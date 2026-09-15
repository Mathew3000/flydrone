"""
Medulla encoder: turns a pair of camera frames into drive current for
specific downstream cell-type populations -- NOT into the retina.

This mirrors the key fix the boat.horse/fly writeup describes: injecting
signal at the retina did nothing ("visual projection neurons stay at 0.9 Hz
whatever the image"), so instead this encoder drives the motion- and
looming-detector populations directly, the way their real firing is actually
driven biologically:

  - Motion (optic flow direction/speed) -> T4/T5 cells
    T4/T5 are the fly's well-established elementary motion detectors
    (four subtypes T4a-d / T5a-d each tuned to one of the four cardinal
    motion directions; T4 responds to brightness increments, T5 to
    decrements). See e.g. the eLife 2017 "comprehensive connectome of a
    neural substrate for 'ON' motion detection" paper.
  - Looming (expanding flow field / radial divergence) -> LC4 / LPLC2 cells
    LPLC2 in particular is characterized as an "ultra-selective" looming
    detector via radial motion opponency (Klapoetke et al. 2017); LC4 is
    the other well-studied looming-selective lobula columnar cell type.

Exact MaleCNS type-name strings (e.g. is it "T4a" or "T4a_R" etc., and
whether LPLC2 exists under that exact name in this dataset) need to be
confirmed once real neuPrint data is pulled -- see
docs/connectome-data-access.md. Until then this module works against an
abstract `cell_type_indices: dict[str, np.ndarray]` mapping a cell-type name
to the neuron indices in the LIFNetwork that represent it, so swapping in
real body-ID-based indices later is a one-line change at the call site, not
a rewrite of this module.
"""
from __future__ import annotations

import numpy as np
import cv2


def optical_flow(frame_prev: np.ndarray, frame_curr: np.ndarray) -> np.ndarray:
    """Dense optical flow (Farneback) between two grayscale/RGB frames.

    Returns an (H, W, 2) array of (dx, dy) per pixel, in pixels/frame.
    """
    def to_gray(f):
        if f.ndim == 3 and f.shape[2] >= 3:
            return cv2.cvtColor(f[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return f.astype(np.uint8)

    prev_gray = to_gray(frame_prev)
    curr_gray = to_gray(frame_curr)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=9,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
    )
    return flow


def directional_motion_energy(flow: np.ndarray) -> dict[str, float]:
    """Mean flow energy in each of the 4 cardinal directions (T4/T5-style tuning)."""
    dx, dy = flow[..., 0], flow[..., 1]
    return {
        "right": float(np.mean(np.clip(dx, 0, None))),
        "left": float(np.mean(np.clip(-dx, 0, None))),
        "down": float(np.mean(np.clip(dy, 0, None))),
        "up": float(np.mean(np.clip(-dy, 0, None))),
    }


def looming_energy(flow: np.ndarray) -> float:
    """Radial divergence of the flow field -- positive for an expanding
    (approaching/looming) pattern, near zero for pure translation or rotation.
    """
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    rx, ry = xx - cx, yy - cy
    r_norm = np.sqrt(rx**2 + ry**2) + 1e-6
    radial_component = (flow[..., 0] * rx + flow[..., 1] * ry) / r_norm
    return float(np.mean(radial_component))


def encode_to_drive(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    n_neurons: int,
    cell_type_indices: dict[str, np.ndarray],
    gain: float = 1.0,
) -> np.ndarray:
    """Build the external_input vector for one LIFNetwork.step() call.

    cell_type_indices keys used, if present (all optional -- missing keys are
    just skipped, so this also works with a toy network that only has some
    of these populations):
      "T4_right", "T4_left", "T5_up", "T5_down" (or however directions are
      split across the T4/T5 subtypes in your index map) -> directional motion drive
      "LC4", "LPLC2" -> looming drive
    """
    drive = np.zeros(n_neurons, dtype=np.float64)
    flow = optical_flow(frame_prev, frame_curr)
    motion = directional_motion_energy(flow)
    loom = looming_energy(flow)

    direction_map = {
        "T4_right": motion["right"], "T5_right": motion["right"],
        "T4_left": motion["left"], "T5_left": motion["left"],
        "T4_up": motion["up"], "T5_up": motion["up"],
        "T4_down": motion["down"], "T5_down": motion["down"],
    }
    for key, energy in direction_map.items():
        idx = cell_type_indices.get(key)
        if idx is not None and len(idx):
            drive[idx] = gain * energy

    for key in ("LC4", "LPLC2"):
        idx = cell_type_indices.get(key)
        if idx is not None and len(idx) and loom > 0:
            drive[idx] = gain * loom

    return drive


def encode_to_drive_hemifield(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    n_neurons: int,
    cell_type_indices: dict[str, np.ndarray],
    gain: float = 1.0,
) -> np.ndarray:
    """Real-connectome variant of encode_to_drive(): splits the camera frame
    into left/right halves (a simplified stand-in for the fly's two
    retinotopic visual hemifields) and drives the "visual_L"/"visual_R"
    populations (see connectome.local_maleCNS.local_fetch_lr_subnetwork)
    with that half's motion energy -- anatomical side, not cardinal
    direction, is the split that's actually grounded in the local
    MaleCNS data (somaSide) without guessing at T4/T5 subtype conventions.

    Deliberately simpler than encode_to_drive()'s directional/looming split:
    this feeds overall |flow| magnitude per half, not direction-tuned or
    ON/OFF (T4 vs T5) selective drive. That's a reasonable first cut given
    T4 and T5 are driven identically here; refining it (T4 from brightness
    increments, T5 from decrements) is a natural next step, not done yet.
    """
    drive = np.zeros(n_neurons, dtype=np.float64)
    flow = optical_flow(frame_prev, frame_curr)
    h, w = flow.shape[:2]
    mid = w // 2
    left_energy = float(np.abs(flow[:, :mid]).mean())
    right_energy = float(np.abs(flow[:, mid:]).mean())

    idx_l = cell_type_indices.get("visual_L")
    idx_r = cell_type_indices.get("visual_R")
    if idx_l is not None and len(idx_l):
        drive[idx_l] = gain * left_energy
    if idx_r is not None and len(idx_r):
        drive[idx_r] = gain * right_energy
    return drive

def encode_to_drive_progressive(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    n_neurons: int,
    cell_type_indices: dict[str, np.ndarray],
    gain: float = 1.0,
) -> np.ndarray:
    """Direction-selective variant of encode_to_drive_hemifield().

    encode_to_drive_hemifield() feeds each hemifield the mean of |flow|, which
    discards the sign of horizontal motion -- and the sign is the entire
    stimulus in an optomotor experiment. Measured on a rotating drum
    (scripts/m3_optomotor.py): the signed horizontal flow swings cleanly from
    -1.14/-1.02 (drum CCW) to +0.99/+1.16 (drum CW), while the |flow| the
    encoder actually used barely moved (0.94/0.69 vs 0.66/1.06, and that
    residual difference tracks scene asymmetry, not drum direction). The
    connectome consequently could not distinguish the two directions at all.

    This version keeps the anatomical hemifield mapping intact and adds the
    selectivity where the fly has it. Horizontal-system cells in one optic lobe
    are selective for PROGRESSIVE (front-to-back) motion across that eye, so:

        visual_L  <- front-to-back motion in the left half of the image
        visual_R  <- front-to-back motion in the right half

    With a forward-looking camera, "front" is the image centre and "back" is
    the outer edge, so front-to-back is leftward (dx < 0) on the left and
    rightward (dx > 0) on the right. A body or drum rotation therefore
    excites exactly one side -- which is the asymmetry the motor decoder's
    left-minus-right already reads -- while pure forward translation excites
    both equally and cancels in yaw, as it should.

    Rectified (negative values clipped to zero) rather than signed, because
    LIFNetwork drive is a current into a population whose firing rate cannot go
    below zero; the opposing direction is represented by the other side's
    population, not by negative drive on this one.
    """
    drive = np.zeros(n_neurons, dtype=np.float64)
    flow = optical_flow(frame_prev, frame_curr)
    mid = flow.shape[1] // 2
    dx_left, dx_right = flow[:, :mid, 0], flow[:, mid:, 0]

    left_energy = float(np.mean(np.clip(-dx_left, 0, None)))
    right_energy = float(np.mean(np.clip(dx_right, 0, None)))

    idx_l = cell_type_indices.get("visual_L")
    idx_r = cell_type_indices.get("visual_R")
    if idx_l is not None and len(idx_l):
        drive[idx_l] = gain * left_energy
    if idx_r is not None and len(idx_r):
        drive[idx_r] = gain * right_energy
    return drive
