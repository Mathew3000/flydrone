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
