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


def reichardt_response(frame_prev: np.ndarray, frame_curr: np.ndarray,
                       spacing: int = 2, blur_sigma: float = 1.0,
                       axis: int = 1) -> np.ndarray:
    """Elementary motion detector of the Hassenstein-Reichardt type.

    The other encoders in this module hand the network a velocity estimate
    computed by Farneback optical flow. That is an algorithm standing in for
    the very computation T4/T5 are famous for, and it has a measurable
    consequence: the pipeline comes out tuned to image VELOCITY, where real
    flies are tuned to TEMPORAL FREQUENCY (measured in
    scripts/m3_tuning_curve.py -- 12 and 24 bars peak at the same 0.6 rad/s
    rather than at the same Hz).

    A correlation detector has that property by construction. It multiplies a
    delayed signal from one sampling point with the undelayed signal from its
    neighbour, and subtracts the mirror-image pairing:

        resp = c(x, t-D) * c(x+s, t)  -  c(x, t) * c(x+s, t-D)

    For a drifting grating of spatial frequency k and velocity v this evaluates
    to  resp ~ sin(k*s) * sin(k*v*D), so the response peaks when k*v*D = pi/2,
    i.e. at temporal frequency f = 1/(4D) -- independent of k, hence
    independent of bar width. sin(k*s) only sets the amplitude. That is exactly
    the fly signature the flow-based encoders cannot produce.

    The delay D is fixed here at one camera frame (1/30 s), which puts the
    predicted optimum at 7.5 Hz. Making D a free parameter would need the
    detector to carry state between calls (a low-pass filter), and therefore a
    stateful object rather than the two-frame signature the rest of this module
    and all its call sites use. Fixed D is the honest, minimal version: it
    tests the tuning claim without restructuring the pipeline.

    `axis` selects the correlation direction: 1 (default) pairs horizontal
    neighbours and detects left/right motion, 0 pairs vertical neighbours and
    detects up/down motion. A horizontal channel alone cannot distinguish a
    rotating panorama from an approaching object -- both produce outward
    horizontal motion -- because rotation simply has no vertical component to
    give itself away. Measured: on the drum at 1.2 rad/s the vertical channel
    reads 0.015 against the horizontal channel's 0.182, while an approaching
    object reads 0.067 vertical against 0.094 horizontal.

    Returns a signed array; positive means rightward (axis=1) or downward
    (axis=0) motion.
    Magnitude scales with contrast SQUARED -- a multiplication, not a
    normalised estimate -- so it is not contrast-invariant. Real flies largely
    are, via adaptation this model does not have.
    """
    def contrast(f):
        g = f[..., :3].astype(np.uint8) if (f.ndim == 3 and f.shape[2] >= 3) else f.astype(np.uint8)
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) if g.ndim == 3 else g
        g = cv2.GaussianBlur(g.astype(np.float64), (0, 0), blur_sigma)  # optics blur
        return (g - g.mean()) / 128.0   # contrast, not luminance: the product of two
                                        # positive luminances is dominated by its DC term

    c_prev, c_curr = contrast(frame_prev), contrast(frame_curr)
    if axis == 0:
        c_prev, c_curr = c_prev.T, c_curr.T
    s = spacing
    resp = c_prev[:, :-s] * c_curr[:, s:] - c_curr[:, :-s] * c_prev[:, s:]
    return resp.T if axis == 0 else resp


def encode_to_drive_reichardt(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    n_neurons: int,
    cell_type_indices: dict[str, np.ndarray],
    gain: float = 1.0,
    spacing: int = 2,
) -> np.ndarray:
    """Drop-in replacement for encode_to_drive_progressive() built on
    reichardt_response() instead of Farneback flow.

    Identical anatomy: each hemifield's population is driven by progressive
    (front-to-back) motion across its own eye, rectified, so one drum direction
    excites exactly one side. Only the motion measurement underneath differs --
    which is the whole point, since that is where the velocity-versus-temporal-
    frequency question lives.
    """
    drive = np.zeros(n_neurons, dtype=np.float64)
    resp = reichardt_response(frame_prev, frame_curr, spacing=spacing)
    mid = resp.shape[1] // 2

    # front-to-back is leftward (negative) on the left, rightward (positive) on
    # the right -- same convention as encode_to_drive_progressive()
    left_energy = float(np.mean(np.clip(-resp[:, :mid], 0, None)))
    right_energy = float(np.mean(np.clip(resp[:, mid:], 0, None)))

    idx_l = cell_type_indices.get("visual_L")
    idx_r = cell_type_indices.get("visual_R")
    if idx_l is not None and len(idx_l):
        drive[idx_l] = gain * left_energy
    if idx_r is not None and len(idx_r):
        drive[idx_r] = gain * right_energy
    return drive


def encode_to_drive_looming(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    n_neurons: int,
    cell_type_indices: dict[str, np.ndarray],
    gain: float = 1.0,
    spacing: int = 2,
) -> np.ndarray:
    """Radial-expansion drive for the LC4/LPLC2 looming populations.

    Expansion points away from the image centre everywhere; a yaw turn points
    one way everywhere. So the discriminator is the RADIAL component of motion,
    integrated across the centre:

        radial = mean over the image of  resp(x) * sign(x - centre)

    Under rotation resp is roughly constant, so the two halves contribute
    opposite signs and cancel. Under expansion both halves contribute
    positively. That cancellation is the whole mechanism, and it is what
    "opponency" means.

    The order of operations is the entire point, and getting it wrong is what
    made four earlier attempts fail. Rectifying per pixel BEFORE pooling
    destroys the cancellation: the inward-moving half contributes zero instead
    of a negative, so rotation survives the pooling intact. Measured, expansion
    against translation of the SAME texture:

        rectify per pixel, then pool    1.10x isotropic,  1.00x striped
        pool the signed radial, then rectify   translation exactly 0.00000
                                               isotropic,  8.5x striped

    For comparison, the earlier attempts, all of which rectified first:
    hemifield outward 0.52x, its spatial divergence 0.48x, retinotopic 4x8
    grid 0.65x, and a horizontal-plus-vertical conjunction that measured 4.39x
    but only because it compared a striped rotating drum against a
    mosaic-textured approaching object -- a vertical grating has no vertical
    luminance gradient, so that channel read zero on it whether it rotated or
    expanded. That number separated two textures, not two motions.

    Lateralisation is applied as a weighting, not as a second opponent
    computation: the rectified global radial signal is the looming magnitude
    (it is what rejects rotation, and it only rejects rotation because it
    integrates across the centre), and it is then split between the two
    populations in proportion to where the motion energy actually sits. An
    object approaching off to one side therefore excites that side more without
    reopening the hole that per-hemifield pooling would.
    """
    drive = np.zeros(n_neurons, dtype=np.float64)
    resp = reichardt_response(frame_prev, frame_curr, spacing=spacing, axis=1)
    h, w = resp.shape
    centre = w / 2.0
    outward_sign = np.where(np.arange(w) < centre, -1.0, 1.0)

    # pool first, rectify second -- see docstring
    looming = max(float(np.mean(resp * outward_sign)), 0.0)

    mid = int(centre)
    energy_l = float(np.mean(np.abs(resp[:, :mid])))
    energy_r = float(np.mean(np.abs(resp[:, mid:])))
    total = energy_l + energy_r
    frac_l, frac_r = (0.5, 0.5) if total <= 0 else (energy_l / total, energy_r / total)

    idx_l = cell_type_indices.get("looming_L")
    idx_r = cell_type_indices.get("looming_R")
    if idx_l is not None and len(idx_l):
        drive[idx_l] = gain * looming * 2.0 * frac_l
    if idx_r is not None and len(idx_r):
        drive[idx_r] = gain * looming * 2.0 * frac_r
    return drive
