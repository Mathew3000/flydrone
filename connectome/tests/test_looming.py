"""
Regression test for the looming pathway (encode_to_drive_looming ->
LC4/LPLC2 -> escape descending neurons), the result scripts/m5_looming.py
demonstrates in the simulator.

Simulator-free like test_optomotor.py: an expanding texture and a translating
one are all this needs, and they isolate the property under test better than a
drum does.

What this guards is specificity, and it currently records a FAILURE to achieve
it. Four encoder formulations were measured, approach-to-rotation (under 1.0
means the escape circuit fires harder at a turn than at an impending
collision): horizontal outward motion pooled per hemifield 0.52x, its spatial
divergence 0.48x, the same on a retinotopic 4x8 grid 0.65x, and horizontal AND
vertical outward combined conjunctively 4.39x -- except that last number is
confounded, because it compared a mosaic-textured approaching object against a
vertically striped rotating drum, and a vertical grating has no vertical
luminance gradient for the vertical channel to find. Matched textures give
1.5x. The xfail below keeps that honest and will announce itself the day
someone fixes it.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from connectome.local_maleCNS import (local_fetch_looming_subnetwork,
                                      DATA_DIR_DEFAULT, WEIGHTS_FILE)
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive_looming
from connectome import motor_decoder as md

GAIN = 40.0          # calibrated in scripts/m5_looming.py, for the simulator's stimulus
# The synthetic stimulus here is about an order of magnitude weaker than the
# simulator's: a high-contrast textured box expanding against a background
# drives the correlator far harder than a smooth full-field zoom, and the
# correlator's output goes as contrast SQUARED. The network test therefore
# needs its own gain to reach the same operating point. The two numbers are not
# comparable and neither is wrong.
SYNTHETIC_GAIN = 600.0
WEIGHT_SCALE = 0.0015

DATA_AVAILABLE = os.path.isfile(os.path.join(DATA_DIR_DEFAULT, WEIGHTS_FILE))
IDX = {"looming_L": np.arange(0, 10), "looming_R": np.arange(10, 20)}


def _texture(size=(48, 64), seed=0):
    """Smooth broadband noise.

    Not a blocky mosaic: with hard 4 px block edges and nearest-neighbour
    resampling, a purely horizontal shift makes those edges jump and the
    VERTICAL correlator reads 0.054 against the horizontal 0.112 -- an artefact
    of the stimulus, not of the encoder. In the simulator the same comparison
    is 0.015 against 0.182. A blocky synthetic stimulus would therefore fail
    the specificity test for reasons that have nothing to do with the code
    under test.
    """
    rng = np.random.default_rng(seed)
    field = rng.normal(128, 60, size=size)
    return np.clip(cv2.GaussianBlur(field, (0, 0), 1.6), 0, 255)


def _sample(field, scale=1.0, shift=0.0):
    """Resample `field` about its centre, scaled and/or shifted horizontally.

    Bilinear, via an affine warp: integer indexing quantises sub-pixel motion
    into jumps, which is exactly the aliasing a correlation detector is
    sensitive to.
    """
    h, w = field.shape
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[scale, 0.0, cx - scale * cx + shift],
                  [0.0, scale, cy - scale * cy]], dtype=np.float64)
    img = cv2.warpAffine(field, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return np.stack([img] * 3, axis=-1)


def drive_for(scale=1.0, shift=0.0, seed=0):
    field = _texture(seed=seed)
    d = encode_to_drive_looming(_sample(field), _sample(field, scale, shift), 20, IDX, gain=GAIN)
    return float(d[IDX["looming_L"]].mean() + d[IDX["looming_R"]].mean())


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation, kept as a live marker rather than deleted. The encoder "
    "is not selective for expansion over translation when both carry the same "
    "texture: measured 1.5x, where anything near 1.0 means an escape circuit "
    "that fires at a turn. The 4.39x reported from scripts/m5_looming.py is "
    "confounded -- its rotation control used the vertically striped drum and "
    "its looming object an isotropic mosaic, and a vertically striped pattern "
    "has no vertical luminance gradient at all, so the vertical channel reads "
    "exactly 0.00000 on it whether it translates OR expands. That measurement "
    "separated two textures, not two motions. Flip this to a passing test only "
    "with an encoder that discriminates the spatial PATTERN of motion "
    "direction (uniform for rotation, diverging for approach) rather than the "
    "presence of vertical motion."))
def test_expansion_beats_translation():
    """The property a looming detector must have and this one does not yet."""
    expansion = drive_for(scale=1.10)
    translation = drive_for(shift=3.0)
    assert expansion > 3 * translation, (
        f"expansion {expansion:.5f} vs translation {translation:.5f}"
    )


def test_vertical_channel_responds_to_vertical_motion():
    """reichardt_response(axis=0) must actually detect vertical motion.

    Worth pinning separately: the axis parameter is the one piece of the
    looming work that is unambiguously correct, and everything else built on it
    depends on it staying that way.
    """
    from connectome.medulla_encoder import reichardt_response
    field = _texture()
    still = _sample(field)
    shifted_down = np.stack([np.roll(field, 3, axis=0)] * 3, axis=-1)
    resp = reichardt_response(still, shifted_down, axis=0)
    assert abs(float(np.mean(resp))) > 1e-4, "vertical channel is blind to vertical motion"


def test_contraction_is_weaker_than_expansion():
    """A receding object must not drive the pathway as hard as an approaching one.

    The threshold is the measured ratio (0.155) with headroom, not a round
    number: per-pixel rectification before pooling gives any noisy field a
    positive mean, so exact silence is not achievable here and asserting it
    would only encode a wish.
    """
    contraction = drive_for(scale=1 / 1.10)
    expansion = drive_for(scale=1.10)
    assert contraction < 0.25 * expansion, (
        f"contraction {contraction:.5f} should stay well below expansion {expansion:.5f}"
    )


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_escape_neurons_fire_for_expansion_only():
    """End to end on the real LC4/LPLC2 -> escape DN subnetwork."""
    net = local_fetch_looming_subnetwork()
    cti = net["cell_type_indices"]
    W = net["weights"].multiply(WEIGHT_SCALE).tocsr()
    field = _texture()

    def run(scale=1.0, shift=0.0, n_frames=6):
        lif = LIFNetwork(n_neurons=net["n_neurons"], weights=W, dt=0.1e-3)
        peak = 0.0
        for _ in range(n_frames):
            drive = encode_to_drive_looming(_sample(field), _sample(field, scale, shift),
                                            net["n_neurons"], cti, gain=SYNTHETIC_GAIN)
            for _ in range(333):
                lif.step(external_input=drive)
            rate = md.spike_counts_to_rate(lif.reset_spike_window(), 333 * 0.1e-3)
            left, right = md.decode_lr_pools(rate, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)
            peak = max(peak, left + right)
        return peak

    expansion = run(scale=1.10)
    translation = run(shift=3.0)
    assert expansion > 0.0, "an expanding pattern must fire the escape neurons"
    # NOT asserting silence for translation: the encoder cannot deliver that yet
    # (see the xfail above). What is pinned is that expansion is the stronger of
    # the two, which is the weakest claim the pathway has to support to be worth
    # anything at all.
    assert expansion > translation, (
        f"expansion {expansion:.5f} must at least exceed translation {translation:.5f}"
    )
