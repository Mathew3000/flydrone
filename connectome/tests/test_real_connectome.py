"""
Regression test for the M1 pipeline running on the REAL MaleCNS connectome
subnetwork (connectome.local_maleCNS.local_fetch_lr_subnetwork) instead of
the synthetic make_toy_network() -- this is the real-data upgrade path
mentioned in test_toy_network.py's docstring.

Requires the local bulk data files in data/raw/ (see
docs/connectome-data-access.md) -- skipped automatically if they're not
present, so this doesn't break CI/environments without the ~1.1GB download.

Calibration note (the whole point of this file existing): the toy network's
GAIN=8.0 was tuned against random ~0.15-mean synaptic weights. The real
MaleCNS weights are raw synapse counts (mean ~2.6, max 56 in the T4/T5 +
HS/H1/H2 subnetwork), and a handful of motor neurons (HS/H1/H2, only 6 per
side) each receive a *cumulative* incoming weight on the order of 10,000+
from their ~6,800 upstream visual neurons combined. Feeding drive straight
into that network with the toy network's gain saturates every motor neuron
to max firing rate regardless of which side is actually stimulated (both
sides decode to exactly 0.2500, the motor_decoder ceiling) -- confirmed by a
weight_scale/gain sweep (see docs/connectome-data-access.md). The fix is
WEIGHT_SCALE below: it rescales the raw synapse-count weights down before
they're used as LIFNetwork synaptic weights, which is the real-data
equivalent of the toy network's small random weights. GAIN stays at the
same 8.0 used elsewhere (it scales the encoder's *input* drive, not the
network's internal weights, so there's no reason to re-tune it separately).
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from connectome.local_maleCNS import local_fetch_lr_subnetwork, DATA_DIR_DEFAULT, WEIGHTS_FILE
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive_hemifield
from connectome import motor_decoder as md

GAIN = 8.0
WEIGHT_SCALE = 0.0015  # see module docstring -- calibrated against the real
                        # T4/T5 -> HS/H1/H2 subnetwork's raw synapse-count weights.

DATA_AVAILABLE = os.path.isfile(os.path.join(DATA_DIR_DEFAULT, WEIGHTS_FILE))


def make_lr_checkerboard(n_frames=15, size=(48, 64), square=8, shift_left=3, shift_right=0):
    """Like test_toy_network.make_shifting_checkerboard(), but shifts the
    left and right halves of the frame independently, so a stimulus can be
    confined to one anatomical hemifield -- shift_right=0 means "no motion
    on the right", matching what encode_to_drive_hemifield's left/right
    split actually looks at."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    mid = w // 2
    frames = []
    for i in range(n_frames):
        shifted_x = xx.copy()
        shifted_x[:, :mid] = xx[:, :mid] + i * shift_left
        shifted_x[:, mid:] = xx[:, mid:] + i * shift_right
        pattern = (((shifted_x // square) + (yy // square)) % 2) * 255
        frame = np.stack([pattern, pattern, pattern], axis=-1).astype(np.float64)
        frames.append(frame)
    return np.stack(frames)


_NET_CACHE = {}


def _get_net():
    if "net" not in _NET_CACHE:
        _NET_CACHE["net"] = local_fetch_lr_subnetwork()
    return _NET_CACHE["net"]


def run_pipeline(frames, weight_scale=WEIGHT_SCALE, gain=GAIN, dt=0.1e-3,
                  decode_window_s=20e-3, steps_per_frame_gap=500):
    net = _get_net()
    idx = net["cell_type_indices"]
    W = net["weights"].multiply(weight_scale).tocsr()
    lif = LIFNetwork(n_neurons=net["n_neurons"], weights=W, dt=dt)
    decode_every = int(round(decode_window_s / dt))

    steps_since_decode = 0
    results = []
    for i in range(1, len(frames)):
        drive = encode_to_drive_hemifield(frames[i - 1], frames[i], net["n_neurons"], idx, gain=gain)
        for _ in range(steps_per_frame_gap):
            lif.step(external_input=drive)
            steps_since_decode += 1
            if steps_since_decode >= decode_every:
                counts = lif.reset_spike_window()
                rate_hz = md.spike_counts_to_rate(counts, decode_window_s)
                left, right = md.decode_lr_pools(rate_hz, idx["motor_left"], idx["motor_right"], baseline_hz=0.0)
                results.append((left, right))
                steps_since_decode = 0
    return np.array(results)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_left_only_motion_produces_left_dominant_output():
    frames = make_lr_checkerboard(shift_left=3, shift_right=0)
    results = run_pipeline(frames)
    left_mean, right_mean = results[:, 0].mean(), results[:, 1].mean()
    assert left_mean > right_mean, (
        f"left-only visual motion should drive motor_left more than motor_right "
        f"(got left={left_mean:.4f}, right={right_mean:.4f})"
    )
    assert right_mean < 0.05, f"motor_right should stay near-quiet for left-only motion (got {right_mean:.4f})"
    assert left_mean > 0.15, f"motor_left should show a clear response to left-only motion (got {left_mean:.4f})"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_right_only_motion_produces_right_dominant_output():
    frames = make_lr_checkerboard(shift_left=0, shift_right=3)
    results = run_pipeline(frames)
    left_mean, right_mean = results[:, 0].mean(), results[:, 1].mean()
    assert right_mean > left_mean, (
        f"right-only visual motion should drive motor_right more than motor_left "
        f"(got left={left_mean:.4f}, right={right_mean:.4f})"
    )


@pytest.mark.skipif(not DATA_AVAILABLE, reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_symmetric_motion_produces_balanced_output():
    """Sanity/fairness check: symmetric motion on both sides should give a
    near-zero yaw signal (left ~= right), not an inherent left or right
    bias from the (slightly unequal, e.g. 6790 vs 6795 neuron) anatomy."""
    frames = make_lr_checkerboard(shift_left=3, shift_right=3)
    results = run_pipeline(frames)
    left_mean, right_mean = results[:, 0].mean(), results[:, 1].mean()
    assert abs(left_mean - right_mean) < 0.01, (
        f"symmetric motion should not produce a strong left/right bias "
        f"(got left={left_mean:.4f}, right={right_mean:.4f})"
    )


if __name__ == "__main__":
    if not DATA_AVAILABLE:
        print("SKIPPED: local MaleCNS bulk data not present -- see docs/connectome-data-access.md")
    else:
        test_left_only_motion_produces_left_dominant_output()
        test_right_only_motion_produces_right_dominant_output()
        test_symmetric_motion_produces_balanced_output()
        print("OK: real-connectome M1 pipeline test passed (left/right-selective, symmetric-balanced motor output).")
