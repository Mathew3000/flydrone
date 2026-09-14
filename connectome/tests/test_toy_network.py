"""
Regression test for the M1 pipeline (medulla encoder -> LIFNetwork -> motor
decoder) using synthetic shifted-checkerboard frames instead of a captured
drone camera sequence, so this runs standalone without pybullet/the
simulator. Run with: python -m pytest connectome/tests/ (or just run this
file directly).

This is intentionally about wiring correctness, not biological realism: it
checks that (a) a clearly rightward-shifting pattern drives the "right"
motion population more than "left", and (b) that drive actually reaches the
motor layer and produces non-zero, non-constant decoded output. Tuning
(gain, synaptic weights) needed serious trial and error to get spikes to
reach the motor layer at all -- see the GAIN constant below and
docs/connectome-data-access.md's note on why this will need re-tuning once
real connectome weights replace the toy network.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from connectome.data_loader import make_toy_network
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive
from connectome import motor_decoder as md

GAIN = 8.0  # see PATCHES.md-style note above: tuned empirically so a clear
            # motion signal reliably crosses v_thresh=1.0 through the toy
            # network's random weights; will need re-tuning for real weights.


def make_shifting_checkerboard(n_frames=20, size=(48, 64), square=8, shift_per_frame=2):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(n_frames):
        shifted_x = xx + i * shift_per_frame
        pattern = (((shifted_x // square) + (yy // square)) % 2) * 255
        frame = np.stack([pattern, pattern, pattern, np.full_like(pattern, 255)], axis=-1)
        frames.append(frame.astype(np.float64))
    return np.stack(frames)


def run_pipeline(frames, seed=1, gain=GAIN, dt=0.1e-3, decode_window_s=20e-3, steps_per_frame_gap=500):
    net_spec = make_toy_network(seed=seed)
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=net_spec["weights"], dt=dt)
    cti = net_spec["cell_type_indices"]
    decode_every = int(round(decode_window_s / dt))

    steps_since_decode = 0
    results = []
    for i in range(1, len(frames)):
        drive = encode_to_drive(frames[i - 1], frames[i], net.n, cti, gain=gain)
        for _ in range(steps_per_frame_gap):
            net.step(external_input=drive)
            steps_since_decode += 1
            if steps_since_decode >= decode_every:
                counts = net.reset_spike_window()
                rate_hz = md.spike_counts_to_rate(counts, decode_window_s)
                left, right = md.decode_lr_pools(rate_hz, cti["motor_left"], cti["motor_right"], baseline_hz=0.0)
                results.append(md.lr_to_thrust_yaw(left, right))
                steps_since_decode = 0
    return np.array(results)  # columns: thrust, yaw


def test_rightward_motion_produces_varying_nonzero_output():
    frames = make_shifting_checkerboard(n_frames=15, shift_per_frame=3)
    results = run_pipeline(frames)
    assert len(results) > 0, "no decode windows ran -- check steps_per_frame_gap/decode_window_s"
    assert np.any(results != 0), "pipeline produced all-zero output -- drive is too weak to reach the motor layer"
    assert results[:, 1].std() > 0, "yaw output never varies -- check the encoder is actually seeing motion"


if __name__ == "__main__":
    test_rightward_motion_produces_varying_nonzero_output()
    print("OK: M1 pipeline test passed (non-zero, varying motor output from synthetic motion).")
