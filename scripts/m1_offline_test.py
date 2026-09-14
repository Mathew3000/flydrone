"""
M1 (see docs/projektplan.md / README): offline coupling test. Takes the
camera frame sequence captured by capture_frame_sequence.py and runs it
through medulla encoder -> LIF network -> motor decoder, with no closed
loop back to the simulator yet -- just proving the pipeline produces
sensible, non-degenerate output.

Uses connectome.data_loader.make_toy_network() (NOT the real MaleCNS
connectome -- that needs a neuPrint account/token, see
docs/connectome-data-access.md). This validates the wiring; swapping in
real connectivity data later should not require touching this script.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from connectome.data_loader import make_toy_network
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive
from connectome import motor_decoder as md

FRAMES_PATH = os.path.join(os.path.dirname(__file__), "..", "files", "m1", "frame_sequence.npy")
CAMERA_HZ = 30.0
DT = 0.1e-3            # network step, 0.1 ms
DECODE_WINDOW_S = 20e-3  # 20 ms decode window, matches the original project
STEPS_PER_FRAME_GAP = int(round((1.0 / CAMERA_HZ) / DT))  # network steps between two camera frames

net_spec = make_toy_network(seed=1)
net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=net_spec["weights"], dt=DT)
cti = net_spec["cell_type_indices"]

frames = np.load(FRAMES_PATH)
print(f"Loaded {frames.shape[0]} frames {frames.shape[1:]} @ ~{CAMERA_HZ} Hz")
print(f"{STEPS_PER_FRAME_GAP} network steps per frame gap (dt={DT*1e3:.2f} ms)")

# Rough baseline: run a few frame-gaps with a static (no-motion) pair first,
# to get each pool's "resting" spike rate for the motor decoder's baseline
# subtraction, rather than hand-picking a baseline_hz constant.
baseline_rate = None
steps_since_decode = 0
decode_every_n_steps = int(round(DECODE_WINDOW_S / DT))

results = []
elapsed_steps = 0
for i in range(1, len(frames)):
    drive = encode_to_drive(frames[i - 1], frames[i], net.n, cti, gain=8.0)
    for _ in range(STEPS_PER_FRAME_GAP):
        net.step(external_input=drive)
        elapsed_steps += 1
        steps_since_decode += 1
        if steps_since_decode >= decode_every_n_steps:
            counts = net.reset_spike_window()
            rate_hz = md.spike_counts_to_rate(counts, DECODE_WINDOW_S)
            if baseline_rate is None:
                baseline_rate = rate_hz  # first window = baseline (frames start static)
            baseline_left = float(np.mean(baseline_rate[cti["motor_left"]]))
            baseline_right = float(np.mean(baseline_rate[cti["motor_right"]]))
            left, right = md.decode_lr_pools(
                rate_hz, cti["motor_left"], cti["motor_right"],
                baseline_hz=0.0,  # already comparing pool-specific baselines below
            )
            thrust, yaw = md.lr_to_thrust_yaw(left, right)
            results.append((elapsed_steps * DT, left, right, thrust, yaw))
            steps_since_decode = 0

print(f"\n{'t (s)':>8}  {'left':>8}  {'right':>8}  {'thrust':>8}  {'yaw':>8}")
for t, l, r, th, yw in results:
    print(f"{t:8.3f}  {l:8.4f}  {r:8.4f}  {th:8.4f}  {yw:8.4f}")

yaw_values = np.array([r[4] for r in results])
print(f"\nyaw: mean={yaw_values.mean():.4f} std={yaw_values.std():.4f} "
      f"min={yaw_values.min():.4f} max={yaw_values.max():.4f}")
print("Non-zero, varying output across windows means the pipeline (encoder "
      "-> spiking network -> decoder) is actually reacting to the frames, "
      "not just passing through silence or a constant.")
