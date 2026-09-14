"""
Grabs a short sequence of camera frames from the drone simulator while it
does a small yaw spin, so there is real optical flow to feed the medulla
encoder with (a single static frame has no motion to detect). Saves the
sequence as a .npy stack for scripts/m1_offline_test.py to consume offline
-- keeps the simulator and the connectome/medulla-encoder test decoupled.
"""
import os
import numpy as np
from gym_pybullet_drones.envs.VisionAviary import VisionAviary

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "m1")
os.makedirs(OUT_DIR, exist_ok=True)

env = VisionAviary(num_drones=1, gui=False, record=False, freq=240)
env.reset()

hover = env.HOVER_RPM
# Small yaw torque: front-left/back-right vs front-right/back-left RPM offset.
yaw_offset = hover * 0.02
action = {0: np.array([hover + yaw_offset, hover - yaw_offset, hover + yaw_offset, hover - yaw_offset])}

frames = []
capture_every = 8  # ~30 Hz camera at a 240 Hz physics step
n_steps = 240 * 4  # 4 seconds
for i in range(n_steps):
    obs, *_ = env.step(action)
    if i % capture_every == 0:
        frames.append(obs["0"]["rgb"].copy())

frames = np.stack(frames)
np.save(os.path.join(OUT_DIR, "frame_sequence.npy"), frames)
print(f"Captured {frames.shape[0]} frames, shape {frames.shape[1:]}, saved to {OUT_DIR}/frame_sequence.npy")
env.close()
