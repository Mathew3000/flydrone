"""
M0 sanity check for the fly-drone-sim project.

Goal: prove the two things the plan needs from the simulator actually work,
headless, before any connectome code is written:
  1. Direct per-motor RPM control (no built-in flight controller in the way)
  2. Reading back an RGB camera frame from the drone's own point of view

This does NOT talk to the connectome sim yet. It just drives the drone at a
fixed hover RPM for a few seconds and dumps one camera frame to disk, so we
have concrete proof M0 works before wiring in the medulla encoder.
"""
import os
import numpy as np
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import ImageType

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "sanity")
os.makedirs(OUT_DIR, exist_ok=True)

env = VisionAviary(num_drones=1, gui=False, record=False, freq=240)
obs = env.reset()

hover_rpm = env.HOVER_RPM
print(f"HOVER_RPM = {hover_rpm:.1f}, MAX_RPM = {env.MAX_RPM:.1f}")

action = {0: np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])}

n_steps = 240 * 3  # 3 seconds of sim time at 240 Hz
last_rgb = None
for i in range(n_steps):
    obs, reward, done, info = env.step(action)
    if i == n_steps - 1:
        last_rgb = obs["0"]["rgb"]

pos = env.pos[0]
print(f"Position after {n_steps} steps at fixed hover RPM: {pos}")
print(f"RGB frame shape: {last_rgb.shape}, dtype: {last_rgb.dtype}")

env._exportImage(
    img_type=ImageType.RGB,
    img_input=last_rgb,
    path=OUT_DIR + "/",
    frame_num=0,
)
print(f"Saved frame to {OUT_DIR}/frame_0.png")
env.close()
