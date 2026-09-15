"""
Render world.room once, from outside and from the drone's own camera, so the
environment can be eyeballed instead of only inferred from flight numbers.

Writes to files/room/ (gitignored):
  overview.png   third-person view of the whole room
  drone_pov.png  the actual 64x48 VisionAviary camera frame the medulla
                 encoder receives, upscaled 8x with nearest-neighbour so the
                 individual pixels stay visible -- this is genuinely all the
                 resolution the connectome gets.
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import pybullet as p
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from world.room import build_room

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "room")
os.makedirs(OUT_DIR, exist_ok=True)

env = VisionAviary(num_drones=1, gui=False, record=False, freq=240)
env.reset()
build_room(env.CLIENT, plane_id=env.PLANE_ID)

# Fly up to hover height first, so the POV shot is the view the connectome
# actually works with rather than a shot from the floor.
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
target = np.array([0.0, 0.0, 1.0])
for _ in range(240 * 3):
    s = env._getDroneStateVector(0)
    rpm, _, _ = ctrl.computeControl(control_timestep=1/240,
                                    cur_pos=s[0:3], cur_quat=s[3:7], cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                    target_pos=target, target_rpy=np.array([0.0, 0.0, 0.6]))
    env.step({0: rpm})

view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 1.0], distance=13.0,
                                           yaw=35, pitch=-52, roll=0, upAxisIndex=2,
                                           physicsClientId=env.CLIENT)
proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=4/3, nearVal=0.1, farVal=100.0,
                                    physicsClientId=env.CLIENT)
_, _, rgb, _, _ = p.getCameraImage(width=960, height=720, viewMatrix=view, projectionMatrix=proj,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
Image.fromarray(np.reshape(rgb, (720, 960, 4))[:, :, :3].astype(np.uint8)).save(
    os.path.join(OUT_DIR, "overview.png"))

pov = env._getDroneImages(0, segmentation=False)[0]
pov_img = Image.fromarray(pov[:, :, :3].astype(np.uint8))
pov_img.resize((pov_img.width * 8, pov_img.height * 8), Image.NEAREST).save(
    os.path.join(OUT_DIR, "drone_pov.png"))

env.close()
print(f"wrote {OUT_DIR}/overview.png and {OUT_DIR}/drone_pov.png")
print(f"drone POV frame: {pov.shape}, brightness min/max = {pov[:,:,:3].min():.0f}/{pov[:,:,:3].max():.0f}")
