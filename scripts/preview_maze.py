"""Render world.build_maze once, from above and from the drone's own camera."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import pybullet as pb
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from world.room import build_maze

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "maze")
os.makedirs(OUT_DIR, exist_ok=True)

env = VisionAviary(num_drones=1, gui=False, record=False, freq=240)
env.reset()
maze = build_maze(env.CLIENT, plane_id=env.PLANE_ID)
print("start", maze["start"], "goal", maze["goal"], "walls", len(maze["walls"]))

pos = np.array([maze["start"][0], maze["start"][1], 1.2])
quat = pb.getQuaternionFromEuler([0, 0, 0])
pb.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(pos), quat, physicsClientId=env.CLIENT)
env.pos[0], env.quat[0] = pos, np.array(quat)

view = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=26,
                                            yaw=90, pitch=-89, roll=0, upAxisIndex=2,
                                            physicsClientId=env.CLIENT)
proj = pb.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0,
                                     physicsClientId=env.CLIENT)
_, _, rgb, _, _ = pb.getCameraImage(width=700, height=700, viewMatrix=view, projectionMatrix=proj,
                                    renderer=pb.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
Image.fromarray(np.reshape(rgb, (700, 700, 4))[:, :, :3].astype(np.uint8)).save(
    os.path.join(OUT_DIR, "overview.png"))

pov = env._getDroneImages(0, segmentation=False)[0]
img = Image.fromarray(pov[:, :, :3].astype(np.uint8))
img.resize((img.width * 6, img.height * 6), Image.NEAREST).save(
    os.path.join(OUT_DIR, "drone_pov.png"))
print(f"wrote {OUT_DIR}/overview.png and drone_pov.png")
env.close()
