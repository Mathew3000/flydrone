"""
Free flight: the connectome steers a drone through a maze.

Everything before this held the drone in place so the only motion in the image
was the stimulus. Here it flies, and the stimulus is whatever its own motion
produces -- which is the situation the pathway evolved for.

Steering runs on the LC4/LPLC2 -> escape DN subnetwork from M5, but NOT on its
looming readout. Which readout works here was settled by measurement, and the
obvious one lost:

  Flying a corridor towards a wall, drive against distance-to-wall:
      global radial (encode_to_drive_looming)  0.0014 -> 0.0039, non-monotonic,
                                               maximum 10.8 m from the wall
      frontal third only                       0.0001 -> 0.0007, then exactly 0
                                               inside 1.5 m
      lateral imbalance                        0.0036 -> 0.0104, monotonic

  The geometry explains it. In forward translation the flow is a radial
  expansion about the direction of travel, so the wall being flown into sits at
  the focus of expansion where image motion is zero, while the side walls -- at
  constant distance, high angular speed -- dominate any global pooling. A
  looming detector reports an object approaching a stationary observer well and
  a wall approached by a moving one badly.

So steering uses encode_to_drive_centring: the lateral flow imbalance, which
steers away from the closer wall. That is the centring response bees and flies
navigate tunnels with. Same 323-neuron subnetwork, same decoder -- only the
readout differs.

The SIGN of the steering is ours, as it has been for every channel in this
project; the connectome does not say which pool means "turn left". Both signs
are run, and a run with steering disabled gives the baseline any claim has to
beat.

What it achieves and where it stops: the drone flies the first corridor and
takes the first corner, which the unsteered run cannot -- that one flies
straight into the wall and stays there. Over a longer run it follows the next
corridor and then wedges itself against the wall at the second turn. The
40 s figure of zero wall contacts is real but is a window shorter than the
failure; at 120 s there are contacts.

The limit is the one encode_to_drive_centring names: centring keeps the drone
off the side walls, and nothing in this pipeline sees the wall AHEAD. Finishing
a maze needs a frontal close-range channel, which means retinotopic receptive
fields rather than pooling over image halves -- the same conclusion the looming
work reached from its own direction.

Outputs (files/maze/, gitignored): trajectories.png
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as pb

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from connectome.local_maleCNS import local_fetch_looming_subnetwork
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive_centring
from connectome import motor_decoder as md
from world.room import build_maze

PHYSICS_HZ, CAMERA_HZ = 240, 30
STEPS_PER_FRAME = PHYSICS_HZ // CAMERA_HZ
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE = 0.0015
# Calibrated for cruise flow, which is an order of magnitude weaker than the
# approaching-object stimulus m5_looming.py used: 60 barely steers, 150 halves
# the wall contacts, 400 removes them entirely.
GAIN = float(os.environ.get("M6_GAIN", "400"))
CRUISE_SPEED = 1.2           # m/s
TURN_GAIN = float(os.environ.get("TURN_GAIN", "8.0"))
# Absolute, and heavy on purpose. The proximity channel sits near its ceiling
# for most of a flight (median 0.404 against a ceiling of 0.5), so this brake is
# fully on in about 70% of frames and the drone crawls. That looks like a bug
# and is in fact the mechanism: the centring signal needs TIME to swing the
# drone around a corner, because nothing here sees the wall ahead -- it sits at
# the focus of expansion where image motion is zero (see
# encode_to_drive_centring).
#
# Measured by replacing it with a baseline-subtracting adaptive brake so the
# drone keeps its speed: it covers more ground per second and then crashes --
# 692 to 784 contact frames against 0 with this one. Turning harder does not
# rescue it either (TURN_GAIN 20 -> 691 contacts, 45 -> unstable at 1040).
# Slowing down in clutter is what makes the corner work, and it is what insects
# do in clutter too.
BRAKE_GAIN = float(os.environ.get("BRAKE_GAIN", "2.5"))
DURATION_S = float(os.environ.get("DURATION_S", "120.0"))
ALTITUDE = 1.2

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "maze")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading LC4/LPLC2 -> escape DN subnetwork...")
t0 = time.time()
net_spec = local_fetch_looming_subnetwork()
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses "
      f"({time.time()-t0:.1f}s)")


def run(steering: bool, turn_gain: float = TURN_GAIN):
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    maze = build_maze(env.CLIENT, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)

    start = np.array([maze["start"][0], maze["start"][1], ALTITUDE])
    quat = pb.getQuaternionFromEuler([0, 0, -np.pi / 2])   # face down the first corridor
    pb.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(start), quat,
                                       physicsClientId=env.CLIENT)
    pb.resetBaseVelocity(env.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=env.CLIENT)

    target = start.copy()
    yaw = -np.pi / 2
    prev, log, contacts = None, [], 0

    for frame_i in range(int(DURATION_S * CAMERA_HZ)):
        curr = env._getDroneImages(0, segmentation=False)[0]
        left = right = 0.0
        if prev is not None:
            drive = encode_to_drive_centring(prev, curr, net.n, cti, gain=GAIN)
            for _ in range(NET_STEPS):
                net.step(external_input=drive)
            rate = md.spike_counts_to_rate(net.reset_spike_window(), NET_STEPS * NET_DT)
            left, right = md.decode_lr_pools(rate, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)

        proximity, imbalance = left + right, left - right
        if steering:
            yaw += turn_gain * imbalance / CAMERA_HZ
            speed = CRUISE_SPEED * max(0.15, 1.0 - BRAKE_GAIN * proximity)
        else:
            speed = CRUISE_SPEED
        target = target + np.array([np.cos(yaw), np.sin(yaw), 0.0]) * speed / CAMERA_HZ
        target[2] = ALTITUDE

        for _ in range(STEPS_PER_FRAME):
            s = env._getDroneStateVector(0)
            rpm, _, _ = ctrl.computeControl(
                control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3], cur_quat=s[3:7],
                cur_vel=s[10:13], cur_ang_vel=s[13:16], target_pos=target,
                target_rpy=np.array([0.0, 0.0, yaw]))
            env.step({0: rpm})

        s = env._getDroneStateVector(0)
        if pb.getContactPoints(bodyA=env.DRONE_IDS[0], physicsClientId=env.CLIENT):
            contacts += 1
        log.append((frame_i / CAMERA_HZ, s[0], s[1], s[2], yaw, proximity, imbalance, speed))
        prev = curr

    env.close()
    return np.array(log), contacts, maze


def summarize(label, log, contacts):
    xy = log[:, 1:3]
    path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    # frames where the drone was pinned against something: barely moving
    stuck = int(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 0.005))
    print(f"{label:>26} path {path:6.1f} m   contact frames {contacts:4d}   "
          f"stalled {stuck:4d}/{len(log)}   final z {log[-1,3]:.2f}")
    return path, contacts


results = {}
print("\n[1/3] no steering (straight ahead)")
results["no steering"] = run(steering=False)
print("[2/3] connectome steering")
results["connectome"] = run(steering=True)
print("[3/3] connectome steering, sign flipped")
results["sign flipped"] = run(steering=True, turn_gain=-TURN_GAIN)

print()
for label, (log, contacts, _) in results.items():
    summarize(label, log, contacts)

maze = results["connectome"][2]
cell, rows = maze["cell"], maze["layout"]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
n_rows, n_cols = len(rows), max(len(r) for r in rows)
for r, row in enumerate(rows):
    for c, ch in enumerate(row):
        if ch == "#":
            x = (c - (n_cols - 1) / 2.0) * cell
            y = ((n_rows - 1) / 2.0 - r) * cell
            ax.add_patch(plt.Rectangle((x - cell/2, y - cell/2), cell, cell,
                                       color="0.75", zorder=0))
for (label, (log, _, _)), colour in zip(results.items(),
                                        ("#7f8c8d", "#2471a3", "#c0392b")):
    ax.plot(log[:, 1], log[:, 2], color=colour, lw=2, label=label, zorder=2)
    ax.plot(log[0, 1], log[0, 2], "o", color=colour, ms=7, zorder=3)
ax.plot(*maze["goal"], "*", color="#27ae60", ms=18, label="goal", zorder=3)
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Maze flight: connectome steering from the centring response")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "trajectories.png"), dpi=130)
print(f"\nwrote {OUT_DIR}/trajectories.png")
