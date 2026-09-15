"""
M4 / Phase B: roll and pitch from the vertical system.

M3 took yaw from the horizontal system (T4/T5 -> HS/H1/H2 -> descending
neurons). This is the same architecture one axis over: T4/T5 -> VS ->
descending neurons, with vertical image motion as the input.

M4 was previously recorded as blocked, on the grounds that MaleCNS had no
vertical-system cells. It has them. The search that concluded otherwise asked
for `VS\\d+`, and MaleCNS collapses all eight subtypes into a single type named
exactly "VS" -- its flywireType field spells this out as
"VS1,...,VS8". 18 cells, 9 per side, receiving total weight 158026 from T4/T5
(8779 per cell against an HS cell's 13164). The block was a regex, not a gap in
the data.

No new decoder is needed. Roll turns the two hemifields' vertical motion in
opposite directions and pitch turns them the same way, so
motor_decoder.lr_to_thrust_yaw()'s difference and sum already separate them --
the identical pair of channels that carry yaw and thrust on the horizontal
side.

Conditions, each an imposed rotation at constant rate, with static baselines
between and the order counterbalanced across repeats:

  roll +/-   expect the roll channel (left minus right) to reverse
  pitch +/-  expect the pitch channel (left plus right); only ONE direction is
             expected to register, since each population is rectified and
             signals downward motion only (see encode_to_drive_vertical)
  yaw        the specificity control: a turn must not masquerade as a roll

Rotations are imposed directly rather than flown, for the reason
scripts/m5_looming.py gives: the PID's own transients would land in the signal
being measured.

Outputs (files/vertical/, gitignored): response.png
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

from connectome.local_maleCNS import local_fetch_vertical_subnetwork
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import encode_to_drive_vertical
from connectome import motor_decoder as md
from world.room import build_drum

PHYSICS_HZ, CAMERA_HZ = 240, 30
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE = 0.0015
GAIN = float(os.environ.get("M4_GAIN", "40"))
DN_SCALE = float(os.environ.get("M4_DN_SCALE", "12"))
OMEGA = 1.2
N_REPEATS = 3
HOVER_XYZ = np.array([0.0, 0.0, 1.5])

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "vertical")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading T4/T5 -> VS -> descending subnetwork...")
t0 = time.time()
net_spec = local_fetch_vertical_subnetwork()
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()
W = scale_incoming(W, np.concatenate([cti["motor_left"], cti["motor_right"]]), DN_SCALE)
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses "
      f"({time.time()-t0:.1f}s)  gain={GAIN} dn_scale={DN_SCALE}")


def protocol():
    phases = []
    for rep in range(N_REPEATS):
        axes = [("roll", +1), ("roll", -1), ("pitch", +1), ("pitch", -1), ("yaw", +1)]
        if rep % 2:
            axes = axes[::-1]
        for axis, sign in axes:
            phases.append(("static", 0.8, None, 0))
            phases.append((axis, 1.2, axis, sign))
    return phases


def run():
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    build_drum(env.CLIENT, n_bars=24, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    for _ in range(PHYSICS_HZ * 2):                        # settle at hover
        s = env._getDroneStateVector(0)
        rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3],
                                        cur_quat=s[3:7], cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                        target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
        env.step({0: rpm})

    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)
    rpy = np.zeros(3)
    prev, log, t = None, [], 0.0

    for label, duration, axis, sign in protocol():
        if axis is None:
            # Level the drone for every baseline. Without this the attitude
            # accumulates across blocks -- each stimulus block turns the drone
            # by omega*duration and never gives it back -- so later repeats
            # start from extreme attitudes and measure something else. It
            # showed up as a standard deviation equal to the mean: one repeat
            # responded, the next did not.
            rpy[:] = 0.0
        for _ in range(int(duration * CAMERA_HZ)):
            if axis is not None:
                rpy[{"roll": 0, "pitch": 1, "yaw": 2}[axis]] += sign * OMEGA / CAMERA_HZ
            quat = pb.getQuaternionFromEuler(list(rpy))
            pb.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(HOVER_XYZ), quat,
                                               physicsClientId=env.CLIENT)
            env.pos[0], env.quat[0] = HOVER_XYZ.copy(), np.array(quat)
            curr = env._getDroneImages(0, segmentation=False)[0]

            if prev is not None:
                drive = encode_to_drive_vertical(prev, curr, net.n, cti, gain=GAIN)
                for _ in range(NET_STEPS):
                    net.step(external_input=drive)
                rate = md.spike_counts_to_rate(net.reset_spike_window(), NET_STEPS * NET_DT)
                left, right = md.decode_lr_pools(rate, cti["motor_left"], cti["motor_right"],
                                                 baseline_hz=0.0)
                pitch_ch, roll_ch = md.lr_to_thrust_yaw(left, right)   # sum, difference
            else:
                pitch_ch, roll_ch = 0.0, 0.0

            key = "static" if axis is None else f"{axis} {'+' if sign > 0 else '-'}"
            log.append((t, key, roll_ch, pitch_ch))
            prev = curr
            t += 1.0 / CAMERA_HZ

    env.close()
    return log


log = run()
keys = ["static", "roll +", "roll -", "pitch +", "pitch -", "yaw +"]
blocks = {k: [] for k in keys}
start = 0
for i in range(1, len(log) + 1):
    if i == len(log) or log[i][1] != log[start][1]:
        k = log[start][1]
        seg = log[start:i]
        blocks[k].append((float(np.mean([r[2] for r in seg])),
                          float(np.mean([r[3] for r in seg]))))
        start = i

print(f"\n{'condition':>10} {'n':>3} {'roll channel':>14} {'sd':>9} {'pitch channel':>15}")
means = {}
for k in keys:
    v = np.array(blocks[k])
    if not len(v):
        continue
    means[k] = (v[:, 0].mean(), v[:, 1].mean())
    print(f"{k:>10} {len(v):3d} {v[:,0].mean():+14.5f} {v[:,0].std():9.5f} {v[:,1].mean():15.5f}")

rp, rm = np.array([b[0] for b in blocks["roll +"]]), np.array([b[0] for b in blocks["roll -"]])
separated = rp.min() > rm.max() or rm.min() > rp.max()
print(f"\nroll reversal (n={len(rp)} per direction)")
print(f"  roll+ {rp.mean():+.5f} +- {rp.std():.5f}   roll- {rm.mean():+.5f} +- {rm.std():.5f}")
print(f"  opposite signs: {rp.mean()*rm.mean() < 0}   non-overlapping: {separated}")
print(f"  => roll is read out from the vertical system: "
      f"{rp.mean()*rm.mean() < 0 and separated}")

yaw_roll = abs(means["yaw +"][0])
roll_mag = max(abs(rp.mean()), abs(rm.mean()))
print(f"\nspecificity: roll {roll_mag:.5f} vs yaw leaking into the roll channel "
      f"{yaw_roll:.5f}  ({roll_mag/max(yaw_roll,1e-9):.1f}x)")
print(f"pitch channel: pitch+ {means['pitch +'][1]:.5f}  pitch- {means['pitch -'][1]:.5f} "
      f"(one direction only is expected -- see encode_to_drive_vertical)")

fig, ax = plt.subplots(figsize=(11, 4))
t = [r[0] for r in log]
ax.plot(t, [r[2] for r in log], label="roll channel (left - right)", color="#c0392b")
ax.plot(t, [r[3] for r in log], label="pitch channel (left + right)", color="#2471a3", alpha=0.7)
colours = {"roll +": "#c0392b", "roll -": "#e67e22", "pitch +": "#2471a3",
           "pitch -": "#27ae60", "yaw +": "#7f8c8d"}
start = 0
for i in range(1, len(log) + 1):
    if i == len(log) or log[i][1] != log[start][1]:
        k = log[start][1]
        if k != "static":
            ax.axvspan(log[start][0], log[i-1][0], color=colours[k], alpha=0.13)
        start = i
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("time [s]"); ax.set_ylabel("decoded output")
ax.set_title("Vertical system: T4/T5 -> VS -> descending neurons "
             "(shaded: red/orange roll, blue/green pitch, grey yaw)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "response.png"), dpi=130)
print(f"\nwrote {OUT_DIR}/response.png")
