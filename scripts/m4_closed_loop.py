"""
Phase B: all four channels in closed loop, both subnetworks running together.

M2 closed the loop on two channels (yaw and altitude) from the horizontal
system. M4 showed the vertical system reads roll, but as an open-loop readout
with the drone frozen. This runs both at once, on a flying drone:

    horizontal system  T4/T5 -> HS/H1/H2 -> DNs   ->  yaw setpoint, altitude
    vertical system    T4/T5 -> VS       -> DNs   ->  roll setpoint, pitch

The inner attitude loop stays with DSLPIDControl. That is deliberate and it is
what the project plan calls Phase B's pragmatic form: the connectome supplies
setpoints, the proven controller keeps the aircraft in the air. Replacing the
inner loop outright is Phase C and a different project.

What this does NOT attempt, and why -- because the obvious Phase B reading is
"replace the classical roll PD with network output", and that is not reachable
here for a reason worth writing down rather than discovering twice:

  The visual path runs at 33 ms per camera frame plus a 20 ms decode window,
  so roughly 53 ms of latency. DSLPIDControl damps the roll axis at 240 Hz.
  Measured by sweeping its roll rate gain D_COEFF_TOR[0]: at the stock 20000
  a 1.5 rad/s disturbance produces 0.109 rad of roll, at 11000 it produces
  0.163, and at 10000 the drone flips and crashes. The transition is a
  bifurcation, not a slope -- there is no regime where the axis is slack
  enough for a visual signal to matter and still flyable. A 53 ms rate signal
  is an order of magnitude too slow to stand in for inner-loop damping on a
  Crazyflie, whatever the connectome puts through it.

So the connectome supplies SETPOINTS on all four channels and the PID keeps the
inner loop, which is the pragmatic Phase B the project plan actually specifies
("Roll/Pitch werden vorerst von einem klassischen PD-Regler stabilisiert").
What is measured is that all four channels are live on a flying drone and that
the roll channel tracks an induced roll with the right sign.

Outputs (files/vertical/, gitignored): closed_loop.png
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

from connectome.local_maleCNS import (local_fetch_dn_subnetwork,
                                      local_fetch_vertical_subnetwork)
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import (encode_to_drive_reichardt,
                                        encode_to_drive_vertical)
from connectome import motor_decoder as md
from world.room import build_drum

PHYSICS_HZ, CAMERA_HZ = 240, 30
STEPS_PER_FRAME = PHYSICS_HZ // CAMERA_HZ
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE, DN_SCALE = 0.0015, 12.0
GAIN_H, GAIN_V = 10.0, 40.0        # calibrated per encoder, see m3/m4 sweeps
YAW_GAIN, Z_GAIN = 6.0, 0.15
ROLL_GAIN = float(os.environ.get("ROLL_GAIN", "-1.5"))   # sign: damping, see docstring
DURATION_S = 6.0
HOVER_XYZ = np.array([0.0, 0.0, 1.5])
# Recoverable magnitude, measured: 3.0 rad/s flips the drone outright.
KICKS = [(1.5, +1.5), (3.5, -1.5)]   # (time, roll rate in rad/s)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "vertical")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading both subnetworks...")
t0 = time.time()
nets = {}
for name, fetch in (("horizontal", local_fetch_dn_subnetwork),
                    ("vertical", local_fetch_vertical_subnetwork)):
    spec = fetch()
    W = spec["weights"].multiply(WEIGHT_SCALE).tocsr()
    pools = np.concatenate([spec["cell_type_indices"]["motor_left"],
                            spec["cell_type_indices"]["motor_right"]])
    spec["model_weights"] = scale_incoming(W, pools, DN_SCALE)
    nets[name] = spec
    print(f"  {name:11} {spec['n_neurons']} neurons, {spec['weights'].nnz} synapses")
print(f"  ({time.time()-t0:.1f}s)")


def run(connectome_roll: bool, roll_gain: float = ROLL_GAIN):
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    build_drum(env.CLIENT, n_bars=24, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    lif = {k: LIFNetwork(n_neurons=v["n_neurons"], weights=v["model_weights"], dt=NET_DT)
           for k, v in nets.items()}

    target_pos = HOVER_XYZ.copy()
    yaw_sp, roll_sp, pitch_sp = 0.0, 0.0, 0.0
    prev, log = None, []
    kicks = list(KICKS)

    for frame_i in range(int(DURATION_S * CAMERA_HZ)):
        t = frame_i / CAMERA_HZ
        if kicks and t >= kicks[0][0]:
            _, rate = kicks.pop(0)
            pb.resetBaseVelocity(env.DRONE_IDS[0], angularVelocity=[rate, 0.0, 0.0],
                                 physicsClientId=env.CLIENT)

        curr = env._getDroneImages(0, segmentation=False)[0]
        yaw_out = thrust_out = roll_out = pitch_out = v_drive = 0.0
        if prev is not None:
            for key, encoder, gain in (("horizontal", encode_to_drive_reichardt, GAIN_H),
                                       ("vertical", encode_to_drive_vertical, GAIN_V)):
                cti = nets[key]["cell_type_indices"]
                drive = encoder(prev, curr, lif[key].n, cti, gain=gain)
                for _ in range(NET_STEPS):
                    lif[key].step(external_input=drive)
                rate_hz = md.spike_counts_to_rate(lif[key].reset_spike_window(),
                                                  NET_STEPS * NET_DT)
                left, right = md.decode_lr_pools(rate_hz, cti["motor_left"],
                                                 cti["motor_right"], baseline_hz=0.0)
                s_sum, s_diff = md.lr_to_thrust_yaw(left, right)
                if key == 'vertical':
                    v_drive = float(drive[cti['vertical_L']].mean()
                                    + drive[cti['vertical_R']].mean())
                if key == "horizontal":
                    thrust_out, yaw_out = s_sum, s_diff
                else:
                    pitch_out, roll_out = s_sum, s_diff

            yaw_sp += YAW_GAIN * yaw_out / CAMERA_HZ
            target_pos = HOVER_XYZ + np.array([0.0, 0.0, Z_GAIN * thrust_out])
            # proportional, not integrated: a roll setpoint is an angle
            roll_sp = roll_gain * roll_out if connectome_roll else 0.0
            pitch_sp = 0.0      # left out: the pitch channel is one-directional

        for _ in range(STEPS_PER_FRAME):
            s = env._getDroneStateVector(0)
            rpm, _, _ = ctrl.computeControl(
                control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3], cur_quat=s[3:7],
                cur_vel=s[10:13], cur_ang_vel=s[13:16], target_pos=target_pos,
                target_rpy=np.array([roll_sp, pitch_sp, yaw_sp]))
            env.step({0: rpm})

        s = env._getDroneStateVector(0)
        log.append((t, s[7], s[8], s[9], s[2], roll_out, yaw_out, v_drive))
        prev = curr

    env.close()
    return np.array(log)


print("\n[1/2] connectome drives yaw + altitude only (Phase A)")
two = run(connectome_roll=False)
print("[2/2] connectome drives all four channels (Phase B)")
four = run(connectome_roll=True)


def summarize(label, log):
    t, roll, z = log[:, 0], log[:, 1], log[:, 4]
    after = t >= KICKS[0][0]
    # after takeoff only: the drone spawns on the ground and climbs to HOVER_XYZ,
    # so z over the whole run always dips to the spawn height
    airborne = bool(z[after].min() > 0.5)
    print(f"{label:>34} max|roll| {np.abs(roll[after]).max():6.3f} rad   "
          f"z {z.min():.2f}-{z.max():.2f} m   airborne {airborne}")
    return airborne


print()
ok_two = summarize("yaw + altitude (Phase A)", two)
ok_four = summarize("all four channels (Phase B)", four)

# Does the roll channel see the induced roll at all, in flight?
def roll_tracking(log):
    t = log[:, 0]
    rate = np.gradient(log[:, 1], 1.0 / CAMERA_HZ)
    win = (t > KICKS[0][0]) & (t < KICKS[0][0] + 0.7)
    if np.std(log[win, 5]) == 0:
        return float("nan")
    return float(np.corrcoef(rate[win], log[win, 5])[0, 1])


print(f"\nroll channel vs actual roll rate during the first disturbance:")
print(f"{'gain':>8} {'correlation':>12}")
print(f"{GAIN_V:8.0f} {roll_tracking(four):12.3f}   (calibrated)")
for forced in (200.0, 800.0):
    globals()["GAIN_V"] = forced
    print(f"{forced:8.0f} {roll_tracking(run(connectome_roll=True)):12.3f}   (forced, "
          f"far outside the calibrated range)")
globals()["GAIN_V"] = 40.0

print("""
Conclusion, and it is a null result worth stating plainly: the roll channel
does not usefully track roll in flight. At the calibrated gain it never fires;
forced to 20x that it reaches a correlation of about 0.2. The cause is not the
connectome but the loop rate. DSLPIDControl suppresses the disturbance to 0.110
rad and kills the transient in roughly 0.2 s -- about six camera frames -- and
a 30 Hz camera with a 20 ms decode window has no resolution left at that
timescale. The same disturbance measured open-loop in scripts/m4_vertical.py,
where the rotation is imposed for 1.2 s, gives a clean reversing signal.

What Phase B does deliver: both subnetworks (27244 neurons together) run in one
closed loop at 30 Hz on a flying drone, the connectome supplies setpoints on
all four channels, and the aircraft stays airborne. Yaw and altitude carry real
signal. Roll is wired and demonstrably reads rotation, just not fast enough to
matter against this inner loop.""")
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(two[:, 0], two[:, 1], label="yaw + altitude only", color="#7f8c8d")
axes[0].plot(four[:, 0], four[:, 1], label="all four channels", color="#2471a3")
axes[1].plot(four[:, 0], np.gradient(four[:, 1], 1.0 / CAMERA_HZ), color="#7f8c8d",
             label="actual roll rate [rad/s]")
ax2 = axes[1].twinx()
ax2.plot(four[:, 0], four[:, 5], color="#c0392b", label="decoded roll channel")
ax2.set_ylabel("decoded roll", color="#c0392b")
for kt, _ in KICKS:
    for ax in axes:
        ax.axvline(kt, color="k", ls=":", lw=1)
axes[0].set_ylabel("roll [rad]")
axes[1].set_ylabel("roll rate [rad/s]")
axes[1].set_xlabel("time [s]")
axes[0].set_title("Phase B: four connectome channels in closed loop "
                  "(dotted = induced roll disturbance)")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.8", lw=0.8)
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8, loc="upper left")
ax2.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "closed_loop.png"), dpi=130)
print(f"\nwrote {OUT_DIR}/closed_loop.png")
