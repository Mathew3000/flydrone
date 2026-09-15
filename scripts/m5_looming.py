"""
M5: does the looming pathway fire an escape transient when something
approaches -- and stay quiet when nothing is on a collision course?

The course-control work (M3/M3b) showed a pathway that turns continuously in
proportion to rotation. This is the opposite kind of circuit and the data says
so before any simulation: LC4/LPLC2 -> DNp04 carries synaptic weight 14995 and
-> DNp01 (the Giant Fiber) 11224, where the entire optomotor pathway
HS/H1/H2 -> DNb03 manages 1266. That is a circuit built to fire hard and
rarely. The expected signature is therefore a transient, not a proportional
signal -- which is also why commit 1f09a4f pivoted away from this pathway for
steering.

Four conditions, two of them controls:

  approach slow   object closes at 2 m/s   -> expect a burst
  approach fast   object closes at 4 m/s   -> expect a burst, earlier in time
                  but at the same angular size if the trigger is size-based
  recede          object retreats at 2 m/s -> expect silence: inward motion
                  rectifies to zero in encode_to_drive_looming()
  rotation        drum spins, object parked far away -> the specificity test

That prediction was made and it came true: the first version of
encode_to_drive_looming() pooled horizontal outward motion per hemifield, and
the rotation control did not merely leak -- it dominated, firing the escape
DNs at 90 Hz while an actual approach produced nothing at all. Two more
attempts (spatial divergence; a retinotopic 4x8 patch grid) stayed below 1.0
selectivity too. The cause was not resolution: the encoder had no vertical
motion channel, and rotation versus expansion is precisely the distinction
that needs one. Requiring both axes to read outward brought it to 4.39x at the
encoder and to complete silence in both controls at the network. The history
is kept in encode_to_drive_looming()'s docstring because the two failed
formulations are the ones anybody would reach for first.

Encoder drive peaks at a fixed ANGULAR SIZE (~34 deg) independent of approach
speed, measured before wiring the network up. That is the classic description
of Giant Fiber triggering, and it falls out of the detector and the geometry
rather than being tuned for. This part holds.

READ THE SPECIFICITY NUMBERS WITH CARE. The controls in this script come out
at exactly 0.00000 while the approaches respond, which looks like a clean
result and is not one. The rotation control uses the vertically striped drum;
the approaching object is mosaic-textured. A vertical grating has no vertical
luminance gradient, so the vertical channel reads exactly zero on it -- whether
it rotates OR expands. The comparison therefore separates two textures, not two
motions. With matched textures the encoder manages 1.5x
(connectome/tests/test_looming.py, where the specificity test is an explicit
xfail). A vertically striped object approaching this drone would be missed
entirely, and an isotropically textured rotating room would nearly trigger it.

Outputs (files/looming/, gitignored): response.png
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
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import encode_to_drive_looming
from connectome import motor_decoder as md
from world.room import build_drum, rotate_drum, spawn_object, move_object

PHYSICS_HZ, CAMERA_HZ = 240, 30
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE = 0.0015
# Calibrated here, not inherited: 15 is silent everywhere, 80 lets the rotation
# control leak through (0.00906 where it should be 0). At 40 both approach
# speeds respond and both controls read exactly zero.
GAIN = float(os.environ.get("M5_GAIN", "40"))
DN_SCALE = float(os.environ.get("M5_DN_SCALE", "1.0"))
HOVER_XYZ = np.array([0.0, 0.0, 1.0])
OBJ_HALF = 0.5
START_DIST, STOP_DIST = 9.0, 0.7

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "looming")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading LC4/LPLC2 -> escape DN subnetwork...")
t0 = time.time()
net_spec = local_fetch_looming_subnetwork()
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()
if DN_SCALE != 1.0:
    W = scale_incoming(W, np.concatenate([cti["motor_left"], cti["motor_right"]]), DN_SCALE)
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses "
      f"({time.time()-t0:.1f}s)  gain={GAIN} dn_scale={DN_SCALE}")
DN_TYPES = sorted({k[3:-2] for k in cti if k.startswith("dn_")})


def angular_size(d):
    return float(np.rad2deg(2 * np.arctan(OBJ_HALF / max(d, 1e-3))))


def run_trial(condition, speed=2.0, omega=1.2):
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    drum = build_drum(env.CLIENT, n_bars=24, plane_id=env.PLANE_ID)
    obj = spawn_object(env.CLIENT, half_size=OBJ_HALF, position=(START_DIST, 0, 1.0))
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    for _ in range(PHYSICS_HZ * 2):                       # settle at hover
        s = env._getDroneStateVector(0)
        rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3],
                                        cur_quat=s[3:7], cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                        target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
        env.step({0: rpm})

    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)
    dist = START_DIST if condition != "recede" else STOP_DIST
    drum_angle, yaw_setpoint, prev, log = 0.0, 0.0, None, []
    n_frames = int(abs(START_DIST - STOP_DIST) / speed * CAMERA_HZ)

    for i in range(n_frames):
        if condition in ("approach", "recede"):
            dist += (-speed if condition == "approach" else +speed) / CAMERA_HZ
            move_object(env.CLIENT, obj, (dist, 0, 1.0))
        elif condition == "rotation":                     # drum spins, drone still
            drum_angle += omega / CAMERA_HZ
            rotate_drum(env.CLIENT, drum, drum_angle)
            move_object(env.CLIENT, obj, (START_DIST, 0, 1.0))
        else:                                             # condition == "yaw"
            # The drone turns instead: the whole scene sweeps, including the
            # isotropically textured floor. The drum control alone would not
            # settle the question, because its walls are a vertical grating and
            # a grating is an easy stimulus to look selective on for the wrong
            # reason -- that is exactly how the earlier conjunctive encoder
            # produced a confounded 4.39x.
            yaw_setpoint += omega / CAMERA_HZ
            # Imposed directly rather than flown: the trial loop deliberately
            # does not advance physics (every other condition wants the drone
            # perfectly still, so that the only motion in the image is the
            # stimulus). Asking the PID to turn would require stepping physics
            # and would add its own transients to the very signal being
            # measured.
            quat = pb.getQuaternionFromEuler([0.0, 0.0, yaw_setpoint])
            pb.resetBasePositionAndOrientation(env.DRONE_IDS[0], list(HOVER_XYZ), quat,
                                               physicsClientId=env.CLIENT)
            env.pos[0], env.quat[0] = HOVER_XYZ.copy(), np.array(quat)
            move_object(env.CLIENT, obj, (START_DIST, 0, 1.0))
        curr = env._getDroneImages(0, segmentation=False)[0]

        if prev is not None:
            drive = encode_to_drive_looming(prev, curr, net.n, cti, gain=GAIN)
            for _ in range(NET_STEPS):
                net.step(external_input=drive)
            rate = md.spike_counts_to_rate(net.reset_spike_window(), NET_STEPS * NET_DT)
            left, right = md.decode_lr_pools(rate, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)
            escape = left + right                          # the sum channel: escape magnitude
            drive_sum = float(drive[cti["looming_L"]].mean() + drive[cti["looming_R"]].mean())
            per_type = [float(rate[cti[f"dn_{t}_L"]].mean() + rate[cti[f"dn_{t}_R"]].mean()) / 2
                        for t in DN_TYPES]
        else:
            escape, drive_sum, per_type = 0.0, 0.0, [0.0] * len(DN_TYPES)

        log.append([i / CAMERA_HZ, dist, angular_size(dist), drive_sum, escape] + per_type)
        prev = curr

    env.close()
    return np.array(log)


CONDITIONS = [("approach slow", dict(condition="approach", speed=2.0)),
              ("approach fast", dict(condition="approach", speed=4.0)),
              ("recede", dict(condition="recede", speed=2.0)),
              ("rotation", dict(condition="rotation", speed=2.0)),
              ("self-yaw", dict(condition="yaw", speed=2.0))]

results = {}
for label, kwargs in CONDITIONS:
    print(f"\n[{label}]")
    log = results[label] = run_trial(**kwargs)
    esc = log[:, 4]
    i = int(np.argmax(esc))
    print(f"  peak escape output {esc.max():.5f} at t={log[i,0]:.2f}s, "
          f"dist {log[i,1]:.2f} m, angular size {log[i,2]:.1f} deg")
    print(f"  frames with any output: {int((esc > 0).sum())}/{len(esc)}")

print("\nper-DN-type peak rate (Hz), pooled over both sides:")
print(f"{'condition':>14} " + " ".join(f"{t:>8}" for t in DN_TYPES))
for label, log in results.items():
    print(f"{label:>14} " + " ".join(f"{log[:, 5+j].max():8.1f}" for j in range(len(DN_TYPES))))

peak_sizes = {l: results[l][int(np.argmax(results[l][:, 4])), 2] for l, _ in CONDITIONS[:2]}
print(f"\nangular size at peak: slow {peak_sizes['approach slow']:.1f} deg, "
      f"fast {peak_sizes['approach fast']:.1f} deg")
print("Same angular size across approach speeds -> a size-threshold trigger,")
print("as the Giant Fiber is usually described.")

quiet = {l: float(results[l][:, 4].max()) for l in ("recede", "rotation", "self-yaw")}
loud = max(results[l][:, 4].max() for l, _ in CONDITIONS[:2])
print(f"\nspecificity: approach {loud:.5f} vs recede {quiet['recede']:.5f} "
      f"vs drum rotation {quiet['rotation']:.5f} vs self-yaw {quiet['self-yaw']:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for label, log in results.items():
    axes[0].plot(log[:, 0], log[:, 4], label=label)
for label in ("approach slow", "approach fast"):
    log = results[label]
    axes[1].plot(log[:, 2], log[:, 4], "o-", ms=3, label=label)
axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("escape output (DN pools)")
axes[0].set_title("all conditions over time")
axes[1].set_xlabel("angular size of object [deg]")
axes[1].set_title("approaches, against angular size")
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Looming pathway: LC4/LPLC2 -> escape descending neurons")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "response.png"), dpi=130)
np.savez(os.path.join(OUT_DIR, "looming.npz"), **{l.replace(" ", "_"): a for l, a in results.items()})
print(f"\nwrote {OUT_DIR}/response.png")
