"""
Velocity vs temporal-frequency tuning of the optomotor response.

scripts/m3_optomotor.py established that the response reverses with drum
direction -- at one drum speed (1.2 rad/s). That leaves the obvious question
open: is the pathway tuned, and tuned to what?

The classic fly result (Goetz) is that the optomotor response is tuned to
TEMPORAL FREQUENCY, not to angular velocity. Temporal frequency is
    f = omega / bar_period,  bar_period = 2*pi / n_bars
so widening the bars shifts the velocity optimum but leaves the
temporal-frequency optimum where it was. That is the signature of a
correlation-type (Hassenstein-Reichardt) motion detector, which measures a
correlation delay rather than a speed.

This script sweeps omega at two bar counts and plots the response both ways.
The curves collapsing onto one line when plotted against temporal frequency
means fly-like tuning; collapsing when plotted against omega means the model is
measuring velocity instead.

Result (added after running): on this drum the sweep does NOT adjudicate. The
response curves are broad and flat, argmax and centroid disagree, and the drum
is not a single spatial frequency to begin with -- its bar period in pixels
varies across the field of view and the static mosaic floor is broadband. The
tuning claim is therefore made in connectome/tests/test_optomotor.py against a
single-spatial-frequency grating, where the correlator's optimum sits at 0.25
cycles per frame for both a 16 px and a 32 px period, exactly as predicted.
What this sweep does show cleanly is that the correlator's drive is smooth and
monotonic up to its optimum, where Farneback's becomes erratic above ~5 px of
displacement per frame and eventually reverses.

Prediction, stated before running: this model should show VELOCITY tuning. Its
direction selectivity does not come from the connectome at all -- it is
computed in connectome.medulla_encoder by Farneback optical flow, which
estimates true velocity, and only then injected into T4/T5. The connectome
downstream of that cannot recover a temporal-frequency dependence that the
encoder has already thrown away. A velocity-tuned result is therefore not a
failure of the connectome; it localises exactly which abstraction in this
pipeline is standing in for fly biology.

At high speeds the response should also break down for a mundane reason: once
the drum moves more than half a bar between frames, the flow estimate becomes
ambiguous (the same aliasing that made world.room randomize its bar widths).
That limit is reported alongside, so it is not mistaken for tuning.

Encoder drive and network response are reported separately per point, because
the first run of this conflated them: the response was flat zero below 1.2
rad/s and that looked like low-speed tuning. It was not. The encoder responds
monotonically down to 0.15 rad/s (drive 0.12 at 24 bars); what removes the low
end is the descending layer's firing threshold, which needs a drive around 0.9
to produce any spikes at all. Two gains are swept for the same reason -- to
show whether the usable window moves or widens.
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

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from connectome.local_maleCNS import local_fetch_dn_subnetwork
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import (encode_to_drive_progressive,
                                        encode_to_drive_reichardt)

ENCODERS = {"progressive": encode_to_drive_progressive,
            "reichardt": encode_to_drive_reichardt}
from connectome import motor_decoder as md
from world.room import build_drum, rotate_drum

PHYSICS_HZ, CAMERA_HZ = 240, 30
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE = 0.0015
DN_SCALE = 12.0
SECONDS_PER_POINT = 1.2
# (encoder, gain) pairs, each gain calibrated for its encoder by
# scripts/m3_gain_sweep.py against this drum. The comparison is the point:
# progressive measures velocity via Farneback flow, reichardt correlates a
# delayed signal with its neighbour the way T4/T5 are thought to.
CONFIGS = [("progressive", 4.0), ("reichardt", 10.0)]
HOVER_XYZ = np.array([0.0, 0.0, 1.0])

BAR_COUNTS = [12, 24]
# Finer and reaching higher than the first pass, which left the correlator's
# peak outside the sweep for 12 bars (response still rising at the last point).
# The correlator optimum is predicted at 0.25 cycles/frame = 7.5 Hz, i.e.
# omega = 7.5 * bar_period: 3.93 rad/s at 12 bars, 1.96 at 24. That is always
# exactly half the aliasing limit (0.5 cycles/frame), so it is resolvable for
# any bar width -- but only if the grid goes far enough.
OMEGAS = [0.3, 0.6, 1.0, 1.4, 2.0, 2.8, 3.9, 5.5, 7.0]

# Horizontal extent the camera sees, in radians of drum -- 60 deg FOV. Used to
# express the stimulus in image pixels per frame, which is what decides whether
# the flow estimate is still unambiguous.
CAMERA_FOV_RAD = np.deg2rad(60.0)
IMG_WIDTH_PX = 64

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "optomotor")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading real MaleCNS subnetwork (decode layer: dn)...")
t0 = time.time()
net_spec = local_fetch_dn_subnetwork()
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()
W = scale_incoming(W, np.concatenate([cti["motor_left"], cti["motor_right"]]), DN_SCALE)
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses "
      f"({time.time()-t0:.1f}s)")


def measure(env, drum, omega, encoder_name, gain):
    """Drum turns at `omega`, drone held still.

    Returns (mean decoded yaw, mean encoder drive difference L-R). The drive is
    returned alongside so a zero response can be attributed to the right stage
    instead of guessed at.
    """
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    angle, prev, out, drives = 0.0, None, [], []
    for _ in range(int(SECONDS_PER_POINT * CAMERA_HZ)):
        angle += omega / CAMERA_HZ
        rotate_drum(env.CLIENT, drum, angle)
        curr = env._getDroneImages(0, segmentation=False)[0]
        if prev is not None:
            drive = ENCODERS[encoder_name](prev, curr, net.n, cti, gain=gain)
            drives.append(float(drive[cti["visual_L"]].mean() - drive[cti["visual_R"]].mean()))
            for _ in range(NET_STEPS):
                net.step(external_input=drive)
            rate_hz = md.spike_counts_to_rate(net.reset_spike_window(), NET_STEPS * NET_DT)
            left, right = md.decode_lr_pools(rate_hz, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)
            out.append(md.lr_to_thrust_yaw(left, right)[1])
        for _ in range(PHYSICS_HZ // CAMERA_HZ):
            s = env._getDroneStateVector(0)
            rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3],
                                            cur_quat=s[3:7], cur_vel=s[10:13],
                                            cur_ang_vel=s[13:16], target_pos=HOVER_XYZ,
                                            target_rpy=np.zeros(3))
            env.step({0: rpm})
        prev = curr
    return float(np.mean(out)), float(np.mean(drives))


results = {}
for n_bars in BAR_COUNTS:
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    drum = build_drum(env.CLIENT, n_bars=n_bars, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    for _ in range(PHYSICS_HZ * 2):        # settle at hover
        s = env._getDroneStateVector(0)
        rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3],
                                        cur_quat=s[3:7], cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                        target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
        env.step({0: rpm})

    bar_period = 2 * np.pi / n_bars
    bar_px = IMG_WIDTH_PX * bar_period / CAMERA_FOV_RAD
    print(f"\n{n_bars} bars (bar period {bar_period:.3f} rad, ~{bar_px:.1f} px in the frame)")
    print(f"{'omega':>7} {'temp.f':>7} {'px/frm':>7} {'encoder':>12} {'drive L-R':>10} "
          f"{'response':>10}  {'note':>9}")
    for encoder_name, gain in CONFIGS:
        rows = []
        for omega in OMEGAS:
            # (CCW - CW)/2: the signed, direction-selective part, with any
            # direction-independent offset cancelled out.
            r_ccw, d_ccw = measure(env, drum, +omega, encoder_name, gain)
            r_cw, d_cw = measure(env, drum, -omega, encoder_name, gain)
            resp, drive = (r_ccw - r_cw) / 2.0, (d_ccw - d_cw) / 2.0
            temporal_f = omega / bar_period
            px_per_frame = IMG_WIDTH_PX * (omega / CAMERA_HZ) / CAMERA_FOV_RAD
            aliased = px_per_frame > bar_px / 2
            note = "ALIASED" if aliased else ("silent" if resp == 0 else "")
            rows.append((omega, temporal_f, px_per_frame, drive, resp, float(aliased)))
            print(f"{omega:7.2f} {temporal_f:7.2f} {px_per_frame:7.2f} {encoder_name:>12} "
                  f"{drive:+10.4f} {resp:+10.5f}  {note:>9}")
        results[(n_bars, encoder_name)] = np.array(rows)
    env.close()

# Raw points saved alongside the figure: this sweep costs ~10 minutes of
# simulation, and every plotting tweak would otherwise re-run it.
np.savez(os.path.join(OUT_DIR, "tuning.npz"),
         **{f"bars{n}_{enc}": arr for (n, enc), arr in results.items()})

# --- plot: does either encoder collapse the two bar widths onto one curve? ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
colours = {12: "#c0392b", 24: "#2471a3"}
styles = {"progressive": "--", "reichardt": "-"}

for (n_bars, enc), arr in results.items():
    omega, tf, px, drive, resp, aliased = arr.T
    label = f"{n_bars} bars, {enc}"
    axes[0].plot(omega, np.abs(drive), "o" + styles[enc], color=colours[n_bars], label=label)
    axes[1].plot(omega, np.abs(resp), "o" + styles[enc], color=colours[n_bars], label=label)
    axes[2].plot(tf, np.abs(resp), "o" + styles[enc], color=colours[n_bars], label=label)

axes[0].set_title("encoder drive")
axes[0].set_ylabel("|drive L-R|")
axes[0].set_yscale("log")
axes[0].set_xlabel("drum velocity [rad/s]")
axes[1].set_title("network response vs VELOCITY")
axes[1].set_ylabel("|direction-selective response|")
axes[1].set_xlabel("drum velocity [rad/s]")
axes[2].set_title("network response vs TEMPORAL FREQUENCY")
axes[2].set_xlabel("temporal frequency [Hz]")

from matplotlib.ticker import ScalarFormatter
for ax, ticks in zip(axes, (OMEGAS, OMEGAS, None)):
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.minorticks_off()   # default log minor ticks overlap illegibly here
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
fig.suptitle("Which axis collapses the two bar widths? (dashed = optic flow, solid = correlator)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "tuning.png"), dpi=130)
print(f"\nwrote {OUT_DIR}/tuning.png")

print("\nTuning summary. Read the flatness column first: an argmax on a flat curve")
print("is not a peak, and on this stimulus the curves are flat.")
print(f"{'encoder':>12} {'bars':>5} {'argmax Hz':>10} {'centroid Hz':>12} "
      f"{'cyc/frame':>10} {'flatness':>9}")
for (n_bars, enc), arr in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    live = arr[arr[:, 5] == 0]          # aliased points are not part of any tuning curve
    r = np.abs(live[:, 4])
    if not len(live) or r.sum() == 0:
        print(f"{enc:>12} {n_bars:5d}      silent")
        continue
    i = int(np.argmax(r))
    centroid = float((live[:, 1] * r).sum() / r.sum())
    # How much of the peak does the curve keep across the whole sampled range?
    # Near 1.0 means there is no peak to speak of.
    flatness = float(r[r > 0].min() / r.max())
    edge = "*" if i in (0, len(live) - 1) else " "
    print(f"{enc:>12} {n_bars:5d} {live[i,1]:9.2f}{edge} {centroid:12.2f} "
          f"{centroid/CAMERA_HZ:10.3f} {flatness:9.2f}")
print("  * = argmax sits at the edge of the sweep, i.e. the peak is outside it")

print("\nSame temporal frequency across bar counts -> correlation tuning, as in the fly.")
print("Same velocity                             -> an optic-flow estimator.")
print("\nCaveat this sweep cannot escape: a rotating drum is not a single spatial")
print("frequency. Bar period in pixels varies across the field of view, and the")
print("mosaic floor is broadband and does not rotate. Any temporal-frequency")
print("optimum is smeared across that mixture. The controlled version of this")
print("measurement -- one grating, one spatial frequency -- lives in")
print("connectome/tests/test_optomotor.py and is where the tuning claim is made.")
