"""
M2 in a textured room: does giving the fly something to look at revive the
closed loop?

scripts/m2_real_connectome.py flies the real MaleCNS connectome in
VisionAviary's default world -- a bare ground plane. There is nothing in that
image to produce optical flow, so the connectome's decoded output drops to
exactly 0.000 once the drone stops climbing and the yaw setpoint freezes: the
PID alone keeps it airborne. This script runs the identical control loop twice,
once in that empty world and once inside world.room's bar-textured room, and
prints both so the difference is measured rather than asserted.

The interesting number is not "does it stay airborne" (the PID handles that
either way) but `active`: the fraction of camera frames where the connectome
produced any non-zero output at all, measured over the steady-state window
after takeoff.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from connectome.local_maleCNS import local_fetch_lr_subnetwork
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import encode_to_drive_hemifield
from connectome import motor_decoder as md
from world.room import build_room

PHYSICS_HZ = 240
CAMERA_HZ = 30
STEPS_PER_CAMERA_FRAME = PHYSICS_HZ // CAMERA_HZ
NET_DT = 0.1e-3
NET_STEPS_PER_CAMERA_FRAME = int(round((1.0 / CAMERA_HZ) / NET_DT))
GAIN = 8.0
WEIGHT_SCALE = 0.0015   # see connectome/tests/test_real_connectome.py
YAW_GAIN = 6.0
Z_GAIN = 0.15
DURATION_S = 6.0
HOVER_XYZ = np.array([0.0, 0.0, 1.0])

print("Loading real MaleCNS T4/T5 -> HS/H1/H2 subnetwork from local bulk data...")
t0 = time.time()
net_spec = local_fetch_lr_subnetwork()
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses ({time.time()-t0:.1f}s)")
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()


def run(with_room: bool):
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    # After reset(): reset() calls p.resetSimulation(), which would wipe the room.
    if with_room:
        build_room(env.CLIENT, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)

    target_pos = HOVER_XYZ.copy()
    yaw_setpoint = 0.0
    prev_frame = None
    control_timestep = 1.0 / PHYSICS_HZ
    log = []

    for frame_i in range(int(DURATION_S * CAMERA_HZ)):
        curr_frame = env._getDroneImages(0, segmentation=False)[0]

        if prev_frame is not None:
            drive = encode_to_drive_hemifield(prev_frame, curr_frame, net.n, cti, gain=GAIN)
            for _ in range(NET_STEPS_PER_CAMERA_FRAME):
                net.step(external_input=drive)
            counts = net.reset_spike_window()
            rate_hz = md.spike_counts_to_rate(counts, NET_STEPS_PER_CAMERA_FRAME * NET_DT)
            left, right = md.decode_lr_pools(rate_hz, cti["motor_left"], cti["motor_right"], baseline_hz=0.0)
            thrust_out, yaw_out = md.lr_to_thrust_yaw(left, right)
            yaw_setpoint += YAW_GAIN * yaw_out * (1.0 / CAMERA_HZ)
            target_pos = HOVER_XYZ + np.array([0.0, 0.0, Z_GAIN * thrust_out])
        else:
            thrust_out, yaw_out = 0.0, 0.0

        for _ in range(STEPS_PER_CAMERA_FRAME):
            state = env._getDroneStateVector(0)
            rpm, _, _ = ctrl.computeControl(
                control_timestep=control_timestep,
                cur_pos=state[0:3], cur_quat=state[3:7], cur_vel=state[10:13], cur_ang_vel=state[13:16],
                target_pos=target_pos, target_rpy=np.array([0.0, 0.0, yaw_setpoint]),
            )
            env.step({0: rpm})

        state = env._getDroneStateVector(0)
        log.append((frame_i / CAMERA_HZ, *state[7:10], state[2], yaw_setpoint, thrust_out, yaw_out))
        prev_frame = curr_frame

    env.close()
    return np.array(log)


def summarize(label, log):
    t, roll, pitch, yaw, z, yaw_sp, thrust_o, yaw_o = log.T
    steady = slice(len(log) * 2 // 3, None)
    stable = (np.abs(roll[steady]).max() < 0.05 and np.abs(pitch[steady]).max() < 0.05
              and abs(z[steady].mean() - HOVER_XYZ[2]) < 0.1)
    active = float(np.mean((np.abs(thrust_o[steady]) > 0) | (np.abs(yaw_o[steady]) > 0)))
    print(f"\n--- {label} ---")
    print(f"stable at steady state        : {stable}")
    print(f"z at steady state             : {z[steady].mean():.3f} m (target {HOVER_XYZ[2]})")
    print(f"yaw range over run            : {yaw.max()-yaw.min():.4f} rad")
    print(f"yaw setpoint end              : {yaw_sp[-1]:+.4f} rad")
    print(f"connectome active (steady)    : {active*100:.0f}% of frames")
    print(f"mean |yaw_out| (steady)       : {np.abs(yaw_o[steady]).mean():.5f}")
    print(f"mean |thrust_out| (steady)    : {np.abs(thrust_o[steady]).mean():.5f}")
    return stable, active


print("\n[1/2] empty world (as in scripts/m2_real_connectome.py)")
log_empty = run(with_room=False)
print("\n[2/2] bar-textured room (world.room)")
log_room = run(with_room=True)

stable_e, active_e = summarize("EMPTY WORLD", log_empty)
stable_r, active_r = summarize("TEXTURED ROOM", log_room)

print(f"\n=> drone airborne in both worlds: {stable_e and stable_r}")
print(f"=> connectome alive at steady state: empty {active_e*100:.0f}% -> room {active_r*100:.0f}%")
