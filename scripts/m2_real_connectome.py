"""
M2, real-connectome version: the actual fruit fly connectome (MaleCNS
T4/T5 -> HS/H1/H2, the real optomotor course-control pathway, fetched from
the local bulk data -- see connectome/local_maleCNS.py) controlling a
simulated drone's yaw/altitude in closed loop, instead of the synthetic
make_toy_network() used in scripts/m2_closed_loop.py.

Same architecture as m2_closed_loop.py (kept as-is, as a pure
software-wiring regression test that doesn't need the ~1.1GB local data
download): camera frame -> medulla encoder -> LIF network -> motor decoder
-> nudges DSLPIDControl's yaw/altitude setpoint -> drone moves -> repeat.
Roll/pitch/position holding is still DSLPIDControl's job (a proven PID
controller, not reimplemented here) -- only yaw and altitude setpoints come
from the fly connectome, exactly as in the project plan's Phase A.

The two real-data-specific pieces vs. m2_closed_loop.py:
  - local_fetch_lr_subnetwork() instead of make_toy_network(): a real
    13,597-neuron / 207,059-synapse subnetwork, split by anatomical
    somaSide into visual_L/visual_R (T4/T5) and motor_left/motor_right
    (HS/H1/H2) pools.
  - encode_to_drive_hemifield() instead of encode_to_drive(): drives the
    visual_L/visual_R pools from left/right image-half motion energy (the
    anatomically-grounded retinotopic split), and WEIGHT_SCALE, which
    rescales the real connectome's raw synapse-count weights down before
    they're used as LIFNetwork weights -- see
    connectome/tests/test_real_connectome.py's docstring for why this is
    needed (unscaled, the real weights saturate every motor neuron
    regardless of which side is stimulated).
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

PHYSICS_HZ = 240
CAMERA_HZ = 30
STEPS_PER_CAMERA_FRAME = PHYSICS_HZ // CAMERA_HZ  # 8
NET_DT = 0.1e-3
DECODE_WINDOW_S = 20e-3
NET_STEPS_PER_CAMERA_FRAME = int(round((1.0 / CAMERA_HZ) / NET_DT))  # ~333
GAIN = 8.0                   # encoder input drive gain (same as the toy network)
WEIGHT_SCALE = 0.0015        # real-connectome synapse-weight rescale -- see
                              # connectome/tests/test_real_connectome.py
YAW_GAIN = 6.0                # rad/s of yaw-setpoint drift per unit (thrust,yaw) decoder output
Z_GAIN = 0.15                 # m of altitude-setpoint nudge per unit decoder output
DURATION_S = 6.0
HOVER_XYZ = np.array([0.0, 0.0, 1.0])

print("Loading real MaleCNS T4/T5 -> HS/H1/H2 subnetwork from local bulk data...")
t0 = time.time()
net_spec = local_fetch_lr_subnetwork()
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses ({time.time()-t0:.1f}s)")
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()

env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
env.reset()
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)

target_pos = HOVER_XYZ.copy()
yaw_setpoint = 0.0
prev_frame = None
control_timestep = 1.0 / PHYSICS_HZ

log = []  # (t, roll, pitch, yaw, z, yaw_setpoint, thrust_out, yaw_out)
n_frames = int(DURATION_S * CAMERA_HZ)
step_count = 0

for frame_i in range(n_frames):
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
        cur_pos, cur_quat, cur_rpy = state[0:3], state[3:7], state[7:10]
        cur_vel, cur_ang_vel = state[10:13], state[13:16]
        rpm, _, _ = ctrl.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos, cur_quat=cur_quat, cur_vel=cur_vel, cur_ang_vel=cur_ang_vel,
            target_pos=target_pos, target_rpy=np.array([0.0, 0.0, yaw_setpoint]),
        )
        env.step({0: rpm})
        step_count += 1

    state = env._getDroneStateVector(0)
    log.append((frame_i / CAMERA_HZ, *state[7:10], state[2], yaw_setpoint, thrust_out, yaw_out))
    prev_frame = curr_frame

env.close()

log = np.array(log)
t, roll, pitch, yaw, z, yaw_sp, thrust_o, yaw_o = log.T
print(f"\n{'t':>6} {'roll':>7} {'pitch':>7} {'yaw':>7} {'z':>6} {'yaw_sp':>7} {'thr_out':>8} {'yaw_out':>8}")
for row in log[::5]:
    print(" ".join(f"{v:7.3f}" for v in row))

print(f"\nroll:  max|.|={np.abs(roll).max():.3f} rad")
print(f"pitch: max|.|={np.abs(pitch).max():.3f} rad")
print(f"z:     min={z.min():.3f} max={z.max():.3f} (target {HOVER_XYZ[2]})")
print(f"yaw:   start={yaw[0]:.3f} end={yaw[-1]:.3f} range={yaw.max()-yaw.min():.3f} rad")
print(f"yaw_setpoint: start={yaw_sp[0]:.3f} end={yaw_sp[-1]:.3f}")

# Steady-state window only (last third of the run) -- the first second is
# takeoff from the ground spawn point, which is expected to pass through low
# z / larger errors and isn't a "did it stay stable" question.
steady = slice(len(log) * 2 // 3, None)
stable = (np.abs(roll[steady]).max() < 0.05 and np.abs(pitch[steady]).max() < 0.05
          and abs(z[steady].mean() - HOVER_XYZ[2]) < 0.1)
drifting = (yaw.max() - yaw.min()) > 0.01
print(f"\nSTABLE at steady state (roll/pitch < ~3 deg, holds altitude): {stable}")
print(f"YAW DRIFTED in response to vision during the run (range > 0.01 rad): {drifting}")
print(f"\n=> REAL FLY CONNECTOME (T4/T5 -> HS/H1/H2) KEEPS DRONE AIRBORNE: {stable}")
