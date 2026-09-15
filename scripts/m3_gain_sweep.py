"""
M3 calibration: find an encoder gain at which the optomotor stimulus lands in
the decoder's responsive range instead of pinning it.

scripts/m3_optomotor.py at the inherited GAIN=8.0 returns mean thrust 0.49898
during BOTH drum directions -- that is 0.25+0.25, the exact ceiling of
motor_decoder.decode_pool (master=0.25 per pool). Both motor pools sit at max
firing, so yaw = left - right is ~0 by construction and the experiment cannot
resolve direction. This is the same failure the WEIGHT_SCALE note in
connectome/tests/test_real_connectome.py describes, re-triggered from the other
side: that note calibrated the network's internal weights against the static
toy stimulus, and a rotating high-contrast drum delivers far more drive than
that stimulus ever did.

GAIN is the right knob here rather than WEIGHT_SCALE: the network's internal
weights are already calibrated so that its left/right structure is intact
(test_real_connectome.py passes, including the symmetric-balance test); what is
mis-scaled is how hard the encoder pushes the visual populations for this
stimulus.

Prints, per gain, the raw motor-pool rates and the decoded values for both drum
directions. What to look for: pool rates comparable to the decoder's 40 Hz
tau -- far above that and the exponential has flattened -- and a left/right
difference that changes sign with drum direction.
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

from connectome.local_maleCNS import (local_fetch_lr_subnetwork,
                                      local_fetch_dn_subnetwork)
from connectome.lif_network import LIFNetwork
from connectome.medulla_encoder import (encode_to_drive_hemifield,
                                        encode_to_drive_progressive)
from connectome import motor_decoder as md
from world.room import build_drum, rotate_drum

ENCODERS = {"hemifield": encode_to_drive_hemifield,
            "progressive": encode_to_drive_progressive}

PHYSICS_HZ, CAMERA_HZ = 240, 30
NET_DT = 0.1e-3
NET_STEPS = int(round((1.0 / CAMERA_HZ) / NET_DT))
WEIGHT_SCALE = 0.0015
OMEGA = 1.2
SECONDS_PER_DIRECTION = 1.2
HOVER_XYZ = np.array([0.0, 0.0, 1.0])
# Override for a finer pass, e.g. SWEEP_GAINS=0.5,0.6,0.7,0.8
GAINS = [float(g) for g in os.environ.get("SWEEP_GAINS", "0.05,0.1,0.25,0.5,1.0,2.0,8.0").split(",")]
ENCODER_NAME = os.environ.get("M3_ENCODER", "progressive")
encoder = ENCODERS[ENCODER_NAME]

# M3_NETWORK=lr decodes from HS/H1/H2 (tangential cells, interneurons);
# dn adds the descending-neuron layer they drive and decodes there instead.
NETWORK = os.environ.get("M3_NETWORK", "dn")
_FETCH = {"lr": local_fetch_lr_subnetwork, "dn": local_fetch_dn_subnetwork}[NETWORK]
print(f"Loading real MaleCNS subnetwork (decode layer: {NETWORK})...")
t0 = time.time()
net_spec = _FETCH()
print(f"  {net_spec['n_neurons']} neurons, {net_spec['weights'].nnz} synapses ({time.time()-t0:.1f}s)")
cti = net_spec["cell_type_indices"]
W = net_spec["weights"].multiply(WEIGHT_SCALE).tocsr()
# The DN layer integrates ~12 HS cells where an HS cell integrates ~6800 visual
# neurons, so it needs its own scale -- see lif_network.scale_incoming().
DN_SCALE = float(os.environ.get("DN_SCALE", "1.0"))
if NETWORK == "dn" and DN_SCALE != 1.0:
    from connectome.lif_network import scale_incoming
    import numpy as _np
    W = scale_incoming(W, _np.concatenate([cti["motor_left"], cti["motor_right"]]), DN_SCALE)
    print(f"DN incoming weights scaled by {DN_SCALE}")

env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
env.reset()
drum = build_drum(env.CLIENT, plane_id=env.PLANE_ID)
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

# Hold the drone still for the whole sweep: the only thing that must differ
# between measurements is the gain and the drum direction.
for _ in range(PHYSICS_HZ * 2):
    s = env._getDroneStateVector(0)
    rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3], cur_quat=s[3:7],
                                    cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                    target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
    env.step({0: rpm})


def measure(gain, omega):
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)
    angle = 0.0
    prev = None
    rates_l, rates_r, decoded = [], [], []
    for _ in range(int(SECONDS_PER_DIRECTION * CAMERA_HZ)):
        angle += omega / CAMERA_HZ
        rotate_drum(env.CLIENT, drum, angle)
        curr = env._getDroneImages(0, segmentation=False)[0]
        if prev is not None:
            drive = encoder(prev, curr, net.n, cti, gain=gain)
            for _ in range(NET_STEPS):
                net.step(external_input=drive)
            rate_hz = md.spike_counts_to_rate(net.reset_spike_window(), NET_STEPS * NET_DT)
            rates_l.append(float(np.mean(rate_hz[cti["motor_left"]])))
            rates_r.append(float(np.mean(rate_hz[cti["motor_right"]])))
            left, right = md.decode_lr_pools(rate_hz, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)
            decoded.append(md.lr_to_thrust_yaw(left, right))
        # keep the drone exactly where it is
        for _ in range(PHYSICS_HZ // CAMERA_HZ):
            s = env._getDroneStateVector(0)
            rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ, cur_pos=s[0:3],
                                            cur_quat=s[3:7], cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                            target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
            env.step({0: rpm})
        prev = curr
    d = np.array(decoded)
    return np.mean(rates_l), np.mean(rates_r), d[:, 0].mean(), d[:, 1].mean()


print(f"\nencoder: {ENCODER_NAME}")
print(f"\n{'gain':>6} {'dir':>4} {'rate_L Hz':>10} {'rate_R Hz':>10} {'thrust':>8} {'yaw':>10}")
results = {}
for gain in GAINS:
    row = {}
    for tag, omega in [("CCW", +OMEGA), ("CW", -OMEGA)]:
        rl, rr, thrust, yaw = measure(gain, omega)
        row[tag] = yaw
        sat = " <- ceiling" if thrust > 0.497 else ""
        print(f"{gain:6.2f} {tag:>4} {rl:10.1f} {rr:10.1f} {thrust:8.4f} {yaw:+10.5f}{sat}")
    results[gain] = row

env.close()

print(f"\n{'gain':>6} {'yaw CCW':>10} {'yaw CW':>10}  reverses?")
for gain, row in results.items():
    print(f"{gain:6.2f} {row['CCW']:+10.5f} {row['CW']:+10.5f}  {row['CCW']*row['CW'] < 0}")
