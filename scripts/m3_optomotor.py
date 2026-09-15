"""
The optomotor experiment: a striped drum rotates around the drone, and we ask
whether the fly connectome turns the drone with it.

This is the falsifiable version of "the connectome controls the drone".
scripts/m2_room.py only showed the loop is *alive* in a textured world (0% ->
43% non-zero output); it did not show the output means anything. Here the
stimulus is controlled: the drum turns one way, then the other, with static
baselines between, and the prediction is specific -- the decoded yaw signal
must reverse sign when the drum reverses, and stay near zero when it is still.

Two conditions, mirroring how the experiment is actually run on flies:

  tethered  The drone's pose is frozen (PID holds position, altitude and
            yaw=0). Nothing the connectome outputs is applied. The only motion
            in the image is the drum, so this is a clean open-loop readout of
            the pathway's direction selectivity -- the equivalent of a fly
            glued to a pin inside the drum.
  free      The closed loop of scripts/m2_room.py: yaw setpoint integrates the
            decoded yaw, altitude setpoint the decoded thrust. The drone can
            now turn, which changes what it sees -- so a following response
            here is a genuine behavioural result, not just a readout.

Honest caveat on what a positive result does and does not show: the SIGN of the
response is not predicted by the connectome. Which anatomical side drives which
motor pool is our mapping choice in local_fetch_lr_subnetwork()/
encode_to_drive_hemifield(), not something derived from the data. What is a
real test is the *reversal*: whatever sign the CCW phase produces, the CW phase
must produce the opposite, and the static phases neither. That cannot be
arranged by a mapping convention.

Three passes, so the decode layer is compared rather than swapped silently:

  tethered / lr    decode from HS/H1/H2 -- lobula plate tangential cells, i.e.
                   interneurons. This is the established result.
  tethered / dn    decode from the descending neurons those tangential cells
                   drive (DNb03, DNg41, DNp15, DNp17, DNa02). Descending
                   neurons are the brain's actual output to the nerve cord, so
                   this removes one layer of stand-in from the claim.
  free / dn        closed loop on the descending-neuron decode, recorded.

The DN layer needs its own synaptic scale (DN_SCALE) because it integrates ~12
tangential cells where those integrate ~6800 visual neurons -- see
lif_network.scale_incoming(). That scale is a modelling choice, not something
the connectome dictates; the data gives connectivity, not biophysics.

Outputs (files/optomotor/, gitignored):
  response.png  time series of drum velocity, decoded yaw, drone yaw
  flight.mp4    the free-condition run, external view + the drone's own camera
"""
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulator"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p

from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from connectome.local_maleCNS import (local_fetch_lr_subnetwork,
                                      local_fetch_dn_subnetwork)
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import (encode_to_drive_hemifield,
                                        encode_to_drive_progressive,
                                        encode_to_drive_reichardt)
from connectome import motor_decoder as md
from world.room import build_drum, rotate_drum

ENCODERS = {"hemifield": encode_to_drive_hemifield,
            "progressive": encode_to_drive_progressive,
            "reichardt": encode_to_drive_reichardt}

PHYSICS_HZ = 240
CAMERA_HZ = 30
STEPS_PER_CAMERA_FRAME = PHYSICS_HZ // CAMERA_HZ
NET_DT = 0.1e-3
NET_STEPS_PER_CAMERA_FRAME = int(round((1.0 / CAMERA_HZ) / NET_DT))
# Calibrated per encoder by scripts/m3_gain_sweep.py. The 8.0 inherited from
# the toy network pins BOTH motor pools at the decoder's 0.25 ceiling for any
# drum direction (measured: thrust 0.49898 = 0.25+0.25), which annihilates
# yaw = left - right.
#   hemifield   0.7 -- best of a narrow window (0.4 silent, 0.8 already loses
#                      whatever reversal there was)
#   progressive 1.0 -- pool rates 32/38 Hz, just under the decoder's 40 Hz tau,
#                      with the opposite pool completely silent
GAIN_BY_ENCODER = {"hemifield": 0.7, "progressive": 1.0}
WEIGHT_SCALE = 0.0015   # see connectome/tests/test_real_connectome.py
YAW_GAIN = 6.0
Z_GAIN = 0.15
HOVER_XYZ = np.array([0.0, 0.0, 1.0])

# Drum speed: 24 bars means one bar every 0.26 rad; the 60 deg camera spans
# ~1.05 rad of drum, so a bar is ~16 px wide in the 64 px frame. 1.2 rad/s is
# ~2.4 px/frame at 30 Hz -- enough displacement for Farneback to resolve, and
# far below the ~16 px/frame that would alias one bar onto the next.
OMEGA = 1.2
N_REPEATS = 3

def _protocol():
    """Repeated trials with counterbalanced direction order.

    A single CCW block followed by a single CW block is not enough to claim a
    reversal: measured that way at GAIN=0.7 the sweep (fresh network per
    direction) reversed cleanly while the continuous run did not, which means
    the outcome depended on the state the network entered the block with, not
    only on the stimulus. Repeating each direction and swapping which comes
    first cancels order and carry-over effects, and gives a spread to judge
    whether a difference between directions is bigger than trial-to-trial
    noise.
    """
    phases = []
    for rep in range(N_REPEATS):
        order = [+OMEGA, -OMEGA] if rep % 2 == 0 else [-OMEGA, +OMEGA]
        for w in order:
            phases.append(("static", 1.0, 0.0))
            phases.append(("drum CCW" if w > 0 else "drum CW", 1.5, w))
    return phases

PHASES = _protocol()

CAPTURE_EVERY = 2          # every 2nd camera frame -> 15 fps, i.e. real time
VIDEO_FPS = CAMERA_HZ / CAPTURE_EVERY

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "optomotor")
os.makedirs(OUT_DIR, exist_ok=True)

# lr decodes from HS/H1/H2 (tangential cells, interneurons); dn adds the
# descending-neuron layer they drive and decodes there instead.
DN_SCALE = 12.0   # calibrated by scripts/m3_gain_sweep.py -- see module docstring
_FETCH = {"lr": local_fetch_lr_subnetwork, "dn": local_fetch_dn_subnetwork}
_NETS = {}


def get_net(name):
    """Load and prepare a network once; both passes reuse it."""
    if name not in _NETS:
        print(f"Loading real MaleCNS subnetwork (decode layer: {name})...")
        t0 = time.time()
        spec = _FETCH[name]()
        W = spec["weights"].multiply(WEIGHT_SCALE).tocsr()
        if name == "dn":
            pools = np.concatenate([spec["cell_type_indices"]["motor_left"],
                                    spec["cell_type_indices"]["motor_right"]])
            W = scale_incoming(W, pools, DN_SCALE)
        spec["model_weights"] = W
        print(f"  {spec['n_neurons']} neurons, {spec['weights'].nnz} synapses "
              f"({time.time()-t0:.1f}s)")
        _NETS[name] = spec
    return _NETS[name]


def run(tethered: bool, encoder_name: str, network: str, record_gif: bool = False):
    encoder = ENCODERS[encoder_name]
    gain = GAIN_BY_ENCODER[encoder_name]
    net_spec = get_net(network)
    cti = net_spec["cell_type_indices"]
    W = net_spec["model_weights"]
    dn_types = sorted({k[3:-2] for k in cti if k.startswith("dn_")})
    dn_rates = {dn_t: {"CCW": [], "CW": []} for dn_t in dn_types}
    env = VisionAviary(num_drones=1, gui=False, record=False, freq=PHYSICS_HZ)
    env.reset()
    drum = build_drum(env.CLIENT, plane_id=env.PLANE_ID)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    net = LIFNetwork(n_neurons=net_spec["n_neurons"], weights=W, dt=NET_DT)

    target_pos = HOVER_XYZ.copy()
    yaw_setpoint = 0.0
    drum_angle = 0.0
    prev_frame = None
    log, gif_frames = [], []

    # Settle at hover first, so the measured phases start from a steady drone
    # rather than mixing the takeoff transient into the static baseline.
    for _ in range(PHYSICS_HZ * 2):
        s = env._getDroneStateVector(0)
        rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ,
                                        cur_pos=s[0:3], cur_quat=s[3:7],
                                        cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                        target_pos=HOVER_XYZ, target_rpy=np.zeros(3))
        env.step({0: rpm})

    t = 0.0
    for phase_name, duration, omega in PHASES:
        for _ in range(int(duration * CAMERA_HZ)):
            drum_angle += omega / CAMERA_HZ
            rotate_drum(env.CLIENT, drum, drum_angle)
            curr_frame = env._getDroneImages(0, segmentation=False)[0]

            if prev_frame is not None:
                drive = encoder(prev_frame, curr_frame, net.n, cti, gain=gain)
                for _ in range(NET_STEPS_PER_CAMERA_FRAME):
                    net.step(external_input=drive)
                counts = net.reset_spike_window()
                rate_hz = md.spike_counts_to_rate(counts, NET_STEPS_PER_CAMERA_FRAME * NET_DT)
                left, right = md.decode_lr_pools(rate_hz, cti["motor_left"], cti["motor_right"],
                                                 baseline_hz=0.0)
                thrust_out, yaw_out = md.lr_to_thrust_yaw(left, right)
                if omega != 0.0:
                    tag = "CCW" if omega > 0 else "CW"
                    for dn_t in dn_types:   # not `t` -- that is the clock
                        dn_rates[dn_t][tag].append(
                            float(rate_hz[cti[f"dn_{dn_t}_L"]].mean() -
                                  rate_hz[cti[f"dn_{dn_t}_R"]].mean()))
            else:
                thrust_out, yaw_out = 0.0, 0.0

            if not tethered:
                yaw_setpoint += YAW_GAIN * yaw_out * (1.0 / CAMERA_HZ)
                target_pos = HOVER_XYZ + np.array([0.0, 0.0, Z_GAIN * thrust_out])

            for _ in range(STEPS_PER_CAMERA_FRAME):
                s = env._getDroneStateVector(0)
                rpm, _, _ = ctrl.computeControl(control_timestep=1/PHYSICS_HZ,
                                                cur_pos=s[0:3], cur_quat=s[3:7],
                                                cur_vel=s[10:13], cur_ang_vel=s[13:16],
                                                target_pos=target_pos,
                                                target_rpy=np.array([0.0, 0.0, yaw_setpoint]))
                env.step({0: rpm})

            s = env._getDroneStateVector(0)
            log.append((t, omega, drum_angle, thrust_out, yaw_out, s[9], s[2]))
            if record_gif and len(log) % CAPTURE_EVERY == 0:
                gif_frames.append(_composite(env, curr_frame, phase_name, omega,
                                             t, s[9], drum_angle))
            prev_frame = curr_frame
            t += 1.0 / CAMERA_HZ

    env.close()
    return np.array(log), gif_frames, dn_rates


def _font(size):
    """DejaVuSans ships with matplotlib, so no system font lookup is needed.
    PIL's built-in default font is a fixed ~11px bitmap, too small to read on a
    480px frame."""
    from PIL import ImageFont
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                        "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _project(pos, view, proj, width, height):
    """World point -> pixel coordinates, using the same matrices as the render.

    getCameraImage's matrices are column-major 16-element lists, hence order='F'.
    """
    V = np.array(view, dtype=np.float64).reshape(4, 4, order="F")
    P = np.array(proj, dtype=np.float64).reshape(4, 4, order="F")
    clip = P @ (V @ np.array([pos[0], pos[1], pos[2], 1.0]))
    if abs(clip[3]) < 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * width, (1.0 - (ndc[1] * 0.5 + 0.5)) * height)


def _composite(env, pov, phase_name, omega, t, drone_yaw, drum_angle):
    """External view, annotated, with the drone's own 64x48 camera inset.

    The annotation is not decoration. At this camera distance the Crazyflie is
    about six pixels across and sits behind the drum bars, so neither its
    position nor its heading is legible from the raw render -- which is the one
    thing a recording of this experiment has to show. Hence: a ring marking
    where the drone is, and a compass giving its heading against the drum's.
    """
    from PIL import ImageDraw

    width, height = 480, 360
    # Steep enough to see into the drum rather than at its outside, far enough
    # that the whole drum stays in frame, then zoomed with a narrower FOV.
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 1.0], distance=13.0,
                                               yaw=40, pitch=-55, roll=0, upAxisIndex=2,
                                               physicsClientId=env.CLIENT)
    proj = p.computeProjectionMatrixFOV(fov=45.0, aspect=width / height, nearVal=0.1,
                                        farVal=100.0, physicsClientId=env.CLIENT)
    _, _, rgb, _, _ = p.getCameraImage(width=width, height=height, viewMatrix=view,
                                       projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
                                       physicsClientId=env.CLIENT)
    img = Image.fromarray(np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")

    direction = "CCW" if omega > 0 else ("CW" if omega < 0 else "still")
    colour = (46, 204, 113) if omega > 0 else ((230, 126, 34) if omega < 0 else (180, 180, 180))

    # ring around the drone, so it can be found at all
    drone_pos = env._getDroneStateVector(0)[0:3]
    xy = _project(drone_pos, view, proj, width, height)
    if xy is not None:
        x, y = xy
        draw.ellipse([x - 16, y - 16, x + 16, y + 16], outline=(41, 128, 185), width=3)

    # header
    draw.rectangle([0, 0, width, 34], fill=(0, 0, 0, 165))
    draw.text((8, 8), f"t = {t:4.1f} s", font=_font(15), fill=(235, 235, 235))
    draw.text((110, 8), f"drum {direction}", font=_font(15), fill=colour)
    draw.text((235, 8), f"drone yaw {drone_yaw:+.3f} rad", font=_font(15), fill=(120, 190, 255))

    # compass: blue needle = drone heading, drum-coloured tick = drum phase
    cx, cy, r = width - 46, 76, 30
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 150), outline=(210, 210, 210))
    draw.line([cx, cy, cx + r * 0.85 * np.cos(-drone_yaw), cy + r * 0.85 * np.sin(-drone_yaw)],
              fill=(41, 128, 185), width=4)
    draw.line([cx + r * 0.72 * np.cos(-drum_angle), cy + r * 0.72 * np.sin(-drum_angle),
               cx + r * 0.98 * np.cos(-drum_angle), cy + r * 0.98 * np.sin(-drum_angle)],
              fill=colour, width=4)

    inset = Image.fromarray(pov[:, :, :3].astype(np.uint8)).resize((192, 144), Image.NEAREST)
    img.paste(inset, (width - 192 - 8, height - 144 - 8))
    draw.rectangle([width - 192 - 8, height - 144 - 8, width - 9, height - 9],
                   outline=(210, 210, 210), width=2)
    draw.text((width - 192 - 4, height - 144 - 22), "drone camera (64x48)", font=_font(12),
              fill=(235, 235, 235))
    return img


def _trial_means(log):
    """Mean decoded yaw per contiguous drum block, keyed by direction."""
    omega, yaw_o = log[:, 1], log[:, 4]
    blocks = {"drum CCW": [], "drum CW": [], "static": []}
    start = 0
    for i in range(1, len(omega) + 1):
        if i == len(omega) or not np.isclose(omega[i], omega[start]):
            w = omega[start]
            key = "static" if np.isclose(w, 0.0) else ("drum CCW" if w > 0 else "drum CW")
            blocks[key].append(float(yaw_o[start:i].mean()))
            start = i
    return blocks


def summarize(label, log):
    blocks = _trial_means(log)
    thrust_o, omega = log[:, 3], log[:, 1]
    print(f"\n--- {label} ---")
    print(f"{'condition':>10} {'n':>3} {'mean yaw_out':>14} {'sd':>10} {'per-trial means':>34}")
    for key in ("static", "drum CCW", "drum CW"):
        v = np.array(blocks[key])
        per = " ".join(f"{x:+.4f}" for x in v)
        print(f"{key:>10} {len(v):3d} {v.mean():+14.5f} {v.std():10.5f}   {per:>32}")
    for key, w in (("drum CCW", +OMEGA), ("drum CW", -OMEGA)):
        m = np.isclose(omega, w)
        print(f"{'':>10} mean thrust during {key}: {thrust_o[m].mean():.5f}"
              + ("  <- at decoder ceiling" if thrust_o[m].mean() > 0.497 else ""))
    return blocks


print("\n[1/3] tethered, decode from HS/H1/H2 (tangential cells)")
log_hs, _, _ = run(tethered=True, encoder_name="progressive", network="lr")
print("[2/3] tethered, decode from descending neurons")
log_dn, _, dn_rates = run(tethered=True, encoder_name="progressive", network="dn")
print("[3/3] free, decode from descending neurons (closed loop)")
log_free, gif, _ = run(tethered=False, encoder_name="progressive", network="dn",
                       record_gif=True)

b_hs = summarize("TETHERED / decode at HS/H1/H2", log_hs)
b_dn = summarize("TETHERED / decode at descending neurons", log_dn)
b_free = summarize("FREE / decode at descending neurons", log_free)


def reversal(label, blocks):
    ccw, cw = np.array(blocks["drum CCW"]), np.array(blocks["drum CW"])
    separated = ccw.min() > cw.max() or cw.min() > ccw.max()
    ok = (ccw.mean() * cw.mean() < 0) and separated
    print(f"\n{label} (n={len(ccw)} trials per direction)")
    print(f"  CCW {ccw.mean():+.5f} +- {ccw.std():.5f}   CW {cw.mean():+.5f} +- {cw.std():.5f}")
    print(f"  opposite signs on the means : {ccw.mean() * cw.mean() < 0}")
    print(f"  trials fully non-overlapping: {separated}")
    print(f"  => direction-selective optomotor response: {ok}")
    return ok


reversal("reversal test, decode at HS/H1/H2", b_hs)
reversal("reversal test, decode at descending neurons", b_dn)
reversal("reversal test, free / descending neurons", b_free)

print("\nper-DN-type direction selectivity (mean rate_L - rate_R, Hz):")
print(f"{'type':>10} {'CCW':>10} {'CW':>10}  reverses?")
for t, v in dn_rates.items():
    ccw, cw = np.mean(v["CCW"]), np.mean(v["CW"])
    print(f"{t:>10} {ccw:+10.2f} {cw:+10.2f}  {ccw * cw < 0}")

free_yaw = log_free[:, 5]
print(f"free condition: drone yaw {free_yaw.min():+.3f} .. {free_yaw.max():+.3f} rad "
      f"(net {free_yaw[-1]-free_yaw[0]:+.3f})")

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, (log, name) in zip(axes[:2], [(log_hs, "tethered\ndecode: HS/H1/H2"),
                                      (log_dn, "tethered\ndecode: DNs")]):
    t, omega, _, _, yaw_o, drone_yaw, _ = log.T
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(t, yaw_o, color="#c0392b", lw=1.4, label="decoded yaw (connectome)")
    ax.set_ylabel(f"{name}\nyaw_out")
    ax.legend(loc="upper left", fontsize=8)
axes[2].plot(log_dn[:, 0], log_dn[:, 5], label="drone yaw, tethered", color="#7f8c8d")
axes[2].plot(log_free[:, 0], log_free[:, 5], label="drone yaw, free (DN decode)",
             color="#2471a3", lw=1.8)
axes[2].set_ylabel("drone yaw [rad]")
axes[2].set_xlabel("time [s]")
axes[2].legend(loc="upper left", fontsize=8)

# shade the drum phases across all panels
edges, acc = [], 0.0
for name, dur, w in PHASES:
    edges.append((acc, acc + dur, w, name))
    acc += dur
for ax in axes:
    for lo, hi, w, name in edges:
        if w != 0:
            ax.axvspan(lo, hi, color=("#2ecc71" if w > 0 else "#e67e22"), alpha=0.12)
axes[0].set_title(f"Optomotor response, drum ±{OMEGA} rad/s "
                  f"(green = CCW, orange = CW, white = static)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "response.png"), dpi=130)
print(f"\nwrote {OUT_DIR}/response.png")

def write_video(frames, path, fps):
    """Write an MP4. H.264 (avc1) first, MPEG-4 (mp4v) as fallback.

    Both are available in the bundled opencv-python-headless build here, but
    which codecs a given OpenCV wheel carries is not guaranteed, so the
    fallback is real rather than defensive decoration. cv2 wants BGR uint8.
    """
    import cv2
    size = (frames[0].width, frames[0].height)
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames:
            writer.write(np.asarray(f.convert("RGB"))[:, :, ::-1])
        writer.release()
        if os.path.getsize(path) > 0:
            return fourcc
    raise RuntimeError("no usable MP4 codec in this OpenCV build")


if gif:
    video_path = os.path.join(OUT_DIR, "flight.mp4")
    codec = write_video(gif, video_path, VIDEO_FPS)
    print(f"wrote {video_path} ({len(gif)} frames, {VIDEO_FPS:.0f} fps, {codec}, "
          f"{os.path.getsize(video_path)/1e6:.1f} MB)")
