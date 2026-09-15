"""
Regression test for the direction-selective optomotor pathway
(medulla_encoder.encode_to_drive_progressive -> real MaleCNS T4/T5 -> HS/H1/H2
-> motor_decoder), the result scripts/m3_optomotor.py demonstrates in the
simulator.

Deliberately simulator-free: the stimulus is a synthetic full-field bar pattern
drifting left or right, which is what a rotating drum looks like to a
forward-facing camera anyway. That keeps this in the normal `pytest
connectome/tests/` run (PYTHONPATH=., no PYTHONPATH=simulator, no pybullet) and
takes seconds instead of the several minutes a full simulator run needs.

What this guards against, concretely: the pathway silently losing direction
selectivity again. It was absent for the entire life of the project until
2026-09-15 because encode_to_drive_hemifield() feeds |flow| and discards the
sign of motion -- and the failure mode does not look like a failure, it looks
like a clean null result. The gain is part of what is being pinned here: too
high and both motor pools saturate at the decoder's 0.25 ceiling, which also
produces a null result (see test_real_connectome.py on WEIGHT_SCALE, and
scripts/m3_gain_sweep.py).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from connectome.local_maleCNS import (local_fetch_lr_subnetwork, local_fetch_dn_subnetwork,
                                      DATA_DIR_DEFAULT, WEIGHTS_FILE)
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import encode_to_drive_progressive
from connectome import motor_decoder as md

GAIN = 1.0        # calibrated in scripts/m3_gain_sweep.py against the drum stimulus
WEIGHT_SCALE = 0.0015

DATA_AVAILABLE = os.path.isfile(os.path.join(DATA_DIR_DEFAULT, WEIGHTS_FILE))


def make_drifting_bars(n_frames=12, size=(48, 64), drift_px=2, seed=0):
    """Full-field vertical bars drifting horizontally; +drift_px moves the
    pattern rightward, -drift_px leftward.

    Built by sliding a window over one wide random field rather than by
    recomputing a regular grating per frame. A perfectly periodic square-wave
    grating with identical rows returns exactly zero Farneback flow (verified):
    the gradient is zero inside every bar, the coarse pyramid levels blur the
    edges away, and a periodic pattern is ambiguous to a correlation-based
    estimator anyway. Randomized bar widths plus faint 2D structure give the
    estimator something locally unique to track -- which is exactly why the
    real drum in world.room works, and why it is randomized there too.

    The noise field slides with the bars. Static per-pixel noise would anchor
    the flow estimate to zero, which is the same trap in a different costume.
    """
    h, w = size
    rng = np.random.default_rng(seed)
    # room for the window to travel in either direction from its start offset
    pad = 2 * abs(drift_px) * n_frames + w + 16

    strip = np.empty(pad, dtype=np.float64)
    x = 0
    light = True
    while x < pad:
        width = int(rng.integers(5, 13))
        strip[x:x + width] = (240 if light else 90) + rng.integers(-15, 16)
        light = not light
        x += width

    field = np.repeat(strip[None, :], h, axis=0)
    field += rng.normal(0, 6, size=(h, pad))          # slides with the pattern
    field += np.linspace(-12, 12, h)[:, None]          # gentle vertical shading
    field = np.clip(field, 0, 255)

    start = abs(drift_px) * n_frames + 8
    frames = []
    for i in range(n_frames):
        off = start - i * drift_px
        img = field[:, off:off + w]
        assert img.shape == (h, w), f"window ran off the field: {img.shape}"
        frames.append(np.stack([img, img, img], axis=-1))
    return np.stack(frames)


DN_SCALE = 12.0   # calibrated in scripts/m3_gain_sweep.py -- see test_dn_layer below

_NET_CACHE = {}


def _get_net(kind="lr"):
    if kind not in _NET_CACHE:
        _NET_CACHE[kind] = (local_fetch_lr_subnetwork() if kind == "lr"
                            else local_fetch_dn_subnetwork())
    return _NET_CACHE[kind]


def run_pipeline(frames, gain=GAIN, dt=0.1e-3, decode_window_s=20e-3, steps_per_frame_gap=500,
                 kind="lr"):
    net = _get_net(kind)
    idx = net["cell_type_indices"]
    W = net["weights"].multiply(WEIGHT_SCALE).tocsr()
    if kind == "dn":
        W = scale_incoming(W, np.concatenate([idx["motor_left"], idx["motor_right"]]), DN_SCALE)
    lif = LIFNetwork(n_neurons=net["n_neurons"], weights=W, dt=dt)
    decode_every = int(round(decode_window_s / dt))

    steps_since_decode = 0
    results = []
    for i in range(1, len(frames)):
        drive = encode_to_drive_progressive(frames[i - 1], frames[i], net["n_neurons"], idx,
                                            gain=gain)
        for _ in range(steps_per_frame_gap):
            lif.step(external_input=drive)
            steps_since_decode += 1
            if steps_since_decode >= decode_every:
                rate_hz = md.spike_counts_to_rate(lif.reset_spike_window(), decode_window_s)
                left, right = md.decode_lr_pools(rate_hz, idx["motor_left"], idx["motor_right"],
                                                 baseline_hz=0.0)
                results.append(md.lr_to_thrust_yaw(left, right))
                steps_since_decode = 0
    return np.array(results)  # columns: thrust, yaw


def test_progressive_encoder_separates_motion_direction():
    """Encoder-only, no connectome and no data download needed.

    Front-to-back motion on one side must drive that side's visual population
    and leave the other side's silent. This is the property
    encode_to_drive_hemifield() lacks, and the reason the optomotor experiment
    returned a null result before it existed.
    """
    idx = {"visual_L": np.arange(0, 10), "visual_R": np.arange(10, 20)}
    left_drift = make_drifting_bars(n_frames=3, drift_px=-2)
    right_drift = make_drifting_bars(n_frames=3, drift_px=+2)

    d_left = encode_to_drive_progressive(left_drift[0], left_drift[1], 20, idx, gain=GAIN)
    d_right = encode_to_drive_progressive(right_drift[0], right_drift[1], 20, idx, gain=GAIN)

    # Not an exact zero on the opposing side: rectification happens per pixel, so
    # a handful of noise pixels with the opposite sign always survive the mean.
    # What matters is that they are negligible against the driven side.
    for drive, driven, quiet, label in [(d_left, "visual_L", "visual_R", "leftward"),
                                        (d_right, "visual_R", "visual_L", "rightward")]:
        on = drive[idx[driven]].mean()
        off = drive[idx[quiet]].mean()
        assert on > 0.5, f"{label} motion should drive {driven} (got {on:.4f})"
        assert off < 0.01 * on, (
            f"{label} motion should leave {quiet} effectively silent "
            f"(got {off:.6f} against {on:.4f} on {driven})"
        )


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_optomotor_yaw_reverses_with_stimulus_direction():
    """The headline result: decoded yaw must flip sign with stimulus direction."""
    yaw_left = run_pipeline(make_drifting_bars(drift_px=-2))[:, 1]
    yaw_right = run_pipeline(make_drifting_bars(drift_px=+2))[:, 1]

    assert yaw_left.mean() * yaw_right.mean() < 0, (
        f"decoded yaw must reverse with motion direction "
        f"(leftward {yaw_left.mean():+.5f}, rightward {yaw_right.mean():+.5f})"
    )
    assert min(abs(yaw_left.mean()), abs(yaw_right.mean())) > 0.01, (
        f"response too weak to be meaningful (leftward {yaw_left.mean():+.5f}, "
        f"rightward {yaw_right.mean():+.5f})"
    )


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_motor_pools_are_not_saturated():
    """Guard the gain from drifting back up.

    Both pools pinned at the decoder's 0.25 ceiling gives thrust 0.5 and
    yaw = left - right = 0 -- a null result that looks like a clean
    measurement. This is exactly what GAIN=8.0 produced.
    """
    thrust = run_pipeline(make_drifting_bars(drift_px=+2))[:, 0]
    assert thrust.max() < 0.45, (
        f"motor pools are saturating (max thrust {thrust.max():.4f} of a 0.5 ceiling) -- "
        "re-run scripts/m3_gain_sweep.py"
    )


if __name__ == "__main__":
    test_progressive_encoder_separates_motion_direction()
    if not DATA_AVAILABLE:
        print("SKIPPED connectome tests: local MaleCNS bulk data not present")
    else:
        test_optomotor_yaw_reverses_with_stimulus_direction()
        test_motor_pools_are_not_saturated()
        print("OK: optomotor direction selectivity intact.")


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_dn_layer_is_direction_selective():
    """Same reversal, but decoded from the descending neurons rather than from
    the tangential cells that drive them.

    Pins DN_SCALE as much as the direction selectivity. The DN layer integrates
    ~12 tangential cells where those integrate ~6800 visual neurons, so with
    the single WEIGHT_SCALE the descending neurons never spike at all -- silent
    from encoder gain 1.0 through 16.0, measured. A scale that is too low is
    therefore indistinguishable from a broken pathway, and one that is too high
    saturates: both look like a clean null result.
    """
    yaw_left = run_pipeline(make_drifting_bars(drift_px=-2), kind="dn")[:, 1]
    yaw_right = run_pipeline(make_drifting_bars(drift_px=+2), kind="dn")[:, 1]

    assert yaw_left.mean() * yaw_right.mean() < 0, (
        f"decoded yaw at the DN layer must reverse with motion direction "
        f"(leftward {yaw_left.mean():+.5f}, rightward {yaw_right.mean():+.5f})"
    )
    assert min(abs(yaw_left.mean()), abs(yaw_right.mean())) > 0.002, (
        f"DN response too weak -- check DN_SCALE (leftward {yaw_left.mean():+.5f}, "
        f"rightward {yaw_right.mean():+.5f})"
    )
