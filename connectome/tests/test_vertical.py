"""
Regression test for the vertical system (encode_to_drive_vertical -> T4/T5 ->
VS -> descending neurons), the roll/pitch pathway scripts/m4_vertical.py
demonstrates.

Simulator-free, like the other pathway tests: synthetic warps of one texture
isolate roll, pitch and yaw far more cleanly than flying a drone does.

Two things here are easy to break silently and both have bitten already:

  - The type name. MaleCNS collapses all eight vertical-system subtypes into a
    single type called exactly "VS"; a search for `VS\\d+` finds nothing and
    led to M4 being recorded as blocked for lack of data. If that name ever
    changes, the network test below fails loudly rather than the pathway
    quietly disappearing.
  - The order of pooling and rectification. Roll is an opponent channel --
    left hemifield up against right hemifield down -- so a clip() in front of
    a mean() destroys it, exactly as it destroyed the looming pathway. See
    connectome/tests/test_looming.py.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from connectome.local_maleCNS import (local_fetch_vertical_subnetwork,
                                      DATA_DIR_DEFAULT, WEIGHTS_FILE)
from connectome.lif_network import LIFNetwork, scale_incoming
from connectome.medulla_encoder import encode_to_drive_vertical
from connectome import motor_decoder as md

DATA_AVAILABLE = os.path.isfile(os.path.join(DATA_DIR_DEFAULT, WEIGHTS_FILE))
IDX = {"vertical_L": np.arange(0, 9), "vertical_R": np.arange(9, 18)}
WEIGHT_SCALE, DN_SCALE = 0.0015, 12.0


def _texture(size=(48, 64), seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(cv2.GaussianBlur(rng.normal(128, 60, size=size), (0, 0), 1.6), 0, 255)


def _warp(field, angle_deg=0.0, dx=0.0, dy=0.0):
    h, w = field.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    img = cv2.warpAffine(field, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return np.stack([img] * 3, axis=-1)


def channels(gain=1.0, **warp_kwargs):
    """Returns (roll, pitch) = (left - right, left + right) of the drive."""
    field = _texture()
    d = encode_to_drive_vertical(_warp(field), _warp(field, **warp_kwargs), 18, IDX, gain=gain)
    left, right = float(d[IDX["vertical_L"]].mean()), float(d[IDX["vertical_R"]].mean())
    return left - right, left + right


def test_roll_reverses_with_direction():
    """Rolling one way must excite one hemifield, the other way the other."""
    roll_right, _ = channels(angle_deg=+6)
    roll_left, _ = channels(angle_deg=-6)
    assert roll_right * roll_left < 0, (
        f"roll channel must reverse (right {roll_right:+.5f}, left {roll_left:+.5f}) -- "
        "is the encoder rectifying per hemifield before pooling?"
    )


def test_pitch_lands_in_the_sum_channel():
    """Pitch turns both hemifields the same way, so it belongs to the sum."""
    roll, pitch = channels(dy=+3)
    assert pitch > 0, "downward field motion must drive both populations"
    assert abs(roll) < 0.4 * pitch, (
        f"pitch leaked into the roll channel (roll {roll:+.5f}, pitch {pitch:.5f})"
    )


def test_yaw_does_not_masquerade_as_roll():
    """The specificity control: horizontal motion is the other pathway's job."""
    roll_yaw, _ = channels(dx=+3)
    roll_real, _ = channels(angle_deg=+6)
    assert abs(roll_yaw) < 0.5 * abs(roll_real), (
        f"horizontal motion produced {roll_yaw:+.5f} in the roll channel against "
        f"{roll_real:+.5f} for an actual roll"
    )


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_vs_cells_exist_and_are_wired():
    """Guards the type name and the pathway's shape.

    "VS" is a single collapsed type in MaleCNS covering VS1-VS8, which is why a
    `VS\\d+` search once concluded the vertical system was missing entirely.
    """
    net = local_fetch_vertical_subnetwork()
    cti = net["cell_type_indices"]
    assert len(cti["vertical_L"]) > 0 and len(cti["vertical_R"]) > 0, "no VS cells found"
    assert len(cti["motor_left"]) > 0 and len(cti["motor_right"]) > 0, "no descending targets"

    W = net["weights"].tocsr()
    visual = np.concatenate([cti["visual_L"], cti["visual_R"]])
    vertical = np.concatenate([cti["vertical_L"], cti["vertical_R"]])
    incoming = W[vertical, :][:, visual].sum()
    assert incoming > 0, "T4/T5 do not reach the VS cells -- the pathway is not connected"


@pytest.mark.skipif(not DATA_AVAILABLE,
                    reason="local MaleCNS bulk data not downloaded (see docs/connectome-data-access.md)")
def test_roll_reaches_the_descending_neurons():
    """End to end: the roll signal must survive the network, with its sign."""
    net = local_fetch_vertical_subnetwork()
    cti = net["cell_type_indices"]
    W = net["weights"].multiply(WEIGHT_SCALE).tocsr()
    W = scale_incoming(W, np.concatenate([cti["motor_left"], cti["motor_right"]]), DN_SCALE)
    field = _texture()

    def run(angle_deg):
        lif = LIFNetwork(n_neurons=net["n_neurons"], weights=W, dt=0.1e-3)
        peak = 0.0
        for _ in range(6):
            drive = encode_to_drive_vertical(_warp(field), _warp(field, angle_deg=angle_deg),
                                             net["n_neurons"], cti, gain=600.0)
            for _ in range(333):
                lif.step(external_input=drive)
            rate = md.spike_counts_to_rate(lif.reset_spike_window(), 333 * 0.1e-3)
            left, right = md.decode_lr_pools(rate, cti["motor_left"], cti["motor_right"],
                                             baseline_hz=0.0)
            roll = md.lr_to_thrust_yaw(left, right)[1]
            peak = roll if abs(roll) > abs(peak) else peak
        return peak

    right, left = run(+6), run(-6)
    assert right * left < 0, (
        f"roll must still reverse at the descending neurons (right {right:+.5f}, left {left:+.5f})"
    )
