"""
Motor decoder: spike counts -> normalized [0,1] actuator value.

Same formula as the original boat.horse/fly project (from the write-up):

    value = 1 - exp(-(rate - baseline) / 40Hz)

then gamma + master scaling. `rate` is spikes per 20ms window per neuron
(or per pool-mean, for a group of neurons); `baseline` is that same neuron's
(or pool's) resting rate, subtracted so we show *change* from rest rather
than absolute activity -- this is what the original project's writeup
described doing to get meaningful behavioral signal instead of noise.
"""
from __future__ import annotations

import numpy as np


def spike_counts_to_rate(spike_counts: np.ndarray, window_seconds: float) -> np.ndarray:
    """Spikes accumulated over a window -> Hz."""
    return spike_counts / window_seconds


def decode_pool(
    rate_hz: np.ndarray,
    pool_indices: np.ndarray,
    baseline_hz: float,
    tau_hz: float = 40.0,
    gamma: float = 2.2,
    master: float = 0.25,
) -> float:
    """Mean rate over one neuron pool -> single normalized value in [0, master]."""
    if len(pool_indices) == 0:
        return 0.0
    pool_rate = float(np.mean(rate_hz[pool_indices]))
    value = 1.0 - np.exp(-max(pool_rate - baseline_hz, 0.0) / tau_hz)
    return master * (value ** gamma)


def decode_lr_pools(
    rate_hz: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    baseline_hz: float,
    **kwargs,
) -> tuple[float, float]:
    """Convenience wrapper for the Phase-A drone mapping: returns (left, right) values."""
    left = decode_pool(rate_hz, left_indices, baseline_hz, **kwargs)
    right = decode_pool(rate_hz, right_indices, baseline_hz, **kwargs)
    return left, right


def lr_to_thrust_yaw(left: float, right: float) -> tuple[float, float]:
    """Phase A mapping from the project plan: sum -> collective thrust offset,
    difference -> yaw. Both returned in roughly [-1, 1] (caller applies its
    own scale/clip before adding to a hover RPM baseline)."""
    thrust = left + right
    yaw = left - right
    return thrust, yaw
