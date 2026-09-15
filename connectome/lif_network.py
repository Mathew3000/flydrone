"""
Generic sparse leaky-integrate-and-fire (LIF) network engine.

Deliberately backend-agnostic and correctness-first: uses scipy.sparse for
the weight matrix and plain numpy for the membrane update, so it runs
anywhere (no CUDA/Numba/GeNN setup needed) and is easy to unit-test with a
handful of neurons. This is NOT the 166,700-neuron/25.6M-synapse scale the
real MaleCNS connectome needs -- once we have real connectivity data and are
running on the GPU server, swap this for a GeNN (GPU) or Numba (CPU) backend
with the same constructor/step() interface. See docs/connectome-data-access.md
for how the real data gets in here.

Dynamics (matches the "leaky integrate-and-fire ... 0.1ms resolution" model
mentioned in the boat.horse/fly writeup), with an exponentially filtered
synaptic current (a plain "add the weight for one timestep" model turned out
too brief to let sparse, infrequent spikes ever accumulate to threshold --
see connectome/tests/test_toy_network.py's notes -- a synaptic time constant
is the standard fix and also more biologically realistic):

    I_syn[i](t)  = I_syn[i](t-dt) * (1 - dt/tau_syn) + sum_j W[i,j] * spike[j](t-1)
    v[i](t+dt)   = v[i](t) + (dt/tau_mem) * ( (v_rest - v[i](t)) + R * (external_input[i](t) + I_syn[i](t)) )
    spike when v[i] >= v_thresh -> record spike, v[i] = v_reset, refractory for `refractory_steps`
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LIFNetwork:
    def __init__(
        self,
        n_neurons: int,
        weights: sp.spmatrix,
        dt: float = 0.1e-3,        # 0.1 ms, matches the original project's resolution
        tau_mem: float = 20e-3,    # 20 ms membrane time constant (typical fly neuron order of magnitude)
        tau_syn: float = 5e-3,     # 5 ms synaptic current decay
        v_rest: float = 0.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        r_mem: float = 1.0,
        refractory_steps: int = 20,  # 2 ms refractory at dt=0.1ms
    ):
        assert weights.shape == (n_neurons, n_neurons), "weights must be (n_neurons, n_neurons)"
        self.n = n_neurons
        self.W = sp.csr_matrix(weights)
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.r_mem = r_mem
        self.refractory_steps = refractory_steps

        self.v = np.full(n_neurons, v_rest, dtype=np.float64)
        self.i_syn = np.zeros(n_neurons, dtype=np.float64)
        self.refractory_counter = np.zeros(n_neurons, dtype=np.int32)
        self.last_spikes = np.zeros(n_neurons, dtype=np.float64)  # 0/1, previous step's spikes
        self.spike_count_window = np.zeros(n_neurons, dtype=np.int64)  # for the motor decoder's 20ms windows

    def step(self, external_input: np.ndarray | None = None) -> np.ndarray:
        """Advance the network by one dt. Returns this step's spike vector (0/1 per neuron)."""
        ext = external_input if external_input is not None else 0.0

        self.i_syn = self.i_syn * (1.0 - self.dt / self.tau_syn) + self.W.dot(self.last_spikes)
        total_input = ext + self.i_syn

        not_refractory = self.refractory_counter <= 0
        dv = (self.dt / self.tau_mem) * ((self.v_rest - self.v) + self.r_mem * total_input)
        self.v = np.where(not_refractory, self.v + dv, self.v_reset)

        spikes = (self.v >= self.v_thresh) & not_refractory
        self.v[spikes] = self.v_reset
        self.refractory_counter[spikes] = self.refractory_steps
        self.refractory_counter = np.maximum(self.refractory_counter - 1, 0)

        spike_vec = spikes.astype(np.float64)
        self.last_spikes = spike_vec
        self.spike_count_window += spikes.astype(np.int64)
        return spike_vec

    def reset_spike_window(self) -> np.ndarray:
        """Return and zero the per-neuron spike counts accumulated since the last call."""
        counts = self.spike_count_window.copy()
        self.spike_count_window[:] = 0
        return counts


def scale_incoming(weights, target_indices, factor: float):
    """Multiply every synapse *onto* `target_indices` by `factor`.

    Why a per-layer scale is needed at all: connectome weights are raw synapse
    counts, and the LIF engine gives every neuron the same threshold and
    membrane resistance, so a layer's operating point is set by its fan-in. In
    the T4/T5 -> HS/H1/H2 -> DN network those differ by two orders of
    magnitude: an HS cell integrates ~6800 visual neurons, a descending neuron
    integrates 12 HS cells. The single WEIGHT_SCALE calibrated to keep HS out
    of saturation therefore leaves the DNs at a steady-state membrane voltage
    of roughly 0.1 against a threshold of 1.0 -- they never spike at any
    encoder gain (measured: silent from gain 1.0 through 16.0).

    Scaling the DN layer's inputs separately is the same statement as saying
    descending neurons are more excitable, or have a lower threshold, than
    tangential cells. It is a modelling choice the connectome does not make for
    us -- the data gives connectivity, not biophysics -- and it should be
    reported as such rather than buried.

    Returns a new CSR matrix; the input is not modified. weights[post, pre], so
    "incoming" means rows.
    """
    scaled = sp.lil_matrix(weights.shape, dtype=np.float64)
    scaled[:] = weights
    scaled[np.asarray(target_indices), :] = weights[np.asarray(target_indices), :] * factor
    return sp.csr_matrix(scaled)
