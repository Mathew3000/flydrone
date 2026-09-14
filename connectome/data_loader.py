"""
Connectivity data loading.

Two sources, same output shape -- a dict with:
    "n_neurons": int
    "weights": scipy.sparse matrix (n_neurons, n_neurons), weights[post, pre]
    "cell_type_indices": dict[str, np.ndarray of neuron indices]

1. `make_toy_network()` -- a small hand-built 3-layer network (input ->
   motion/looming-like cells -> motor-like cells) with no real biology, used
   to validate the whole pipeline (medulla encoder -> LIFNetwork -> motor
   decoder) mechanically. This is what scripts/m1_offline_test.py currently
   uses. See docs/connectome-data-access.md for why we don't have real
   MaleCNS data wired in yet.

2. `fetch_from_neuprint()` -- sketches how the real data comes in later via
   neuprint-python once Matthias has a neuPrint account + API token. NOT
   runnable as-is without that token; documented here so the shape of the
   real integration is decided now rather than as an afterthought, even
   though it can't be exercised yet.
"""
from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp


def _load_dotenv(path: str | None = None) -> None:
    """Minimal .env loader (no python-dotenv dependency): reads KEY=VALUE
    lines from flydrone/.env, if present, into os.environ -- but never
    overwrites a variable that's already set in the real environment.
    Deliberately tiny/dependency-free; see .env.example for the expected
    format. Windows env vars are enough of a hassle that a project-local
    .env file (gitignored) is the recommended path -- see
    docs/connectome-data-access.md.
    """
    env_path = path or os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


def make_toy_network(seed: int = 0):
    """A small synthetic network standing in for the real connectome, purely
    to prove the pipeline wiring end-to-end (see M1 in the project plan).

    Layout:
      - 64 "photoreceptor-like" input neurons (not directly driven -- the
        medulla encoder skips the retina on purpose, same as the real project)
      - 32 motion cells: T4_right / T4_left / T5_up / T5_down (8 each)
      - 16 looming cells: LC4 (8), LPLC2 (8)
      - 16 "motor-like" output neurons, split left/right (8 each), each
        pooling from both motion and looming layers with random weights
    """
    rng = np.random.default_rng(seed)
    n_input, n_motion, n_loom, n_motor = 64, 32, 16, 16
    n = n_input + n_motion + n_loom + n_motor

    offset_motion = n_input
    offset_loom = offset_motion + n_motion
    offset_motor = offset_loom + n_loom

    cell_type_indices = {
        "T4_right": np.arange(offset_motion, offset_motion + 8),
        "T4_left": np.arange(offset_motion + 8, offset_motion + 16),
        "T5_up": np.arange(offset_motion + 16, offset_motion + 24),
        "T5_down": np.arange(offset_motion + 24, offset_motion + 32),
        "LC4": np.arange(offset_loom, offset_loom + 8),
        "LPLC2": np.arange(offset_loom + 8, offset_loom + 16),
        "motor_left": np.arange(offset_motor, offset_motor + 8),
        "motor_right": np.arange(offset_motor + 8, offset_motor + 16),
    }

    rows, cols, vals = [], [], []

    def connect(pre_idx, post_idx, p_connect, w_mean, w_std):
        for post in post_idx:
            for pre in pre_idx:
                if rng.random() < p_connect:
                    w = max(rng.normal(w_mean, w_std), 0.0)
                    rows.append(post)
                    cols.append(pre)
                    vals.append(w)

    # motion/looming layers feed the motor layer; motor_left biased toward
    # T4_left/T5_up/LC4, motor_right toward T4_right/T5_down/LPLC2 -- purely
    # arbitrary, just needs to be asymmetric enough to be testable.
    connect(cell_type_indices["T4_left"], cell_type_indices["motor_left"], 0.5, 0.15, 0.05)
    connect(cell_type_indices["T5_up"], cell_type_indices["motor_left"], 0.5, 0.15, 0.05)
    connect(cell_type_indices["LC4"], cell_type_indices["motor_left"], 0.5, 0.15, 0.05)
    connect(cell_type_indices["T4_right"], cell_type_indices["motor_right"], 0.5, 0.15, 0.05)
    connect(cell_type_indices["T5_down"], cell_type_indices["motor_right"], 0.5, 0.15, 0.05)
    connect(cell_type_indices["LPLC2"], cell_type_indices["motor_right"], 0.5, 0.15, 0.05)

    weights = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return {
        "n_neurons": n,
        "weights": weights,
        "cell_type_indices": cell_type_indices,
    }


def fetch_from_neuprint(cell_types: list[str], dataset: str = "male-cns:v0.9", token: str | None = None):
    """Sketch of the real data path -- requires `pip install neuprint-python`
    and a neuPrint account/token (see docs/connectome-data-access.md).
    Not exercised by any test yet; the exact dataset tag and cell-type name
    strings (e.g. is it "T4a" or something else in MaleCNS) need confirming
    once real access is available.
    """
    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

    token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError(
            "No neuPrint token found. Set NEUPRINT_APPLICATION_CREDENTIALS "
            "or pass token= explicitly. See docs/connectome-data-access.md."
        )
    Client("neuprint.janelia.org", dataset=dataset, token=token)

    neuron_df, _ = fetch_neurons(NC(type=cell_types))
    body_ids = neuron_df["bodyId"].to_numpy()
    _, conn_df = fetch_adjacencies(NC(bodyId=body_ids), NC(bodyId=body_ids))

    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}
    n = len(body_ids)
    rows = conn_df["bodyId_post"].map(id_to_idx).to_numpy()
    cols = conn_df["bodyId_pre"].map(id_to_idx).to_numpy()
    vals = conn_df["weight"].to_numpy(dtype=np.float64)
    weights = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    cell_type_indices = {}
    for t in cell_types:
        mask = neuron_df["type"].astype(str).str.fullmatch(t) if any(c in t for c in ".*") \
            else neuron_df["type"] == t
        cell_type_indices[t] = np.where(mask.to_numpy())[0]

    return {"n_neurons": n, "weights": weights, "cell_type_indices": cell_type_indices}
