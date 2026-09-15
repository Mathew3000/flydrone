"""
Query the local MaleCNS bulk-download files directly (see
docs/connectome-data-access.md) instead of neuPrint's API.

Why this exists: the neuPrint API works (connectome/data_loader.py::
fetch_from_neuprint), but every query goes over the network from Matthias's
machine (this Cowork sandbox can't reach neuprint.janelia.org at all -- see
docs), and broad "what connects to what across the whole brain" questions
are slow/awkward one query at a time. The MaleCNS bulk files are public,
direct downloads (no token) and, once local, this module answers the same
kinds of questions with zero network calls and no query-size limits.

Files expected in data/raw/ (see docs/connectome-data-access.md for the
download links):
  - body-annotations-male-cns-v1.0-minconf-0.5.feather  (~14MB, one row per
    body: bodyId, type, instance, superclass, class, ...)
  - connectome-weights-male-cns-v1.0-minconf-0.5.feather (~1.1GB, ~152M rows:
    body_pre, body_post, weight -- the WHOLE CNS connectivity graph, not
    filtered to any cell type)

Memory note: this sandbox VM has ~3.8GB RAM and no swap. `pd.read_feather()`
on the weights file OOM-kills the process (confirmed 2026-09-14) --
materializing 152M rows x 3 int64 columns as a pandas DataFrame, plus
decompression overhead, doesn't fit. Every function here instead streams the
weights file batch-by-batch via pyarrow's IPC reader (get_batch(i) one at a
time, ~65536 rows each, discarded after filtering) -- confirmed to scan all
~152M rows in ~5 seconds using negligible memory. Do not replace this with
pd.read_feather(weights_path) without re-checking memory.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

DATA_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def load_annotations(data_dir: str = DATA_DIR_DEFAULT, columns=("bodyId", "type", "instance")):
    path = os.path.join(data_dir, ANNOTATIONS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found -- see docs/connectome-data-access.md for the download link.")
    return pd.read_feather(path, columns=list(columns))


def resolve_type_ids(ann_df: pd.DataFrame, type_patterns: list[str]) -> np.ndarray:
    """Same matching convention as fetch_from_neuprint()/NeuronCriteria: a
    pattern containing regex metacharacters (only '.' and '*' expected, e.g.
    'T4.*') is matched with re.fullmatch; anything else must match exactly
    (e.g. 'LC4')."""
    mask = np.zeros(len(ann_df), dtype=bool)
    types = ann_df["type"].astype(str)
    for pat in type_patterns:
        if any(c in pat for c in ".*"):
            mask |= types.str.fullmatch(pat).fillna(False).to_numpy()
        else:
            mask |= (types == pat).to_numpy()
    return ann_df.loc[mask, "bodyId"].to_numpy()


def _stream_weights(data_dir: str, pre_ids: np.ndarray | None, post_ids: np.ndarray | None,
                     require_both: bool = False):
    """Yields (pre, post, weight) numpy arrays per batch, filtered so that
    (pre in pre_ids if given) [AND/OR, per require_both] (post in post_ids if given).
    require_both=True -> both sides must match (for building a subnetwork's
    internal adjacency). require_both=False with only pre_ids given -> "pre
    matches, post is anything" (for downstream exploration).
    """
    path = os.path.join(data_dir, WEIGHTS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found -- see docs/connectome-data-access.md for the download link.")
    pre_set = None if pre_ids is None else np.asarray(pre_ids)
    post_set = None if post_ids is None else np.asarray(post_ids)

    with pa.memory_map(path, "r") as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            weight = batch.column("weight").to_numpy()

            pre_mask = np.isin(pre, pre_set) if pre_set is not None else np.ones(len(pre), dtype=bool)
            post_mask = np.isin(post, post_set) if post_set is not None else np.ones(len(post), dtype=bool)
            mask = (pre_mask & post_mask) if require_both else (
                pre_mask if post_set is None else
                post_mask if pre_set is None else
                (pre_mask | post_mask)
            )
            if mask.any():
                yield pre[mask], post[mask], weight[mask]


def local_downstream_types(source_types: list[str], top_n: int = 25, data_dir: str = DATA_DIR_DEFAULT) -> pd.DataFrame:
    """Local, network-free equivalent of data_loader.explore_downstream_types().
    No query-size limit -- fine to pass all of T4.*/T5.* (13.5k source
    neurons) at once, unlike the neuPrint version."""
    ann = load_annotations(data_dir)
    source_ids = resolve_type_ids(ann, source_types)
    if len(source_ids) == 0:
        raise RuntimeError(f"No bodies matched {source_types}")

    pre_chunks, post_chunks, w_chunks = [], [], []
    for pre, post, weight in _stream_weights(data_dir, pre_ids=source_ids, post_ids=None):
        pre_chunks.append(pre)
        post_chunks.append(post)
        w_chunks.append(weight)
    if not pre_chunks:
        raise RuntimeError(f"{source_types} matched {len(source_ids)} bodies but they have 0 outgoing edges.")

    post_all = np.concatenate(post_chunks)
    w_all = np.concatenate(w_chunks)
    edges = pd.DataFrame({"bodyId_post": post_all, "weight": w_all})
    edges = edges.merge(ann[["bodyId", "type"]], left_on="bodyId_post", right_on="bodyId", how="left")

    summary = (
        edges.groupby("type")
        .agg(total_weight=("weight", "sum"), n_bodies=("bodyId_post", "nunique"))
        .sort_values("total_weight", ascending=False)
        .reset_index()
    )
    return summary.head(top_n)


def local_fetch_subnetwork(cell_types: list[str], data_dir: str = DATA_DIR_DEFAULT) -> dict:
    """Local, network-free equivalent of data_loader.fetch_from_neuprint():
    same return shape ({"n_neurons", "weights", "cell_type_indices"}), built
    entirely from the local bulk files -- both endpoints of each edge must
    be in the requested cell_types (internal connectivity only, matching
    fetch_from_neuprint's omit_rois=True adjacency)."""
    ann = load_annotations(data_dir)
    body_ids = resolve_type_ids(ann, cell_types)
    if len(body_ids) == 0:
        raise RuntimeError(f"No bodies matched {cell_types}")

    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}
    n = len(body_ids)

    rows_all, cols_all, vals_all = [], [], []
    for pre, post, weight in _stream_weights(data_dir, pre_ids=body_ids, post_ids=body_ids, require_both=True):
        rows_all.append(np.array([id_to_idx[b] for b in post]))
        cols_all.append(np.array([id_to_idx[b] for b in pre]))
        vals_all.append(weight.astype(np.float64))

    if rows_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        vals = np.concatenate(vals_all)
    else:
        rows = cols = vals = np.array([])
    weights = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    types = ann.set_index("bodyId").loc[body_ids, "type"].to_numpy()
    cell_type_indices = {}
    for t in cell_types:
        if any(c in t for c in ".*"):
            keep = pd.Series(types).str.fullmatch(t).fillna(False).to_numpy()
        else:
            keep = (types == t)
        cell_type_indices[t] = np.where(keep)[0]

    return {"n_neurons": n, "weights": weights, "cell_type_indices": cell_type_indices}


def local_fetch_lr_subnetwork(
    visual_types: list[str] = ["T4.*", "T5.*"],
    course_types: list[str] = ["HSE", "HSN", "HSS", "HST", "H1", "H2"],
    data_dir: str = DATA_DIR_DEFAULT,
) -> dict:
    """Real-connectome subnetwork split by anatomical side (somaSide: L/R),
    for use as a drop-in replacement of data_loader.make_toy_network().

    visual_types (default T4/T5, confirmed >90% of HS/H1/H2's modeled input
    in this subnetwork -- see docs/connectome-data-access.md) become the
    "visual_L"/"visual_R" drive populations: feed motion energy from the
    left/right half of the camera image into the neurons on that
    anatomical side (a real, if simplified, retinotopic mapping -- the fly's
    two optic lobes each see one visual hemifield).

    course_types (default HSE/HSN/HSS/HST/H1/H2 -- the real lobula plate
    tangential cells T4/T5 drive, confirmed by local_downstream_types())
    become the "motor_left"/"motor_right" decode populations, read the same
    way connectome.motor_decoder already reads the toy network's
    motor_left/motor_right pools.

    Returns the same shape as local_fetch_subnetwork(), with extra
    "visual_L"/"visual_R"/"motor_left"/"motor_right" keys added to
    cell_type_indices alongside the individual per-type keys.
    """
    all_types = list(visual_types) + list(course_types)
    net = local_fetch_subnetwork(all_types, data_dir=data_dir)

    ann = load_annotations(data_dir, columns=("bodyId", "type", "somaSide"))
    body_ids = resolve_type_ids(ann, all_types)
    sides = ann.set_index("bodyId").loc[body_ids, "somaSide"].to_numpy()

    visual_mask = np.zeros(len(body_ids), dtype=bool)
    for t in visual_types:
        visual_mask |= np.isin(np.arange(len(body_ids)), net["cell_type_indices"][t])
    course_mask = np.zeros(len(body_ids), dtype=bool)
    for t in course_types:
        course_mask |= np.isin(np.arange(len(body_ids)), net["cell_type_indices"][t])

    net["cell_type_indices"]["visual_L"] = np.where(visual_mask & (sides == "L"))[0]
    net["cell_type_indices"]["visual_R"] = np.where(visual_mask & (sides == "R"))[0]
    net["cell_type_indices"]["motor_left"] = np.where(course_mask & (sides == "L"))[0]
    net["cell_type_indices"]["motor_right"] = np.where(course_mask & (sides == "R"))[0]
    return net

# Data-driven, not picked from the literature: these are the descending neurons
# that actually appear downstream of HS/H1/H2 in this dataset, ranked by total
# synaptic weight (local_downstream_types(["HSE","HSN","HSS","HST","H1","H2"])),
# restricted to types with a clean bilateral pair so a left/right decode is
# possible at all. DNa02 is the canonical steering DN of the literature and is
# in here on its own merits -- it does receive HS input (weight 182) -- but it
# is far from the strongest, which is exactly why the type list stays a
# parameter and the optomotor script reports selectivity per DN type instead of
# assuming which one carries the turning signal.
DEFAULT_DN_TYPES = ["DNb03", "DNg41", "DNp15", "DNp17", "DNa02"]


def local_fetch_dn_subnetwork(
    visual_types: list[str] = ["T4.*", "T5.*"],
    course_types: list[str] = ["HSE", "HSN", "HSS", "HST", "H1", "H2"],
    dn_types: list[str] | None = None,
    data_dir: str = DATA_DIR_DEFAULT,
) -> dict:
    """Three-layer subnetwork: T4/T5 -> HS/H1/H2 -> descending neurons.

    Why this exists on top of local_fetch_lr_subnetwork(): that one decodes the
    motor command straight off HS/H1/H2, which are lobula plate tangential
    cells -- interneurons, not an output of the brain. Reading a "motor" signal
    off them is a stand-in. Descending neurons are the real thing: they are the
    brain's output to the ventral nerve cord, and the data shows HS/H1/H2 drive
    a handful of them directly. Decoding there removes one layer of pretending
    from the claim that the connectome drives the drone.

    cell_type_indices gains, alongside the per-type keys:
      visual_L / visual_R    T4/T5 by somaSide -- the encoder's drive targets
      course_L / course_R    HS/H1/H2 by somaSide (what the two-layer version
                             decoded from; kept so the layers can be compared)
      motor_left / motor_right   the DNs by somaSide -- the new decode pools
      dn_<type>_L / dn_<type>_R  each DN type separately, so direction
                             selectivity can be attributed per type rather than
                             assumed

    Pools are small (1-6 neurons per side per type, 11 per side in total with
    the default list) -- comparable to the 6 per side the HS/H1/H2 version
    used, so decode noise should be of the same order, but it is worth checking
    rather than assuming: see scripts/m3_gain_sweep.py.
    """
    dn_types = list(DEFAULT_DN_TYPES if dn_types is None else dn_types)
    all_types = list(visual_types) + list(course_types) + dn_types
    net = local_fetch_subnetwork(all_types, data_dir=data_dir)

    ann = load_annotations(data_dir, columns=("bodyId", "type", "somaSide"))
    body_ids = resolve_type_ids(ann, all_types)
    sides = ann.set_index("bodyId").loc[body_ids, "somaSide"].to_numpy()

    def _mask_for(types):
        mask = np.zeros(len(body_ids), dtype=bool)
        for t in types:
            mask |= np.isin(np.arange(len(body_ids)), net["cell_type_indices"][t])
        return mask

    cti = net["cell_type_indices"]
    visual, course, motor = _mask_for(visual_types), _mask_for(course_types), _mask_for(dn_types)
    cti["visual_L"] = np.where(visual & (sides == "L"))[0]
    cti["visual_R"] = np.where(visual & (sides == "R"))[0]
    cti["course_L"] = np.where(course & (sides == "L"))[0]
    cti["course_R"] = np.where(course & (sides == "R"))[0]
    cti["motor_left"] = np.where(motor & (sides == "L"))[0]
    cti["motor_right"] = np.where(motor & (sides == "R"))[0]
    for t in dn_types:
        t_mask = _mask_for([t])
        cti[f"dn_{t}_L"] = np.where(t_mask & (sides == "L"))[0]
        cti[f"dn_{t}_R"] = np.where(t_mask & (sides == "R"))[0]

    return net


# Data-driven like DEFAULT_DN_TYPES: the descending neurons that actually come
# out of local_downstream_types(["LC4", "LPLC2"]) by weight, restricted to
# bilateral pairs. DNp01 is the Giant Fiber. Note how much stronger this
# pathway is than the course-control one -- LC4/LPLC2 -> DNp04 carries weight
# 14995 and -> DNp01 11224, where the whole of HS/H1/H2 -> DNb03 manages 1266.
# That asymmetry is the point of commit 1f09a4f's finding: this is an escape
# reflex, built to fire hard and rarely, not to steer continuously.
DEFAULT_ESCAPE_DN_TYPES = ["DNp01", "DNp04", "DNp02", "DNp11", "DNp03", "DNp06"]


def local_fetch_looming_subnetwork(
    looming_types: list[str] = ["LC4", "LPLC2"],
    dn_types: list[str] | None = None,
    data_dir: str = DATA_DIR_DEFAULT,
) -> dict:
    """Looming pathway: LC4/LPLC2 -> escape descending neurons.

    The companion to local_fetch_dn_subnetwork(), which wires the optomotor
    course-control pathway. Same shape, same conventions, different behaviour:
    that one turns continuously in response to rotation, this one is expected
    to do nothing at all until something approaches and then fire a transient.

    cell_type_indices gains:
      looming_L / looming_R    LC4+LPLC2 by somaSide -- the encoder's targets
      motor_left / motor_right the escape DNs by somaSide, so the existing
                               motor_decoder pools work unchanged
      dn_<type>_L / dn_<type>_R  per DN type, to attribute a response rather
                               than assume which neuron carries it (the same
                               approach that found DNg41 rather than DNa02
                               carrying the turning signal)
    """
    dn_types = list(DEFAULT_ESCAPE_DN_TYPES if dn_types is None else dn_types)
    all_types = list(looming_types) + dn_types
    net = local_fetch_subnetwork(all_types, data_dir=data_dir)

    ann = load_annotations(data_dir, columns=("bodyId", "type", "somaSide"))
    body_ids = resolve_type_ids(ann, all_types)
    sides = ann.set_index("bodyId").loc[body_ids, "somaSide"].to_numpy()

    def _mask_for(types):
        mask = np.zeros(len(body_ids), dtype=bool)
        for t in types:
            mask |= np.isin(np.arange(len(body_ids)), net["cell_type_indices"][t])
        return mask

    cti = net["cell_type_indices"]
    looming, motor = _mask_for(looming_types), _mask_for(dn_types)
    cti["looming_L"] = np.where(looming & (sides == "L"))[0]
    cti["looming_R"] = np.where(looming & (sides == "R"))[0]
    cti["motor_left"] = np.where(motor & (sides == "L"))[0]
    cti["motor_right"] = np.where(motor & (sides == "R"))[0]
    for t in dn_types:
        t_mask = _mask_for([t])
        cti[f"dn_{t}_L"] = np.where(t_mask & (sides == "L"))[0]
        cti[f"dn_{t}_R"] = np.where(t_mask & (sides == "R"))[0]
    return net


# The vertical system's descending targets, from
# local_downstream_types(["VS"]) by weight, bilateral pairs only.
DEFAULT_VERTICAL_DN_TYPES = ["DNp20", "DNp17", "DNge043", "DNp53", "DNb06", "DNp22"]


def local_fetch_vertical_subnetwork(
    visual_types: list[str] = ["T4.*", "T5.*"],
    vertical_types: list[str] = ["VS"],
    dn_types: list[str] | None = None,
    data_dir: str = DATA_DIR_DEFAULT,
) -> dict:
    """Vertical system: T4/T5 -> VS -> descending neurons. The roll/pitch
    counterpart of local_fetch_dn_subnetwork()'s yaw pathway.

    Finding VS at all took a correction. An earlier pass concluded the vertical
    system was absent from MaleCNS because a search for `VS\\d+` returned
    nothing, and M4 was marked blocked on that. The cells are there: MaleCNS
    collapses all eight subtypes into a single type named exactly "VS", which
    the annotations' flywireType column spells out as
    "VS1,VS2,VS3,VS4,VS5,VS6,VS7,VS8". A regex demanding a digit could not
    match it. 18 cells, 9 per side.

    The pathway is real and not a curiosity: T4/T5 -> VS carries total synaptic
    weight 158026, which is 8779 per VS cell against 13164 per HS cell -- the
    same order, from the same input population.

    cell_type_indices gains visual_L/visual_R (the encoder's targets),
    vertical_L/vertical_R (the VS cells by somaSide) and motor_left/motor_right
    (the descending neurons), plus dn_<type>_L/_R per type.
    """
    dn_types = list(DEFAULT_VERTICAL_DN_TYPES if dn_types is None else dn_types)
    all_types = list(visual_types) + list(vertical_types) + dn_types
    net = local_fetch_subnetwork(all_types, data_dir=data_dir)

    ann = load_annotations(data_dir, columns=("bodyId", "type", "somaSide"))
    body_ids = resolve_type_ids(ann, all_types)
    sides = ann.set_index("bodyId").loc[body_ids, "somaSide"].to_numpy()

    def _mask_for(types):
        mask = np.zeros(len(body_ids), dtype=bool)
        for t in types:
            mask |= np.isin(np.arange(len(body_ids)), net["cell_type_indices"][t])
        return mask

    cti = net["cell_type_indices"]
    visual, vertical, motor = (_mask_for(visual_types), _mask_for(vertical_types),
                               _mask_for(dn_types))
    cti["visual_L"] = np.where(visual & (sides == "L"))[0]
    cti["visual_R"] = np.where(visual & (sides == "R"))[0]
    cti["vertical_L"] = np.where(vertical & (sides == "L"))[0]
    cti["vertical_R"] = np.where(vertical & (sides == "R"))[0]
    cti["motor_left"] = np.where(motor & (sides == "L"))[0]
    cti["motor_right"] = np.where(motor & (sides == "R"))[0]
    for t in dn_types:
        t_mask = _mask_for([t])
        cti[f"dn_{t}_L"] = np.where(t_mask & (sides == "L"))[0]
        cti[f"dn_{t}_R"] = np.where(t_mask & (sides == "R"))[0]
    return net
