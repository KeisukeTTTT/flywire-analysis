"""Per-neuron processing-stage / layer assignment for the optic lobe.

Lateral inhibition acts *within a processing stage*. In the fly optic lobe the
natural stage unit is the neuropil -- lamina (LA), medulla (ME), lobula (LO),
lobula plate (LOP) and accessory medulla (AME) -- following the FlyWire "parts
list" taxonomy (Nern/Matsliah et al. 2024), where optic-lobe intrinsic neurons are
columnar / local-interneuron / cross-neuropil-tangential / cross-neuropil-amacrine.

``assign_stage`` combines three evidence sources:

1. ``visual_neuron_types.family`` -> :data:`FAMILY_TO_STAGE` (literature-curated
   backbone; span / cross-neuropil families are left ``None`` and deferred).
2. ``neuropil_synapse_table`` dominant input / output neuropil (data-driven; gives
   ``input_stage`` / ``output_stage`` and fills neurons missing a family).
3. ``classification.flow`` -> ``is_intrinsic`` (input and output in the same region;
   the gate for a true lateral-inhibition *mediator* such as Lai / Dm / Pm).

For span types (T4, T5, Tm, CT1, ...) the single ``stage`` label is necessarily a
simplification, so ``input_stage`` / ``output_stage`` are kept separate -- the
"same stage" gate downstream should be evaluated on the *input* side, where lateral
pooling happens (T4 input = ME, T5 input = LO).

The fine M1-M10 medulla depth (:func:`assign_medulla_layer`) is an opt-in refinement
that reads the 864 MB ``synapse_coordinates.csv``; it is not needed for the coarse
neuropil-stage gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DATA_DIR

# --- neuropil -> coarse stage --------------------------------------------------
OPTIC_STAGES = ("RETINA", "LA", "ME", "AME", "LO", "LOP")

NEUROPIL_TO_STAGE = {
    "LA": "LA",
    "ME": "ME",
    "AME": "AME",
    "LO": "LO",
    "LOP": "LOP",
}

_IN_PREFIX = "input synapses in "
_OUT_PREFIX = "output synapses in "


def neuropil_base(name):
    """Strip the ``_L`` / ``_R`` hemisphere suffix (``ME_R`` -> ``ME``).

    ``LO`` and ``LOP`` are distinct bases and are preserved -- only the hemisphere
    suffix is removed, never a substring of the neuropil name itself.
    """
    if not isinstance(name, str):
        return name
    if name.endswith("_L") or name.endswith("_R"):
        return name[:-2]
    return name


def neuropil_to_stage(name):
    """Map a (possibly hemisphere-tagged) neuropil to a coarse stage.

    Optic neuropils map to themselves; any central-brain neuropil maps to
    ``"central"``; non-strings (NaN) pass through as NaN.
    """
    if not isinstance(name, str):
        return np.nan
    return NEUROPIL_TO_STAGE.get(neuropil_base(name), "central")


# --- family -> coarse stage (literature-curated) -------------------------------
# Single-neuropil families get a concrete stage. Span / projection / cross-neuropil
# / central families are mapped to ``None`` *on purpose* so the data-driven neuropil
# table decides their stage (and keeps input/output separate). Every ``family`` value
# observed in visual_neuron_types.csv is a key here, so lookups never KeyError.
FAMILY_TO_STAGE = {
    # retina
    "Photo Receptors": "RETINA",
    # lamina
    "Lamina Monopolar": "LA",
    "Lamina Intrinsic": "LA",
    "Lamina Wide Field": "LA",
    "Lamina Tangential": "LA",
    # medulla (intrinsic / local)
    "Distal Medulla": "ME",
    "Proximal Medulla": "ME",
    "Medulla Intrinsic": "ME",
    "Serpentine Medulla": "ME",
    "Distal Medulla Dorsal Rim Area": "ME",
    # lobula / lobula plate (intrinsic)
    "Lobula Intrinsic": "LO",
    "Lobula Plate Intrinsic": "LOP",
    # accessory medulla
    "aMe": "AME",
    # --- deferred to the neuropil table (span / projection / cross-neuropil / central)
    "Transmedullary": None,
    "Transmedullary Y": None,
    "Centrifugal": None,
    "T1 Neuron": None,
    "T2 Neuron": None,
    "T3 Neuron": None,
    "T4 Neuron": None,
    "T5 Neuron": None,
    "Y Neuron": None,
    "MeTu": None,
    "MT": None,
    "MC": None,
    "MeMe": None,
    "MeLp": None,
    "LC": None,
    "LT": None,
    "LPLC": None,
    "LLPC": None,
    "LPC": None,
    "LPT": None,
    "cLP": None,
    "cL": None,
    "cM": None,
    "cLLPM": None,
    "cMLLP": None,
    "cML": None,
    "cLLP": None,
    "cLM": None,
    "Medulla Lobula Tangential": None,
    "Lobula Medulla Tangential": None,
    "Lobula Medulla Amacrine": None,
    "Lobula Lobula Plate Tangential": None,
    "Proximal Distal Medulla Tangential": None,
    "Medulla Lobula Lobula Plate Amacrine": None,
    "Translobula Plate": None,
    "VS": None,
    "HS": None,
    "H": None,
    "Nod": None,
    "CB": None,
    "DN": None,
    "OA": None,
    "mAL": None,
    "AN": None,
    "LN": None,
    "AVLP": None,
    "AOTU": None,
    "VC": None,
    "PLP": None,
    "Weirdos": None,
    "other": None,
    "": None,
}

# Families whose members are local interneurons / amacrine cells -- the canonical
# lateral-inhibition mediators (Lai, Dm, Pm, Sm, Li, LPi, CT1-like amacrines).
LOCAL_INTERNEURON_FAMILIES = frozenset(
    {
        "Lamina Intrinsic",
        "Lamina Wide Field",
        "Distal Medulla",
        "Proximal Medulla",
        "Medulla Intrinsic",
        "Serpentine Medulla",
        "Distal Medulla Dorsal Rim Area",
        "Lobula Intrinsic",
        "Lobula Plate Intrinsic",
        "Lobula Medulla Amacrine",
        "Medulla Lobula Lobula Plate Amacrine",
    }
)


def dominant_neuropils(neuropil_table_df):
    """Reduce the wide ``neuropil_synapse_table`` to dominant input/output neuropil.

    Returns a frame keyed by ``root_id`` with ``dominant_in_np`` / ``dominant_out_np``
    (e.g. ``ME_R``; NaN when the neuron has no input/output synapses) and the input /
    output synapse totals.
    """
    ntab = neuropil_table_df.drop_duplicates("root_id").set_index("root_id")
    in_cols = [c for c in ntab.columns if c.startswith(_IN_PREFIX)]
    out_cols = [c for c in ntab.columns if c.startswith(_OUT_PREFIX)]
    in_mat, out_mat = ntab[in_cols], ntab[out_cols]
    in_sum, out_sum = in_mat.sum(axis=1), out_mat.sum(axis=1)
    dom_in = in_mat.idxmax(axis=1).str[len(_IN_PREFIX):].where(in_sum > 0)
    dom_out = out_mat.idxmax(axis=1).str[len(_OUT_PREFIX):].where(out_sum > 0)
    return pd.DataFrame(
        {
            "dominant_in_np": dom_in,
            "dominant_out_np": dom_out,
            "n_in_syn": in_sum.astype(int),
            "n_out_syn": out_sum.astype(int),
        }
    ).reset_index()


def assign_stage(neurons_df, *, visual_types_df=None, neuropil_table_df=None):
    """Assign a processing stage / layer to each neuron.

    Args:
        neurons_df: must contain ``root_id`` and ``flow`` (and ideally
            ``primary_type``) -- e.g. ``FlyWireDataManager.optic_lobe_neurons_df``.
        visual_types_df: ``visual_neuron_types.csv`` (``root_id``, ``family``).
        neuropil_table_df: ``neuropil_synapse_table.csv`` (wide per-neuropil counts).

    Returns:
        DataFrame keyed by ``root_id`` with ``primary_type, family, flow,
        is_intrinsic, dominant_in_np, dominant_out_np, input_stage, output_stage,
        stage, stage_source, stage_confidence, is_local_interneuron_family``.
    """
    n = neurons_df.drop_duplicates("root_id").set_index("root_id")
    out = pd.DataFrame(index=n.index)
    out["primary_type"] = n["primary_type"] if "primary_type" in n.columns else pd.NA
    out["flow"] = n["flow"] if "flow" in n.columns else pd.NA
    out["is_intrinsic"] = out["flow"].eq("intrinsic")

    # (1) family -> curated stage
    if visual_types_df is not None:
        fam = visual_types_df.drop_duplicates("root_id").set_index("root_id")["family"]
        out["family"] = out.index.map(fam)
    else:
        out["family"] = pd.NA
    out["family_stage"] = out["family"].map(
        lambda f: FAMILY_TO_STAGE.get(f) if isinstance(f, str) else None
    )

    # (2) neuropil table -> data-driven input / output stage
    if neuropil_table_df is not None:
        dom = dominant_neuropils(neuropil_table_df).set_index("root_id")
        out["dominant_in_np"] = out.index.map(dom["dominant_in_np"])
        out["dominant_out_np"] = out.index.map(dom["dominant_out_np"])
    else:
        out["dominant_in_np"] = pd.NA
        out["dominant_out_np"] = pd.NA
    out["input_stage"] = out["dominant_in_np"].map(neuropil_to_stage)
    out["output_stage"] = out["dominant_out_np"].map(neuropil_to_stage)

    # single coarse stage: family first, else dominant input, else dominant output
    out["stage"] = out["family_stage"]
    out["stage_source"] = np.where(out["family_stage"].notna(), "family", None)
    for col, src in (("input_stage", "neuropil_in"), ("output_stage", "neuropil_out")):
        mask = out["stage"].isna() & out[col].notna()
        out.loc[mask, "stage"] = out.loc[mask, col]
        out.loc[mask, "stage_source"] = src
    out["stage"] = out["stage"].fillna("other")
    out["stage_source"] = out["stage_source"].fillna("none")

    # confidence: does the curated family stage agree with the data-driven one?
    ns = out["input_stage"].where(out["input_stage"].notna(), out["output_stage"])
    fs = out["family_stage"]
    out["stage_confidence"] = np.where(
        fs.isna() | ns.isna(), "single_source", np.where(fs == ns, "agree", "mismatch")
    )

    out["is_local_interneuron_family"] = out["family"].isin(LOCAL_INTERNEURON_FAMILIES)
    return out.reset_index()


def assign_stage_from_manager(manager, *, use_neuropil_table=True):
    """Convenience: build the stage table directly from a ``FlyWireDataManager``.

    Pulls ``optic_lobe_neurons_df`` plus the lazily-loaded ``visual_neuron_types`` and
    (optionally) ``neuropil_synapse_table`` accessors.
    """
    ntab = manager.get_neuropil_synapse_table_df() if use_neuropil_table else None
    return assign_stage(
        manager.optic_lobe_neurons_df,
        visual_types_df=manager.get_visual_neuron_types_df(),
        neuropil_table_df=ntab,
    )


def is_intrinsic_inhibitory_mediator(stage_table, output_inh_frac=None, *, min_inh_frac=0.5):
    """Boolean Series flagging likely lateral-inhibition mediators.

    A mediator is ``is_intrinsic`` (flow == intrinsic; input and output in the same
    region) AND a local-interneuron / amacrine ``family``. If ``output_inh_frac`` (a
    Series keyed by ``root_id`` giving the inhibitory fraction of each neuron's
    output, e.g. from :func:`src.lateral.classifier.output_sign_fraction`) is given,
    the neuron must also be inhibition-dominant (``>= min_inh_frac``).
    """
    st = stage_table.set_index("root_id") if "root_id" in stage_table.columns else stage_table
    flag = st["is_intrinsic"].fillna(False) & st["is_local_interneuron_family"].fillna(False)
    if output_inh_frac is not None:
        inh = st.index.map(output_inh_frac).astype(float)
        flag = flag & (pd.Series(inh, index=st.index).fillna(0.0) >= min_inh_frac)
    return flag


# --- opt-in fine layer: medulla M1-M10 depth -----------------------------------
@dataclass
class MedullaDepthRuler:
    """Local medulla depth axis from the Mi1 synapse-centroid PCA (see Q13)."""

    normal: np.ndarray
    c0: np.ndarray
    lo: float
    hi: float
    flipped: bool

    def depth(self, xyz):
        return (np.asarray(xyz, dtype=float) - self.c0) @ self.normal

    def rel_depth(self, xyz):
        r = (self.depth(xyz) - self.lo) / (self.hi - self.lo)
        return (1.0 - r) if self.flipped else r


def assign_medulla_layer(
    neurons_df,
    *,
    side="right",
    types=None,
    data_dir=DATA_DIR,
    reference_type="Mi1",
    min_syn_for_centroid=20,
    distal_types=("Dm1", "Dm4", "Dm12"),
    proximal_types=("Pm04", "Pm08", "Pm09"),
):
    """Reconstruct relative medulla depth (0 = distal/M1 .. 1 = proximal/M10).

    Opt-in and heavy: reads ``synapse_coordinates.csv`` (~864 MB). Faithful refactor
    of ``lateral_inhibition_extended.py`` Q13 -- PCA min-variance axis of the
    ``reference_type`` (Mi1) synapse-centroid cloud is the local depth normal,
    calibrated to [0, 1] by the reference's 1-99 percentile, oriented so distal
    families sit at small depth.

    Returns ``(sc, ruler)``: ``sc`` has ``pre_root_id, x, y, z, ptype, depth,
    rel_depth`` for the reference + requested ``types`` on ``side``; ``ruler`` is a
    :class:`MedullaDepthRuler`.
    """
    neur = neurons_df.drop_duplicates("root_id").set_index("root_id")
    ptype, pside = neur["primary_type"], neur["side"]
    wanted = set(distal_types) | set(proximal_types) | {reference_type}
    if types is not None:
        wanted |= set(types)
    keep_ids = set(neur.index[(pside == side) & ptype.isin(wanted)])
    ref_ids = set(neur.index[(pside == side) & (ptype == reference_type)])

    syn_path = os.path.join(data_dir, "raw", "flywire", "csv", "synapse_coordinates.csv")
    sc = pd.read_csv(syn_path, dtype={"pre_root_id": str}, usecols=["pre_root_id", "x", "y", "z"])
    sc["pre_root_id"] = sc["pre_root_id"].ffill()
    sc = sc.dropna(subset=["pre_root_id"])
    sc = sc[sc["pre_root_id"].isin(keep_ids)].copy()
    sc["ptype"] = sc["pre_root_id"].map(ptype)

    ref_sc = sc[sc["pre_root_id"].isin(ref_ids)]
    cnt = ref_sc.groupby("pre_root_id").size()
    cents = ref_sc.groupby("pre_root_id")[["x", "y", "z"]].mean()[cnt >= min_syn_for_centroid]
    if len(cents) < 2:
        raise ValueError(
            f"too few {reference_type!r} reference cells on side {side!r} with "
            f">= {min_syn_for_centroid} synapses ({len(cents)} found); the depth axis "
            "cannot be fit -- lower min_syn_for_centroid or check reference_type/side."
        )
    Xc = cents.values - cents.values.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    normal, c0 = Vt[-1], cents.values.mean(axis=0)
    sc["depth"] = (sc[["x", "y", "z"]].values - c0) @ normal

    ref_depth = sc.loc[sc["ptype"] == reference_type, "depth"]
    if ref_depth.empty:
        raise ValueError(
            f"no {reference_type!r} synapses on side {side!r} to calibrate the depth "
            "range; check reference_type/side."
        )
    lo, hi = np.percentile(ref_depth, [1, 99])
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    flipped = False
    rel = (sc["depth"] - lo) / (hi - lo)
    distal_med = rel[sc["ptype"].isin(distal_types)].median()
    proximal_med = rel[sc["ptype"].isin(proximal_types)].median()
    if pd.notna(distal_med) and pd.notna(proximal_med) and distal_med > proximal_med:
        flipped = True
        rel = 1.0 - rel
    sc["rel_depth"] = rel
    return sc, MedullaDepthRuler(normal=normal, c0=c0, lo=lo, hi=hi, flipped=flipped)
