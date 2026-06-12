"""Operational classifier for inhibitory connections.

Combines stage (:mod:`stage`), edge region (:mod:`edge_region`), Δcolumn geometry
(:mod:`hexgeom`) and sign (:mod:`nt_sign`) to label every inhibitory edge by what it
*is*, so that genuine lateral inhibition stops being conflated with feedforward /
feedback inhibition that merely happens to be spatially wide.

Labels (stage + geometry):

* ``direct_lateral``        -- same home stage, columnar source offset by
  ``>= min_offset_cols`` columns (a neighbor-column monosynaptic contact).
* ``wide_field_lateral``    -- same home stage, source has no single column *and*
  belongs to a local-interneuron / amacrine family (the canonical lateral-inhibition
  mediator). The family gate (``wide_field_requires_local_family``, on by default)
  stops a merely-missing column coordinate -- a central neuron projecting in, an
  unreconstructed cell, a type never column-mapped -- from masquerading as wide-field.
* ``uncolumned_other``      -- same home stage, no column coordinate, but *not* a
  local-interneuron family: an uncolumned source whose wide-field morphology is not
  established. Kept separate so it is neither counted as lateral nor silently dropped.
* ``co_columnar``           -- same home stage but within the home column
  (``Δcol < min_offset_cols``); spatially central, not lateral.
* ``feedforward_inhibition``-- pre is an earlier stage than post (RETINA<LA<ME<LO/LOP).
* ``feedback_inhibition``   -- pre is a later stage than post.
* ``cross_parallel``        -- different stage, equal rank (e.g. LO <-> LOP).
* ``unknown``               -- a stage is missing.

``is_disinhibitory`` (post is inhibition-dominant) and ``pre_intrinsic`` are added as
orthogonal flags rather than competing labels, so e.g. a lateral contact onto an
inhibitory cell is both ``direct_lateral`` and ``is_disinhibitory``.

The disynaptic *mediated* surround (exc->inh->T) is a two-hop construct and is
quantified separately via :class:`src.lateral.pathtrace.RadialKernels`; here the
monosynaptic ``wide_field_lateral`` edges are the first hop of that motif.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .edge_region import tag_edges
from .hexgeom import hex_distance, pq_hemi_maps
from .nt_sign import add_sign

# Processing-flow rank for feedforward/feedback ordering. LO and LOP are parallel
# outputs of the medulla (equal rank); AME runs parallel to ME.
STAGE_RANK = {"RETINA": 0, "LA": 1, "ME": 2, "AME": 2, "LO": 3, "LOP": 3}

# Fine rank used when same_stage_def="sublayer": the medulla M1-M10 sublayers are
# ordered distal -> proximal *within* the (1, 3) gap between LA and LO/LOP, so a
# cross-sublayer ME->ME inhibitory edge (e.g. distal -> proximal) sorts into
# feedforward/feedback instead of being lumped as same-stage. An unresolved bare "ME"
# sits at the neutral middle.
SUBLAYER_RANK = {
    **STAGE_RANK,
    "ME": 2.5,
    "ME:distal": 2.2,
    "ME:medial": 2.5,
    "ME:proximal": 2.8,
}


def stage_rank(stage):
    return STAGE_RANK.get(stage, np.nan)


def output_sign_fraction(conn, *, by="pre_root_id"):
    """Per node (or per type) inhibitory / excitatory output synapse fractions.

    Returns a frame indexed by ``by`` with ``inh, exc, other, total, inh_frac,
    exc_frac``. Used to identify inhibition-dominant cells (mediators / disinhibition
    targets).
    """
    if "sign" not in conn.columns:
        conn = add_sign(conn)
    g = conn.groupby([by, "sign"])["syn_count"].sum().unstack(fill_value=0)
    for c in ("inh", "exc", "other"):
        if c not in g.columns:
            g[c] = 0
    g["total"] = g[["inh", "exc", "other"]].sum(axis=1)
    denom = g["total"].clip(lower=1)
    g["inh_frac"] = g["inh"] / denom
    g["exc_frac"] = g["exc"] / denom
    return g


@dataclass
class LateralInhibitionCriteria:
    """Thresholds for :func:`classify_inhibition`."""

    min_syn: int = 5
    same_stage_def: str = "home"  # "home" or "syn" (see edge_region.tag_edges)
    min_offset_cols: float = 1.0  # Δcolumn >= this counts as lateral (excludes home column)
    inh_dominant_frac: float = 0.5  # post inh_frac >= this -> disinhibitory
    # Require an uncolumned source to be a local-interneuron / amacrine family before
    # it is labelled wide_field_lateral; otherwise it falls to uncolumned_other.
    wide_field_requires_local_family: bool = True


def classify_inhibition(conn, stage_table, *, col_assign=None, geometry=None, criteria=None):
    """Label every inhibitory edge by stage relationship and lateral geometry.

    Args:
        conn: connection table (needs ``pre/post_root_id``, ``pre/post_primary_type``,
            ``syn_count``, ``neuropil``; ``sign`` added if missing).
        stage_table: :func:`src.lateral.stage.assign_stage` output.
        col_assign / geometry: provide column coordinates for Δcolumn (either the raw
            column table or a :class:`src.lateral.hexgeom.ColumnGeometry`).
        criteria: :class:`LateralInhibitionCriteria`.

    Returns:
        The inhibitory edges with added columns: stage tags (from
        :func:`src.lateral.edge_region.tag_edges`), ``delta_col``, ``pre_rank``,
        ``post_rank``, ``post_inh_frac``, ``is_disinhibitory`` and ``label``.
    """
    criteria = criteria or LateralInhibitionCriteria()
    if "sign" not in conn.columns:
        conn = add_sign(conn)

    inh = conn[(conn["sign"] == "inh") & (conn["syn_count"] >= criteria.min_syn)].copy()
    inh = tag_edges(inh, stage_table, same_stage_def=criteria.same_stage_def)

    if geometry is not None:
        P, Q, _ = geometry.pq_maps()
    elif col_assign is not None:
        P, Q, _ = pq_hemi_maps(col_assign)
    else:
        P = Q = {}
    for side in ("pre", "post"):
        inh[f"{side}_p"] = inh[f"{side}_root_id"].map(P)
        inh[f"{side}_q"] = inh[f"{side}_root_id"].map(Q)
    have_col = inh[["pre_p", "pre_q", "post_p", "post_q"]].notna().all(axis=1)
    inh["delta_col"] = np.nan
    inh.loc[have_col, "delta_col"] = hex_distance(
        inh.loc[have_col, "pre_p"] - inh.loc[have_col, "post_p"],
        inh.loc[have_col, "pre_q"] - inh.loc[have_col, "post_q"],
    )

    # A post cell is "inhibition-dominant" if *its own output* is mostly inhibitory,
    # so group output sign by the source (pre) type, then look up the post type there.
    osf_out = output_sign_fraction(conn, by="pre_primary_type")
    inh["post_inh_frac"] = inh["post_primary_type"].map(osf_out["inh_frac"]).fillna(0.0)
    inh["is_disinhibitory"] = inh["post_inh_frac"] >= criteria.inh_dominant_frac

    # When gating on the medulla sublayer, rank pre/post by their fine stage so that
    # cross-sublayer ME->ME edges resolve to feedforward / feedback; otherwise rank by
    # the coarse neuropil stage.
    if criteria.same_stage_def == "sublayer" and "pre_fine_stage" in inh.columns:
        inh["pre_rank"] = inh["pre_fine_stage"].map(SUBLAYER_RANK)
        inh["post_rank"] = inh["post_fine_stage"].map(SUBLAYER_RANK)
    else:
        inh["pre_rank"] = inh["pre_stage"].map(STAGE_RANK)
        inh["post_rank"] = inh["post_stage"].map(STAGE_RANK)

    same = inh["same_stage"].to_numpy()
    offset_ok = (inh["delta_col"] >= criteria.min_offset_cols).to_numpy()
    no_col = inh["delta_col"].isna().to_numpy()
    pre_local = inh["pre_is_local_interneuron"].to_numpy(dtype=bool)
    if criteria.wide_field_requires_local_family:
        wide_field = same & no_col & pre_local
        uncolumned_other = same & no_col & ~pre_local
    else:
        wide_field = same & no_col
        uncolumned_other = np.zeros(len(inh), dtype=bool)
    ff = (inh["pre_rank"] < inh["post_rank"]).to_numpy()
    fb = (inh["pre_rank"] > inh["post_rank"]).to_numpy()
    unknown = (inh["pre_stage"].isna() | inh["post_stage"].isna()).to_numpy()

    inh["label"] = np.select(
        [
            unknown,
            same & offset_ok,
            wide_field,
            uncolumned_other,
            same,  # same stage, has column, below offset -> co-columnar
            ff,
            fb,
        ],
        [
            "unknown",
            "direct_lateral",
            "wide_field_lateral",
            "uncolumned_other",
            "co_columnar",
            "feedforward_inhibition",
            "feedback_inhibition",
        ],
        default="cross_parallel",
    )
    return inh


def lateral_inhibition_index(classified):
    """Summarize a :func:`classify_inhibition` table into synapse-weighted fractions.

    The rigorous replacement for the headline "fraction inhibitory / how wide" metric:
    it reports how inhibition splits across same-stage lateral (direct vs wide-field)
    vs cross-stage feedforward / feedback, plus the disinhibitory share.
    """
    w = classified.groupby("label")["syn_count"].sum()
    total = float(w.sum())
    if total == 0:
        return pd.Series(dtype=float)

    def frac(label):
        return float(w.get(label, 0)) / total

    lateral = frac("direct_lateral") + frac("wide_field_lateral")
    disinh = float(classified.loc[classified["is_disinhibitory"], "syn_count"].sum()) / total
    return pd.Series(
        {
            "total_inh_syn": total,
            "frac_lateral": lateral,
            "frac_direct_lateral": frac("direct_lateral"),
            "frac_wide_field_lateral": frac("wide_field_lateral"),
            "frac_uncolumned_other": frac("uncolumned_other"),
            "frac_co_columnar": frac("co_columnar"),
            "frac_feedforward": frac("feedforward_inhibition"),
            "frac_feedback": frac("feedback_inhibition"),
            "frac_cross_parallel": frac("cross_parallel"),
            "frac_unknown": frac("unknown"),
            "frac_disinhibitory": disinh,
        }
    )
