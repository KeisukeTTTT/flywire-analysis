"""Edge-level processing-stage / region tagging.

Joins the per-neuron stage table (:mod:`src.lateral.stage`) onto a connection table
and adds the boolean ``same_stage`` that the original notebooks never computed -- the
gate that separates *within-stage* lateral inhibition from *cross-stage* feedforward
/ feedback inhibition.

The connection table carries a ``neuropil`` column giving the synapse *location*
(e.g. ``ME_R``), so each edge also gets ``syn_region`` = the stage where the synapse
physically occurs. This is why ``same_stage`` can be made strict: a lamina cell
synapsing onto a medulla cell *in the medulla* (L1->Mi1) shares the synapse neuropil
but not a home stage -- it is feedforward, not lateral, and is excluded.

Two definitions are provided (both columns are always added; ``same_stage_def``
selects which one populates ``same_stage``):

* ``home`` -- pre and post share a home stage (primary lateral vs feedforward signal).
* ``syn``  -- additionally the synapse occurs in that shared stage's neuropil
  (strict; excludes e.g. T4->T4 contacts that happen in the lobula plate while both
  cells' home stage is the medulla).
"""

from __future__ import annotations

import pandas as pd

from .stage import neuropil_base

_STAGE_COLS = ("stage", "input_stage", "output_stage", "is_intrinsic")


def tag_edges(conn_df, stage_table, *, same_stage_def="home"):
    """Annotate each connection edge with pre/post stage and ``same_stage``.

    Args:
        conn_df: connection table with ``pre_root_id``, ``post_root_id`` and
            (optionally) ``neuropil`` -- e.g. ``optic_lobe_connections_df``.
        stage_table: output of :func:`src.lateral.stage.assign_stage`.
        same_stage_def: ``"home"`` (default) or ``"syn"`` -- which definition fills
            the ``same_stage`` column.

    Returns:
        Copy of ``conn_df`` with added columns: ``syn_region``, ``{pre,post}_stage``,
        ``{pre,post}_input_stage``, ``{pre,post}_output_stage``, ``{pre,post}_intrinsic``,
        ``same_stage_home``, ``same_stage_syn`` and ``same_stage``.
    """
    if same_stage_def not in ("home", "syn"):
        raise ValueError(f"unknown same_stage_def: {same_stage_def!r} (use 'home' or 'syn')")

    st = stage_table.drop_duplicates("root_id").set_index("root_id")
    out = conn_df.copy()

    out["syn_region"] = out["neuropil"].map(neuropil_base) if "neuropil" in out.columns else pd.NA

    for side in ("pre", "post"):
        ids = out[f"{side}_root_id"]
        out[f"{side}_stage"] = ids.map(st["stage"])
        out[f"{side}_input_stage"] = ids.map(st["input_stage"])
        out[f"{side}_output_stage"] = ids.map(st["output_stage"])
        out[f"{side}_intrinsic"] = ids.map(st["is_intrinsic"]).fillna(False).astype(bool)

    out["same_stage_home"] = (out["pre_stage"] == out["post_stage"]) & out["pre_stage"].notna()
    out["same_stage_syn"] = out["same_stage_home"] & (out["syn_region"] == out["pre_stage"])
    out["same_stage"] = out["same_stage_syn"] if same_stage_def == "syn" else out["same_stage_home"]
    return out
