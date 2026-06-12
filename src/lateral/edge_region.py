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

Three definitions are provided (all columns are always added; ``same_stage_def``
selects which one populates ``same_stage``):

* ``home``     -- pre and post share a home stage (primary lateral vs feedforward signal).
* ``syn``      -- additionally the synapse occurs in that shared stage's neuropil
  (strict; excludes e.g. T4->T4 contacts that happen in the lobula plate while both
  cells' home stage is the medulla).
* ``sublayer`` -- pre and post share a *fine* stage: the medulla M1-M10 sublayer
  (``fine_stage``; see :func:`src.lateral.stage.attach_medulla_sublayer`) when both are
  ME, else the coarse home stage. This separates within-ME sub-laminar feedforward /
  feedback (e.g. distal Dm -> proximal Pm) that the neuropil-level ``home`` gate cannot.
  Requires the stage table to carry a ``fine_stage`` column; without it this falls back
  to the home definition.
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
        ``{pre,post}_is_local_interneuron``, ``{pre,post}_fine_stage``,
        ``same_stage_home``, ``same_stage_syn``, ``same_stage_sublayer`` and
        ``same_stage``.
    """
    if same_stage_def not in ("home", "syn", "sublayer"):
        raise ValueError(
            f"unknown same_stage_def: {same_stage_def!r} (use 'home', 'syn' or 'sublayer')"
        )

    st = stage_table.drop_duplicates("root_id").set_index("root_id")
    out = conn_df.copy()

    out["syn_region"] = out["neuropil"].map(neuropil_base) if "neuropil" in out.columns else pd.NA

    has_local_family = "is_local_interneuron_family" in st.columns
    has_fine_stage = "fine_stage" in st.columns
    for side in ("pre", "post"):
        ids = out[f"{side}_root_id"]
        out[f"{side}_stage"] = ids.map(st["stage"])
        out[f"{side}_input_stage"] = ids.map(st["input_stage"])
        out[f"{side}_output_stage"] = ids.map(st["output_stage"])
        out[f"{side}_intrinsic"] = ids.map(st["is_intrinsic"]).fillna(False).astype(bool)
        # Local-interneuron / amacrine family membership (the canonical wide-field
        # lateral mediators). Used to ground the wide_field_lateral label on
        # morphology rather than on a merely-missing column coordinate. Defaults to
        # False when the stage table predates the flag or the neuron is absent.
        out[f"{side}_is_local_interneuron"] = (
            ids.map(st["is_local_interneuron_family"]).fillna(False).astype(bool)
            if has_local_family
            else False
        )
        # Fine stage = medulla sublayer when present, else the coarse home stage.
        out[f"{side}_fine_stage"] = (
            ids.map(st["fine_stage"]) if has_fine_stage else out[f"{side}_stage"]
        )

    out["same_stage_home"] = (out["pre_stage"] == out["post_stage"]) & out["pre_stage"].notna()
    out["same_stage_syn"] = out["same_stage_home"] & (out["syn_region"] == out["pre_stage"])
    out["same_stage_sublayer"] = (
        out["pre_fine_stage"] == out["post_fine_stage"]
    ) & out["pre_fine_stage"].notna()
    out["same_stage"] = {
        "syn": out["same_stage_syn"],
        "sublayer": out["same_stage_sublayer"],
    }.get(same_stage_def, out["same_stage_home"])
    return out
