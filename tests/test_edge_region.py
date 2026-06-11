"""Unit tests for edge-level stage / same_stage tagging."""

import pandas as pd

from src.lateral.edge_region import tag_edges


def _stage_table():
    return pd.DataFrame(
        {
            "root_id": ["mi_c", "mi_n", "l1", "t4a", "t4b"],
            "stage": ["ME", "ME", "LA", "ME", "ME"],
            "input_stage": ["ME", "ME", "LA", "ME", "ME"],
            "output_stage": ["ME", "ME", "ME", "LOP", "LOP"],
            "is_intrinsic": [True, True, True, True, True],
        }
    )


def _conn():
    return pd.DataFrame(
        {
            "pre_root_id": ["mi_n", "l1", "t4a"],
            "post_root_id": ["mi_c", "mi_c", "t4b"],
            "neuropil": ["ME_R", "ME_R", "LOP_R"],
            "syn_count": [10, 8, 5],
        }
    )


def test_same_stage_home_distinguishes_lateral_from_feedforward():
    out = tag_edges(_conn(), _stage_table(), same_stage_def="home").set_index(
        ["pre_root_id", "post_root_id"]
    )
    # Mi1 -> Mi1 within the medulla: same home stage
    assert bool(out.loc[("mi_n", "mi_c"), "same_stage_home"]) is True
    # L1 (lamina) -> Mi1 (medulla), synapse in ME: NOT same home stage -> feedforward
    assert bool(out.loc[("l1", "mi_c"), "same_stage_home"]) is False
    assert out.loc[("l1", "mi_c"), "syn_region"] == "ME"


def test_same_stage_syn_is_stricter_for_span_cells():
    out = tag_edges(_conn(), _stage_table(), same_stage_def="syn").set_index(
        ["pre_root_id", "post_root_id"]
    )
    # T4 -> T4 share a home stage (ME) but the synapse is in the lobula plate (LOP),
    # so the strict syn-definition excludes it while the home-definition keeps it.
    row = out.loc[("t4a", "t4b")]
    assert bool(row["same_stage_home"]) is True
    assert bool(row["same_stage_syn"]) is False
    assert bool(row["same_stage"]) is False  # syn def selected
    assert row["syn_region"] == "LOP"


def test_intrinsic_flags_present():
    out = tag_edges(_conn(), _stage_table())
    assert {"pre_intrinsic", "post_intrinsic"} <= set(out.columns)
    assert out["pre_intrinsic"].all()
