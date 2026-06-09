"""Tier B: stage assignment vs known optic-lobe biology (real data)."""

import os

import pandas as pd
import pytest

from src.lateral.stage import assign_stage

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def stage_table(flywire_csv_dir):
    cls = pd.read_csv(os.path.join(flywire_csv_dir, "classification.csv"), dtype={"root_id": str})[
        ["root_id", "flow"]
    ]
    pt = pd.read_csv(
        os.path.join(flywire_csv_dir, "consolidated_cell_types.csv"), dtype={"root_id": str}
    )[["root_id", "primary_type"]]
    neurons = cls.merge(pt, on="root_id", how="left")
    vnt = pd.read_csv(os.path.join(flywire_csv_dir, "visual_neuron_types.csv"), dtype={"root_id": str})
    ntab = pd.read_csv(
        os.path.join(flywire_csv_dir, "neuropil_synapse_table.csv"), dtype={"root_id": str}
    )
    st = assign_stage(neurons, visual_types_df=vnt, neuropil_table_df=ntab)
    return st.merge(vnt[["root_id", "type"]], on="root_id", how="left")


def _frac(st, type_mask, col, expect):
    sub = st[type_mask]
    assert len(sub) > 50, "anchor group unexpectedly small"
    return (sub[col] == expect).mean()


def test_lamina_monopolar_in_lamina(stage_table):
    st = stage_table
    assert _frac(st, st["type"].isin(["L1", "L2", "L3", "L4", "L5"]), "stage", "LA") >= 0.99


def test_medulla_intrinsic_families_in_medulla(stage_table):
    st = stage_table
    for prefix in ("Dm", "Pm", "Mi"):
        assert _frac(st, st["type"].astype(str).str.startswith(prefix), "stage", "ME") >= 0.99


def test_t4_spans_medulla_to_lobula_plate(stage_table):
    st = stage_table
    t4 = st["type"].astype(str).str.startswith("T4")
    assert _frac(st, t4, "input_stage", "ME") >= 0.95
    assert _frac(st, t4, "output_stage", "LOP") >= 0.95


def test_t5_input_is_lobula(stage_table):
    st = stage_table
    t5 = st["type"].astype(str).str.startswith("T5")
    assert _frac(st, t5, "input_stage", "LO") >= 0.95
    assert _frac(st, t5, "output_stage", "LOP") >= 0.95


def test_photoreceptors_are_retina(stage_table):
    st = stage_table
    assert _frac(st, st["type"].astype(str).str.startswith("R"), "stage", "RETINA") >= 0.99
