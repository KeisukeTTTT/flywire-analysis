"""Unit tests for stage / layer assignment (synthetic data)."""

import numpy as np
import pandas as pd

from src.lateral.stage import (
    FAMILY_TO_STAGE,
    MEDULLA_SUBLAYERS,
    OPTIC_STAGES,
    assign_stage,
    dominant_neuropils,
    medulla_sublayer_from_rel_depth,
    neuropil_base,
    neuropil_to_stage,
)


def test_neuropil_base_strips_only_hemisphere_suffix():
    assert neuropil_base("ME_R") == "ME"
    assert neuropil_base("LO_R") == "LO"
    assert neuropil_base("LOP_L") == "LOP"  # LO is NOT a false match
    assert neuropil_base("AME_R") == "AME"
    assert neuropil_base("FB") == "FB"


def test_neuropil_to_stage():
    assert neuropil_to_stage("ME_R") == "ME"
    assert neuropil_to_stage("LOP_R") == "LOP"
    assert neuropil_to_stage("LO_L") == "LO"
    assert neuropil_to_stage("SMP_L") == "central"  # central brain
    assert pd.isna(neuropil_to_stage(np.nan))


def test_family_to_stage_curated_values_are_valid_stages():
    for fam, stage in FAMILY_TO_STAGE.items():
        assert stage is None or stage in OPTIC_STAGES, (fam, stage)
    # a few literature anchors
    assert FAMILY_TO_STAGE["Lamina Monopolar"] == "LA"
    assert FAMILY_TO_STAGE["Distal Medulla"] == "ME"
    assert FAMILY_TO_STAGE["Photo Receptors"] == "RETINA"
    # span families are intentionally deferred to the neuropil table
    assert FAMILY_TO_STAGE["T4 Neuron"] is None
    assert FAMILY_TO_STAGE["Transmedullary"] is None


def test_dominant_neuropils_excludes_partner_columns():
    ntab = pd.DataFrame(
        {
            "root_id": ["a"],
            "input synapses in ME_R": [80],
            "input synapses in LA_R": [50],
            "input partners in LA_R": [9999],  # must be ignored
            "output synapses in LOP_R": [90],
            "output synapses in ME_R": [10],
        }
    )
    dom = dominant_neuropils(ntab).set_index("root_id")
    assert dom.loc["a", "dominant_in_np"] == "ME_R"
    assert dom.loc["a", "dominant_out_np"] == "LOP_R"
    assert dom.loc["a", "n_in_syn"] == 130


def _synthetic_inputs():
    neurons = pd.DataFrame(
        {
            "root_id": ["n1", "n2", "n3"],
            "flow": ["intrinsic", "intrinsic", "afferent"],
            "primary_type": ["L1", "T4a", "R1-6"],
        }
    )
    vnt = pd.DataFrame(
        {
            "root_id": ["n1", "n2", "n3"],
            "family": ["Lamina Monopolar", "T4 Neuron", "Photo Receptors"],
        }
    )
    ntab = pd.DataFrame(
        {
            "root_id": ["n1", "n2", "n3"],
            "input synapses in LA_R": [50, 0, 0],
            "input synapses in ME_R": [80, 100, 0],
            "output synapses in ME_R": [70, 0, 0],
            "output synapses in LOP_R": [0, 90, 0],
            "output synapses in LA_R": [0, 0, 30],
        }
    )
    return neurons, vnt, ntab


def test_assign_stage_family_primary_and_intrinsic_flag():
    neurons, vnt, ntab = _synthetic_inputs()
    st = assign_stage(neurons, visual_types_df=vnt, neuropil_table_df=ntab).set_index("root_id")
    # L1: family-curated LA wins for the home stage, even though dominant input is ME
    assert st.loc["n1", "stage"] == "LA"
    assert st.loc["n1", "stage_source"] == "family"
    assert st.loc["n1", "input_stage"] == "ME"
    assert st.loc["n1", "stage_confidence"] == "mismatch"
    assert bool(st.loc["n1", "is_intrinsic"]) is True


def test_assign_stage_span_type_uses_neuropil_input_output():
    neurons, vnt, ntab = _synthetic_inputs()
    st = assign_stage(neurons, visual_types_df=vnt, neuropil_table_df=ntab).set_index("root_id")
    # T4 family is deferred -> stage falls back to dominant input neuropil (ME),
    # and output_stage is LOP
    assert st.loc["n2", "stage"] == "ME"
    assert st.loc["n2", "stage_source"] == "neuropil_in"
    assert st.loc["n2", "input_stage"] == "ME"
    assert st.loc["n2", "output_stage"] == "LOP"


def test_medulla_sublayer_from_rel_depth_bins():
    assert medulla_sublayer_from_rel_depth(0.0) == "ME:distal"
    assert medulla_sublayer_from_rel_depth(0.2) == "ME:distal"
    assert medulla_sublayer_from_rel_depth(0.5) == "ME:medial"
    assert medulla_sublayer_from_rel_depth(0.9) == "ME:proximal"
    # boundaries are left-closed: 1/3 -> medial, 2/3 -> proximal
    assert medulla_sublayer_from_rel_depth(1.0 / 3.0) == "ME:medial"
    assert medulla_sublayer_from_rel_depth(2.0 / 3.0) == "ME:proximal"
    assert pd.isna(medulla_sublayer_from_rel_depth(np.nan))
    assert set(MEDULLA_SUBLAYERS) == {"ME:distal", "ME:medial", "ME:proximal"}


def test_assign_stage_photoreceptor_and_afferent():
    neurons, vnt, ntab = _synthetic_inputs()
    st = assign_stage(neurons, visual_types_df=vnt, neuropil_table_df=ntab).set_index("root_id")
    assert st.loc["n3", "stage"] == "RETINA"
    assert bool(st.loc["n3", "is_intrinsic"]) is False
