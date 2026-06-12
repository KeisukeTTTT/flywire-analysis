"""Unit tests for the inhibition classifier (synthetic data)."""

import pandas as pd

from src.lateral.classifier import (
    LateralInhibitionCriteria,
    classify_inhibition,
    lateral_inhibition_index,
    output_sign_fraction,
)


def _stage_table():
    return pd.DataFrame(
        {
            "root_id": ["mi_c", "mi_n", "dm", "dmpost", "l1", "lo1", "tm"],
            "primary_type": ["Mi1", "Mi1", "Dm9", "Dm9", "L1", "Li1", "Tm9"],
            "stage": ["ME", "ME", "ME", "ME", "LA", "LO", "ME"],
            "input_stage": ["ME", "ME", "ME", "ME", "LA", "LO", "ME"],
            "output_stage": ["ME", "ME", "ME", "ME", "ME", "LO", "ME"],
            "is_intrinsic": [True, True, True, True, True, True, True],
            # Dm/Mi/Li are local-interneuron / amacrine families (wide-field
            # mediators); L1 (monopolar) and Tm9 (transmedullary) are not.
            "is_local_interneuron_family": [True, True, True, True, False, True, False],
        }
    )


def _col_assign():
    return pd.DataFrame(
        {
            "root_id": ["mi_c", "mi_n", "l1"],
            "hemisphere": "right",
            "type": ["Mi1", "Mi1", "L1"],
            "column_id": ["0", "1", "0"],
            "x": 0,
            "y": 0,
            "p": [0, 1, 0],
            "q": [0, 0, 0],
        }
    )


def _conn():
    # inhibitory edges of interest + excitatory output for Mi1 so it is not
    # falsely flagged inhibition-dominant.
    return pd.DataFrame(
        {
            "pre_root_id": ["mi_n", "dm", "l1", "lo1", "dm", "mi_c", "mi_c", "tm"],
            "post_root_id": ["mi_c", "mi_c", "mi_c", "mi_c", "dmpost", "mi_n", "x", "mi_c"],
            "pre_primary_type": ["Mi1", "Dm9", "L1", "Li1", "Dm9", "Mi1", "Mi1", "Tm9"],
            "post_primary_type": ["Mi1", "Mi1", "Mi1", "Mi1", "Dm9", "Mi1", "X", "Mi1"],
            "neuropil": ["ME_R"] * 8,
            "nt_type": ["GABA", "GABA", "GABA", "GABA", "GABA", "GABA", "ACH", "GABA"],
            "syn_count": [10, 20, 8, 9, 6, 3, 100, 7],
        }
    )


def test_classify_inhibition_labels():
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5),
    )
    label = {(r.pre_root_id, r.post_root_id): r.label for r in cl.itertuples()}
    assert label[("mi_n", "mi_c")] == "direct_lateral"  # neighbor column, same stage
    # Dm9 has no column but IS a local-interneuron family -> genuine wide-field
    assert label[("dm", "mi_c")] == "wide_field_lateral"
    # Tm9 has no column and is NOT a local-interneuron family -> uncolumned_other,
    # not silently counted as a wide-field mediator
    assert label[("tm", "mi_c")] == "uncolumned_other"
    assert label[("l1", "mi_c")] == "feedforward_inhibition"  # LA -> ME
    assert label[("lo1", "mi_c")] == "feedback_inhibition"  # LO -> ME
    # the 3-synapse Mi1->Mi1 edge is below min_syn and excluded
    assert ("mi_c", "mi_n") not in label


def test_wide_field_family_gate_can_be_disabled():
    # With the family gate off, the uncolumned Tm9 source falls back to the legacy
    # "no column -> wide_field_lateral" behaviour.
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5, wide_field_requires_local_family=False),
    )
    label = {(r.pre_root_id, r.post_root_id): r.label for r in cl.itertuples()}
    assert label[("tm", "mi_c")] == "wide_field_lateral"
    assert "uncolumned_other" not in set(label.values())


def test_disinhibition_flag_uses_output_sign():
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5),
    ).set_index(["pre_root_id", "post_root_id"])
    # dmpost is a Dm9 whose only output here is inhibitory -> disinhibitory target
    assert bool(cl.loc[("dm", "dmpost"), "is_disinhibitory"]) is True
    # mi_c (Mi1) has a strong excitatory output -> not a disinhibition target
    assert bool(cl.loc[("mi_n", "mi_c"), "is_disinhibitory"]) is False


def test_output_sign_fraction():
    osf = output_sign_fraction(_conn(), by="pre_primary_type")
    # Mi1 outputs: 10+30 inh + 100 exc -> mostly excitatory
    assert osf.loc["Mi1", "exc_frac"] > 0.5
    # Dm9 outputs: all inhibitory
    assert osf.loc["Dm9", "inh_frac"] == 1.0


def _sublayer_stage_table():
    return pd.DataFrame(
        {
            "root_id": ["a_dist", "b_dist", "c_prox"],
            "primary_type": ["Dm9", "Dm9", "Pm04"],
            "stage": ["ME", "ME", "ME"],
            "input_stage": ["ME", "ME", "ME"],
            "output_stage": ["ME", "ME", "ME"],
            "is_intrinsic": [True, True, True],
            "is_local_interneuron_family": [True, True, True],
            "fine_stage": ["ME:distal", "ME:distal", "ME:proximal"],
        }
    )


def _sublayer_conn():
    return pd.DataFrame(
        {
            "pre_root_id": ["a_dist", "a_dist", "c_prox"],
            "post_root_id": ["b_dist", "c_prox", "a_dist"],
            "pre_primary_type": ["Dm9", "Dm9", "Pm04"],
            "post_primary_type": ["Dm9", "Pm04", "Dm9"],
            "neuropil": ["ME_R"] * 3,
            "nt_type": ["GABA"] * 3,
            "syn_count": [10, 10, 10],
        }
    )


def test_sublayer_gate_reclassifies_cross_sublayer_me_as_ff_fb():
    crit = LateralInhibitionCriteria(min_syn=5, same_stage_def="sublayer")
    cl = classify_inhibition(_sublayer_conn(), _sublayer_stage_table(), criteria=crit)
    label = {(r.pre_root_id, r.post_root_id): r.label for r in cl.itertuples()}
    # distal -> distal stays same-stage lateral (no column coords -> wide_field)
    assert label[("a_dist", "b_dist")] == "wide_field_lateral"
    # distal -> proximal: lower rank -> higher rank -> feedforward
    assert label[("a_dist", "c_prox")] == "feedforward_inhibition"
    # proximal -> distal: higher rank -> lower rank -> feedback
    assert label[("c_prox", "a_dist")] == "feedback_inhibition"


def test_home_gate_lumps_cross_sublayer_me_as_lateral():
    # with the coarse home gate the same distal->proximal edge is NOT split out
    crit = LateralInhibitionCriteria(min_syn=5, same_stage_def="home")
    cl = classify_inhibition(_sublayer_conn(), _sublayer_stage_table(), criteria=crit)
    label = {(r.pre_root_id, r.post_root_id): r.label for r in cl.itertuples()}
    assert label[("a_dist", "c_prox")] == "wide_field_lateral"


def test_lateral_inhibition_index_fractions_sum_sensibly():
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5),
    )
    idx = lateral_inhibition_index(cl)
    label_fracs = [
        "frac_direct_lateral", "frac_wide_field_lateral", "frac_uncolumned_other",
        "frac_co_columnar", "frac_feedforward", "frac_feedback",
        "frac_cross_parallel", "frac_unknown",
    ]
    assert abs(sum(idx[k] for k in label_fracs) - 1.0) < 1e-9
    assert idx["frac_lateral"] == idx["frac_direct_lateral"] + idx["frac_wide_field_lateral"]
