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
            "root_id": ["mi_c", "mi_n", "dm", "dmpost", "l1", "lo1"],
            "primary_type": ["Mi1", "Mi1", "Dm9", "Dm9", "L1", "Li1"],
            "stage": ["ME", "ME", "ME", "ME", "LA", "LO"],
            "input_stage": ["ME", "ME", "ME", "ME", "LA", "LO"],
            "output_stage": ["ME", "ME", "ME", "ME", "ME", "LO"],
            "is_intrinsic": [True, True, True, True, True, True],
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
            "pre_root_id": ["mi_n", "dm", "l1", "lo1", "dm", "mi_c", "mi_c"],
            "post_root_id": ["mi_c", "mi_c", "mi_c", "mi_c", "dmpost", "mi_n", "x"],
            "pre_primary_type": ["Mi1", "Dm9", "L1", "Li1", "Dm9", "Mi1", "Mi1"],
            "post_primary_type": ["Mi1", "Mi1", "Mi1", "Mi1", "Dm9", "Mi1", "X"],
            "neuropil": ["ME_R"] * 7,
            "nt_type": ["GABA", "GABA", "GABA", "GABA", "GABA", "GABA", "ACH"],
            "syn_count": [10, 20, 8, 9, 6, 3, 100],
        }
    )


def test_classify_inhibition_labels():
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5),
    )
    label = {(r.pre_root_id, r.post_root_id): r.label for r in cl.itertuples()}
    assert label[("mi_n", "mi_c")] == "direct_lateral"  # neighbor column, same stage
    assert label[("dm", "mi_c")] == "wide_field_lateral"  # no column -> wide-field
    assert label[("l1", "mi_c")] == "feedforward_inhibition"  # LA -> ME
    assert label[("lo1", "mi_c")] == "feedback_inhibition"  # LO -> ME
    # the 3-synapse Mi1->Mi1 edge is below min_syn and excluded
    assert ("mi_c", "mi_n") not in label


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


def test_lateral_inhibition_index_fractions_sum_sensibly():
    cl = classify_inhibition(
        _conn(), _stage_table(), col_assign=_col_assign(),
        criteria=LateralInhibitionCriteria(min_syn=5),
    )
    idx = lateral_inhibition_index(cl)
    label_fracs = [
        "frac_direct_lateral", "frac_wide_field_lateral", "frac_co_columnar",
        "frac_feedforward", "frac_feedback", "frac_cross_parallel", "frac_unknown",
    ]
    assert abs(sum(idx[k] for k in label_fracs) - 1.0) < 1e-9
    assert idx["frac_lateral"] == idx["frac_direct_lateral"] + idx["frac_wide_field_lateral"]
