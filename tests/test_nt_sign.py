"""Unit tests for neurotransmitter sign classification."""

import numpy as np
import pandas as pd

from src.lateral.nt_sign import SIGN_VALUE, add_sign, classify_nt


def test_classify_nt_truth_table():
    assert classify_nt("GABA") == "inh"
    assert classify_nt("GLUT") == "inh"
    assert classify_nt("HIS") == "inh"
    assert classify_nt("ACH") == "exc"
    for modulatory in ("DA", "SER", "OCT"):
        assert classify_nt(modulatory) == "other"
    assert classify_nt(None) == "other"
    assert classify_nt(np.nan) == "other"
    assert classify_nt("nonsense") == "other"


def test_sign_value_signs():
    assert SIGN_VALUE["exc"] == 1
    assert SIGN_VALUE["inh"] == -1
    assert SIGN_VALUE["other"] == 0


def test_add_sign_maps_column_without_mutating_input():
    conn = pd.DataFrame({"nt_type": ["GABA", "ACH", "SER", "HIS"]})
    out = add_sign(conn)
    assert out["sign"].tolist() == ["inh", "exc", "other", "inh"]
    assert "sign" not in conn.columns  # original untouched
