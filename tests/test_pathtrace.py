"""Unit tests for signed path tracing and radial kernels (hand-built graphs)."""

import pandas as pd
import pytest

from src.lateral.pathtrace import (
    RadialKernels,
    SignedConnGraph,
    disinhibition_onto,
    ffi_motifs,
    net_sign_to_target,
    rms_radius,
)


def _type_graph():
    # A -exc-> B -inh-> T ; C -inh-> T ; A -exc-> T ; X -inh-> B -inh-> T2
    conn = pd.DataFrame(
        {
            "pre_primary_type": ["A", "B", "C", "A", "X", "B"],
            "post_primary_type": ["B", "T", "T", "T", "B", "T2"],
            "sign": ["exc", "inh", "inh", "exc", "inh", "inh"],
            "syn_count": [10, 5, 3, 7, 4, 6],
        }
    )
    return SignedConnGraph(conn)


def test_trace_paths_exc_inh_surround():
    g = _type_graph()
    p = g.trace_paths(["exc", "inh"], targets={"T"})
    assert list(zip(p["n0"], p["n1"], p["n2"])) == [("A", "B", "T")]
    assert float(p["w"].iloc[0]) == 50.0  # 10 * 5
    assert int(p["net_sign"].iloc[0]) == -1  # exc * inh
    assert int(p["length"].iloc[0]) == 2


def test_trace_paths_disinhibition_is_net_positive():
    g = _type_graph()
    p = g.trace_paths(["inh", "inh"], targets={"T2"})
    assert list(zip(p["n0"], p["n2"])) == [("X", "T2")]
    assert int(p["net_sign"].iloc[0]) == 1  # inh * inh
    assert float(p["w"].iloc[0]) == 24.0  # 4 * 6


def test_disinhibition_onto():
    g = _type_graph()
    d = disinhibition_onto(g, ["T2"])
    assert d.set_index("target").loc["T2", "total_w"] == 24.0


def test_net_sign_to_target_splits_by_length_and_sign():
    g = _type_graph()
    ns = net_sign_to_target(g, ["T"], max_len=2).set_index(["length", "net_sign"])
    # monosynaptic excitatory drive A->T = 7
    assert ns.loc[(1, 1), "total_w"] == 7.0
    # monosynaptic inhibition B->T + C->T = 8
    assert ns.loc[(1, -1), "total_w"] == 8.0
    # disynaptic exc->inh surround A->B->T = 50
    assert ns.loc[(2, -1), "total_w"] == 50.0


def test_ffi_motifs_finds_common_driver():
    # X drives target T (exc) and also drives the inhibitor Z (exc); Z inhibits T.
    conn = pd.DataFrame(
        {
            "pre_primary_type": ["X", "X", "Z"],
            "post_primary_type": ["T", "Z", "T"],
            "sign": ["exc", "exc", "inh"],
            "syn_count": [300, 250, 400],
        }
    )
    out = ffi_motifs(conn, "T", min_xz=200)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["driver_X"], row["inhibitor_Z"]) == ("X", "Z")
    assert row["X_to_Z_exc"] == 250 and row["Z_to_T_inh"] == 400


def _radial_data():
    # target t0 at home (0,0); inhibitory mediator j0 (no column);
    # excitatory sources e1@(0,0) and e0@(2,0) feed j0.
    col = pd.DataFrame(
        {
            "root_id": ["t0", "e0", "e1"],
            "hemisphere": "right",
            "type": ["T", "Mi1", "Mi1"],
            "column_id": ["0", "1", "2"],
            "x": 0,
            "y": 0,
            "p": [0, 2, 0],
            "q": [0, 0, 0],
        }
    )
    conn = pd.DataFrame(
        {
            "pre_root_id": ["j0", "e0", "e1"],
            "post_root_id": ["t0", "j0", "j0"],
            "pre_primary_type": ["Pm", "Mi1", "Mi1"],
            "sign": ["inh", "exc", "exc"],
            "syn_count": [10, 4, 6],
        }
    )
    return RadialKernels.from_data(conn, col)


def test_direct_kernel_misses_widefield_source():
    # the only direct inhibitor of t0 is j0, which has no column -> dropped.
    rk = _radial_data()
    direct = rk.direct_kernel("T", "inh", min_nbrs=0)
    assert direct.sum() == 0


def test_disyn_kernel_recovers_surround():
    rk = _radial_data()
    k = rk.disyn_kernel("T", min_nbrs=0).to_dict()
    # j0 pools e1 (d=0, weight 10*6/10=6) and e0 (d=2, weight 10*4/10=4)
    assert k == {0: 6.0, 2: 4.0}


def test_rms_radius():
    s = pd.Series({0: 6.0, 2: 4.0})
    sigma, norm = rms_radius(s, maxd=8)
    # variance = (6*0 + 4*4) / 10 = 1.6 -> sigma = sqrt(1.6)
    assert sigma == pytest.approx(1.6 ** 0.5)
