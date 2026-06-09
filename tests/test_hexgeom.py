"""Unit tests for hex / column geometry."""

import numpy as np
import pandas as pd
import pytest

from src.lateral.hexgeom import (
    HEXN,
    ColumnGeometry,
    axial_to_cart,
    hex_distance,
    interior_cells,
)


def test_hex_distance_identities():
    assert hex_distance(0, 0) == 0.0
    # all six unit neighbors are at distance 1
    assert all(hex_distance(dp, dq) == 1.0 for dp, dq in HEXN)
    assert hex_distance(2, 0) == 2.0
    assert hex_distance(0, 3) == 3.0
    assert hex_distance(3, -3) == 3.0  # (3 + 3 + 0) / 2


def test_hex_distance_symmetry():
    for dp, dq in [(1, 0), (2, -1), (3, 2), (-4, 1)]:
        assert hex_distance(dp, dq) == hex_distance(-dp, -dq)


def test_hex_distance_float_vs_integer_agree_on_integers():
    pts = [(0, 0), (1, 0), (2, -1), (3, 2), (-4, 1)]
    for dp, dq in pts:
        assert hex_distance(dp, dq, integer=True) == int(hex_distance(dp, dq))


def test_hex_distance_handles_non_integer_centroids():
    # centroid displacements must not crash and round sanely
    assert hex_distance(0.5, 0.5) == 1.0
    assert hex_distance(0.5, 0.5, integer=True) == 1


def test_axial_to_cart_basics():
    x, y = axial_to_cart(0, 0)
    assert (float(x), float(y)) == (0.0, 0.0)
    x, y = axial_to_cart(1, 0)
    assert (float(x), float(y)) == (1.0, 0.0)
    # q axis is sheared by 60 degrees
    x, y = axial_to_cart(0, 1)
    assert float(x) == pytest.approx(0.5)
    assert float(y) == pytest.approx(np.sqrt(3) / 2)


def _flower():
    """Center column (0,0) + its 6 hex neighbors, all of type Mi1, right hemi."""
    coords = [(0, 0)] + [(dp, dq) for dp, dq in HEXN]
    return pd.DataFrame(
        {
            "root_id": [f"c{i}" for i in range(len(coords))],
            "hemisphere": "right",
            "type": "Mi1",
            "column_id": [str(i) for i in range(len(coords))],
            "x": 0,
            "y": 0,
            "p": [p for p, _ in coords],
            "q": [q for _, q in coords],
        }
    )


def test_column_geometry_flower_neighbors_and_distance():
    geo = ColumnGeometry.from_assignment(_flower(), reference_type="Mi1")
    cols = geo.columns.set_index(["p", "q"])
    # the center is surrounded by all 6 neighbors
    assert cols.loc[(0, 0), "n_neighbors"] == 6
    # the 6 outer cells sit on the boundary (distance 0), center is one hop in
    assert cols.loc[(0, 0), "boundary_distance"] == 1
    assert cols.loc[(1, 0), "boundary_distance"] == 0
    assert (geo.columns["n_neighbors"] <= 6).all()


def test_interior_cells_neighbor_threshold():
    fl = _flower()
    only_center = interior_cells(fl, "Mi1", "right", min_nbrs=5)
    assert list(only_center.index) == ["c0"]  # center has 6 same-type neighbors
    everyone = interior_cells(fl, "Mi1", "right", min_nbrs=0)
    assert len(everyone) == 7
