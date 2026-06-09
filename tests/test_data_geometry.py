"""Tier B: consolidated geometry reproduces the validated column lattice (real data)."""

import pytest

from src.lateral.hexgeom import ColumnGeometry, load_column_assignment

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def geometry(flywire_csv_dir):
    return ColumnGeometry.from_assignment(load_column_assignment())


def test_reference_grid_is_hexagonal(geometry):
    # column_assignment_validation.ipynb established a near-perfect hex lattice:
    # the modal/median interior column has exactly 6 neighbors.
    assert geometry.columns["n_neighbors"].median() == 6
    assert (geometry.columns["n_neighbors"] <= 6).all()


def test_both_hemispheres_present(geometry):
    assert set(geometry.columns["hemisphere"]) >= {"left", "right"}


def test_regions_partition_the_grid(geometry):
    regions = set(geometry.columns["region"])
    assert regions <= {"rim", "middle", "center"}
    # a real grid has all three
    assert {"rim", "center"} <= regions
