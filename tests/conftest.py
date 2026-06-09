"""Shared pytest fixtures and the ``data`` marker auto-skip.

Tests marked ``@pytest.mark.data`` need the real FlyWire CSVs under
``DROSOPHILA_DATA_DIR``; they are skipped automatically when that data is absent so
the fast unit tier always runs (e.g. in CI).
"""

import os

import pytest

from src.config import DATA_DIR

_FLYWIRE_CSV = os.path.join(DATA_DIR, "raw", "flywire", "csv")


def has_flywire_data():
    return os.path.exists(os.path.join(_FLYWIRE_CSV, "column_assignment.csv"))


def pytest_collection_modifyitems(config, items):
    if has_flywire_data():
        return
    skip = pytest.mark.skip(reason="real FlyWire data not available (set DROSOPHILA_DATA_DIR)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def flywire_csv_dir():
    if not has_flywire_data():
        pytest.skip("real FlyWire data not available")
    return _FLYWIRE_CSV


@pytest.fixture(scope="session")
def flywire_manager():
    """Load the full FlyWireDataManager once (heavy: ~1 GB of connections)."""
    if not has_flywire_data():
        pytest.skip("real FlyWire data not available")
    from src.data import FlyWireDataManager

    return FlyWireDataManager()
