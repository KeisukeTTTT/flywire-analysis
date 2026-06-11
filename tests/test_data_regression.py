"""Tier B: RadialKernels reproduce the original extended.py kernels (golden).

The golden in ``tests/fixtures/extended_golden.json`` was captured after verifying
(``pd.testing.assert_series_equal``) that :class:`RadialKernels` exactly reproduces
the original ``direct_kernel`` / ``disyn_inh_kernel`` on real data. This test guards
against future regressions in the refactored kernels.
"""

import json
from pathlib import Path

import pytest

from src.config import DATA_DIR
from src.lateral.hexgeom import load_column_assignment
from src.lateral.nt_sign import add_sign
from src.lateral.pathtrace import RadialKernels, rms_radius

pytestmark = pytest.mark.data

_GOLDEN = Path(__file__).parent / "fixtures" / "extended_golden.json"


@pytest.fixture(scope="module")
def kernels(flywire_manager):
    conn = add_sign(flywire_manager.optic_lobe_connections_df.copy())
    return RadialKernels.from_data(conn, load_column_assignment(DATA_DIR))


def test_a1_table_matches_golden(kernels):
    golden = {row["target"]: row for row in json.loads(_GOLDEN.read_text())["a1_table"]}
    for target, exp in golden.items():
        e = kernels.direct_kernel(target, "exc")
        i = kernels.disyn_kernel(target)
        sig_c, _ = rms_radius(e, 8)
        sig_s, _ = rms_radius(i, 8)
        assert sig_c == pytest.approx(exp["exc_rms_cols"], abs=1e-3), target
        assert sig_s == pytest.approx(exp["inh_surround_rms_cols"], abs=1e-3), target
        ratio = sig_s / max(sig_c, 1e-6)
        assert ratio == pytest.approx(exp["surround_center_ratio"], abs=1e-3), target
        # the disynaptic inhibitory surround is broader than the excitatory center
        assert ratio > 1.0, target
