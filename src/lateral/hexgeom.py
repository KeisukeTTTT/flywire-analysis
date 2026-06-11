"""Hexagonal column geometry for the FlyWire optic lobe retinotopic map.

The lamina / medulla / lobula are organized as a hexagonal lattice of ~750-800
columns (one per ommatidium). Column positions live in ``column_assignment.csv`` as
axial coordinates ``(p, q)``. The *lateral* distance between two neurons is the hex
distance between their home columns -- the horizontal / retinotopic axis along which
lateral inhibition acts (orthogonal to the medulla M1-M10 depth axis).

This module consolidates geometry helpers that were duplicated across
``notebook/lateral_inhibition.py``, ``lateral_inhibition_extended.py``,
``edge_compensation_t4t5_lateral.py`` and ``column_assignment_validation.py``.

Two distinct notions of "neighbor count" exist in the original notebooks and are
both preserved here:

* :func:`interior_cells` counts neighbors *within a single cell type's own columns*
  (as in ``lateral_inhibition_extended.py``), used to pick well-surrounded cells for
  the center-surround kernels.
* :class:`ColumnGeometry` counts neighbors / BFS boundary distance on the ``Mi1``
  reference grid (as in ``edge_compensation_t4t5_lateral.py``), used for rim/center
  edge-compensation geometry.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DATA_DIR

# Six axial-hex neighbor offsets (dp, dq).
HEXN = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# Mi1 is ~1 cell per column -> the canonical column reference grid.
REFERENCE_COLUMN_TYPE = "Mi1"
RIM_MAX_DISTANCE = 1
CENTER_MIN_DISTANCE = 3

# Inward direction: how the per-column "toward the interior" axis is defined.
#   "local"    -> local edge normal = direction to the centroid of the lattice columns
#                 actually present within LOCAL_NORMAL_RADIUS hops (points away from the
#                 missing/off-lattice side; accurate at the rim where the edge curves).
#   "centroid" -> direction to the global lattice centroid (legacy; ~45 deg off the true
#                 local edge normal at the rim on the curved medulla outline).
# Interior cells (full neighborhood -> ~zero local vector) fall back to the centroid
# direction so the inward axis stays a proper unit vector everywhere.
INWARD_METHODS = ("local", "centroid")
DEFAULT_INWARD_METHOD = "local"
LOCAL_NORMAL_RADIUS = 3
_LOCAL_NORMAL_EPS = 1e-6


def _hex_disk_offsets(radius):
    """All ``(dp, dq)`` offsets within ``radius`` hex hops of the origin (excl. origin)."""
    return [
        (dp, dq)
        for dp in range(-radius, radius + 1)
        for dq in range(-radius, radius + 1)
        if (dp, dq) != (0, 0) and max(abs(dp), abs(dq), abs(dp + dq)) <= radius
    ]


def axial_to_cart(p, q):
    """Axial hex ``(p, q)`` -> Cartesian ``(x, y)`` on a 60-degree basis."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return p + 0.5 * q, q * (np.sqrt(3) / 2.0)


def hex_distance(dp, dq, *, integer=False):
    """Axial hex distance of a displacement ``(dp, dq)``.

    Uses the cube-projection form ``(|dp| + |dq| + |dp+dq|) / 2`` which is also valid
    for non-integer (synapse-weighted centroid) displacements. For integer column
    coordinates the numerator is always even, so this equals the integer floor form
    used in ``column_assignment_validation.py``; pass ``integer=True`` to round to int
    (matching the ``np.rint(...).astype(int)`` convention used by the kernels).
    """
    dp = np.asarray(dp, dtype=float)
    dq = np.asarray(dq, dtype=float)
    d = (np.abs(dp) + np.abs(dq) + np.abs(dp + dq)) / 2.0
    if integer:
        return np.rint(d).astype(int)
    return d


def load_column_assignment(data_dir=DATA_DIR, *, dedupe=True):
    """Load ``column_assignment.csv`` -- the single canonical loader.

    Columns: ``root_id, hemisphere, type, column_id, x, y, p, q``. ``p``/``q`` are
    cast to int; ``root_id``/``column_id`` stay str. One row per neuron when ``dedupe``.
    """
    path = os.path.join(data_dir, "raw", "flywire", "csv", "column_assignment.csv")
    df = pd.read_csv(path, dtype={"root_id": str, "column_id": str})
    df["p"] = df["p"].astype(int)
    df["q"] = df["q"].astype(int)
    if dedupe:
        df = df.drop_duplicates("root_id").copy()
    return df


def pq_hemi_maps(col_assign):
    """Return ``(P, Q, HM)`` dicts ``root_id -> p / q / hemisphere`` over all columns.

    These are the per-source lookups the center-surround kernels use to locate
    inhibitory / excitatory source cells in column space.
    """
    idx = (
        col_assign.dropna(subset=["p", "q"]).drop_duplicates("root_id").set_index("root_id")
    )
    return idx["p"].to_dict(), idx["q"].to_dict(), idx["hemisphere"].to_dict()


def interior_cells(col_assign, cell_type, hemisphere, *, min_nbrs=5):
    """Cells of ``cell_type`` whose home column has ``>= min_nbrs`` neighbors among
    the *same type's* columns (faithful to ``lateral_inhibition_extended.py``).

    Returns a frame indexed by ``root_id`` with the original columns plus
    ``n_neighbors``. Used to select well-surrounded target cells so the surround of a
    radial kernel is not truncated by the lattice edge.
    """
    cells = col_assign[
        (col_assign["type"] == cell_type) & (col_assign["hemisphere"] == hemisphere)
    ].copy()
    s = set(zip(cells["p"], cells["q"]))
    cells["n_neighbors"] = [
        sum((p + dp, q + dq) in s for dp, dq in HEXN) for p, q in zip(cells["p"], cells["q"])
    ]
    cells = cells[cells["n_neighbors"] >= min_nbrs].copy()
    return cells.set_index("root_id")


def region_from_boundary_distance(distance, *, rim_max=RIM_MAX_DISTANCE, center_min=CENTER_MIN_DISTANCE):
    """Classify a column by its BFS distance from the lattice edge."""
    if pd.isna(distance):
        return "outside_reference_grid"
    if distance <= rim_max:
        return "rim"
    if distance >= center_min:
        return "center"
    return "middle"


def _build_column_geometry(
    reference_assignment,
    *,
    rim_max=RIM_MAX_DISTANCE,
    center_min=CENTER_MIN_DISTANCE,
    inward_method=DEFAULT_INWARD_METHOD,
    local_radius=LOCAL_NORMAL_RADIUS,
):
    """Per-hemisphere lattice geometry (faithful to ``build_column_geometry``).

    Computes both inward conventions per column -- ``inward_centroid_{x,y}`` (toward the
    global lattice centroid) and ``inward_local_{x,y}`` (local edge normal: toward the
    centroid of columns present within ``local_radius`` hops, falling back to the global
    centroid where the neighborhood is full) -- and exposes the selected one as the
    active ``inward_unit_{x,y}`` (see ``inward_method`` / :data:`INWARD_METHODS`).
    """
    if inward_method not in INWARD_METHODS:
        raise ValueError(f"inward_method must be one of {INWARD_METHODS}, got {inward_method!r}")
    disk = _hex_disk_offsets(local_radius)
    rows = []
    for hemi, hemi_df in reference_assignment.groupby("hemisphere"):
        coords = set(zip(hemi_df["p"], hemi_df["q"]))
        n_neighbors = {
            coord: sum((coord[0] + dp, coord[1] + dq) in coords for dp, dq in HEXN)
            for coord in coords
        }
        boundary = [coord for coord, count in n_neighbors.items() if count < 6]
        distance = {coord: 0 for coord in boundary}
        queue = deque(boundary)
        while queue:
            p, q = queue.popleft()
            for dp, dq in HEXN:
                neighbor = (p + dp, q + dq)
                if neighbor in coords and neighbor not in distance:
                    distance[neighbor] = distance[(p, q)] + 1
                    queue.append(neighbor)

        ordered = sorted(coords)
        all_p = np.asarray([c[0] for c in ordered])
        all_q = np.asarray([c[1] for c in ordered])
        all_x, all_y = axial_to_cart(all_p, all_q)
        pos = {c: (float(x), float(y)) for c, x, y in zip(ordered, all_x, all_y)}
        center_x, center_y = float(all_x.mean()), float(all_y.mean())
        for p, q in sorted(coords):
            x, y = pos[(p, q)]
            # centroid-based inward (legacy)
            cvx, cvy = center_x - x, center_y - y
            cnorm = float(np.hypot(cvx, cvy))
            cux, cuy = (cvx / cnorm, cvy / cnorm) if cnorm else (0.0, 0.0)
            # local edge normal: mean direction toward present neighbors within local_radius
            lvx = lvy = 0.0
            for dp, dq in disk:
                nb = (p + dp, q + dq)
                if nb in coords:
                    nx, ny = pos[nb]
                    lvx += nx - x
                    lvy += ny - y
            lnorm = float(np.hypot(lvx, lvy))
            if lnorm > _LOCAL_NORMAL_EPS:
                lux, luy = lvx / lnorm, lvy / lnorm
            else:  # full neighborhood (deep interior) -> direction is arbitrary; fall back
                lux, luy = cux, cuy
            active = (lux, luy) if inward_method == "local" else (cux, cuy)
            rows.append(
                {
                    "hemisphere": hemi,
                    "p": p,
                    "q": q,
                    "x": x,
                    "y": y,
                    "n_neighbors": n_neighbors[(p, q)],
                    "boundary_distance": distance[(p, q)],
                    "region": region_from_boundary_distance(
                        distance[(p, q)], rim_max=rim_max, center_min=center_min
                    ),
                    "inward_centroid_x": cux,
                    "inward_centroid_y": cuy,
                    "inward_local_x": lux,
                    "inward_local_y": luy,
                    "inward_unit_x": active[0],
                    "inward_unit_y": active[1],
                }
            )
    return pd.DataFrame(rows)


@dataclass
class ColumnGeometry:
    """Hex-lattice geometry built from a reference column grid (default ``Mi1``).

    ``columns`` has one row per ``(hemisphere, p, q)`` with cartesian ``(x, y)``,
    ``n_neighbors``, ``boundary_distance`` (BFS hops from the lattice edge),
    ``region`` (rim/middle/center) and the inward unit vector. Two inward conventions
    are always present -- ``inward_local_{x,y}`` (local edge normal, default) and
    ``inward_centroid_{x,y}`` (global centroid, legacy) -- with the selected
    ``inward_method`` mirrored into the active ``inward_unit_{x,y}``. ``cells`` is every
    column-assigned neuron joined to that geometry.
    """

    columns: pd.DataFrame
    cells: pd.DataFrame
    reference_type: str = REFERENCE_COLUMN_TYPE
    inward_method: str = DEFAULT_INWARD_METHOD

    @classmethod
    def from_assignment(
        cls,
        col_assign=None,
        *,
        reference_type=REFERENCE_COLUMN_TYPE,
        rim_max=RIM_MAX_DISTANCE,
        center_min=CENTER_MIN_DISTANCE,
        inward_method=DEFAULT_INWARD_METHOD,
        data_dir=DATA_DIR,
    ):
        if col_assign is None:
            col_assign = load_column_assignment(data_dir)
        reference = col_assign[col_assign["type"] == reference_type]
        columns = _build_column_geometry(reference, rim_max=rim_max, center_min=center_min,
                                         inward_method=inward_method)
        cells = col_assign.merge(
            columns, on=["hemisphere", "p", "q"], how="left", validate="many_to_one"
        )
        cells["region"] = cells["boundary_distance"].map(
            lambda d: region_from_boundary_distance(d, rim_max=rim_max, center_min=center_min)
        )
        cells["in_reference_grid"] = cells["boundary_distance"].notna()
        return cls(columns=columns, cells=cells, reference_type=reference_type,
                   inward_method=inward_method)

    def pq_maps(self):
        """``(P, Q, HM)`` dicts ``root_id -> p / q / hemisphere``."""
        return pq_hemi_maps(self.cells)

    def region_of(self, p, q, hemisphere):
        m = self.columns[
            (self.columns["hemisphere"] == hemisphere)
            & (self.columns["p"] == p)
            & (self.columns["q"] == q)
        ]
        return m["region"].iloc[0] if len(m) else "outside_reference_grid"
