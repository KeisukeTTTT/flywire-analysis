"""Per-cell (column x type) lateral-inhibition footprints -- one figure per cell.

For each individual cell (L1_1, L1_2, ... one column of one type), draw where in
retinotopic column space that single cell's drive comes from. Three lattice panels,
centred on the cell's home column (cyan star), full lattice in grey for edge context:

* feedforward excitation -- columnar excitatory sources (same-or-earlier stage).
* direct lateral inhibition -- ``direct_lateral`` columnar inhibitory sources.
* mediated lateral surround -- disynaptic exc->inh(J)->cell: the columns this cell's
  inhibitory partners J pool excitation from, weighted by w(J->cell)*norm w(src->J)
  (how wide-field / amacrine inhibition becomes spatially localizable; 2-D version of
  :meth:`src.lateral.RadialKernels.disyn_kernel`).

Cells are selected to span the rim->centre boundary-distance range, capped by
``--max-per-type``. Caveats: ``nt_type`` sign is mostly ML-predicted; synapse count is
a structural proxy.

Run from the project root::

    uv run python scripts/lateral_inhibition/generate_lateral_percell_footprints.py --types L1,Mi1 --max-per-type 6
    uv run python scripts/lateral_inhibition/generate_lateral_percell_footprints.py --types Mi1 --max-per-type 0  # every Mi1 cell

Output: outputs/lateral_inhibition/lateral_per_cell/<type>/<type>_<idx>_p<p>_q<q>_<root_id>.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "src").is_dir() and (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless batch rendering
import matplotlib.pyplot as plt

from src.lateral import (
    STAGE_RANK,
    ColumnGeometry,
    LateralInhibitionCriteria,
    add_sign,
    assign_stage_from_manager,
    axial_to_cart,
    classify_inhibition,
    load_column_assignment,
    pq_hemi_maps,
)
from src.config import DATA_DIR
from src.data import FlyWireDataManager

OUT_ROOT = REPO_ROOT / "outputs" / "lateral_inhibition" / "lateral_per_cell"
EXCLUDED = {"R7", "R8", "R1-6"}
WINDOW = 11.0           # zoom half-width (cart units ~ columns) around the home column


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", default=None,
                    help="comma-separated receiver types (default: all columnar types)")
    ap.add_argument("--hemisphere", default="right", choices=["right", "left"])
    ap.add_argument("--max-per-type", type=int, default=6,
                    help="cells per type, spanning rim->centre (0 = all cells of the type)")
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=110)
    return ap.parse_args()


def select_cells(cells, geom, hemi, n):
    """Pick up to n cells spanning the rim->centre boundary-distance range."""
    bd = geom.columns[geom.columns["hemisphere"] == hemi].set_index(["p", "q"])["boundary_distance"]
    cells = cells.copy()
    cells["bd"] = [bd.get((p, q), np.nan) for p, q in zip(cells["p"], cells["q"])]
    cells = cells.sort_values("bd", na_position="last").reset_index(drop=True)
    if n and len(cells) > n:
        idx = np.linspace(0, len(cells) - 1, n).round().astype(int)
        cells = cells.iloc[idx].reset_index(drop=True)
    return cells


def main():
    args = parse_args()
    print("Loading FlyWire data...")
    t0 = time.perf_counter()
    m = FlyWireDataManager()
    conn = add_sign(m.optic_lobe_connections_df.copy())
    col_assign = load_column_assignment(DATA_DIR)
    geom = ColumnGeometry.from_assignment(col_assign)
    stage = assign_stage_from_manager(m)
    P, Q, HM = pq_hemi_maps(col_assign)
    rank = stage.assign(r=stage["stage"].map(STAGE_RANK)).set_index("root_id")["r"].to_dict()
    hemi = args.hemisphere
    print(f"  loaded in {time.perf_counter() - t0:.1f}s ({len(conn):,} edges)")

    cl = classify_inhibition(conn, stage, col_assign=col_assign,
                             criteria=LateralInhibitionCriteria(min_syn=args.min_syn))
    cl_lat = cl[cl["label"].isin(["direct_lateral", "wide_field_lateral"])]

    if args.types:
        types = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        vc = col_assign[col_assign["hemisphere"] == hemi]["type"].value_counts()
        types = sorted(t for t in vc.index if t not in EXCLUDED and vc[t] >= 50)

    bg = col_assign[col_assign["hemisphere"] == hemi].drop_duplicates(["p", "q"])
    bgx, bgy = axial_to_cart(bg["p"].to_numpy(), bg["q"].to_numpy())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    made = 0
    for t in types:
        cells = (col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)]
                 [["root_id", "p", "q"]].drop_duplicates("root_id"))
        if len(cells) == 0:
            continue
        cells = select_cells(cells, geom, hemi, args.max_per_type)
        cids = set(cells["root_id"])

        exc = conn[(conn["sign"] == "exc") & conn["post_root_id"].isin(cids)].copy()
        exc["sp"] = exc["pre_root_id"].map(P); exc["sq"] = exc["pre_root_id"].map(Q)
        exc["sh"] = exc["pre_root_id"].map(HM)
        exc["pr"] = exc["pre_root_id"].map(rank); exc["por"] = exc["post_root_id"].map(rank)
        exc = exc.dropna(subset=["sp", "sq"])
        exc = exc[(exc["sh"] == hemi) & (exc["pr"] <= exc["por"])]
        ff_by_cell = dict(tuple(exc.groupby("post_root_id")))

        dlat = cl_lat[cl_lat["post_root_id"].isin(cids) & (cl_lat["label"] == "direct_lateral")].copy()
        dlat["sp"] = dlat["pre_root_id"].map(P); dlat["sq"] = dlat["pre_root_id"].map(Q)
        dlat["sh"] = dlat["pre_root_id"].map(HM)
        dlat = dlat[dlat["sh"] == hemi]
        dlat_by_cell = dict(tuple(dlat.groupby("post_root_id")))

        lat_e = cl_lat[cl_lat["post_root_id"].isin(cids)][["pre_root_id", "post_root_id", "syn_count"]]
        lat_e = lat_e.rename(columns={"pre_root_id": "J", "syn_count": "w1"})
        J = set(lat_e["J"])
        jin = conn[(conn["sign"] == "exc") & conn["post_root_id"].isin(J)].copy()
        jin["sp"] = jin["pre_root_id"].map(P); jin["sq"] = jin["pre_root_id"].map(Q)
        jin["sh"] = jin["pre_root_id"].map(HM)
        jin = jin.dropna(subset=["sp", "sq"]); jin = jin[jin["sh"] == hemi]
        jin["w2n"] = jin["syn_count"] / jin.groupby("post_root_id")["syn_count"].transform("sum")
        jin = jin.rename(columns={"post_root_id": "J"})[["J", "sp", "sq", "w2n"]]
        med = lat_e.merge(jin, on="J", how="inner")
        med["w"] = med["w1"] * med["w2n"]
        med = med.groupby(["post_root_id", "sp", "sq"])["w"].sum().reset_index()
        med_by_cell = dict(tuple(med.groupby("post_root_id")))

        all_lat = cl_lat[cl_lat["post_root_id"].isin(cids)]
        wf_by_cell = (all_lat[all_lat["label"] == "wide_field_lateral"]
                      .groupby("post_root_id")["syn_count"].sum().to_dict())
        dl_tot_by_cell = (all_lat[all_lat["label"] == "direct_lateral"]
                          .groupby("post_root_id")["syn_count"].sum().to_dict())
        ff_tot_by_cell = exc.groupby("post_root_id")["syn_count"].sum().to_dict()

        out_dir = OUT_ROOT / t
        out_dir.mkdir(parents=True, exist_ok=True)
        panels = [
            ("feedforward excitation", ff_by_cell, "syn_count", "Reds"),
            ("direct lateral inhibition", dlat_by_cell, "syn_count", "Blues"),
            ("mediated lateral surround", med_by_cell, "w", "Blues"),
        ]
        for i, row in cells.reset_index(drop=True).iterrows():
            rid, p0, q0 = row["root_id"], int(row["p"]), int(row["q"])
            hx, hy = axial_to_cart(p0, q0)
            fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharex=True, sharey=True)
            for ax, (title, byc, wcol, cmap) in zip(axes, panels):
                ax.scatter(bgx, bgy, c="0.88", s=30, marker="H", linewidths=0)
                df = byc.get(rid)
                ncol = 0
                if df is not None and len(df):
                    ncol = len(df)
                    sx, sy = axial_to_cart(df["sp"].to_numpy(), df["sq"].to_numpy())
                    sc = ax.scatter(sx, sy, c=df[wcol].to_numpy(), cmap=cmap, s=80, marker="H",
                                    edgecolors="0.3", linewidths=0.2)
                    fig.colorbar(sc, ax=ax, shrink=0.7)
                ax.plot(hx, hy, "*", color="cyan", ms=15, mec="black", mew=0.6, zorder=5)
                ax.set(aspect="equal", title=f"{title}\n({ncol} source columns)",
                       xlim=(hx - WINDOW, hx + WINDOW), ylim=(hy - WINDOW, hy + WINDOW))
                ax.set_xticks([]); ax.set_yticks([])
            fft = ff_tot_by_cell.get(rid, 0); dlt = dl_tot_by_cell.get(rid, 0); wft = wf_by_cell.get(rid, 0)
            bd = row.get("bd", np.nan)
            reg = ("rim" if bd <= 1 else "centre" if bd >= 3 else "middle") if pd.notna(bd) else "?"
            fig.suptitle(
                f"{t}_{i+1}   column (p={p0}, q={q0})  [{hemi}, {reg}, bd={bd if pd.notna(bd) else '?'}]   "
                f"root_id={rid}\nsynapses  FF-exc={int(fft)}   direct-lat={int(dlt)}   "
                f"wide-field-lat={int(wft)}   (cyan star = home column)", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(out_dir / f"{t}_{i+1:03d}_p{p0}_q{q0}_{rid}.png", dpi=args.dpi)
            plt.close(fig)
            made += 1
        print(f"  {t:10s}: {len(cells):4d} figures -> {out_dir.relative_to(REPO_ROOT)}  (total {made})")

    print(f"\nDone. {made} figures under {OUT_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
