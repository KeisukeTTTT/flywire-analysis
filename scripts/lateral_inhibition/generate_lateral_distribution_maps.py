"""Per-type retinotopic maps of lateral inhibition (one figure per columnar type).

For every columnar receiver type, draw the spatial distribution across columns of the
lateral inhibition each cell receives -- one hexagon per cell at its true hex
position, no aggregation across columns. Three lattice panels:

* feedforward excitation -- excitatory input synapses from the same-or-earlier stage.
* lateral inhibition -- same-stage ``direct_lateral`` + ``wide_field_lateral`` input.
* lateral fraction ``lat / (lat + ff_exc)`` -- structural I/(I+E) balance (fixed 0..1).

The full hemisphere lattice is drawn in grey so edge vs. centre columns are visible.
Caveats: ``nt_type`` sign is mostly ML-predicted; synapse count is a structural proxy.

Run from the project root::

    uv run python scripts/lateral_inhibition/generate_lateral_distribution_maps.py --types Mi1,L2,T4a
    uv run python scripts/lateral_inhibition/generate_lateral_distribution_maps.py   # all columnar types

Output: outputs/lateral_inhibition/lateral_distribution/<type>.png
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
    LateralInhibitionCriteria,
    add_sign,
    assign_stage_from_manager,
    axial_to_cart,
    classify_inhibition,
    load_column_assignment,
)
from src.config import DATA_DIR
from src.data import FlyWireDataManager

OUT_DIR = REPO_ROOT / "outputs" / "lateral_inhibition" / "lateral_distribution"
EXCLUDED = {"R7", "R8", "R1-6"}
LATERAL_LABELS = {"direct_lateral", "wide_field_lateral"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", default=None,
                    help="comma-separated receiver types (default: all columnar types)")
    ap.add_argument("--hemisphere", default="right", choices=["right", "left"])
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--min-syn", type=int, default=5,
                    help="min synapses per connection (default 5 = LateralInhibitionCriteria default)")
    ap.add_argument("--dpi", type=int, default=110)
    return ap.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FlyWire data...")
    t0 = time.perf_counter()
    m = FlyWireDataManager()
    conn = add_sign(m.optic_lobe_connections_df.copy())
    col_assign = load_column_assignment(DATA_DIR)
    stage = assign_stage_from_manager(m)
    rank = stage.assign(r=stage["stage"].map(STAGE_RANK)).set_index("root_id")["r"].to_dict()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s ({len(conn):,} edges)")

    # per-cell feedforward excitation (same-or-earlier stage)
    exc = conn[conn["sign"] == "exc"][["pre_root_id", "post_root_id", "syn_count"]].copy()
    exc["pre_r"] = exc["pre_root_id"].map(rank)
    exc["post_r"] = exc["post_root_id"].map(rank)
    ff = exc[exc["pre_r"] <= exc["post_r"]]   # NaN comparisons -> False -> excluded
    ff_exc = ff.groupby("post_root_id")["syn_count"].sum()

    # per-cell lateral inhibition (same-stage direct / wide-field)
    cl = classify_inhibition(conn, stage, col_assign=col_assign,
                             criteria=LateralInhibitionCriteria(min_syn=args.min_syn))
    lat = cl[cl["label"].isin(LATERAL_LABELS)]
    lat_in = lat.groupby("post_root_id")["syn_count"].sum()
    print(f"  feedforward-exc cells: {len(ff_exc):,}  lateral-inh cells: {len(lat_in):,}")

    hemi = args.hemisphere
    if args.types:
        types = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        vc = col_assign[col_assign["hemisphere"] == hemi]["type"].value_counts()
        types = sorted(t for t in vc.index if t not in EXCLUDED and vc[t] >= args.min_cells)
    print(f"\nReceiver types: {len(types)} | hemisphere: {hemi}")

    bg = col_assign[col_assign["hemisphere"] == hemi].drop_duplicates(["p", "q"])
    bgx, bgy = axial_to_cart(bg["p"].to_numpy(), bg["q"].to_numpy())

    panels = [
        ("feedforward excitation", "ff_exc", "Reds", None),
        ("lateral inhibition", "lat_in", "Blues", None),
        ("lateral fraction  lat/(lat+ff_exc)", "ratio", "viridis", (0.0, 1.0)),
    ]
    made = 0
    for t in types:
        cells = (col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)]
                 [["root_id", "p", "q"]].drop_duplicates("root_id").reset_index(drop=True))
        if len(cells) < (1 if args.types else args.min_cells):
            continue
        cells["ff_exc"] = cells["root_id"].map(ff_exc).fillna(0.0)
        cells["lat_in"] = cells["root_id"].map(lat_in).fillna(0.0)
        denom = cells["lat_in"] + cells["ff_exc"]
        cells["ratio"] = np.where(denom > 0, cells["lat_in"] / denom, np.nan)
        cx, cy = axial_to_cart(cells["p"].to_numpy(), cells["q"].to_numpy())

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
        for ax, (title, col, cmap, vlim) in zip(axes, panels):
            ax.scatter(bgx, bgy, c="0.88", s=26, marker="H", linewidths=0)
            vals = cells[col].to_numpy()
            vmin, vmax = (vlim if vlim else (0.0, float(np.nanmax(vals)) if np.isfinite(vals).any() and np.nanmax(vals) > 0 else 1.0))
            sc = ax.scatter(cx, cy, c=vals, cmap=cmap, s=34, marker="H",
                            vmin=vmin, vmax=vmax, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.7)
            med = np.nanmedian(vals)
            ax.set(aspect="equal", title=f"{title}\n(median={med:.2f})")
            ax.set_xticks([]); ax.set_yticks([])
        stg = stage.set_index("root_id")["stage"].get(cells["root_id"].iloc[0], "?")
        fig.suptitle(f"{t}   [{hemi} hemisphere, stage={stg}, n_cells={len(cells)}]   "
                     "lateral inhibition across columns  (grey = full lattice)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT_DIR / f"{t}.png", dpi=args.dpi)
        plt.close(fig)
        made += 1
        print(f"  {t:10s} n_cells={len(cells):4d}  median lat_frac={np.nanmedian(cells['ratio']):.2f}  ({made}/{len(types)})")

    print(f"\nDone. {made} figures -> {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
