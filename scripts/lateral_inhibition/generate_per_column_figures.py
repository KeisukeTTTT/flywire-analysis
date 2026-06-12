"""Per-column presynaptic input footprints for columnar receiver cell types.

For every postsynaptic column (each cell of a columnar receiver type in
``column_assignment.csv``), draw a 2-panel figure of its *columnar* presynaptic
input footprint: excitatory (left) and inhibitory (right) source columns placed
at their true hex positions on the medulla lattice, coloured by synapse count.
The receiver's own (home) column is marked, and the full lattice is shown in
grey so edge vs. centre position is visible.

Caveat: only presynaptic partners that have column coordinates (= columnar
sources in ``column_assignment.csv``) can be placed. Wide-field inhibitory
sources (Dm/Pm/Lai/CT1 ...) have no column and are omitted, so the inh panel is
a partial view (median coverage ~26% of inh synapses vs ~90% of exc).

Run from the project root::

    # smoke test (a few Mi1 columns)
    uv run python scripts/lateral_inhibition/generate_per_column_figures.py --types Mi1 --max-per-type 3
    # full batch (all 29 columnar types, thousands of figures)
    uv run python scripts/lateral_inhibition/generate_per_column_figures.py

Output: outputs/lateral_inhibition/per_column/<type>/<type>_p<p>_q<q>_<root_id>.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless batch rendering
import matplotlib.pyplot as plt

from src.config import DATA_DIR
from src.data import FlyWireDataManager
from src.lateral import axial_to_cart, classify_nt, load_column_assignment

OUT_ROOT = REPO_ROOT / "outputs" / "lateral_inhibition" / "per_column"
WINDOW = 9.0  # zoom half-width (cart units ~ hex columns) around the home column
EXCLUDED_RECEIVERS = {"R7", "R8"}  # photoreceptor inputs, not edge-effect receivers


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", default=None,
                    help="comma-separated receiver types (default: all columnar types)")
    ap.add_argument("--hemisphere", default="right", choices=["right", "left"])
    ap.add_argument("--max-per-type", type=int, default=None,
                    help="cap number of columns per type (for testing)")
    ap.add_argument("--dpi", type=int, default=100)
    return ap.parse_args()


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading FlyWire data...")
    t0 = time.perf_counter()
    m = FlyWireDataManager()
    conn = m.optic_lobe_connections_df.copy()
    conn["sign"] = conn["nt_type"].map(classify_nt)
    col_assign = load_column_assignment(DATA_DIR)
    col_map = col_assign.set_index("root_id")[["p", "q", "hemisphere"]]
    print(f"  loaded in {time.perf_counter() - t0:.1f}s ({len(conn):,} edges)")

    hemi = args.hemisphere
    if args.types:
        target_types = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        target_types = sorted(
            t for t in col_assign["type"].dropna().unique() if t not in EXCLUDED_RECEIVERS
        )

    # Background lattice (all columns of this hemisphere), drawn grey on every panel.
    bg = col_assign[col_assign["hemisphere"] == hemi].drop_duplicates(["p", "q"])
    bgx, bgy = axial_to_cart(bg["p"].to_numpy(), bg["q"].to_numpy())

    # Pre-count for the user.
    counts = {}
    for t in target_types:
        n = col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)][
            "root_id"
        ].nunique()
        if args.max_per_type:
            n = min(n, args.max_per_type)
        counts[t] = n
    total = sum(counts.values())
    print(f"\nReceiver types: {len(target_types)} | hemisphere: {hemi} | "
          f"figures to generate: {total:,}")

    panels = [("exc", "Blues"), ("inh", "Reds")]
    made = 0
    for t in target_types:
        cells = (
            col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)]
            [["root_id", "p", "q"]]
            .drop_duplicates("root_id")
            .reset_index(drop=True)
        )
        if len(cells) == 0:
            continue
        if args.max_per_type:
            cells = cells.head(args.max_per_type)
        cids = set(cells["root_id"])

        # All columnar-source inputs to this type's cells, in the same hemisphere.
        inc = conn[conn["post_root_id"].isin(cids) & conn["sign"].isin(["exc", "inh"])].copy()
        inc["p_src"] = inc["pre_root_id"].map(col_map["p"])
        inc["q_src"] = inc["pre_root_id"].map(col_map["q"])
        inc["hemi_src"] = inc["pre_root_id"].map(col_map["hemisphere"])
        inc = inc.dropna(subset=["p_src", "q_src"])
        inc = inc[inc["hemi_src"] == hemi]
        agg = (
            inc.groupby(["post_root_id", "sign", "p_src", "q_src"])["syn_count"]
            .sum()
            .reset_index()
        )
        by_cell = dict(tuple(agg.groupby("post_root_id")))

        out_dir = OUT_ROOT / t
        out_dir.mkdir(parents=True, exist_ok=True)

        tt = time.perf_counter()
        for _, row in cells.iterrows():
            rid, p0, q0 = row["root_id"], int(row["p"]), int(row["q"])
            cell_df = by_cell.get(rid)
            hx, hy = axial_to_cart(p0, q0)
            # Per-cell colour scale (shared by both panels) so each figure is self-readable.
            vmax = float(cell_df["syn_count"].max()) if cell_df is not None and len(cell_df) else 1.0

            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharex=True, sharey=True)
            for ax, (sgn, cmap) in zip(axes, panels):
                ax.scatter(bgx, bgy, c="lightgray", s=40, marker="H", alpha=0.35, linewidths=0)
                n_cols = 0
                if cell_df is not None:
                    sub = cell_df[cell_df["sign"] == sgn]
                    n_cols = len(sub)
                    if n_cols:
                        sx, sy = axial_to_cart(sub["p_src"].to_numpy(), sub["q_src"].to_numpy())
                        sc = ax.scatter(sx, sy, c=sub["syn_count"], cmap=cmap, s=70, marker="H",
                                        edgecolors="black", linewidths=0.3, vmin=0, vmax=vmax)
                        fig.colorbar(sc, ax=ax, shrink=0.7, label="syn from source column")
                # home column marker
                ax.plot(hx, hy, "*", color="cyan", markersize=13,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)
                # zoom around home so the (small) footprint is readable; for edge cells the
                # grey lattice is truncated inside the window, making edge proximity visible.
                ax.set(aspect="equal", title=f"{sgn} input — {n_cols} source columns",
                       xlim=(hx - WINDOW, hx + WINDOW), ylim=(hy - WINDOW, hy + WINDOW))
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{t}  column (p={p0}, q={q0})  [{hemi} hemi]  root_id={rid}\n"
                         "columnar presynaptic input footprint  (cyan star = home column)",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(out_dir / f"{t}_p{p0}_q{q0}_{rid}.png", dpi=args.dpi)
            plt.close(fig)
            made += 1
        print(f"  {t:8s}: {len(cells):4d} figures in {time.perf_counter() - tt:5.1f}s "
              f"-> {out_dir.relative_to(REPO_ROOT)}  ({made:,}/{total:,})")

    print(f"\nDone. {made:,} figures written under {OUT_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
