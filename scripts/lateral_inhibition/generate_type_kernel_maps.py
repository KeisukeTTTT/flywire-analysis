"""Per-type 2-D Δcolumn kernels: population aggregate of the per-cell footprints.

For each columnar type, re-center every cell's source columns onto its own home
column (Δcolumn = source - home) and average over cells -> the population
center-surround receptive field in column space. The 2-D generalization of
:meth:`src.lateral.RadialKernels.direct_kernel` / ``disyn_kernel`` (which collapse the
same thing to a 1-D radius); keeping it 2-D preserves anisotropy such as the L4->L2
directional surround or the T4/T5 motion-offset surround.

The edge effect is evaluated *continuously* along ``boundary_distance`` (BFS hops
from the lattice edge) rather than by an interior/edge threshold. Each drive component
(feedforward excitation / direct lateral inhibition / mediated lateral surround
exc->inh(J)->cell) is shown in two complementary, separately-saved views:

* a quantitative *edge profile* -- two line plots of a per-cell kernel summary vs the
  continuous boundary_distance: the inward asymmetry ``A = sum(w*inx)/sum(w)``
  (weighted-mean inward-aligned Δx; ~0 deep inside, grows positive at the rim because
  outward sources are missing) and the relative drive magnitude ``sum(w)`` normalised
  to each component's interior median (compensation: does drive drop at the edge?).
* a qualitative *kernel ladder* -- inward-aligned 2-D Δcolumn kernels binned by
  boundary_distance (bd<=1 rim ... bd>=5 centre), so the kernel's continuous
  deformation toward the edge is visible. Values are mean synapses per cell per bin.

Caveats: ``nt_type`` sign is mostly ML-predicted; synapse count is a structural proxy.

Run from the project root::

    uv run python scripts/lateral_inhibition/generate_type_kernel_maps.py --types Mi1,L4,T4a
    uv run python scripts/lateral_inhibition/generate_type_kernel_maps.py        # all columnar types

Output:
    outputs/lateral_inhibition/type_kernels/<type>.png          (kernel ladder)
    outputs/lateral_inhibition/type_edge_profiles/<type>.png    (edge profile)
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

OUT_DIR = REPO_ROOT / "outputs" / "lateral_inhibition" / "type_kernels"
OUT_PROFILE_DIR = REPO_ROOT / "outputs" / "lateral_inhibition" / "type_edge_profiles"
EXCLUDED = {"R7", "R8", "R1-6"}
WINDOW = 8.0           # half-width of the Δcolumn window (columns)
NBINS = 41             # bin edges across [-WINDOW, WINDOW]
CENTER_MIN_BD = 3      # boundary_distance >= this = interior reference (magnitude norm)
# boundary_distance bins for the kernel ladder: (label, lo, hi) inclusive
BD_SERIES = [("bd≤1 (rim)", 0, 1), ("bd=2", 2, 2), ("bd=3", 3, 3),
             ("bd=4", 4, 4), ("bd≥5 (centre)", 5, np.inf)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", default=None,
                    help="comma-separated receiver types (default: all columnar types)")
    ap.add_argument("--hemisphere", default="right", choices=["right", "left"])
    ap.add_argument("--inward-method", default="local", choices=["local", "centroid"],
                    help="inward axis: 'local' edge normal (default) or 'centroid' (legacy)")
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=120)
    return ap.parse_args()


def hist2d(x, y, w):
    edges = np.linspace(-WINDOW, WINDOW, NBINS)
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=w)
    return H, edges


def per_cell_asym_mag(d):
    """Per-cell kernel summary: inward asymmetry A=sum(w*inx)/sum(w), magnitude M=sum(w)."""
    if len(d) == 0:
        return pd.DataFrame(columns=["post_root_id", "A", "M", "bd"])
    M = d.groupby("post_root_id")["w"].sum()
    winx = (d["w"].to_numpy() * d["inx"].to_numpy())
    A = pd.Series(winx, index=d["post_root_id"].to_numpy()).groupby(level=0).sum() / M
    bd = d.groupby("post_root_id")["bd"].first()
    return pd.DataFrame({"A": A, "M": M, "bd": bd}).reset_index(names="post_root_id")


def draw(ax, H, n_cells, cmap, inward=False, vmax=None):
    Hm = H.T / max(n_cells, 1)          # mean synapses per cell per bin
    if vmax is None:
        vmax = float(Hm.max()) if Hm.max() > 0 else 1.0
    if inward:
        ax.axvspan(-WINDOW, 0, color="0.92", zorder=0)  # outward / off-lattice side
    im = ax.imshow(Hm, origin="lower", extent=[-WINDOW, WINDOW, -WINDOW, WINDOW],
                   cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", zorder=1)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4); ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set(aspect="equal", xlim=(-WINDOW, WINDOW), ylim=(-WINDOW, WINDOW))
    ax.set_xticks([]); ax.set_yticks([])
    return im


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FlyWire data...")
    t0 = time.perf_counter()
    m = FlyWireDataManager()
    conn = add_sign(m.optic_lobe_connections_df.copy())
    col_assign = load_column_assignment(DATA_DIR)
    geom = ColumnGeometry.from_assignment(col_assign, inward_method=args.inward_method)
    stage = assign_stage_from_manager(m)
    P, Q, HM = pq_hemi_maps(col_assign)
    rank = stage.assign(r=stage["stage"].map(STAGE_RANK)).set_index("root_id")["r"].to_dict()
    cells_geo = geom.cells.dropna(subset=["region"]).drop_duplicates("root_id").set_index("root_id")
    IUX, IUY, BD = cells_geo["inward_unit_x"].to_dict(), cells_geo["inward_unit_y"].to_dict(), cells_geo["boundary_distance"].to_dict()
    hemi = args.hemisphere
    print(f"  loaded in {time.perf_counter() - t0:.1f}s ({len(conn):,} edges)")

    # precompute coordinate-resolved edge tables (once)
    exc = conn[conn["sign"] == "exc"][["pre_root_id", "post_root_id", "syn_count"]].copy()
    exc["sp"] = exc["pre_root_id"].map(P); exc["sq"] = exc["pre_root_id"].map(Q)
    exc["sh"] = exc["pre_root_id"].map(HM)
    exc = exc.dropna(subset=["sp", "sq"]); exc = exc[exc["sh"] == hemi]
    exc["pr"] = exc["pre_root_id"].map(rank); exc["por"] = exc["post_root_id"].map(rank)
    FF = exc[exc["pr"] <= exc["por"]][["post_root_id", "sp", "sq", "syn_count"]]
    JIN = exc[["pre_root_id", "post_root_id", "sp", "sq", "syn_count"]].rename(columns={"post_root_id": "J"})
    JIN["w2n"] = JIN["syn_count"] / JIN.groupby("J")["syn_count"].transform("sum")
    JIN = JIN[["J", "sp", "sq", "w2n"]]

    cl = classify_inhibition(conn, stage, col_assign=col_assign,
                             criteria=LateralInhibitionCriteria(min_syn=args.min_syn))
    LAT = cl[cl["label"].isin(["direct_lateral", "wide_field_lateral"])][
        ["pre_root_id", "post_root_id", "syn_count", "label"]]
    DLAT = LAT[LAT["label"] == "direct_lateral"].copy()
    DLAT["sp"] = DLAT["pre_root_id"].map(P); DLAT["sq"] = DLAT["pre_root_id"].map(Q)
    DLAT["sh"] = DLAT["pre_root_id"].map(HM)
    DLAT = DLAT[DLAT["sh"] == hemi][["post_root_id", "sp", "sq", "syn_count"]]

    if args.types:
        types = [t.strip() for t in args.types.split(",") if t.strip()]
    else:
        vc = col_assign[col_assign["hemisphere"] == hemi]["type"].value_counts()
        types = sorted(t for t in vc.index if t not in EXCLUDED and vc[t] >= args.min_cells)
    print(f"\nReceiver types: {len(types)} | hemisphere: {hemi}")

    cols = [("feedforward excitation", "Reds"), ("direct lateral inhibition", "Blues"),
            ("mediated lateral surround", "Blues")]
    made = 0
    for t in types:
        cdf = col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)].drop_duplicates("root_id")
        cids = set(cdf["root_id"])
        if len(cids) < (1 if args.types else args.min_cells):
            continue

        ff = FF[FF["post_root_id"].isin(cids)].rename(columns={"syn_count": "w"})
        dl = DLAT[DLAT["post_root_id"].isin(cids)].rename(columns={"syn_count": "w"})
        lat_t = LAT[LAT["post_root_id"].isin(cids)][["pre_root_id", "post_root_id", "syn_count"]] \
            .rename(columns={"pre_root_id": "J", "syn_count": "w1"})
        jin_t = JIN[JIN["J"].isin(set(lat_t["J"]))]
        med = lat_t.merge(jin_t, on="J", how="inner")
        med["w"] = med["w1"] * med["w2n"]
        med = med[["post_root_id", "sp", "sq", "w"]]

        comps = [ff, dl, med]
        for d in comps:   # re-center each point onto its post cell's home column, Cartesian Δ
            if len(d) == 0:
                d["dx"] = []; d["dy"] = []; d["inx"] = []; d["iny"] = []; d["bd"] = []; continue
            dpx, dpy = axial_to_cart(d["sp"].to_numpy() - d["post_root_id"].map(P).to_numpy(),
                                     d["sq"].to_numpy() - d["post_root_id"].map(Q).to_numpy())
            ux = d["post_root_id"].map(IUX).to_numpy(); uy = d["post_root_id"].map(IUY).to_numpy()
            d["dx"], d["dy"] = dpx, dpy
            d["inx"] = dpx * ux + dpy * uy          # inward-aligned x (toward centre)
            d["iny"] = -dpx * uy + dpy * ux         # inward-aligned y (tangential)
            d["bd"] = d["post_root_id"].map(BD)

        n_all = len(cids)
        cbd = pd.Series({c: BD.get(c, np.nan) for c in cids})   # per-cell boundary_distance
        n_rim = int((cbd <= 1).sum())

        line_meta = [("feedforward excitation", "tab:red"),
                     ("direct lateral inhibition", "tab:blue"),
                     ("mediated lateral surround", "tab:purple")]
        scalars = [per_cell_asym_mag(comps[j]) for j in range(3)]
        stg = stage.set_index("root_id")["stage"].get(next(iter(cids)), "?")

        # ============ figure 1: continuous edge profile (scalar vs boundary_distance) ============
        figp, (ax_asym, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.6))
        for (name, color), sc in zip(line_meta, scalars):
            sc = sc.dropna(subset=["bd"])
            if len(sc) == 0:
                continue
            sc = sc.assign(bdi=sc["bd"].round().astype(int))
            gA = sc.groupby("bdi")["A"]
            x = gA.median().index.to_numpy()
            ax_asym.plot(x, gA.median().to_numpy(), "-o", color=color, ms=4, label=name)
            ax_asym.fill_between(x, gA.quantile(.25).to_numpy(), gA.quantile(.75).to_numpy(),
                                 color=color, alpha=0.13)
            ref = sc.loc[sc["bd"] >= CENTER_MIN_BD, "M"].median()
            if ref and np.isfinite(ref) and ref > 0:
                gM = sc.assign(rel=sc["M"] / ref).groupby("bdi")["rel"]
                xm = gM.median().index.to_numpy()
                ax_mag.plot(xm, gM.median().to_numpy(), "-o", color=color, ms=4, label=name)
                ax_mag.fill_between(xm, gM.quantile(.25).to_numpy(), gM.quantile(.75).to_numpy(),
                                    color=color, alpha=0.13)
        ax_asym.axhline(0, color="gray", lw=0.7, ls=":")
        ax_asym.set(xlabel="boundary_distance  (0 = rim → centre)",
                    ylabel="inward asymmetry ⟨inx⟩ (columns)",
                    title="kernel inward shift vs edge distance\n(⟨inx⟩>0 → pushed inward; outward sources missing)")
        ax_asym.grid(alpha=0.3); ax_asym.legend(fontsize=8)
        ax_mag.axhline(1, color="gray", lw=0.7, ls=":")
        ax_mag.set(xlabel="boundary_distance  (0 = rim → centre)",
                   ylabel="relative drive  Σw / interior median",
                   title="drive magnitude vs edge distance\n(drop at low bd → drive falls; flat → compensation)")
        ax_mag.grid(alpha=0.3); ax_mag.legend(fontsize=8)
        figp.suptitle(f"{t}   [{hemi} hemisphere, stage={stg}, n_cells={n_all}, "
                      f"inward={args.inward_method}]   "
                      "continuous edge profile (median ± IQR over cells)", fontsize=12)
        figp.tight_layout(rect=(0, 0, 1, 0.95))
        figp.savefig(OUT_PROFILE_DIR / f"{t}.png", dpi=args.dpi)
        plt.close(figp)

        # ============ figure 2: kernel ladder (inward-aligned kernels binned by bd) ============
        nbins = len(BD_SERIES)
        fig = plt.figure(figsize=(13, 2.7 * nbins))
        gs = fig.add_gridspec(nbins, 3, hspace=0.12, wspace=0.18)
        for j, (title, cmap) in enumerate(cols):
            d = comps[j]
            Hs, ns = [], []
            for (lab, lo, hi) in BD_SERIES:
                n = int(((cbd >= lo) & (cbd <= hi)).sum())
                dm = d[(d["bd"] >= lo) & (d["bd"] <= hi)] if len(d) else d
                H, _ = hist2d(np.asarray(dm["inx"]), np.asarray(dm["iny"]), np.asarray(dm["w"])) \
                    if len(dm) else (np.zeros((NBINS - 1, NBINS - 1)), None)
                Hs.append(H); ns.append(n)
            vmax = max([float((H.T / max(n, 1)).max()) for H, n in zip(Hs, ns)] + [1e-9])
            col_axes = []
            for r, (lab, lo, hi) in enumerate(BD_SERIES):
                ax = fig.add_subplot(gs[r, j])
                im = draw(ax, Hs[r], ns[r], cmap, inward=True, vmax=vmax)
                col_axes.append(ax)
                if j == 0:
                    ax.set_ylabel(f"{lab}\n(n={ns[r]})", fontsize=9)
                if r == 0:
                    ax.set_title(title, fontsize=11)
                if r == 0 and j == 1:
                    ax.annotate("inward →", (0.6, 0.93), xycoords="axes fraction", fontsize=9)
                    ax.annotate("← outward (off-lattice)", (0.02, 0.04),
                                xycoords="axes fraction", fontsize=8, color="0.4")
            fig.colorbar(im, ax=col_axes, shrink=0.6, location="right", pad=0.01)

        fig.suptitle(f"{t}   [{hemi} hemisphere, stage={stg}, n_cells={n_all}, "
                     f"inward={args.inward_method}]   "
                     "inward-aligned Δcolumn kernels binned by boundary_distance "
                     "(origin = home column / post-neuron; mean synapses/cell/bin)",
                     fontsize=12, y=0.998)
        fig.savefig(OUT_DIR / f"{t}.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  {t:10s} n_cells={n_all:4d} rim={n_rim:4d}  ({made}/{len(types)})")

    print(f"\nDone. {made} type(s) -> {OUT_DIR.relative_to(REPO_ROOT)} (ladder) "
          f"+ {OUT_PROFILE_DIR.relative_to(REPO_ROOT)} (profile)")


if __name__ == "__main__":
    main()
