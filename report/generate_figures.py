"""Regenerate the figures used in report/main.tex.

Run from the project root::

    uv run python report/generate_figures.py

Output: report/figures/fig*.png (about 8 figures, ~60 s total).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from src.config import DATA_DIR
from src.data import FlyWireDataManager

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

INHIBITORY_NT = {"GABA", "GLUT", "HIS"}
EXCITATORY_NT = {"ACH"}


def classify_nt(x):
    if x in INHIBITORY_NT:
        return "inh"
    if x in EXCITATORY_NT:
        return "exc"
    return "other"


def axial_to_cart(p, q):
    return p + 0.5 * q, q * (np.sqrt(3) / 2)


def hex_distance(dp, dq):
    return (np.abs(dp) + np.abs(dp + dq) + np.abs(dq)) // 2


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


print("Loading FlyWire data...")
t0 = time.perf_counter()
m = FlyWireDataManager()
neurons = m.optic_lobe_neurons_df.copy()
conn = m.optic_lobe_connections_df.copy()
conn["sign"] = conn["nt_type"].map(classify_nt)
conn["same_type"] = conn["pre_primary_type"] == conn["post_primary_type"]
print(f"  loaded in {time.perf_counter() - t0:.1f}s "
      f"({len(neurons):,} neurons, {len(conn):,} edges)")

col_assign = pd.read_csv(
    Path(DATA_DIR) / "raw" / "flywire" / "csv" / "column_assignment.csv",
    dtype={"root_id": str, "column_id": str},
)
col_map = col_assign.set_index("root_id")[["p", "q", "hemisphere"]]
pre_side_map = neurons.drop_duplicates("root_id").set_index("root_id")["side"]


# --------------------------------------------------------------------------
# Fig 1: Q1 nt_type breakdown
# --------------------------------------------------------------------------
print("\nFig 1: Q1 nt_type breakdown")
nt_syn = conn.groupby("nt_type", dropna=False)["syn_count"].sum().sort_values()
colors = [
    "tab:red" if x in INHIBITORY_NT else ("tab:blue" if x in EXCITATORY_NT else "gray")
    for x in nt_syn.index
]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
nt_syn.plot.barh(ax=ax, color=colors)
ax.set(xlabel="# synapses",
       title="Synapse count by nt_type (red=inh, blue=exc, gray=modulatory/unknown)")
plt.tight_layout()
save(fig, "fig1_nt_distribution.png")


# --------------------------------------------------------------------------
# Q2 / Q4 setup
# --------------------------------------------------------------------------
type_io = conn.groupby(["pre_primary_type", "sign"])["syn_count"].sum().unstack(fill_value=0)
for col in ("inh", "exc", "other"):
    if col not in type_io.columns:
        type_io[col] = 0
type_io["total"] = type_io["inh"] + type_io["exc"] + type_io["other"]
type_io["inh_frac"] = type_io["inh"] / type_io["total"].clip(lower=1)
active = type_io[type_io["total"] >= 1000]


# --------------------------------------------------------------------------
# Fig 2: Q2 inh fraction distribution per cell type
# --------------------------------------------------------------------------
print("\nFig 2: Q2 inh fraction distribution")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.hist(active["inh_frac"], bins=50, color="steelblue", edgecolor="white")
ax.set(xlabel="inhibitory output fraction (per cell type)",
       ylabel="# cell types",
       title=f"Distribution of inh output fraction (n={len(active)} types, >=1000 outgoing syn)")
plt.tight_layout()
save(fig, "fig2_inh_fraction_dist.png")


# --------------------------------------------------------------------------
# Q4 column-based lateral spread
# --------------------------------------------------------------------------
print("\nQ4 setup: column-based lateral spread")
ec = conn[["pre_root_id", "post_root_id", "syn_count", "pre_primary_type", "sign"]].copy()
ec["p_post"] = ec["post_root_id"].map(col_map["p"])
ec["q_post"] = ec["post_root_id"].map(col_map["q"])
ec["hemi_post"] = ec["post_root_id"].map(col_map["hemisphere"])
ec["hemi_pre"] = ec["pre_root_id"].map(pre_side_map)
ec = ec.dropna(subset=["p_post", "q_post", "hemi_pre"])
ec = ec[ec["hemi_pre"] == ec["hemi_post"]]

ec["wp"] = ec["p_post"] * ec["syn_count"]
ec["wq"] = ec["q_post"] * ec["syn_count"]
per_pre = ec.groupby("pre_root_id").agg(
    wp_sum=("wp", "sum"),
    wq_sum=("wq", "sum"),
    w_sum=("syn_count", "sum"),
    n_targets=("syn_count", "size"),
)
per_pre["pc"] = per_pre["wp_sum"] / per_pre["w_sum"]
per_pre["qc"] = per_pre["wq_sum"] / per_pre["w_sum"]
per_pre = per_pre[(per_pre["w_sum"] >= 100) & (per_pre["n_targets"] >= 5)]

ec["pc"] = ec["pre_root_id"].map(per_pre["pc"])
ec["qc"] = ec["pre_root_id"].map(per_pre["qc"])
ec = ec.dropna(subset=["pc"])
dp = ec["p_post"] - ec["pc"]
dq = ec["q_post"] - ec["qc"]
ec["d_hex"] = np.sqrt((dp ** 2 + dq ** 2 + (dp + dq) ** 2) / 2)
ec["wd"] = ec["d_hex"] * ec["syn_count"]

spread = ec.groupby("pre_root_id").agg(
    wd_sum=("wd", "sum"),
    w_sum=("syn_count", "sum"),
    n_targets=("syn_count", "size"),
)
spread["spread_wmean"] = spread["wd_sum"] / spread["w_sum"]
ec["col_tup"] = list(zip(ec["p_post"].astype(int), ec["q_post"].astype(int)))
spread["n_unique_cols"] = ec.groupby("pre_root_id")["col_tup"].nunique()
type_map = neurons.drop_duplicates("root_id").set_index("root_id")[["primary_type", "nt_type"]]
spread = spread.join(type_map).dropna(subset=["primary_type"])
spread["sign"] = spread["nt_type"].map(classify_nt)

per_type_col = (
    spread.groupby("primary_type")
    .agg(
        n_neurons=("spread_wmean", "size"),
        type_spread=("spread_wmean", "median"),
        type_n_cols=("n_unique_cols", "median"),
        dominant_sign=("sign", lambda x: x.value_counts().idxmax()),
    )
    .query("n_neurons >= 5")
)


# --------------------------------------------------------------------------
# Fig 3: Q4 column spread distribution inh vs exc per cell type
# --------------------------------------------------------------------------
print("\nFig 3: Q4 column spread inh vs exc")
inh_t = per_type_col[per_type_col["dominant_sign"] == "inh"]
exc_t = per_type_col[per_type_col["dominant_sign"] == "exc"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
upper = float(np.percentile(np.concatenate([inh_t["type_spread"], exc_t["type_spread"]]), 99))
bins = np.linspace(0, upper, 25)
axes[0].hist(exc_t["type_spread"], bins=bins, alpha=0.5, density=True,
             label=f"exc (n={len(exc_t)})", color="tab:blue")
axes[0].hist(inh_t["type_spread"], bins=bins, alpha=0.5, density=True,
             label=f"inh (n={len(inh_t)})", color="tab:red")
axes[0].axvline(exc_t["type_spread"].median(), color="tab:blue", ls="--", lw=0.8)
axes[0].axvline(inh_t["type_spread"].median(), color="tab:red", ls="--", lw=0.8)
axes[0].set(xlabel="weighted-mean Delta-column from centroid (hex)",
            ylabel="density",
            title=f"Lateral spread per cell type (inh/exc ratio = {inh_t['type_spread'].median()/exc_t['type_spread'].median():.2f})")
axes[0].legend(fontsize=8)

upper2 = float(np.percentile(np.concatenate([inh_t["type_n_cols"], exc_t["type_n_cols"]]), 99))
bins2 = np.linspace(0, upper2, 25)
axes[1].hist(exc_t["type_n_cols"], bins=bins2, alpha=0.5, density=True,
             label=f"exc (n={len(exc_t)})", color="tab:blue")
axes[1].hist(inh_t["type_n_cols"], bins=bins2, alpha=0.5, density=True,
             label=f"inh (n={len(inh_t)})", color="tab:red")
axes[1].axvline(exc_t["type_n_cols"].median(), color="tab:blue", ls="--", lw=0.8)
axes[1].axvline(inh_t["type_n_cols"].median(), color="tab:red", ls="--", lw=0.8)
axes[1].set(xlabel="median # unique target columns per neuron",
            ylabel="density",
            title=f"# target columns per neuron (inh/exc ratio = {inh_t['type_n_cols'].median()/exc_t['type_n_cols'].median():.2f})")
axes[1].legend(fontsize=8)
plt.tight_layout()
save(fig, "fig3_column_spread_inh_vs_exc.png")


# --------------------------------------------------------------------------
# Fig 4: Q4 hex maps of representative cells
# --------------------------------------------------------------------------
print("\nFig 4: Q4 hex maps of representative cells")
example_cells = ["Pm08", "Pm04", "Dm4", "Dm12", "Lai", "Mi1"]
fig, axes = plt.subplots(1, len(example_cells), figsize=(3.4 * len(example_cells), 4),
                          sharex=True, sharey=True)
panel_data = []
for ctype in example_cells:
    cands = spread[spread["primary_type"] == ctype].sort_values("w_sum", ascending=False)
    if len(cands) == 0:
        panel_data.append(None)
        continue
    chosen_id = cands.index[0]
    edges_c = ec[ec["pre_root_id"] == chosen_id]
    col_syn = edges_c.groupby(["p_post", "q_post"], as_index=False)["syn_count"].sum()
    hemi = edges_c["hemi_post"].iloc[0]
    panel_data.append((ctype, chosen_id, col_syn, hemi))

vmax = max((d[2]["syn_count"].max() for d in panel_data if d is not None), default=1)
for ax, item in zip(axes, panel_data):
    if item is None:
        ax.set_axis_off()
        continue
    ctype, chosen_id, col_syn, hemi = item
    bg = col_assign[col_assign["hemisphere"] == hemi].drop_duplicates(["p", "q"])[["p", "q"]]
    bx, by = axial_to_cart(bg["p"].values, bg["q"].values)
    ax.scatter(bx, by, c="lightgray", s=70, marker="H", alpha=0.45, linewidths=0)
    sx, sy = axial_to_cart(col_syn["p_post"].values, col_syn["q_post"].values)
    sc = ax.scatter(sx, sy, c=col_syn["syn_count"], cmap="hot_r", s=95, marker="H",
                    edgecolors="black", linewidths=0.3, vmin=0, vmax=vmax)
    pc, qc = per_pre.loc[chosen_id, ["pc", "qc"]]
    cx, cy = axial_to_cart(pc, qc)
    ax.plot(cx, cy, "x", color="cyan", markersize=12, markeredgewidth=3)
    sign = spread.loc[chosen_id, "sign"]
    spread_val = spread.loc[chosen_id, "spread_wmean"]
    ax.set(aspect="equal", title=f"{ctype} ({sign})\n{len(col_syn)} cols, spread={spread_val:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(sc, ax=axes, shrink=0.65, location="right", label="syn count")
plt.suptitle("Single-cell output footprint on hex column lattice (x = weighted centroid)",
             y=1.04, fontsize=11)
save(fig, "fig4_hex_footprint.png")


# --------------------------------------------------------------------------
# Fig 5: Q6 Dm8 input footprint
# --------------------------------------------------------------------------
print("\nFig 5: Q6 Dm8 input footprint")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, dm in zip(axes, ["Dm8a", "Dm8b"]):
    inn = conn[(conn["post_primary_type"] == dm) & (conn["pre_primary_type"] == "R7")].copy()
    inn["p_pre"] = inn["pre_root_id"].map(col_map["p"])
    inn["q_pre"] = inn["pre_root_id"].map(col_map["q"])
    inn["hemi_pre"] = inn["pre_root_id"].map(col_map["hemisphere"])
    inn["hemi_post"] = inn["post_root_id"].map(pre_side_map)
    inn = inn.dropna(subset=["p_pre", "q_pre", "hemi_post"])
    inn = inn[inn["hemi_pre"] == inn["hemi_post"]]
    if len(inn) == 0:
        ax.set_axis_off()
        continue
    inn["col_tup"] = list(zip(inn["p_pre"].astype(int), inn["q_pre"].astype(int)))
    n_per = inn.groupby("post_root_id").agg(
        n_cols=("col_tup", "nunique"), syn=("syn_count", "sum"))
    chosen_id = n_per.sort_values(["n_cols", "syn"], ascending=False).index[0]
    chosen_edges = inn[inn["post_root_id"] == chosen_id]
    col_syn = chosen_edges.groupby(["p_pre", "q_pre"], as_index=False)["syn_count"].sum()
    hemi = chosen_edges["hemi_post"].iloc[0]
    bg = col_assign[col_assign["hemisphere"] == hemi].drop_duplicates(["p", "q"])[["p", "q"]]
    bx, by = axial_to_cart(bg["p"].values, bg["q"].values)
    ax.scatter(bx, by, c="lightgray", s=70, marker="H", alpha=0.45, linewidths=0)
    sx, sy = axial_to_cart(col_syn["p_pre"].values, col_syn["q_pre"].values)
    sc = ax.scatter(sx, sy, c=col_syn["syn_count"], cmap="hot_r", s=100, marker="H",
                    edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="R7 syn -> this Dm8", shrink=0.7)
    ax.set(aspect="equal", title=f"{dm}: R7 input from {len(col_syn)} columns")
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("Single Dm8 cell: R7 (UV photoreceptor) input column footprint",
             y=1.02, fontsize=11)
plt.tight_layout()
save(fig, "fig5_dm8_input_footprint.png")


# --------------------------------------------------------------------------
# Q7 edge analysis on Mi1
# --------------------------------------------------------------------------
print("\nQ7 setup: edge categorization for Mi1")
mi1_r = col_assign[(col_assign["type"] == "Mi1") & (col_assign["hemisphere"] == "right")]
mi1_pq = mi1_r[["root_id", "p", "q"]].copy()
all_pq = set(zip(mi1_pq["p"], mi1_pq["q"]))
hex_nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
mi1_pq["n_neighbors"] = mi1_pq.apply(
    lambda r: sum((r["p"] + dp, r["q"] + dq) in all_pq for dp, dq in hex_nbrs), axis=1)
mi1_pq["edge_cat"] = pd.cut(mi1_pq["n_neighbors"], bins=[-1, 3, 5, 6],
                            labels=["corner", "edge", "interior"])
mi1_ids = set(mi1_pq["root_id"])
incoming = conn[conn["post_root_id"].isin(mi1_ids)]
sbs = (incoming.groupby(["post_root_id", "sign"])["syn_count"].sum()
       .unstack(fill_value=0)
       .reindex(columns=["inh", "exc", "other"], fill_value=0))
mi1_pq = mi1_pq.set_index("root_id").join(sbs, how="left").reset_index()
for c in ["inh", "exc", "other"]:
    mi1_pq[c] = mi1_pq[c].fillna(0).astype(int)
mi1_pq["inh_frac"] = mi1_pq["inh"] / (mi1_pq["inh"] + mi1_pq["exc"]).clip(lower=1)


# --------------------------------------------------------------------------
# Fig 6: Q7 edge effect box plot for Mi1
# --------------------------------------------------------------------------
print("\nFig 6: Q7 edge effect on Mi1")
cats = ["corner", "edge", "interior"]
colors_cat = ["tab:red", "tab:orange", "tab:blue"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
configs = [
    ("inh", "inh syn per Mi1", "Inhibitory input"),
    ("exc", "exc syn per Mi1", "Excitatory input"),
    ("inh_frac", "inh / (inh + exc)", "E/I balance (preserved)"),
]
for ax, (col, ylabel, title) in zip(axes, configs):
    data = [mi1_pq[mi1_pq["edge_cat"] == c][col].values for c in cats]
    bp = ax.boxplot(data, tick_labels=cats, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors_cat):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    medians = [float(np.median(d)) for d in data]
    for x, m_val in enumerate(medians, 1):
        txt = f"  {m_val:.2f}" if col == "inh_frac" else f"  {m_val:.0f}"
        ax.text(x, m_val, txt, va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set(ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3, axis="y")
    if col == "inh_frac":
        ax.set_ylim(0, 1)
plt.suptitle("Q7: Mi1 receives less inh AND exc at edge, but E/I balance preserved",
             y=1.04, fontsize=12)
plt.tight_layout()
save(fig, "fig6_edge_effect_mi1.png")


# --------------------------------------------------------------------------
# Fig 7: Q8 family-level survey
# --------------------------------------------------------------------------
print("\nFig 7: Q8 family-level inh interneuron survey")


def family_of(t):
    t = str(t)
    if t == "CT1":
        return "CT1 (global)"
    if t == "Lai":
        return "Lai (lamina amacrine)"
    if t == "T1":
        return "T1"
    if t.startswith("Lawf"):
        return "Lawf"
    if t.startswith("DmDRA") or t.startswith("Dm"):
        return "Dm (distal medulla)"
    if t.startswith("Pm"):
        return "Pm (proximal medulla)"
    if t.startswith("Sm"):
        return "Sm (small medulla)"
    if t.startswith("LPi"):
        return "LPi"
    if t.startswith("LPLC") or t.startswith("LC"):
        return "LC/LPLC"
    if t.startswith("Lat"):
        return "Lat"
    if t.startswith("LT"):
        return "LT"
    if t.startswith("Li"):
        return "Li"
    if t.startswith("Mi"):
        return "Mi (medulla intrinsic)"
    if t.startswith("TmY"):
        return "TmY"
    if t.startswith("Tm"):
        return "Tm"
    if t in ("C2", "C3"):
        return "C2/C3"
    if t.startswith("MeTu"):
        return "MeTu"
    if t.startswith("MeMe"):
        return "MeMe"
    return "other"


inh_survey = type_io[(type_io["inh_frac"] >= 0.5) & (type_io["inh"] >= 1000)].copy()
inh_survey["family"] = inh_survey.index.map(family_of)
rows = []
for ctype in inh_survey.index:
    rows.append({
        "type": ctype,
        "family": inh_survey.loc[ctype, "family"],
        "inh_syn": int(inh_survey.loc[ctype, "inh"]),
        "col_spread": float(per_type_col.loc[ctype, "type_spread"])
        if ctype in per_type_col.index else float("nan"),
    })
inh_detail = pd.DataFrame(rows).sort_values("inh_syn", ascending=False)

topN = inh_detail.head(25).copy()
family_list = list(topN["family"].unique())
palette = plt.cm.tab20(np.linspace(0, 1, max(len(family_list), 3)))
family_colors = dict(zip(family_list, palette))
colors_topN = [family_colors[f] for f in topN["family"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.barh(range(len(topN)), topN["inh_syn"], color=colors_topN, edgecolor="white")
ax.set_yticks(range(len(topN)))
ax.set_yticklabels(topN["type"], fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="total inh syn output",
       title=f"Top {len(topN)} inhibitory cell types (colored by family)")
from matplotlib.patches import Patch
handles = [Patch(facecolor=family_colors[f], label=f) for f in family_list]
ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)

ax = axes[1]
fam_summary = (
    inh_detail.groupby("family")
    .agg(n=("type", "size"),
         total_inh=("inh_syn", "sum"),
         median_spread=("col_spread", "median"))
    .sort_values("total_inh", ascending=False)
)
data_per_fam = [
    inh_detail[inh_detail["family"] == f]["col_spread"].dropna().values
    for f in fam_summary.index
]
valid = [(f, d) for f, d in zip(fam_summary.index, data_per_fam) if len(d) > 0]
labels = [f + f"\n(n={len(d)})" for f, d in valid]
data_valid = [d for _, d in valid]
colors_valid = [family_colors.get(f, "lightgray") for f, _ in valid]
bp = ax.boxplot(data_valid, tick_labels=labels, patch_artist=True,
                showfliers=False, vert=False)
for patch, c in zip(bp["boxes"], colors_valid):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set(xlabel="column spread (hex units)",
       title="Column spread distribution by family")
ax.grid(True, alpha=0.3, axis="x")
plt.suptitle("Q8: comprehensive survey of inhibitory interneurons",
             y=1.01, fontsize=12)
plt.tight_layout()
save(fig, "fig7_inh_family_survey.png")


# --------------------------------------------------------------------------
# Fig 8: Q9 multi-cell-type edge analysis
# --------------------------------------------------------------------------
print("\nFig 8: Q9 multi-cell-type edge analysis")
import scipy.stats as st


def edge_stats_for_type(cell_type, hemisphere="right", min_cells=30):
    cells = col_assign[(col_assign["type"] == cell_type) & (col_assign["hemisphere"] == hemisphere)]
    if len(cells) < min_cells:
        return None
    cpq = cells[["root_id", "p", "q"]].copy()
    all_pq_t = set(zip(cpq["p"], cpq["q"]))
    cpq["n_neighbors"] = cpq.apply(
        lambda r: sum((r["p"] + dp, r["q"] + dq) in all_pq_t for dp, dq in hex_nbrs), axis=1)
    cpq["edge_cat"] = pd.cut(cpq["n_neighbors"], bins=[-1, 3, 5, 6],
                             labels=["corner", "edge", "interior"])
    cids = set(cpq["root_id"])
    inc = conn[conn["post_root_id"].isin(cids)]
    sbs_t = (inc.groupby(["post_root_id", "sign"])["syn_count"].sum()
             .unstack(fill_value=0)
             .reindex(columns=["inh", "exc", "other"], fill_value=0))
    cpq = cpq.set_index("root_id").join(sbs_t, how="left").reset_index()
    for c in ["inh", "exc", "other"]:
        cpq[c] = cpq[c].fillna(0).astype(int)
    cpq["inh_frac"] = cpq["inh"] / (cpq["inh"] + cpq["exc"]).clip(lower=1)
    return cpq


target_types = ["Mi1", "Mi4", "Mi9", "L1", "L2", "L3", "L4", "L5",
                "Tm1", "Tm2", "Tm3", "Tm9", "Tm20", "Tm21",
                "T1", "T2", "T2a", "T3", "C2", "C3",
                "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
rows = []
for t in target_types:
    df = edge_stats_for_type(t)
    if df is None:
        continue
    sub_c = df[df["edge_cat"] == "corner"]
    sub_i = df[df["edge_cat"] == "interior"]
    if len(sub_c) < 5 or len(sub_i) < 30:
        continue
    rows.append({
        "cell_type": t,
        "inh_corner/interior": float(sub_c["inh"].median() / max(sub_i["inh"].median(), 1)),
        "exc_corner/interior": float(sub_c["exc"].median() / max(sub_i["exc"].median(), 1)),
        "corner_inh_frac": float(sub_c["inh_frac"].median()),
        "interior_inh_frac": float(sub_i["inh_frac"].median()),
    })
scaling_df = pd.DataFrame(rows).sort_values("inh_corner/interior")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
xs = scaling_df["exc_corner/interior"].values
ys = scaling_df["inh_corner/interior"].values
ax.scatter(xs, ys, s=45, alpha=0.7, color="tab:purple",
           edgecolors="black", linewidths=0.5)
maxv = max(xs.max(), ys.max(), 1.05) * 1.05
ax.plot([0, maxv], [0, maxv], "k--", lw=0.7, alpha=0.5, label="y = x (proportional)")
ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
ax.axvline(1.0, ls=":", color="gray", alpha=0.5)
for _, r in scaling_df.iterrows():
    ax.annotate(r["cell_type"], (r["exc_corner/interior"], r["inh_corner/interior"]),
                fontsize=6.5, alpha=0.85, xytext=(2, 2), textcoords="offset points")
ax.set(xlim=(0, maxv), ylim=(0, maxv),
       xlabel="exc corner/interior ratio",
       ylabel="inh corner/interior ratio",
       title="Edge scaling: inh vs exc per cell type")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
d_sorted = scaling_df.sort_values("inh_corner/interior", ascending=False)
colors_bar = ["tab:green" if r > 0.9 else ("tab:orange" if r > 0.6 else "tab:red")
              for r in d_sorted["inh_corner/interior"]]
ax.barh(range(len(d_sorted)), d_sorted["inh_corner/interior"],
        color=colors_bar, edgecolor="white")
ax.set_yticks(range(len(d_sorted)))
ax.set_yticklabels(d_sorted["cell_type"], fontsize=7.5)
ax.invert_yaxis()
ax.axvline(1.0, color="black", ls="--", alpha=0.5, label="no edge effect")
ax.set(xlabel="corner/interior inh ratio",
       title="Edge sensitivity per cell type (green=<10% drop)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Q9: edge effect across 28 columnar cell types", y=1.02, fontsize=12)
plt.tight_layout()
save(fig, "fig8_multi_type_edge.png")


print("\n--- done ---")
print(f"figures saved in: {FIG_DIR}")
