"""Regenerate figures for the lateral inhibition report.

Run from the project root::

    uv run python report/lateral_inhibition/generate_figures.py

Output: report/lateral_inhibition/figures/fig*.png (about 9 figures, ~70 s total).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "tab:blue" if x in INHIBITORY_NT else ("tab:red" if x in EXCITATORY_NT else "gray")
    for x in nt_syn.index
]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
nt_syn.plot.barh(ax=ax, color=colors)
ax.set(xlabel="# synapses",
       title="Synapse count by nt_type (red=exc, blue=inh, gray=modulatory/unknown)")
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
type_io["dominant_sign"] = type_io[["inh", "exc", "other"]].idxmax(axis=1)
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
ec_all = ec.copy()
print(f"  observable same-hemi columnar synapses: "
      f"{ec_all['syn_count'].sum() / conn['syn_count'].sum():.1%}")

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
spread["cell_sign_mode"] = spread["nt_type"].map(classify_nt)
spread["sign"] = spread["cell_sign_mode"]  # display label for single-cell panels

per_type_col = (
    spread.groupby("primary_type")
    .agg(
        n_neurons=("spread_wmean", "size"),
        type_spread=("spread_wmean", "median"),
        type_n_cols=("n_unique_cols", "median"),
        cell_sign_mode=("cell_sign_mode", lambda x: x.value_counts().idxmax()),
    )
    .query("n_neurons >= 5")
)
per_type_col = per_type_col.join(
    type_io[["inh", "exc", "other", "total", "inh_frac", "dominant_sign"]],
    how="left",
)
q4_observable_syn_by_type = ec_all.groupby("pre_primary_type")["syn_count"].sum()
total_syn_by_type = conn.groupby("pre_primary_type")["syn_count"].sum()
active_q4_coverage = q4_observable_syn_by_type.div(total_syn_by_type).reindex(active.index).dropna()
print(f"  active types with observable Q4 synapses: "
      f"{len(active_q4_coverage)} / {len(active)}, "
      f"median coverage={active_q4_coverage.median():.1%}")


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
             label=f"exc (n={len(exc_t)})", color="tab:red")
axes[0].hist(inh_t["type_spread"], bins=bins, alpha=0.5, density=True,
             label=f"inh (n={len(inh_t)})", color="tab:blue")
axes[0].axvline(exc_t["type_spread"].median(), color="tab:red", ls="--", lw=0.8)
axes[0].axvline(inh_t["type_spread"].median(), color="tab:blue", ls="--", lw=0.8)
axes[0].set(xlabel="weighted-mean Delta-column from centroid (hex)",
            ylabel="density",
            title=f"Lateral spread per cell type (inh/exc ratio = {inh_t['type_spread'].median()/exc_t['type_spread'].median():.2f})")
axes[0].legend(fontsize=8)

upper2 = float(np.percentile(np.concatenate([inh_t["type_n_cols"], exc_t["type_n_cols"]]), 99))
bins2 = np.linspace(0, upper2, 25)
axes[1].hist(exc_t["type_n_cols"], bins=bins2, alpha=0.5, density=True,
             label=f"exc (n={len(exc_t)})", color="tab:red")
axes[1].hist(inh_t["type_n_cols"], bins=bins2, alpha=0.5, density=True,
             label=f"inh (n={len(inh_t)})", color="tab:blue")
axes[1].axvline(exc_t["type_n_cols"].median(), color="tab:red", ls="--", lw=0.8)
axes[1].axvline(inh_t["type_n_cols"].median(), color="tab:blue", ls="--", lw=0.8)
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


excluded_edge_types = {"R7", "R8"}
target_types = sorted(
    t for t in col_assign["type"].dropna().unique()
    if t not in excluded_edge_types
)
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

plt.suptitle(f"Q9: edge effect across {len(scaling_df)} columnar cell types",
             y=1.02, fontsize=12)
plt.tight_layout()
save(fig, "fig8_multi_type_edge.png")


# --------------------------------------------------------------------------
# Fig 9: Q9 supplement -- lateral inhibition vs bottom-up excitation per column
# --------------------------------------------------------------------------
# Q7's E/I balance lumped home-column feedforward drive into "exc". Here the two
# functional streams are separated and their ratio mapped in real hex space:
#   bottom-up exc = home-column (Delta=0) excitatory syn (feedforward drive, e.g. L1->Mi1)
#   lateral inh   = total inhibitory syn onto the column (sign-based, full coverage;
#                   columnar cells are inhibited almost entirely by wide-field surround)
# R = lateral_inh / bottom_up_exc. A rim that brightens => surround inhibition is
# relatively stronger than feedforward at the edge (compensation); darker => weaker.
print("\nFig 9: Q9 supplement -- lateral inh / bottom-up exc per column")
hemi = "right"

# bottom-up excitation: home-column (same (p, q) as the receiver) exc syn per receiver cell
conn_exc = conn[conn["sign"] == "exc"]
foot_rows = []
for t in target_types:
    cells = col_assign[(col_assign["type"] == t) & (col_assign["hemisphere"] == hemi)]
    if len(cells) < 30:
        continue
    home = cells[["root_id", "p", "q"]].drop_duplicates("root_id").set_index("root_id")
    inc = conn_exc[conn_exc["post_root_id"].isin(set(home.index))].copy()
    inc["p_src"] = inc["pre_root_id"].map(col_map["p"])
    inc["q_src"] = inc["pre_root_id"].map(col_map["q"])
    inc["hemi_src"] = inc["pre_root_id"].map(col_map["hemisphere"])
    inc = inc.dropna(subset=["p_src", "q_src"])
    inc = inc[inc["hemi_src"] == hemi]
    if len(inc) == 0:
        continue
    inc["dp"] = (inc["p_src"] - inc["post_root_id"].map(home["p"])).astype(int)
    inc["dq"] = (inc["q_src"] - inc["post_root_id"].map(home["q"])).astype(int)
    foot_rows.append(inc[["post_root_id", "dp", "dq", "syn_count"]].assign(type=t))
foot = pd.concat(foot_rows, ignore_index=True)
bottom_up = (foot[(foot["dp"] == 0) & (foot["dq"] == 0)]
             .groupby(["type", "post_root_id"])["syn_count"].sum()
             .reset_index().rename(columns={"syn_count": "bottom_up_exc"}))

# lateral inhibition: total inh syn onto each columnar receiver (sign-based, full coverage)
recv = (col_assign[(col_assign["hemisphere"] == hemi) & col_assign["type"].isin(target_types)]
        .drop_duplicates("root_id")[["root_id", "type", "p", "q"]].copy())
inh_tot = (conn[conn["post_root_id"].isin(set(recv["root_id"])) & (conn["sign"] == "inh")]
           .groupby("post_root_id")["syn_count"].sum())
recv["lateral_inh"] = recv["root_id"].map(inh_tot).fillna(0.0)
recv = recv.merge(bottom_up.rename(columns={"post_root_id": "root_id"}),
                  on=["type", "root_id"], how="left")
recv["bottom_up_exc"] = recv["bottom_up_exc"].fillna(0.0)
# columns with negligible feedforward drive give an unstable ratio -> NaN (blank hex)
recv["ratio"] = np.where(recv["bottom_up_exc"] >= 5,
                         recv["lateral_inh"] / recv["bottom_up_exc"].clip(lower=1), np.nan)
recv["x"], recv["y"] = axial_to_cart(recv["p"].values, recv["q"].values)
print(f"  columns with usable ratio: {recv['ratio'].notna().sum():,} / {len(recv):,}")
print(f"  R (lateral_inh / bottom_up_exc) median = {recv['ratio'].median():.2f}, "
      f"IQR = [{recv['ratio'].quantile(.25):.2f}, {recv['ratio'].quantile(.75):.2f}]")

types_c = sorted(recv["type"].unique())
ncol = 6
nrow = int(np.ceil(len(types_c) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, t in zip(axes, types_c):
    gg = recv[recv["type"] == t]
    vmx = float(np.nanpercentile(gg["ratio"], 95)) if gg["ratio"].notna().any() else 1.0
    ax.scatter(gg["x"], gg["y"], c=gg["ratio"], cmap="magma", s=16, marker="H",
               edgecolors="none", vmin=0, vmax=vmx)
    ax.set(aspect="equal", title=f"{t} (n={gg['ratio'].notna().sum()})")
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes[len(types_c):]:
    ax.set_axis_off()
plt.suptitle("Q9 supplement: lateral inhibition / bottom-up excitation ratio per column "
             "(all columnar receivers, right hemi)\n"
             "per-panel 95th-pctl colour scale; bright rim = surround inhibition stronger "
             "than feedforward at the edge",
             y=1.01, fontsize=12)
plt.tight_layout()
save(fig, "fig9_lateral_inh_vs_bottomup.png")


# ==========================================================================
# EXTENDED ANALYSES (Q10-Q13): from structure to computation
# ==========================================================================
# Shared per-cell column maps for the column-resolved input analyses (A2, A1).
_P = col_map["p"].to_dict()
_Q = col_map["q"].to_dict()
_HM = col_map["hemisphere"].to_dict()


def _interior_cells(cell_type, hemisphere, min_nbrs=5):
    cells = col_assign[(col_assign["type"] == cell_type)
                       & (col_assign["hemisphere"] == hemisphere)].drop_duplicates("root_id")
    s = set(zip(cells["p"], cells["q"]))
    nn = [sum((p + dp, q + dq) in s for dp, dq in hex_nbrs) for p, q in zip(cells["p"], cells["q"])]
    return cells.assign(n_neighbors=nn).query("n_neighbors >= @min_nbrs").set_index("root_id")


# --------------------------------------------------------------------------
# Fig 10: Q10 (A2) T4/T5 input spatial offset -> direction selectivity
# --------------------------------------------------------------------------
print("\nFig 10: Q10 (A2) T4/T5 input spatial offset")


def _per_cell_centroids(subtype, src_types, hemi):
    cells = _interior_cells(subtype, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    inc = conn[conn["post_root_id"].isin(set(cells.index))
               & conn["pre_primary_type"].isin(src_types)].copy()
    inc["sp"] = inc["pre_root_id"].map(_P)
    inc["sq"] = inc["pre_root_id"].map(_Q)
    inc["sh"] = inc["pre_root_id"].map(_HM)
    inc = inc.dropna(subset=["sp", "sq"])
    inc = inc[inc["sh"] == hemi]
    inc["dp"] = inc["sp"] - inc["post_root_id"].map(hp)
    inc["dq"] = inc["sq"] - inc["post_root_id"].map(hq)
    inc["wp"] = inc["dp"] * inc["syn_count"]
    inc["wq"] = inc["dq"] * inc["syn_count"]
    g = inc.groupby(["post_root_id", "pre_primary_type"]).agg(
        wp=("wp", "sum"), wq=("wq", "sum"), w=("syn_count", "sum"))
    g["cdp"] = g["wp"] / g["w"]
    g["cdq"] = g["wq"] / g["w"]
    return g[["cdp", "cdq"]].unstack("pre_primary_type")


def _circ_stats(ang):
    n = len(ang)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    C, S = np.cos(ang).sum(), np.sin(ang).sum()
    R = np.hypot(C, S) / n
    mean = np.degrees(np.arctan2(S, C)) % 360
    Z = n * R * R
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n)
                      - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n * n))
    return mean, R, float(min(max(p, 0.0), 1.0)), n


def _dipole_angles(subtype, src_types, hemi, arm_pos, arm_neg):
    cen = _per_cell_centroids(subtype, src_types, hemi)
    need = [("cdp", arm_pos), ("cdq", arm_pos), ("cdp", arm_neg), ("cdq", arm_neg)]
    if any(c not in cen.columns for c in need):
        return None
    sub = cen[need].dropna()
    px, py = axial_to_cart(sub[("cdp", arm_pos)].values, sub[("cdq", arm_pos)].values)
    nx, ny = axial_to_cart(sub[("cdp", arm_neg)].values, sub[("cdq", arm_neg)].values)
    return np.arctan2(py - ny, px - nx)


def _angdiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


A2_CONFIG = {
    "T4 (ON)": dict(subs=["T4a", "T4b", "T4c", "T4d"], srcs=["Mi1", "Tm3", "Mi9", "Mi4"], pos="Mi9", neg="Mi4"),
    "T5 (OFF)": dict(subs=["T5a", "T5b", "T5c", "T5d"], srcs=["Tm1", "Tm2", "Tm4", "Tm9"], pos="Tm9", neg="Tm2"),
}
sub_colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
fig, axes = plt.subplots(2, 2, figsize=(11, 11), subplot_kw=dict(projection="polar"))
for i_pw, pathway in enumerate(A2_CONFIG):
    cfg = A2_CONFIG[pathway]
    for j_h, hemi in enumerate(["right", "left"]):
        ax = axes[i_pw, j_h]
        means = {}
        for sub in cfg["subs"]:
            ang = _dipole_angles(sub, cfg["srcs"], hemi, cfg["pos"], cfg["neg"])
            if ang is None or len(ang) < 3:
                continue
            mean, R, p, n = _circ_stats(ang)
            means[sub] = mean
            c = sub_colors[sub[-1]]
            counts, edges = np.histogram(ang % (2 * np.pi), bins=24, range=(0, 2 * np.pi))
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(np.append(centers, centers[0]), np.append(counts, counts[0]),
                    color=c, alpha=0.55, lw=1.3, label=f"{sub} (R={R:.2f}, n={n})")
            ax.annotate("", xy=(np.radians(mean), max(counts) * (R + 0.15)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
        ax.set_title(f"{pathway}  {hemi}\ndipole = {cfg['pos']}-{cfg['neg']}", fontsize=10)
        ax.set_theta_zero_location("E")
        ax.set_yticklabels([])
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.16, 1.12))
        if len(means) == 4:
            sb = cfg["subs"]
            d_ab = _angdiff(means[sb[0]], means[sb[1]])
            d_cd = _angdiff(means[sb[2]], means[sb[3]])
            d_axes = _angdiff((means[sb[0]] % 180) * 2, (means[sb[2]] % 180) * 2) / 2
            print(f"  {pathway} {hemi}: {sb[0]}|{sb[1]} d={d_ab:.0f}, {sb[2]}|{sb[3]} d={d_cd:.0f} "
                  f"(~180); axes d={d_axes:.0f} (~90)")
plt.suptitle("Q10 (A2): per-cell input dipole angle rotates with T4/T5 subtype (preferred direction)\n"
             "lines = per-subtype angle histogram; arrows = circular mean x resultant R", y=1.0, fontsize=11)
save(fig, "fig10_t4t5_offset.png")


# --------------------------------------------------------------------------
# Fig 11: Q11 (A1) center-surround receptive field (Mexican-hat)
# --------------------------------------------------------------------------
print("\nFig 11: Q11 (A1) center-surround receptive field")


def _direct_kernel(target, sign, hemi="right"):
    cells = _interior_cells(target, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    inc = conn[conn["post_root_id"].isin(set(cells.index)) & (conn["sign"] == sign)].copy()
    inc["sp"] = inc["pre_root_id"].map(_P)
    inc["sq"] = inc["pre_root_id"].map(_Q)
    inc["sh"] = inc["pre_root_id"].map(_HM)
    inc = inc.dropna(subset=["sp", "sq"])
    inc = inc[inc["sh"] == hemi]
    dp = (inc["sp"] - inc["post_root_id"].map(hp)).astype(int)
    dq = (inc["sq"] - inc["post_root_id"].map(hq)).astype(int)
    inc["d"] = hex_distance(dp, dq)
    return inc.groupby("d")["syn_count"].sum()


def _disyn_inh_kernel(target, hemi="right", top_inh_types=12):
    cells = _interior_cells(target, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    inh_to_t = conn[conn["post_root_id"].isin(set(cells.index)) & (conn["sign"] == "inh")].copy()
    top_types = (inh_to_t.groupby("pre_primary_type")["syn_count"].sum()
                 .sort_values(ascending=False).head(top_inh_types).index)
    inh_to_t = (inh_to_t[inh_to_t["pre_primary_type"].isin(top_types)]
                [["pre_root_id", "post_root_id", "syn_count"]].rename(columns={"syn_count": "w1"}))
    jin = conn[conn["post_root_id"].isin(set(inh_to_t["pre_root_id"])) & (conn["sign"] == "exc")].copy()
    jin["sp"] = jin["pre_root_id"].map(_P)
    jin["sq"] = jin["pre_root_id"].map(_Q)
    jin["sh"] = jin["pre_root_id"].map(_HM)
    jin = jin.dropna(subset=["sp", "sq"])
    jin = jin[jin["sh"] == hemi]
    jin["w2n"] = jin["syn_count"] / jin.groupby("post_root_id")["syn_count"].transform("sum")
    jin = jin.rename(columns={"post_root_id": "j"})[["j", "sp", "sq", "w2n"]]
    mg = inh_to_t.merge(jin, left_on="pre_root_id", right_on="j", how="inner")
    dp = (mg["sp"] - mg["post_root_id"].map(hp)).astype(int)
    dq = (mg["sq"] - mg["post_root_id"].map(hq)).astype(int)
    mg["d"] = hex_distance(dp, dq)
    mg["w"] = mg["w1"] * mg["w2n"]
    return mg.groupby("d")["w"].sum()


def _norm_kernel(kern, maxd=8):
    s = kern.reindex(range(0, maxd + 1), fill_value=0).astype(float)
    s = s / s.sum() if s.sum() > 0 else s
    d = np.arange(0, maxd + 1)
    return s, float(np.sqrt((s.values * d ** 2).sum()))


targets_a1 = ["Mi1", "Tm9", "L2", "T4a"]
maxd = 8
fig, axes = plt.subplots(2, len(targets_a1), figsize=(3.6 * len(targets_a1), 7))
rr = np.arange(0, maxd + 1)
for k, target in enumerate(targets_a1):
    e, sig_c = _norm_kernel(_direct_kernel(target, "exc"), maxd)
    iv, sig_s = _norm_kernel(_disyn_inh_kernel(target), maxd)
    ax = axes[0, k]
    ax.plot(rr, e.values, "-o", color="tab:red", label=f"exc E (s={sig_c:.2f})")
    ax.plot(rr, iv.values, "--^", color="tab:blue", label=f"inh surround I (s={sig_s:.2f})")
    ax.set(xlabel="Delta-column (hex)", ylabel="fraction of input", title=target)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax = axes[1, k]
    f = np.linspace(0, 0.5, 200)
    amp_c = float(e.iloc[0]) or 1.0
    amp_s = float(iv.iloc[0]) or 1.0
    sc_, ss_ = max(sig_c, 0.3), max(sig_s, sig_c + 0.1)
    resp = (amp_c * sc_ ** 2 * np.exp(-2 * np.pi ** 2 * sc_ ** 2 * f ** 2)
            - amp_s * ss_ ** 2 * np.exp(-2 * np.pi ** 2 * ss_ ** 2 * f ** 2))
    resp = resp / np.abs(resp).max()
    ax.plot(f, resp, color="tab:green", lw=2)
    ax.axhline(0, color="gray", lw=0.6)
    peak_f = f[np.argmax(resp)]
    ax.axvline(peak_f, color="tab:red", ls=":", label=f"peak~{peak_f:.2f} cyc/col")
    ax.set(xlabel="spatial freq (cycles/column)", ylabel="predicted resp (norm)",
           title=f"{target}: DoG bandpass")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.suptitle("Q11 (A1): excitation is center-weighted, disynaptic inhibition forms a broad surround\n"
             "(top) received radial profiles; (bottom) Difference-of-Gaussians -> bandpass spatial-frequency tuning",
             y=1.02, fontsize=11)
save(fig, "fig11_center_surround.png")


# --------------------------------------------------------------------------
# Fig 12: Q12 (B1) inhibitory microcircuit motif census
# --------------------------------------------------------------------------
print("\nFig 12: Q12 (B1) inhibitory motif census")
dom = type_io["dominant_sign"]
inh_b1 = conn[conn["sign"] == "inh"].copy()
inh_b1["post_dom"] = inh_b1["post_primary_type"].map(dom)
by_post = inh_b1.groupby("post_dom")["syn_count"].sum()
tot_inh = by_post.sum()
print(f"  disinhibition: {by_post.get('inh', 0) / tot_inh:.1%} of inhibition lands on inh-dominant cells")

te = conn[conn["sign"] == "inh"].groupby(["pre_primary_type", "post_primary_type"])["syn_count"].sum()
te_map = {(a, b): s for (a, b), s in te.items()}
seen, recip = set(), []
for (a, b), s_ab in te_map.items():
    if a == b or (b, a) in seen:
        continue
    seen.add((a, b))
    s_ba = te_map.get((b, a), 0)
    if s_ab >= 500 and s_ba >= 500:
        recip.append((a, b, int(min(s_ab, s_ba))))
recip_df = pd.DataFrame(recip, columns=["A", "B", "min_syn"]).sort_values("min_syn", ascending=False)
print(f"  reciprocal-inhibition type-pairs (both >=500 inh syn): {len(recip_df)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
d = recip_df.head(15).iloc[::-1]
axes[0].barh(range(len(d)), d["min_syn"], color="tab:purple", edgecolor="white")
axes[0].set_yticks(range(len(d)))
axes[0].set_yticklabels([f"{a}<->{b}" for a, b in zip(d["A"], d["B"])], fontsize=8)
axes[0].set(xlabel="min(A->B, B->A) inh syn", title="Reciprocal inhibition: top type-pairs")
axes[0].grid(True, alpha=0.3, axis="x")
vals = [by_post.get("exc", 0), by_post.get("inh", 0), by_post.get("other", 0)]
axes[1].bar(["onto exc-dom\n(FF/FB inh)", "onto inh-dom\n(disinhibition)", "onto other"], vals,
            color=["tab:red", "tab:blue", "gray"], edgecolor="white")
for x_pos, v in enumerate(vals):
    axes[1].text(x_pos, v, f"{v / tot_inh:.1%}", ha="center", va="bottom", fontweight="bold")
axes[1].set(ylabel="total inhibitory synapses", title="Where does inhibition land?")
axes[1].grid(True, alpha=0.3, axis="y")
plt.suptitle("Q12 (B1): inhibitory microcircuit motif census", y=1.02, fontsize=12)
plt.tight_layout()
save(fig, "fig12_motif_census.png")


# --------------------------------------------------------------------------
# Fig 13: Q13 (B2) M-layer (depth) atlas of inhibition
# --------------------------------------------------------------------------
print("\nFig 13: Q13 (B2) M-layer depth atlas")
_neur = neurons.drop_duplicates("root_id").set_index("root_id")
_type, _side = _neur["primary_type"], _neur["side"]
CURATED = {"Dm1": "inh", "Dm4": "inh", "Dm9": "inh", "Dm12": "inh",
           "Pm04": "inh", "Pm08": "inh", "Pm09": "inh", "CT1": "inh", "Mi4": "inh", "Mi9": "inh",
           "Mi1": "exc-ref", "L5": "exc-ref", "Tm3": "exc-ref", "Tm9": "exc-ref"}
right_curated = set(_neur.index[(_side == "right") & (_type.isin(CURATED))])
mi1_right_ids = set(_neur.index[(_side == "right") & (_type == "Mi1")])
syn_path = Path(DATA_DIR) / "raw" / "flywire" / "csv" / "synapse_coordinates.csv"
t_b2 = time.perf_counter()
syndf = pd.read_csv(syn_path, dtype={"pre_root_id": str}, usecols=["pre_root_id", "x", "y", "z"])
syndf["pre_root_id"] = syndf["pre_root_id"].ffill()
syndf = syndf.dropna(subset=["pre_root_id"])
syndf = syndf[syndf["pre_root_id"].isin(right_curated | mi1_right_ids)].copy()
syndf["ptype"] = syndf["pre_root_id"].map(_type)
print(f"  loaded+filtered {len(syndf):,} right-hemi curated synapses in {time.perf_counter() - t_b2:.1f}s")

mi1_sc = syndf[syndf["pre_root_id"].isin(mi1_right_ids)]
cnt = mi1_sc.groupby("pre_root_id").size()
cents = mi1_sc.groupby("pre_root_id")[["x", "y", "z"]].mean()[cnt >= 20]
Xc = cents.values - cents.values.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
normal, c0 = Vt[-1], cents.values.mean(axis=0)
syndf["depth"] = (syndf[["x", "y", "z"]].values - c0) @ normal
lo, hi = np.percentile(syndf.loc[syndf["ptype"] == "Mi1", "depth"], [1, 99])
lo, hi = min(lo, hi), max(lo, hi)
syndf["rel_depth"] = (syndf["depth"] - lo) / (hi - lo)
if (syndf.loc[syndf["ptype"].isin(["Dm1", "Dm4", "Dm12"]), "rel_depth"].median()
        > syndf.loc[syndf["ptype"].isin(["Pm04", "Pm08", "Pm09"]), "rel_depth"].median()):
    syndf["rel_depth"] = 1 - syndf["rel_depth"]
WIN = (-0.25, 1.25)
order = [t for t in ["L5", "Mi1", "Tm3", "Dm1", "Dm9", "Dm12", "Dm4", "Mi9", "Mi4", "Tm9", "Pm08", "Pm09", "Pm04"]
         if (syndf["ptype"] == t).any()]
order_plot = [t for t in order if syndf.loc[(syndf["ptype"] == t) & syndf["rel_depth"].between(*WIN)].shape[0] >= 50]
print(f"  per-type median depth: "
      f"{ {t: round(float(syndf.loc[(syndf['ptype']==t) & syndf['rel_depth'].between(*WIN), 'rel_depth'].median()), 2) for t in order_plot} }")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_sign = {"inh": "tab:blue", "exc-ref": "tab:red"}
data = [syndf.loc[(syndf["ptype"] == t) & syndf["rel_depth"].between(*WIN), "rel_depth"].values for t in order_plot]
ax = axes[0]
vp = ax.violinplot(data, positions=range(len(order_plot)), vert=False, showmedians=True, widths=0.85)
for kk, body in enumerate(vp["bodies"]):
    body.set_facecolor(colors_sign[CURATED[order_plot[kk]]])
    body.set_alpha(0.45)
ax.set_yticks(range(len(order_plot)))
ax.set_yticklabels([f"{t} ({CURATED[t][:3]})" for t in order_plot], fontsize=9)
ax.set(xlabel="relative medulla depth (0=distal/M1 .. 1=proximal/M10)", xlim=WIN,
       title="(A) Synapse depth by cell type (red=exc ref, blue=inh)")
ax.axvline(0, color="gray", ls=":", alpha=0.5)
ax.axvline(1, color="gray", ls=":", alpha=0.5)
ax.grid(True, alpha=0.3, axis="x")
ax = axes[1]
inh_d = syndf.loc[(syndf["ptype"].map(CURATED) == "inh") & syndf["rel_depth"].between(*WIN), "rel_depth"]
exc_d = syndf.loc[(syndf["ptype"].map(CURATED) == "exc-ref") & syndf["rel_depth"].between(*WIN), "rel_depth"]
bins = np.linspace(WIN[0], WIN[1], 31)
ax.hist(exc_d, bins=bins, density=True, alpha=0.5, color="tab:red", label=f"exc ref (n={len(exc_d):,})")
ax.hist(inh_d, bins=bins, density=True, alpha=0.5, color="tab:blue", label=f"inh (n={len(inh_d):,})")
ax.set(xlabel="relative medulla depth", ylabel="density",
       title="(B) inh vs exc-reference depth (inh has extra distal mode)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.suptitle("Q13 (B2): layer-resolved inhibition in the medulla (Mi1-PCA depth ruler; "
             "global-normal proxy excludes multi-neuropil CT1)", y=1.02, fontsize=11)
plt.tight_layout()
save(fig, "fig13_mlayer_atlas.png")


print("\n--- done ---")
print(f"figures saved in: {FIG_DIR}")
