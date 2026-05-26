# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `column_assignment.csv` の妥当性検証
#
# FlyWire の `column_assignment.csv` (45,061 行 / 796 column / 17+ cell type) は medulla の個眼柱を hex 格子 `(x, y)` または `(p, q)` で指定する。これを **lateral inhibition の物理スケール (Δcolumn) として使う前提が成り立つか** を 2 面から検証する。
#
# 1. **物理座標の妥当性 (Section 2-4)** — column 指定と「シナプス分布の実位置」が整合するか。**soma 座標は unipolar fly neuron では機能部位と独立なので使わず、`synapse_coordinates.csv` のシナプス実位置から centroid を計算する**。同じ column の細胞は物理的に同じ場所にあるはず。
# 2. **接続パターンの妥当性 (Section 5)** — column 指定と既知の retinotopic 回路が整合するか。L1 → Mi1 や Mi1 → T4 のような **同 column 1-to-1 retinotopic 接続** が data 上で Δcolumn = 0 にピークを持つはず。
#
# **注意**: column_assignment は `(column_id, hemisphere)` ペアで一意 (column_id 単体は左右で共有される)。同じ hemisphere 内でだけ距離を計算する必要がある。

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
if (REPO_ROOT / "src").is_dir() is False:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DIR
from src.data import FlyWireDataManager

m = FlyWireDataManager()
neurons = m.optic_lobe_neurons_df
conn = m.optic_lobe_connections_df

col_assign = pd.read_csv(
    Path(DATA_DIR) / "raw" / "flywire" / "csv" / "column_assignment.csv",
    dtype={"root_id": str, "column_id": str},
)
print(f"column_assignment rows: {len(col_assign):,}")
print(f"unique column_id    : {col_assign['column_id'].nunique():,}")
print(f"unique (col_id, hemi): {col_assign.groupby(['column_id', 'hemisphere']).ngroups:,}")
print(f"cell types          : {col_assign['type'].nunique()}")

# %% [markdown]
# ## Section 1. column_assignment の概観
#
# どの cell type が含まれているか、何 neuron / column か、を確認。

# %%
# Per-type count + column count + 1-per-column か?
per_type = col_assign.groupby("type").agg(
    n_neurons=("root_id", "size"),
    n_columns=("column_id", "nunique"),
    n_col_hemi=("column_id", lambda x: col_assign.loc[x.index].groupby(["column_id", "hemisphere"]).ngroups),
)
per_type["neurons_per_col_hemi"] = per_type["n_neurons"] / per_type["n_col_hemi"]
per_type = per_type.sort_values("n_neurons", ascending=False)
print("Per cell type — 1-per-(column, hemisphere) check:")
display(per_type.head(20))

# T4 だけは 4 subtype × 1-per-column = 4 per column (合計)。確認:
t4_cells = col_assign[col_assign["type"].str.startswith("T4")].groupby(["column_id", "hemisphere"]).size()
print(f"\nT4* per (column, hemi): mean={t4_cells.mean():.2f}, mode={t4_cells.mode().iloc[0]}")

# %% [markdown]
# ## Section 2. (x, y) vs (p, q) — どちらが hex 格子か?
#
# 右半球の Mi1 (1 per column) を両座標系で plot して、hex 格子に見える方を採用する。

# %%
mi1_r = col_assign[(col_assign["type"] == "Mi1") & (col_assign["hemisphere"] == "right")]
print(f"Mi1 right-hemi: {len(mi1_r):,}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(mi1_r["x"], mi1_r["y"], s=8, alpha=0.7, color="tab:blue")
axes[0].set(xlabel="x", ylabel="y", aspect="equal", title=f"Mi1 right-hemi in (x, y)  [{mi1_r['x'].min()}..{mi1_r['x'].max()}, {mi1_r['y'].min()}..{mi1_r['y'].max()}]")
axes[1].scatter(mi1_r["p"], mi1_r["q"], s=8, alpha=0.7, color="tab:orange")
axes[1].set(xlabel="p", ylabel="q", aspect="equal", title=f"Mi1 right-hemi in (p, q)  [{mi1_r['p'].min()}..{mi1_r['p'].max()}, {mi1_r['q'].min()}..{mi1_r['q'].max()}]")
plt.tight_layout()

# (x, y) vs (p, q) の相関
corr = mi1_r[["x", "y", "p", "q"]].corr()
print("\nCorrelation (Mi1 right-hemi):")
display(corr.round(3))


# %%
# (p, q) を axial hex coords とみなして hex distance を定義
def hex_distance(dp, dq):
    """Axial hex coords でのマンハッタン距離 (cube projection)"""
    return (np.abs(dp) + np.abs(dp + dq) + np.abs(dq)) // 2

# 隣接判定: Mi1 cells の中で hex_dist = 1 のペアを数えて、6 個眼隣接 (六方格子) になっているか確認
pq = mi1_r[["p", "q"]].to_numpy()
dp = pq[:, None, 0] - pq[None, :, 0]
dq = pq[:, None, 1] - pq[None, :, 1]
dist_mat = hex_distance(dp, dq)
np.fill_diagonal(dist_mat, -1)  # exclude self

neighbors_per_cell = (dist_mat == 1).sum(axis=1)
print(f"Mi1 right-hemi — number of (p,q) hex neighbors per cell")
print(f"  mean = {neighbors_per_cell.mean():.2f}, median = {np.median(neighbors_per_cell):.0f}")
print(f"  histogram: {dict(zip(*np.unique(neighbors_per_cell, return_counts=True)))}")
print("  (六方格子なら 内部 cell は 6, 縁/辺 cell は 3-5)")

# %% [markdown]
# ## Section 3. 物理座標との整合 — Mi1 のシナプス centroid
#
# Mi1 は medulla-intrinsic で 1-per-column なので、出力シナプスの centroid がそのまま「物理 column 位置」を表すはず。column hex coord `(p, q)` と centroid `(px, py, pz)` の対応が滑らかかどうかを見る。

# %%
syn_path = Path(DATA_DIR) / "raw" / "flywire" / "csv" / "synapse_coordinates.csv"
t0 = time.perf_counter()
syn_coords = pd.read_csv(syn_path, dtype={"pre_root_id": str, "post_root_id": str}, usecols=["pre_root_id", "x", "y", "z"])
syn_coords["pre_root_id"] = syn_coords["pre_root_id"].ffill()  # fill-down format
syn_coords = syn_coords.dropna(subset=["pre_root_id"])
print(f"loaded {len(syn_coords):,} synapses in {time.perf_counter()-t0:.1f}s")

# Mi1 (right hemi) の出力シナプスだけ抽出
mi1_r_ids = set(mi1_r["root_id"])
syn_mi1 = syn_coords[syn_coords["pre_root_id"].isin(mi1_r_ids)].copy()
print(f"Mi1 right-hemi outgoing synapses: {len(syn_mi1):,}")

# 各 Mi1 ニューロンの centroid
mi1_centroids = syn_mi1.groupby("pre_root_id")[["x", "y", "z"]].agg(["mean", "size"])
mi1_centroids.columns = ["px", "px_n", "py", "py_n", "pz", "pz_n"]
mi1_centroids = mi1_centroids[mi1_centroids["px_n"] >= 20]
mi1_full = mi1_r.set_index("root_id").join(mi1_centroids[["px", "py", "pz"]], how="inner")
print(f"Mi1 cells with >=20 syn AND column assignment: {len(mi1_full):,}")
display(mi1_full[["p", "q", "px", "py", "pz"]].head())

# %%
# (p, q) を 2D scatter にして、物理座標 (px / py / pz) を色で表示。
# column 指定が物理位置と整合していれば、それぞれの軸で滑らかなグラデーションが見える。
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, comp, label in zip(axes, ["px", "py", "pz"], ["physical x", "physical y", "physical z"]):
    sc = ax.scatter(mi1_full["p"], mi1_full["q"], c=mi1_full[comp], cmap="viridis", s=22)
    plt.colorbar(sc, ax=ax, label=f"{label} (voxel)")
    ax.set(xlabel="p (hex)", ylabel="q (hex)", aspect="equal",
           title=f"Mi1 R: column (p,q) vs {label}")
plt.suptitle("If column_assignment is correct → smooth color gradient in each panel", y=1.02)
plt.tight_layout()

# %%
# 定量: hex distance と物理距離の散布図 + 相関係数
from itertools import combinations

rng = np.random.default_rng(0)
rows = mi1_full.reset_index().to_dict("records")
n = len(rows)
# サンプル: 全ペア n*(n-1)/2 = 1.2M 個は重いので、20k ペアをランダムサンプル
n_sample = 20000
i_idx = rng.integers(0, n, size=n_sample)
j_idx = rng.integers(0, n, size=n_sample)
mask = i_idx != j_idx
i_idx, j_idx = i_idx[mask], j_idx[mask]

p_i = np.array([rows[k]["p"] for k in i_idx]); q_i = np.array([rows[k]["q"] for k in i_idx])
p_j = np.array([rows[k]["p"] for k in j_idx]); q_j = np.array([rows[k]["q"] for k in j_idx])
px_i = np.array([rows[k]["px"] for k in i_idx]); py_i = np.array([rows[k]["py"] for k in i_idx]); pz_i = np.array([rows[k]["pz"] for k in i_idx])
px_j = np.array([rows[k]["px"] for k in j_idx]); py_j = np.array([rows[k]["py"] for k in j_idx]); pz_j = np.array([rows[k]["pz"] for k in j_idx])

hex_d = hex_distance(p_j - p_i, q_j - q_i)
phys_d = np.sqrt((px_j - px_i)**2 + (py_j - py_i)**2 + (pz_j - pz_i)**2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(hex_d, phys_d, s=4, alpha=0.15, color="tab:blue")
# bin 平均
max_hex = int(hex_d.max())
bins = np.arange(0, max_hex + 1)
med_per_bin = [np.median(phys_d[hex_d == b]) if (hex_d == b).any() else np.nan for b in bins]
ax.plot(bins, med_per_bin, color="tab:red", lw=2, marker="o", label="median per hex distance")
ax.set(xlabel="hex distance (column units)", ylabel="physical distance (voxel)",
       title="Mi1 R: hex distance vs physical distance between centroids")
ax.legend()
plt.tight_layout()

from scipy.stats import spearmanr
try:
    rho, p_val = spearmanr(hex_d, phys_d)
    print(f"Spearman rho = {rho:.3f} (p={p_val:.2g})")
except ImportError:
    # numpy corrcoef でも代用
    rho = np.corrcoef(hex_d, phys_d)[0, 1]
    print(f"Pearson r = {rho:.3f} (scipy 未インストール)")


# %% [markdown]
# ## Section 4. 同 column の cell type 間で centroid 位置が一致するか
#
# Mi1 と Mi9 (どちらも ME-intrinsic で 1-per-column) が同じ `(column_id, hemisphere)` に属するなら、両者の物理 centroid は近くにあるはず。

# %%
def centroids_for_type(type_name, hemisphere="right"):
    sub = col_assign[(col_assign["type"] == type_name) & (col_assign["hemisphere"] == hemisphere)]
    ids = set(sub["root_id"])
    s = syn_coords[syn_coords["pre_root_id"].isin(ids)]
    counts = s.groupby("pre_root_id").size()
    cents = s.groupby("pre_root_id")[["x", "y", "z"]].mean()
    cents.columns = ["px", "py", "pz"]  # col_assign の (x, y) と衝突しないよう rename
    cents = cents[counts >= 20]
    return sub.set_index("root_id").join(cents, how="inner")

mi1_c = centroids_for_type("Mi1")
mi9_c = centroids_for_type("Mi9")
print(f"Mi1 R: {len(mi1_c):,}, Mi9 R: {len(mi9_c):,}")

# 各 column について Mi1 と Mi9 の centroid distance
merged = mi1_c[["column_id", "px", "py", "pz"]].merge(
    mi9_c[["column_id", "px", "py", "pz"]],
    on="column_id", suffixes=("_mi1", "_mi9"),
)
print(f"columns with both Mi1 and Mi9: {len(merged):,}")
same_col_dist = np.sqrt((merged["px_mi1"] - merged["px_mi9"])**2 +
                        (merged["py_mi1"] - merged["py_mi9"])**2 +
                        (merged["pz_mi1"] - merged["pz_mi9"])**2)

# 比較: ランダム Mi1-Mi9 ペア
rng = np.random.default_rng(0)
mi1_arr = mi1_c[["px", "py", "pz"]].to_numpy()
mi9_arr = mi9_c[["px", "py", "pz"]].to_numpy()
i_idx = rng.integers(0, len(mi1_arr), 10000)
j_idx = rng.integers(0, len(mi9_arr), 10000)
random_dist = np.linalg.norm(mi1_arr[i_idx] - mi9_arr[j_idx], axis=1)

print(f"\nSame-column Mi1-Mi9 distance:  median={np.median(same_col_dist):,.0f}, mean={same_col_dist.mean():,.0f}")
print(f"Random Mi1-Mi9 distance:        median={np.median(random_dist):,.0f}, mean={random_dist.mean():,.0f}")
print(f"ratio (same-col / random)     = {np.median(same_col_dist) / np.median(random_dist):.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
upper = float(np.percentile(np.concatenate([same_col_dist, random_dist]), 99))
bins = np.linspace(0, upper, 60)
ax.hist(random_dist, bins=bins, alpha=0.5, density=True, label=f"random pair (n={len(random_dist)})", color="tab:gray")
ax.hist(same_col_dist, bins=bins, alpha=0.7, density=True, label=f"same-column (n={len(same_col_dist)})", color="tab:green")
ax.set(xlabel="Mi1 ↔ Mi9 centroid distance (voxel)", ylabel="density",
       title="Same-column Mi1-Mi9 should be MUCH closer than random pairs")
ax.legend()
plt.tight_layout()


# %% [markdown]
# ## Section 5. 接続パターンの妥当性 — L1 → Mi1 retinotopic 検証
#
# L1 と Mi1 は同 column で 1-to-1 接続する古典的 retinotopic 経路。各 (L1 pre, Mi1 post) 接続について `(p, q)` の hex distance を計算し、syn_count の分布を見る。**Δ column = 0 にピークがあれば column_assignment は retinotopic 接続を正しく捉えている**。

# %%
def connectivity_vs_hex_dist(pre_type, post_type, hemisphere="right"):
    """指定 cell type ペアの接続を column 距離別に集計"""
    col_h = col_assign[col_assign["hemisphere"] == hemisphere].set_index("root_id")[["p", "q", "column_id"]]
    pre_ids  = set(col_assign[(col_assign["type"] == pre_type)  & (col_assign["hemisphere"] == hemisphere)]["root_id"])
    post_ids = set(col_assign[(col_assign["type"] == post_type) & (col_assign["hemisphere"] == hemisphere)]["root_id"])
    edges = conn[(conn["pre_primary_type"] == pre_type) &
                 (conn["post_primary_type"] == post_type) &
                 (conn["pre_root_id"].isin(pre_ids)) &
                 (conn["post_root_id"].isin(post_ids))].copy()
    print(f"{pre_type} → {post_type} (hemi={hemisphere}): {len(edges):,} edges, {int(edges['syn_count'].sum()):,} syn")

    edges["p_pre"]  = edges["pre_root_id"].map(col_h["p"])
    edges["q_pre"]  = edges["pre_root_id"].map(col_h["q"])
    edges["p_post"] = edges["post_root_id"].map(col_h["p"])
    edges["q_post"] = edges["post_root_id"].map(col_h["q"])
    edges = edges.dropna(subset=["p_pre", "p_post"])
    edges["hex_dist"] = hex_distance(edges["p_post"] - edges["p_pre"], edges["q_post"] - edges["q_pre"]).astype(int)
    return edges

def plot_conn_hist(edges, pre_type, post_type, ax_top, ax_bot, max_dist=6):
    e = edges[edges["hex_dist"] <= max_dist]
    edges_count = e.groupby("hex_dist").size()
    syn_sum     = e.groupby("hex_dist")["syn_count"].sum()
    syn_mean    = e.groupby("hex_dist")["syn_count"].mean()

    ax_top.bar(syn_sum.index, syn_sum.values, color="steelblue", edgecolor="white")
    ax_top.set(xlabel="hex column distance", ylabel="total syn_count",
               title=f"{pre_type} → {post_type}: total synapses by Δcolumn")

    ax_bot.bar(syn_mean.index, syn_mean.values, color="tab:orange", edgecolor="white")
    ax_bot.set(xlabel="hex column distance", ylabel="mean syn_count per edge",
               title=f"{pre_type} → {post_type}: mean syn per edge by Δcolumn")

    # peak check
    peak = syn_mean.idxmax()
    print(f"  Δ=0 edges: {edges_count.get(0, 0):,} ({edges_count.get(0, 0)/edges_count.sum():.0%} of all edges)")
    print(f"  syn_mean peak at Δ={peak} ({syn_mean.iloc[0]:.1f} syn/edge at Δ=0 vs {syn_mean.get(2, 0):.1f} at Δ=2)")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, (pre, post) in enumerate([("L1", "Mi1"), ("L1", "Mi9"), ("Mi1", "T4a")]):
    print(f"\n=== {pre} → {post} ===")
    e = connectivity_vs_hex_dist(pre, post)
    plot_conn_hist(e, pre, post, axes[0, col], axes[1, col])
plt.tight_layout()

# %% [markdown]
# ## Section 6. まとめ — 検証結果
#
# | 検証項目 | 期待 | 実測 | 判定 |
# |---|---|---|---|
# | **Sec 1**: 1-per-(col, hemi) 性 | Mi1/L1/Mi9/T4 系で 1.0 | Mi1=L1=Mi9=Tm1=…=1.0、T4* は 3.79 ≈ 4 (4 subtype × 1) | ✓ |
# | **Sec 2**: (p, q) が hex 格子か | 内部 cell の hex-neighbor 数 = 6 | median 6、668/796 が 6-近傍、縁 47+40+41 が 3-5 近傍 | ✓ |
# | **Sec 3**: 物理対応 (column → 物理位置) | hex 距離 ↔ 物理距離が単調 | Spearman ρ = **0.936** | ✓ |
# | **Sec 4**: cell type 間整合 (Mi1 と Mi9 同 column) | 同 column < random | 同 column 中央値 9,471 vs random 90,430 (**10 倍タイト**) | ✓ |
# | **Sec 5a**: L1 → Mi1 retinotopic | Δ=0 にピーク | edges の **86%** が Δ=0、Δ=0 で **124.3 syn/edge** vs Δ=2 で 4.1 (30×) | ✓ |
# | **Sec 5b**: L1 → Mi9 retinotopic | Δ=0 にピーク | edges の 88% が Δ=0、ただし弱い経路で 1.5 vs 1.0 | ✓ |
# | **Sec 5c**: Mi1 → T4a 接続 | Δ=0 にピーク | Δ=0 は edge の 16% のみ (T4 は複数 column 入力)、ただし **mean syn は Δ=0 が 29.3 vs Δ=2 が 11.4** で peak | ✓ |
#
# ### 結論
#
# `column_assignment.csv` は連結体の retinotopic 構造を **強く正確に** 捉えている:
#
# - 物理座標との対応は Spearman ρ ≈ 0.94 と極めて強い (この強さは soma 座標では絶対に出ない — `synapse_coordinates.csv` を使ったから検出できた)
# - cell type 間で同 column が一貫し、random pair より 10 倍タイト
# - 古典 retinotopic 経路 (L1 → Mi1) が Δ=0 でピーク (syn 数で 30 倍)
#
# → 以降の lateral inhibition 解析では、**`(p, q)` を axial hex coords とみなした hex distance を「Δcolumn = 何個眼またぐか」の物理単位として安心して使える**。Q4 の 3D voxel spread を Δcolumn 単位の lateral spread に置き換えれば、stratification の confound を完全に排除できる。
#
# ### Caveat
#
# - column_assignment に含まれるのは **columnar cell type のみ** (31 種、Mi1/L1/L2/L3/L5/Tm/T2/T3/T4/T5/C2/C3 など)。**Dm/Pm/Lai のような wide-field 抑制 interneuron は含まれない** ので、これらの「カバー柱数」を測りたいときは pre 側でなく **post 側 (columnar target) の column 分布** から逆算する必要がある。
# - 全 31 cell type について validate したわけではなく、Mi1 (Sec 2,3)、Mi9 (Sec 4)、L1/Mi1/Mi9/T4a (Sec 5) のみ。他の type も同じく信頼できる前提だが、念のため気になる type は単独で再確認したい。
