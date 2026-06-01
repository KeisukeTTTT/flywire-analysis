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
# # FlyWire 視覚系における側抑制 (lateral inhibition) の検証
#
# ショウジョウバエ視覚系では「至るところで側抑制が起きている」とよく言われる。FlyWire の連結体だけからこの主張がどこまで支持できるかを以下の問いで確認する。
#
# 1. **Q1** 視覚系の全シナプスのうち、抑制性 (GABA / GLUT / HIS) はどれくらいの割合か?
# 2. **Q2** 抑制性出力をメインに持つ cell type はどれくらい広範に存在するか?
# 3. **Q3** 同タイプ間 (within-type) の抑制接続 — 最も直接的な側抑制シグナル — はどの程度起きているか?
# 4. **Q4** wide-field な抑制 (= 1 ニューロンの出力シナプスが広い範囲に分布) は本当に抑制 dominant cell type で顕著か?
# 5. **Q5** 既知の側抑制回路 (Lai, Dm/Pm 系) が実際にデータで再現されるか?
# 6. **Q6** Dm8 のように output 側が column assignment 外の cell type では、input 側 footprint で receptive field を測れるか?
# 7. **Q7** 端 column の cell は側抑制入力が不足するのか、あるいは補償されるのか?
# 8. **Q8** 抑制性 interneuron family 全体を網羅すると、どの family が wide-field なのか?
# 9. **Q9** Q7 の edge effect は columnar cell type 全般に成立するか?
#
# **抑制性 nt の定義** Drosophila では GABA (GABA-A Cl-)、HIS (HCl1, R1-6 用)、GLUT (GluCl 経由でしばしば抑制性) を抑制性とみなす (`{GABA, GLUT, HIS}`)。`{ACH}` を興奮性。
#
# **caveat** `nt_type` の多くは ML 推定。R1-6 のみドメイン補正で HIS。個々の予測には誤りがあるので aggregate を見る。また connectome から見えるのは配線構造であり、機能的な inhibition の直接証明には activity / physiology が必要。
#

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

INHIBITORY_NT = {"GABA", "GLUT", "HIS"}
EXCITATORY_NT = {"ACH"}

m = FlyWireDataManager()
neurons = m.optic_lobe_neurons_df.copy()
conn = m.optic_lobe_connections_df.copy()

def classify_nt(x):
    if x in INHIBITORY_NT:
        return "inh"
    if x in EXCITATORY_NT:
        return "exc"
    return "other"

conn["sign"] = conn["nt_type"].map(classify_nt)
conn["same_type"] = conn["pre_primary_type"] == conn["post_primary_type"]
print(f"neurons={len(neurons):,}, edges={len(conn):,}")
print(f"sign distribution by edges: {conn['sign'].value_counts().to_dict()}")

# %% [markdown]
# ## Q1. 全シナプスの I/E バランス
#
# 視覚系全体で抑制性シナプスがどの程度を占めるか。30〜40% を超えるようなら抑制が「至るところ」にあると言える。

# %%
syn_by_sign = conn.groupby("sign")["syn_count"].sum()
inh = int(syn_by_sign.get("inh", 0))
exc = int(syn_by_sign.get("exc", 0))
other = int(syn_by_sign.get("other", 0))
total = inh + exc + other
ie_share = inh / (inh + exc)
print(f"Total synapses     : {total:,}")
print(f"  Inhibitory  (GABA/GLUT/HIS): {inh:,} ({inh/total:.1%})")
print(f"  Excitatory  (ACH)          : {exc:,} ({exc/total:.1%})")
print(f"  Other       (DA/SER/OCT)   : {other:,} ({other/total:.1%})")
print(f"  I / (I+E)                   = {ie_share:.1%}")

nt_syn = conn.groupby("nt_type", dropna=False)["syn_count"].sum().sort_values()
colors = ["tab:red" if x in INHIBITORY_NT else ("tab:blue" if x in EXCITATORY_NT else "gray") for x in nt_syn.index]
fig, ax = plt.subplots(figsize=(8, 4))
nt_syn.plot.barh(ax=ax, color=colors)
ax.set(xlabel="# synapses", title="Synapse count by nt_type  (red=inh, blue=exc, gray=other)")
plt.tight_layout()

# %% [markdown]
# ## Q2. 抑制性出力を持つ cell type の広がり
#
# 視覚系の cell type のうち、出力のメインが抑制性であるものがどれくらいあるか。多数あれば「至るところで抑制」に整合。

# %%
type_io = (
    conn.groupby(["pre_primary_type", "sign"])["syn_count"].sum().unstack(fill_value=0)
)
for col in ("inh", "exc", "other"):
    if col not in type_io.columns:
        type_io[col] = 0
type_io["total"] = type_io["inh"] + type_io["exc"] + type_io["other"]
type_io["inh_frac"] = type_io["inh"] / type_io["total"].clip(lower=1)

active = type_io[type_io["total"] >= 1000]
n_active = len(active)
n_pure_exc  = int((active["inh_frac"] < 0.05).sum())
n_mixed     = int(((active["inh_frac"] >= 0.05) & (active["inh_frac"] < 0.5)).sum())
n_mostly_i  = int(((active["inh_frac"] >= 0.5)  & (active["inh_frac"] < 0.95)).sum())
n_pure_inh  = int((active["inh_frac"] >= 0.95).sum())
print(f"Cell types with >=1000 outgoing syns : {n_active}")
print(f"  pure excitatory   (inh<5%)         : {n_pure_exc}")
print(f"  mixed             (5-50%)          : {n_mixed}")
print(f"  mostly inhibitory (50-95%)         : {n_mostly_i}")
print(f"  pure inhibitory   (>=95%)          : {n_pure_inh}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(active["inh_frac"], bins=50, color="steelblue", edgecolor="white")
ax.set(xlabel="inhibitory output fraction (per cell type)", ylabel="# cell types",
       title=f"Distribution of inhibitory output fraction (n={n_active} types, >=1000 syn)")
plt.tight_layout()

print()
print("Top 15 cell types by raw inhibitory output (inh_frac>=50%):")
display(active[active["inh_frac"] >= 0.5].sort_values("inh", ascending=False).head(15)[["inh", "exc", "inh_frac"]])

# %% [markdown]
# ## Q3. 同タイプ内 (within-type) 抑制
#
# 同じ primary_type のニューロン間に抑制接続があると、これは典型的な側抑制 (同じ機能ユニットの隣同士で抑え合う)。
# 各 cell type について、興奮 / 抑制それぞれの出力のうち何 % が「同じ type」を狙うかを比較する。
# **抑制側の自タイプ率 > 興奮側の自タイプ率 (= 対角線より上)** であれば、その type は他より自分の仲間を選択的に抑える傾向がある。
#
# 注: 多くの lateral inhibition は専用の抑制ニューロン (Lai, Dm, Pm) 経由で起きるので、within-type だけが lateral inhibition の指標ではないことに注意。

# %%
agg = (
    conn[conn["sign"].isin(["inh", "exc"])]
    .groupby(["pre_primary_type", "sign", "same_type"])["syn_count"].sum()
    .unstack(fill_value=0)
)
agg.columns = ["other_type_syn", "self_type_syn"]
agg["total"] = agg["other_type_syn"] + agg["self_type_syn"]
agg["self_frac"] = agg["self_type_syn"] / agg["total"].clip(lower=1)

wide = agg.reset_index().pivot(index="pre_primary_type", columns="sign", values=["self_frac", "total"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]

viable = wide[(wide["total_inh"].fillna(0) >= 500) & (wide["total_exc"].fillna(0) >= 500)].copy()
n_above = int((viable["self_frac_inh"] > viable["self_frac_exc"]).sum())
print(f"Cell types with >=500 syn in BOTH inh and exc output: {len(viable)}")
print(f"  inh self-targeting > exc self-targeting: {n_above} / {len(viable)} ({n_above/len(viable):.0%})")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(viable["self_frac_exc"], viable["self_frac_inh"], s=18, alpha=0.6, color="tab:purple")
lim = max(viable[["self_frac_inh", "self_frac_exc"]].max().max(), 0.1) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.5, label="y = x")
ax.set(xlim=(0, lim), ylim=(0, lim),
       xlabel="excitatory self-type syn fraction",
       ylabel="inhibitory self-type syn fraction",
       title="Within-type targeting rate: inh vs exc (each dot = cell type)")
for t, row in viable.iterrows():
    if row["self_frac_inh"] - row["self_frac_exc"] > 0.05:
        ax.annotate(t, (row["self_frac_exc"], row["self_frac_inh"]), fontsize=8, alpha=0.8)
ax.legend()
plt.tight_layout()

print()
print("Top 15 types by (inh_self_frac - exc_self_frac) — clearest within-type lateral inhibition:")
display(viable.assign(diff=viable["self_frac_inh"] - viable["self_frac_exc"])
        .sort_values("diff", ascending=False).head(15)
        [["self_frac_inh", "self_frac_exc", "total_inh", "total_exc", "diff"]])

# %% [markdown]
# ## Q4. Lateral spread (Δcolumn 単位)
#
# 別ノートブック [`column_assignment_validation.ipynb`](column_assignment_validation.ipynb) で `column_assignment.csv` が連結体の retinotopic 構造を強く正確に捉えていることを確認した (Spearman ρ ≈ 0.94, L1→Mi1 は edges の 86% が Δcolumn=0, mean syn が Δ=0 で 30× ピーク)。これにより 3D voxel spread (= stratification と lateral が混在する confound 入りの metric) の代わりに、**Δcolumn 単位の lateral spread** を直接計算できる。
#
# **方法**: 各 pre neuron について
# 1. 出力先のうち column_assignment に含まれる細胞 (= columnar target) だけに絞る
# 2. 同じ半球内の edges に限る (column 距離は半球内でしか意味を持たない)
# 3. post 側の `(p, q)` を syn_count で重み付けして hex centroid を計算
# 4. 各 target の centroid からの hex 距離を weighted mean
# 5. cell type 単位で集計し、**Q2 と同じ synapse-weighted な全出力 sign** で inh vs exc を比較する
#
# これで stratification 軸の伸び (Mi1 が M1-M10 を縦断する等) は完全に消え、純粋に「何個眼まで広がっているか」だけが残る。**Dm/Pm/Lai のような wide-field 抑制 interneuron なら大きな Δcolumn spread を持つはず**。
#
# *Pre 側は cell type 任意 (Dm/Pm/Lai 等を含む)。Post 側のみ columnar に限る。このため Q4 は全出力の部分観測であり、coverage も併記する。*
#

# %%
col_assign = pd.read_csv(
    Path(DATA_DIR) / "raw" / "flywire" / "csv" / "column_assignment.csv",
    dtype={"root_id": str, "column_id": str},
)
col_map = col_assign.set_index("root_id")[["p", "q", "hemisphere"]]
pre_side_map = neurons.drop_duplicates("root_id").set_index("root_id")["side"]

# Edge-level の column 情報
ec = conn[["pre_root_id", "post_root_id", "syn_count", "pre_primary_type", "sign"]].copy()
ec["p_post"]    = ec["post_root_id"].map(col_map["p"])
ec["q_post"]    = ec["post_root_id"].map(col_map["q"])
ec["hemi_post"] = ec["post_root_id"].map(col_map["hemisphere"])
ec["hemi_pre"]  = ec["pre_root_id"].map(pre_side_map)
ec = ec.dropna(subset=["p_post", "q_post", "hemi_pre"])
ec = ec[ec["hemi_pre"] == ec["hemi_post"]]
ec_all = ec.copy()
print(f"Edges where post is columnar AND same hemisphere: {len(ec_all):,} / {len(conn):,}")
print(f"  synapse coverage before per-cell filters: {ec_all['syn_count'].sum() / conn['syn_count'].sum():.1%}")

# Per-pre weighted centroid in (p, q)
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
print(f"pre neurons with >=100 syn to >=5 columnar targets: {len(per_pre):,}")

# Per-edge: centroid からの hex 距離 (cube projection 風の近似で非整数 centroid に対応)
ec["pc"] = ec["pre_root_id"].map(per_pre["pc"])
ec["qc"] = ec["pre_root_id"].map(per_pre["qc"])
ec = ec.dropna(subset=["pc"])
dp = ec["p_post"] - ec["pc"]
dq = ec["q_post"] - ec["qc"]
ec["d_hex"] = np.sqrt((dp**2 + dq**2 + (dp + dq)**2) / 2)
ec["wd"] = ec["d_hex"] * ec["syn_count"]

# Per-pre weighted-mean spread + unique target columns
spread = ec.groupby("pre_root_id").agg(
    wd_sum=("wd", "sum"),
    w_sum=("syn_count", "sum"),
    n_targets=("syn_count", "size"),
)
spread["spread_wmean"] = spread["wd_sum"] / spread["w_sum"]
# Unique (p, q) target columns
ec["col_tup"] = list(zip(ec["p_post"].astype(int), ec["q_post"].astype(int)))
spread["n_unique_cols"] = ec.groupby("pre_root_id")["col_tup"].nunique()

# Cell type / sign を join
type_map = neurons.drop_duplicates("root_id").set_index("root_id")[["primary_type", "nt_type"]]
spread = spread.join(type_map).dropna(subset=["primary_type"])
spread["sign"] = spread["nt_type"].map(classify_nt)
print(f"pre neurons with full info: {len(spread):,}")

# %%
# Per cell type 集計
per_type_col = (
    spread.groupby("primary_type")
    .agg(
        n_neurons=("spread_wmean", "size"),
        type_spread=("spread_wmean", "median"),
        type_n_cols=("n_unique_cols", "median"),
        total_syn_observed=("w_sum", "sum"),
        cell_sign_mode=("sign", lambda x: x.value_counts().idxmax()),
    )
    .query("n_neurons >= 5")
)

# Q4 の inh/exc 比較は、Q2 と同じく「全出力 synapse 数で重み付けた sign」で分類する。
# Q4 subset 内の neuron nt_type 多数決を使うと、Lai / Dm17 / Dm19 / R1-6 などが other 扱いになり得る。
type_sign = type_io[["inh", "exc", "other", "total", "inh_frac"]].copy()
type_sign["dominant_sign"] = type_sign[["inh", "exc", "other"]].idxmax(axis=1)
per_type_col = per_type_col.join(type_sign, how="left")

q4_observable_syn_by_type = ec_all.groupby("pre_primary_type")["syn_count"].sum()
q4_modeled_syn_by_type = ec.groupby("pre_primary_type")["syn_count"].sum()
total_syn_by_type = conn.groupby("pre_primary_type")["syn_count"].sum()
per_type_col["q4_observable_syn_frac"] = per_type_col.index.map(q4_observable_syn_by_type.div(total_syn_by_type))
per_type_col["q4_modeled_syn_frac"] = per_type_col.index.map(q4_modeled_syn_by_type.div(total_syn_by_type))

inh_t = per_type_col[per_type_col["dominant_sign"] == "inh"]
exc_t = per_type_col[per_type_col["dominant_sign"] == "exc"]
other_t = per_type_col[per_type_col["dominant_sign"] == "other"]
spread_ratio_col = float(inh_t["type_spread"].median() / exc_t["type_spread"].median())
ncol_ratio       = float(inh_t["type_n_cols"].median()  / exc_t["type_n_cols"].median())

active_q4_coverage = q4_observable_syn_by_type.div(total_syn_by_type).reindex(active.index).dropna()
print(f"Q4 observable same-hemi columnar synapse coverage: {ec_all['syn_count'].sum() / conn['syn_count'].sum():.1%} of optic-lobe synapses")
print(f"Active cell types with any observable Q4 synapses: {len(active_q4_coverage)} / {len(active)}; median coverage = {active_q4_coverage.median():.1%}")
print(f"INH-dominant cell types: n={len(inh_t):3d}, per-type median Δcolumn spread = {inh_t['type_spread'].median():.2f}, median # target cols = {inh_t['type_n_cols'].median():.0f}")
print(f"EXC-dominant cell types: n={len(exc_t):3d}, per-type median Δcolumn spread = {exc_t['type_spread'].median():.2f}, median # target cols = {exc_t['type_n_cols'].median():.0f}")
print(f"OTHER-dominant cell types in Q4 table: n={len(other_t):3d} (excluded from inh/exc ratio)")
print(f"ratio inh/exc (spread)   = {spread_ratio_col:.2f}")
print(f"ratio inh/exc (# cols)   = {ncol_ratio:.2f}")

mismatch = per_type_col[per_type_col["dominant_sign"] != per_type_col["cell_sign_mode"]]
if len(mismatch):
    print("\nTypes where synapse-weighted sign differs from modal neuron nt_type in the Q4 subset:")
    display(mismatch.sort_values("total", ascending=False)
            [["n_neurons", "type_spread", "type_n_cols", "cell_sign_mode", "dominant_sign", "inh_frac", "q4_observable_syn_frac"]]
            .head(12))

# 2 パネル: weighted-mean spread と # unique target columns
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
upper = float(np.percentile(np.concatenate([inh_t["type_spread"], exc_t["type_spread"]]), 99))
bins = np.linspace(0, upper, 30)
ax.hist(exc_t["type_spread"], bins=bins, alpha=0.5, density=True, label=f"exc (n={len(exc_t)})", color="tab:blue")
ax.hist(inh_t["type_spread"], bins=bins, alpha=0.5, density=True, label=f"inh (n={len(inh_t)})", color="tab:red")
ax.axvline(exc_t["type_spread"].median(), color="tab:blue", ls="--", lw=0.8)
ax.axvline(inh_t["type_spread"].median(), color="tab:red",  ls="--", lw=0.8)
ax.set(xlabel="weighted-mean Δcolumn from centroid (hex units)", ylabel="density",
       title=f"Lateral spread per cell type\ninh/exc median = {spread_ratio_col:.2f}")
ax.legend(fontsize=8)

ax = axes[1]
upper = float(np.percentile(np.concatenate([inh_t["type_n_cols"], exc_t["type_n_cols"]]), 99))
bins = np.linspace(0, upper, 30)
ax.hist(exc_t["type_n_cols"], bins=bins, alpha=0.5, density=True, label=f"exc (n={len(exc_t)})", color="tab:blue")
ax.hist(inh_t["type_n_cols"], bins=bins, alpha=0.5, density=True, label=f"inh (n={len(inh_t)})", color="tab:red")
ax.axvline(exc_t["type_n_cols"].median(), color="tab:blue", ls="--", lw=0.8)
ax.axvline(inh_t["type_n_cols"].median(), color="tab:red",  ls="--", lw=0.8)
ax.set(xlabel="median # unique target columns per neuron", ylabel="density",
       title=f"# columnar target columns per neuron\ninh/exc median = {ncol_ratio:.2f}")
ax.legend(fontsize=8)

plt.suptitle("Q4 (column-based): lateral spread of inh vs exc cell types — stratification confound removed", y=1.05)
plt.tight_layout()

# 具体的な type で比較
examples = ["Dm9", "Dm4", "Dm12", "Dm3p", "Lai", "Pm04", "Pm08",
            "Mi1", "Tm9", "T4a", "T5a", "L1", "L2", "L3"]
example_cols = ["n_neurons", "type_spread", "type_n_cols", "dominant_sign", "inh_frac", "q4_observable_syn_frac", "q4_modeled_syn_frac"]
example_table = per_type_col.loc[per_type_col.index.intersection(examples), example_cols].sort_values("type_spread", ascending=False)
print()
print("Selected cell types — column-unit lateral spread (sorted by spread):")
display(example_table)


# %% [markdown]
# ### Q4 補助可視化: lateral inhibition の射程を直感的に見る
#
# 数値での比較 (上の散布) に加え、以下 2 つの可視化を入れる:
#
# 1. **Single-cell hex-map footprint** — 代表 cell 1 個ずつについて、その出力先 column を hex 格子上に描き syn_count で色付け。Pm08/Pm04/Dm4/Dm12 のような wide-field 抑制 interneuron が「広い赤いパッチ」を作るのに対し、Mi1 のような columnar 興奮ニューロンは「1 つだけ赤い hex」になることが目で見て分かる。
# 2. **Radial output profile** — 各 cell type について、出力 syn が centroid から Δcolumn 離れたところに何 % あるかを線で表示。columnar exc は Δ=0 に鋭いピーク (~80-90%)、wide-field inh は緩やかな decay (Δ=0 で 20-40%、Δ=5-10 でもまだ残る)。

# %%
# 可視化 (1): 代表ニューロン 1 個ずつの column-level 出力 footprint
# Pm/Dm (wide-field 抑制) と Mi1 (columnar 興奮) を並べて「lateral inhibition の射程」を視覚化

# axial hex coords (p, q) → cartesian (x, y) [basis 60度]
def axial_to_cart(p, q):
    return p + 0.5 * q, q * (np.sqrt(3) / 2)

example_cells = ['Pm08', 'Pm04', 'Dm4', 'Dm12', 'Lai', 'Mi1']
fig, axes = plt.subplots(1, len(example_cells), figsize=(4 * len(example_cells), 4.2), sharex=True, sharey=True)

# 各 type の代表 cell (w_sum 最大) を選定
panel_data = []
for ctype in example_cells:
    cands = spread[spread['primary_type'] == ctype].sort_values('w_sum', ascending=False)
    if len(cands) == 0:
        panel_data.append(None); continue
    chosen_id = cands.index[0]
    edges_c = ec[ec['pre_root_id'] == chosen_id]
    col_syn = edges_c.groupby(['p_post', 'q_post'], as_index=False)['syn_count'].sum()
    hemi = edges_c['hemi_post'].iloc[0]
    panel_data.append((ctype, chosen_id, col_syn, hemi))

vmax = max((d[2]['syn_count'].max() for d in panel_data if d is not None), default=1)

for ax, item in zip(axes, panel_data):
    if item is None:
        ax.set_title('no data'); ax.set_axis_off(); continue
    ctype, chosen_id, col_syn, hemi = item
    # 背景: 該当半球の全 column を薄い灰色で
    bg_cols = col_assign[col_assign['hemisphere'] == hemi].drop_duplicates(['p', 'q'])[['p', 'q']]
    bx, by = axial_to_cart(bg_cols['p'].values, bg_cols['q'].values)
    ax.scatter(bx, by, c='lightgray', s=80, marker='H', alpha=0.45, linewidths=0)
    # syn_count をカラーで重ねる
    sx, sy = axial_to_cart(col_syn['p_post'].values, col_syn['q_post'].values)
    sc = ax.scatter(sx, sy, c=col_syn['syn_count'], cmap='hot_r', s=110, marker='H',
                    edgecolors='black', linewidths=0.3, vmin=0, vmax=vmax)
    # centroid を ✕ でマーク
    pc, qc = per_pre.loc[chosen_id, ['pc', 'qc']]
    cx, cy = axial_to_cart(pc, qc)
    ax.plot(cx, cy, 'x', color='cyan', markersize=14, markeredgewidth=3)
    sign = spread.loc[chosen_id, 'sign']
    n_target_cols = len(col_syn)
    spread_val = spread.loc[chosen_id, 'spread_wmean']
    ax.set(aspect='equal', title=f'{ctype} (sign={sign}, hemi={hemi})\n{n_target_cols} target cols, spread={spread_val:.2f}')
    ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(sc, ax=axes, shrink=0.7, location='right', label='syn_count to each column')
plt.suptitle('Column-level output footprint of single representative cells (x = weighted centroid)\n'
             'Wide red patch = wide-field inhibition; single red hex = columnar (focal) output',
             y=1.04, fontsize=11)

# %%
# 可視化 (2): cell type 別の radial output profile
# 各 pre cell の出力 syn を Δcolumn (centroid からの hex 距離) 毎に集計し、
# 同 type の cell をプールして 「平均的にどれくらい広がるか」を線で表示。
# 横軸: Δcolumn / 縦軸: その距離に向かう syn の fraction

profile_types = ['Pm08', 'Pm04', 'Dm4', 'Dm12', 'Lai', 'Mi1', 'Tm9', 'T4a', 'L2']
max_d_show = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for ax, yscale in zip(axes, ['linear', 'log']):
    for ctype in profile_types:
        if ctype not in per_type_col.index:
            continue
        pre_ids = spread[spread['primary_type'] == ctype].index
        type_edges = ec[ec['pre_root_id'].isin(pre_ids)]
        if len(type_edges) == 0:
            continue
        # 整数 Δ にビン (cube 近似で d_hex は非整数なので round)
        d_int = np.minimum(np.round(type_edges['d_hex']).astype(int), max_d_show)
        syn_by_d = type_edges.groupby(d_int)['syn_count'].sum()
        syn_by_d = syn_by_d / syn_by_d.sum()  # 各 type 内で正規化
        sign = per_type_col.loc[ctype, 'dominant_sign']
        color = 'tab:red' if sign == 'inh' else ('tab:blue' if sign == 'exc' else 'tab:gray')
        ls = '-' if sign == 'inh' else ('--' if sign == 'exc' else ':')
        ax.plot(syn_by_d.index, syn_by_d.values, marker='o', label=f'{ctype} ({sign})',
                color=color, alpha=0.75, linestyle=ls, linewidth=1.6, markersize=5)
    ax.set(xlabel='delta-column from cell-specific centroid (hex)',
           ylabel='fraction of synapses at that delta',
           title=f'Radial output profile per cell type ({yscale} scale)')
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper right' if yscale == 'linear' else 'lower left')

plt.suptitle('Wide-field inhibitory cells (Pm/Dm, red solid) decay slowly; '
             'columnar excitatory cells (Mi1/Tm9/L2, blue dashed) peak sharply at delta=0',
             y=1.04, fontsize=11)
plt.tight_layout()


# %% [markdown]
# ## Q5. 既知の側抑制回路の検証
#
# ### Q5a. Lamina amacrine (Lai)
# 古典的にラミナで R1-6 → L1/L2/L3 の活動を、隣の個眼柱由来の信号で抑える GABA 性インターニューロン。

# %%
def summarize_type(type_name, top_n=10):
    out = conn[conn["pre_primary_type"] == type_name]
    if len(out) == 0:
        print(f"  (no outgoing edges for {type_name})")
        return None
    print(f"{type_name}: {len(out):,} edges, {int(out['syn_count'].sum()):,} syn")
    by_nt_edges = out["nt_type"].value_counts().to_dict()
    by_nt_syn   = out.groupby("nt_type")["syn_count"].sum().to_dict()
    print(f"  nt_type (by edges) : {by_nt_edges}")
    print(f"  nt_type (by syns)  : { {k: int(v) for k, v in by_nt_syn.items()} }")
    print(f"  neuropils (top)    : {out['neuropil'].value_counts().head(4).to_dict()}")
    return (out.groupby("post_primary_type")["syn_count"].sum()
            .sort_values(ascending=False).head(top_n).to_frame("syn"))

print("=== Lai (lamina amacrine) ===")
display(summarize_type("Lai", top_n=10))

print("\n=== R1-6 (HIS, photoreceptor) top targets — feedforward into lamina ===")
r16 = conn[conn["pre_primary_type"] == "R1-6"]
display(r16.groupby("post_primary_type")["syn_count"].sum().sort_values(ascending=False).head(8).to_frame("syn"))

# %% [markdown]
# ### Q5b. Medulla の Dm 系 (distal medulla intrinsic)
# Dm9 は M9-M10 で GABA 性 wide-field 抑制を提供することが教科書的に知られる。他の Dm も多くが GABA / GLUT。

# %%
dm_types = sorted([t for t in conn["pre_primary_type"].dropna().unique() if str(t).startswith("Dm")])
dm_summary = (
    type_io.loc[type_io.index.intersection(dm_types)]
    .sort_values("inh", ascending=False)
    [["inh", "exc", "other", "total", "inh_frac"]]
)
print(f"{len(dm_types)} Dm types in dataset. Sorted by total inhibitory output syn:")
display(dm_summary.head(15))

pm_types = sorted([t for t in conn["pre_primary_type"].dropna().unique() if str(t).startswith("Pm")])
pm_summary = (
    type_io.loc[type_io.index.intersection(pm_types)]
    .sort_values("inh", ascending=False)
    [["inh", "exc", "other", "total", "inh_frac"]]
)
print(f"\n{len(pm_types)} Pm types in dataset:")
display(pm_summary.head(10))

# %%
for candidate in ["Dm9", "Dm4", "Dm12", "Dm3p"]:
    if candidate in conn["pre_primary_type"].values:
        print(f"\n--- {candidate} ---")
        display(summarize_type(candidate, top_n=8))


# %% [markdown]
# ## Q6. Dm8 input-side column footprint — Q4 metric の補完
#
# Q4 の output spread metric は **post が column_assignment 入り (= 1-per-column な columnar 細胞)** に限定されるので、**output が wide-field 細胞 (Sm/Tm5*/Dm 系等) ばかりに向かう** cell type は評価できない。代表例が UV color circuit の中核 **Dm8 (Dm8a / Dm8b)**:
#
# - Dm8 自体は column_assignment 未収録 (1-per-column ではない wide-field intrinsic)
# - 主要 output (Tm5a/b/c/d, Sm 系, Dm9) も全て column_assignment 外
# - 一方で **主要 input である R7 (UV photoreceptor) は column_assignment 内** (922 cells, ~1-per-column)
# - → **入力側 R7 column footprint で Dm8 の receptive field を測定**
#
# 教科書記述 (Gao et al. 2008, Karuppudurai et al. 2014): Dm8 は home column の R7 を中心に **~14 個眼** の R7 入力を集積する wide-field 局所インターニューロン (UV 検出の空間積分)。連結体データで再現できるか確認する。

# %%
# 各 Dm8 cell が何個の unique input columns から入力を受けているか
def input_column_footprint(pre_type, post_type, min_syn=5):
    inn = conn[(conn['post_primary_type'] == post_type) &
               (conn['pre_primary_type'] == pre_type)].copy()
    inn['p_pre'] = inn['pre_root_id'].map(col_map['p'])
    inn['q_pre'] = inn['pre_root_id'].map(col_map['q'])
    inn['hemi_pre'] = inn['pre_root_id'].map(col_map['hemisphere'])
    inn['hemi_post'] = inn['post_root_id'].map(pre_side_map)
    inn = inn.dropna(subset=['p_pre', 'q_pre', 'hemi_post'])
    inn = inn[inn['hemi_pre'] == inn['hemi_post']]
    inn['col_tup'] = list(zip(inn['p_pre'].astype(int), inn['q_pre'].astype(int)))
    per_post = inn.groupby('post_root_id').agg(
        n_input_cells=('pre_root_id', 'nunique'),
        n_input_cols=('col_tup', 'nunique'),
        total_syn=('syn_count', 'sum'),
    )
    return per_post[per_post['total_syn'] >= min_syn]

dm8_input_summary = {}
for dm in ['Dm8a', 'Dm8b']:
    rows = []
    for src in ['R7', 'R8', 'Mi1', 'Mi4', 'Mi9', 'L1']:
        ft = input_column_footprint(src, dm, min_syn=5)
        if len(ft) == 0:
            continue
        row = {
            'input_type': src,
            'n_dm8_with_input': len(ft),
            'median_n_input_cols': float(ft['n_input_cols'].median()),
            'min_n_cols': int(ft['n_input_cols'].min()),
            'max_n_cols': int(ft['n_input_cols'].max()),
            'total_syn': int(ft['total_syn'].sum()),
        }
        rows.append(row)
        dm8_input_summary[(dm, src)] = row
    print(f'\n=== {dm} input column footprint per cell ===')
    display(pd.DataFrame(rows))


# %%
# 1 個の Dm8 cell の R7 input column footprint をヘックスマップで可視化
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, dm in zip(axes, ['Dm8a', 'Dm8b']):
    inn = conn[(conn['post_primary_type'] == dm) & (conn['pre_primary_type'] == 'R7')].copy()
    inn['p_pre'] = inn['pre_root_id'].map(col_map['p'])
    inn['q_pre'] = inn['pre_root_id'].map(col_map['q'])
    inn['hemi_pre'] = inn['pre_root_id'].map(col_map['hemisphere'])
    inn['hemi_post'] = inn['post_root_id'].map(pre_side_map)
    inn = inn.dropna(subset=['p_pre', 'q_pre', 'hemi_post'])
    inn = inn[inn['hemi_pre'] == inn['hemi_post']]
    if len(inn) == 0:
        ax.set_title(f'{dm}: no data'); continue
    inn['col_tup'] = list(zip(inn['p_pre'].astype(int), inn['q_pre'].astype(int)))
    # 最多 input column を持つ Dm8 cell を選定 (= classical wide-field 代表)
    n_per = inn.groupby('post_root_id').agg(n_cols=('col_tup', 'nunique'), syn=('syn_count', 'sum'))
    chosen_id = n_per.sort_values(['n_cols', 'syn'], ascending=False).index[0]
    chosen_edges = inn[inn['post_root_id'] == chosen_id]
    col_syn = chosen_edges.groupby(['p_pre', 'q_pre'], as_index=False)['syn_count'].sum()
    hemi = chosen_edges['hemi_post'].iloc[0]
    # 背景: 全 column
    bg = col_assign[col_assign['hemisphere'] == hemi].drop_duplicates(['p', 'q'])[['p', 'q']]
    bx, by = axial_to_cart(bg['p'].values, bg['q'].values)
    ax.scatter(bx, by, c='lightgray', s=80, marker='H', alpha=0.45, linewidths=0)
    # R7 input column を syn_count で色付け
    sx, sy = axial_to_cart(col_syn['p_pre'].values, col_syn['q_pre'].values)
    sc = ax.scatter(sx, sy, c=col_syn['syn_count'], cmap='hot_r', s=110, marker='H',
                    edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='R7 syn -> this Dm8', shrink=0.7)
    ax.set(aspect='equal', title=f'{dm}: R7 input from {len(col_syn)} columns (max-coverage cell)')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Single Dm8 cell - R7 input column footprint (classical home column + surrounding ~14 ommatidia)',
             y=1.04, fontsize=11)
plt.tight_layout()

# %% [markdown]
# ### Q6 続き: Dm8 の input/output footprint を可視化する
#
# Dm8 の output 側 cell type (Tm5*/Sm 系) は column_assignment 外で直接 column 距離が測れない。そこで **Mi1 を近傍探索のプロキシ** とする:
#
# - Mi1 は 1-per-column / ME-intrinsic で column_assignment にあり、各 Mi1 cell の物理 centroid と (p, q) column の対応が分かる (column_assignment_validation.ipynb で Spearman ρ ≈ 0.94 を確認済み)
# - 任意の 3D 物理座標 (例: Dm8 の output synapse 位置) に対し、最も近い Mi1 を求めればその Mi1 の column が「その synapse が属する column」の近似値になる
# - これで Dm8 output synapse 群を column 空間に投影可能 → 入力 (R7) と出力 (Mi1-proxy) を同じ hex 格子上に並べて比較
#
# **I/O 構造の検証**: Dm8 は wide-field R7 input を受ける一方、output は home column 周辺に絞られるかを確認する。これは Pm08 のような「広く撒く」wide-field inhibition とは別の circuit motif である。
#

# %%
# synapse_coordinates をロード (Q4 で必要だったが column-based 版では消去したので再読込)
syn_path = Path(DATA_DIR) / 'raw' / 'flywire' / 'csv' / 'synapse_coordinates.csv'
t0 = time.perf_counter()
syn_coords = pd.read_csv(syn_path, dtype={'pre_root_id': str, 'post_root_id': str},
                         usecols=['pre_root_id', 'x', 'y', 'z'])
syn_coords['pre_root_id'] = syn_coords['pre_root_id'].ffill()
syn_coords = syn_coords.dropna(subset=['pre_root_id'])
print(f'loaded {len(syn_coords):,} synapses in {time.perf_counter()-t0:.1f}s')

# Mi1 centroid を column proxy として KDTree 構築 (半球別)
from scipy.spatial import cKDTree

def build_mi1_index(hemi):
    mi1 = col_assign[(col_assign['type'] == 'Mi1') & (col_assign['hemisphere'] == hemi)]
    mi1_ids = set(mi1['root_id'])
    s = syn_coords[syn_coords['pre_root_id'].isin(mi1_ids)]
    counts = s.groupby('pre_root_id').size()
    cents = s.groupby('pre_root_id')[['x','y','z']].mean()
    cents.columns = ['cx', 'cy', 'cz']
    cents = cents[counts >= 20]
    full = mi1.set_index('root_id').join(cents, how='inner').dropna()
    tree = cKDTree(full[['cx','cy','cz']].values)
    cols = full[['p','q']].values.astype(int)
    return tree, cols, full

mi1_idx = {h: build_mi1_index(h) for h in ['right', 'left']}
for h, (_, _, f) in mi1_idx.items():
    print(f'Mi1 {h}-hemi indexed: {len(f)} cells')

def dm8_io_footprint(dm8_id):
    hemi = pre_side_map.get(dm8_id)
    if hemi not in mi1_idx:
        return None
    tree, cols, _ = mi1_idx[hemi]
    inn = conn[(conn['post_root_id'] == dm8_id) & (conn['pre_primary_type'] == 'R7')].copy()
    inn['p'] = inn['pre_root_id'].map(col_map['p'])
    inn['q'] = inn['pre_root_id'].map(col_map['q'])
    inn['hemi_pre'] = inn['pre_root_id'].map(col_map['hemisphere'])
    inn = inn[inn['hemi_pre'] == hemi].dropna(subset=['p','q'])
    inp = (inn.assign(p=inn['p'].astype(int), q=inn['q'].astype(int))
              .groupby(['p','q'], as_index=False)['syn_count'].sum()
              .rename(columns={'syn_count': 'syn'}))
    out_s = syn_coords[syn_coords['pre_root_id'] == dm8_id]
    if len(out_s) == 0:
        return inp, pd.DataFrame(columns=['p','q','syn']), hemi
    _, idx = tree.query(out_s[['x','y','z']].values)
    out_pq = cols[idx]
    counts = pd.DataFrame({'p': out_pq[:, 0], 'q': out_pq[:, 1]}).value_counts().reset_index()
    counts.columns = ['p', 'q', 'syn']
    return inp, counts, hemi

def pick_top_dm8(dm_type):
    df = conn[(conn['post_primary_type'] == dm_type) & (conn['pre_primary_type'] == 'R7')].copy()
    df['p'] = df['pre_root_id'].map(col_map['p'])
    df['q'] = df['pre_root_id'].map(col_map['q'])
    df = df.dropna(subset=['p', 'q'])
    df['t'] = list(zip(df['p'].astype(int), df['q'].astype(int)))
    g = df.groupby('post_root_id').agg(n=('t','nunique'), s=('syn_count','sum'))
    return g.sort_values(['n', 's'], ascending=False).index[0]

top_a = pick_top_dm8('Dm8a')
top_b = pick_top_dm8('Dm8b')

fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for row, (dm8_id, dm_type) in enumerate([(top_a, 'Dm8a'), (top_b, 'Dm8b')]):
    inp, outp, hemi = dm8_io_footprint(dm8_id)
    bg = col_assign[col_assign['hemisphere'] == hemi].drop_duplicates(['p', 'q'])
    bx, by = axial_to_cart(bg['p'].values, bg['q'].values)
    panels = [(inp, 'Input from R7 columns', 'Blues'),
              (outp, 'Output to columns (via Mi1 proxy)', 'Reds')]
    for col, (df, label, cmap) in enumerate(panels):
        ax = axes[row, col]
        ax.scatter(bx, by, c='lightgray', s=70, marker='H', alpha=0.4, linewidths=0)
        if len(df) > 0:
            sx, sy = axial_to_cart(df['p'].values, df['q'].values)
            sc = ax.scatter(sx, sy, c=df['syn'], cmap=cmap, s=100, marker='H',
                            edgecolors='black', linewidths=0.3)
            plt.colorbar(sc, ax=ax, label='syn count', shrink=0.7)
        title = dm_type + ' (hemi=' + str(hemi) + ', n_cols=' + str(len(df)) + ') -- ' + label
        ax.set(aspect='equal', title=title)
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Dm8 single-cell lateral inhibition: '
             'wide-field R7 input (blue, left) -> wide-field inhibitory output (red, right)  '
             '[Mi1-proxy = nearest-Mi1 column for each output synapse]',
             y=1.01, fontsize=10)
plt.tight_layout()

# %%
# Aggregate: 全 Dm8 cell について input col 数 vs output col 数
# 最適化: syn_coords を Dm8 IDs に pre-filter してから groupby で高速ルックアップ
all_dm8_ids = set(neurons[neurons['primary_type'].isin(['Dm8a', 'Dm8b'])]['root_id'])
syn_dm8 = syn_coords[syn_coords['pre_root_id'].isin(all_dm8_ids)]
syn_dm8_groups = dict(tuple(syn_dm8[['pre_root_id', 'x', 'y', 'z']].groupby('pre_root_id')))
print(f'pre-grouped {len(syn_dm8):,} Dm8 output syns into {len(syn_dm8_groups)} cells')

def io_col_counts(dm_type, min_out_syn=20):
    rows = []
    inn = conn[(conn['post_primary_type'] == dm_type) & (conn['pre_primary_type'] == 'R7')].copy()
    inn['p'] = inn['pre_root_id'].map(col_map['p'])
    inn['q'] = inn['pre_root_id'].map(col_map['q'])
    inn = inn.dropna(subset=['p', 'q'])
    inn['t'] = list(zip(inn['p'].astype(int), inn['q'].astype(int)))
    n_in = inn.groupby('post_root_id').agg(n_in=('t', 'nunique'), syn_in=('syn_count', 'sum'))
    for dm8_id in n_in.index:
        hemi = pre_side_map.get(dm8_id)
        if hemi not in mi1_idx: continue
        tree, cols, _ = mi1_idx[hemi]
        out_s = syn_dm8_groups.get(dm8_id)
        if out_s is None or len(out_s) < min_out_syn: continue
        _, idx = tree.query(out_s[['x','y','z']].values)
        out_pq = cols[idx]
        n_out = len(set(map(tuple, out_pq)))
        rows.append((dm8_id, int(n_in.loc[dm8_id, 'n_in']), n_out, len(out_s)))
    return pd.DataFrame(rows, columns=['dm8_id', 'n_in_cols', 'n_out_cols', 'n_out_syn']).set_index('dm8_id')

io_a = io_col_counts('Dm8a')
io_b = io_col_counts('Dm8b')
dm8_io_summary = {}
for name, df in [('Dm8a', io_a), ('Dm8b', io_b)]:
    ratio = (df['n_out_cols'] / df['n_in_cols']).median() if len(df) else float('nan')
    dm8_io_summary[name] = {
        'n_cells': int(len(df)),
        'median_n_in_cols': float(df['n_in_cols'].median()) if len(df) else float('nan'),
        'median_n_out_cols': float(df['n_out_cols'].median()) if len(df) else float('nan'),
        'median_out_in_ratio': float(ratio),
    }
    print(f'{name}: n cells = {len(df)}')
    print(f'  median n_in_cols  = {df["n_in_cols"].median():.0f}, median n_out_cols = {df["n_out_cols"].median():.0f}')
    print(f'  median output/input column ratio = {ratio:.2f}')

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(io_a['n_in_cols'], io_a['n_out_cols'], s=14, alpha=0.5, color='tab:purple', label='Dm8a (n=' + str(len(io_a)) + ')')
ax.scatter(io_b['n_in_cols'], io_b['n_out_cols'], s=14, alpha=0.5, color='tab:green',  label='Dm8b (n=' + str(len(io_b)) + ')')
lim_x = max(io_a['n_in_cols'].max(), io_b['n_in_cols'].max()) + 2
lim_y = max(io_a['n_out_cols'].max(), io_b['n_out_cols'].max()) + 5
m = max(lim_x, lim_y)
ax.plot([0, m], [0, m], 'k--', lw=0.7, alpha=0.5, label='y = x')
ax.set(xlim=(0, lim_x), ylim=(0, lim_y),
       xlabel='# input columns (R7-based)',
       ylabel='# output columns (Mi1-proxy)',
       title='Per Dm8 cell: input column count vs output column count\n'
             '(typical Dm8: wide R7 input ~7 cols, narrower output ~4-5 cols at home column)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# ## Q7. 端 column と中央 column の inhibition 比較
#
# medulla 個眼柱は hex 格子で並んでおり、**端の column (3-5 近傍) は中央 (6 近傍) より隣接 column が少ない**。lateral inhibition が「隣接 column から来る」のなら、端 column は本来少ない抑制しか受けられないはず — 視覚処理の境界条件問題。
#
# 連結体データで以下を確認する:
# 1. **inh 入力量の差**: Mi1 (1-per-column の columnar 興奮ニューロン) を受け手として、各 Mi1 が受ける総 inh syn 数を端 / 中央で比較
# 2. **空間ヒートマップ**: 各 column の Mi1 が受ける inh 量を hex 格子上に色付け、端〜中央の勾配を見る
# 3. **入力源の内訳**: inh 入力源を **local 抑制 (Dm/Pm/Lai — 空間的に限定的)** と **global 抑制 (CT1 — 全 medulla を 1 cell でカバー)** に分解。端では local が減り global で補償されているという仮説をテスト
#
# CT1 は medulla 全体を 1 cell でカバーする巨大 GABA 性ニューロン (medulla 内に 1 個ずつしか無い、Lin & Meinertzhagen 2013) なので、端 column でも中央 column でも均等に inh を提供できる構造上、端での compensation 源として理論上最有力。

# %%
# Mi1 (right hemi) の edge category を hex 近傍数で定義
mi1_r = col_assign[(col_assign['type'] == 'Mi1') & (col_assign['hemisphere'] == 'right')]
mi1_pq = mi1_r[['root_id', 'p', 'q']].copy()

# 各 column の hex 近傍数 (Mi1 が存在する近傍の数 = 有効近傍数)
all_pq = set(zip(mi1_pq['p'], mi1_pq['q']))
hex_neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
def n_neighbors(p, q):
    return sum((p+dp, q+dq) in all_pq for dp, dq in hex_neighbors)
mi1_pq['n_neighbors'] = mi1_pq.apply(lambda r: n_neighbors(r['p'], r['q']), axis=1)
mi1_pq['edge_cat'] = pd.cut(mi1_pq['n_neighbors'], bins=[-1, 3, 5, 6],
                            labels=['corner (<=3 nbrs)', 'edge (4-5 nbrs)', 'interior (6 nbrs)'])

print('Mi1 right-hemi edge categories:')
print(mi1_pq['edge_cat'].value_counts().to_dict())

# 各 Mi1 が受ける inh / exc / other syn 合計
mi1_ids = set(mi1_pq['root_id'])
incoming = conn[conn['post_root_id'].isin(mi1_ids)]
syn_by_sign = (incoming.groupby(['post_root_id', 'sign'])['syn_count'].sum()
               .unstack(fill_value=0)
               .reindex(columns=['inh', 'exc', 'other'], fill_value=0))
mi1_pq = mi1_pq.set_index('root_id').join(syn_by_sign, how='left').reset_index()
# Categorical 列 (edge_cat) を保ったまま、数値列のみ fillna(0)
for c in ['inh', 'exc', 'other']:
    mi1_pq[c] = mi1_pq[c].fillna(0).astype(int)
mi1_pq['inh_frac'] = mi1_pq['inh'] / (mi1_pq['inh'] + mi1_pq['exc']).clip(lower=1)

g = mi1_pq.groupby('edge_cat', observed=True).agg(
    n=('root_id', 'size'),
    median_inh=('inh', 'median'),
    median_exc=('exc', 'median'),
    median_inh_frac=('inh_frac', 'median'),
)
print()
print('Per-Mi1 syn input by edge category (median across cells):')
print(g.to_string())

# %%
# 空間ヒートマップ: 各 Mi1 column の inh / exc 入力 + n_neighbors
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cx, cy = axial_to_cart(mi1_pq['p'].values, mi1_pq['q'].values)

# (1) inh
ax = axes[0]
sc = ax.scatter(cx, cy, c=mi1_pq['inh'], cmap='Reds', s=110, marker='H',
                edgecolors='black', linewidths=0.3)
plt.colorbar(sc, ax=ax, label='total inh syn to Mi1')
ax.set(aspect='equal', title='Inhibitory input per Mi1 column')
ax.set_xticks([]); ax.set_yticks([])

# (2) exc
ax = axes[1]
sc = ax.scatter(cx, cy, c=mi1_pq['exc'], cmap='Blues', s=110, marker='H',
                edgecolors='black', linewidths=0.3)
plt.colorbar(sc, ax=ax, label='total exc syn to Mi1')
ax.set(aspect='equal', title='Excitatory input per Mi1 column')
ax.set_xticks([]); ax.set_yticks([])

# (3) n_neighbors (edge map)
ax = axes[2]
sc = ax.scatter(cx, cy, c=mi1_pq['n_neighbors'], cmap='viridis', s=110, marker='H',
                edgecolors='black', linewidths=0.3, vmin=3, vmax=6)
plt.colorbar(sc, ax=ax, label='# valid hex neighbors')
ax.set(aspect='equal', title='Edge map (3-5 = edge/corner, 6 = interior)')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Spatial heatmap of inhibitory input per Mi1 column (right hemi) vs edge category', y=1.02)
plt.tight_layout()

# 端で inh が落ちるなら inh ヒートマップが外周で淡くなるはず
print()
print('Correlation between n_neighbors (edge proximity) and inh input:')
import scipy.stats as st
r, p = st.spearmanr(mi1_pq['n_neighbors'], mi1_pq['inh'])
print(f'  Spearman rho (n_neighbors vs inh syn): {r:.3f} (p = {p:.2g})')
r2, p2 = st.spearmanr(mi1_pq['n_neighbors'], mi1_pq['exc'])
print(f'  Spearman rho (n_neighbors vs exc syn): {r2:.3f} (p = {p2:.2g})')

# %%
# 入力源の内訳: local (Dm/Pm/Lai) vs global (CT1) vs その他
mi1_in_inh = conn[conn['post_root_id'].isin(mi1_ids) & (conn['sign'] == 'inh')].copy()

def categorize(t):
    t = str(t)
    if t == 'CT1': return 'CT1 (global, medulla-wide)'
    if t == 'Lai': return 'Lai (lamina amacrine)'
    if t.startswith('Dm'): return 'Dm (distal medulla wide-field)'
    if t.startswith('Pm'): return 'Pm (proximal medulla wide-field)'
    return 'other inhibitory'

mi1_in_inh['src_cat'] = mi1_in_inh['pre_primary_type'].map(categorize)

by_src = mi1_in_inh.groupby(['post_root_id', 'src_cat'])['syn_count'].sum().unstack(fill_value=0)

# Mi1 ごとに edge_cat を貼って source 内訳を集計
detail = mi1_pq.set_index('root_id')[['edge_cat']].join(by_src, how='left')
src_cols = list(by_src.columns)
# 数値列のみ fillna (edge_cat は Categorical なので除外)
for c in src_cols:
    detail[c] = detail[c].fillna(0).astype(int)

agg = detail.groupby('edge_cat', observed=True)[src_cols].mean()
agg['total_inh_mean'] = agg[src_cols].sum(axis=1)
ratio = agg[src_cols].div(agg['total_inh_mean'], axis=0)

print('Mean inh syn per Mi1 by edge category and source:')
print(agg.round(0).to_string())
print()
print('Fraction of inh from each source:')
print(ratio.round(3).to_string())

# プロット: 端〜中央でカテゴリ別 syn 数の積み上げ
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
agg[src_cols].plot.bar(stacked=True, ax=axes[0], colormap='tab10', edgecolor='white')
axes[0].set(xlabel='edge category', ylabel='mean inh syn per Mi1',
            title='Per Mi1 absolute inh syn input by source category')
axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
axes[0].tick_params(axis='x', rotation=15)

ratio[src_cols].plot.bar(stacked=True, ax=axes[1], colormap='tab10', edgecolor='white')
axes[1].set(xlabel='edge category', ylabel='fraction of inh input',
            title='Per Mi1 fraction of inh input by source category', ylim=(0, 1))
axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
axes[1].tick_params(axis='x', rotation=15)
plt.suptitle('Edge compensation: does global (CT1) inhibition fraction increase at the edge?', y=1.04)
plt.tight_layout()

# %%
# Q7 の key finding を 1 枚で見る: edge category 別の inh / exc / inh_frac 分布
import numpy as np

cats_order = ['corner (<=3 nbrs)', 'edge (4-5 nbrs)', 'interior (6 nbrs)']
colors_cat = ['tab:red', 'tab:orange', 'tab:blue']

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
configs = [
    ('inh',      'inh syn count per Mi1',  'Inhibitory input (absolute)'),
    ('exc',      'exc syn count per Mi1',  'Excitatory input (absolute)'),
    ('inh_frac', 'inh / (inh + exc)',      'E/I balance (PRESERVED across edge/center)'),
]
for ax, (col, ylabel, title) in zip(axes, configs):
    data = [mi1_pq[mi1_pq['edge_cat'] == c][col].values for c in cats_order]
    bp = ax.boxplot(data, tick_labels=[c.replace(' (', '\n(') for c in cats_order],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_cat):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    medians = [float(np.median(d)) for d in data]
    for x, m in enumerate(medians, 1):
        txt = f'  {m:.2f}' if col == 'inh_frac' else f'  {m:.0f}'
        ax.text(x, m, txt, va='center', ha='left', fontsize=9, fontweight='bold')
    ax.set(ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3, axis='y')
    if col == 'inh_frac':
        ax.set_ylim(0, 1)

plt.suptitle('Q7 key finding: edge Mi1 receives proportionally less inh AND exc, but E/I balance is preserved',
             y=1.05, fontsize=12)
plt.tight_layout()

# Bonus: radial profile (medulla 中心からの距離に対する inh / exc / inh_frac)
center_p = mi1_pq['p'].mean(); center_q = mi1_pq['q'].mean()
dp = mi1_pq['p'] - center_p; dq = mi1_pq['q'] - center_q
mi1_pq['dist_from_center'] = np.sqrt((dp**2 + dq**2 + (dp + dq)**2) / 2)

dist_bins = np.linspace(0, mi1_pq['dist_from_center'].max() + 0.5, 10)
mi1_pq['dist_bin'] = pd.cut(mi1_pq['dist_from_center'], bins=dist_bins)
radial = mi1_pq.groupby('dist_bin', observed=True).agg(
    n=('inh', 'size'),
    mean_inh=('inh', 'mean'),
    sem_inh=('inh', lambda x: x.std() / max(np.sqrt(len(x)), 1)),
    mean_exc=('exc', 'mean'),
    sem_exc=('exc', lambda x: x.std() / max(np.sqrt(len(x)), 1)),
    mean_inh_frac=('inh_frac', 'mean'),
    sem_inh_frac=('inh_frac', lambda x: x.std() / max(np.sqrt(len(x)), 1)),
)
x_bin = np.array([b.mid for b in radial.index])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.errorbar(x_bin, radial['mean_inh'], yerr=radial['sem_inh'], marker='o',
            label='inh syn (mean +/- SEM)', color='tab:red', linewidth=2)
ax.errorbar(x_bin, radial['mean_exc'], yerr=radial['sem_exc'], marker='s',
            label='exc syn (mean +/- SEM)', color='tab:blue', linewidth=2)
ax.set(xlabel='hex distance from medulla center',
       ylabel='mean syn per Mi1 (absolute)',
       title='Radial profile: input strength decreases monotonically toward the edge')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.errorbar(x_bin, radial['mean_inh_frac'], yerr=radial['sem_inh_frac'], marker='^',
             linestyle='--', color='tab:green', linewidth=1.5, label='inh fraction', alpha=0.8)
ax2.set_ylabel('inh / (inh+exc)  [green dashed]', color='tab:green')
ax2.set_ylim(0.5, 0.85)
ax2.tick_params(axis='y', colors='tab:green')
ax2.legend(loc='lower right')

plt.suptitle('inh and exc fall together with distance from center, while their RATIO stays near constant',
             y=1.01, fontsize=11)
plt.tight_layout()


# %% [markdown]
# ### Q7 補足: 平均化しない生データ可視化でエッジ現象を多角的に見る
#
# Q7 本体は corner/edge/interior の 3 カテゴリ化と median/Spearman への集約に依存しており、
# 「medulla は等方的」「カテゴリ内の細胞は単峰性」を暗黙に仮定する。そのため局所異常・二峰性・
# 方向依存・データアーティファクトを見落とす恐れがある。以下では Mi1 (right hemi) の生データを
# なるべく集約せずに別角度から見る:
#
# - **A. 空間をつぶさない**: `inh_frac` の hex マップ + radial trend からの残差マップ
# - **B. 分布をつぶさない**: edge_cat 別の violin + 全細胞 strip (boxplot は二峰性を隠す)
# - **C. 連続軸でみる**: n_neighbors を 0-6 個別に, さらに inh vs exc の生散布 (距離で色付け)
# - **D. 異方性**: 中心からの方位角別に inh 分布 (radial profile が潰している方向依存の検証)
# - **F. 交絡チェック**: edge deficit が arbor truncation / proofreading 由来でないか
#   (総出力 syn・未分類 nt 割合が端で増えるか)

# %%
# 共通ヘルパー: violin + 全点 strip を重ねる。
# データ点が少ない / 分散ゼロの group は violin の KDE が不安定なので strip のみにフォールバック。
def violin_strip(ax, data_groups, positions, colors, rng, jitter=0.13, point_size=8, alpha=0.4):
    if isinstance(colors, str):
        colors = [colors] * len(data_groups)
    vio_idx = [i for i, d in enumerate(data_groups) if len(d) >= 2 and np.std(d) > 0]
    if vio_idx:
        vp = ax.violinplot([data_groups[i] for i in vio_idx],
                           positions=[positions[i] for i in vio_idx],
                           showmedians=True, widths=0.8)
        for body, i in zip(vp['bodies'], vio_idx):
            body.set_facecolor(colors[i]); body.set_alpha(0.3)
    for pos, d, c in zip(positions, data_groups, colors):
        if len(d) == 0:
            continue
        jit = rng.uniform(-jitter, jitter, size=len(d))
        ax.scatter(np.full(len(d), pos) + jit, d, s=point_size, alpha=alpha, color=c, linewidths=0)

# %%
# A. 空間をつぶさない: inh_frac の hex マップ + radial trend 残差マップ
cx, cy = axial_to_cart(mi1_pq['p'].values, mi1_pq['q'].values)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) E/I 比率の空間マップ — 絶対量 (Q7 既出) ではなく比率の局所ポケットを探す
ax = axes[0]
sc = ax.scatter(cx, cy, c=mi1_pq['inh_frac'], cmap='RdBu_r', s=110, marker='H',
                edgecolors='black', linewidths=0.3,
                vmin=mi1_pq['inh_frac'].quantile(0.02), vmax=mi1_pq['inh_frac'].quantile(0.98))
plt.colorbar(sc, ax=ax, label='inh / (inh+exc)')
ax.set(aspect='equal', title='E/I balance per Mi1 column (spatial, no binning)')
ax.set_xticks([]); ax.set_yticks([])

# (2) radial trend からの残差: 中心距離の 2 次多項式で inh を予測し、その残差を色付け。
#     単純な radial decay で説明できない局所異常 (特定の辺だけ強い/弱い) を浮かせる。
d = mi1_pq['dist_from_center'].values
coef = np.polyfit(d, mi1_pq['inh'].values, 2)
resid = mi1_pq['inh'].values - np.polyval(coef, d)
ax = axes[1]
vlim = np.percentile(np.abs(resid), 98)
sc = ax.scatter(cx, cy, c=resid, cmap='coolwarm', s=110, marker='H',
                edgecolors='black', linewidths=0.3, vmin=-vlim, vmax=vlim)
plt.colorbar(sc, ax=ax, label='inh syn  -  radial-model prediction')
ax.set(aspect='equal', title='Residual of inh after removing radial trend\n(red = more inh than radius predicts)')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('A. Spatial views without binning: local E/I pockets and non-radial anomalies', y=1.02)
plt.tight_layout()

# %%
# B. 分布をつぶさない: edge_cat 別に violin + 全細胞 strip (boxplot は二峰性を隠す)
cats_order = ['corner (<=3 nbrs)', 'edge (4-5 nbrs)', 'interior (6 nbrs)']
colors_cat = ['tab:red', 'tab:orange', 'tab:blue']
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs = [('inh', 'inh syn per Mi1'), ('exc', 'exc syn per Mi1'), ('inh_frac', 'inh / (inh+exc)')]
for ax, (col, ylabel) in zip(axes, configs):
    data = [mi1_pq[mi1_pq['edge_cat'] == c][col].values for c in cats_order]
    violin_strip(ax, data, list(range(len(cats_order))), colors_cat, rng, point_size=7, alpha=0.35)
    ax.set_xticks(range(len(cats_order)))
    ax.set_xticklabels([c.replace(' (', '\n(') for c in cats_order], fontsize=8)
    ax.set(ylabel=ylabel, title=ylabel)
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('B. Full per-cell distributions (violin + every cell): is the edge population unimodal?',
             y=1.03, fontsize=12)
plt.tight_layout()

# %%
# C. 連続軸でみる: n_neighbors を 3bin ではなく 0-6 個別に, さらに inh vs exc 生散布
nn_vals = sorted(mi1_pq['n_neighbors'].unique())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) n_neighbors 個別の inh 生分布
ax = axes[0]
data = [mi1_pq[mi1_pq['n_neighbors'] == n]['inh'].values for n in nn_vals]
violin_strip(ax, data, nn_vals, 'tab:red', np.random.default_rng(1))
ax.set(xlabel='# valid hex neighbors (0-6)', ylabel='inh syn per Mi1',
       title='Inh input per exact neighbor count\n(3-category binning collapses 0,1,2,3)')
ax.grid(True, alpha=0.3, axis='y')

# (2) 同じく exc
ax = axes[1]
data = [mi1_pq[mi1_pq['n_neighbors'] == n]['exc'].values for n in nn_vals]
violin_strip(ax, data, nn_vals, 'tab:blue', np.random.default_rng(2))
ax.set(xlabel='# valid hex neighbors (0-6)', ylabel='exc syn per Mi1',
       title='Exc input per exact neighbor count')
ax.grid(True, alpha=0.3, axis='y')

# (3) inh vs exc 生散布, 中心距離で色付け: 縁細胞が E/I 直線上を滑るか, 直線から外れるか
ax = axes[2]
sc = ax.scatter(mi1_pq['exc'], mi1_pq['inh'], c=mi1_pq['dist_from_center'],
                cmap='viridis', s=22, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=ax, label='hex distance from center')
mask = mi1_pq['exc'] > 0
slope = np.polyfit(mi1_pq.loc[mask, 'exc'], mi1_pq.loc[mask, 'inh'], 1)
xline = np.array([0, mi1_pq['exc'].max()])
ax.plot(xline, np.polyval(slope, xline), 'k--', lw=0.8, alpha=0.6, label=f'linear fit (slope={slope[0]:.2f})')
ax.set(xlabel='exc syn per Mi1', ylabel='inh syn per Mi1',
       title='inh vs exc per cell (color = edge proximity)\nedge cells = dark; do they stay on the line?')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('C. Continuous-axis raw views: per-neighbor distributions and raw inh-vs-exc scatter',
             y=1.03, fontsize=12)
plt.tight_layout()

# %%
# D. 異方性: 中心からの方位角別に inh 分布を見る (radial profile は方向を潰している)。
#    外周リング (n_neighbors <= 5) の細胞のみ対象に、背側縁/腹側縁などで差が無いか検証する。
edge_cells = mi1_pq[mi1_pq['n_neighbors'] <= 5].copy()
ex, ey = axial_to_cart((edge_cells['p'] - center_p).values, (edge_cells['q'] - center_q).values)
edge_cells['angle'] = np.degrees(np.arctan2(ey, ex)) % 360
n_sectors = 8
edge_cells['sector'] = (edge_cells['angle'] // (360 / n_sectors)).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (1) 縁細胞を方位角でグループ化し inh の生分布 (等方なら sector 間でフラット)
ax = axes[0]
sectors = sorted(edge_cells['sector'].unique())
data = [edge_cells[edge_cells['sector'] == s]['inh'].values for s in sectors]
labels = [f'{int(s*360/n_sectors)}-{int((s+1)*360/n_sectors)}°\n(n={len(dd)})' for s, dd in zip(sectors, data)]
violin_strip(ax, data, list(range(len(sectors))), 'tab:purple', np.random.default_rng(3))
ax.set_xticks(range(len(sectors)))
ax.set_xticklabels(labels, fontsize=7)
ax.set(ylabel='inh syn per edge Mi1',
       title='Inh input of edge cells by azimuthal direction\n(isotropic edge = flat across sectors)')
ax.grid(True, alpha=0.3, axis='y')

# (2) 縁細胞の位置を inh で色付け, どのリム方向が弱いか直接みる
ax = axes[1]
allx, ally = axial_to_cart(mi1_pq['p'].values, mi1_pq['q'].values)
ax.scatter(allx, ally, c='lightgray', s=70, marker='H', alpha=0.4, linewidths=0)
ecx, ecy = axial_to_cart(edge_cells['p'].values, edge_cells['q'].values)
sc = ax.scatter(ecx, ecy, c=edge_cells['inh'], cmap='Reds', s=110, marker='H',
                edgecolors='black', linewidths=0.3)
plt.colorbar(sc, ax=ax, label='inh syn (edge cells only)')
ax.set(aspect='equal', title='Edge cells colored by inh input\n(look for one rim systematically weaker)')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('D. Anisotropy check: does the edge effect depend on direction (e.g. dorsal rim)?',
             y=1.02, fontsize=12)
plt.tight_layout()

# %%
# F. 交絡チェック: edge deficit が "arbor truncation" や proofreading 由来でないか。
#    縁細胞は imaging volume の境界で arbor が切れ, 入力も出力も一律減るだけかもしれない。
#    -> 総出力 syn も端で減るなら inhibition 特異的でなく cell 全体の効果 (truncation 示唆)。
mi1_out = conn[conn['pre_root_id'].isin(mi1_ids)].groupby('pre_root_id')['syn_count'].sum()
mi1_in_all = conn[conn['post_root_id'].isin(mi1_ids)].groupby('post_root_id')['syn_count'].sum()
mi1_pq['total_out'] = mi1_pq['root_id'].map(mi1_out).fillna(0)
mi1_pq['total_in'] = mi1_pq['root_id'].map(mi1_in_all).fillna(0)
mi1_pq['other_frac'] = mi1_pq['other'] / (mi1_pq['inh'] + mi1_pq['exc'] + mi1_pq['other']).clip(lower=1)

nn_vals = sorted(mi1_pq['n_neighbors'].unique())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) n_neighbors vs 総出力 syn — 入力と同程度に落ちるなら truncation を示唆
ax = axes[0]
data = [mi1_pq[mi1_pq['n_neighbors'] == n]['total_out'].values for n in nn_vals]
violin_strip(ax, data, nn_vals, 'tab:gray', np.random.default_rng(4))
ax.set(xlabel='# valid hex neighbors', ylabel='total OUTPUT syn per Mi1',
       title='Output also drops at edge?\n(yes => arbor truncation, not inh-specific)')
ax.grid(True, alpha=0.3, axis='y')

# (2) 総入力 vs 総出力 (端細胞が原点側に寄るか), 色 = n_neighbors
ax = axes[1]
sc = ax.scatter(mi1_pq['total_in'], mi1_pq['total_out'], c=mi1_pq['n_neighbors'],
                cmap='viridis', s=22, alpha=0.7, vmin=3, vmax=6, edgecolors='none')
plt.colorbar(sc, ax=ax, label='# neighbors')
ax.set(xlabel='total input syn', ylabel='total output syn',
       title='Per-cell input vs output (color = edge proximity)')
ax.grid(True, alpha=0.3)

# (3) 未分類 nt (other) の割合が端で増えるか = データ品質劣化のサイン
ax = axes[2]
data = [mi1_pq[mi1_pq['n_neighbors'] == n]['other_frac'].values for n in nn_vals]
violin_strip(ax, data, nn_vals, 'tab:green', np.random.default_rng(5))
ax.set(xlabel='# valid hex neighbors', ylabel="'other' (unclassified nt) fraction of input",
       title='Unclassified-nt fraction vs edge\n(rises at edge => proofreading/quality artifact)')
ax.grid(True, alpha=0.3, axis='y')

r_out, p_out = st.spearmanr(mi1_pq['n_neighbors'], mi1_pq['total_out'])
r_in, p_in = st.spearmanr(mi1_pq['n_neighbors'], mi1_pq['total_in'])
print(f"Spearman rho (n_neighbors vs total OUTPUT syn): {r_out:.3f} (p={p_out:.2g})")
print(f"Spearman rho (n_neighbors vs total INPUT syn) : {r_in:.3f} (p={p_in:.2g})")
print("If output drops as steeply as input, the edge deficit is likely arbor truncation / boundary,")
print("not a specifically reduced-inhibition phenomenon.")

plt.suptitle('F. Confound check: is the edge deficit cell-wide (truncation/quality) or inhibition-specific?',
             y=1.02, fontsize=12)
plt.tight_layout()


# %% [markdown]
# ## Q8. 抑制性インターニューロンの網羅的サーベイ
#
# これまでは Lai / Pm / Dm / Dm8 など個別の cell type を見てきたが、視覚系の inhibitory interneuron は他にも数多い (Sm 系、CT1、LPi、Li、Lat、Lawf 等)。それぞれが異なる neuropil・spatial scale で側抑制を提供しているはずなので、**全ての inh-dominant cell type をリストアップし、family ごとに集計**する。
#
# **family の定義** (FlyWire naming convention に基づく):
# - **Lai**: lamina amacrine (LA 内の側抑制)
# - **Lawf**: lamina wide-field (LA 内の wide-field 抑制)
# - **Dm**: distal medulla intrinsic (medulla 浅層、~M1-M6)
# - **Pm**: proximal medulla intrinsic (medulla 深層、~M7-M10)
# - **Sm**: small medulla intrinsic (modular, 色処理系に多い)
# - **Mi**: medulla intrinsic (ほとんどは exc dominant な columnar, 一部 inh)
# - **Tm/TmY**: transmedullary (ME→LO/LOP projection)
# - **CT1**: 全 medulla を 1 cell でカバーする超大型 GABA cell
# - **T1**: medulla intrinsic neuron (col 単位)
# - **C2/C3**: lamina→medulla centrifugal feedback
# - **Li / Lat / LT / LC / LPi / LPLC**: lobula / lobula plate 系
#
# 各 family について、cell type 数 / total inh syn / 代表 column spread を比較する。

# %%
# 全 inh-dominant cell type をリストアップして family 分類
def family_of(t):
    t = str(t)
    if t == 'CT1':                                       return 'CT1 (global)'
    if t == 'Lai':                                       return 'Lai (lamina amacrine)'
    if t == 'T1':                                        return 'T1 (medulla intrinsic)'
    if t.startswith('Lawf'):                             return 'Lawf (lamina wide-field)'
    if t.startswith('DmDRA') or t.startswith('Dm'):      return 'Dm (distal medulla)'
    if t.startswith('Pm'):                               return 'Pm (proximal medulla)'
    if t.startswith('Sm'):                               return 'Sm (small medulla)'
    if t.startswith('LPi'):                              return 'LPi (lobula plate intrinsic)'
    if t.startswith('LPLC') or t.startswith('LC'):       return 'LC/LPLC (lobula columnar)'
    if t.startswith('Lat'):                              return 'Lat (lobula amacrine)'
    if t.startswith('LT'):                               return 'LT (lobula tangential)'
    if t.startswith('Li'):                               return 'Li (lobula intrinsic)'
    if t.startswith('Mi'):                               return 'Mi (medulla intrinsic)'
    if t.startswith('TmY'):                              return 'TmY (transmedullary Y)'
    if t.startswith('Tm'):                               return 'Tm (transmedullary)'
    if t in ('C2', 'C3'):                                return 'C2/C3 (centrifugal)'
    if t.startswith('MeTu'):                             return 'MeTu (medulla-tubercular)'
    if t.startswith('MeMe'):                             return 'MeMe (medulla-medulla)'
    return 'other'

# inh-dominant (>=50% inh) かつ >=1000 inh syn を集める
inh_survey = type_io[(type_io['inh_frac'] >= 0.5) & (type_io['inh'] >= 1000)].copy()
inh_survey['family'] = inh_survey.index.map(family_of)
print(f'inh-dominant cell types (>=50% inh, >=1000 inh syn): {len(inh_survey)}')

# 各 type の home neuropil と top targets を計算
def neuropil_base(np_name):
    s = str(np_name)
    return s[:-2] if (s.endswith('_R') or s.endswith('_L')) else s

def type_stats(ctype):
    out = conn[conn['pre_primary_type'] == ctype]
    if len(out) == 0:
        return {'home_np': 'NA', 'home_share': 0.0, 'top_targets': ''}
    neuropils = out.groupby('neuropil')['syn_count'].sum()
    top_np = str(neuropils.idxmax())
    home = neuropil_base(top_np)
    home_share = sum(v for n, v in neuropils.items() if neuropil_base(str(n)) == home) / neuropils.sum()
    top_tgts = out.groupby('post_primary_type')['syn_count'].sum().sort_values(ascending=False).head(3)
    return {
        'home_np': home,
        'home_share': float(home_share),
        'top_targets': ', '.join(f'{t}({int(s):,})' for t, s in top_tgts.items()),
    }

n_cells_by_type = neurons['primary_type'].value_counts()
rows = []
for ctype in inh_survey.index:
    st = type_stats(ctype)
    rows.append({
        'type': ctype,
        'family': inh_survey.loc[ctype, 'family'],
        'n_cells': int(n_cells_by_type.get(ctype, 0)),
        'inh_syn': int(inh_survey.loc[ctype, 'inh']),
        'inh_frac': round(inh_survey.loc[ctype, 'inh_frac'], 2),
        'home_np': st['home_np'],
        'home_share': round(st['home_share'], 2),
        'col_spread':   round(per_type_col.loc[ctype, 'type_spread'], 2) if ctype in per_type_col.index else None,
        'n_target_cols': int(per_type_col.loc[ctype, 'type_n_cols']) if ctype in per_type_col.index else None,
        'top_targets': st['top_targets'],
    })
inh_detail = pd.DataFrame(rows).sort_values('inh_syn', ascending=False)

print()
print(f'Top 25 inhibitory cell types by total inh syn output:')
display(inh_detail.head(25))

# %%
# Family-level summary
fam_summary = (
    inh_detail.groupby('family')
    .agg(n_types=('type', 'size'),
         n_cells_total=('n_cells', 'sum'),
         inh_syn_total=('inh_syn', 'sum'),
         median_inh_frac=('inh_frac', 'median'),
         median_home_share=('home_share', 'median'),
         median_col_spread=('col_spread', 'median'),
         median_n_target_cols=('n_target_cols', 'median'))
    .sort_values('inh_syn_total', ascending=False)
)
print('Per-family summary (sorted by total inh syn):')
display(fam_summary)

# %%
# 可視化: (A) top 30 cell types by inh syn, colored by family
import numpy as np

top_n = 30
topN = inh_detail.head(top_n).copy()
family_list = list(topN['family'].unique())
palette = plt.cm.tab20(np.linspace(0, 1, max(len(family_list), 3)))
family_colors = dict(zip(family_list, palette))
colors_topN = [family_colors[f] for f in topN['family']]

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

ax = axes[0]
ax.barh(range(len(topN)), topN['inh_syn'], color=colors_topN, edgecolor='white')
ax.set_yticks(range(len(topN)))
ax.set_yticklabels(topN['type'], fontsize=8)
ax.invert_yaxis()
ax.set(xlabel='total inh syn output', title=f'Top {top_n} inhibitory cell types (colored by family)')

# family legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=family_colors[f], label=f) for f in family_list]
ax.legend(handles=handles, fontsize=7, loc='lower right', framealpha=0.9)

# (B) family-level column spread boxplot
ax = axes[1]
fam_for_box = fam_summary.index.tolist()
data_per_fam = [inh_detail[inh_detail['family'] == f]['col_spread'].dropna().values for f in fam_for_box]
sizes = [len(d) for d in data_per_fam]
# 中身があるものだけ
valid_idx = [i for i, s in enumerate(sizes) if s > 0]
data_valid = [data_per_fam[i] for i in valid_idx]
labels_valid = [f + f' (n={sizes[i]})' for i, f in zip(valid_idx, [fam_for_box[i] for i in valid_idx])]
colors_valid = [family_colors.get(fam_for_box[i], 'lightgray') for i in valid_idx]

bp = ax.boxplot(data_valid, tick_labels=labels_valid, patch_artist=True, showfliers=False, vert=False)
for patch, c in zip(bp['boxes'], colors_valid):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set(xlabel='column spread (hex units, weighted-mean Δcol from centroid)',
       title='Column spread distribution by family\n(only cells with columnar targets in column_assignment)')
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Q8: comprehensive survey of inhibitory interneurons in the FlyWire optic lobe', y=1.02, fontsize=12)
plt.tight_layout()

# %% [markdown]
# ## Q9. 多 cell type の端 vs 中央 — Q7 の一般性
#
# Q7 では Mi1 1 種類だけで「端は絶対量が減るが E/I balance は保たれる」結果を見た。これが Mi1 特有なのか、columnar 受け手全般に成り立つのか、column_assignment にある cell type で同じ解析を回す。
#
# 解析対象は `column_assignment.csv` から自動生成する。ただし R7/R8 は photoreceptor input cell なので、edge effect の受け手解析からは除外する。現在のデータでは 31 種中、R7/R8 を除いた 29 種を対象にする。
#
# **問い**:
# 1. **Spearman 相関**: n_neighbors と inh/exc 入力量の相関は全 cell type で正 (= 端で減る) か?
# 2. **E/I balance**: corner と interior の inh_frac は全 type で似ているか? それとも一部の type は「端で inh が相対的に増える (= compensation)」を示すか?
# 3. **family による違い**: 浅層 (LA: L1-L5) と中層 (ME: Mi/Tm) と深層 (LOP: T4/T5) で端処理の戦略が違うか?
#

# %%
# 多 cell type 用に edge_stats 関数化、各 type について edge categorization + inh/exc 集計
import scipy.stats as st
hex_neighbors_v = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

def edge_stats_for_type(cell_type, hemisphere='right', min_cells=30):
    cells = col_assign[(col_assign['type'] == cell_type) & (col_assign['hemisphere'] == hemisphere)]
    if len(cells) < min_cells:
        return None
    cpq = cells[['root_id', 'p', 'q']].copy()
    all_pq = set(zip(cpq['p'], cpq['q']))
    cpq['n_neighbors'] = cpq.apply(lambda r: sum((r['p']+dp, r['q']+dq) in all_pq for dp, dq in hex_neighbors_v), axis=1)
    cpq['edge_cat'] = pd.cut(cpq['n_neighbors'], bins=[-1, 3, 5, 6], labels=['corner', 'edge', 'interior'])
    cids = set(cpq['root_id'])
    inc = conn[conn['post_root_id'].isin(cids)]
    sbs = (inc.groupby(['post_root_id', 'sign'])['syn_count'].sum()
           .unstack(fill_value=0)
           .reindex(columns=['inh', 'exc', 'other'], fill_value=0))
    cpq = cpq.set_index('root_id').join(sbs, how='left').reset_index()
    for c in ['inh', 'exc', 'other']:
        cpq[c] = cpq[c].fillna(0).astype(int)
    cpq['inh_frac'] = cpq['inh'] / (cpq['inh'] + cpq['exc']).clip(lower=1)
    return cpq

# 解析対象 cell type: column_assignment 入りの cell type から photoreceptor input (R7/R8) を除く
excluded_edge_types = {'R7', 'R8'}
target_types = sorted(t for t in col_assign['type'].dropna().unique() if t not in excluded_edge_types)
print(f'target columnar receiver types: {len(target_types)} / {col_assign["type"].nunique()} column_assignment types')

# 各 type で集計
per_type_stats = {}
for t in target_types:
    df = edge_stats_for_type(t)
    if df is None:
        continue
    per_type_stats[t] = df

print(f'analyzed {len(per_type_stats)} cell types in right hemi')

# Per-cell-type 集計 (corner / interior の median + Spearman)
rows = []
for t, df in per_type_stats.items():
    sub_c = df[df['edge_cat'] == 'corner']
    sub_i = df[df['edge_cat'] == 'interior']
    if len(sub_c) < 5 or len(sub_i) < 30:
        continue
    rho_inh, p_inh = st.spearmanr(df['n_neighbors'], df['inh'])
    rho_exc, p_exc = st.spearmanr(df['n_neighbors'], df['exc'])
    rows.append({
        'cell_type': t,
        'n_cells': len(df),
        'n_corner': len(sub_c),
        'n_interior': len(sub_i),
        'corner_inh': float(sub_c['inh'].median()),
        'interior_inh': float(sub_i['inh'].median()),
        'corner_exc': float(sub_c['exc'].median()),
        'interior_exc': float(sub_i['exc'].median()),
        'inh_corner/interior': float(sub_c['inh'].median() / max(sub_i['inh'].median(), 1)),
        'exc_corner/interior': float(sub_c['exc'].median() / max(sub_i['exc'].median(), 1)),
        'corner_inh_frac': float(sub_c['inh_frac'].median()),
        'interior_inh_frac': float(sub_i['inh_frac'].median()),
        'inh_frac_delta': float(sub_c['inh_frac'].median() - sub_i['inh_frac'].median()),
        'rho_inh_vs_n_nbrs': round(rho_inh, 3),
        'rho_exc_vs_n_nbrs': round(rho_exc, 3),
    })
scaling_df = pd.DataFrame(rows).sort_values('inh_corner/interior')
print()
print(f'Per cell type: corner vs interior comparison ({len(scaling_df)} types with enough samples)')
display(scaling_df.round(3))

# %%
# 可視化: (A) corner/interior 比のスキャッタ (inh vs exc軸), (B) inh_frac shift, (C) per-type radial-ish bar
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (A) corner/interior ratio: inh vs exc
ax = axes[0]
xs = scaling_df['exc_corner/interior'].values
ys = scaling_df['inh_corner/interior'].values
ax.scatter(xs, ys, s=50, alpha=0.7, color='tab:purple', edgecolors='black', linewidths=0.5)
m = max(xs.max(), ys.max(), 1.05) * 1.05
ax.plot([0, m], [0, m], 'k--', lw=0.7, alpha=0.5, label='y = x (proportional)')
ax.axhline(1.0, ls=':', color='gray', alpha=0.5)
ax.axvline(1.0, ls=':', color='gray', alpha=0.5)
for _, r in scaling_df.iterrows():
    ax.annotate(r['cell_type'], (r['exc_corner/interior'], r['inh_corner/interior']),
                fontsize=7, alpha=0.85, xytext=(3, 3), textcoords='offset points')
ax.set(xlim=(0, m), ylim=(0, m),
       xlabel='exc corner/interior ratio',
       ylabel='inh corner/interior ratio',
       title='Edge scaling: inh vs exc reduction ratio per cell type\n(above y=x: exc drops more than inh = relative inh compensation)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (B) inh_frac shift: corner vs interior
ax = axes[1]
xs = scaling_df['interior_inh_frac'].values
ys = scaling_df['corner_inh_frac'].values
ax.scatter(xs, ys, s=50, alpha=0.7, color='tab:green', edgecolors='black', linewidths=0.5)
lim = max(xs.max(), ys.max(), 0.5) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.7, alpha=0.5, label='y = x (preserved)')
for _, r in scaling_df.iterrows():
    ax.annotate(r['cell_type'], (r['interior_inh_frac'], r['corner_inh_frac']),
                fontsize=7, alpha=0.85, xytext=(3, 3), textcoords='offset points')
ax.set(xlim=(0, lim), ylim=(0, lim),
       xlabel='interior inh fraction', ylabel='corner inh fraction',
       title='E/I balance shift at the edge per cell type\n(above y=x: corner has more inh share = compensation)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (C) bar chart of corner/interior inh ratio per type, sorted
ax = axes[2]
d = scaling_df.sort_values('inh_corner/interior', ascending=False)
colors_bar = ['tab:green' if r > 0.9 else ('tab:orange' if r > 0.6 else 'tab:red')
              for r in d['inh_corner/interior']]
ax.barh(range(len(d)), d['inh_corner/interior'], color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(d)))
ax.set_yticklabels(d['cell_type'], fontsize=8)
ax.invert_yaxis()
ax.axvline(1.0, color='black', ls='--', alpha=0.5, label='1.0 (no edge effect)')
ax.set(xlabel='corner/interior inh ratio', title='How much inh drops at the edge per cell type\n(green: <10% drop, orange: 10-40%, red: >40%)')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle(f'Q9: edge vs center across {len(scaling_df)} columnar cell types — is Q7 Mi1 pattern universal?', y=1.03, fontsize=12)
plt.tight_layout()

# %% [markdown]
# ## Q9 補足: 入力非対称性の空間フィールド (位置で集計しない生の空間マップ)
#
# Q7/Q9 は inh/exc の「総量」を, また corner/edge/interior の分類は「位置を 3 段階に集計」していた。
# エッジ仮説の核心は **「端 column は入力を集める近傍が片側で切れている／非対称か」** という空間構造であり、
# これは位置で集計 (binning) した瞬間に潰れる。そこで **空間情報を一切潰さず**、各 column を実際の (p,q) 位置に
# 置いたまま、その column 固有の **入力の片寄り (lateral input centroid offset) をベクトルとして** 描く。
#
# **各 column の指標 (集計なし, 1 cell = 1 ベクトル)**:
# 1. 受け手 = `column_assignment` 入りの columnar cell type (Q9 と同じ 29 種、R7/R8 除く、right hemi)
# 2. その cell の presynaptic partner のうち **column 座標を持つ columnar source** を home からの相対 (Δp, Δq) に
#    (Dm/Pm/Lai/CT1 等の wide-field 抑制源は座標を持たず除外 → exc に厚く inh に薄い bias。coverage 併記)
# 3. home (Δ=0) を除いた周辺 source の **syn 重み付き重心 = offset ベクトル** を計算。対称なら ~0, 片側欠損なら外れる
# 4. これを **実際の hex 位置に矢印で描く**: 端 column は内側向き (入力源が内側に偏る) の長い矢印, 中央は ~0 のはず
#
# **見方** (どれも位置で集計しない):
# - **A. 空間ベクトル場**: 代表 type で全 column を真の位置に置き, offset ベクトルを矢印 + |offset| でヒートマップ
# - **B. 個別 footprint montage**: 最も非対称な cell を 1 個ずつ生 footprint で (平均なし)
# - **C. 全 type の空間マップ**: 29 種すべてを small multiples で並べ, rim の非対称が type 横断で出るか空間的に確認

# %%
# Q9 補足の前処理: 各 columnar 受け手 type について source-column footprint を集める
hemi = 'right'
foot_rows = []
coverage_rows = []

# 境界法線の推定方式 (改良版):
#   旧: 直近 6 近傍のうち「欠けている」方向のベクトル和 -> 離散で ±30° 量子化, 内部の単発穴に弱い, 両側欠損で相殺。
#   新: 半径 R_NORM 内の "存在する" 同 type 細胞の重心方向の逆 (= 存在密度勾配の外向き)。
#       多数の present 細胞で決まるので連続的な角度が出て, 内部の単発穴の影響も希釈される。
#   edge_cat の定義 (corner/edge/interior) は Q9 と揃えるため直近 6 近傍のままにする。
R_NORM = 3
def _hexdist(dp, dq):
    return (abs(dp) + abs(dq) + abs(dp + dq)) / 2
_norm_offsets = [(dp, dq, *axial_to_cart(dp, dq))
                 for dp in range(-R_NORM, R_NORM + 1)
                 for dq in range(-R_NORM, R_NORM + 1)
                 if not (dp == 0 and dq == 0) and _hexdist(dp, dq) <= R_NORM]

def boundary_normal(p, q, all_pq):
    """半径 R_NORM 内の present 細胞重心の逆向き = 外向き境界法線の角度と強度を返す。"""
    vx = vy = 0.0
    for dp, dq, x, y in _norm_offsets:
        if (p + dp, q + dq) in all_pq:
            vx += x
            vy += y
    return float(np.arctan2(-vy, -vx)), float(np.hypot(vx, vy))

for t in target_types:
    cells = col_assign[(col_assign['type'] == t) & (col_assign['hemisphere'] == hemi)]
    if len(cells) < 30:
        continue
    cpq = cells[['root_id', 'p', 'q']].drop_duplicates('root_id').copy()
    all_pq = set(zip(cpq['p'], cpq['q']))
    cpq['theta'] = [boundary_normal(p, q, all_pq)[0] for p, q in zip(cpq['p'], cpq['q'])]
    home = cpq.set_index('root_id')
    cids = set(cpq['root_id'])
    inc = conn[conn['post_root_id'].isin(cids) & conn['sign'].isin(['inh', 'exc'])].copy()
    denom = inc.groupby('sign')['syn_count'].sum()  # coverage 分母 (全 inh/exc 入力)
    inc['p_src'] = inc['pre_root_id'].map(col_map['p'])
    inc['q_src'] = inc['pre_root_id'].map(col_map['q'])
    inc['hemi_src'] = inc['pre_root_id'].map(col_map['hemisphere'])
    inc = inc.dropna(subset=['p_src', 'q_src'])
    inc = inc[inc['hemi_src'] == hemi]
    numer = inc.groupby('sign')['syn_count'].sum()  # columnar source で配置できた分
    coverage_rows.append({
        'type': t,
        'exc_cov': float(numer.get('exc', 0) / max(denom.get('exc', 0), 1)),
        'inh_cov': float(numer.get('inh', 0) / max(denom.get('inh', 0), 1)),
        'n_cells': len(cpq),
    })
    if len(inc) == 0:
        continue
    inc['p0'] = inc['post_root_id'].map(home['p'])
    inc['q0'] = inc['post_root_id'].map(home['q'])
    inc['theta'] = inc['post_root_id'].map(home['theta'])
    inc['dp'] = (inc['p_src'] - inc['p0']).astype(int)
    inc['dq'] = (inc['q_src'] - inc['q0']).astype(int)
    foot_rows.append(inc[['post_root_id', 'dp', 'dq', 'syn_count', 'sign', 'theta']].assign(type=t))

foot = pd.concat(foot_rows, ignore_index=True)
coverage = pd.DataFrame(coverage_rows).set_index('type')

# 全 column の実 (p,q) 位置 (right hemi)。空間マップで矢印を置く座標として使う。
id_pq = (col_assign[col_assign['hemisphere'] == hemi]
         .drop_duplicates('root_id').set_index('root_id')[['p', 'q']])

# 各 column につき 1 ベクトル: home (Δ=0) を除いた columnar source の syn 重み付き重心 = offset。
#   対称な surround -> ~0, 片側が欠ける (端) -> 内側を向く長いベクトル。位置での集計 (binning) は一切しない。
def cell_offsets(sign):
    lat = foot[(foot['sign'] == sign) & ~((foot['dp'] == 0) & (foot['dq'] == 0))].copy()
    lx, ly = axial_to_cart(lat['dp'].values, lat['dq'].values)
    lat['lx'] = lx * lat['syn_count']
    lat['ly'] = ly * lat['syn_count']
    g = lat.groupby(['type', 'post_root_id']).agg(
        lx=('lx', 'sum'), ly=('ly', 'sum'), w=('syn_count', 'sum'), n_src=('dp', 'size'))
    g['ox'] = g['lx'] / g['w']     # offset ベクトル (片側欠損なら内側を向く)
    g['oy'] = g['ly'] / g['w']
    g['offset'] = np.hypot(g['ox'], g['oy'])
    g = g.reset_index()
    g['p'] = g['post_root_id'].map(id_pq['p'])
    g['q'] = g['post_root_id'].map(id_pq['q'])
    g = g.dropna(subset=['p', 'q'])
    g['x'], g['y'] = axial_to_cart(g['p'].values, g['q'].values)
    return g

cell_exc = cell_offsets('exc')
cell_inh = cell_offsets('inh')
print(f'per-cell offset vectors: exc={len(cell_exc):,}, inh={len(cell_inh):,} '
      f'(types: {cell_exc["type"].nunique()})')
print(f'columnar-source coverage of input syn (median across types): '
      f'exc {coverage["exc_cov"].median():.1%}, inh {coverage["inh_cov"].median():.1%} '
      f'(inh は wide-field 源が多く低い: 構造上の bias)')

# %%
# A. 空間ベクトル場: 代表 type を選び, 全 column を真の hex 位置に置いて offset ベクトルを矢印で描く。
#    背景 hex を |offset| で塗る。端の column は内側向きの長い矢印 (入力源が内側に偏る) になるはず。集計なし。
rep = coverage[coverage['exc_cov'] >= 0.3].copy()
rep['n_exc_cells'] = rep.index.map(cell_exc['type'].value_counts()).fillna(0)
montage_type = rep.sort_values('n_exc_cells', ascending=False).index[0]

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
for ax, (g, sgn) in zip(axes, [(cell_exc, 'exc'), (cell_inh, 'inh')]):
    gg = g[g['type'] == montage_type]
    if len(gg) == 0:
        ax.set_title(f'{montage_type} {sgn}: no data'); ax.set_axis_off(); continue
    sc = ax.scatter(gg['x'], gg['y'], c=gg['offset'], cmap='viridis', s=170, marker='H',
                    edgecolors='none', alpha=0.9)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='|lateral input centroid offset| (hex)')
    ax.quiver(gg['x'], gg['y'], gg['ox'], gg['oy'], angles='xy', scale_units='xy',
              scale=0.4, width=0.003, color='black', alpha=0.85)
    ax.set(aspect='equal', title=f'{montage_type} — {sgn} columnar input offset field (n={len(gg)} columns)')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('A. Per-column input-asymmetry vector field at true hex positions (NO spatial binning)\n'
             'Arrow = where each column gathers its columnar input from; edge columns point inward, center ~0',
             y=1.02, fontsize=12)
plt.tight_layout()

# %%
# B. 個別 footprint montage: 代表 type で最も非対称な (offset 大) column を 1 個ずつ生 footprint 表示。
#    平均なし。home=星, 緑矢印=境界法線方向, 黄矢印=実際の入力 offset 方向。source は境界側で欠けるはず。
gg = cell_exc[cell_exc['type'] == montage_type].sort_values('offset', ascending=False)
show_ids = gg['post_root_id'].head(12).tolist()
off_xy = gg.set_index('post_root_id')[['ox', 'oy']]
mt = foot[(foot['type'] == montage_type) & (foot['sign'] == 'exc') & foot['post_root_id'].isin(show_ids)].copy()
cell_theta = mt.drop_duplicates('post_root_id').set_index('post_root_id')['theta']

ncol = 4
nrow = int(np.ceil(len(show_ids) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
vmax_m = mt['syn_count'].max()
for ax, pid in zip(axes, show_ids):
    d = mt[mt['post_root_id'] == pid]
    sx, sy = axial_to_cart(d['dp'].values, d['dq'].values)
    sc = ax.scatter(sx, sy, c=d['syn_count'], cmap='hot_r', s=120, marker='H',
                    edgecolors='black', linewidths=0.3, vmin=0, vmax=vmax_m)
    ax.plot(0, 0, '*', color='cyan', markersize=16, markeredgecolor='black', markeredgewidth=0.5)
    theta = cell_theta.get(pid, 0.0)
    ax.annotate('', xy=(2.2 * np.cos(theta), 2.2 * np.sin(theta)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='lime', lw=2))
    ox, oy = off_xy.loc[pid]
    ax.annotate('', xy=(4 * ox, 4 * oy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='gold', lw=2))
    ax.set(aspect='equal', title=f'{int(d["dp"].nunique())} src cols, offset={np.hypot(ox, oy):.2f}')
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(show_ids):]:
    ax.set_axis_off()
fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, label='syn from each source column')
plt.suptitle(f'B. Raw per-cell columnar exc input footprint — {montage_type} most-asymmetric columns\n'
             '(star = home, green = boundary-normal, gold = input offset (x4); sources clipped on the boundary side)',
             y=1.02, fontsize=12)

# %%
# C. 全 type の空間マップ: 29 種を small multiples で。各 type 全 column を真位置に置き |offset| 塗り + 矢印。
#    位置で集計しない。rim の column が type 横断で一貫して光る (内側向き矢印) なら非対称は普遍的かつ空間的。
types_c = sorted(cell_exc['type'].unique())
ncol = 6
nrow = int(np.ceil(len(types_c) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
axes = np.atleast_1d(axes).ravel()
vmax_c = float(np.percentile(cell_exc['offset'], 97))
for ax, t in zip(axes, types_c):
    gg = cell_exc[cell_exc['type'] == t]
    sc = ax.scatter(gg['x'], gg['y'], c=gg['offset'], cmap='viridis', s=16, marker='H',
                    edgecolors='none', vmin=0, vmax=vmax_c)
    ax.quiver(gg['x'], gg['y'], gg['ox'], gg['oy'], angles='xy', scale_units='xy', scale=0.6,
              width=0.005, color='black', alpha=0.6)
    ax.set(aspect='equal', title=f'{t} (n={len(gg)})')
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(types_c):]:
    ax.set_axis_off()
fig.colorbar(sc, ax=axes.tolist(), shrink=0.5, label='|lateral input centroid offset| (hex)')
plt.suptitle('C. Per-column input-asymmetry field across all 29 columnar receiver types (no spatial aggregation)\n'
             'Rim columns light up with inward offset consistently -> edge input-field asymmetry is general & spatial',
             y=1.01, fontsize=13)


# %% [markdown]
# ### Q9 補足 (続き): lateral 抑制 vs bottom-up 興奮 の空間マップ
#
# これまでの footprint は exc/inh を sign で分けていたが、exc 側には **home column の bottom-up 入力 (Δcol=0,
# feedforward drive) と lateral exc が混在**していた。より機能的な対比は **「surround 抑制 vs feedforward 駆動」**:
#
# - **bottom-up 興奮** = home column (Δcol=0) からの exc syn (feedforward drive, 例 L1->Mi1)。exc は座標を持つので ~90% 配置可能。
# - **lateral 抑制** = その column が受ける **総 inh syn (sign ベース)**。視覚葉の columnar cell への抑制はほぼ wide-field
#   surround 由来なので lateral と見なせる。sign ベースなので **全カバレッジ** (wide-field 源の座標問題を回避)。
#
# 各 column で比 **R = lateral_inh / bottom_up_exc** を計算し、真の hex 位置に色付け (位置で集計しない)。
# これは Q7 の E/I バランスを「lateral exc 混入を除いた」正しい形にしたもの。**端で R が落ちれば surround 抑制が
# feedforward に対して相対的に弱まる (補償なし)、上がれば相対的に強まる (補償あり)** ことを意味する。

# %%
# bottom-up 興奮: home column (Δcol=0) の exc syn (foot は同半球 columnar source なので feedforward 入力を捉える)
bottom_up = (foot[(foot['sign'] == 'exc') & (foot['dp'] == 0) & (foot['dq'] == 0)]
             .groupby(['type', 'post_root_id'])['syn_count'].sum()
             .reset_index().rename(columns={'syn_count': 'bottom_up_exc'}))

# lateral 抑制: その column が受ける総 inh syn (sign ベース・全カバレッジ; foot ではなく conn から直接)
recv = (col_assign[(col_assign['hemisphere'] == hemi) & col_assign['type'].isin(target_types)]
        .drop_duplicates('root_id')[['root_id', 'type', 'p', 'q']].copy())
inh_tot = (conn[conn['post_root_id'].isin(set(recv['root_id'])) & (conn['sign'] == 'inh')]
           .groupby('post_root_id')['syn_count'].sum())
recv['lateral_inh'] = recv['root_id'].map(inh_tot).fillna(0.0)
recv = recv.merge(bottom_up.rename(columns={'post_root_id': 'root_id'}), on=['type', 'root_id'], how='left')
recv['bottom_up_exc'] = recv['bottom_up_exc'].fillna(0.0)
# bottom-up が極小の column は比が不安定なので NaN (灰色) にする
recv['ratio'] = np.where(recv['bottom_up_exc'] >= 5,
                         recv['lateral_inh'] / recv['bottom_up_exc'].clip(lower=1), np.nan)
recv['x'], recv['y'] = axial_to_cart(recv['p'].values, recv['q'].values)
print(f'columns with usable ratio: {recv["ratio"].notna().sum():,} / {len(recv):,}')
print(f'ratio (lateral_inh / bottom_up_exc) median = {recv["ratio"].median():.2f}, '
      f'IQR = [{recv["ratio"].quantile(.25):.2f}, {recv["ratio"].quantile(.75):.2f}]')

# %%
# (i) 代表 type を大きく + (ii) 全 type small multiples。色 = R (surround 抑制 / feedforward 駆動)。集計なし。
vmax_r = float(np.nanpercentile(recv['ratio'], 95))

fig, ax = plt.subplots(figsize=(8.5, 7))
gg = recv[recv['type'] == montage_type]
sc = ax.scatter(gg['x'], gg['y'], c=gg['ratio'], cmap='magma', s=150, marker='H',
                edgecolors='none', vmin=0, vmax=vmax_r)
plt.colorbar(sc, ax=ax, shrink=0.8, label='R = lateral inh / bottom-up exc')
ax.set(aspect='equal', title=f'{montage_type}: surround-inhibition / feedforward-drive ratio per column\n'
       '(bright = inhibition-heavy; does the rim differ from the center?)')
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()

# %%
types_c = sorted(recv['type'].unique())
ncol = 6
nrow = int(np.ceil(len(types_c) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
axes = np.atleast_1d(axes).ravel()
for ax, t in zip(axes, types_c):
    gg = recv[recv['type'] == t]
    vmx = float(np.nanpercentile(gg['ratio'], 95)) if gg['ratio'].notna().any() else 1.0
    sc = ax.scatter(gg['x'], gg['y'], c=gg['ratio'], cmap='magma', s=16, marker='H',
                    edgecolors='none', vmin=0, vmax=vmx)
    ax.set(aspect='equal', title=f'{t} (n={gg["ratio"].notna().sum()})')
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(types_c):]:
    ax.set_axis_off()
plt.suptitle('Lateral inhibition / bottom-up excitation ratio per column, all columnar receivers\n'
             '(per-panel 95pct colour scale; bright rim = surround inhibition stronger relative to feedforward at the edge)',
             y=1.01, fontsize=13)
plt.tight_layout()


# %% [markdown]
# ## まとめ

# %%
summary = {
    "Q1_inhibitory_share_of_IE":      f"{ie_share:.1%}",
    "Q1_excitatory_share_total":      f"{exc/total:.1%}",
    "Q1_inhibitory_share_total":      f"{inh/total:.1%}",
    "Q2_active_types":                 n_active,
    "Q2_types_inh_dominant":           n_mostly_i + n_pure_inh,
    "Q2_pure_inh_(>=95%)":             n_pure_inh,
    "Q3_viable_types":                 len(viable),
    "Q3_inh_self>exc_self":            n_above,
    "Q4_inh_dominant_types":           len(inh_t),
    "Q4_exc_dominant_types":           len(exc_t),
    "Q4_observable_synapse_coverage":  f"{ec_all['syn_count'].sum() / conn['syn_count'].sum():.1%}",
    "Q4_Delta_col_spread_inh/exc":     round(spread_ratio_col, 2),
    "Q4_target_columns_ratio_inh/exc": round(ncol_ratio, 2),
    "Q6_Dm8a_R7_input_cols_median":    dm8_input_summary.get(("Dm8a", "R7"), {}).get("median_n_input_cols", np.nan),
    "Q6_Dm8b_R7_input_cols_median":    dm8_input_summary.get(("Dm8b", "R7"), {}).get("median_n_input_cols", np.nan),
    "Q6_Dm8a_out/in_col_ratio":        round(dm8_io_summary.get("Dm8a", {}).get("median_out_in_ratio", np.nan), 2),
    "Q6_Dm8b_out/in_col_ratio":        round(dm8_io_summary.get("Dm8b", {}).get("median_out_in_ratio", np.nan), 2),
    "Q7_inh_median_corner":            int(mi1_pq[mi1_pq['edge_cat']=='corner (<=3 nbrs)']['inh'].median()),
    "Q7_inh_median_interior":          int(mi1_pq[mi1_pq['edge_cat']=='interior (6 nbrs)']['inh'].median()),
    "Q7_corner_as_fraction_of_interior": round(mi1_pq[mi1_pq['edge_cat']=='corner (<=3 nbrs)']['inh'].median() / mi1_pq[mi1_pq['edge_cat']=='interior (6 nbrs)']['inh'].median(), 2),
    "Q7_inh_frac_corner":              round(mi1_pq[mi1_pq['edge_cat']=='corner (<=3 nbrs)']['inh_frac'].median(), 2),
    "Q7_inh_frac_interior":            round(mi1_pq[mi1_pq['edge_cat']=='interior (6 nbrs)']['inh_frac'].median(), 2),
    "Q9_analyzed_cell_types":          len(scaling_df),
}
for k, v in summary.items():
    print(f"  {k:38s} = {v}")


# %% [markdown]
# ### 連結体から見える側抑制の実態
#
# - **Q1** 全シナプスの **約 56%** が興奮性 (ACH)、**約 44%** が抑制性 (GABA+GLUT+HIS)。`I/(I+E) ≈ 44%`。視覚系の入力配線の半分近くを抑制が占めるという結果は「至るところで抑制がある」という主張と整合する。
#
# - **Q2** 主要 cell type (>=1000 outgoing syn) 344 個のうち **約 60%** (206 個) が抑制性出力 dominant、**196 個 (57%)** は >=95% 抑制 (= pure inhibitory)。抑制ニューロンは少数の専門集団ではなく、cell type レベルで広く分布する。
#
# - **Q3** within-type の自タイプ抑制を出せる程に E/I 両方を持つ type は 58 個と少ない。そのうち抑制側の自タイプ率が興奮側より高いのは **24/58 (41%)**。within-type lateral inhibition は強いシグナルではなく、**多くの lateral inhibition は同タイプ内ではなく専用の抑制 interneuron 経由で起きる** ことを示唆。
#
# - **Q4** column_assignment.csv (別 notebook で validate 済) を使い、**Δcolumn 単位**で各 pre neuron の出力先 columnar cells の lateral spread を測定。sign は Q2 と同じ synapse-weighted な全出力 sign で分類した:
#   - Q4 で直接観測できるのは post が同半球 columnar target の synapse で、全 optic-lobe synapse の **約 35%**。この制約は Dm8 / Sm / CT1 など一部 type を過小評価する方向の bias。
#   - **INH-dominant cell type は EXC-dominant cell type より lateral spread が 3.35 倍、ターゲット column 数で 5.73 倍広い**。3D voxel spread の confound (stratification 軸の伸びが混入) を column 単位で排除しても、wide-field inhibition が clear に検出された。
#   - 具体例: **Pm08 (73 column) ≫ Pm04 (62) > Dm12 (30) > Dm4 (23) ≫ Mi1 (12, spread 0.74) ≈ Tm9 (11)** ≫ L2 (7 cols, spread 0.24)。Pm/Dm 系の抑制 interneuron は 1 個体で数十柱を支配しており、これが lateral inhibition の物理的基盤。
#
# - **Q5**
#   - **Lai**: 出力 neuropil は予想通り LA_R/LA_L 主体で、ターゲットは L1/L2/L3/R1-6/Lai (古典回路を再現)。シナプス数で重み付けすると **GABA 30,505 > ACH 28,874** で僅かに GABA dominant。Q4 でも Lai は 9 column 程度をカバー (lamina amacrine らしい中規模 lateral)。
#   - **Dm 24 種 / Pm 14 種** のうち多数が GABA/GLUT dominant で Medulla の Mi/Tm 系を広く支配 → 教科書的な wide-field inhibition と整合 (Dm4, Dm12, Dm3p, Pm 全種など)。
#   - **Dm9 は例外**: 文献では GABA 性とされるが FlyWire の ML 予測では出力の **97% が ACH** (シナプス数ベース)。Q4 でも Dm9 は 14 column / spread 1.13 と中規模 wide-field な形をしており、形態は Dm 系列と矛盾しない — つまり ML の nt_type 予測の誤りか、Dm9 ラベルに別細胞が混入している可能性が高い。
#
# - **Q6** Dm8 は Q4 output metric では直接評価できないため、R7 input footprint と Mi1-proxy output footprint を併用した。Dm8a / Dm8b は R7 input が median 7 columns、output は median 4-5 columns 程度で、**wide input integration + narrower home-column output** という古典的 Dm8 architecture と整合。
#
# - **Q7** 端 column の Mi1 と中央 column の Mi1 を比較 — **connectome 上の source-switching 型 compensation は存在しない**:
#   - 端 (corner, ≤3 hex 近傍) の Mi1 が受ける inh syn は中央 (6 近傍) の **約 42%** (median 131 vs 315)
#   - exc syn も同様に減る (73 vs 138)
#   - **E/I 比率はほぼ保たれている** (corner 0.66, interior 0.69)
#   - 入力源の内訳 (CT1 global / Dm/Pm local / Lai) も corner と interior で **比率は同じ** → CT1 等の global source による補償は無い (CT1 は Mi1 入力としてそもそも極小)
#   - つまり連結体レベルでは "**端ではすべての入力が proportionally 減るが E/I balance だけ保たれる**" 仕様。視覚処理の境界では絶対応答強度が弱くなることを許容している可能性がある。
#
# - **Q9** R7/R8 を除く column_assignment cell type **29 種**で Q7 と同じ解析を回すと、全 type で n_neighbors と inh/exc 入力量が正相関。edge effect は普遍的だが、T4/T5 や Tm3/Tm4/Tm21 では相対的に弱い。大半の type で `inh_frac` shift は小さく、E/I balance は保たれる。
#
# → 結論: 連結体レベルでは、ショウジョウバエ視覚系における側抑制は **(1)** 全シナプスの約 44% を占める量的な広さ、**(2)** 多数の専用 inhibitory interneuron 群の存在、**(3)** Δcolumn 単位で抑制 cell type が興奮 cell type の **約 3.35 倍 lateral に広がる**物理的構造、**(4)** Lamina/Medulla 両方での古典的回路の再現、という四点で支持される。
#
# ### 残る限界
#
# - `nt_type` の多くは ML 予測。`nt_type_score` の確信度フィルタを入れていない (`nt_type_score < 0.5` 除外で数値はやや変わる)。Dm9 のように literature と食い違うラベルは個別に検証必要。
# - GLUT を一律抑制扱いした。Drosophila でも GLUT 興奮性シナプスは存在する。
# - Q4 の column-based metric は post 側が column_assignment にある (= columnar) target のみを見ている。Dm/Pm/Sm/CT1 等が他の wide-field cell に向けて出すシナプスは除外される。これは true output spread を過小評価する方向の bias。
# - Mi1-proxy による Dm8 output column 投影は近似であり、M-layer / depth による誤差が残る。
# - 「lateral inhibition の機能的検証」には activity / behavioral データが必要。
#
