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
# # 側抑制の厳密な再評価 — `src/lateral/` 基盤のデモ
#
# 既存の側抑制解析には2つの方法論的欠陥があった:
#
# 1. **「同じ階層」を判定していない** — Δcolumn(横方向)は厳密だが、抑制 pre/post が *同一処理段
#    (neuropil)* にあるかを検査していない。段をまたぐ feedforward/feedback 抑制が「広いだけ」で
#    真の側抑制と混同され得る。
# 2. **直接抑制 vs 介在ニューロン経由を分離していない** — 看板 metrics(I/E 比・spread)は両者を混ぜていた。
#
# **文献的定義**(網膜の水平/アマクリン細胞、ショウジョウバエの Lai/Dm/Pm): 側抑制 = *介在ニューロンが
# 近傍をプールし抑制を返す* center-surround。すなわち (a) **同一 neuropil 内**(lamina/medulla/lobula/
# lobula plate; Nern et al. 2024)で、(b) **カラム方向に広がり**、(c) **多くは介在経由(=間接)**。
#
# このノートは `src/lateral/` 基盤でこれらを直交軸(stage × Δcolumn × sign × path-length)として扱い、
# 主張を分解して再評価する。**既存ノートの数値は不変**(本ノートは別経路)。

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
if (REPO_ROOT / "src").is_dir() is False:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_DIR
from src.data import FlyWireDataManager
from src.lateral import (
    LateralInhibitionCriteria,
    RadialKernels,
    add_sign,
    assign_stage_from_manager,
    classify_inhibition,
    lateral_inhibition_index,
    load_column_assignment,
    rms_radius,
)

EXC_COLOR, INH_COLOR = "tab:red", "tab:blue"  # repo convention: excitation=red, inhibition=blue

m = FlyWireDataManager()
conn = add_sign(m.optic_lobe_connections_df.copy())
col_assign = load_column_assignment(DATA_DIR)
stage = assign_stage_from_manager(m)
print(f"neurons={len(m.optic_lobe_neurons_df):,}  edges={len(conn):,}  staged={len(stage):,}")

# %% [markdown]
# ## R1. ステージ割当の生物学的検証
#
# `assign_stage` は ① `visual_neuron_types.family`(文献キュレーション)② `neuropil_synapse_table` の
# 優勢入出力 neuropil(データ駆動)③ `flow`(intrinsic か)を統合する。既知の細胞型でアンカー検証する。

# %%
st = stage.merge(
    m.get_visual_neuron_types_df()[["root_id", "type"]].drop_duplicates("root_id"),
    on="root_id", how="left",
)


def anchor(label, mask, col, expect):
    sub = st[mask]
    return dict(group=label, n=int(len(sub)),
               frac=round(float((sub[col] == expect).mean()), 3) if len(sub) else np.nan,
               check=f"{col}=={expect}")


anchors = pd.DataFrame([
    anchor("L1-L5 (lamina monopolar)", st["type"].isin(["L1", "L2", "L3", "L4", "L5"]), "stage", "LA"),
    anchor("Dm* (distal medulla)", st["type"].astype(str).str.startswith("Dm"), "stage", "ME"),
    anchor("Pm* (proximal medulla)", st["type"].astype(str).str.startswith("Pm"), "stage", "ME"),
    anchor("Mi* (medulla intrinsic)", st["type"].astype(str).str.startswith("Mi"), "stage", "ME"),
    anchor("T4* input", st["type"].astype(str).str.startswith("T4"), "input_stage", "ME"),
    anchor("T4* output", st["type"].astype(str).str.startswith("T4"), "output_stage", "LOP"),
    anchor("T5* input", st["type"].astype(str).str.startswith("T5"), "input_stage", "LO"),
    anchor("T5* output", st["type"].astype(str).str.startswith("T5"), "output_stage", "LOP"),
    anchor("R* (photoreceptors)", st["type"].astype(str).str.startswith("R"), "stage", "RETINA"),
])
print("Stage assignment vs known biology:")
display(anchors)

print("\nfamily-curated vs neuropil-derived stage agreement:")
print(st["stage_confidence"].value_counts(dropna=False).to_string())

# %% [markdown]
# 既知のアンカーは ~96-100% 一致。`stage_confidence == mismatch` の主因は **lamina monopolar (L1-L5)**:
# 同定上は lamina (family→LA) だが、**シナプス入力数では medulla 優勢**(光受容体入力は少数の大シナプス、
# medulla 側の入力が数で勝る)。これは欠陥ではなく、`stage_confidence` が surface すべき実在の非自明性で、
# `home stage`(family 優先)を採る設計判断の根拠でもある。

# %%
fig, ax = plt.subplots(figsize=(8, 4))
order = ["RETINA", "LA", "ME", "AME", "LO", "LOP", "central", "other"]
counts = st["stage"].value_counts().reindex(order).dropna()
counts.plot.bar(ax=ax, color="slategray", edgecolor="white")
ax.set(ylabel="# neurons", title="Optic-lobe neurons by assigned processing stage")
plt.tight_layout()

# %% [markdown]
# ## R2. 「至るところで側抑制」を分解する【欠陥1への回答】
#
# 全抑制シナプスを `classify_inhibition` でラベル付けし、**same-stage lateral(direct / wide-field)**
# と **cross-stage feedforward / feedback** に分ける。`same_stage_def="home"`(home stage 一致が
# lateral vs FF/FB を分ける主信号)、`min_offset_cols=1`(home column を除く)。

# %%
crit = LateralInhibitionCriteria(min_syn=5, same_stage_def="home", min_offset_cols=1.0)
cl = classify_inhibition(conn, stage, col_assign=col_assign, criteria=crit)

idx = lateral_inhibition_index(cl)
print("Inhibitory synapse budget (fraction of all inhibitory synapses, min_syn>=5):")
display(idx.round(3).to_frame("fraction"))

by_label = cl.groupby("label")["syn_count"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
lateral_labels = {"direct_lateral", "wide_field_lateral"}
colors = ["tab:green" if lbl in lateral_labels else
          ("tab:orange" if "feed" in lbl else "lightgray") for lbl in by_label.index]
by_label.plot.bar(ax=ax, color=colors, edgecolor="white")
ax.set(ylabel="# inhibitory synapses",
       title="Where inhibition actually sits (green=same-stage lateral, orange=feedforward/feedback)")
plt.tight_layout()

# %% [markdown]
# **読み方**: 単純な「抑制シナプスの割合」(従来の I/E 比)は same-stage 側抑制と段またぎ FF/FB 抑制を
# 区別しない。この分解で、抑制のうち真に *同一段・横方向* なのはどれだけか、そのうち **direct(隣接カラム
# 単シナプス)か wide-field(座標を持たない介在 = 介在経由の第一ホップ)か**が見える。

# %%
# どの neuropil(段)で side 抑制が起きているか
opt = cl[cl["syn_region"].isin(["LA", "ME", "LO", "LOP", "AME"])]
piv = (opt.assign(kind=np.where(opt["label"].isin(lateral_labels), "same-stage lateral",
                                np.where(opt["label"].str.contains("feed"), "feedforward/feedback", "other")))
       .groupby(["syn_region", "kind"])["syn_count"].sum().unstack(fill_value=0))
piv = piv.reindex(["LA", "ME", "LO", "LOP", "AME"]).dropna(how="all")
print("Inhibitory synapses by synapse-location neuropil x kind:")
display(piv.astype(int))

# %% [markdown]
# ## R3. 直接 vs 介在経由の center-surround【欠陥2への回答】
#
# 各標的が受ける抑制を **単シナプス(direct)** と **二シナプス(mediated; exc→inh→T)** に分けて
# Δcolumn の関数として再構成する。文献どおり、**直接抑制は home 集中に見え、surround は介在(disynaptic)
# 再構成でのみ現れる**。

# %%
rk = RadialKernels.from_data(conn, col_assign)
targets = ["Mi1", "Tm9", "L2", "T4a"]
maxd = 8
rr = np.arange(0, maxd + 1)
fig, axes = plt.subplots(1, len(targets), figsize=(4.3 * len(targets), 4))
summary = []
for ax, T in zip(axes, targets):
    E = rk.direct_kernel(T, "exc")
    Idir = rk.direct_kernel(T, "inh")
    Idis = rk.disyn_kernel(T)
    sig_e, e = rms_radius(E, maxd)
    sig_dir, idr = rms_radius(Idir, maxd)
    sig_dis, ids = rms_radius(Idis, maxd)
    ax.plot(rr, e.values, "-o", color=EXC_COLOR, label=f"exc center (σ={sig_e:.2f})")
    ax.plot(rr, idr.values, "--s", color="tab:purple", label=f"direct inh (σ={sig_dir:.2f})")
    ax.plot(rr, ids.values, "-^", color=INH_COLOR, label=f"mediated inh (σ={sig_dis:.2f})")
    ax.set(xlabel="Δcolumn (hex)", ylabel="fraction", title=T)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    summary.append(dict(target=T, exc_center_rms=round(sig_e, 2), direct_inh_rms=round(sig_dir, 2),
                        mediated_inh_rms=round(sig_dis, 2),
                        direct_inh_at_home=round(float(idr.iloc[0]), 2),
                        mediated_inh_at_home=round(float(ids.iloc[0]), 2)))
plt.suptitle("Direct (monosynaptic) inhibition is home-concentrated; the broad surround is mediated (disynaptic)",
             y=1.03, fontsize=11)
plt.tight_layout()
display(pd.DataFrame(summary))

# %% [markdown]
# `direct inh`(紫)は home column に集中し、`mediated inh`(青; exc→inh→T)が広い surround を作る。
# これが「側抑制は介在経由」という文献的事実の連結体での裏付けであり、**direct と mediated を分けて初めて**
# center-surround が正しく見える。

# %% [markdown]
# ## R4 (opt-in). M1–M10 細層での抑制の役割分担
#
# `RUN_MEDULLA_LAYER=True` にすると `synapse_coordinates.csv`(~864MB)を読み、Mi1-PCA depth 定規で
# Dm(遠位)/ Pm(近位)の層分離を復元する(粗い neuropil ステージの opt-in refinement)。

# %%
RUN_MEDULLA_LAYER = False
if RUN_MEDULLA_LAYER:
    from src.lateral import assign_medulla_layer

    sc, ruler = assign_medulla_layer(m.optic_lobe_neurons_df, side="right")
    med = (sc[sc["rel_depth"].between(-0.25, 1.25)]
           .groupby("ptype")["rel_depth"].median().sort_values())
    print("median relative medulla depth (0=distal/M1 .. 1=proximal/M10):")
    display(med.to_frame("median_rel_depth"))
else:
    print("RUN_MEDULLA_LAYER=False (set True to reconstruct M1-M10 depth; loads ~864MB)")

# %% [markdown]
# ## まとめ
#
# | 欠陥 | 基盤での対応 |
# |---|---|
# | 1. 同一階層を未判定 | `stage`(neuropil)+ `same_stage` で side 抑制を段またぎ FF/FB から分離(R2) |
# | 2. direct/mediated 未分離 | `RadialKernels` の direct vs disyn、`SignedConnGraph` の符号付き経路(R3) |
#
# いずれも `nt_type`(ML 推定)に依存する点は限界(確率的符号ロバスト性は別 Issue)。機能の確定には
# activity/physiology が必要という連結体解析全般の限界も従来どおり残る。
