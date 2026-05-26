# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FlyWire connectome — quick EDA
#
# `FlyWireDataManager` で raw CSV を読み込み、ニューロン分類・接続・神経伝達物質・光受容体 (R1-6) 座標をひととおり眺める。
#
# **前提:**
# - `DROSOPHILA_DATA_DIR` が `data/raw/flywire/csv/` を含むディレクトリを指していること (`.env.local` 参照)
# - 依存: `uv sync --extra notebook` で `matplotlib` 等が入ること

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
from src.data import FlyWireDataManager, FLYWIRE_DESCRIPTOR

print(f"DATA_DIR = {DATA_DIR}")
print(f"descriptor = {FLYWIRE_DESCRIPTOR.name} ({FLYWIRE_DESCRIPTOR.display_name})")

# %%
# 全 CSV 読み込み (~15s)。初回のみ時間がかかる。
m = FlyWireDataManager()

print(f"neurons_df:               {m.neurons_df.shape}")
print(f"optic_lobe_neurons_df:    {m.optic_lobe_neurons_df.shape}")
print(f"optic_lobe_connections_df:{m.optic_lobe_connections_df.shape}")

# %% [markdown]
# ## 1. 全脳ニューロンの super_class 分布

# %%
super_class_counts = m.neurons_df["super_class"].value_counts(dropna=False)
display(super_class_counts.to_frame("count"))

fig, ax = plt.subplots(figsize=(8, 4))
super_class_counts.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("# neurons")
ax.set_title("FlyWire neurons by super_class")
plt.tight_layout()

# %% [markdown]
# ## 2. Optic Lobe 内の主要な cell type
#
# `group` (neuropil グループ) と `primary_type` (T4a / Tm9 等) で見る。

# %%
group_counts = m.optic_lobe_neurons_df["group"].value_counts().head(15)
display(group_counts.to_frame("count"))

top_types = m.optic_lobe_neurons_df["primary_type"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(9, 5))
top_types.sort_values().plot.barh(ax=ax, color="tab:orange")
ax.set_xlabel("# neurons")
ax.set_title("Top 20 primary_type in Optic Lobe")
plt.tight_layout()

# %% [markdown]
# ## 3. R1-6 光受容体
#
# `FLYWIRE_DESCRIPTOR.input_neuron_types == ("R1-6",)`。 retina 表面に投影された tip 座標を散布図に。

# %%
r16_df = m.optic_lobe_neurons_df[m.optic_lobe_neurons_df["primary_type"] == "R1-6"]
print(f"R1-6 count: {len(r16_df)}")
display(r16_df[["root_id", "side", "position_x", "position_y", "position_z", "nt_type"]].head())

# %%
# Retina 表面に投影した座標 (surface fitting cache 必須)
coords_df, meta = m.get_photoreceptor_coordinates_with_directions()
print(f"projected R1-6: {len(coords_df)} (source={meta['source']})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(coords_df["position_x"], coords_df["position_y"], s=2, alpha=0.6)
axes[0].set(xlabel="x", ylabel="y", aspect="equal", title="R1-6 (x, y)")
axes[1].scatter(coords_df["position_y"], coords_df["position_z"], s=2, alpha=0.6, color="tab:red")
axes[1].set(xlabel="y", ylabel="z", aspect="equal", title="R1-6 (y, z)")
plt.tight_layout()

# %% [markdown]
# ## 4. 接続データ
#
# Optic Lobe 内の接続 (R1-6→R1-6 直結は除外済み)。`syn_count` がエッジの重み。

# %%
conn = m.optic_lobe_connections_df
print(f"total edges:   {len(conn):,}")
print(f"unique pre:    {conn['pre_root_id'].nunique():,}")
print(f"unique post:   {conn['post_root_id'].nunique():,}")
print(f"total synapses:{conn['syn_count'].sum():,}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(conn["syn_count"].clip(lower=1)), bins=60, color="steelblue", edgecolor="white")
ax.set(xlabel="log10(syn_count)", ylabel="# edges", title="Synapse count distribution")
plt.tight_layout()

# %% [markdown]
# ## 5. R1-6 のシナプス出力先
#
# L1/L2/L3 などのラミナ単極細胞や AMC 系へ向かう。

# %%
r16_out = conn[conn["pre_primary_type"] == "R1-6"]
print(f"R1-6 outgoing edges: {len(r16_out):,}, synapses: {r16_out['syn_count'].sum():,}")

downstream = (
    r16_out.groupby("post_primary_type")["syn_count"].sum()
    .sort_values(ascending=False).head(15)
)
display(downstream.to_frame("total_syn"))

fig, ax = plt.subplots(figsize=(8, 4))
downstream.sort_values().plot.barh(ax=ax, color="tab:green")
ax.set(xlabel="# synapses", title="Top R1-6 downstream targets")
plt.tight_layout()

# %% [markdown]
# ## 6. 神経伝達物質 (nt_type) 分布
#
# FlyWire は機械推定の `nt_type` を持つ (ACH / GABA / GLUT / DA / SER / OCT)。
#
# **注:** R1-6 はドメイン知識から histaminergic として **接続 DataFrame の `pre_primary_type==R1-6` のエッジに限り `nt_type=HIS` / `alpha=-1`** に補正している (`optic_lobe_neurons_df` 側は補正なしの元 csv 値)。

# %%
nt_counts = m.optic_lobe_neurons_df["nt_type"].value_counts(dropna=False)
display(nt_counts.to_frame("count"))

# R1-6 の出力エッジが HIS にリラベルされていることを確認
r16_edge_nt = conn[conn["pre_primary_type"] == "R1-6"]["nt_type"].value_counts()
print(f"R1-6 outgoing edge nt_type: {dict(r16_edge_nt)}")

fig, ax = plt.subplots(figsize=(7, 4))
nt_counts.sort_values().plot.barh(ax=ax, color="tab:purple")
ax.set(xlabel="# neurons", title="Optic Lobe neurons by predicted nt_type")
plt.tight_layout()

# %% [markdown]
# ---
#
# ここまでで主要な切り口は把握できたはず。さらに深掘りしたい例:
#
# - `m.neurons_df[m.neurons_df["primary_type"].isin(["T4a", "T4b", "T4c", "T4d"])]` のような T4/T5 抽出
# - `m.optic_lobe_connections_df` を NetworkX に流して受容野解析
# - `FlyWireDataManager(extra_csv_stems=["neuropil_synapse_table"])` で neuropil 別 input/output 統計を追加ロード
