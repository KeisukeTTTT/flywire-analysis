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
# # 視野端の側抑制補填と T4/T5 lateral circuit
#
# ## 背景
#
# ハエの optic lobe（視葉）は、複眼の各 ommatidium に対応する **column（カラム）** が
# 六方格子状に並ぶ retinotopic な構造をもつ。各 column 内外のニューロンは近傍 column との間で
# 側方結合（lateral connection）を作り、中心–周辺拮抗のような空間フィルタを形成する。
# ところが視野の **端（rim）** にある column は、利用できる隣接 column が物理的に欠けるため、
# 中心部と同じ配線則をそのまま当てはめると入力が目減りするはずである。実際の脳がこの目減りを
# どう扱っているか（放置するのか、何らかの形で補うのか）は、視野端での運動・コントラスト検出の
# 挙動を理解するうえで重要な問いになる。
#
# ## 検証する二つの仮説
#
# FlyWire optic-lobe connectome を用いて、次の二つを構造の観点から検証する。
#
# 1. **視野端の補填 (edge compensation)**: rim column では利用可能な隣接 column が減る。
#    もし観測される入力が「単純な幾何学的切断（隣が無い分だけ減る）」で説明できる量を超えて
#    維持されているなら、その差を生む候補機構として
#    (a) 残存する edge 1 本あたりの増強、(b) column を持たない wide-field source への切り替え、
#    (c) kernel を視野内側へずらす再中心化（recentering）、などが考えられる。
# 2. **T4/T5 lateral circuit**: 方向選択性ニューロン T4/T5 には、既知の feedforward 入力
#    （Mi, Tm, C 系など）に加えて、同亜型どうしの homotypic coupling、LPi/CT1 などを介した
#    recurrent loop、視野端だけで変わる rim-specific な配線が存在する可能性がある。
#
# ## 解析の立場
#
# connectome（配線図）だけから機能的な補償や抑制を **証明** することはできない。シナプスの
# 機能的符号は受容体に依存し、本解析の興奮/抑制ラベルは伝達物質に基づく便宜的なものに過ぎない。
# そこで本 notebook は結論を出すことを目的とせず、次段階のモデル・生理実験で検証すべき
# **構造的候補を漏れなく洗い出して保存する** ことを目的とする。各指標は「候補を絞り込むための
# heuristic」であり、統計的有意性や機能的効果を主張するものではない。
#
# ## 繰り返し出てくる用語
#
# - **column / reference grid**: `Mi1` の column 割り当てを基準格子とする（§2）。
# - **boundary distance**: その column から視野端までの graph 距離。`rim (<=1)` /
#   `middle (==2)` / `center (>=3)` の 3 領域に分ける。
# - **opportunity（機会数）**: ある距離に「そもそも存在しうる」source column 数。観測値を
#   この機会数で割ることで、配線変更と幾何学的切断を分離する（§3, §6, §10）。
# - **putative inhibitory / excitatory**: `GABA/GLUT/HIS` を抑制性、`ACH` を興奮性とみなす
#   便宜ラベル（§1）。
#
# ## 出力方針
#
# 集計前の **edge-level 表**、**column opportunity 表**、**per-cell 表**、**per-type 表** を
# 段階的に `outputs/edge_compensation_t4t5_lateral/` 配下へ保存する。各表は番号付き
# （`00_`, `01_`, …）で生成順に対応し、後段の解析が前段のどの表から導かれたかを追跡できる。
# 大きな表は `csv.gz` で圧縮する。raw データの読み込み元は `DATA_DIR`（隣接する drosophila
# リポジトリ）のままとし、派生物だけを本リポジトリに書き出す。同じ notebook を再実行すると
# 同名ファイルを上書き更新し、最後に `manifest.json` へ解析条件と全成果物の一覧を残す（§13）。

# %%
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
if (REPO_ROOT / "src").is_dir() is False:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401  # import するだけで matplotlib に日本語フォントを適用する
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import DATA_DIR
from src.data import FlyWireDataManager

MIN_SYN = 5
RIM_MAX_DISTANCE = 1
CENTER_MIN_DISTANCE = 3
REFERENCE_COLUMN_TYPE = "Mi1"
INHIBITORY_NT = {"GABA", "GLUT", "HIS"}
EXCITATORY_NT = {"ACH"}
MOTION_PREFIXES = ("T4", "T5")
EXCLUDED_RECEIVER_TYPES = {"R7", "R8"}
HEXN = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

OUT_DIR = REPO_ROOT / "outputs" / "edge_compensation_t4t5_lateral"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
artifacts = []


def classify_nt(value):
    """符号の操作的ラベル。機能的な符号は受容体依存であり、これはあくまで便宜的な分類である。"""
    if value in INHIBITORY_NT:
        return "inh"
    if value in EXCITATORY_NT:
        return "exc"
    return "other"


def is_motion_type(value):
    return str(value).startswith(MOTION_PREFIXES)


def axial_to_cart(p, q):
    """axial hex 座標を 60 度基底のデカルト座標に変換する。"""
    return np.asarray(p) + 0.5 * np.asarray(q), np.asarray(q) * (np.sqrt(3) / 2)


def hexd(dp, dq):
    """整数の column 座標に対する厳密な axial hex 距離。"""
    return (np.abs(dp) + np.abs(dq) + np.abs(dp + dq)) / 2


def region_from_boundary_distance(distance):
    if pd.isna(distance):
        return "outside_reference_grid"
    if distance <= RIM_MAX_DISTANCE:
        return "rim"
    if distance >= CENTER_MIN_DISTANCE:
        return "center"
    return "middle"


def save_table(df, stem, *, compress=False):
    """表を保存し、notebook 出力を後から監査できるようメタデータも記録する。"""
    suffix = ".csv.gz" if compress else ".csv"
    path = OUT_DIR / f"{stem}{suffix}"
    df.to_csv(path, index=False, compression="gzip" if compress else None)
    artifacts.append(
        {
            "name": stem,
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "bytes": int(path.stat().st_size),
        }
    )
    print(f"saved {path.name}: {len(df):,} rows")
    return path


def save_figure(fig, stem):
    path = FIG_DIR / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    artifacts.append({"name": stem, "path": str(path), "bytes": int(path.stat().st_size)})
    print(f"saved {path.name}")


def safe_div(num, den):
    return num / den.replace(0, np.nan)


def entropy_from_fraction(fraction):
    fraction = fraction[fraction > 0]
    return float(-(fraction * np.log2(fraction)).sum())


def robust_abs_limit(values, percentile=97, minimum=1e-9):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return minimum
    return max(float(np.percentile(np.abs(values), percentile)), minimum)


def hex_metric(ax, df, metric, *, title, cmap="viridis", vmin=None, vmax=None, center=None):
    """空間ビニングを行わず、column ごとの指標をそのまま描画する。"""
    x, y = axial_to_cart(df["receiver_p"], df["receiver_q"])
    values = df[metric].astype(float)
    if center is not None:
        limit = robust_abs_limit(values - center)
        vmin, vmax = center - limit, center + limit
    scatter = ax.scatter(
        x,
        y,
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker="h",
        s=32,
        linewidths=0,
    )
    rim = df["region"] == "rim"
    ax.scatter(
        np.asarray(x)[rim],
        np.asarray(y)[rim],
        facecolors="none",
        edgecolors="black",
        marker="h",
        s=38,
        linewidths=0.25,
    )
    ax.set(aspect="equal", title=title)
    ax.axis("off")
    return scatter


print(f"output directory: {OUT_DIR}")

# %% [markdown]
# ## 1. データ読み込みと解析条件
#
# ### 狙い
# 以降の全解析が乗る土台となる edge テーブルを用意し、解析全体の前提（シナプス数しきい値、
# 興奮/抑制の符号付け）をここで一括して固定する。
#
# ### 方法
# `FlyWireDataManager(use_no_threshold_connections=True)` で、optic lobe 内部のニューロン間 edge を
# **しきい値なし** でまず読み込む。各 edge には伝達物質型 `nt_type` から求めた符号 `sign`
# （`inh` / `exc` / `other`）を付与する。主要解析では結合の信頼性を担保するため
# `syn_count >= 5`（1 つの cell–cell 結合あたり 5 シナプス以上）を適用した `conn` を使う。
#
# ### しきい値感度
# しきい値を `1, 3, 5, 10` と変えたときの edge 数・総シナプス数・抑制性比率を
# `00_threshold_sensitivity` に保存する。これにより「5 という閾値選択が結論を左右していないか」を
# 後から確認できる。閾値適用済みの生 edge は `01_optic_lobe_edges_min5_raw` として残す。
#
# ### 符号付けの注意
# `GABA / GLUT / HIS` を putative inhibitory、`ACH` を putative excitatory とする。これは
# 受容体情報を含まない operational label であり、**機能的な符号そのものではない**。たとえば
# 同じ GABA でも受容体次第で効果は変わりうる点に留意する。`other` はこのいずれにも入らない
# 伝達物質（不明を含む）。

# %%
m = FlyWireDataManager(use_no_threshold_connections=True)
neurons = m.optic_lobe_neurons_df.copy()
conn_all = m.optic_lobe_connections_df.copy()
conn_all["sign"] = conn_all["nt_type"].map(classify_nt)
conn = conn_all[conn_all["syn_count"] >= MIN_SYN].copy()

threshold_rows = []
for threshold in [1, 3, 5, 10]:
    d = conn_all[conn_all["syn_count"] >= threshold]
    g = d.groupby("sign")["syn_count"].sum()
    inh = int(g.get("inh", 0))
    exc = int(g.get("exc", 0))
    threshold_rows.append(
        {
            "min_syn": threshold,
            "n_edges": len(d),
            "n_synapses": int(d["syn_count"].sum()),
            "inh_synapses": inh,
            "exc_synapses": exc,
            "inh_fraction_of_ie": inh / (inh + exc),
        }
    )
threshold_sensitivity = pd.DataFrame(threshold_rows)
save_table(threshold_sensitivity, "00_threshold_sensitivity")
save_table(conn, "01_optic_lobe_edges_min5_raw", compress=True)
print(threshold_sensitivity.to_string(index=False))

# %% [markdown]
# ### 図: しきい値感度の可視化
#
# しきい値を上げると edge / シナプスがどれだけ減り、興奮/抑制比がどう動くかを確認する。
# 灰色の破線は主要解析で採用する `MIN_SYN`。

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
axes[0].plot(
    threshold_sensitivity["min_syn"], threshold_sensitivity["n_edges"], marker="o", label="edge 数"
)
axes[0].plot(
    threshold_sensitivity["min_syn"], threshold_sensitivity["n_synapses"], marker="s", label="シナプス総数"
)
axes[0].axvline(MIN_SYN, color="gray", linestyle="--", linewidth=1)
axes[0].set(
    xlabel="最小シナプス数しきい値",
    ylabel="件数（対数軸）",
    yscale="log",
    title="しきい値による edge / シナプス数の減少",
)
axes[0].legend(frameon=False)
axes[1].plot(
    threshold_sensitivity["min_syn"],
    threshold_sensitivity["inh_fraction_of_ie"],
    marker="o",
    color="tab:purple",
)
axes[1].axvline(MIN_SYN, color="gray", linestyle="--", linewidth=1)
axes[1].set(
    xlabel="最小シナプス数しきい値",
    ylabel="抑制性シナプスの割合  inh / (inh + exc)",
    title="しきい値による抑制性比率の変化",
)
fig.suptitle("シナプス数しきい値の感度")
save_figure(fig, "fig_threshold_sensitivity")
plt.show()

# %% [markdown]
# ## 2. Reference column grid と境界距離
#
# ### 狙い
# 「視野端からどれだけ内側か」を全 column に与える。これが以降の rim / center 比較の座標軸になる。
#
# ### なぜ Mi1 を基準にするか
# `Mi1` は 1 column につきほぼ 1 つ存在し、retinotopic な格子のマーカーとして信頼できる。
# そこで `column_assignment.csv` の `Mi1` が存在する `(p, q)` を「その column が実在する」印として
# 採用し、hemisphere ごとに reference grid を作る。`(p, q)` は六方格子の axial 座標である。
#
# ### 計算手順
# 1. 各 column について、六近傍 `HEXN` のうち格子内に存在する数 `n_neighbors` を数える。
#    6 未満なら格子の縁にあたる。
# 2. `n_neighbors < 6` の column を境界（距離 0）とし、幅優先探索 (BFS) で各 column の
#    **最寄り境界までの graph distance** `boundary_distance` を求める。
# 3. 距離に応じて領域を 3 つに分ける。
#    - `rim`: boundary distance `<= 1`（縁とそのすぐ内側）
#    - `middle`: boundary distance `== 2`
#    - `center`: boundary distance `>= 3`（十分内側で、隣接 column が原則そろう）
# 4. 後段の「再中心化」検証用に、各 column から格子の重心へ向かう **内向き単位ベクトル**
#    `(inward_unit_x, inward_unit_y)` も計算しておく。lateral 入力がこの向きに偏れば、
#    kernel が視野内側へずれている候補になる。
#
# ### 出力と注意
# column ごとの座標・近傍数・境界距離・領域・内向きベクトルを `02_reference_column_geometry`
# に保存する。`Mi1` の割り当て漏れ（本来あるはずの column が欠ける穴）は、偽の境界として
# 誤認されうる。geometry そのものを表に残すことで、こうしたアーティファクトを後から点検できる。
# 続いて全 cell に column 情報を結合し (`03_column_assigned_cells_raw`)、
# photoreceptor (`R7/R8`) を除いて reference grid 内にあるニューロンを receiver として抽出する。

# %%
col_assign = pd.read_csv(
    Path(DATA_DIR) / "raw" / "flywire" / "csv" / "column_assignment.csv",
    dtype={"root_id": str, "column_id": str},
)
col_assign["p"] = col_assign["p"].astype(int)
col_assign["q"] = col_assign["q"].astype(int)
col_assign = col_assign.drop_duplicates("root_id").copy()


def build_column_geometry(reference_assignment):
    rows = []
    for hemi, hemi_df in reference_assignment.groupby("hemisphere"):
        coords = set(zip(hemi_df["p"], hemi_df["q"]))
        n_neighbors = {
            coord: sum((coord[0] + dp, coord[1] + dq) in coords for dp, dq in HEXN)
            for coord in coords
        }
        boundary = [coord for coord, count in n_neighbors.items() if count < 6]
        distance = {coord: 0 for coord in boundary}
        queue = deque(boundary)
        while queue:
            p, q = queue.popleft()
            for dp, dq in HEXN:
                neighbor = (p + dp, q + dq)
                if neighbor in coords and neighbor not in distance:
                    distance[neighbor] = distance[(p, q)] + 1
                    queue.append(neighbor)

        all_p = np.asarray([coord[0] for coord in coords])
        all_q = np.asarray([coord[1] for coord in coords])
        all_x, all_y = axial_to_cart(all_p, all_q)
        center_x, center_y = float(all_x.mean()), float(all_y.mean())
        for p, q in sorted(coords):
            x, y = axial_to_cart(p, q)
            inward_x, inward_y = center_x - x, center_y - y
            norm = float(np.hypot(inward_x, inward_y))
            rows.append(
                {
                    "hemisphere": hemi,
                    "p": p,
                    "q": q,
                    "x": float(x),
                    "y": float(y),
                    "n_neighbors": n_neighbors[(p, q)],
                    "boundary_distance": distance[(p, q)],
                    "region": region_from_boundary_distance(distance[(p, q)]),
                    "inward_unit_x": inward_x / norm if norm else 0.0,
                    "inward_unit_y": inward_y / norm if norm else 0.0,
                }
            )
    return pd.DataFrame(rows)


reference_assignment = col_assign[col_assign["type"] == REFERENCE_COLUMN_TYPE]
column_geometry = build_column_geometry(reference_assignment)
save_table(column_geometry, "02_reference_column_geometry")

cell_columns = col_assign.merge(
    column_geometry,
    on=["hemisphere", "p", "q"],
    how="left",
    validate="many_to_one",
)
cell_columns["region"] = cell_columns["boundary_distance"].map(region_from_boundary_distance)
cell_columns["in_reference_grid"] = cell_columns["boundary_distance"].notna()
save_table(cell_columns, "03_column_assigned_cells_raw", compress=True)

receiver_cells = cell_columns[
    ~cell_columns["type"].isin(EXCLUDED_RECEIVER_TYPES) & cell_columns["in_reference_grid"]
].copy()
print(receiver_cells["region"].value_counts().to_string())

# %% [markdown]
# ## 3. Column opportunity table
#
# ### 狙い（この解析の中心アイデア）
# rim column では、距離 `d` に「そもそも存在しうる」source column の数自体が少ない。
# したがって観測シナプス数をそのまま rim と center で比べると、
# **幾何学的切断（隣が無い）** と **配線変更（つなぎ方が違う）** を区別できない。
# そこで「分母」になる *機会数 (opportunity)* を先に用意する。
#
# ### 方法
# 各 hemisphere 内で、reference grid の全 column を receiver 候補・source 候補として総当たり
# （cross join）し、相対変位 `(dp, dq)`・hex 距離 `distance`・実空間変位 `(dx, dy)`、および
# receiver の内向きベクトルへの射影 `inward_projection` を付与する。`is_home` は同一 column
# （距離 0）を指す。これを全ペアぶん `04_column_opportunities_raw` に残す。
#
# ### 出力
# receiver column × 距離ごとに「利用可能な source column 数」を集計した
# `05_column_opportunity_counts` を作る。これが §6 で観測値を割って規格化するときの分母となり、
# 「rim で入力が減るのは単に近傍が無いからか、それとも配線が変わっているからか」を切り分ける。

# %%
opportunity_parts = []
for hemi, geom in column_geometry.groupby("hemisphere"):
    recv = geom[
        ["hemisphere", "p", "q", "boundary_distance", "region", "inward_unit_x", "inward_unit_y"]
    ].rename(
        columns={
            "p": "receiver_p",
            "q": "receiver_q",
            "boundary_distance": "receiver_boundary_distance",
            "region": "receiver_region",
        }
    )
    src = geom[["p", "q"]].rename(columns={"p": "source_p", "q": "source_q"})
    pair = recv.merge(src, how="cross")
    pair["hemisphere"] = hemi
    pair["dp"] = pair["source_p"] - pair["receiver_p"]
    pair["dq"] = pair["source_q"] - pair["receiver_q"]
    pair["distance"] = hexd(pair["dp"], pair["dq"]).astype(int)
    pair["dx"], pair["dy"] = axial_to_cart(pair["dp"], pair["dq"])
    pair["inward_projection"] = (
        pair["dx"] * pair["inward_unit_x"] + pair["dy"] * pair["inward_unit_y"]
    )
    pair["is_home"] = pair["distance"] == 0
    opportunity_parts.append(pair)

column_opportunities = pd.concat(opportunity_parts, ignore_index=True)
save_table(column_opportunities, "04_column_opportunities_raw", compress=True)

opportunity_counts = (
    column_opportunities.groupby(
        [
            "hemisphere",
            "receiver_p",
            "receiver_q",
            "receiver_boundary_distance",
            "receiver_region",
            "distance",
        ],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "n_available_source_columns"})
)
save_table(opportunity_counts, "05_column_opportunity_counts")

# %% [markdown]
# ### 図: 距離ごとの利用可能 source column 数
#
# rim では各距離で利用できる source column が center より少ないことを確認する。これが §6 以降で
# 観測値を機会数で割って補正する根拠になる。

# %%
fig, ax = plt.subplots(figsize=(8, 5))
region_styles = {"rim": ("tab:red", "-"), "middle": ("tab:orange", "--"), "center": ("tab:blue", "-")}
opp_by_region = (
    opportunity_counts[opportunity_counts["distance"] <= 8]
    .groupby(["receiver_region", "distance"], as_index=False)["n_available_source_columns"]
    .mean()
)
for region in ["center", "middle", "rim"]:
    group = opp_by_region[opp_by_region["receiver_region"] == region].sort_values("distance")
    if group.empty:
        continue
    color, linestyle = region_styles[region]
    ax.plot(
        group["distance"],
        group["n_available_source_columns"],
        marker="o",
        color=color,
        linestyle=linestyle,
        label=region,
    )
ax.set(
    xlabel="距離（hex column）",
    ylabel="利用可能な source column 数（平均）",
    title="距離ごとの利用可能 source column 数（領域別）",
)
ax.legend(frameon=False, title="receiver の領域")
fig.tight_layout()
save_figure(fig, "fig_column_opportunity_by_distance")
plt.show()

# %% [markdown]
# ## 4. Columnar receiver への raw input edges
#
# ### 狙い
# 以降の per-cell / per-type 集計の素になる、**集計前の主表** を組み立てる（1 行 = 1 本の入力 edge）。
#
# ### 各 edge に付ける情報
# receiver 側には型・hemisphere・column 座標・境界距離・領域・内向きベクトルを、source 側には
# その pre ニューロンの column 情報を結合する。そのうえで edge ごとに次を計算する。
#
# - `source_has_column`: source が column を持つか（持たない＝ wide-field 系の候補）。
# - `source_same_hemisphere`: source と receiver が同じ hemisphere の column か。
# - `dp, dq, distance`: 同一 hemisphere の column 入力に限った相対変位と hex 距離。
# - `inward_projection`: その変位の内向き成分（正なら視野内側から入る入力）。
# - `is_home (distance==0)` / `is_lateral (distance>0)`: 自 column 入力か側方入力か。
# - `source_scope`: `column_assigned` か `widefield_or_unassigned` か。
#
# ### 注意
# wide-field source（Dm, Pm, CT1 など column を一意に割り当てられないもの）は `dp/dq/distance`
# が `NaN` になるが、**edge 自体は捨てない**。これらは rim での「source 切り替え」候補の主役に
# なりうるため、`widefield_or_unassigned` として明示的に保持する。結果は
# `06_columnar_receiver_edges_raw` に保存する。

# %%
receiver_meta = receiver_cells[
    [
        "root_id",
        "type",
        "hemisphere",
        "p",
        "q",
        "n_neighbors",
        "boundary_distance",
        "region",
        "inward_unit_x",
        "inward_unit_y",
    ]
].rename(
    columns={
        "root_id": "post_root_id",
        "type": "receiver_type",
        "hemisphere": "receiver_hemisphere",
        "p": "receiver_p",
        "q": "receiver_q",
    }
)
source_meta = cell_columns[["root_id", "type", "hemisphere", "p", "q"]].rename(
    columns={
        "root_id": "pre_root_id",
        "type": "source_column_type",
        "hemisphere": "source_hemisphere",
        "p": "source_p",
        "q": "source_q",
    }
)

receiver_edges = conn[conn["post_root_id"].isin(set(receiver_meta["post_root_id"]))].copy()
receiver_edges = receiver_edges.merge(receiver_meta, on="post_root_id", how="inner", validate="many_to_one")
receiver_edges = receiver_edges.merge(source_meta, on="pre_root_id", how="left", validate="many_to_one")
receiver_edges["source_has_column"] = receiver_edges["source_p"].notna()
receiver_edges["source_same_hemisphere"] = (
    receiver_edges["source_has_column"]
    & (receiver_edges["source_hemisphere"] == receiver_edges["receiver_hemisphere"])
)
receiver_edges["dp"] = np.where(
    receiver_edges["source_same_hemisphere"],
    receiver_edges["source_p"] - receiver_edges["receiver_p"],
    np.nan,
)
receiver_edges["dq"] = np.where(
    receiver_edges["source_same_hemisphere"],
    receiver_edges["source_q"] - receiver_edges["receiver_q"],
    np.nan,
)
receiver_edges["distance"] = hexd(receiver_edges["dp"], receiver_edges["dq"])
receiver_edges["dx"], receiver_edges["dy"] = axial_to_cart(receiver_edges["dp"], receiver_edges["dq"])
receiver_edges["inward_projection"] = (
    receiver_edges["dx"] * receiver_edges["inward_unit_x"]
    + receiver_edges["dy"] * receiver_edges["inward_unit_y"]
)
receiver_edges["is_home"] = receiver_edges["distance"] == 0
receiver_edges["is_lateral"] = receiver_edges["distance"] > 0
receiver_edges["source_scope"] = np.where(receiver_edges["source_has_column"], "column_assigned", "widefield_or_unassigned")
save_table(receiver_edges, "06_columnar_receiver_edges_raw", compress=True)

# %% [markdown]
# ## 5. Per-cell input metrics: 補填候補を分解する
#
# ### 狙い
# 冒頭で挙げた 3 つの補填機構（edge 増強 / source 切り替え / 再中心化）を、ニューロン 1 個ずつの
# 指標に **分解** する（1 行 = 1 receiver neuron）。後で boundary distance に対する依存を見れば、
# どの機構が効いていそうかを切り分けられる。
#
# ### 計算する指標と、対応する補填機構
# - **入力の総量**: total synapse / edge / partner 数、興奮・抑制・other 別の内訳。
# - **edge 増強**: `inh_syn_per_edge`（抑制性 edge 1 本あたりシナプス数）。rim で上がれば、
#   本数が減った分を 1 本ごとの増強で補っている候補。
# - **source 切り替え**:
#     - `inh_widefield_or_unassigned_fraction`（抑制入力に占める wide-field/未割当の比率）。
#     - `inh_source_type_count` と `inh_source_entropy`（抑制 source 型の数と多様性）。
#   rim でこれらが上がれば、column 性 source から wide-field/別種 source へ切り替えている候補。
# - **再中心化**: lateral（distance>0）な抑制入力をシナプス数で重み付けした重心
#   `inh_offset_(x,y)`・その大きさ `inh_offset_magnitude`・内向き射影の平均
#   `inh_mean_inward_projection`。rim で内向き射影が上がれば、kernel を視野内側へずらしている候補。
#
# ### 中間生成物
# 符号 × source_scope 別の集計 (`07_…`)、抑制 source 型ごとの寄与 (`08_…`)、lateral 入力の
# offset (`09_…`) を個別に保存し、最後にすべてを cell 単位へ統合して
# `10_per_cell_input_metrics_raw` を出力する。entropy はシナプス比率から `-Σ p log2 p` で計算し、
# source が 1 種に偏れば 0、多種に分散すれば大きくなる。
#
# ### 注意
# ここでの各指標は「候補を分解して可視化する」ためのもので、有意差検定は §7 の Spearman 相関や
# per-type 集計に委ねる。

# %%
cell_key = [
    "post_root_id",
    "receiver_type",
    "receiver_hemisphere",
    "receiver_p",
    "receiver_q",
    "n_neighbors",
    "boundary_distance",
    "region",
]
per_cell = (
    receiver_edges.groupby(cell_key, as_index=False)
    .agg(
        total_syn=("syn_count", "sum"),
        total_edges=("syn_count", "size"),
        total_partners=("pre_root_id", "nunique"),
    )
)

by_sign = (
    receiver_edges.groupby(cell_key + ["sign"], as_index=False)
    .agg(
        syn=("syn_count", "sum"),
        edges=("syn_count", "size"),
        partners=("pre_root_id", "nunique"),
    )
)
by_sign_wide = by_sign.pivot(index=cell_key, columns="sign", values=["syn", "edges", "partners"]).fillna(0)
by_sign_wide.columns = [f"{metric}_{sign}" for metric, sign in by_sign_wide.columns]
by_sign_wide = by_sign_wide.reset_index()
per_cell = per_cell.merge(by_sign_wide, on=cell_key, how="left", validate="one_to_one")

for sign in ["inh", "exc", "other"]:
    for metric in ["syn", "edges", "partners"]:
        column = f"{metric}_{sign}"
        if column not in per_cell:
            per_cell[column] = 0
per_cell["inh_syn_per_edge"] = safe_div(per_cell["syn_inh"], per_cell["edges_inh"])
per_cell["exc_syn_per_edge"] = safe_div(per_cell["syn_exc"], per_cell["edges_exc"])
per_cell["inh_fraction_of_ie"] = safe_div(per_cell["syn_inh"], per_cell["syn_inh"] + per_cell["syn_exc"])

column_scope = (
    receiver_edges.groupby(cell_key + ["sign", "source_scope"], as_index=False)["syn_count"]
    .sum()
    .pivot(index=cell_key + ["sign"], columns="source_scope", values="syn_count")
    .fillna(0)
    .reset_index()
)
for column in ["column_assigned", "widefield_or_unassigned"]:
    if column not in column_scope:
        column_scope[column] = 0
column_scope["widefield_or_unassigned_fraction"] = safe_div(
    column_scope["widefield_or_unassigned"],
    column_scope["column_assigned"] + column_scope["widefield_or_unassigned"],
)
save_table(column_scope, "07_per_cell_source_scope_by_sign")

inh_scope = column_scope[column_scope["sign"] == "inh"][
    cell_key + ["column_assigned", "widefield_or_unassigned", "widefield_or_unassigned_fraction"]
].rename(
    columns={
        "column_assigned": "inh_column_assigned_syn",
        "widefield_or_unassigned": "inh_widefield_or_unassigned_syn",
        "widefield_or_unassigned_fraction": "inh_widefield_or_unassigned_fraction",
    }
)
per_cell = per_cell.merge(inh_scope, on=cell_key, how="left", validate="one_to_one")

inh_source_by_cell = (
    receiver_edges[receiver_edges["sign"] == "inh"]
    .groupby(cell_key + ["pre_primary_type"], as_index=False)["syn_count"]
    .sum()
    .rename(columns={"syn_count": "inh_syn_from_source_type", "pre_primary_type": "source_type"})
)
inh_source_by_cell["inh_source_fraction"] = inh_source_by_cell.groupby("post_root_id")[
    "inh_syn_from_source_type"
].transform(lambda values: values / values.sum())
save_table(inh_source_by_cell, "08_inh_source_type_by_cell_raw", compress=True)

inh_diversity = (
    inh_source_by_cell.groupby("post_root_id", as_index=False)
    .agg(
        inh_source_type_count=("source_type", "nunique"),
        inh_source_entropy=("inh_source_fraction", entropy_from_fraction),
    )
)
per_cell = per_cell.merge(inh_diversity, on="post_root_id", how="left", validate="one_to_one")

lateral_edges = receiver_edges[
    receiver_edges["source_same_hemisphere"] & receiver_edges["is_lateral"]
].copy()
lateral_edges["weighted_dx"] = lateral_edges["dx"] * lateral_edges["syn_count"]
lateral_edges["weighted_dy"] = lateral_edges["dy"] * lateral_edges["syn_count"]
lateral_edges["weighted_inward_projection"] = lateral_edges["inward_projection"] * lateral_edges["syn_count"]
lateral_offset = (
    lateral_edges.groupby(cell_key + ["sign"], as_index=False)
    .agg(
        lateral_syn=("syn_count", "sum"),
        weighted_dx=("weighted_dx", "sum"),
        weighted_dy=("weighted_dy", "sum"),
        weighted_inward_projection=("weighted_inward_projection", "sum"),
    )
)
lateral_offset["offset_x"] = lateral_offset["weighted_dx"] / lateral_offset["lateral_syn"]
lateral_offset["offset_y"] = lateral_offset["weighted_dy"] / lateral_offset["lateral_syn"]
lateral_offset["offset_magnitude"] = np.hypot(lateral_offset["offset_x"], lateral_offset["offset_y"])
lateral_offset["mean_inward_projection"] = (
    lateral_offset["weighted_inward_projection"] / lateral_offset["lateral_syn"]
)
save_table(lateral_offset, "09_per_cell_lateral_input_offset_by_sign")

inh_offset = lateral_offset[lateral_offset["sign"] == "inh"][
    cell_key + ["lateral_syn", "offset_x", "offset_y", "offset_magnitude", "mean_inward_projection"]
].rename(
    columns={
        "lateral_syn": "inh_lateral_columnar_syn",
        "offset_x": "inh_offset_x",
        "offset_y": "inh_offset_y",
        "offset_magnitude": "inh_offset_magnitude",
        "mean_inward_projection": "inh_mean_inward_projection",
    }
)
per_cell = per_cell.merge(inh_offset, on=cell_key, how="left", validate="one_to_one")
save_table(per_cell, "10_per_cell_input_metrics_raw", compress=True)

# %% [markdown]
# ### 図: source の種別構成（column 性 vs wide-field）
#
# 表 07 を領域 × 符号で集約し、入力シナプスが column 性 source と wide-field/未割当 source の
# どちらから来るかを見る。rim で wide-field 比率が上がれば source 切り替えの候補（興奮性側も確認）。

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
region_order = ["center", "middle", "rim"]
scope_region = column_scope.groupby(["region", "sign"], as_index=False)[
    ["column_assigned", "widefield_or_unassigned"]
].sum()
for ax, sign in zip(axes, ["inh", "exc"]):
    sub = scope_region[scope_region["sign"] == sign].set_index("region").reindex(region_order)
    total = (sub["column_assigned"] + sub["widefield_or_unassigned"]).replace(0, np.nan)
    col_frac = (sub["column_assigned"] / total).fillna(0).to_numpy()
    wf_frac = (sub["widefield_or_unassigned"] / total).fillna(0).to_numpy()
    x = np.arange(len(region_order))
    ax.bar(x, col_frac, color="tab:blue", label="column 性 source")
    ax.bar(x, wf_frac, bottom=col_frac, color="tab:orange", label="wide-field / 未割当 source")
    ax.set(xticks=x, ylim=(0, 1), ylabel="入力シナプスの構成比", title=f"{sign}")
    ax.set_xticklabels(region_order)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("source の種別構成（column 性 vs wide-field）: 領域別・符号別")
save_figure(fig, "fig_source_scope_by_region")
plt.show()

# %% [markdown]
# ### 図: lateral 入力の内向き射影（領域別・符号別）
#
# 表 09 を領域 × 符号で集約し、lateral（側方）入力が視野内側から来る度合いをシナプス加重平均で見る。
# rim で内向き射影が大きくなれば、kernel を視野内側へずらす再中心化の候補。

# %%
fig, ax = plt.subplots(figsize=(8, 5))
region_order = ["center", "middle", "rim"]
sign_colors = {"inh": "tab:blue", "exc": "tab:red"}
width = 0.38
x = np.arange(len(region_order))
for i, sign in enumerate(["inh", "exc"]):
    sub = lateral_offset[lateral_offset["sign"] == sign]
    values = []
    for region in region_order:
        group = sub[sub["region"] == region]
        weight = group["lateral_syn"].sum()
        values.append(
            float((group["mean_inward_projection"] * group["lateral_syn"]).sum() / weight)
            if weight
            else 0.0
        )
    ax.bar(x + (i - 0.5) * width, values, width=width, color=sign_colors[sign], label=sign)
ax.axhline(0, color="black", linewidth=0.8)
ax.set(
    xticks=x,
    ylabel="lateral 入力の内向き射影（シナプス加重平均）",
    title="lateral 入力の内向き射影: 領域別・符号別",
)
ax.set_xticklabels(region_order)
ax.legend(frameon=False, title="符号")
fig.tight_layout()
save_figure(fig, "fig_lateral_inward_projection_by_region")
plt.show()

# %% [markdown]
# ## 6. Opportunity-corrected radial kernel と単純切断モデル
#
# ### 狙い
# 「もし配線則が center と同じで、視野端では単に隣が欠けるだけ（＝幾何学的切断）」という
# **帰無モデル (null model)** を作り、観測値と比べる。観測がこの期待を上回れば、切断だけでは
# 説明できない入力維持＝補填の候補になる。
#
# ### 手順
# 1. **kernel 推定**: center（boundary distance >= 3、近傍がそろう領域）の cell だけを使い、
#    type × hemisphere × sign × 距離ごとに `synapses / available source columns`
#    （機会 1 つあたりの平均シナプス数）を求める。これを「視野端の影響を受けていない素の動径
#    kernel」とみなす（`13_center_radial_kernel_opportunity_corrected`）。
# 2. **期待値の構成**: その kernel を、各 cell が実際に持つ距離別の機会数（§3 の opportunity）に
#    掛けて距離方向に足し合わせ、その cell の `geometry_expected_columnar_syn`
#    （切断モデルが予測する column 性入力量）を作る。
# 3. **残差**: 観測した column 性シナプス量と比べ、
#    `observed_over_geometry = observed / geometry_expected` を cell ごとに出す
#    （`14_per_cell_geometry_model_residual_raw`、要約は `15_geometry_model_residual_summary`）。
#
# ### 読み方
# - `observed_over_geometry ≈ 1`: 入力減は近傍欠如だけで説明でき、配線変更の証拠は弱い。
# - `> 1`: 単純切断より入力が維持されている＝補填の候補。
# - `< 1`: 切断以上に入力が落ちている。
#
# ### 注意
# これは **等方的な切断モデルに対する構造残差** であって、機能的補償の証明ではない。また
# T5 のように column 性の抑制 source がほとんど無い型では、この比は分母が小さく不安定になるため、
# §7 では all-source 入力や source 構成で補って解釈する。

# %%
cell_opportunities = receiver_cells[
    ["root_id", "type", "hemisphere", "p", "q", "boundary_distance", "region"]
].rename(
    columns={
        "root_id": "post_root_id",
        "type": "receiver_type",
        "hemisphere": "receiver_hemisphere",
        "p": "receiver_p",
        "q": "receiver_q",
    }
).merge(
    opportunity_counts[
        ["hemisphere", "receiver_p", "receiver_q", "distance", "n_available_source_columns"]
    ],
    left_on=["receiver_hemisphere", "receiver_p", "receiver_q"],
    right_on=["hemisphere", "receiver_p", "receiver_q"],
    how="left",
    validate="many_to_many",
).drop(columns="hemisphere")
save_table(cell_opportunities, "11_per_cell_opportunities_raw", compress=True)

observed_radial = (
    receiver_edges[receiver_edges["source_same_hemisphere"]]
    .groupby(cell_key + ["sign", "distance"], as_index=False)
    .agg(
        observed_syn=("syn_count", "sum"),
        observed_edges=("syn_count", "size"),
        observed_partners=("pre_root_id", "nunique"),
    )
)
observed_radial["distance"] = observed_radial["distance"].astype(int)
observed_radial = observed_radial.merge(
    cell_opportunities[["post_root_id", "distance", "n_available_source_columns"]],
    on=["post_root_id", "distance"],
    how="left",
    validate="many_to_one",
)
observed_radial["syn_per_available_column"] = safe_div(
    observed_radial["observed_syn"], observed_radial["n_available_source_columns"]
)
save_table(observed_radial, "12_per_cell_radial_input_raw", compress=True)

center_opp = (
    cell_opportunities[cell_opportunities["region"] == "center"]
    .groupby(["receiver_type", "receiver_hemisphere", "distance"], as_index=False)
    .agg(
        center_opportunities=("n_available_source_columns", "sum"),
        n_center_cells=("post_root_id", "nunique"),
    )
)
center_syn = (
    observed_radial[observed_radial["region"] == "center"]
    .groupby(["receiver_type", "receiver_hemisphere", "sign", "distance"], as_index=False)[
        "observed_syn"
    ]
    .sum()
    .rename(columns={"observed_syn": "center_observed_syn"})
)
kernel_signs = pd.DataFrame({"sign": ["inh", "exc", "other"]})
center_kernel = center_opp.merge(kernel_signs, how="cross").merge(
    center_syn,
    on=["receiver_type", "receiver_hemisphere", "sign", "distance"],
    how="left",
)
center_kernel["center_observed_syn"] = center_kernel["center_observed_syn"].fillna(0)
center_kernel["syn_per_available_column"] = safe_div(
    center_kernel["center_observed_syn"], center_kernel["center_opportunities"]
)
save_table(center_kernel, "13_center_radial_kernel_opportunity_corrected")

expected_by_cell = cell_opportunities.merge(
    center_kernel[
        ["receiver_type", "receiver_hemisphere", "distance", "sign", "syn_per_available_column"]
    ],
    on=["receiver_type", "receiver_hemisphere", "distance"],
    how="left",
    validate="many_to_many",
)
expected_by_cell["expected_syn_contribution"] = (
    expected_by_cell["n_available_source_columns"] * expected_by_cell["syn_per_available_column"]
)
expected_by_cell = (
    expected_by_cell.groupby(
        [
            "post_root_id",
            "receiver_type",
            "receiver_hemisphere",
            "receiver_p",
            "receiver_q",
            "boundary_distance",
            "region",
            "sign",
        ],
        as_index=False,
    )["expected_syn_contribution"]
    .sum()
    .rename(columns={"expected_syn_contribution": "geometry_expected_columnar_syn"})
)
observed_columnar_by_cell = (
    receiver_edges[receiver_edges["source_same_hemisphere"]]
    .groupby(["post_root_id", "sign"], as_index=False)["syn_count"]
    .sum()
    .rename(columns={"syn_count": "observed_columnar_syn"})
)
compensation_by_cell = expected_by_cell.merge(
    observed_columnar_by_cell,
    on=["post_root_id", "sign"],
    how="left",
    validate="one_to_one",
)
compensation_by_cell["observed_columnar_syn"] = compensation_by_cell["observed_columnar_syn"].fillna(0)
compensation_by_cell["observed_over_geometry"] = safe_div(
    compensation_by_cell["observed_columnar_syn"],
    compensation_by_cell["geometry_expected_columnar_syn"],
)
save_table(compensation_by_cell, "14_per_cell_geometry_model_residual_raw", compress=True)

compensation_summary = (
    compensation_by_cell.groupby(["receiver_type", "receiver_hemisphere", "sign", "region"], as_index=False)
    .agg(
        n_cells=("post_root_id", "nunique"),
        median_observed_columnar_syn=("observed_columnar_syn", "median"),
        median_geometry_expected_syn=("geometry_expected_columnar_syn", "median"),
        median_observed_over_geometry=("observed_over_geometry", "median"),
    )
)
save_table(compensation_summary, "15_geometry_model_residual_summary")

# %% [markdown]
# ### 図: Center で推定した動径 kernel
#
# 切断モデルの基準となる「機会あたりシナプス数」を距離方向のプロファイルとして可視化する。
# ここでは notebook の主役である T4/T5 を表示する（表 13 は全 receiver type を含む）。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
kernel_motion = center_kernel[
    center_kernel["receiver_type"].map(is_motion_type) & (center_kernel["distance"] <= 6)
]
for ax, sign in zip(axes, ["inh", "exc"]):
    profile = (
        kernel_motion[kernel_motion["sign"] == sign]
        .groupby(["receiver_type", "distance"], as_index=False)["syn_per_available_column"]
        .mean()
    )
    for receiver_type, group in profile.groupby("receiver_type"):
        group = group.sort_values("distance")
        ax.plot(group["distance"], group["syn_per_available_column"], marker="o", label=receiver_type, alpha=0.8)
    ax.set(
        xlabel="距離（hex column）",
        ylabel="機会あたりシナプス数",
        title=f"{sign}: center 動径 kernel",
    )
    ax.legend(ncol=2, fontsize=7, frameon=False)
fig.suptitle("Center で推定した opportunity 補正済み動径 kernel（T4/T5, 左右半球平均）")
save_figure(fig, "fig_center_radial_kernel")
plt.show()

# %% [markdown]
# ## 7. Source switching と per-type rim-center summary
#
# ### 狙い
# §5 の per-cell 指標を **type × hemisphere 単位** に集約し、rim と center の差として
# 補填候補を一望できるようにする。あわせて「どの source 型が rim で増減するか」を順位付けする。
#
# ### 方法
# - **source switching**: 抑制 source 型ごとに領域内構成比を出し（`16_inh_source_type_by_region`）、
#   rim と center の構成比差 `rim_minus_center_fraction` で並べる
#   （`17_inh_source_switch_rim_minus_center`）。raw な source 型別表を残してあるので、
#   CT1・LPi・Dm・Pm など特定の source を後から個別に追える。
# - **per-type 要約**: total/抑制シナプス、抑制比率、edge あたりシナプス、source 多様性、
#   wide-field 比率、内向き offset などの中央値を領域別に出し（`18_per_type_region_input_metrics`）、
#   rim と center を横並びにして比 `*_rim_over_center` を付けた
#   `19_per_type_rim_center_comparison` を作る。
# - **相関**: 各 type で boundary distance と主要指標の Spearman 相関を取り
#   （`20_boundary_distance_correlations`）、rim/center の二値比較だけでなく
#   「距離に対する単調傾向」があるかを確認する。中央値を使うのは外れ値に頑健にするため。

# %%
source_by_region = (
    inh_source_by_cell.groupby(
        ["receiver_type", "receiver_hemisphere", "region", "source_type"],
        as_index=False,
    )
    .agg(
        inh_syn_from_source_type=("inh_syn_from_source_type", "sum"),
        n_receiver_cells=("post_root_id", "nunique"),
    )
)
source_by_region["source_fraction_within_region"] = source_by_region.groupby(
    ["receiver_type", "receiver_hemisphere", "region"]
)["inh_syn_from_source_type"].transform(lambda values: values / values.sum())
save_table(source_by_region, "16_inh_source_type_by_region")

source_switch = source_by_region.pivot_table(
    index=["receiver_type", "receiver_hemisphere", "source_type"],
    columns="region",
    values="source_fraction_within_region",
    fill_value=0,
).reset_index()
for region in ["rim", "center"]:
    if region not in source_switch:
        source_switch[region] = 0.0
source_switch["rim_minus_center_fraction"] = source_switch["rim"] - source_switch["center"]
source_switch = source_switch.sort_values("rim_minus_center_fraction", ascending=False)
save_table(source_switch, "17_inh_source_switch_rim_minus_center")

metric_columns = [
    "total_syn",
    "syn_inh",
    "syn_exc",
    "inh_fraction_of_ie",
    "edges_inh",
    "partners_inh",
    "inh_syn_per_edge",
    "inh_source_type_count",
    "inh_source_entropy",
    "inh_widefield_or_unassigned_fraction",
    "inh_offset_magnitude",
    "inh_mean_inward_projection",
]
per_type_region = (
    per_cell.groupby(["receiver_type", "receiver_hemisphere", "region"], as_index=False)[metric_columns]
    .median()
)
save_table(per_type_region, "18_per_type_region_input_metrics")

rim_center = per_type_region[per_type_region["region"].isin(["rim", "center"])].pivot(
    index=["receiver_type", "receiver_hemisphere"],
    columns="region",
    values=metric_columns,
)
rim_center.columns = [f"{metric}_{region}" for metric, region in rim_center.columns]
rim_center = rim_center.reset_index()
for metric in metric_columns:
    rim = f"{metric}_rim"
    center = f"{metric}_center"
    if rim in rim_center and center in rim_center:
        rim_center[f"{metric}_rim_over_center"] = safe_div(rim_center[rim], rim_center[center])
save_table(rim_center, "19_per_type_rim_center_comparison")

correlation_rows = []
for (receiver_type, hemi), group in per_cell.groupby(["receiver_type", "receiver_hemisphere"]):
    for metric in ["syn_inh", "inh_syn_per_edge", "inh_widefield_or_unassigned_fraction", "inh_mean_inward_projection"]:
        valid = group[["boundary_distance", metric]].dropna()
        rho, pvalue = spearmanr(valid["boundary_distance"], valid[metric]) if len(valid) >= 3 else (np.nan, np.nan)
        correlation_rows.append(
            {
                "receiver_type": receiver_type,
                "receiver_hemisphere": hemi,
                "metric": metric,
                "n_cells": len(valid),
                "spearman_rho_vs_boundary_distance": rho,
                "spearman_pvalue": pvalue,
            }
        )
boundary_correlations = pd.DataFrame(correlation_rows)
save_table(boundary_correlations, "20_boundary_distance_correlations")

# %% [markdown]
# ### 図: 抑制性 source switching（全 receiver type）
#
# fig06 は T4/T5 限定なので、ここでは全 receiver type について、rim−center の構成比差が大きい
# 抑制性 source を俯瞰する（行は |差| 上位の receiver type、左右半球平均）。

# %%
top_switch_sources = (
    source_switch.assign(abs_delta=source_switch["rim_minus_center_fraction"].abs())
    .groupby("source_type")["abs_delta"]
    .sum()
    .nlargest(20)
    .index
)
switch_heat = (
    source_switch[source_switch["source_type"].isin(top_switch_sources)]
    .groupby(["receiver_type", "source_type"], as_index=False)["rim_minus_center_fraction"]
    .mean()
    .pivot(index="receiver_type", columns="source_type", values="rim_minus_center_fraction")
    .fillna(0)
)
top_recv = switch_heat.abs().sum(axis=1).nlargest(40).index
switch_heat = switch_heat.loc[top_recv].sort_index()
fig, ax = plt.subplots(figsize=(13, max(5, 0.24 * len(switch_heat))), constrained_layout=True)
limit = robust_abs_limit(switch_heat.to_numpy())
image = ax.imshow(switch_heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set(
    xticks=np.arange(len(switch_heat.columns)),
    yticks=np.arange(len(switch_heat.index)),
    title="抑制性 source の構成比差 rim − center（全 receiver type, 左右半球平均）",
)
ax.set_xticklabels(switch_heat.columns, rotation=90, fontsize=7)
ax.set_yticklabels(switch_heat.index, fontsize=6)
fig.colorbar(image, ax=ax, shrink=0.6, label="rim − center 構成比")
save_figure(fig, "fig_inh_source_switch_all_types")
plt.show()

# %% [markdown]
# ### 図: boundary distance との相関
#
# 主要 4 指標が境界距離に対して単調に変化するかを Spearman 相関で見る。正なら内側 (center) ほど
# 大きく、負なら rim ほど大きい。|相関| が大きい上位 type を表示（表 20 は全 type を含む）。

# %%
metric_labels = {
    "syn_inh": "抑制性シナプス数",
    "inh_syn_per_edge": "抑制性シナプス / edge",
    "inh_widefield_or_unassigned_fraction": "wide-field 比率",
    "inh_mean_inward_projection": "内向き射影",
}
corr_pivot = (
    boundary_correlations.groupby(["receiver_type", "metric"], as_index=False)[
        "spearman_rho_vs_boundary_distance"
    ]
    .mean()
    .pivot(index="receiver_type", columns="metric", values="spearman_rho_vs_boundary_distance")
    .reindex(columns=list(metric_labels.keys()))
)
top_corr_types = corr_pivot.abs().max(axis=1).nlargest(40).index
corr_pivot = corr_pivot.loc[top_corr_types].sort_values("syn_inh")
fig, ax = plt.subplots(figsize=(7, max(5, 0.24 * len(corr_pivot))), constrained_layout=True)
limit = robust_abs_limit(corr_pivot.to_numpy(), percentile=100)
image = ax.imshow(corr_pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
ax.set(
    xticks=np.arange(len(corr_pivot.columns)),
    yticks=np.arange(len(corr_pivot.index)),
    title="boundary distance との Spearman 相関（|相関| 上位 type）",
)
ax.set_xticklabels([metric_labels[c] for c in corr_pivot.columns], rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(corr_pivot.index, fontsize=6)
fig.colorbar(image, ax=ax, shrink=0.6, label="Spearman ρ")
save_figure(fig, "fig_boundary_distance_correlations")
plt.show()

# %% [markdown]
# ### 補填候補の heuristic summary
#
# ### 狙い
# 機構候補を見落とさないよう、主要な rim-center 差分を 1 表
# `20a_compensation_candidate_heuristic_summary` に束ね、type ごとに「どの機構が疑わしいか」を
# フラグで俯瞰する。
#
# ### 4 つの候補フラグ（しきい値はあくまで探索用の目安）
# - `candidate_edge_strengthening`: `inh_syn_per_edge` の rim/center 比 > 1.10
#   → edge 1 本あたりの増強。
# - `candidate_source_switching`: wide-field 比率の rim−center 差 > 0.05
#   → column 性から wide-field source への切り替え。
# - `candidate_inward_recentering`: 内向き射影の rim−center 差 > 0.20
#   → kernel の視野内側へのずれ。
# - `candidate_columnar_residual_maintenance`: §6 の幾何残差 rim/center 比 >= 1.0
#   → 切断モデルを超える column 性入力の維持。
#
# ### 注意
# これらの boolean は候補探索のための heuristic であり、統計的有意性や機能的補償を意味しない。
# とくに T5 のように column-assigned な抑制 source がほぼ存在しない type では
# `geometry_residual_rim_over_center` は不安定なので解釈せず、all-source 入力と
# source composition の側で判断する。

# %%
geometry_inh = compensation_summary[
    (compensation_summary["sign"] == "inh") & compensation_summary["region"].isin(["rim", "center"])
].pivot(
    index=["receiver_type", "receiver_hemisphere"],
    columns="region",
    values="median_observed_over_geometry",
)
geometry_inh.columns = [f"geometry_residual_{region}" for region in geometry_inh.columns]
geometry_inh = geometry_inh.reset_index()
for region in ["rim", "center"]:
    column = f"geometry_residual_{region}"
    if column not in geometry_inh:
        geometry_inh[column] = np.nan
geometry_inh["geometry_residual_rim_over_center"] = safe_div(
    geometry_inh["geometry_residual_rim"], geometry_inh["geometry_residual_center"]
)

candidate_summary = rim_center.merge(
    geometry_inh,
    on=["receiver_type", "receiver_hemisphere"],
    how="left",
    validate="one_to_one",
)
candidate_summary["inh_widefield_fraction_rim_minus_center"] = (
    candidate_summary["inh_widefield_or_unassigned_fraction_rim"]
    - candidate_summary["inh_widefield_or_unassigned_fraction_center"]
)
candidate_summary["inh_inward_projection_rim_minus_center"] = (
    candidate_summary["inh_mean_inward_projection_rim"]
    - candidate_summary["inh_mean_inward_projection_center"]
)
candidate_summary["candidate_edge_strengthening"] = (
    candidate_summary["inh_syn_per_edge_rim_over_center"] > 1.10
)
candidate_summary["candidate_source_switching"] = (
    candidate_summary["inh_widefield_fraction_rim_minus_center"] > 0.05
)
candidate_summary["candidate_inward_recentering"] = (
    candidate_summary["inh_inward_projection_rim_minus_center"] > 0.20
)
candidate_summary["candidate_columnar_residual_maintenance"] = (
    candidate_summary["geometry_residual_rim_over_center"] >= 1.0
)
save_table(candidate_summary, "20a_compensation_candidate_heuristic_summary")

# %% [markdown]
# ## 8. T4/T5 raw edges と direct lateral coupling
#
# ### 狙い
# ここから仮説 2（T4/T5 lateral circuit）に移る。まず T4/T5 を中心とした生 edge を 3 種類に分けて
# 保存し、後段の coupling / motif 解析の素材をそろえる。
#
# ### 保存する 3 つの生表
# - `21_t4t5_input_edges_raw`: T4/T5 への全入力 edge（§4 の主表から T4/T5 受け手を抽出）。
# - `22_t4t5_output_edges_raw`: T4/T5 からの全出力 edge。post 側の column 座標を結合し、
#   同一 hemisphere の相対変位 `(dp, dq)`・距離・実空間変位を付ける。
# - `23_t4t5_to_t4t5_edges_raw`: T4/T5 → T4/T5 の直接 edge。送り手と受け手の関係を
#   `homotypic`（同亜型, 例 `T4a→T4a`）/ `same_pathway_other_subtype`（同経路の別亜型）/
#   `cross_pathway` に分類する。
#
# ### なぜ相対 column 座標を残すか
# homotypic edge に source→target の相対変位を保持しておくと、coupling が
# **等方的な一様増幅** なのか、preferred-direction 軸に沿った **異方的 coupling** なのかを
# 後から検定できる（§10 の異方性ベクトル解析につながる）。type レベルで距離・変位別に集計した
# `24_t4t5_direct_lateral_kernel_raw` と動径要約 `25_t4t5_direct_lateral_radial_summary` も出力する。

# %%
motion_cells = receiver_cells[receiver_cells["type"].map(is_motion_type)].copy()
motion_ids = set(motion_cells["root_id"])
motion_inputs = receiver_edges[receiver_edges["post_root_id"].isin(motion_ids)].copy()
save_table(motion_inputs, "21_t4t5_input_edges_raw", compress=True)

motion_source_meta = motion_cells[
    ["root_id", "type", "hemisphere", "p", "q", "boundary_distance", "region"]
].rename(
    columns={
        "root_id": "pre_root_id",
        "type": "motion_source_type",
        "hemisphere": "motion_source_hemisphere",
        "p": "motion_source_p",
        "q": "motion_source_q",
        "boundary_distance": "motion_source_boundary_distance",
        "region": "motion_source_region",
    }
)
post_column_meta = cell_columns[["root_id", "type", "hemisphere", "p", "q"]].rename(
    columns={
        "root_id": "post_root_id",
        "type": "post_column_type",
        "hemisphere": "post_column_hemisphere",
        "p": "post_column_p",
        "q": "post_column_q",
    }
)
motion_outputs = conn[conn["pre_root_id"].isin(motion_ids)].copy()
motion_outputs = motion_outputs.merge(motion_source_meta, on="pre_root_id", how="inner", validate="many_to_one")
motion_outputs = motion_outputs.merge(post_column_meta, on="post_root_id", how="left", validate="many_to_one")
motion_outputs["post_has_column"] = motion_outputs["post_column_p"].notna()
motion_outputs["post_same_hemisphere"] = (
    motion_outputs["post_has_column"]
    & (motion_outputs["post_column_hemisphere"] == motion_outputs["motion_source_hemisphere"])
)
motion_outputs["dp"] = np.where(
    motion_outputs["post_same_hemisphere"],
    motion_outputs["post_column_p"] - motion_outputs["motion_source_p"],
    np.nan,
)
motion_outputs["dq"] = np.where(
    motion_outputs["post_same_hemisphere"],
    motion_outputs["post_column_q"] - motion_outputs["motion_source_q"],
    np.nan,
)
motion_outputs["distance"] = hexd(motion_outputs["dp"], motion_outputs["dq"])
motion_outputs["dx"], motion_outputs["dy"] = axial_to_cart(motion_outputs["dp"], motion_outputs["dq"])
save_table(motion_outputs, "22_t4t5_output_edges_raw", compress=True)

motion_to_motion = motion_outputs[motion_outputs["post_primary_type"].map(is_motion_type)].copy()
motion_to_motion["relation"] = np.select(
    [
        motion_to_motion["motion_source_type"] == motion_to_motion["post_primary_type"],
        motion_to_motion["motion_source_type"].str[:2] == motion_to_motion["post_primary_type"].str[:2],
    ],
    ["homotypic", "same_pathway_other_subtype"],
    default="cross_pathway",
)
save_table(motion_to_motion, "23_t4t5_to_t4t5_edges_raw", compress=True)

motion_direct_kernel = (
    motion_to_motion[motion_to_motion["post_same_hemisphere"]]
    .groupby(
        [
            "motion_source_type",
            "post_primary_type",
            "motion_source_hemisphere",
            "motion_source_region",
            "relation",
            "dp",
            "dq",
            "distance",
        ],
        as_index=False,
    )
    .agg(synapses=("syn_count", "sum"), edges=("syn_count", "size"), source_cells=("pre_root_id", "nunique"))
)
save_table(motion_direct_kernel, "24_t4t5_direct_lateral_kernel_raw")

motion_direct_radial = (
    motion_direct_kernel.groupby(
        [
            "motion_source_type",
            "post_primary_type",
            "motion_source_hemisphere",
            "motion_source_region",
            "relation",
            "distance",
        ],
        as_index=False,
    )[["synapses", "edges", "source_cells"]]
    .sum()
)
save_table(motion_direct_radial, "25_t4t5_direct_lateral_radial_summary")

# %% [markdown]
# ### 図: T4/T5 の直接 lateral coupling（関係種別の動径プロファイル）
#
# homotypic / 同経路の別亜型 / cross-pathway という関係種別ごとに、T4/T5 → T4/T5 の直接結合が
# 距離方向にどう分布するかを比較する。

# %%
fig, ax = plt.subplots(figsize=(8, 5))
relation_colors = {
    "homotypic": "tab:red",
    "same_pathway_other_subtype": "tab:green",
    "cross_pathway": "tab:gray",
}
direct_radial = (
    motion_direct_radial[motion_direct_radial["distance"] <= 6]
    .groupby(["relation", "distance"], as_index=False)["synapses"]
    .sum()
)
for relation, color in relation_colors.items():
    group = direct_radial[direct_radial["relation"] == relation].sort_values("distance")
    if group.empty:
        continue
    ax.plot(group["distance"], group["synapses"], marker="o", color=color, label=relation)
ax.set(
    xlabel="距離（hex column）",
    ylabel="シナプス総数",
    title="T4/T5 → T4/T5 の直接 coupling: 関係種別の動径プロファイル",
)
ax.legend(frameon=False, title="送り手と受け手の関係")
fig.tight_layout()
save_figure(fig, "fig_t4t5_direct_lateral_radial")
plt.show()

# %% [markdown]
# ## 9. T4/T5 recurrent motif candidates
#
# ### 狙い
# T4/T5 が中間ニューロン J を介して T4/T5 に戻る 2 ステップ経路 `T4/T5 → J → T4/T5` を洗い出す。
# とくに後半 `J → T4/T5` が putative inhibitory な motif は、opponent inhibition（方向反対の抑制）や
# recurrent inhibition の回路候補になる。
#
# ### 方法
# 経路を 2 本の leg に分けて扱う。
# - **leg1** `T4/T5 → J`: §8 の出力 edge を流用（`26_t4t5_motif_leg1_output_edges_raw`）。
# - **leg2** `J → T4/T5`: §8 の入力 edge を流用し、J の出力符号 `intermediate_output_sign` を保持
#   （`27_t4t5_motif_leg2_input_edges_raw`）。
#
# 両 leg を中間型 J で結合し、`leg1_synapses × leg2_synapses` を `path_score` として type レベルで
# 順位付けする（`28_t4t5_two_step_motif_type_ranking`）。さらに `J → T4/T5` が抑制性のものだけを
# 抜き出した `29_t4t5_inhibitory_two_step_motif_candidates` を作る。
#
# ### なぜ type レベルに留めるか
# CT1 のように発散・収束が大きい中間ニューロンを cell 単位で完全展開すると、経路数が組合せ爆発する。
# そこで raw な edge leg は保存しつつ、ランキングは type レベルに留める。有望な候補を選んだ後に、
# その候補だけ cell-level・column-level へ展開する運用を想定している。

# %%
motif_leg1_raw = motion_outputs.rename(
    columns={
        "motion_source_type": "motion_pre_type",
        "post_root_id": "intermediate_root_id",
        "post_primary_type": "intermediate_type",
        "syn_count": "leg1_syn_count",
    }
)
motif_leg2_raw = motion_inputs.rename(
    columns={
        "pre_root_id": "intermediate_root_id",
        "pre_primary_type": "intermediate_type",
        "post_primary_type": "motion_post_type",
        "syn_count": "leg2_syn_count",
        "sign": "intermediate_output_sign",
    }
)
save_table(motif_leg1_raw, "26_t4t5_motif_leg1_output_edges_raw", compress=True)
save_table(motif_leg2_raw, "27_t4t5_motif_leg2_input_edges_raw", compress=True)

motif_leg1_type = (
    motif_leg1_raw.groupby(["motion_pre_type", "intermediate_type"], as_index=False)
    .agg(leg1_synapses=("leg1_syn_count", "sum"), leg1_edges=("leg1_syn_count", "size"))
)
motif_leg2_type = (
    motif_leg2_raw.groupby(["intermediate_type", "motion_post_type", "intermediate_output_sign"], as_index=False)
    .agg(leg2_synapses=("leg2_syn_count", "sum"), leg2_edges=("leg2_syn_count", "size"))
)
motif_candidates = motif_leg1_type.merge(motif_leg2_type, on="intermediate_type", how="inner")
motif_candidates["path_score_synapse_product"] = (
    motif_candidates["leg1_synapses"] * motif_candidates["leg2_synapses"]
)
motif_candidates = motif_candidates.sort_values("path_score_synapse_product", ascending=False)
save_table(motif_candidates, "28_t4t5_two_step_motif_type_ranking")

inhibitory_motif_candidates = motif_candidates[
    motif_candidates["intermediate_output_sign"] == "inh"
].copy()
save_table(inhibitory_motif_candidates, "29_t4t5_inhibitory_two_step_motif_candidates")

# %% [markdown]
# ### 図: 2-step motif の中間 type ランキング（出力符号別）
#
# fig10 は抑制性 motif のみを扱うので、ここでは全符号を含めて、経路スコアが大きい中間 type J を
# 俯瞰する。各バーを J の出力符号（exc/inh/other）で積み上げる。

# %%
motif_by_sign = motif_candidates.groupby(
    ["intermediate_type", "intermediate_output_sign"], as_index=False
)["path_score_synapse_product"].sum()
top_intermediate = (
    motif_by_sign.groupby("intermediate_type")["path_score_synapse_product"].sum().nlargest(20).index
)
sign_colors = {"exc": "tab:red", "inh": "tab:blue", "other": "tab:gray"}
xpos = np.arange(len(top_intermediate))
bottoms = np.zeros(len(top_intermediate))
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
for sign, color in sign_colors.items():
    values = []
    for intermediate_type in top_intermediate:
        match = motif_by_sign[
            (motif_by_sign["intermediate_type"] == intermediate_type)
            & (motif_by_sign["intermediate_output_sign"] == sign)
        ]["path_score_synapse_product"]
        values.append(float(match.iloc[0]) if len(match) else 0.0)
    ax.bar(xpos, values, bottom=bottoms, color=color, label=sign)
    bottoms += np.asarray(values)
ax.set(
    xticks=xpos,
    ylabel="path score の合計",
    title="T4/T5 2-step motif: 中間 type ランキング（出力符号別, 上位20）",
)
ax.set_xticklabels(top_intermediate, rotation=90, fontsize=7)
ax.legend(frameon=False, title="J の出力符号")
save_figure(fig, "fig_t4t5_two_step_motif_ranking")
plt.show()

# %% [markdown]
# ## 10. T4/T5 rim-center comparison
#
# ### 狙い
# 仮説 1（視野端の補填）と仮説 2（lateral circuit）を T4/T5 に絞って突き合わせる。
# T4/T5 について、入力 source の構成・幾何残差・homotypic coupling を rim と center で比較する。
#
# ### 出力
# - `30_t4t5_input_source_by_region`: T4/T5 への入力を source 型 × 符号 × 領域で集計し、
#   領域内構成比を付ける。
# - `31_t4t5_geometry_model_residual_raw` / `32_t4t5_per_cell_input_metrics_raw`:
#   §5–§6 の per-cell 指標・幾何残差を T4/T5 だけ抜き出したもの。
# - `33_t4t5_homotypic_radial_by_region`: homotypic（同亜型）入力を距離別・領域別に集計し、
#   source cell 1 個あたりシナプス数も出す。次のセルでこれを opportunity 補正する。

# %%
t4t5_input_source_region = (
    motion_inputs.groupby(
        ["receiver_type", "receiver_hemisphere", "region", "pre_primary_type", "sign"],
        as_index=False,
    )
    .agg(synapses=("syn_count", "sum"), edges=("syn_count", "size"), receiver_cells=("post_root_id", "nunique"))
)
t4t5_input_source_region["source_fraction_within_region"] = t4t5_input_source_region.groupby(
    ["receiver_type", "receiver_hemisphere", "region"]
)["synapses"].transform(lambda values: values / values.sum())
save_table(t4t5_input_source_region, "30_t4t5_input_source_by_region")

t4t5_compensation = compensation_by_cell[
    compensation_by_cell["receiver_type"].map(is_motion_type)
].copy()
save_table(t4t5_compensation, "31_t4t5_geometry_model_residual_raw")

t4t5_per_cell = per_cell[per_cell["receiver_type"].map(is_motion_type)].copy()
save_table(t4t5_per_cell, "32_t4t5_per_cell_input_metrics_raw")

homotypic = motion_to_motion[
    (motion_to_motion["relation"] == "homotypic") & motion_to_motion["post_same_hemisphere"]
].copy()
homotypic_summary = (
    homotypic.groupby(
        ["motion_source_type", "motion_source_hemisphere", "motion_source_region", "distance"],
        as_index=False,
    )
    .agg(
        synapses=("syn_count", "sum"),
        edges=("syn_count", "size"),
        source_cells=("pre_root_id", "nunique"),
        target_cells=("post_root_id", "nunique"),
    )
)
homotypic_summary["synapses_per_source_cell"] = safe_div(
    homotypic_summary["synapses"], homotypic_summary["source_cells"]
)
save_table(homotypic_summary, "33_t4t5_homotypic_radial_by_region")

# %% [markdown]
# ### 図: Homotypic coupling の動径プロファイル（補正前）
#
# opportunity 補正前の「source cell あたりシナプス数」。補正後の fig02 / fig08 と見比べることで、
# 機会数補正がどの程度効くか（rim と center の見かけの差がどう変わるか）を確認できる。

# %%
fig, ax = plt.subplots(figsize=(9, 5))
homotypic_radial_uncorr = (
    homotypic_summary[homotypic_summary["distance"] <= 6]
    .groupby(["motion_source_type", "motion_source_region", "distance"], as_index=False)[
        "synapses_per_source_cell"
    ]
    .mean()
)
for (subtype, region), group in homotypic_radial_uncorr.groupby(
    ["motion_source_type", "motion_source_region"]
):
    if region not in {"rim", "center"}:
        continue
    group = group.sort_values("distance")
    ax.plot(
        group["distance"],
        group["synapses_per_source_cell"],
        marker="o",
        label=f"{subtype} {region}",
        alpha=0.8 if region == "rim" else 0.45,
        linestyle="-" if region == "rim" else "--",
    )
ax.set(
    xlabel="距離（hex column）",
    ylabel="source cell あたりシナプス数",
    title="T4/T5 homotypic coupling: 補正前の動径プロファイル（rim 実線 / center 破線）",
)
ax.legend(ncol=4, fontsize=7, frameon=False)
fig.tight_layout()
save_figure(fig, "fig_t4t5_homotypic_radial_uncorrected")
plt.show()

# %% [markdown]
# ### Homotypic coupling の opportunity correction
#
# ### なぜ補正が要るか
# `synapses / source cell` だけでは、rim では「つなぐ相手（同亜型の target cell）」自体が少ないという
# 効果を除けない。これは §3 と同じ「機会数で割る」発想を homotypic coupling に適用する操作である。
#
# ### 方法
# T4/T5 の subtype × hemisphere ごとに、実際に column を持つ source–target ペアを総当たりで列挙し、
# 距離・変位 `(dp, dq)` 別の **可能ペア数 `n_possible_pairs`**（＝機会数）を数える。自己結合候補
# `source == target`（autapse）は除外する。これを観測 homotypic edge と突き合わせ、
# - `connection_probability = edges / n_possible_pairs`
# - `synapses_per_possible_pair = synapses / n_possible_pairs`
# を求める（`34_…opportunity`, `35_…kernel_corrected`, `36_…radial_corrected`）。
#
# ### 異方性ベクトル
# さらに変位をシナプス数で重み付けした平均オフセット `(mean_offset_x, y)`・その大きさ・角度を
# 領域別に出し（`37_t4t5_homotypic_anisotropy_summary`）、coupling が等方か、特定方向
# （preferred-direction 軸）に偏るかを定量する。

# %%
homotypic_opportunity_parts = []
for (subtype, hemi), cells in motion_cells.groupby(["type", "hemisphere"]):
    source = cells[["root_id", "p", "q", "region"]].rename(
        columns={
            "root_id": "source_root_id",
            "p": "source_p",
            "q": "source_q",
            "region": "source_region",
        }
    )
    target = cells[["root_id", "p", "q"]].rename(
        columns={"root_id": "target_root_id", "p": "target_p", "q": "target_q"}
    )
    pairs = source.merge(target, how="cross")
    pairs = pairs[pairs["source_root_id"] != pairs["target_root_id"]].copy()
    pairs["dp"] = pairs["target_p"] - pairs["source_p"]
    pairs["dq"] = pairs["target_q"] - pairs["source_q"]
    pairs["distance"] = hexd(pairs["dp"], pairs["dq"]).astype(int)
    opportunities = (
        pairs.groupby(["source_region", "dp", "dq", "distance"], as_index=False)
        .size()
        .rename(columns={"size": "n_possible_pairs"})
    )
    opportunities["motion_source_type"] = subtype
    opportunities["motion_source_hemisphere"] = hemi
    homotypic_opportunity_parts.append(opportunities)

homotypic_opportunities = pd.concat(homotypic_opportunity_parts, ignore_index=True)
save_table(homotypic_opportunities, "34_t4t5_homotypic_opportunity_by_displacement")

homotypic_observed = (
    homotypic.groupby(
        [
            "motion_source_type",
            "motion_source_hemisphere",
            "motion_source_region",
            "dp",
            "dq",
            "distance",
        ],
        as_index=False,
    )
    .agg(synapses=("syn_count", "sum"), edges=("syn_count", "size"))
)
homotypic_corrected = homotypic_opportunities.merge(
    homotypic_observed,
    left_on=[
        "motion_source_type",
        "motion_source_hemisphere",
        "source_region",
        "dp",
        "dq",
        "distance",
    ],
    right_on=[
        "motion_source_type",
        "motion_source_hemisphere",
        "motion_source_region",
        "dp",
        "dq",
        "distance",
    ],
    how="left",
).drop(columns="motion_source_region")
homotypic_corrected["synapses"] = homotypic_corrected["synapses"].fillna(0)
homotypic_corrected["edges"] = homotypic_corrected["edges"].fillna(0)
homotypic_corrected["connection_probability"] = safe_div(
    homotypic_corrected["edges"], homotypic_corrected["n_possible_pairs"]
)
homotypic_corrected["synapses_per_possible_pair"] = safe_div(
    homotypic_corrected["synapses"], homotypic_corrected["n_possible_pairs"]
)
save_table(homotypic_corrected, "35_t4t5_homotypic_kernel_opportunity_corrected")

homotypic_corrected_radial = (
    homotypic_corrected.groupby(
        ["motion_source_type", "motion_source_hemisphere", "source_region", "distance"],
        as_index=False,
    )[["n_possible_pairs", "synapses", "edges"]]
    .sum()
)
homotypic_corrected_radial["connection_probability"] = safe_div(
    homotypic_corrected_radial["edges"], homotypic_corrected_radial["n_possible_pairs"]
)
homotypic_corrected_radial["synapses_per_possible_pair"] = safe_div(
    homotypic_corrected_radial["synapses"], homotypic_corrected_radial["n_possible_pairs"]
)
save_table(homotypic_corrected_radial, "36_t4t5_homotypic_radial_opportunity_corrected")

homotypic_vector = homotypic_corrected.copy()
homotypic_vector["dx"], homotypic_vector["dy"] = axial_to_cart(
    homotypic_vector["dp"], homotypic_vector["dq"]
)
homotypic_vector["weighted_dx"] = homotypic_vector["dx"] * homotypic_vector["synapses"]
homotypic_vector["weighted_dy"] = homotypic_vector["dy"] * homotypic_vector["synapses"]
homotypic_anisotropy = (
    homotypic_vector.groupby(
        ["motion_source_type", "motion_source_hemisphere", "source_region"],
        as_index=False,
    )
    .agg(
        synapses=("synapses", "sum"),
        weighted_dx=("weighted_dx", "sum"),
        weighted_dy=("weighted_dy", "sum"),
    )
)
homotypic_anisotropy["mean_offset_x"] = safe_div(
    homotypic_anisotropy["weighted_dx"], homotypic_anisotropy["synapses"]
)
homotypic_anisotropy["mean_offset_y"] = safe_div(
    homotypic_anisotropy["weighted_dy"], homotypic_anisotropy["synapses"]
)
homotypic_anisotropy["mean_offset_magnitude"] = np.hypot(
    homotypic_anisotropy["mean_offset_x"], homotypic_anisotropy["mean_offset_y"]
)
homotypic_anisotropy["mean_offset_angle_deg"] = (
    np.degrees(
        np.arctan2(homotypic_anisotropy["mean_offset_y"], homotypic_anisotropy["mean_offset_x"])
    )
    % 360
)
save_table(homotypic_anisotropy, "37_t4t5_homotypic_anisotropy_summary")

# %% [markdown]
# ### 図: Homotypic coupling の機会数（可能ペア数）
#
# 補正の分母にあたる「同亜型ペアが存在しうる数」を距離・領域別に可視化する（表 34）。rim では
# 各距離で可能ペアが少ないため、補正前後（補正前の図と fig02/fig08）で見え方が変わる理由になる。

# %%
fig, ax = plt.subplots(figsize=(8, 5))
region_styles = {"center": ("tab:blue", "-"), "middle": ("tab:orange", "--"), "rim": ("tab:red", "-")}
homo_opp = (
    homotypic_opportunities[homotypic_opportunities["distance"] <= 6]
    .groupby(
        ["motion_source_type", "motion_source_hemisphere", "source_region", "distance"],
        as_index=False,
    )["n_possible_pairs"]
    .sum()
    .groupby(["source_region", "distance"], as_index=False)["n_possible_pairs"]
    .mean()
)
for region in ["center", "middle", "rim"]:
    group = homo_opp[homo_opp["source_region"] == region].sort_values("distance")
    if group.empty:
        continue
    color, linestyle = region_styles[region]
    ax.plot(
        group["distance"],
        group["n_possible_pairs"],
        marker="o",
        color=color,
        linestyle=linestyle,
        label=region,
    )
ax.set(
    xlabel="距離（hex column）",
    ylabel="可能な homotypic ペア数（subtype 平均）",
    title="Homotypic coupling の機会数: 距離・領域別",
)
ax.legend(frameon=False, title="source の領域")
fig.tight_layout()
save_figure(fig, "fig_t4t5_homotypic_opportunity_by_distance")
plt.show()

# %% [markdown]
# ### T4/T5 source-switch ranking
#
# T4/T5 への入力を source 型 × 符号ごとに、rim と center の領域内構成比の差
# `rim_minus_center_fraction` で順位付けする（`38_t4t5_input_source_switch_ranking`）。
# 既知の主要回路だけに絞らず、低順位の候補も残すことで、rim 特異的に出入りする少数派 source を
# 見落とさないようにする。正の値は rim で増える source、負の値は rim で減る source を表す。

# %%
t4t5_source_switch = t4t5_input_source_region.pivot_table(
    index=["receiver_type", "receiver_hemisphere", "pre_primary_type", "sign"],
    columns="region",
    values="source_fraction_within_region",
    fill_value=0,
).reset_index()
for region in ["rim", "center"]:
    if region not in t4t5_source_switch:
        t4t5_source_switch[region] = 0.0
t4t5_source_switch["rim_minus_center_fraction"] = t4t5_source_switch["rim"] - t4t5_source_switch["center"]
t4t5_source_switch = t4t5_source_switch.sort_values("rim_minus_center_fraction", ascending=False)
save_table(t4t5_source_switch, "38_t4t5_input_source_switch_ranking")

# %% [markdown]
# ## 11. Quick-look figures
#
# raw table を主成果としつつ、候補のあたりを付けるための概要図を 2 枚出力する。
#
# - **fig01**: T4 / T5 各亜型について（左右パネル）、§6 の幾何残差 `observed / geometry-expected` を
#   center→rim で結んだ図。破線 `y=1` が「切断モデル通り」の基準線で、これを上回れば column 性
#   抑制入力が切断以上に維持されている候補。
# - **fig02**: T4/T5 homotypic coupling の opportunity 補正済み動径プロファイル
#   （`synapses / possible pair` を距離に対して）。rim を実線、center を破線で示し、
#   視野端で近傍 coupling が強まる/弱まる傾向を見る。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, pathway in zip(axes, ["T4", "T5"]):
    plot_df = compensation_summary[
        (compensation_summary["sign"] == "inh")
        & compensation_summary["region"].isin(["rim", "center"])
        & compensation_summary["receiver_type"].str.startswith(pathway)
    ].copy()
    for (receiver_type, hemi), group in plot_df.groupby(["receiver_type", "receiver_hemisphere"]):
        values = group.set_index("region")["median_observed_over_geometry"]
        if {"rim", "center"}.issubset(values.index):
            ax.plot(
                [0, 1],
                [values["center"], values["rim"]],
                marker="o",
                alpha=0.65,
                label=f"{receiver_type} {hemi}",
            )
    ax.axhline(1, color="black", linewidth=1, linestyle="--")
    ax.set(
        xticks=[0, 1],
        xticklabels=["center", "rim"],
        ylabel="観測値 / 幾何期待値（columnar 抑制性シナプス）",
        title=pathway,
    )
    ax.legend(ncol=2, fontsize=7, frameon=False)
fig.suptitle("等方的な幾何切断モデルに対する columnar 抑制性入力の残差")
save_figure(fig, "fig01_t4t5_geometry_residual")
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
for (subtype, region), group in homotypic_corrected_radial.groupby(["motion_source_type", "source_region"]):
    if region not in {"rim", "center"}:
        continue
    group = group.sort_values("distance")
    ax.plot(
        group["distance"],
        group["synapses_per_possible_pair"],
        marker="o",
        label=f"{subtype} {region}",
        alpha=0.8 if region == "rim" else 0.45,
        linestyle="-" if region == "rim" else "--",
    )
ax.set(
    xlabel="source-target 間距離（hex column）",
    ylabel="homotypic シナプス数 / 可能なペア数",
    title="T4/T5 homotypic coupling: opportunity 補正済みの動径プロファイル",
)
ax.legend(ncol=4, fontsize=7, frameon=False)
fig.tight_layout()
save_figure(fig, "fig02_t4t5_homotypic_radial")
plt.show()

# %% [markdown]
# ## 12. Interpretability figures
#
# raw table だけでは候補を見落としやすいため、解釈用の図をまとめて保存する。各図の対応は次の通り。
#
# - **fig03** reference grid と boundary distance: §2 の格子と rim（赤枠）の位置を可視化し、
#   割り当ての穴などのアーティファクトを目視点検する。
# - **fig04 / fig05** 候補機構 heatmap: §7 の `candidate_summary` を、全 receiver type と T4/T5 に
#   ついて、5 指標（抑制シナプス比・edge あたり・wide-field 比率・内向き射影・幾何残差）の
#   rim-center 差で色分けする。
# - **fig06 / fig07** source 構成: T4/T5 入力の source switching（rim−center の構成比差）と、
#   center vs rim の積み上げ構成比。
# - **fig08 / fig09** homotypic coupling: opportunity 補正済みの hex kernel（亜型 × center/rim）と、
#   平均オフセットベクトル（左右 × center/rim）で coupling の異方性を見る。
# - **fig10** recurrent inhibitory motif: §9 の抑制性 2-step motif 候補を中間型 × 端点で
#   heatmap 表示する。
# - **per-type バルク図**: 全 receiver type・左右半球ごとの per-cell 空間マップと
#   boundary-distance profile（下のセル）。
#
# 最後の per-type 図は枚数が多いため、`figures/per_type_spatial_maps/` と
# `figures/per_type_boundary_profiles/` に分けて保存する。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, (hemi, group) in zip(axes, column_geometry.groupby("hemisphere")):
    scatter = ax.scatter(
        group["x"],
        group["y"],
        c=group["boundary_distance"],
        cmap="viridis",
        marker="h",
        s=34,
        linewidths=0,
    )
    rim = group["region"] == "rim"
    ax.scatter(
        group.loc[rim, "x"],
        group.loc[rim, "y"],
        facecolors="none",
        edgecolors="red",
        marker="h",
        s=42,
        linewidths=0.35,
    )
    ax.set(aspect="equal", title=f"{hemi}: Mi1 リファレンスグリッド")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, shrink=0.72, label="境界からの距離")
fig.suptitle("リファレンス column のジオメトリ（赤枠 = rim）")
save_figure(fig, "fig03_reference_column_boundary_distance")
plt.show()


def plot_compensation_heatmap(df, stem, title, *, annotate=False):
    specs = [
        ("syn_inh_rim_over_center", "抑制性シナプス\nrim / center", 1.0),
        ("inh_syn_per_edge_rim_over_center", "抑制性シナプス / edge\nrim / center", 1.0),
        ("inh_widefield_fraction_rim_minus_center", "wide-field 比率\nrim - center", 0.0),
        ("inh_inward_projection_rim_minus_center", "内向き射影\nrim - center", 0.0),
        ("geometry_residual_rim_over_center", "columnar 残差\nrim / center", 1.0),
    ]
    table = df.sort_values(["receiver_type", "receiver_hemisphere"]).copy()
    labels = table["receiver_type"] + " " + table["receiver_hemisphere"].str[0].str.upper()
    fig, axes = plt.subplots(1, len(specs), figsize=(15, max(5, 0.25 * len(table))), constrained_layout=True)
    for ax, (metric, label, midpoint) in zip(axes, specs):
        values = table[metric].astype(float).to_numpy()
        limit = robust_abs_limit(values - midpoint)
        image = ax.imshow(
            values[:, None],
            aspect="auto",
            cmap="coolwarm",
            vmin=midpoint - limit,
            vmax=midpoint + limit,
        )
        ax.set(xticks=[0], xticklabels=[label], yticks=np.arange(len(table)))
        ax.set_yticklabels(labels if ax is axes[0] else [])
        if annotate:
            for row, value in enumerate(values):
                if np.isfinite(value):
                    ax.text(0, row, f"{value:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle(title)
    save_figure(fig, stem)
    plt.show()


plot_compensation_heatmap(
    candidate_summary,
    "fig04_all_types_compensation_candidate_heatmap",
    "全 columnar receiver type: edge 補填候補の指標",
)
plot_compensation_heatmap(
    candidate_summary[candidate_summary["receiver_type"].map(is_motion_type)],
    "fig05_t4t5_compensation_candidate_heatmap",
    "T4/T5: edge 補填候補の指標",
    annotate=True,
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
for ax, sign in zip(axes, ["inh", "exc"]):
    subset = t4t5_source_switch[t4t5_source_switch["sign"] == sign].copy()
    top_sources = (
        subset.groupby("pre_primary_type")["rim_minus_center_fraction"]
        .apply(lambda values: values.abs().sum())
        .nlargest(18)
        .index
    )
    heat = subset[subset["pre_primary_type"].isin(top_sources)].pivot_table(
        index=["receiver_type", "receiver_hemisphere"],
        columns="pre_primary_type",
        values="rim_minus_center_fraction",
        fill_value=0,
    )
    heat = heat.sort_index()
    values = heat.to_numpy()
    limit = robust_abs_limit(values)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(
        xticks=np.arange(len(heat.columns)),
        yticks=np.arange(len(heat.index)),
        title=f"{sign}: source 構成比 rim - center",
    )
    ax.set_xticklabels(heat.columns, rotation=90, fontsize=7)
    ax.set_yticklabels([f"{cell_type} {hemi[0].upper()}" for cell_type, hemi in heat.index], fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.72)
fig.suptitle("T4/T5 入力の source switching: 変化が大きいもの")
save_figure(fig, "fig06_t4t5_input_source_switch_heatmap")
plt.show()

# %%
composition = (
    t4t5_input_source_region.groupby(["receiver_type", "region", "pre_primary_type"], as_index=False)["synapses"]
    .sum()
)
top_composition_sources = (
    composition.groupby("pre_primary_type")["synapses"].sum().nlargest(14).index
)
composition["source_plot"] = np.where(
    composition["pre_primary_type"].isin(top_composition_sources),
    composition["pre_primary_type"],
    "other",
)
composition = (
    composition.groupby(["receiver_type", "region", "source_plot"], as_index=False)["synapses"]
    .sum()
)
composition["fraction"] = composition.groupby(["receiver_type", "region"])["synapses"].transform(
    lambda values: values / values.sum()
)
motion_order = [f"T{pathway}{subtype}" for pathway in [4, 5] for subtype in "abcd"]
source_order = list(top_composition_sources) + ["other"]
colors = plt.cm.tab20(np.linspace(0, 1, len(source_order)))
fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
for ax, pathway in zip(axes, ["T4", "T5"]):
    labels = [(subtype, region) for subtype in motion_order if subtype.startswith(pathway) for region in ["center", "rim"]]
    bottoms = np.zeros(len(labels))
    for source, color in zip(source_order, colors):
        values = []
        for subtype, region in labels:
            match = composition[
                (composition["receiver_type"] == subtype)
                & (composition["region"] == region)
                & (composition["source_plot"] == source)
            ]["fraction"]
            values.append(float(match.iloc[0]) if len(match) else 0.0)
        ax.bar(np.arange(len(labels)), values, bottom=bottoms, color=color, width=0.85, label=source)
        bottoms += np.asarray(values)
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=[f"{subtype}\n{region}" for subtype, region in labels],
        ylim=(0, 1),
        ylabel="入力シナプスの構成比",
        title=pathway,
    )
    ax.tick_params(axis="x", labelsize=7)
axes[1].legend(ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle("T4/T5 の入力構成: center vs rim（左右半球をまとめて集計）")
save_figure(fig, "fig07_t4t5_input_source_composition")
plt.show()

# %%
homotypic_hex = (
    homotypic_corrected.groupby(["motion_source_type", "source_region", "dp", "dq", "distance"], as_index=False)
    .agg(n_possible_pairs=("n_possible_pairs", "sum"), synapses=("synapses", "sum"), edges=("edges", "sum"))
)
homotypic_hex["synapses_per_possible_pair"] = safe_div(
    homotypic_hex["synapses"], homotypic_hex["n_possible_pairs"]
)
fig, axes = plt.subplots(8, 2, figsize=(7, 23), constrained_layout=True)
for row, subtype in enumerate(motion_order):
    subtype_data = homotypic_hex[(homotypic_hex["motion_source_type"] == subtype) & (homotypic_hex["distance"] <= 4)]
    vmax = max(float(subtype_data["synapses_per_possible_pair"].max()), 1e-9)
    for col, region in enumerate(["center", "rim"]):
        ax = axes[row, col]
        group = subtype_data[subtype_data["source_region"] == region]
        x, y = axial_to_cart(group["dp"], group["dq"])
        scatter = ax.scatter(
            x,
            y,
            c=group["synapses_per_possible_pair"],
            cmap="magma",
            vmin=0,
            vmax=vmax,
            marker="h",
            s=60,
            linewidths=0,
        )
        ax.scatter([0], [0], c="cyan", marker="+", s=45, linewidths=1)
        ax.set(aspect="equal", title=f"{subtype} {region}")
        ax.axis("off")
    fig.colorbar(scatter, ax=axes[row, :], shrink=0.65, label="シナプス数 / 可能なペア数")
fig.suptitle("T4/T5 homotypic coupling kernel（左右半球をまとめて集計, distance <= 4）")
save_figure(fig, "fig08_t4t5_homotypic_kernel_hexmaps")
plt.show()

# %%
fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
arrow_styles = {
    ("left", "center"): ("tab:blue", "-"),
    ("left", "rim"): ("tab:cyan", "--"),
    ("right", "center"): ("tab:red", "-"),
    ("right", "rim"): ("tab:orange", "--"),
}
for ax, subtype in zip(axes.ravel(), motion_order):
    group = homotypic_anisotropy[
        (homotypic_anisotropy["motion_source_type"] == subtype)
        & homotypic_anisotropy["source_region"].isin(["center", "rim"])
    ]
    for _, row in group.iterrows():
        color, linestyle = arrow_styles[(row["motion_source_hemisphere"], row["source_region"])]
        ax.annotate(
            "",
            xy=(row["mean_offset_x"], row["mean_offset_y"]),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": color, "linestyle": linestyle, "lw": 1.8},
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set(aspect="equal", title=subtype, xlim=(-2, 2), ylim=(-2, 2))
fig.suptitle("T4/T5 homotypic coupling の平均オフセットベクトル: 左右半球 × center/rim")
save_figure(fig, "fig09_t4t5_homotypic_anisotropy_vectors")
plt.show()

# %%
motif_top_types = (
    inhibitory_motif_candidates.groupby("intermediate_type")["path_score_synapse_product"]
    .sum()
    .nlargest(20)
    .index
)
motif_plot = inhibitory_motif_candidates[
    inhibitory_motif_candidates["intermediate_type"].isin(motif_top_types)
].copy()
fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
for ax, endpoint in zip(axes, ["motion_pre_type", "motion_post_type"]):
    heat = motif_plot.pivot_table(
        index="intermediate_type",
        columns=endpoint,
        values="path_score_synapse_product",
        aggfunc="sum",
        fill_value=0,
    )
    heat = heat.reindex(motif_top_types)
    values = np.log10(heat.to_numpy() + 1)
    image = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set(
        xticks=np.arange(len(heat.columns)),
        yticks=np.arange(len(heat.index)),
        title=f"中間 type vs {endpoint}",
    )
    ax.set_xticklabels(heat.columns, rotation=90)
    ax.set_yticklabels(heat.index)
    fig.colorbar(image, ax=ax, shrink=0.75, label="log10(path score の合計 + 1)")
fig.suptitle("T4/T5 putative 抑制性 two-step motif 候補")
save_figure(fig, "fig10_t4t5_inhibitory_two_step_motif_heatmap")
plt.show()

# %% [markdown]
# ### 全 receiver type の探索用バルク図
#
# ### 狙い
# 集計してしまうと消える「空間パターン」と「距離依存性」を type ごとに目で確認できるよう、
# 全 receiver type × hemisphere について 2 種類の図を機械的に量産する。
#
# ### 2 種類の図
# - **空間マップ（四枚組）**: per-cell の抑制シナプス数 / edge あたり / wide-field 比率 /
#   内向き射影を、各 cell の column 位置にそのまま（空間ビニングせず）配置する。黒枠は rim column。
#   視野端付近だけ値が変わるか、勾配があるかを直接見られる。
# - **boundary-distance profile**: 同じ 4 指標の中央値を boundary distance に沿ってプロットし、
#   rim 帯（赤の網掛け）に向かう単調傾向の有無を見る。元になる集計表は
#   `39_per_type_boundary_distance_profiles`。
#
# 各図は冒頭の出力方針どおりファイルに保存し、最後に全図の一覧を `40_figure_index` に残す。

# %%
profile_metrics = [
    ("syn_inh", "putative 抑制性シナプス数"),
    ("inh_syn_per_edge", "putative 抑制性シナプス数 / edge"),
    ("inh_widefield_or_unassigned_fraction", "wide-field / 未割当の抑制性比率"),
    ("inh_mean_inward_projection", "抑制性入力の内向き射影"),
]
profile_rows = []
for (receiver_type, hemi, boundary_distance), group in per_cell.groupby(
    ["receiver_type", "receiver_hemisphere", "boundary_distance"]
):
    row = {
        "receiver_type": receiver_type,
        "receiver_hemisphere": hemi,
        "boundary_distance": boundary_distance,
        "n_cells": len(group),
    }
    for metric, _ in profile_metrics:
        row[f"median_{metric}"] = group[metric].median()
    profile_rows.append(row)
boundary_profiles = pd.DataFrame(profile_rows)
save_table(boundary_profiles, "39_per_type_boundary_distance_profiles")

for (receiver_type, hemi), group in per_cell.groupby(["receiver_type", "receiver_hemisphere"]):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    plot_specs = [
        ("syn_inh", "putative 抑制性シナプス数", "viridis", None, None),
        ("inh_syn_per_edge", "putative 抑制性シナプス数 / edge", "viridis", None, None),
        ("inh_widefield_or_unassigned_fraction", "wide-field / 未割当の比率", "plasma", 0, 1),
        ("inh_mean_inward_projection", "平均内向き射影", "coolwarm", None, None),
    ]
    for ax, (metric, label, cmap, vmin, vmax) in zip(axes.ravel(), plot_specs):
        scatter = hex_metric(
            ax,
            group,
            metric,
            title=label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0 if metric == "inh_mean_inward_projection" else None,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.72)
    fig.suptitle(f"{receiver_type} {hemi}: per-cell 入力マップ（黒枠 = rim）")
    save_figure(fig, f"per_type_spatial_maps/{receiver_type}_{hemi}")
    plt.close(fig)

    profile = boundary_profiles[
        (boundary_profiles["receiver_type"] == receiver_type)
        & (boundary_profiles["receiver_hemisphere"] == hemi)
    ].sort_values("boundary_distance")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), profile_metrics):
        ax.plot(profile["boundary_distance"], profile[f"median_{metric}"], marker="o")
        ax.axvspan(-0.5, RIM_MAX_DISTANCE + 0.5, color="tab:red", alpha=0.08)
        ax.set(xlabel="境界からの距離", ylabel=f"{label}（中央値）")
    fig.suptitle(f"{receiver_type} {hemi}: 境界距離プロファイル")
    save_figure(fig, f"per_type_boundary_profiles/{receiver_type}_{hemi}")
    plt.close(fig)

# %%
figure_index = pd.DataFrame(
    [
        {"name": artifact["name"], "path": artifact["path"], "bytes": artifact["bytes"]}
        for artifact in artifacts
        if artifact["path"].endswith(".png")
    ]
)
save_table(figure_index, "40_figure_index")
print(f"saved figures: {len(figure_index):,}")

# %% [markdown]
# ## 13. Manifest
#
# ### 狙い
# 解析の再現性と監査のため、この実行で何をどう作ったかを 1 つの JSON にまとめる。
#
# ### 内容
# `manifest.json` に、実行時刻 (UTC)・notebook ソース・出力先のほか、解析条件
# （シナプス数しきい値、rim/center の距離定義、reference 型、興奮/抑制の伝達物質集合、
# 除外 receiver 型）と、全成果物の一覧（各表・図の名前・パス・行数・列・バイト数）を記録する。
# `notes` には、機能的符号が受容体依存であることや、`observed_over_geometry` が切断モデルに対する
# 構造残差にすぎないことなど、結果を解釈する際の主要な caveat を明示している。

# %%
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_source": "notebook/edge_compensation_t4t5_lateral.py",
    "output_directory": str(OUT_DIR),
    "parameters": {
        "min_synapses_per_connection": MIN_SYN,
        "rim_max_boundary_distance": RIM_MAX_DISTANCE,
        "center_min_boundary_distance": CENTER_MIN_DISTANCE,
        "reference_column_type": REFERENCE_COLUMN_TYPE,
        "putative_inhibitory_nt": sorted(INHIBITORY_NT),
        "putative_excitatory_nt": sorted(EXCITATORY_NT),
        "excluded_receiver_types": sorted(EXCLUDED_RECEIVER_TYPES),
    },
    "notes": [
        "Functional synaptic sign is receptor-dependent; sign is an operational transmitter-based label.",
        "The analysis includes internal edges between selected optic-lobe neurons, not every visual-system synapse.",
        "Mi1 column assignments define the reference grid; assignment gaps can mimic boundaries.",
        "observed_over_geometry is a structural residual relative to an isotropic truncation model, not proof of functional compensation.",
        "T4/T5 two-step paths are ranked at type level; inspect saved edge legs before interpreting a candidate motif.",
    ],
    "artifacts": artifacts,
}
manifest_path = OUT_DIR / "manifest.json"
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
print(f"saved {manifest_path}")
print(f"artifacts={len(artifacts)}")
