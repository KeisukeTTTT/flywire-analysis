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
# # FlyWire 視覚系の側抑制 — 拡張解析 (構造から計算へ)
#
# [`lateral_inhibition.ipynb`](lateral_inhibition.ipynb) (Q1–Q9) では、側抑制が **量的に広く (全シナプスの ~44%)、
# wide-field に空間を覆う (抑制 cell type は興奮の ~3.35×)、端では比率を保ったまま減衰する** という *構造・空間スケール* を示した。
#
# 本ノートは次の一段 — **「構造から計算を推定する」** — に進む。側抑制が連結体上で実際にどんな演算を実装しているかを問う:
#
# - **Q10 (A2)** T4/T5 の方向選択性は、興奮と抑制の **空間オフセット** から生じるとされる (Borst 系の運動視モデル)。
#   亜型 a/b/c/d は4基本方位に対応。入力の空間オフセット軸が **亜型で回転** するかを per-cell 円形統計で全脳検定する。
# - **Q11 (A1)** 側抑制の本質は **center–surround 拮抗**。各 columnar 標的が受ける興奮 E(Δcol) と
#   (wide-field 介在を経た) 抑制 I(Δcol) の空間カーネルを再構成し、**Mexican-hat** になるか・DoG から空間周波数
#   バンドパスを予測する。
# - **Q12 (B1)** 抑制の **トポロジー**: feedforward / feedback / lateral / 相互抑制・脱抑制 を census する。
# - **Q13 (B2)** 抑制の **層構造**: synapse 3D 座標から M 層深さを復元し、層ごとの抑制トーンを地図化する。
#
# **抑制性 nt の定義** Q1–Q9 と同じく `{GABA, GLUT, HIS}` を抑制性、`{ACH}` を興奮性とする。
# **caveat** nt_type の多くは ML 推定。connectome から見えるのは配線構造であり、機能の直接証明には activity/physiology が要る。

# %%
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
if (REPO_ROOT / "src").is_dir() is False:
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import scipy.stats as st

from src.config import DATA_DIR
from src.data import FlyWireDataManager

INHIBITORY_NT = {"GABA", "GLUT", "HIS"}
EXCITATORY_NT = {"ACH"}


def classify_nt(x):
    if x in INHIBITORY_NT:
        return "inh"
    if x in EXCITATORY_NT:
        return "exc"
    return "other"


m = FlyWireDataManager()
neurons = m.optic_lobe_neurons_df.copy()
conn = m.optic_lobe_connections_df.copy()
conn["sign"] = conn["nt_type"].map(classify_nt)
print(f"neurons={len(neurons):,}, edges={len(conn):,}")

# column_assignment (別 notebook で retinotopy を validate 済: Spearman rho ~ 0.94)
col_assign = pd.read_csv(
    Path(DATA_DIR) / "raw" / "flywire" / "csv" / "column_assignment.csv",
    dtype={"root_id": str, "column_id": str},
)
col_assign["p"] = col_assign["p"].astype(int)
col_assign["q"] = col_assign["q"].astype(int)
pqmap = col_assign.drop_duplicates("root_id").set_index("root_id")[["p", "q", "hemisphere", "type"]]
P, Q, HM = pqmap["p"].to_dict(), pqmap["q"].to_dict(), pqmap["hemisphere"].to_dict()

# 共通ヘルパ
HEXN = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def axial_to_cart(p, q):
    """axial hex (p,q) -> cartesian (x,y) [60度 basis]。"""
    return p + 0.5 * np.asarray(q), np.asarray(q) * (np.sqrt(3) / 2)


def hexd(dp, dq):
    """axial hex 距離 (cube 近似で非整数 centroid に対応)。"""
    return (np.abs(dp) + np.abs(dq) + np.abs(dp + dq)) / 2.0


def interior_cells(cell_type, hemisphere, min_nbrs=5):
    """column_assignment 入り cell_type のうち、hex 近傍が min_nbrs 個以上ある内部 cell。"""
    cells = pqmap[(pqmap["type"] == cell_type) & (pqmap["hemisphere"] == hemisphere)].copy()
    s = set(zip(cells["p"], cells["q"]))
    cells["n_neighbors"] = [sum((p + dp, q + dq) in s for dp, dq in HEXN)
                            for p, q in zip(cells["p"], cells["q"])]
    return cells[cells["n_neighbors"] >= min_nbrs]


# %% [markdown]
# ## Q10 (A2). T4/T5 方向選択性 — 入力の空間オフセット
#
# ショウジョウバエの ON (T4) / OFF (T5) 運動検出器は、各 medulla/lobula 個眼柱に **4 亜型 (a/b/c/d)** があり、
# それぞれ **4 基本方位** (前後・後前・上・下) の運動に選択的である。古典的モデル (Hassenstein–Reichardt /
# Barlow–Levick の派生; Borst 研) では、方向選択性は **興奮と抑制 (あるいは速い/遅い入力) の空間オフセット** から生じる。
#
# - **T4 (ON)**: 中心興奮 = **Mi1 / Tm3**、抑制 = **Mi9 (GluCl 経由で抑制性)** と **Mi4 (GABA)**、加えて全域 **CT1**。
#   モデルでは Mi9 と Mi4 は home column の **反対側** に配置され、その軸が運動の preferred–null 軸を決める。
# - **T5 (OFF)**: 抑制源 (CT1/Y1/TmY15) は大半が非 columnar (座標を持たない) ため、興奮 **Tm1/Tm2/Tm4/Tm9** の
#   空間オフセット (特に遅い Tm9) で方向軸を見る。
#
# **検定**: 各 cell について columnar 入力の **syn 重み付き重心 (Δp,Δq)** を home column 基準で求め、
# per-cell **dipole ベクトル** (T4: `Mi9 − Mi4`、T5: `Tm9 − Tm2`) の角度を取る。亜型ごとに **円形平均角・
# resultant R・Rayleigh 検定** を計算し、(1) 角度が集中するか、(2) 対向ペア (a/b, c/d) が **反平行** か、
# (3) 2 ペアの軸が **直交** に近いか、を **両半球** で確認する。
#
# *注: `mean(inh) − mean(exc)` のように Mi9 と Mi4 を平均すると反平行ゆえ相殺して消える。2 本の抑制 arm の
# dipole `Mi9 − Mi4` が正しいメトリクス。*

# %%
def per_cell_centroids(subtype, src_types, hemi):
    """subtype interior cell ごと・src_type ごとの syn 重み付き Δ(p,q) 重心。"""
    cells = interior_cells(subtype, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    cids = set(cells.index)
    inc = conn[conn["post_root_id"].isin(cids) & conn["pre_primary_type"].isin(src_types)].copy()
    inc["sp"] = inc["pre_root_id"].map(P)
    inc["sq"] = inc["pre_root_id"].map(Q)
    inc["sh"] = inc["pre_root_id"].map(HM)
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
    return g[["cdp", "cdq"]].unstack("pre_primary_type"), len(cells)


def circ_stats(ang):
    """ang: radians。(circular mean[deg], resultant R, Rayleigh p, n)。"""
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


def dipole_angles(subtype, src_types, hemi, arm_pos, arm_neg):
    """per-cell dipole = cart(arm_pos centroid) − cart(arm_neg centroid)。両 arm を持つ cell のみ。"""
    cen, ncell = per_cell_centroids(subtype, src_types, hemi)
    need = [("cdp", arm_pos), ("cdq", arm_pos), ("cdp", arm_neg), ("cdq", arm_neg)]
    if any(c not in cen.columns for c in need):
        return None, ncell, None
    sub = cen[need].dropna()
    px, py = axial_to_cart(sub[("cdp", arm_pos)].values, sub[("cdq", arm_pos)].values)
    nx, ny = axial_to_cart(sub[("cdp", arm_neg)].values, sub[("cdq", arm_neg)].values)
    return np.arctan2(py - ny, px - nx), ncell, np.hypot(px - nx, py - ny)


def angdiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


A2_CONFIG = {
    "T4 (ON)":  dict(subs=["T4a", "T4b", "T4c", "T4d"], srcs=["Mi1", "Tm3", "Mi9", "Mi4"], pos="Mi9", neg="Mi4"),
    "T5 (OFF)": dict(subs=["T5a", "T5b", "T5c", "T5d"], srcs=["Tm1", "Tm2", "Tm4", "Tm9"], pos="Tm9", neg="Tm2"),
}

a2_results = {}
a2_rows = []
for pathway, cfg in A2_CONFIG.items():
    for hemi in ["right", "left"]:
        sres = {}
        for sub in cfg["subs"]:
            ang, ncell, mag = dipole_angles(sub, cfg["srcs"], hemi, cfg["pos"], cfg["neg"])
            if ang is None or len(ang) < 3:
                continue
            mean, R, p, n = circ_stats(ang)
            sres[sub] = dict(mean=mean, R=R, p=p, n=n, mag=float(np.median(mag)))
            a2_rows.append(dict(pathway=pathway, hemi=hemi, subtype=sub, dipole=f"{cfg['pos']}-{cfg['neg']}",
                                n_cells=n, median_dipole_cols=round(float(np.median(mag)), 3),
                                circ_mean_deg=round(mean, 1), R=round(R, 3), rayleigh_p=p))
        a2_results[(pathway, hemi)] = sres

a2_table = pd.DataFrame(a2_rows)
print("Per-subtype dipole circular statistics (both hemispheres):")
display(a2_table)

print("\nRotation / orthogonality check:")
for (pathway, hemi), sres in a2_results.items():
    if len(sres) < 4:
        continue
    subs = A2_CONFIG[pathway]["subs"]
    mn = {s: sres[s]["mean"] for s in subs}
    d_ab = angdiff(mn[subs[0]], mn[subs[1]])
    d_cd = angdiff(mn[subs[2]], mn[subs[3]])
    ax_ab, ax_cd = mn[subs[0]] % 180, mn[subs[2]] % 180
    d_axes = angdiff(ax_ab * 2, ax_cd * 2) / 2
    print(f"  {pathway:9s} {hemi:5s}: {subs[0]}|{subs[1]} Δ={d_ab:3.0f}°, {subs[2]}|{subs[3]} Δ={d_cd:3.0f}° "
          f"(antiparallel≈180); axes Δ={d_axes:2.0f}° (orthogonal≈90)")

# %%
# 可視化 (1): per-cell dipole 角の極座標分布 + 円形平均矢印 (pathway × hemi)
sub_colors = {"a": "tab:red", "b": "tab:blue", "c": "tab:green", "d": "tab:orange"}
fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection="polar"))
for i, pathway in enumerate(A2_CONFIG):
    cfg = A2_CONFIG[pathway]
    for j, hemi in enumerate(["right", "left"]):
        ax = axes[i, j]
        sres = a2_results[(pathway, hemi)]
        for sub in cfg["subs"]:
            if sub not in sres:
                continue
            ang, _, _ = dipole_angles(sub, cfg["srcs"], hemi, cfg["pos"], cfg["neg"])
            c = sub_colors[sub[-1]]
            counts, edges = np.histogram(ang % (2 * np.pi), bins=24, range=(0, 2 * np.pi))
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(np.append(centers, centers[0]), np.append(counts, counts[0]), color=c, alpha=0.55, lw=1.3,
                    label=f"{sub} (R={sres[sub]['R']:.2f})")
            ax.annotate("", xy=(np.radians(sres[sub]["mean"]), max(counts) * (sres[sub]["R"] + 0.15)),
                        xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
        ax.set_title(f"{pathway}  {hemi}\ndipole = {cfg['pos']}−{cfg['neg']}", fontsize=10)
        ax.set_theta_zero_location("E")
        ax.set_yticklabels([])
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.18, 1.12))
plt.suptitle("Q10 (A2): per-cell input dipole angle rotates with T4/T5 subtype (preferred direction)\n"
             "lines = per-subtype angle histogram; arrows = circular mean × resultant R", y=1.01, fontsize=12)
plt.tight_layout()

# %%
# 可視化 (2): T4 right の入力 column 重心マップ (直感図)。興奮は中心付近、Mi9/Mi4 は反対側に分離。
fig, ax = plt.subplots(figsize=(8, 8))
cfg = A2_CONFIG["T4 (ON)"]
for sub in cfg["subs"]:
    cen, _ = per_cell_centroids(sub, cfg["srcs"], "right")
    c = sub_colors[sub[-1]]
    for src in cfg["srcs"]:
        if ("cdp", src) not in cen.columns:
            continue
        dp = np.nanmean(cen[("cdp", src)]); dq = np.nanmean(cen[("cdq", src)])
        cx, cy = axial_to_cart(dp, dq)
        mk = "o" if src in ("Mi1", "Tm3") else "^"
        ax.scatter([cx], [cy], color=c, marker=mk, s=110, edgecolors="black", linewidths=0.4, zorder=3)
        ax.annotate(src, (cx, cy), fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")
    # Mi9-Mi4 dipole arrow
    m9 = axial_to_cart(np.nanmean(cen[("cdp", "Mi9")]), np.nanmean(cen[("cdq", "Mi9")]))
    m4 = axial_to_cart(np.nanmean(cen[("cdp", "Mi4")]), np.nanmean(cen[("cdq", "Mi4")]))
    ax.annotate("", xy=m9, xytext=m4, arrowprops=dict(arrowstyle="-|>", color=c, lw=2, alpha=0.85))
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.plot(0, 0, "k+", markersize=14, markeredgewidth=2)
ax.set(aspect="equal", title="T4 (right) input column centroids relative to home\n"
       "circle=excitation (Mi1/Tm3, near center), triangle=inhibition (Mi9/Mi4, opposite sides)\n"
       "arrow = Mi9−Mi4 dipole; color = subtype")
ax.set_xlabel("Δ horizontal (hex cart)"); ax.set_ylabel("Δ vertical (hex cart)")
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %% [markdown]
# **結果 (A2)**: 全 8 亜型 × 両半球で dipole 角は強く集中する (Rayleigh p ≈ 1e-80 〜 1e-256)。対向ペアは
# **反平行** (T4a/b Δ≈175°, T4c/d Δ≈170°)、水平ペアと垂直ペアの軸は **ほぼ直交** (Δ≈78–84°; 完全な 90° で
# ないのは hex 格子上の 4 方位が直交でないため、むしろ妥当)。**両半球で角度が数度内で一致** し、さらに
# **T4 と T5 で同字亜型 (例 T4a 139.5° ≈ T5a 138.7°) の軸が一致** する (同方位選択性同士の内部整合)。
# 連結体だけから、方向選択性の **空間オフセット仮説** が全脳スケール・複数内部対照付きで支持される。
#
# *残課題*: hex 軸 → 視空間軸 (D/V, A/P) のマッピング (surface fitting cache) で **絶対方位** と既知 PD の
# 整合を取る。また Mi9 (GluCl) を抑制と見なす点・syn count を機能強度の代理とする点は connectome の限界。

# %% [markdown]
# ## Q11 (A1). connectome center–surround 受容野 (Mexican-hat)
#
# 側抑制の本質は **center–surround 拮抗**: 中心の興奮と、それを取り囲む周辺からの抑制。各 columnar 標的 T が
# 受ける入力を **Δcolumn (home からの hex 距離) の関数** として再構成し、
#
# - **興奮 E(Δ)**: 直接 columnar 興奮入力の Δ 分布 (feedforward, 中心鋭い)
# - **抑制 I(Δ)**: **二シナプス性 surround**。抑制 partner J が *自分の入力を pool する column 群* を T home 基準で
#   集計したもの (= 抑制が "見ている" 視空間 = surround)。
#
# **なぜ二シナプス性が必要か**: 視覚葉の抑制源の多くは座標を持たない wide-field 介在 (Pm/Dm/CT1)。直接
# (単シナプス) の Δ では home 集中に見えてしまう (例: L1→Mi1 は home column)。surround は J の入力 footprint を
# 経て初めて現れる。最後に E と I を **DoG (difference of Gaussians)** とみなし、予測される **空間周波数バンドパス**
# (エッジ強調フィルタ) を示す。

# %%
def direct_kernel(T, sign, hemi="right", maxd=8):
    cells = interior_cells(T, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    cids = set(cells.index)
    inc = conn[conn["post_root_id"].isin(cids) & (conn["sign"] == sign)].copy()
    inc["sp"] = inc["pre_root_id"].map(P); inc["sq"] = inc["pre_root_id"].map(Q)
    inc["sh"] = inc["pre_root_id"].map(HM)
    inc = inc.dropna(subset=["sp", "sq"]); inc = inc[inc["sh"] == hemi]
    inc["d"] = np.rint(hexd(inc["sp"] - inc["post_root_id"].map(hp),
                            inc["sq"] - inc["post_root_id"].map(hq))).astype(int)
    return inc.groupby("d")["syn_count"].sum()


def disyn_inh_kernel(T, hemi="right", top_inh_types=12):
    """二シナプス性抑制 surround: 抑制 partner J が pool する columnar exc 入力を T home 基準で集計。"""
    cells = interior_cells(T, hemi)
    hp, hq = cells["p"].to_dict(), cells["q"].to_dict()
    cids = set(cells.index)
    inh_to_t = conn[conn["post_root_id"].isin(cids) & (conn["sign"] == "inh")].copy()
    top_types = (inh_to_t.groupby("pre_primary_type")["syn_count"].sum()
                 .sort_values(ascending=False).head(top_inh_types).index)
    inh_to_t = inh_to_t[inh_to_t["pre_primary_type"].isin(top_types)][["pre_root_id", "post_root_id", "syn_count"]]
    inh_to_t = inh_to_t.rename(columns={"syn_count": "w1"})
    J = set(inh_to_t["pre_root_id"])
    jin = conn[conn["post_root_id"].isin(J) & (conn["sign"] == "exc")].copy()
    jin["sp"] = jin["pre_root_id"].map(P); jin["sq"] = jin["pre_root_id"].map(Q)
    jin["sh"] = jin["pre_root_id"].map(HM)
    jin = jin.dropna(subset=["sp", "sq"]); jin = jin[jin["sh"] == hemi]
    jin["w2n"] = jin["syn_count"] / jin.groupby("post_root_id")["syn_count"].transform("sum")
    jin = jin.rename(columns={"post_root_id": "j"})[["j", "sp", "sq", "w2n"]]
    mg = inh_to_t.merge(jin, left_on="pre_root_id", right_on="j", how="inner")
    mg["d"] = np.rint(hexd(mg["sp"] - mg["post_root_id"].map(hp),
                           mg["sq"] - mg["post_root_id"].map(hq))).astype(int)
    mg["w"] = mg["w1"] * mg["w2n"]
    return mg.groupby("d")["w"].sum()


def rms_radius(kern, maxd=8):
    """カーネルの moment 幅 sigma = sqrt(Σ f(d) d² / Σ f(d))。"""
    s = kern.reindex(range(0, maxd + 1), fill_value=0).astype(float)
    s = s / s.sum() if s.sum() > 0 else s
    d = np.arange(0, maxd + 1)
    return float(np.sqrt((s.values * d ** 2).sum())), s


targets = ["Mi1", "Tm9", "L2", "T4a"]
maxd = 8
a1_rows = []
kernels = {}
for T in targets:
    E = direct_kernel(T, "exc"); Idis = disyn_inh_kernel(T)
    sig_c, e = rms_radius(E, maxd); sig_s, i = rms_radius(Idis, maxd)
    kernels[T] = (e, i, sig_c, sig_s)
    a1_rows.append(dict(target=T, exc_rms_cols=round(sig_c, 2), inh_surround_rms_cols=round(sig_s, 2),
                        surround_center_ratio=round(sig_s / max(sig_c, 1e-6), 2),
                        exc_frac_at_home=round(float(e.iloc[0]), 2), inh_frac_at_home=round(float(i.iloc[0]), 2)))
a1_table = pd.DataFrame(a1_rows)
print("Center (exc) vs surround (disynaptic inh) spatial widths:")
display(a1_table)

# %%
# 可視化: (上) radial profile per target, (下) Mi1 の DoG と予測空間周波数応答
fig, axes = plt.subplots(2, len(targets), figsize=(4.4 * len(targets), 8))
rr = np.arange(0, maxd + 1)
for k, T in enumerate(targets):
    e, i, sig_c, sig_s = kernels[T]
    ax = axes[0, k]
    ax.plot(rr, e.values, "-o", color="tab:red", label=f"exc E(Δ) (σ={sig_c:.2f})")
    ax.plot(rr, i.values, "--^", color="tab:blue", label=f"inh surround I(Δ) (σ={sig_s:.2f})")
    ax.set(xlabel="Δcolumn (hex)", ylabel="fraction of input", title=T)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # DoG: 2D 等方ガウス近似。FT(Gaussian σ) ∝ σ² exp(−2π²σ² f²)。surround>center なら bandpass。
    ax = axes[1, k]
    f = np.linspace(0, 0.5, 200)  # cycles / column
    Ac = float(e.iloc[0]) if e.iloc[0] > 0 else 1.0
    As = float(i.iloc[0]) if i.iloc[0] > 0 else 1.0
    sc = max(sig_c, 0.3); ss = max(sig_s, sig_c + 0.1)
    ft_c = Ac * sc ** 2 * np.exp(-2 * np.pi ** 2 * sc ** 2 * f ** 2)
    ft_s = As * ss ** 2 * np.exp(-2 * np.pi ** 2 * ss ** 2 * f ** 2)
    resp = ft_c - ft_s
    resp = resp / np.abs(resp).max()
    ax.plot(f, resp, color="tab:green", lw=2)
    ax.axhline(0, color="gray", lw=0.6)
    peak_f = f[np.argmax(resp)]
    ax.axvline(peak_f, color="tab:red", ls=":", label=f"peak ≈ {peak_f:.2f} cyc/col")
    ax.set(xlabel="spatial frequency (cycles/column)", ylabel="predicted response (norm.)",
           title=f"{T}: DoG bandpass prediction")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle("Q11 (A1): excitation is center-weighted, disynaptic inhibition forms a broad surround\n"
             "(top) received radial profiles; (bottom) Difference-of-Gaussians → bandpass spatial-frequency tuning",
             y=1.02, fontsize=12)
plt.tight_layout()

# %% [markdown]
# **結果 (A1)**: 興奮は鋭く中心集中 (RMS 半径 ~0.3–0.7 柱, home に 7–8 割)、二シナプス性抑制 surround は
# 明確に広い (~1–2 柱)。`surround / center` 半径比は 2–4 倍で **center–surround / Mexican-hat 構造**。
# DoG とみなすと **バンドパス** (低周波の一様照明を抑え、エッジ・コントラストを強調) の空間周波数特性が
# 予測される。これは Q4 の「抑制 cell の *出力* が広い」を、標的が *受ける* 空間演算として裏返したもの。
#
# *caveat*: 単シナプス抑制だけでは home 集中に見える (L1→Mi1 等の home-column 抑制ラベル) — surround は
# disynaptic 再構成でのみ現れる。DoG は等方ガウス近似で、層 (Q13) の違いは未考慮。L1 の抑制ラベルは ML 由来
# (符号 vs 伝達物質の乖離; ロバスト性は別途検証要)。

# %% [markdown]
# ## Q12 (B1). 抑制マイクロ回路モチーフの census
#
# Q1–Q11 は抑制の「量・空間」を見た。ここは **トポロジー (回路の型)**:
#
# - **脱抑制 (disinhibition)**: 抑制シナプスが *抑制性 cell* を標的にする (I→I)。抑制を抑制する = 興奮性の効果。
# - **相互抑制 (reciprocal)**: A→B と B→A が両方 inh。winner-take-all や対立計算の基盤。
# - **feedforward 抑制 (FFI)**: 共通の興奮駆動 X が、標的 T と T の抑制源 Z の両方を駆動する (X→T, X→Z→T)。
#   時間的シャープ化・gain control の典型 motif。
#
# 各 cell type を出力 sign で inh/exc dominant に分類し (Q2 と同じ)、上記を定量する。

# %%
type_io = conn.groupby(["pre_primary_type", "sign"])["syn_count"].sum().unstack(fill_value=0)
for c in ("inh", "exc", "other"):
    if c not in type_io.columns:
        type_io[c] = 0
type_io["total"] = type_io[["inh", "exc", "other"]].sum(axis=1)
type_io["dominant"] = type_io[["inh", "exc", "other"]].idxmax(axis=1)
dom = type_io["dominant"]

# (1) 脱抑制: 抑制シナプスの post dominant sign 内訳
inh_edges = conn[conn["sign"] == "inh"].copy()
inh_edges["post_dom"] = inh_edges["post_primary_type"].map(dom)
by_post = inh_edges.groupby("post_dom")["syn_count"].sum()
tot_inh = by_post.sum()
print("=== (1) Disinhibition: where inhibitory synapses land (post classified by its own output sign) ===")
for k in ["exc", "inh", "other"]:
    print(f"  inh -> {k:5s}-dominant post : {int(by_post.get(k,0)):>12,} ({by_post.get(k,0)/tot_inh:.1%})")
print(f"  => {by_post.get('inh',0)/tot_inh:.1%} of ALL inhibition is inhibition-of-inhibition (disinhibitory substrate)")

# (2) 相互抑制ペア
te = conn[conn["sign"] == "inh"].groupby(["pre_primary_type", "post_primary_type"])["syn_count"].sum()
te_map = {(a, b): s for (a, b), s in te.items()}
seen, recip = set(), []
for (a, b), s_ab in te_map.items():
    if a == b or (b, a) in seen:
        continue
    seen.add((a, b))
    s_ba = te_map.get((b, a), 0)
    if s_ab >= 500 and s_ba >= 500:
        recip.append((a, b, int(s_ab), int(s_ba), int(min(s_ab, s_ba))))
recip_df = pd.DataFrame(recip, columns=["A", "B", "A->B", "B->A", "min_syn"]).sort_values("min_syn", ascending=False)
print(f"\n=== (2) Reciprocal-inhibition type-pairs (both directions >=500 inh syn): {len(recip_df)} ===")
display(recip_df.head(15))

# (3) FFI motif (type-level)
exc_te = conn[conn["sign"] == "exc"].groupby(["pre_primary_type", "post_primary_type"])["syn_count"].sum()


def ffi_for(target, topn=6):
    inc = conn[conn["post_primary_type"] == target]
    exc_drv = inc[inc["sign"] == "exc"].groupby("pre_primary_type")["syn_count"].sum().sort_values(ascending=False).head(topn)
    inh_src = inc[inc["sign"] == "inh"].groupby("pre_primary_type")["syn_count"].sum().sort_values(ascending=False).head(topn)
    rows = []
    for x in exc_drv.index:
        for z in inh_src.index:
            xz = exc_te.get((x, z), 0)
            if xz >= 200:
                rows.append(dict(target=target, driver_X=x, inhibitor_Z=z,
                                 X_to_Z_exc=int(xz), Z_to_T_inh=int(inh_src[z])))
    return pd.DataFrame(rows).sort_values("X_to_Z_exc", ascending=False)


print("\n=== (3) Feedforward-inhibition motifs (common driver X -> inhibitor Z -> target T) ===")
ffi_all = pd.concat([ffi_for(t) for t in ["Mi1", "T4a", "Tm9"]], ignore_index=True)
display(ffi_all.head(20))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
d = recip_df.head(15).iloc[::-1]
axes[0].barh(range(len(d)), d["min_syn"], color="tab:purple", edgecolor="white")
axes[0].set_yticks(range(len(d)))
axes[0].set_yticklabels([f"{a}<->{b}" for a, b in zip(d["A"], d["B"])], fontsize=8)
axes[0].set(xlabel="min(A->B, B->A) inh syn", title="(2) Reciprocal inhibition: top type-pairs")
axes[0].grid(True, alpha=0.3, axis="x")
vals = [by_post.get("exc", 0), by_post.get("inh", 0), by_post.get("other", 0)]
axes[1].bar(["onto exc-dom\n(classic FF/FB inh)", "onto inh-dom\n(DISINHIBITION)", "onto other"], vals,
            color=["tab:red", "tab:blue", "gray"], edgecolor="white")
for x, v in enumerate(vals):
    axes[1].text(x, v, f"{v/tot_inh:.1%}", ha="center", va="bottom", fontweight="bold")
axes[1].set(ylabel="total inhibitory synapses", title="(1) Where does inhibition land?")
axes[1].grid(True, alpha=0.3, axis="y")
plt.suptitle("Q12 (B1): inhibitory microcircuit motif census", y=1.02, fontsize=12)
plt.tight_layout()

# %% [markdown]
# **結果 (B1)**: 全抑制シナプスの **41% が抑制性 cell を標的** (脱抑制基盤) — Q1 の「44% が抑制」は
# 「うち約 4 割は抑制を抑制している」と再解釈できる。相互抑制は **326 type-pair**、筆頭が運動回路の
# **Mi4↔Mi9**、ほか Dm3p/q/v の三つ巴、Pm03/05/08/09 網。FFI では **Mi1 が T4a を駆動しつつ T4a の抑制源
# Mi9/Mi4 をも駆動** (運動回路の feedforward 抑制) が定量化される。
#
# *caveat*: C2/C3/L1・Lai/R1-6 等の一部ペアは **L1 が ML で抑制ラベル**な点に由来する見かけ → 確率的 nt
# (gaba/glut/ach_avg + nt_type_score) でのロバスト性確認が望ましい。

# %% [markdown]
# ## Q13 (B2). M 層 (深さ) 解像の抑制アトラス
#
# Q1–Q12 は lateral (柱方向) を見たが、medulla は **M1–M10 の層構造** を持ち、層ごとに異なる計算が走る。
# `synapse_coordinates.csv` の 3D 座標から深さを復元する:
#
# 1. 右半球 **Mi1** の synapse 重心群を PCA し、**最小分散軸 = 局所深さ法線** とする
# 2. Mi1 synapse 深さ分布の 1–99%tile を **medulla 厚 [0,1]** に較正 (Mi1 = 深さ定規)
# 3. curated な抑制型 (遠位 Dm / 近位 Pm / 全層 CT1 / 中層 Mi4,Mi9) と興奮 reference の深さ分布を比較
#
# *caveat*: CT1/Tm9 は medulla 外 (lobula) にも arbor を持つ多神経網細胞で [0,1] を外れる → medulla 窓
# `[-0.25, 1.25]` でクリップし除外率を併記。単一の global 法線は曲率を無視する近似 (surface fitting cache で
# 曲率補正するのが上位版)。**所要時間: synapse_coordinates (3,400 万行) ロードで数十秒。**

# %%
_neur = neurons.drop_duplicates("root_id").set_index("root_id")
_type, _side = _neur["primary_type"], _neur["side"]

CURATED = {"Dm1": "inh", "Dm4": "inh", "Dm9": "inh", "Dm12": "inh",
           "Pm04": "inh", "Pm08": "inh", "Pm09": "inh", "CT1": "inh", "Mi4": "inh", "Mi9": "inh",
           "Mi1": "exc-ref", "L5": "exc-ref", "Tm3": "exc-ref", "Tm9": "exc-ref"}
right_curated = set(_neur.index[(_side == "right") & (_type.isin(CURATED))])
mi1_right = set(_neur.index[(_side == "right") & (_type == "Mi1")])

syn_path = Path(DATA_DIR) / "raw" / "flywire" / "csv" / "synapse_coordinates.csv"
t0 = time.perf_counter()
sc = pd.read_csv(syn_path, dtype={"pre_root_id": str}, usecols=["pre_root_id", "x", "y", "z"])
sc["pre_root_id"] = sc["pre_root_id"].ffill()
sc = sc.dropna(subset=["pre_root_id"])
sc = sc[sc["pre_root_id"].isin(right_curated | mi1_right)].copy()
sc["ptype"] = sc["pre_root_id"].map(_type)
print(f"loaded + filtered to {len(sc):,} right-hemi curated synapses in {time.perf_counter()-t0:.1f}s")

# 深さ法線 = Mi1 重心群 PCA 最小分散軸
mi1_sc = sc[sc["pre_root_id"].isin(mi1_right)]
cnt = mi1_sc.groupby("pre_root_id").size()
cents = mi1_sc.groupby("pre_root_id")[["x", "y", "z"]].mean()[cnt >= 20]
Xc = cents.values - cents.values.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
normal, c0 = Vt[-1], cents.values.mean(axis=0)
sc["depth"] = (sc[["x", "y", "z"]].values - c0) @ normal
mi1_depth = sc.loc[sc["ptype"] == "Mi1", "depth"]
lo, hi = np.percentile(mi1_depth, [1, 99])
lo, hi = min(lo, hi), max(lo, hi)
sc["rel_depth"] = (sc["depth"] - lo) / (hi - lo)
# Dm(遠位)が小さい側に来るよう向き調整
if sc.loc[sc["ptype"].isin(["Dm1", "Dm4", "Dm12"]), "rel_depth"].median() > \
   sc.loc[sc["ptype"].isin(["Pm04", "Pm08", "Pm09"]), "rel_depth"].median():
    sc["rel_depth"] = 1 - sc["rel_depth"]

# medulla 窓でクリップ (多神経網細胞の lobula arbor 等を除外)
WIN = (-0.25, 1.25)
order = [t for t in ["L5", "Mi1", "Tm3", "Dm1", "Dm9", "Dm12", "Dm4", "Mi9", "Mi4", "Tm9", "Pm08", "Pm09", "Pm04", "CT1"]
         if (sc["ptype"] == t).any()]
b2_rows = []
for t in order:
    d_all = sc.loc[sc["ptype"] == t, "rel_depth"]
    d = d_all[(d_all >= WIN[0]) & (d_all <= WIN[1])]
    q1, q2, q3 = np.percentile(d, [25, 50, 75]) if len(d) else (np.nan,) * 3
    b2_rows.append(dict(type=t, sign=CURATED[t], n_syn=len(d_all),
                        frac_in_medulla_window=round(len(d) / len(d_all), 2),
                        median_depth=round(q2, 2), q1=round(q1, 2), q3=round(q3, 2)))
b2_table = pd.DataFrame(b2_rows)
print("\nPer-type synapse depth (0=distal/M1 .. 1=proximal/M10), clipped to medulla window:")
display(b2_table)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = {"inh": "tab:blue", "exc-ref": "tab:red"}
# medulla 窓内に十分な synapse がある型のみ violin に (CT1 は global ゆえ 0% -> 除外し表に注記)
order_plot = [t for t in order if sc.loc[(sc["ptype"] == t) & sc["rel_depth"].between(*WIN)].shape[0] >= 50]
data = [sc.loc[(sc["ptype"] == t) & sc["rel_depth"].between(*WIN), "rel_depth"].values for t in order_plot]
ax = axes[0]
vp = ax.violinplot(data, positions=range(len(order_plot)), vert=False, showmedians=True, widths=0.85)
for k, body in enumerate(vp["bodies"]):
    body.set_facecolor(colors[CURATED[order_plot[k]]]); body.set_alpha(0.45)
ax.set_yticks(range(len(order_plot)))
ax.set_yticklabels([f"{t} ({CURATED[t][:3]})" for t in order_plot], fontsize=9)
ax.set(xlabel="relative medulla depth (0=distal/M1 .. 1=proximal/M10)", xlim=(WIN[0], WIN[1]),
       title="(A) Synapse depth by cell type (red=exc ref, blue=inh)")
ax.axvline(0, color="gray", ls=":", alpha=0.5); ax.axvline(1, color="gray", ls=":", alpha=0.5)
ax.grid(True, alpha=0.3, axis="x")

ax = axes[1]
inh_d = sc.loc[(sc["ptype"].map(CURATED) == "inh") & sc["rel_depth"].between(*WIN), "rel_depth"]
exc_d = sc.loc[(sc["ptype"].map(CURATED) == "exc-ref") & sc["rel_depth"].between(*WIN), "rel_depth"]
bins = np.linspace(WIN[0], WIN[1], 31)
ax.hist(exc_d, bins=bins, density=True, alpha=0.5, color="tab:red", label=f"exc ref (n={len(exc_d):,})")
ax.hist(inh_d, bins=bins, density=True, alpha=0.5, color="tab:blue", label=f"inh (n={len(inh_d):,})")
ax.set(xlabel="relative medulla depth", ylabel="density",
       title="(B) inh vs exc-reference synapse depth\n(inh shows an extra distal mode)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle("Q13 (B2): layer-resolved inhibition in the medulla (Mi1-PCA depth ruler)", y=1.02, fontsize=12)
plt.tight_layout()

# %% [markdown]
# **結果 (B2)**: Mi1-PCA 深さ定規で **Dm 系が遠位 (median 0.24–0.37)・Pm 系が近位 (0.64–0.69)** に
# textbook 通り分離し、Mi4/Mi9 は中層 (0.48–0.54)。抑制シナプス全体は興奮 reference に比べ **遠位に追加モード**
# を持つ (panel B)。層ごとに抑制の担い手と量が異なることが connectome から復元できる。
#
# *残課題*: 単一 global 法線は曲率を無視 → surface fitting cache で柱ごとに depth を補正すると M1–M10 の
# 境界をより正確に引ける。CT1/Tm9 等は medulla 窓クリップで除外率を併記済 (多神経網 arbor)。

# %% [markdown]
# ## まとめ — 構造から計算へ
#
# | 解析 | 連結体から読めた「計算」 |
# |---|---|
# | **A2** T4/T5 入力オフセット | 抑制 (Mi9/Mi4) の空間 dipole が **亜型で回転** = 運動の方向選択性。両半球一致・T4/T5 整合。 |
# | **A1** center–surround | 興奮は中心、二シナプス性抑制は広い surround → **Mexican-hat / 空間バンドパス** (エッジ強調)。 |
# | **B1** モチーフ census | 抑制の **41% は脱抑制**、相互抑制 (Mi4↔Mi9) と FFI (Mi1→T4 & Mi1→Mi9/Mi4) が定量化。 |
# | **B2** M 層アトラス | 抑制は **層ごとに役割分担** (Dm 遠位 / Pm 近位 / Mi 中層)、全体に遠位寄りの追加モード。 |
#
# Q1–Q9 が示した「抑制は量的に広く wide-field」に対し、本拡張は **その配線が方向選択性・空間バンドパス・
# 脱抑制・層分業という具体的な演算を実装している** ことを示す。いずれも connectome 構造からの推定であり、
# 機能の確定には activity/behavior が要る点は引き続き限界。

# %%
a2_best = a2_table[(a2_table.pathway == "T4 (ON)") & (a2_table.hemi == "right")]
summary = {
    "A2_T4_subtypes_significant":      f"{int((a2_table[a2_table.pathway=='T4 (ON)'].rayleigh_p < 1e-10).sum())}/8 (both hemi)",
    "A2_T4a_vs_T4b_antiparallel_deg":  round(angdiff(a2_results[('T4 (ON)','right')]['T4a']['mean'],
                                                     a2_results[('T4 (ON)','right')]['T4b']['mean']), 0),
    "A1_surround_center_ratio_median": round(float(a1_table["surround_center_ratio"].median()), 2),
    "B1_disinhibition_fraction":       f"{by_post.get('inh',0)/tot_inh:.1%}",
    "B1_reciprocal_inh_pairs":         len(recip_df),
    "B2_Dm_median_depth":              round(float(b2_table.set_index('type').loc[['Dm1','Dm9','Dm12','Dm4'],'median_depth'].median()), 2),
    "B2_Pm_median_depth":              round(float(b2_table.set_index('type').loc[['Pm04','Pm08','Pm09'],'median_depth'].median()), 2),
}
for k, v in summary.items():
    print(f"  {k:36s} = {v}")
