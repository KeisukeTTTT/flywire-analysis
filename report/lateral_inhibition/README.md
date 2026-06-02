# lateral_inhibition/

FlyWire 連結体データを用いたショウジョウバエ視覚系の側抑制解析結果をまとめた LaTeX レポート。

## ファイル構成

```
report/lateral_inhibition/
  main.tex              # 本体 (日本語、約 8-10 ページ)
  generate_figures.py   # 図 8 枚を再生成する Python スクリプト
  figures/              # 図 (PNG)
    fig1_nt_distribution.png
    fig2_inh_fraction_dist.png
    fig3_column_spread_inh_vs_exc.png
    fig4_hex_footprint.png
    fig5_dm8_input_footprint.png
    fig6_edge_effect_mi1.png
    fig7_inh_family_survey.png
    fig8_multi_type_edge.png
    fig9_lateral_inh_vs_bottomup.png
    fig10_t4t5_offset.png        # Q10 (A2) T4/T5 入力オフセット
    fig11_center_surround.png    # Q11 (A1) center-surround / DoG
    fig12_motif_census.png       # Q12 (B1) 抑制モチーフ census
    fig13_mlayer_atlas.png       # Q13 (B2) M 層深さアトラス
  README.md             # このファイル
```

## Overleaf でコンパイル

1. Overleaf にログインし、新規プロジェクト → **Upload Project** で、この `report/lateral_inhibition/` ディレクトリを zip にしてアップロード
2. プロジェクト画面で **Menu → Compiler** を **LuaLaTeX** に変更
3. **Recompile** ボタンで PDF が生成される

`main.tex` の先頭に `% !TEX program = lualatex` のマジックコメントを入れてあるので、設定が有効になっていれば自動で LuaLaTeX が選択される。

### 日本語フォントについて

`ltjsarticle` (LuaTeX-ja) を使っているので、Overleaf の標準環境にある Noto Sans / Noto Serif CJK が自動で適用される。フォントを明示指定したい場合は `\usepackage{luatexja-fontspec}` の後に `\setmainjfont{...}` 等を追加。

## ローカルで再生成

### 図の再生成

```bash
# プロジェクトルートで
uv run python report/lateral_inhibition/generate_figures.py
```

所要時間 約 2--3 分 (FlyWire データロード 35 秒 + 図生成 fig1--13。fig13 は synapse_coordinates.csv のロードで +数十秒)。
`report/lateral_inhibition/figures/*.png` (fig1--13) が上書きされる。

### LaTeX のコンパイル

ローカルに LuaTeX-ja 環境がある場合:

```bash
cd report/lateral_inhibition
lualatex main.tex
lualatex main.tex   # 参照を解決するため 2 回実行
```

Docker で TeX Live が入っている場合:

```bash
docker run --rm -v $(pwd)/report/lateral_inhibition:/work -w /work texlive/texlive lualatex main.tex
```

## 内容概要

9 つの問い (Q1-Q9) に分けた連結体解析:

- **Q1**: 全シナプスの I/E バランス (約 44% が抑制性)
- **Q2**: 抑制性 cell type の広がり (主要 344 種中 206 種が抑制 dominant)
- **Q3**: within-type 抑制の頻度 (41% — 支配的ではない)
- **Q4**: Δcolumn 単位の lateral spread (抑制は興奮の約 3.35 倍広い)
- **Q5**: 古典回路 (Lai, Dm, Pm 系) の検証
- **Q6**: Dm8 (UV color circuit) の input-side metric (R7 入力 median 7 columns, max 14)
- **Q7**: 端 column の Mi1 (絶対量は減るが E/I balance 保たれる)
- **Q8**: 抑制性インターニューロン全 205 種の family サーベイ (Sm 系が最 wide-field)
- **Q9**: 多 cell type で Q7 パターンが普遍的か (Yes、ただし projection neuron T4/T5 で端効果弱い)

### 拡張解析 (構造から計算へ; `notebook/lateral_inhibition_extended.ipynb`)

- **Q10 (A2)**: T4/T5 の方向選択性 = 入力の空間オフセット (Mi9−Mi4 dipole が亜型で回転、両半球一致、Rayleigh p≤1e-80)
- **Q11 (A1)**: center-surround 受容野 (興奮は中心、二シナプス性抑制は広い surround、surround/center 比 2.5、DoG バンドパス)
- **Q12 (B1)**: 抑制モチーフ census (全抑制の 41% が脱抑制、相互抑制 326 ペア、Mi4↔Mi9、Mi1→T4 の FFI)
- **Q13 (B2)**: M 層深さアトラス (Dm 遠位 / Pm 近位 / Mi 中層、抑制は遠位に追加モード)

詳細は `main.tex` の本文を参照。
