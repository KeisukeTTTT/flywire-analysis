# report/

FlyWire 解析レポート置き場。分析テーマごとにサブディレクトリを分け、**本文 (LaTeX) と人手で選んだ
追跡対象の図**を置く。図の生成スクリプトは `scripts/<analysis_slug>/`、大量の探索的図は `outputs/`(非追跡)
に置く(本文が参照する厳選図のみ `report/.../figures/` に追跡)。

## 構成

```
report/
  README.md
  lateral_inhibition/
    README.md
    main.tex
    figures/             # main.tex が参照する厳選図(追跡)
```

## レポート一覧

| analysis | 内容 | 主な入力 |
| --- | --- | --- |
| `lateral_inhibition/` | 視覚系の側抑制解析。I/E balance、Δcolumn spread、Dm8、edge effect を扱う。 | `src/lateral/` 基盤 + `notebooks/lateral_inhibition_rigorous.ipynb` |

## 新しい分析を追加する時の標準形

```
report/<analysis_slug>/      # 本文と厳選図
  README.md  main.tex  figures/
scripts/<analysis_slug>/     # 図生成スクリプト(uv run python)
  generate_*.py
outputs/<analysis_slug>/     # スクリプト生成の大量図(非追跡)
```

スクリプトは次の形で repo root を解決する(配置場所に依存しない)。

```python
REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "src").is_dir() and (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO_ROOT))
```

## 実行例

```bash
# 本文用の厳選図を再生成 (report/.../figures/ に出力)
uv run python scripts/lateral_inhibition/generate_figures.py

# LaTeX のコンパイル
cd report/lateral_inhibition
lualatex main.tex
lualatex main.tex
```

Python ファイルの実行は、プロジェクトルールに従い必ず `uv run` を使う。
