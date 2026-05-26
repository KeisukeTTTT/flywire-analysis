# report/

FlyWire 解析レポート置き場。今後は分析テーマごとにサブディレクトリを分け、本文・図・図生成スクリプトを同じ場所に置く。

## 構成

```
report/
  README.md
  lateral_inhibition/
    README.md
    main.tex
    generate_figures.py
    figures/
```

## レポート一覧

| analysis | 内容 | 主な入力 |
| --- | --- | --- |
| `lateral_inhibition/` | 視覚系の側抑制解析。I/E balance、Δcolumn spread、Dm8、edge effect を扱う。 | `notebook/lateral_inhibition.ipynb` |

## 新しい分析を追加する時の標準形

新しい分析は `report/<analysis_slug>/` に以下を置く。

```
report/<analysis_slug>/
  README.md            # 実行方法、図、内容概要
  main.tex             # LaTeX 本文
  generate_figures.py  # 図の再生成スクリプト
  figures/             # 生成図
```

ルートから実行できるように、Python は次の形で repo root を解決する。

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
```

## 実行例

```bash
# 図の再生成
uv run python report/lateral_inhibition/generate_figures.py

# LaTeX のコンパイル
cd report/lateral_inhibition
lualatex main.tex
lualatex main.tex
```

Python ファイルの実行は、プロジェクトルールに従い必ず `uv run` を使う。
