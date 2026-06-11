# notebook/

Notebook は git では直接追跡しない。正本は同名の Python ファイル (`*.py`) で、Jupytext の percent-cell (`# %%`) 形式で管理する。

ルートの `.jupytext.toml` で `ipynb,py:percent` のペアリングを設定している。

## 追跡対象

```
notebook/
  column_assignment_validation.py
  edge_compensation_t4t5_lateral.py
  flywire_eda.py
  lateral_inhibition.py
```

`*.ipynb` はローカルで生成する作業ファイルで、`.gitignore` により追跡対象外。

## 同期

プロジェクトルートで実行する。

```bash
uv run jupytext --sync notebook/*.py
```

個別に同期する場合:

```bash
uv run jupytext --sync notebook/lateral_inhibition.py
```

`--sync` は `.py` と `.ipynb` のうち更新時刻が新しい側を正として、もう一方を更新する。生成された notebook は git に add しない。

## 方向を明示する場合

Python から Notebook を生成する。

```bash
uv run jupytext --to ipynb notebook/lateral_inhibition.py -o notebook/lateral_inhibition.ipynb
```

Notebook で編集した内容を Python に戻す。

```bash
uv run jupytext --to py:percent notebook/lateral_inhibition.ipynb -o notebook/lateral_inhibition.py
```

その後、差分を確認して `*.py` だけをコミットする。

## 注意

- `--to ipynb` で生成される notebook は出力セルなしの状態になる。
- 実行結果を保存したい場合はローカル notebook 上で実行する。ただし `*.ipynb` はコミットしない。
- Python ファイルの実行や同期は、プロジェクトルールに従い必ず `uv run` を使う。
