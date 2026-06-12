# notebooks/

探索的な解析ノートブックを **`.ipynb` 形式のまま** git で追跡する。実行で生じる出力(figures・
`execution_count`・揮発メタデータ)は **コミット時に nbstripout が自動で除去**するため、ノートを実行・
保存しても追跡対象の差分は汚れない。

## 正本は .ipynb

```
notebooks/
  column_assignment_validation.ipynb
  edge_compensation_t4t5_lateral.ipynb
  flywire_eda.ipynb
  lateral_inhibition_rigorous.ipynb
  lateral_inhibition_edge_and_motifs.ipynb
  lateral_inhibition_tmy16_tm33_mediators.ipynb
```

複数画像の一括保存などバッチ処理は notebook ではなく **`scripts/`** に置く(`uv run python` で実行)。

## コミット時の出力除去 (nbstripout, git filter)

`.gitattributes` で `*.ipynb` に nbstripout フィルタを適用している。**クローンごとに一度**だけ次を実行して
ローカル git にフィルタを登録する(`uv sync --all-extras` で nbstripout が入る):

```bash
uv run nbstripout --install --attributes .gitattributes
```

これで `git add` 時に出力・実行カウント・揮発メタデータが自動的に剥がれる(作業コピーの表示はそのまま)。
インストール状態の確認:

```bash
git check-attr filter -- notebooks/lateral_inhibition_rigorous.ipynb   # -> filter: nbstripout
```

## 実行

JupyterLab / VS Code で `.ipynb` を開いて Run All。実行や依存解決はプロジェクトルールに従い `uv` を使う
(`uv run jupyter lab` など)。図などをローカルで保存しても、コミット時に出力は除去される。
