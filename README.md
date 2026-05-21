# flywire_analysis

FlyWire connectome dataset を読み込むための軽量ローダ。
姉妹リポジトリ [drosophila](https://github.com/) から `FlyWireDataManager` 周りだけを抽出した最小構成で、モデル・学習・推論・解析・シミュレータは含まれません。

## できること

- FlyWire raw CSV (`neurons`, `classification`, `coordinates`, `consolidated_cell_types`, `connections_princeton_no_threshold` 等) を読み込み、`pandas.DataFrame` として返す
- Optic Lobe 領域内のニューロン/接続を標準化カラム (`root_id`, `primary_type`, `position_x/y/z`, `pre_*`, `post_*`, `weight`, `alpha` 相当) で取得
- R1-6 光受容体を histaminergic / 抑制性 (`nt_type="HIS"`, `alpha=-1`) として補正
- 事前生成済み surface fitting cache (`data/derived/geometry/flywire/surface_fitting_cache.csv`) から投影座標 + 視線方向ベクトルを取得

surface fitting 自体や、シミュレーション・学習などはこのリポジトリには無いので、必要なら drosophila 側で生成してから cache CSV を配置してください。

## データの参照について

このリポジトリは FlyWire の生データ (~55GB) を同梱しません。各環境で
`data/` ディレクトリ (`raw/flywire/csv/`, `derived/geometry/flywire/`,
`neuropil_info.json`) を別途用意し、環境変数で参照パスを指定します。

```bash
cp .env.example .env.local
# .env.local を編集し、DROSOPHILA_DATA_DIR を環境ごとの絶対パスに
# 例: DROSOPHILA_DATA_DIR=/home/keisuke/Utokyo/Lab/drosophila/data
```

`src/config.py` が起動時に `.env` / `.env.local` を読み込み、
`DROSOPHILA_DATA_DIR` を `DATA_DIR` として伝播します。
`.env`, `.env.local` は `.gitignore` 済み (環境固有設定として扱う)。

## セットアップ

```bash
uv sync
uv run python -c "from src.data import FlyWireDataManager; m = FlyWireDataManager(); print(m.neurons_df.shape)"
```

依存は `numpy`, `pandas`, `loguru`, `python-dotenv` の 4 つだけ。

## API 例

```python
from src.data import FlyWireDataManager

m = FlyWireDataManager()

# 全ニューロン (139,255 rows)
m.neurons_df

# Optic Lobe 領域内のニューロン (90,810 rows)
m.optic_lobe_neurons_df

# Optic Lobe 内の接続 (R1-6→R1-6 直結は除外済み、9.1M rows)
m.optic_lobe_connections_df

# BaseDataManager の統一インターフェース
inputs_df = m.get_input_neurons_df()       # R1-6 のみ (8,467 rows)
conn_df   = m.get_connections_df()          # primary_type を付与した接続

# 光受容体座標のみ高速取得 (CSV 1 ファイルだけ読む)
coords_df, meta = FlyWireDataManager.get_photoreceptor_coordinates(side="right")

# 投影座標 + 視線方向 (surface fitting cache 必須)
coords_df, meta = m.get_photoreceptor_coordinates_with_directions()
```

## ディレクトリ構成

```
src/
  config.py                       # DATA_DIR を環境変数から解決
  data/
    __init__.py                   # FlyWireDataManager / DatasetDescriptor / FLYWIRE_DESCRIPTOR を re-export
    base_data_manager.py          # 抽象基底クラス
    dataset_descriptor.py         # FLYWIRE_DESCRIPTOR (scale_factor=6000, weight_column=syn_count, ...)
    flywire_dataloader.py         # 本体 (FlyWireDataManager)
.env.example                       # DROSOPHILA_DATA_DIR を含む環境変数テンプレート
pyproject.toml                     # 依存関係 (最小)
CLAUDE.md, AGENTS.md               # 実行ルール (uv run / Conventional Commits / GitHub Issues)
```

## 実行ルール

`CLAUDE.md` / `AGENTS.md` を参照。

- Python 実行は必ず `uv run`
- コミットメッセージは Conventional Commits
- 課題管理は GitHub Issues
