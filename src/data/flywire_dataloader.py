import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import DATA_DIR
from .base_data_manager import BaseDataManager


class FlyWireDataManager(BaseDataManager):
    # データセット別のデフォルトスケールファクター
    # Optic Lobe v1.1と同程度のスケールにするため6000を使用
    DEFAULT_SCALE_FACTOR = 6000
    # X軸オフセット（スケール調整後）
    DEFAULT_X_OFFSET = 17  # 元の100μmをスケール調整 (100/6 ≈ 17)

    @staticmethod
    def get_photoreceptor_coordinates(
        csv_path: Optional[str] = None,
        target_neuron_types: Optional[List[str]] = None,
        scale_factor: Optional[float] = None,
        x_offset: Optional[float] = None,
        side: str = "right",
        center: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        光受容体（Photoreceptor）の座標データを取得（スケーリング済み）

        他のCSVファイルを読み込まずに、座標データのみを高速に取得する軽量メソッド。

        Args:
            csv_path: CSVファイルパス（未指定時はデフォルト）
            target_neuron_types: 対象ニューロンタイプ（デフォルト: ["R1-6"]）
            scale_factor: nm→μm変換係数（デフォルト: 6000）
            x_offset: X軸オフセット（デフォルト: 17μm）
            side: 左右の選択（"right", "left", "both"）
            center: 重心を原点に移動するかどうか

        Returns:
            df: 座標データを含むDataFrame
                - root_id: ニューロンID
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: スケーリング済み座標（μm）
            metadata: スケーリングパラメータなどのメタデータ
        """
        # デフォルト値の設定
        if csv_path is None:
            csv_path = os.path.join(DATA_DIR, "derived/geometry/flywire/retina_neurons_tips.csv")
        if target_neuron_types is None:
            target_neuron_types = ["R1-6"]
        if scale_factor is None:
            scale_factor = FlyWireDataManager.DEFAULT_SCALE_FACTOR
        if x_offset is None:
            x_offset = FlyWireDataManager.DEFAULT_X_OFFSET

        # CSVを読み込み
        df = pd.read_csv(csv_path, dtype={"root_id": str})

        # サイドでフィルタリング
        if side != "both":
            df = df[df["side"] == side]

        # ニューロンタイプでフィルタリング
        df = df[df["primary_type"].isin(target_neuron_types)]

        logger.debug(f"Loaded {len(df)} photoreceptor neurons (side={side}, types={target_neuron_types})")

        # 座標をスケーリング（nm → μm）
        X = df["tip_x"].values / scale_factor
        Y = df["tip_y"].values / scale_factor
        Z = df["tip_z"].values / scale_factor

        # 中心化とオフセット適用
        center_coords = np.array([0.0, 0.0, 0.0])
        if center:
            center_x, center_y, center_z = np.mean(X), np.mean(Y), np.mean(Z)
            center_coords = np.array([center_x, center_y, center_z])
            X = X - center_x - x_offset
            Y = Y - center_y
            Z = Z - center_z
        elif x_offset != 0:
            X = X - x_offset

        # 結果をDataFrameに格納
        result_df = pd.DataFrame(
            {
                "root_id": df["root_id"].values,
                "primary_type": df["primary_type"].values,
                "position_x": X,
                "position_y": Y,
                "position_z": Z,
            }
        )

        # メタデータ
        metadata = {
            "scale_factor": scale_factor,
            "x_offset": x_offset,
            "center": center_coords.tolist() if center else None,
            "side": side,
            "target_neuron_types": target_neuron_types,
            "num_neurons": len(result_df),
            "coordinate_ranges": {
                "x": [float(X.min()), float(X.max())],
                "y": [float(Y.min()), float(Y.max())],
                "z": [float(Z.min()), float(Z.max())],
            },
        }

        logger.debug(f"Coordinate ranges (μm): X=[{X.min():.1f}, {X.max():.1f}], Y=[{Y.min():.1f}, {Y.max():.1f}], Z=[{Z.min():.1f}, {Z.max():.1f}]")

        return result_df, metadata

    @staticmethod
    def get_photoreceptor_coordinates_with_directions(
        target_neuron_types: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        光受容体の座標と視線方向を取得（UnifiedSurfaceFitterのキャッシュを活用）

        シミュレーターと可視化で同一の座標・視線方向を使用するための統一メソッド。
        キャッシュが存在する場合はそれを読み込み、存在しない場合はUnifiedSurfaceFitterを
        呼び出して計算・キャッシュを生成する。

        Args:
            target_neuron_types: 対象ニューロンタイプ（デフォルト: ["R1-6"]）
            use_cache: キャッシュを使用するかどうか（デフォルト: True）

        Returns:
            df: 座標と視線方向を含むDataFrame
                - root_id: ニューロンID (str)
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: 投影後座標（μm）
                - sensitivity_axis_x, sensitivity_axis_y, sensitivity_axis_z: 視線方向
            metadata: スケーリングパラメータなどのメタデータ
        """
        if target_neuron_types is None:
            target_neuron_types = ["R1-6"]

        cache_path = os.path.join(DATA_DIR, "derived", "geometry", "flywire", "surface_fitting_cache.csv")

        # キャッシュが存在する場合は読み込み
        if use_cache and os.path.exists(cache_path):
            logger.debug(f"Loading photoreceptor coordinates from cache: {cache_path}")
            # メタデータ行（#で始まる）をスキップして読み込み
            with open(cache_path, "r") as f:
                lines = f.readlines()
            metadata_lines = [l for l in lines if l.startswith("#")]
            data_lines = [l for l in lines if not l.startswith("#")]

            from io import StringIO

            df = pd.read_csv(StringIO("".join(data_lines)), dtype={"root_id": str})

            # ニューロンタイプでフィルタリング
            if "neuron_type" in df.columns:
                df = df[df["neuron_type"].isin(target_neuron_types)]

            # 列名を統一
            result_df = pd.DataFrame(
                {
                    "root_id": df["root_id"].astype(str).values,
                    "primary_type": df["neuron_type"].values if "neuron_type" in df.columns else "R1-6",
                    "position_x": df["projected_x"].values,
                    "position_y": df["projected_y"].values,
                    "position_z": df["projected_z"].values,
                    "sensitivity_axis_x": df["sight_direction_x"].values,
                    "sensitivity_axis_y": df["sight_direction_y"].values,
                    "sensitivity_axis_z": df["sight_direction_z"].values,
                }
            )

            # メタデータをパース
            metadata = {
                "source": "cache",
                "cache_path": cache_path,
                "target_neuron_types": target_neuron_types,
                "num_neurons": len(result_df),
            }
            for line in metadata_lines:
                if line.startswith("# surface_params:"):
                    metadata["surface_params"] = line.split(": ", 1)[1].strip()
                elif line.startswith("# center:"):
                    metadata["center"] = line.split(": ", 1)[1].strip()

            logger.debug(f"Loaded {len(result_df)} photoreceptor neurons from cache")
            return result_df, metadata

        # この最小構成では surface fitting 本体は持っていない。
        # キャッシュ (data/derived/geometry/flywire/surface_fitting_cache.csv) を
        # 事前に用意しておくこと (drosophila リポジトリの UnifiedSurfaceFitter で生成)。
        raise FileNotFoundError(
            f"Surface fitting cache not found: {cache_path}. "
            "Generate it in the drosophila repository (UnifiedSurfaceFitter) "
            "and place the CSV at this path, or call get_photoreceptor_coordinates() "
            "instead if sight directions are not required."
        )

    # ========================================
    # BaseDataManager unified interface methods
    # ========================================

    @staticmethod
    def get_input_neuron_coordinates(
        target_neuron_types: Optional[List[str]] = None,
        scale_factor: Optional[float] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        BaseDataManager interface: alias for get_photoreceptor_coordinates.
        """
        return FlyWireDataManager.get_photoreceptor_coordinates(
            target_neuron_types=target_neuron_types,
            scale_factor=scale_factor,
            **kwargs,
        )

    @staticmethod
    def get_input_neuron_coordinates_with_directions(
        target_neuron_types: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        BaseDataManager interface: alias for get_photoreceptor_coordinates_with_directions.
        """
        return FlyWireDataManager.get_photoreceptor_coordinates_with_directions(
            target_neuron_types=target_neuron_types,
            use_cache=use_cache,
        )

    def get_input_neurons_df(self) -> pd.DataFrame:
        """
        BaseDataManager interface: 入力ニューロン（R1-6）の DataFrame を取得。

        Returns:
            DataFrame with standardized columns:
                - root_id: ニューロン ID (str)
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: 座標
        """
        ol_neurons_r_df, _ = self.get_ol_r_dfs()
        return ol_neurons_r_df[ol_neurons_r_df["primary_type"] == "R1-6"]

    def get_connections_df(self) -> pd.DataFrame:
        """
        BaseDataManager interface: 接続 DataFrame を取得。

        Returns:
            DataFrame with standardized columns:
                - pre_root_id, post_root_id: 前/後シナプスニューロン ID
                - weight: シナプス重み (syn_count)
                - alpha: 興奮性(1)/抑制性(-1)
        """
        _, ol_connections_r_df = self.get_ol_r_dfs()
        return ol_connections_r_df

    def __init__(
        self,
        use_no_threshold_connections: bool = True,
        use_fib25_fib19_v2_2: bool = True,
        *,
        load_all_csv: bool = False,
        extra_csv_stems: Optional[List[str]] = None,
    ):
        self.FLYWIRE_CSV_DIR = os.path.join(DATA_DIR, "raw", "flywire", "csv")
        self.use_no_threshold_connections = use_no_threshold_connections
        self.use_fib25_fib19_v2_2 = use_fib25_fib19_v2_2
        # data/raw/flywire/csv 以下の CSV を stem -> DataFrame で保持（例: "neurons" -> neurons_df）
        self.dataframes: dict[str, pd.DataFrame] = {}
        self._csv_paths_by_stem = self._index_csv_paths()
        self.neuropil_info = self._load_neuropil_info()

        # Optional CSV loading controls (constructor overrides env vars)
        if not load_all_csv:
            load_all_csv = str(os.getenv("FLYWIRE_LOAD_ALL_CSV", "0")).lower() in ("1", "true", "yes", "on")
        if extra_csv_stems is None:
            env_extra = os.getenv("FLYWIRE_EXTRA_CSV_STEMS")
            if env_extra:
                extra_csv_stems = [s.strip() for s in env_extra.split(",") if s.strip()]

        stems_to_load = set(self._required_csv_stems())
        if extra_csv_stems:
            stems_to_load |= {str(s) for s in extra_csv_stems if str(s).strip()}
        if load_all_csv:
            stems_to_load = set(self._csv_paths_by_stem.keys())
        self._load_csv_stems(stems_to_load)
        self._process_neurons_df()
        self._process_optic_lobe_data()

    def _required_csv_stems(self) -> list[str]:
        """
        FlyWireDataManager がデフォルトで必要とする CSV (stem)。

        - これらは学習/推論/主要な解析の標準パスで参照される。
        - 追加のCSVが必要な場合は __init__ の extra_csv_stems か load_all_csv を使う。
        """
        stems = ["classification", "neurons", "coordinates", "consolidated_cell_types"]
        stems.append("connections_princeton_no_threshold" if self.use_no_threshold_connections else "connections")
        return stems

    def _index_csv_paths(self) -> dict[str, Path]:
        csv_dir = Path(self.FLYWIRE_CSV_DIR)
        csv_paths = sorted(csv_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
        return {p.stem: p for p in csv_paths}

    def _read_csv_with_id_dtypes(self, path: Path) -> pd.DataFrame:
        # まずヘッダだけ読み、存在するID系カラムにだけdtype=strを適用（不存在カラム指定で落ちるのを回避）
        header_df = pd.read_csv(path, nrows=0)
        cols = set(header_df.columns.tolist())

        dtype_map: dict[str, type] = {}
        if "root_id" in cols:
            dtype_map["root_id"] = str
        for c in cols:
            if c.endswith("_root_id"):
                dtype_map[c] = str
        if "pre_pt_root_id" in cols:
            dtype_map["pre_pt_root_id"] = str
        if "post_pt_root_id" in cols:
            dtype_map["post_pt_root_id"] = str

        df = pd.read_csv(path, dtype=dtype_map if dtype_map else None)

        # Princeton系は列名を既存コード互換に揃える
        if path.stem.startswith("connections_princeton"):
            if "pre_pt_root_id" in df.columns and "post_pt_root_id" in df.columns:
                df = df.rename(columns={"pre_pt_root_id": "pre_root_id", "post_pt_root_id": "post_root_id"})

        return df

    def _load_csv_stems(self, stems: set[str]) -> None:
        logger.debug("Loading FlyWire CSVs (stems=%s)...", sorted(stems))
        missing = sorted([s for s in stems if s not in self._csv_paths_by_stem])
        if missing:
            csv_dir = Path(self.FLYWIRE_CSV_DIR)
            raise FileNotFoundError(f"Missing CSV(s) in {csv_dir}: {missing}")

        for stem in sorted(stems):
            if stem in self.dataframes:
                continue
            path = self._csv_paths_by_stem[stem]
            df = self._read_csv_with_id_dtypes(path)
            self.dataframes[stem] = df
            setattr(self, f"{stem}_df", df)

    def _load_neuropil_info(self):
        with open(os.path.join(DATA_DIR, "neuropil_info.json")) as f:
            return json.load(f)

    def _process_neurons_df(self):
        logger.debug("Processing neurons_df...")
        # 元の neurons.csv に含まれる神経伝達物質推定（nt_type 等）を保持しておく
        base_neurons_df = self.neurons_df.copy()
        nt_cols = [
            c
            for c in [
                "nt_type",
                "nt_type_score",
                "da_avg",
                "ser_avg",
                "gaba_avg",
                "glut_avg",
                "ach_avg",
                "oct_avg",
            ]
            if c in base_neurons_df.columns
        ]
        nt_df = base_neurons_df[["root_id"] + nt_cols].copy() if nt_cols else None

        # classification.csv のスキーマ差分に耐える（手元データでは cell_type 列が無い場合がある）
        if "cell_type" not in self.classification_df.columns:
            if "sub_class" in self.classification_df.columns:
                self.classification_df = self.classification_df.assign(cell_type=self.classification_df["sub_class"])
            elif "class" in self.classification_df.columns:
                self.classification_df = self.classification_df.assign(cell_type=self.classification_df["class"])
            else:
                self.classification_df = self.classification_df.assign(cell_type=pd.NA)

        required_classification_cols = ["root_id", "cell_type", "flow", "super_class", "class", "sub_class", "side"]
        for c in required_classification_cols:
            if c not in self.classification_df.columns:
                self.classification_df[c] = pd.NA

        self.neurons_df = self.neurons_df[["root_id", "group"]].merge(
            self.classification_df[["root_id", "cell_type", "flow", "super_class", "class", "sub_class", "side"]], on="root_id", how="left"
        )
        self.neurons_df = self.neurons_df.merge(self.consolidated_cell_types_df[["root_id", "primary_type"]], on="root_id", how="left")
        self.neurons_df.loc[self.neurons_df["primary_type"].isin(["R1-6", "R7", "R8"]), "group"] = "RE"
        self.neurons_df.loc[:, "group"] = self.neurons_df.loc[:, "group"].apply(lambda x: x.replace(".", "-"))

        coordinates_deduplicated_df = self.coordinates_df.groupby("root_id", as_index=False).first()
        coordinates_deduplicated_df["position"] = coordinates_deduplicated_df["position"].apply(self._str2array)
        coordinates_deduplicated_df[["position_x", "position_y", "position_z"]] = pd.DataFrame(
            coordinates_deduplicated_df["position"].tolist(), index=coordinates_deduplicated_df.index
        )
        self.neurons_df = self.neurons_df.merge(
            coordinates_deduplicated_df[["root_id", "position_x", "position_y", "position_z"]], on="root_id", how="left"
        )

        # nt_type を downstream の predictedNt と互換になるように追加
        if nt_df is not None:
            self.neurons_df = self.neurons_df.merge(nt_df, on="root_id", how="left")
            if "predictedNt" not in self.neurons_df.columns and "nt_type" in self.neurons_df.columns:
                self.neurons_df["predictedNt"] = self.neurons_df["nt_type"]

    def _process_optic_lobe_data(self):
        logger.debug("Processing optic_lobe_neurons_df...")
        self.optic_lobe_neurons_df = self.neurons_df[self.neurons_df["group"].apply(lambda x: self._select_region(x, "Optic Lobe"))]
        self.optic_lobe_neurons_df = self.optic_lobe_neurons_df[
            ~self.optic_lobe_neurons_df["group"].apply(lambda x: any(y in x for y in ["LH", "UNASGD", "AME"]))
        ]

        # 接続データの選択
        # connections_df_to_use = self.connections_no_threshold_df if self.use_no_threshold_connections else self.connections_df
        connections_df_to_use = self.connections_princeton_no_threshold_df if self.use_no_threshold_connections else self.connections_df

        connections_df_to_use = self._merge_primary_type_info(connections_df_to_use, "pre")
        connections_df_to_use = self._merge_primary_type_info(connections_df_to_use, "post")
        self.optic_lobe_connections_df = connections_df_to_use[
            connections_df_to_use["pre_root_id"].isin(self.optic_lobe_neurons_df["root_id"])
            & connections_df_to_use["post_root_id"].isin(self.optic_lobe_neurons_df["root_id"])
        ]

        self.optic_lobe_connections_df = self._merge_group_info(self.optic_lobe_connections_df, "pre")
        self.optic_lobe_connections_df = self._merge_group_info(self.optic_lobe_connections_df, "post")

        if self.use_fib25_fib19_v2_2:
            self.optic_lobe_connections_df = self._merge_fib25_fib19_v2_2(self.optic_lobe_connections_df)

        # Identify direct connections between photoreceptors
        photoreceptor_connections = self.optic_lobe_connections_df[
            (self.optic_lobe_connections_df["pre_group"] == "RE") & (self.optic_lobe_connections_df["post_group"] == "RE")
        ]

        if not photoreceptor_connections.empty:
            logger.debug(f"Direct connections between photoreceptors found/removed: {len(photoreceptor_connections)} connections")

            # Remove direct connections between photoreceptors
            self.optic_lobe_connections_df = self.optic_lobe_connections_df[
                ~((self.optic_lobe_connections_df["pre_group"] == "RE") & (self.optic_lobe_connections_df["post_group"] == "RE"))
            ]
        else:
            logger.debug("No direct connections between photoreceptors were found.")

        # --- Domain knowledge fix (photoreceptors R1-6) ---
        # R1-6 are experimentally known to be histaminergic and inhibitory.
        # Ensure downstream analyses using nt_type/alpha are consistent even if the source dataset differs.
        if "nt_type" not in self.optic_lobe_connections_df.columns:
            self.optic_lobe_connections_df.loc[:, "nt_type"] = pd.NA
        self.optic_lobe_connections_df.loc[self.optic_lobe_connections_df["pre_primary_type"] == "R1-6", "nt_type"] = "HIS"

        if "alpha" not in self.optic_lobe_connections_df.columns:
            self.optic_lobe_connections_df.loc[:, "alpha"] = 1
        self.optic_lobe_connections_df.loc[self.optic_lobe_connections_df["pre_primary_type"] == "R1-6", "alpha"] = -1

        logger.debug(
            f"Final number of connections from photoreceptors: "
            f"{self.optic_lobe_connections_df[self.optic_lobe_connections_df['pre_group'] == 'RE'].shape[0]}"
        )
        logger.debug(f"Final total number of connections: {self.optic_lobe_connections_df.shape[0]}")

    def _merge_group_info(self, df, prefix):
        df = df.merge(self.optic_lobe_neurons_df[["root_id", "group"]], left_on=f"{prefix}_root_id", right_on="root_id", how="left")
        df = df.rename(columns={"group": f"{prefix}_group"})
        return df.drop(columns=["root_id"])

    def _merge_primary_type_info(self, df, prefix):
        df = df.merge(self.optic_lobe_neurons_df[["root_id", "primary_type"]], left_on=f"{prefix}_root_id", right_on="root_id", how="left")
        df = df.rename(columns={"primary_type": f"{prefix}_primary_type"})
        return df.drop(columns=["root_id"])

    def _select_region(self, name, region):
        return any(x in name for x in self.neuropil_info[region]) and all(
            all(x not in name for x in self.neuropil_info[another]) for another in self.neuropil_info if another != region
        )

    def _merge_fib25_fib19_v2_2(self, connections_df):
        # NOTE: リポジトリ内の配置は data/raw/fib25-fib19_v2.2.json が基本だが、
        # 以前の配置（data/ 直下）もあり得るため両方を探す。
        candidate_paths = [
            Path(DATA_DIR) / "fib25-fib19_v2.2.json",
            Path(DATA_DIR) / "raw" / "fib25-fib19_v2.2.json",
        ]
        json_path = next((p for p in candidate_paths if p.exists()), None)
        if json_path is None:
            raise FileNotFoundError("fib25-fib19_v2.2.json not found. Looked in: " + ", ".join(str(p) for p in candidate_paths))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # JSONからエッジ情報を取得
        edges_df = pd.DataFrame(data["edges"])

        # デフォルトでは全ての接続をalpha=1（興奮性）に設定
        connections_df.loc[:, "alpha"] = 1

        neuron_type_mapping = {
            "Am": "Am1",
            "CT1(Lo1)": "CT1",
            "CT1(M10)": "CT1",
            "R1": "R1-6",
            "R2": "R1-6",
            "R3": "R1-6",
            "R4": "R1-6",
            "R5": "R1-6",
            "R6": "R1-6",
            "TmY9": "TmY9q",
        }

        edges_for_merge = edges_df.copy()
        for json_name, df_name in neuron_type_mapping.items():
            edges_for_merge.loc[edges_for_merge["src"] == json_name, "src"] = df_name
            edges_for_merge.loc[edges_for_merge["tar"] == json_name, "tar"] = df_name

        # JSONのエッジ情報をマージするための準備
        edges_mapping = {"src": "pre_primary_type", "tar": "post_primary_type"}

        # カラム名をマッピングしてエッジデータフレームを準備
        edges_for_merge = edges_for_merge.rename(columns=edges_mapping)

        # 重複を避けるためにdrop_duplicatesを追加
        edges_for_merge = edges_for_merge.drop_duplicates(["pre_primary_type", "post_primary_type"])

        # 代わりに、pre_primary_typeとpost_primary_typeの組み合わせに基づいてalpha値を更新
        # 辞書を作成
        alpha_dict = dict(zip(zip(edges_for_merge["pre_primary_type"], edges_for_merge["post_primary_type"]), edges_for_merge["alpha"]))

        # 辞書を使用して更新 (iterrows() の代わりに map を使用)
        connections_df["alpha"] = list(
            map(lambda x: alpha_dict.get(x, 1), zip(connections_df["pre_primary_type"], connections_df["post_primary_type"]))
        )

        connections_df.loc[connections_df["pre_primary_type"] == "R1-6", "alpha"] = -1

        return connections_df

    @staticmethod
    def _str2array(s):
        return [int(x) for x in s[1:-1].split(" ") if x != ""]

    # refs #36, #40: CT1 は wide-field tangential 細胞で、classification.csv の soma side と
    # シナプスを張る neuropil の hemisphere がほぼ反転している (CT1_left soma → LO_R / ME_R に
    # 投射、その逆も同様)。素朴な side== フィルタでは CT1 が T4/T5 への入力経路から完全に
    # 脱落 (edge_mask_t5_base_inhib = 0) する。bilateral 型として side フィルタを免除し、
    # synapse-side neuropil 制約で右半球内の投射のみを残す。
    BILATERAL_TYPES: frozenset = frozenset({"CT1"})

    def _restrict_bilateral_edges_by_neuropil(self, conn_df, hemisphere_suffix: str):
        """bilateral 型 pre/post を含む edge を、シナプス位置 neuropil で hemisphere 制約する。"""
        if "neuropil" not in conn_df.columns:
            return conn_df
        bilateral_ids = self.optic_lobe_neurons_df[
            self.optic_lobe_neurons_df["primary_type"].isin(self.BILATERAL_TYPES)
        ]["root_id"]
        if bilateral_ids.empty:
            return conn_df
        bset = set(bilateral_ids.astype(str))
        pre_str = conn_df["pre_root_id"].astype(str)
        post_str = conn_df["post_root_id"].astype(str)
        is_bilateral_pre = pre_str.isin(bset)
        is_bilateral_post = post_str.isin(bset)
        neuropil_str = conn_df["neuropil"].astype(str)
        is_target_side = neuropil_str.str.endswith(hemisphere_suffix)
        # bilateral が片方でも関与する edge は、シナプス位置が対象半球の neuropil である必要がある
        return conn_df[(~(is_bilateral_pre | is_bilateral_post)) | is_target_side]

    def _ensure_stem_loaded(self, stem: str) -> pd.DataFrame:
        """Lazily load a CSV stem on demand and return its DataFrame.

        Used for optional tables (e.g. visual_neuron_types, neuropil_synapse_table)
        that are not in ``_required_csv_stems()`` so callers need not preload them.
        """
        if stem not in self.dataframes:
            self._load_csv_stems({stem})
        return self.dataframes[stem]

    def get_visual_neuron_types_df(self) -> pd.DataFrame:
        """visual_neuron_types.csv (root_id, type, family, subsystem, category, side).

        Curated optic-lobe taxonomy consumed by ``src.lateral.stage`` for processing
        stage assignment. Loaded lazily on first call (~6 MB).
        """
        return self._ensure_stem_loaded("visual_neuron_types")

    def get_neuropil_synapse_table_df(self) -> pd.DataFrame:
        """neuropil_synapse_table.csv: per-neuron input/output synapse & partner counts
        broken down by neuropil. Used to derive each neuron's dominant input/output
        neuropil. ~90 MB; loaded lazily on first call.
        """
        return self._ensure_stem_loaded("neuropil_synapse_table")

    def get_ol_r_dfs(self):
        is_right = self.optic_lobe_neurons_df["side"] == "right"
        is_bilateral = self.optic_lobe_neurons_df["primary_type"].isin(self.BILATERAL_TYPES)
        ol_neurons_r_df = self.optic_lobe_neurons_df[is_right | is_bilateral]
        candidate_ids = ol_neurons_r_df["root_id"]
        edges = self.optic_lobe_connections_df[
            self.optic_lobe_connections_df["pre_root_id"].isin(candidate_ids)
            & self.optic_lobe_connections_df["post_root_id"].isin(candidate_ids)
        ]
        ol_connections_r_df = self._restrict_bilateral_edges_by_neuropil(edges, "_R")

        return ol_neurons_r_df, ol_connections_r_df


if __name__ == "__main__":
    data_manager = FlyWireDataManager()
    ol_neurons_r_df, ol_connections_r_df = data_manager.get_ol_r_dfs()
    logger.debug(f"\n{ol_neurons_r_df.head()}")
    logger.debug(f"\n{ol_connections_r_df.head()}")
