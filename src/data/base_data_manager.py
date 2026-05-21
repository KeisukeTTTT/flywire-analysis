"""
Base Data Manager Abstract Class

全ての connectome データマネージャが実装すべき共通インターフェースを定義。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd


class BaseDataManager(ABC):
    """
    Connectome データマネージャの抽象基底クラス。

    データセット固有のマネージャ (FlyWireDataManager 等) はこのクラスを継承し、
    統一されたインターフェースを提供する。

    標準化された DataFrame カラム名:
        - root_id (str): ニューロン ID
        - primary_type (str): ニューロンタイプ
        - position_x, position_y, position_z (float): 座標 (μm)
        - sensitivity_axis_x, sensitivity_axis_y, sensitivity_axis_z (float): 視線方向ベクトル

    接続 DataFrame の標準化されたカラム名:
        - pre_root_id, post_root_id (str): 前/後シナプスニューロン ID
        - weight (int): シナプス重み
        - alpha (int): 興奮性(1)/抑制性(-1)/不明(0)
    """

    @staticmethod
    @abstractmethod
    def get_input_neuron_coordinates(
        target_neuron_types: Optional[List[str]] = None,
        scale_factor: Optional[float] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        入力ニューロンの座標データを取得（スケーリング済み）。

        Args:
            target_neuron_types: 対象ニューロンタイプのリスト
            scale_factor: nm → μm 変換係数
            **kwargs: データセット固有の追加パラメータ

        Returns:
            df: 座標データを含む DataFrame
                - root_id: ニューロン ID (str)
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: スケーリング済み座標 (μm)
            metadata: スケーリングパラメータなどのメタデータ
        """
        pass

    @staticmethod
    @abstractmethod
    def get_input_neuron_coordinates_with_directions(
        target_neuron_types: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        入力ニューロンの座標と視線方向を取得。

        シミュレーターと可視化で同一の座標・視線方向を使用するための統一メソッド。

        Args:
            target_neuron_types: 対象ニューロンタイプのリスト
            use_cache: キャッシュを使用するかどうか

        Returns:
            df: 座標と視線方向を含む DataFrame
                - root_id: ニューロン ID (str)
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: 投影後座標 (μm)
                - sensitivity_axis_x, sensitivity_axis_y, sensitivity_axis_z: 視線方向
            metadata: スケーリングパラメータなどのメタデータ
        """
        pass

    @abstractmethod
    def get_input_neurons_df(self) -> pd.DataFrame:
        """
        入力ニューロンの DataFrame を取得（標準化されたカラム名）。

        Returns:
            DataFrame with standardized columns:
                - root_id: ニューロン ID (str)
                - primary_type: ニューロンタイプ
                - position_x, position_y, position_z: 座標
        """
        pass

    @abstractmethod
    def get_connections_df(self) -> pd.DataFrame:
        """
        接続 DataFrame を取得（標準化されたカラム名）。

        Returns:
            DataFrame with standardized columns:
                - pre_root_id, post_root_id: 前/後シナプスニューロン ID
                - weight: シナプス重み
                - alpha: 興奮性(1)/抑制性(-1)/不明(0)
        """
        pass
