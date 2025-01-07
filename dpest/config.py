import numpy as np

prng = np.random.default_rng(seed=42)
import yaml
import os

class ConfigManager:
    _config = None  # 設定を保持するクラス変数

    @classmethod
    def load_config(cls, config_file_name=None):
        """
        指定された設定ファイルを読み込み、設定値を保存
        """
        if cls._config is not None:
            return
        CUR_DIR = os.getcwd()
        if config_file_name is None:
            config_file_name = "common"
        config_file = f"{CUR_DIR}/experiment/config/{config_file_name}.yaml"
        with open(config_file, 'r') as f:
            cls._config = yaml.safe_load(f)

    @classmethod
    def get(cls, key, default=None):
        """
        設定値を取得
        """
        if cls._config is None:
            raise ValueError("Config not loaded. Call 'load_config' first.")
        return cls._config.get(key, default)
