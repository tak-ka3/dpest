import numpy as np

prng = np.random.default_rng(seed=42)
# TODO: 以下のパラメタの調整
LAP_UPPER = 50
LAP_LOWER = -50
LAP_VAL_NUM = 100
EXP_UPPER = 50
EXP_LOWER = 0
EXP_VAL_NUM = 100
INF = 10000
SAMPLING_NUM = 100000
"""
サンプリングによって小数型の出力の確率を求める際に、確率変数の値を区切るグリッドの数
"""
GRID_NUM = 20
