from __future__ import annotations
import numpy as np
from . import Pmf
from dpest.config import ConfigManager, prng
from dpest.input import ArrayItem
from typing import Union

class HistPmf(Pmf):
    # histは二次元配列で、要素があるところに1が立つ
    def __init__(self, hist_dict):
        self.val_to_prob = hist_dict
        self.hist = np.array(list(hist_dict.values()))
        self.name = f"HistPmf({hist_dict})"

class RawPmf(Pmf):
    def __init__(self, val_to_prob=None):
        super().__init__(val_to_prob=val_to_prob)
        self.name = "RawPmf"

class Laplace(Pmf):
    def __init__(self, mu: Union[float, ArrayItem], b: float):
        super().__init__()
        self.child = [mu]
        self.b = b
        self.name = f"Laplace({mu.name if isinstance(mu, ArrayItem) else mu}, {b})"
        self.depend_vars = [self]
        self.upper = ConfigManager.get("LAP_UPPER")
        self.lower = ConfigManager.get("LAP_LOWER")
        self.num = ConfigManager.get("LAP_VAL_NUM")

    def calc_strict_pdf(self, vals: np.ndarray):
        """
        厳密な確率密度を計算する
        """
        return np.exp(-np.abs(vals - self.child[0]) / self.b) / (2 * self.b)
    
    def sampling(self, n_samples: int):
        return prng.laplace(self.child[0], self.b, n_samples)
        

class Exp(Pmf):
    def __init__(self, mu: Union[float, ArrayItem], b: float):
        super().__init__()
        self.child = [mu]
        self.b = b
        self.name = f"Exponential({mu.name if isinstance(mu, ArrayItem) else mu}, {b})"
        self.depend_vars = [self]
        self.upper = ConfigManager.get("EXP_UPPER")
        self.lower = ConfigManager.get("EXP_LOWER")
        self.num = ConfigManager.get("EXP_VAL_NUM")

    def calc_strict_pdf(self, vals: np.ndarray):
        """
        厳密な確率密度を計算する
        """
        return (1 / self.b) * np.exp(-(vals - self.child[0]) / self.b)
    
    def sampling(self):
        return prng.exponential(self.b) + self.child[0]

class Uni(Pmf):
    """
    一様分布
    """
    def __init__(self, lower: int, upper: int):
        prob = 1 / (upper - lower)
        val_to_prob = {i: prob for i in range(lower, upper)}
        super().__init__(val_to_prob=val_to_prob)
        self.name = f"Uni({lower}, {upper})"
        self.depend_vars = [self]
        self.lower = lower
        self.upper = upper
