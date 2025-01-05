from __future__ import annotations
import numpy as np
from . import Pmf
from dpest.config import *
from dpest.input import ArrayItem

class HistPmf(Pmf):
    # histは二次元配列で、要素があるところに1が立つ
    def __init__(self, hist):
        self.hist = hist

class RawPmf(Pmf):
    def __init__(self, val_to_prob=None):
        super().__init__(val_to_prob=val_to_prob)
        self.name = "RawPmf"

class Laplace(Pmf):
    def __init__(self, mu: float | ArrayItem, b: float):
        super().__init__()
        self.child = [mu]
        self.b = b
        self.name = f"Laplace({mu.name if isinstance(mu, ArrayItem) else mu}, {b})"
        self.depend_vars = [self]
        self.upper = LAP_UPPER
        self.lower = LAP_LOWER
        self.num = LAP_VAL_NUM

    def calc_strict_pdf(self, vals: np.ndarray):
        """
        厳密な確率密度を計算する
        """
        return np.exp(-np.abs(vals - self.child[0]) / self.b) / (2 * self.b)
    
    def sampling(self):
        return prng.laplace(self.child[0], self.b)
        

class Exp(Pmf):
    def __init__(self, mu: float | ArrayItem, b: float):
        super().__init__()
        self.child = [mu]
        self.b = b
        self.name = f"Exponential({mu.name if isinstance(mu, ArrayItem) else mu}, {b})"
        self.depend_vars = [self]
        self.upper = EXP_UPPER
        self.lower = EXP_LOWER
        self.num = EXP_VAL_NUM

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
