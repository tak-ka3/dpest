from __future__ import annotations
from . import Pmf

class ArrayItem(Pmf):
    def __init__(self, ind: int, parent: InputArray):
        super().__init__()
        self.ind = ind
        self.parent = parent
        self.arr_size = parent.size
        self.adj = parent.adj
        self.name = f"ArrayItem({ind})"

class InputScalar:
    """
    シンボリックなスカラ値のためのクラス
    入力がスカラ値であるアルゴリズムではこれを用いる
    """
    def __init__(self):
        self.child = []
        self.name = "InputScalar"

class InputArray:
    """
    シンボリックな配列の値のためのクラス
    プログラマはこれを用いてコードを書くが実際の値はこちらがε推定の時に代入する
    """
    def __init__(self, size, adj):
        self.child = []
        self.size = size
        self.name = "InputArray"
        self.adj = adj
    
    def __iter__(self):
        array_iter = [ArrayItem(i, self) for i in range(self.size)]
        return iter(array_iter)
    
    def __getitem__(self, ind):
        return ArrayItem(ind, self)


class InputScalarToArrayItem:
    """
    InputScalarToArrayの要素を取り出すためのクラス
    """
    def __init__(self, ind: int, parent: InputScalarToArray):
        self.name = "InputScalarToArrayItem"
        self.scalar = None
        self.ind = ind
        self.parent = parent
    
    def set_parent_array(self, scalar):
        if self.parent.array is None:
            arr = self.parent.func(scalar)
            self.parent.array = arr

    def get_array_item(self):
        if self.parent.array is None:
            raise ValueError("Parent array is not set")
        return self.parent.array[self.ind]

class InputScalarToArray:
    """
    入力のスカラ値を配列に変換するクラス
    """
    def __init__(self, size: int, func):
        self.size = size
        self.func = func
        self.input_scalar = None
        self.array = None
    
    def __getitem__(self, ind):
        return InputScalarToArrayItem(ind, self)
