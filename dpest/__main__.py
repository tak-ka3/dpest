# 型ヒントを遅延評価できるようになる
from __future__ import annotations
import numpy as np
import copy
from scipy import interpolate
from scipy.integrate import quad
from dpest.input_generator import input_generator
from dpest.search import search_scalar_all
from utils.pr_calc import nonuniform_convolution

prng = np.random.default_rng(seed=42)
# TODO: 以下のパラメタの調整
LAP_UPPER = 50
LAP_LOWER = -50
LAP_VAL_NUM = 1000
INF = 10000

"""
確率質量関数
"""
class Pmf:
    def __init__(self, val_to_prob=None):
        self.val_to_prob = val_to_prob or {}
        self.child = []
        self.name = "Pmf"
        self.is_args_depend = False # 引数間に依存関係があるか
        self.depend_vars = [] # この確率変数が依存している確率変数で、LaplaceかPmfのような元の確率変数しか含まない
        self.is_sampling = False

    def sampling(self):
        """
        プログラムを実行してサンプリングできるようにしておく
        """
        pass

    def _insert_input_val(self, input_val_list: list):
        """
        サンプリングで得られた値を代入する
        """
        pass

    def _resolve_dependency(self, input_val_list: list):
        """
        あとは解析的に計算することが可能な計算グラフが出来上がる
        TODO: 一つ上の階層の関数がいらないかもしれない
        """
        def _resolve_dependency_rec(var, input_val_ind):
            # TODO: Pmfクラスも対応できるように、var.child==[]という条件分岐にする
            if isinstance(var, Laplace):
                assert len(var.child) == 1
                if isinstance(var.child[0], int | float):
                    return var
                elif isinstance(var.child[0], ArrayItem):
                    input_val_ind += 1
                    var.child[0] = input_val_list[input_val_ind-1]
                    vals = np.linspace(var.lower, var.upper, var.num)
                    probs = np.exp(-np.abs(vals - var.child[0]) / var.b) / (2 * var.b)
                    var.val_to_prob = dict(zip(vals, probs))
                    return var
            if var.is_args_depend:
                # ここでサンプリングにより確率密度を計算する
                return Pmf()
            updated_child = [_resolve_dependency_rec(child, input_val_ind) for child in var.child]
            var.child = updated_child
            return var

        input_val_ind = 0
        calc_graph = _resolve_dependency_rec(self, input_val_ind)
        return calc_graph

    def _calc_pdf(self, calc_graph):
        """
        依存関係がないcalc_graphを最も子供から順に確率密度を解析的に計算する
        Returns: 
            var: 最終的に得られた確率変数。確率密度もプロパティに含む。
        """
        def _calc_pdf_rec(var):
            if isinstance(var, Laplace | OPmf):
                return var
            output_var = var.calc_pdf([_calc_pdf_rec(child) for child in var.child])
            return output_var
        return _calc_pdf_rec(calc_graph)
        
    """
    pytorchのbackwordのように計算式を作った後に、最後に計算グラフを遡って、εの値を計算する
    """
    def eps_est(self):
        """
        計算グラフを辿って、Operationの依存関係を調べる
        依存関係は最も子供から調べる。子が依存する確率変数は親も依存するという関係
        """
        # ArrayItemに代入する
        input_list = input_generator("inf", 5)
        for input_set in input_list:
            Y1 = copy.deepcopy(self)
            calc_graph1 = Y1._resolve_dependency(input_set[0])
            Y1 = self._calc_pdf(calc_graph1)
            val1, pdf1 = list(Y1.val_to_prob.keys()), list(Y1.val_to_prob.values())

            Y2 = copy.deepcopy(self)
            calc_graph2 = Y2._resolve_dependency(input_set[1])
            Y2 = self._calc_pdf(calc_graph2)
            val2, pdf2 = list(Y2.val_to_prob.keys()), list(Y2.val_to_prob.values())

            if val1 != val2:
                raise ValueError("Invalid value")
            eps = search_scalar_all(np.array(val1), np.array(pdf1), np.array(pdf2))
            print("eps=", eps)

        print("eps_est")
        pass

# TODO: Pmfクラスの名前を変えて、こちらに採用する
class OPmf(Pmf):
    def __init__(self, val_to_prob=None):
        super().__init__(val_to_prob)
        self.name = "OPmf"

    def __call__(self):
        return prng.laplace(0, 1)

class Laplace(Pmf):
    def __init__(self, mu: float | ArrayItem, b: float):
        super().__init__()
        self.child = [mu]
        self.name = f"Laplace({mu.name}, {b})"
        self.depend_vars = [self]
        self.upper = LAP_UPPER
        self.lower = LAP_LOWER
        self.num = LAP_VAL_NUM
        self.b = b

    def __call__(self, mu: float, b: float):
        return prng.laplace(mu, b)

"""
addクラス
"""
class Add(Pmf):
    def __init__(self, var1: Pmf, var2: Pmf):
        super().__init__()
        self.child = [var1, var2]
        self.name = f"Add({var1.name}, {var2.name})"
        self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
        self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))

    def __call__(self, var1: Pmf, var2: Pmf):
        return var1() + var2()
    
    def calc_pdf(self, child):
        """
        引数から出力の確率密度を厳密に計算する
        """
        val1 = list(child[0].val_to_prob.keys())
        pdf1 = list(child[0].val_to_prob.values())
        val2 = list(child[1].val_to_prob.keys())
        pdf2 = list(child[1].val_to_prob.values())
        if np.round(val1[1] - val1[0], 10) == np.round(val2[1] - val2[0], 10):
            dx = val1[1] - val1[0]
            output_pdf = np.convolve(pdf1, pdf2, mode='full') * dx
            start_val = val1[0] + val2[0]
            output_val_num = len(val1) + len(val2) - 1
            output_val_range = np.arange(start_val, start_val + dx*output_val_num, dx)
            return OPmf(dict(zip(output_val_range, output_pdf)))
        # TODO: こちら側の処理が正しいかを確認する（NoisySumでやや怪しい気がした）
        else:
            # 各PDFを順次畳み込み
            vals = []
            # 全ての確率変数の和の組み合わせを計算する
            for v1 in val1:
                for v2 in val2:
                    vals.append(v1 + v2)
            np_vals = np.unique(np.array(vals))
            split_num = np_vals.size // len(val1)
            output_val_range = np_vals[::split_num]
            integral = "trapz"
            output_pdf = nonuniform_convolution(val1, val2, pdf1, pdf2, output_val_range, integral_way=integral)
            assert output_val_range.size == output_pdf.size
            return OPmf(dict(zip(output_val_range, output_pdf)))


"""
brクラス(未実装)
"""
class Br(Pmf):
    def __init__(self, var1: Pmf, var2: Pmf):
        super().__init__()
        self.child = [var1, var2]
        self.name = f"Br({var1.name}, {var2.name})"
        self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
        self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))

"""
max
"""
class Max(Pmf):
    def __init__(self, var1: Pmf, var2: Pmf):
        super().__init__()
        self.child = [var1, var2]
        self.name = f"Max({var1.name}, {var2.name})"
        self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
        self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))

    def __call__(self, var1: Pmf, var2: Pmf):
        return max(var1(), var2())
    
    def calc_pdf(self, child):
        """
        引数から出力の確率密度を厳密に計算する
        """
        val1 = list(child[0].val_to_prob.keys())
        pdf1 = list(child[0].val_to_prob.values())
        val2 = list(child[1].val_to_prob.keys())
        pdf2 = list(child[1].val_to_prob.values())
        output_vals = np.unique(np.concatenate([val1, val2]))

        f1 = interpolate.CubicSpline(val1, pdf1, bc_type='natural', extrapolate=True)
        f2 = interpolate.CubicSpline(val2, pdf2, bc_type='natural', extrapolate=True)
        output_pdf = []
        for v in output_vals:
            output_pdf.append(f1(v) * quad(f2, -INF, v)[0] + f2(v) * quad(f1, -INF, v)[0])
        return OPmf(dict(zip(output_vals, output_pdf)))

"""
case(未実装)
"""
class Case(Pmf):
    def __init__(self):
        super().__init__()

"""
to_array(未実装)
"""
class ToArray(Pmf):
    def __init__(self):
        super().__init__()

class ArrayItem:
    def __init__(self, ind: int, parent: Array):
        self.ind = ind
        self.parent = parent
        self.name = f"ArrayItem({ind})"

"""
シンボリックな配列の値のためのクラス
プログラマはこれを用いてコードを書くが実際の値はこちらがε推定の時に代入する
"""
class Array:
    def __init__(self, size):
        self.child = []
        self.size = size
        self.name = "Array"
    
    def __iter__(self):
        array_iter = [ArrayItem(i, self) for i in range(self.size)]
        return iter(array_iter)


def laplace_extract(arr: Array, scale: float):
    """
    配列要素それぞれにラプラスノイズを加えて取り出す
    """
    lap_list = []
    for val in arr:
        if isinstance(val, ArrayItem):
            lap_list.append(Laplace(val, scale))
        else:
            raise ValueError("Invalid value")
    return lap_list

"""
何かとあった方が良いかもしれない
"""
def laplace_map():
    pass

"""
プログラマのコード
"""
eps = 0.1
sens = 1
# 配列要素それぞれにラプラスノイズを加えて取り出す
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)
# Y = Max(Max(Max(Max(Lap1, Lap2), Lap3), Lap4), Lap5)
Y = Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5)
# このアルゴリズムで推定されたεの値が出力される


print("------")
eps = Y.eps_est()
print("----------")

""""""

# Lap1 = Laplace(0, 1)
# Lap2 = Laplace(0, 1)
# Lap3 = Laplace(0, 1)
# Lap4 = Laplace(0, 1)
# add = Add(Lap1, Lap2)
# Y = Max(Max(Max(Lap1, Lap2), Lap3), Lap4)
# print(Y.child)
# print(Y.eps_est())
