# 型ヒントを遅延評価できるようになる
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import interpolate
from scipy.integrate import quad
from itertools import product
from dpest.input_generator import input_generator
from dpest.search import search_scalar_all, search_hist
from utils.pr_calc import nonuniform_convolution

prng = np.random.default_rng(seed=42)
# TODO: 以下のパラメタの調整
LAP_UPPER = 50
LAP_LOWER = -50
LAP_VAL_NUM = 1000
INF = 10000
SAMPLING_NUM = 1000000
# サンプリングによって出力の確率を求める際の、確率変数の値を区切るグリッドの数
GRID_NUM = 5

"""
確率質量関数
"""
class Pmf:
    def __init__(self, *args, val_to_prob=None):
        self.val_to_prob = val_to_prob or {}
        self.child = list(args)
        self.name = "Pmf"
        all_vars = set()
        self.is_args_depend = False # 引数間に依存関係があるか
        for child in self.child:
            depend_vars_set = set(child.depend_vars)
            if all_vars & depend_vars_set:
                self.is_args_depend = True
            all_vars.update(depend_vars_set)
        self.depend_vars = list(all_vars)
        # self.is_args_depend = False # 引数間に依存関係があるか
        # self.depend_vars = [] # この確率変数が依存している確率変数で、LaplaceかPmfのような元の確率変数しか含まない
        self.is_sampling = False

    def _resolve_dependency(self, input_comb: tuple):
        """
        あとは解析的に計算することが可能な計算グラフが出来上がる
        TODO: 一つ上の階層の関数がいらないかもしれない
        """
        Y1 = copy.deepcopy(self)
        Y2 = copy.deepcopy(self)
        input_val_list1 = input_comb[0]
        input_val_list2 = input_comb[1]
        def _resolve_dependency_rec(var1: Pmf, var2: Pmf): # var1とvar2はLaplaceの確率密度以外は同じことを前提とする
            # TODO: Pmfクラスも対応できるように、var.child==[]という条件分岐にする
            if isinstance(var1, Laplace):
                assert len(var1.child) == 1
                if isinstance(var1.child[0], int | float):
                    return var1, var2
                elif isinstance(var1.child[0], ArrayItem):
                    var1.child[0] = input_val_list1[var1.child[0].ind]
                    var2.child[0] = input_val_list2[var2.child[0].ind]
                    vals = np.linspace(var1.lower, var1.upper, var1.num)
                    probs1 = np.exp(-np.abs(vals - var1.child[0]) / var1.b) / (2 * var1.b)
                    probs2 = np.exp(-np.abs(vals - var2.child[0]) / var2.b) / (2 * var2.b)
                    var1.val_to_prob = dict(zip(vals, probs1))
                    var2.val_to_prob = dict(zip(vals, probs2))
                    return var1, var2
            if var1.is_args_depend:
                # ここでサンプリングにより確率密度を計算する
                output_var1, output_var2 = calc_pdf_by_sampling(var1, var2, input_comb)
                return output_var1, output_var2
            updated_child1, updated_child2 = [], []
            for child1, child2 in zip(var1.child, var2.child):
                c1, c2 = _resolve_dependency_rec(child1, child2)
                updated_child1.append(c1)
                updated_child2.append(c2)
            var1.child = updated_child1
            var2.child = updated_child2
            return var1, var2

        calc_graph1, calc_graph2 = _resolve_dependency_rec(Y1, Y2)
        return calc_graph1, calc_graph2

    def _calc_pdf(self, calc_graph):
        """
        依存関係がないcalc_graphを最も子供から順に確率密度を解析的に計算する
        Returns: 
            var: 最終的に得られた確率変数。確率密度もプロパティに含む。
        """
        def _calc_pdf_rec(var):
            if isinstance(var, Laplace | OPmf | HistPmf):
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
        # TODO: infか1かの隣接性はプログラマが指定できるようにする
        input_list = input_generator("inf", 5)
        for input_set in input_list:
            calc_graph1, calc_graph2 = self._resolve_dependency(input_set)
            Y1 = self._calc_pdf(calc_graph1)
            Y2 = self._calc_pdf(calc_graph2)

            if isinstance(Y1, HistPmf):
                eps = search_hist(Y1.hist, Y2.hist)
            else:
                val1, pdf1 = list(Y1.val_to_prob.keys()), list(Y1.val_to_prob.values())
                val2, pdf2 = list(Y2.val_to_prob.keys()), list(Y2.val_to_prob.values())
                if val1 != val2:
                    raise ValueError("Invalid value")
                eps = search_scalar_all(np.array(val1), np.array(pdf1), np.array(pdf2))

            print("eps=", eps)
        print("eps_est")

def calc_pdf_by_sampling(var1, var2, input_comb):
    """
    サンプリングにより確率密度を計算し、OPmfに格納して返す
    """
    input_val_list1 = input_comb[0]
    input_val_list2 = input_comb[1]
    def _calc_pdf_by_sampling_rec(var1: Pmf, var2: Pmf): # var1とvar2はLaplaceの確率密度以外は同じことを前提とする
        if isinstance(var1, Laplace):
            if isinstance(var1.child[0], int | float):
                return prng.laplace(var1.child[0], var1.b), prng.laplace(var2.child[0], var2.b)
            elif isinstance(var1.child[0], ArrayItem):
                var1.child[0] = input_val_list1[var1.child[0].ind]
                var2.child[0] = input_val_list2[var2.child[0].ind]
                return prng.laplace(var1.child[0], var1.b), prng.laplace(var2.child[0], var2.b)
            else:
                raise ValueError("Invalid value")
        elif isinstance(var1, ArrayItem):
            return input_val_list1[var1.ind], input_val_list2[var2.ind]
        updated_child1, updated_child2 = [], []
        for child1, child2 in zip(var1.child, var2.child):
            c1, c2 = _calc_pdf_by_sampling_rec(child1, child2)
            updated_child1.append(c1)
            updated_child2.append(c2)
        return var1.func(updated_child1), var2.func(updated_child2)

    # 最小値と最大値を知るためのサンプリング
    # 出力集合は片方の入力セットから作っても問題ないので、var1の出力を使う
    samples = np.array([np.array(_calc_pdf_by_sampling_rec(var1, var2)[0]) for _ in range(100)])
    max_vals = np.max(samples, axis=0) # それぞれの列の最大値が格納される
    min_vals = np.min(samples, axis=0)
    hist_range = []
    for i in range(len(max_vals)):
        hist_range.append((min_vals[i], max_vals[i]))
    outputs1, outputs2 = [], []
    for _ in range(SAMPLING_NUM):
        output1, output2 = _calc_pdf_by_sampling_rec(var1, var2)
        outputs1.append(output1)
        outputs2.append(output2)
    outputs1 = np.asarray(outputs1)
    outputs2 = np.asarray(outputs2)
    hist1, edges1 = np.histogramdd(outputs1, bins=GRID_NUM, range=hist_range)
    hist2, edges2 = np.histogramdd(outputs2, bins=GRID_NUM, range=hist_range)
    return HistPmf(hist1), HistPmf(hist2)

class HistPmf(Pmf):
    # histは二次元配列で、要素があるところに1が立つ
    def __init__(self, hist):
        self.hist = hist


# TODO: Pmfクラスの名前を変えて、こちらに採用する
class OPmf(Pmf):
    def __init__(self, val_to_prob=None):
        super().__init__(val_to_prob=val_to_prob)
        self.name = "OPmf"

    def __call__(self):
        return prng.laplace(0, 1)

class Laplace(Pmf):
    def __init__(self, mu: float | ArrayItem, b: float):
        super().__init__()
        self.child = [mu]
        self.name = f"Laplace({mu.name if isinstance(mu, ArrayItem) else mu}, {b})"
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
        
    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return args[0] + args[1]


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
    
    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return max(args)
    
class Comp(Pmf):
    """
    二つの確率変数を引数に取り、大きかったらTrueを小さかったらFalseを返す
    引数に定数を取るとしたら、第二引数に定数を取るようにする
    e.g. Comp(pmf, const) or Comp(pmf1, pmf2)
    """
    def __init__(self, var1: Pmf, var2: Pmf | int | float):
        super().__init__()
        self.child = [var1, var2]
        self.name = f"Comp({var1.name}, {var2.name})"
        self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
        self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))

    def _calc_pdf(self, children):
        assert len(children) == 2
        child1, child2 = children

        # 片方がArrayItemという定数
        if isinstance(child1, int | float) or isinstance(child2, int | float):
            const, pmf = child1, child2 if isinstance(child1, int | float) else child2, child1
            val = list(pmf.val_to_prob.keys())
            pdf = list(pmf.val_to_prob.values())
            f = interpolate.CubicSpline(val, pdf, bc_type='natural', extrapolate=True)
            lower_pdf = quad(f, -INF, const)[0]
            upper_pr = 1 - lower_pdf # quad(f, const, INF)[0]という計算にした方が良いかもしれない
            assert upper_pr >= 0
            return OPmf({True: upper_pr, False: lower_pdf})
        else: # どちらも確率変数
            pmf1, pmf2 = child1, child2
            val1 = list(pmf1.val_to_prob.keys())
            pdf1 = list(pmf1.val_to_prob.values())
            val2 = list(pmf2.val_to_prob.keys())
            pdf2 = list(pmf2.val_to_prob.values())
            output_vals = np.unique(np.concatenate([val1, val2]))

            f1 = interpolate.CubicSpline(val1, pdf1, bc_type='natural', extrapolate=True)
            f2 = interpolate.CubicSpline(val2, pdf2, bc_type='natural', extrapolate=True)
            true_pdf = 0
            false_pdf = 0
            for v in output_vals:
                true_pdf += f1(v) * quad(f2, -INF, v)[0]
                false_pdf += f2(v) * quad(f1, -INF, v)[0]
            return OPmf({True: true_pdf, False: false_pdf})

    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return args[0] > args[1]

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
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "ToArray"

    def calc_pdf(self, children):
        # 2次元配列を作成
        val_arr_2d = []
        pdf_arr_2d =[]
        for child in children:
            val_arr_2d.append(list(child.val_to_prob.keys()))
            pdf_arr_2d.append(list(child.val_to_prob.values()))
        val_to_pdf = {}
        for val, pdf in zip(product(*val_arr_2d), product(*pdf_arr_2d)):
            val_to_pdf[val] = np.prod(pdf)
        return OPmf(val_to_pdf)

    
    def func(self, args):
        """
        子をまとめて配列として返す
        """
        return args

class ArrayItem(Pmf):
    def __init__(self, ind: int, parent: Array):
        super().__init__()
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

def raw_extract(arr: Array):
    """
    配列要素それぞれにラプラスノイズを加えて取り出す
    """
    lap_list = []
    for val in arr:
        if isinstance(val, ArrayItem):
            lap_list.append(val)
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
if __name__ == "__main__":
    eps = 0.1
    sens = 1
    # 配列要素それぞれにラプラスノイズを加えて取り出す
    Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)
    # Y = Max(Max(Max(Max(Lap1, Lap2), Lap3), Lap4), Lap5)
    # # Y = Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5)
    # Y = ToArray(Lap1, Add(Lap1, Lap2), Add(Add(Lap1, Lap2), Lap3), Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5))
    Arr = Array(5)
    Lap = Laplace(0, 1/eps)
    q1, q2, q3, q4, q5 = raw_extract(Array(5))
    Y = ToArray(Comp(Lap, q1), Comp(Lap, q2), Comp(Lap, q3), Comp(Lap, q4), Comp(Lap, q5))
    # このアルゴリズムで推定されたεの値が出力される


    print("------")
    eps = Y.eps_est()
    print("----------")

""""""
