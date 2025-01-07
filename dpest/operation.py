# 型ヒントを遅延評価できるようになる
from __future__ import annotations
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from itertools import product, combinations
from utils.pr_calc import nonuniform_convolution
from . import Pmf
from dpest.config import ConfigManager, prng
from dpest.distrib import RawPmf
from typing import Union

class Add(Pmf):
    """
    二つの確率変数同士を足すクラス
    """
    def __init__(self, var1: Pmf, var2: Pmf):
        assert isinstance(var1, Pmf) & isinstance(var2, (Pmf, int, float))
        super().__init__()
        self.child = [var1, var2]
        if isinstance(var2, Pmf):
            self.name = f"Add({var1.name}, {var2.name})"
            self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
            self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))
        elif isinstance(var2, (int, float)):
            self.name = f"Add({var1.name}, {var2})"
            self.is_args_depend = False
            self.depend_vars = var1.depend_vars
        else:
            raise ValueError("var2 is an invalid type")

    def __call__(self, var1: Pmf, var2: Pmf):
        return var1() + var2()
    
    def calc_pdf(self, child):
        """
        引数から出力の確率密度を厳密に計算する
        """
        if isinstance(child[1], (int, float)):
            val1, pdf1 = list(child[0].val_to_prob.keys()), list(child[0].val_to_prob.values())
            output_val1 = [v + child[1] for v in val1]
            return RawPmf(dict(zip(output_val1, pdf1)))

        # 二つの確率変数の和の確率密度を計算する
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
            return RawPmf(dict(zip(output_val_range, output_pdf)))
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
            return RawPmf(dict(zip(output_val_range, output_pdf)))
        
    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return args[0] + args[1]

class Mul(Pmf):
    """
    一つの確率変数に定数をかける
    """
    def __init__(self, var: Pmf, const: Union[int, float]):
        assert isinstance(var, Pmf) & isinstance(const, (int, float))
        super().__init__()
        self.child = [var]
        self.name = f"Mul({var.name}, {const})"
        self.is_args_depend = False
        self.depend_vars = var.depend_vars
    
    def __call__(self, var: Pmf, const: Union[int, float]):
        return var() * const
    
    def calc_pdf(self, child):
        """
        引数から出力の確率密度を厳密に計算する
        """
        val = list(child[0].val_to_prob.keys())
        pdf = list(child[0].val_to_prob.values())
        output_vals = [v * child[1] for v in val]
        return RawPmf(dict(zip(output_vals, pdf)))

class Max(Pmf):
    """
    二つの確率変数を比較し、大きい方を返す
    """
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
        INF = ConfigManager.get("INF")
        for v in output_vals:
            output_pdf.append(f1(v) * quad(f2, -INF, v)[0] + f2(v) * quad(f1, -INF, v)[0])
        return RawPmf(dict(zip(output_vals, output_pdf)))
    
    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return max(args)
    
class Br(Pmf):
    """
    二つの確率変数または、一つの確率変数と定数を引数に取り、その大小関係に応じて、確率変数や定数を返す
    Compクラスを包含するクラス
    """
    def __init__(self, input_var1: Pmf, input_var2: Union[Pmf, int, float], output_var1: Union[Pmf, int, float], output_var2: Union[Pmf, int, float]):
        # assert isinstance(input_var1, Pmf) & isinstance(input_var2, Pmf | int | float)
        super().__init__()
        self.child = [input_var1, input_var2, output_var1, output_var2]
        input_var1_name, input_var1_depend_vars = (input_var1.name, input_var1.depend_vars) if isinstance(input_var1, Pmf) else (input_var1, [])
        input_var2_name, input_var2_depend_vars = (input_var2.name, input_var2.depend_vars) if isinstance(input_var2, Pmf) else (input_var2, [])
        output_var1_name, output_var1_depend_vars = (output_var1.name, output_var1.depend_vars) if isinstance(output_var1, Pmf) else (output_var1, [])
        output_var2_name, output_var2_depend_vars = (output_var2.name, output_var2.depend_vars) if isinstance(output_var2, Pmf) else (output_var2, [])
        self.name = f"Br(input:({input_var1_name}, {input_var2_name}), output: ({output_var1_name}, {output_var2_name}))"
        sets = [set(input_var1_depend_vars), set(input_var2_depend_vars), set(output_var1_depend_vars), set(output_var2_depend_vars)]
        self.is_args_depend = False
        for set1, set2 in combinations(sets, 2):
            if set1 & set2:
                self.is_args_depend = True
        self.depend_vars = list(set(input_var1_depend_vars) | set(input_var2_depend_vars) | set(output_var1_depend_vars) | set(output_var2_depend_vars))

    def calc_pdf(self, children):
        assert len(children) == 4
        input_var1, input_var2, output_var1, output_var2 = children
        assert isinstance(input_var1, Pmf) & isinstance(input_var2, (Pmf, int, float)) & isinstance(output_var1, (Pmf, int, float)) & isinstance(output_var2, (Pmf, int, float))
        raise NotImplementedError # Brを用いて解析的に計算するアルゴリズムがないので未実装
    
    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 4
        input_var1, input_var2, output_var1, output_var2 = args
        return output_var1 if input_var1 >= input_var2 else output_var2


class Comp(Pmf):
    """
    二つの確率変数を引数に取り、大きかったらTrueを小さかったらFalseを返す
    引数に定数を取るとしたら、第二引数に定数を取るようにする
    e.g. Comp(pmf, const) or Comp(pmf1, pmf2)
    """
    def __init__(self, var1: Pmf, var2: Union[Pmf, int, float]):
        assert isinstance(var1, Pmf) & isinstance(var2, (Pmf, int, float))
        super().__init__()
        self.child = [var1, var2]
        self.name = f"Comp({var1.name}, {var2.name})"
        self.is_args_depend = len(set(var1.depend_vars) & set(var2.depend_vars)) > 0
        self.depend_vars = list(set(var1.depend_vars) | set(var2.depend_vars))

    def calc_pdf(self, children):
        assert len(children) == 2
        child1, child2 = children
        INF = ConfigManager.get("INF")

        # 片方がArrayItemという定数
        if isinstance(child1, (int, float)) or isinstance(child2, (int, float)):
            const, pmf = child1, child2 if isinstance(child1, (int, float)) else child2, child1
            val = list(pmf.val_to_prob.keys())
            pdf = list(pmf.val_to_prob.values())
            f = interpolate.CubicSpline(val, pdf, bc_type='natural', extrapolate=True)
            lower_pdf = quad(f, -INF, const)[0]
            upper_pr = 1 - lower_pdf # quad(f, const, INF)[0]という計算にした方が良いかもしれない
            assert upper_pr >= 0
            return RawPmf({True: upper_pr, False: lower_pdf})
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
            return RawPmf({True: true_pdf, False: false_pdf})

    def func(self, args: list):
        """
        通常の関数としての振る舞い
        """
        assert len(args) == 2
        return args[0] > args[1]

"""
確率変数を1つ引数に取り、特定の定数に等しい場合に、特定の値や確率変数を返す
"""
class Case(Pmf):
    def __new__(cls, var: Union[Pmf, int, float] = None, case_dict: dict = None):
        if isinstance(var, (int, float)):
            if case_dict.get(var) is None:
                if case_dict.get("otherwise") is None:
                    raise ValueError("Invalid Case")
                return case_dict["otherwise"]
            return case_dict[var]
        return super().__new__(cls)
    def __init__(self, var: Union[Pmf, int, float], case_dict: dict):
        assert isinstance(var, Pmf)
        super().__init__()
        case_dict_instance = CaseDict(case_dict)
        self.name = "Case"
        self.child = [var, case_dict_instance]
        self.is_args_depend = False
        self.depend_vars = var.depend_vars
    
    def calc_pdf(self, children):
        assert len(children) == 2
        assert isinstance(children[0], Pmf) & isinstance(children[1], CaseDict)
        child = children[0]
        output_dict = {}
        sum_case_val_prob = 0
        # TODO: self.child[1].case_dict.items()に含まれるものがPmfか定数かなどの場合に応じて、計算方法が変わる
        if self.child[1].arg_pmf_num == 0:
            for case_val, case_var in self.child[1].case_dict.items():
                if case_val == "otherwise":
                    # otherwiseは最後に来る必要がある
                    assert list(self.child[1].case_dict.keys())[-1] == "otherwise"
                    case_val_prob = 1 - sum(output_dict.values())
                else:
                    case_val_prob = child.val_to_prob[case_val]
                sum_case_val_prob += case_val_prob
                assert sum_case_val_prob <= 1
                if output_dict.get(case_var) is None:
                    output_dict[case_var] = case_val_prob
                else:
                    output_dict[case_var] += case_val_prob
        elif self.child[1].arg_pmf_num == 1 or self.child[1].arg_pmf_num == 2: # case_dictのvalueの方に1or2個Pmfが含まれている
            for case_val, case_var in self.child[1].case_dict.items():
                if case_val == "otherwise":
                    # otherwiseは最後に来る必要がある
                    assert list(self.child[1].case_dict.keys())[-1] == "otherwise"
                    case_val_prob = 1 - sum(output_dict.values())
                else:
                    assert isinstance(case_val, (int, float, np.int64, np.float64))
                    case_val_prob = child.val_to_prob[case_val]
                sum_case_val_prob += case_val_prob
                assert sum_case_val_prob <= 1

                if isinstance(case_var, (int, float)):
                    if output_dict.get(case_var) is None:
                        output_dict[case_var] = case_val_prob
                    else:
                        output_dict[case_var] += case_val_prob
                elif isinstance(case_var, Pmf):
                    val_to_prob = case_var.val_to_prob
                    for val, prob in val_to_prob.items():
                        if output_dict.get(val) is None:
                            output_dict[val] = prob * case_val_prob
                        else:
                            output_dict[val] += prob * case_val_prob
                else:
                    raise ValueError("Invalid value")
        else:
            raise NotImplementedError("Invalid CaseDict")
        return RawPmf(output_dict)
    
    def func(self, args: list):
        """
        valに対応するcase_dictの値を返す
        """
        assert len(args) == 2
        val = args[0]
        case_dict = args[1]
        assert isinstance(val, (int, float, np.int64, np.float64)) & isinstance(case_dict, dict)
        if case_dict.get(val) is None:
            if case_dict.get("otherwise") is None:
                raise ValueError("Invalid Case")
            return case_dict["otherwise"]
        return case_dict[val]

class CaseDict(Pmf):
    def __init__(self, case_dict: dict):
        super().__init__()
        self._set_case_dict(case_dict)
        
    def _set_case_dict(self, case_dict: dict):
        self.case_dict = case_dict
        pmf_children = []
        children = []
        for case_var, case_output in case_dict.items():
            if isinstance(case_var, Pmf):
                pmf_children.append(case_var)
            if isinstance(case_output, Pmf):
                pmf_children.append(case_output)
            children.append(case_var)
            children.append(case_output)
        self.child = children
        if len(pmf_children) == 0:
            self.arg_pmf_num = 0
            self.depend_vars = []
            self.is_args_depend = False
        elif len(pmf_children) == 1:
            self.arg_pmf_num = 1
            self.depend_vars = pmf_children[0].depend_vars
            self.is_args_depend = pmf_children[0].is_args_depend
        else:
            self.arg_pmf_num = len(pmf_children)
            sets = [set(pmf.depend_vars) for pmf in pmf_children]
            self.is_args_depend = False
            for set1, set2 in combinations(sets, 2):
                if set1 & set2:
                    self.is_args_depend = True
            self.depend_vars = list(set.union(*sets))
        self.name = f"CaseDict({self.child})"
    
    def update_case_dict(self, case_dict: dict):
        self._set_case_dict(case_dict)

    def calc_pdf(self, _):
        """
        Caseの引数として使われないので、確率密度ではなく、単にCaseDictを返す
        self.childに情報が含まれているので、childrenの方は無視して良い
        """
        return self
    
    def func(self, args: list) -> dict:
        """
        argsには、childに入れた確率変数が入る
        Return:
            Caesクラスのみが引数にとるので、分岐のために辞書型で返したい
        """
        # 配列を辞書型に変換
        case_dict = {}
        for i in range(0, len(args), 2):
            case_dict[args[i]] = args[i+1]
        return case_dict

class ToArray(Pmf):
    """
    引数の要素を一つの配列にまとめる
    """
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
            if val_to_pdf.get(val):
                raise ValueError("Invalid value")
            val_to_pdf[val] = np.prod(pdf)
        return RawPmf(val_to_pdf)
    
    def func(self, args):
        """
        子をまとめて配列として返す
        """
        return args
