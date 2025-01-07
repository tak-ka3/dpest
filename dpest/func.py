import copy
import numpy as np
from . import Pmf
from dpest.input_generator import input_generator
from dpest.run_graph import input_analysis_rec, insert_input_rec, resolve_dependency_rec, calc_pdf_rec
from dpest.search import search_scalar_all, search_hist
from dpest.distrib import HistPmf

def input_analysis(Y: Pmf):
    """
    計算グラフを走査して、入力の配列のサイズ、隣接性の定義を取得
    """
    return input_analysis_rec(Y, 0, "")

def insert_input(Y: Pmf, input_comb: tuple):
    """
    計算グラフを走査して隣接した入力をそれぞれ挿入する
    """
    Y1 = copy.deepcopy(Y)
    Y2 = copy.deepcopy(Y)
    input_val_list1 = input_comb[0]
    input_val_list2 = input_comb[1]
    return insert_input_rec(Y1, Y2, input_val_list1, input_val_list2)

def resolve_dependency(Y1: Pmf, Y2: Pmf):
    """
    計算グラフを走査して、依存関係を調べる
    Operationの引数に依存関係を見つけたら、その時点でサンプリングにより確率密度を計算する
    TODO: 一つ上の階層の関数がいらないかもしれない
    """

    calc_graph1, calc_graph2 = resolve_dependency_rec(Y1, Y2)
    return calc_graph1, calc_graph2

def calc_pdf(calc_graph):
    """
    依存関係がないcalc_graphを最も子供から順に確率密度を解析的に計算する
    Returns: 
        var: 最終的に得られた確率変数。確率密度もプロパティに含む。
    """
    
    return calc_pdf_rec(calc_graph)

def eps_est(Y: Pmf):
    """
    計算グラフを辿って、Operationの依存関係を調べる
    依存関係は最も子供から調べる。子が依存する確率変数は親も依存するという関係
    """
    # TODO: infか1かの隣接性はプログラマが指定できるようにする
    # 計算グラフをたどり、入力の配列のサイズ、隣接性の定義を取得
    input_size, adj = input_analysis(Y)
    input_list = input_generator(adj, input_size)
    max_eps = 0
    for input_set in input_list:
        calc_graph1, calc_graph2 = insert_input(Y, input_set)
        calc_graph1, calc_graph2 = resolve_dependency(calc_graph1, calc_graph2)
        Y1 = calc_pdf(calc_graph1)
        Y2 = calc_pdf(calc_graph2)

        if isinstance(Y1, HistPmf):
            eps = search_hist(Y1.hist, Y2.hist)
        else:
            val1, pdf1 = list(Y1.val_to_prob.keys()), list(Y1.val_to_prob.values())
            val2, pdf2 = list(Y2.val_to_prob.keys()), list(Y2.val_to_prob.values())
            if val1 != val2:
                raise ValueError("Invalid value")
            eps = search_scalar_all(np.array(val1), np.array(pdf1), np.array(pdf2))

        # print("eps=", eps)
        if max_eps < eps:
            max_eps = eps
    # print("- estimated eps:", max_eps)
    return max_eps
