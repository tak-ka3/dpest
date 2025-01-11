import numpy as np
from collections import Counter
from . import Pmf
from dpest.operation import Case, Br
from dpest.distrib import Laplace, Exp, ArrayItem, HistPmf, RawPmf, Uni
from dpest.input import InputScalarToArrayItem
from dpest.config import ConfigManager, prng
from typing import Union

def input_analysis_rec(var, size, adj):
    if isinstance(var, (Uni, int, float, np.float64, np.int64, int, str)):
        return size, adj
    if isinstance(var, ArrayItem):
        if size == 0:
            size = var.arr_size
        if adj == "":
            adj = var.adj
        return size, adj
    if isinstance(var, InputScalarToArrayItem):
        if size == 0:
            size = 1
        return size, adj
    for child in var.child:
        size, adj = input_analysis_rec(child, size, adj)
    return size, adj

def insert_input_rec(var1, var2, input_val_list1, input_val_list2):
    if isinstance(var1, (Laplace, Exp)):
        assert len(var1.child) == 1
        if isinstance(var1.child[0], (int, float)):
            return var1, var2
        elif isinstance(var1.child[0], ArrayItem):
            var1.child[0] = input_val_list1[var1.child[0].ind]
            var2.child[0] = input_val_list2[var2.child[0].ind]
            vals = np.linspace(var1.lower, var1.upper, var1.num)
            probs1 = var1.calc_strict_pdf(vals)
            probs2 = var2.calc_strict_pdf(vals)
            var1.val_to_prob = dict(zip(vals, probs1))
            var2.val_to_prob = dict(zip(vals, probs2))
            return var1, var2
    elif isinstance(var1, ArrayItem):
        var1 = input_val_list1[var1.ind]
        var2 = input_val_list2[var2.ind]
        return var1, var2
    elif isinstance(var1, Case):
        # case_dictは辞書型だが、inputの値を取得する可能性もあるため、ここで処理する
        for case_input, case_output in var1.child[1].case_dict.items():
            if isinstance(case_output, InputScalarToArrayItem):
                case_output.set_parent_array(input_val_list1[0])
                updated_val = case_output.get_array_item()
                var1.child[1].case_dict[case_input] = updated_val
        # CaseDictインスタンスの更新
        var1.child[1].update_case_dict(var1.child[1].case_dict)

        for case_input, case_output in var2.child[1].case_dict.items():
            if isinstance(case_output, InputScalarToArrayItem):
                case_output.set_parent_array(input_val_list2[0])
                updated_val = case_output.get_array_item()
                var2.child[1].case_dict[case_input] = updated_val
        var2.child[1].update_case_dict(var2.child[1].case_dict)
    elif isinstance(var1, Br):
        for i in range(len(var1.child)):
            if isinstance(var1.child[i], InputScalarToArrayItem):
                var1.child[i].set_parent_array(input_val_list1[0])
                updated_val1 = var1.child[i].get_array_item()
                var1.child[i] = updated_val1
            if isinstance(var2.child[i], InputScalarToArrayItem):
                var2.child[i].set_parent_array(input_val_list2[0])
                updated_val2 = var2.child[i].get_array_item()
                var2.child[i] = updated_val2
    elif not isinstance(var1, Pmf):
        return var1, var2
    updated_child1, updated_child2 = [], []
    for child1, child2 in zip(var1.child, var2.child):
        c1, c2 = insert_input_rec(child1, child2, input_val_list1, input_val_list2)
        updated_child1.append(c1)
        updated_child2.append(c2)
    var1.child = updated_child1
    var2.child = updated_child2
    return var1, var2

def resolve_dependency_rec(var1, var2): # var1とvar2はLaplaceの確率密度以外は同じことを前提とする
    if isinstance(var1, (Laplace, Exp, RawPmf, HistPmf, Uni, np.float64, np.int64, int, str)):
        return var1, var2
    if var1.is_args_depend:
        # ここでサンプリングにより確率密度を計算する
        output_var1, output_var2 = calc_pdf_by_sampling(var1, var2)
        return output_var1, output_var2
    updated_child1, updated_child2 = [], []
    for child1, child2 in zip(var1.child, var2.child):
        c1, c2 = resolve_dependency_rec(child1, child2)
        updated_child1.append(c1)
        updated_child2.append(c2)
    var1.child = updated_child1
    var2.child = updated_child2
    return var1, var2

def calc_pdf_rec(var):
    if isinstance(var, (Laplace, Exp, Uni, RawPmf, HistPmf, np.float64, np.int64, int)):
        return var
    output_var = var.calc_pdf([calc_pdf_rec(child) for child in var.child])
    return output_var

import numpy as np
import multiprocessing
from collections import Counter

# 以下のクラス・関数は、環境に応じてインポート・定義してください
# - Laplace, Exp, Uni, HistPmf, ConfigManager, prng
#  ここでは並列サンプリング部分の再構成にフォーカスしています。

def _calc_pdf_by_sampling_rec(v1, v2, memo):
    """
    再帰的に v1, v2 をサンプリングし、それぞれの値を返す。
    memo は一度サンプリングした変数の値をキャッシュする辞書。
    """
    # すでにキャッシュされていればそれを返す
    if v1 in memo and v2 in memo:
        return memo[v1], memo[v2]

    # スカラーや文字列の場合
    if isinstance(v1, (float, int, np.float64, np.int64, str)):
        memo[v1] = v1
        memo[v2] = v2
        return v1, v2

    # Laplace や Exp の場合（子が int/float である場合のみ）
    if isinstance(v1, (Laplace, Exp)):
        if isinstance(v1.child[0], (int, float)):
            sample1 = v1.sampling()
            sample2 = v2.sampling()
            memo[v1] = sample1
            memo[v2] = sample2
            return sample1, sample2
        else:
            raise ValueError("Invalid value in Laplace or Exp.")

    # Uni の場合
    if isinstance(v1, Uni):
        sample1 = prng.integers(v1.lower, v1.upper)
        sample2 = prng.integers(v2.lower, v2.upper)
        memo[v1] = sample1
        memo[v2] = sample2
        return sample1, sample2

    # 再帰処理: v1, v2 の子要素をそれぞれサンプリングして関数適用
    updated_child1, updated_child2 = [], []
    for c1, c2 in zip(v1.child, v2.child):
        out1, out2 = _calc_pdf_by_sampling_rec(c1, c2, memo)
        updated_child1.append(out1)
        updated_child2.append(out2)

    result1 = v1.func(updated_child1)
    result2 = v2.func(updated_child2)
    memo[v1] = result1
    memo[v2] = result2
    return result1, result2


def _sample_once(args):
    """
    var1, var2 を 1 回サンプリングして (output1, output2) を返す関数。
    multiprocessing.Pool.map で呼び出せるように、引数は (var1, var2) のタプルにする。
    """
    var1, var2 = args
    memo = {}
    return _calc_pdf_by_sampling_rec(var1, var2, memo)


def _gather_samples(var1, var2, n_samples):
    """
    var1, var2 を n_samples 回サンプリングし、その結果のリストを返す。
    multiprocessing の Pool.map を使って並列化している。
    """
    # (var1, var2) を n_samples 回分作って pool.map に渡す
    args_list = [(var1, var2)] * n_samples
    with multiprocessing.Pool() as pool:
        results = pool.map(_sample_once, args_list)

    # results は [(o1, o2), (o1, o2), ...] の形
    outputs1 = [r[0] for r in results]
    outputs2 = [r[1] for r in results]
    return outputs1, outputs2


def calc_pdf_by_sampling(var1, var2):
    """
    var1, var2 をサンプリングして確率密度（HistPmf）を計算し、返す関数。
    """
    SAMPLING_NUM = ConfigManager.get("SAMPLING_NUM")
    GRID_NUM     = ConfigManager.get("GRID_NUM")

    # まず1サンプルをとって出力の型を確認
    one_sample, _ = _sample_once((var1, var2))
    output_type = type(one_sample)

    #----------------------------------------
    # case 1: スカラー (int, float) の場合
    #----------------------------------------
    if isinstance(one_sample, (int, np.int64, float, np.float64)):
        # test_samples回サンプリングして最小値と最大値を推定
        test_samples = 200
        test_out1, _ = _gather_samples(var1, var2, test_samples)
        test_out1 = np.asarray(test_out1, dtype=output_type)

        min_vals, max_vals = np.min(test_out1), np.max(test_out1)

        # 本番サンプリング
        outputs1, outputs2 = _gather_samples(var1, var2, SAMPLING_NUM)
        outputs1 = np.asarray(outputs1)
        outputs2 = np.asarray(outputs2)

        if isinstance(one_sample, (int, np.int64)):
            # 整数の場合はヒストグラムのビンを [min, ..., max+1] とする
            hist_range = np.arange(min_vals, max_vals + 2)
            hist1, edges1 = np.histogram(outputs1, bins=hist_range)
            hist2, edges2 = np.histogram(outputs2, bins=hist_range)
        else:
            # float の場合
            hist_range = (min_vals, max_vals)
            hist1, edges1 = np.histogram(outputs1, bins=GRID_NUM, range=hist_range)
            hist2, edges2 = np.histogram(outputs2, bins=GRID_NUM, range=hist_range)

        dict_hist1 = {edges1[i]: hist1[i] for i in range(len(hist1))}
        dict_hist2 = {edges2[i]: hist2[i] for i in range(len(hist2))}
        return HistPmf(dict_hist1), HistPmf(dict_hist2)

    #----------------------------------------
    # case 2: 配列やリストの場合
    #----------------------------------------
    elif isinstance(one_sample, (np.ndarray, list)):
        test_samples = 200
        # 配列要素に int, float, bool, nan などが混在する可能性があるのでチェック
        samples_obj, _ = _gather_samples(var1, var2, test_samples)

        # 最小・最大を推定するための変数
        float_max = -np.inf
        float_min = np.inf
        type_set  = set()

        # 配列の各要素をチェック
        for arr in samples_obj:
            for val in arr:
                if np.isnan(val):
                    type_set.add(np.nan)
                elif isinstance(val, bool):
                    type_set.add(bool)
                elif isinstance(val, (int, np.int64)):
                    type_set.add(int)
                elif isinstance(val, (float, np.float64)):
                    float_max = max(float_max, val)
                    float_min = min(float_min, val)
                    type_set.add(float)
                else:
                    raise ValueError("Invalid type inside array or list.")

        # float を含むならグリッドを作成
        float_grid, float_grid_labels = [], []
        GRID_NUM = int(GRID_NUM)  # 念のため
        if float in type_set:
            float_grid = np.linspace(float_min, float_max, GRID_NUM)
            float_grid_labels = [
                f"[{float_grid[i]:.2f}, {float_grid[i+1]:.2f})"
                for i in range(len(float_grid) - 1)
            ]

        # 本番サンプリング
        outputs1, outputs2 = _gather_samples(var1, var2, SAMPLING_NUM)

        # それぞれの要素をbin化 or NaNなどに応じて処理
        def process_array(arr):
            processed = []
            for val in arr:
                if np.isnan(val):
                    processed.append(val)  # そのまま NaN として
                elif isinstance(val, float):
                    # float_grid 上のビンを探す
                    bin_index = np.digitize(val, float_grid, right=False) - 1
                    bin_index = max(0, min(bin_index, len(float_grid_labels) - 1))
                    processed.append(float_grid_labels[bin_index])
                else:
                    # int, bool, などそのまま
                    processed.append(val)
            return tuple(processed)

        outputs1_processed = [process_array(o) for o in outputs1]
        outputs2_processed = [process_array(o) for o in outputs2]

        # カウントして共通キーのみ残す
        counts_dict1 = Counter(outputs1_processed)
        counts_dict2 = Counter(outputs2_processed)
        common_keys = set(counts_dict1.keys()) & set(counts_dict2.keys())

        filtered_counts_dict1 = {k: counts_dict1[k] for k in common_keys}
        filtered_counts_dict2 = {k: counts_dict2[k] for k in common_keys}

        return HistPmf(filtered_counts_dict1), HistPmf(filtered_counts_dict2)

    else:
        raise ValueError("Output type is invalid or unsupported.")