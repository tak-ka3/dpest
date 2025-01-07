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

def calc_pdf_by_sampling(var1, var2):
    """
    サンプリングにより確率密度を計算し、RawPmfに格納して返す
    """
    SAMPLING_NUM = ConfigManager.get("SAMPLING_NUM")
    GRID_NUM = ConfigManager.get("GRID_NUM")
    # sampling_val_memoはサンプリングした値を記録するための辞書
    def _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo: dict): # var1とvar2はLaplaceの確率密度以外は同じことを前提とする
        if isinstance(var1, (float, int, np.float64, np.int64, str)):
            return var1, var2
        elif sampling_val_memo.get(var1) is not None:
            return sampling_val_memo[var1], sampling_val_memo[var2]
        elif isinstance(var1, (Laplace, Exp)):
            if isinstance(var1.child[0], (int, float)):
                lap_sample1, lap_sample2 = var1.sampling(), var2.sampling()
                sampling_val_memo[var1], sampling_val_memo[var2] = lap_sample1, lap_sample2
                return lap_sample1, lap_sample2
            else:
                raise ValueError("Invalid value")
        elif isinstance(var1, Uni):
            uni_sample1, uni_sample2 = prng.integers(var1.lower, var1.upper), prng.integers(var2.lower, var2.upper)
            sampling_val_memo[var1], sampling_val_memo[var2] = uni_sample1, uni_sample2
            return uni_sample1, uni_sample2
        updated_child1, updated_child2 = [], []
        for child1, child2 in zip(var1.child, var2.child):
            c1, c2 = _calc_pdf_by_sampling_rec(child1, child2, sampling_val_memo)
            updated_child1.append(c1)
            updated_child2.append(c2)
        return var1.func(updated_child1), var2.func(updated_child2)

    # 最小値と最大値を知るためのサンプリング
    # 出力集合は片方の入力セットから作っても問題ないので、var1の出力を使う
    sampling_val_memo = {}
    one_sample = _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo)[0]
    output_type = type(one_sample)

    # εを求めるためのサンプリング
    if isinstance(one_sample, (int, np.int64, float, np.float64)):
        # 最小値と最大値を求めるためのサンプリング
        test_samples = 200
        samples = np.empty(test_samples, dtype=output_type)
        for i in range(test_samples):
            sampling_val_memo = {}
            samples[i] = _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo)[0]
        max_vals = np.max(samples)
        min_vals = np.min(samples)
        outputs1, outputs2 = [], []
        for _ in range(SAMPLING_NUM):
            sampling_val_memo = {}
            output1, output2 = _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo)
            outputs1.append(output1)
            outputs2.append(output2)
        outputs1 = np.asarray(outputs1)
        outputs2 = np.asarray(outputs2)
        # 出力が整数値の場合は、それぞれ整数値ごとにヒストグラムを作成する
        if isinstance(samples[0], (np.int64, int)):
            # 整数値ごとのヒストグラムを作る際には、このように+2する必要がある
            hist_range = np.arange(min_vals, max_vals + 2)
            hist1, edges1 = np.histogram(outputs1, bins=hist_range)
            hist2, edges2 = np.histogram(outputs2, bins=hist_range)
        else:
            hist_range = (min_vals, max_vals)
            hist1, edges1 = np.histogram(outputs1, bins=GRID_NUM, range=hist_range)
            hist2, edges2 = np.histogram(outputs2, bins=GRID_NUM, range=hist_range)
        dict_hist1 = {edges1[i]: hist1[i] for i in range(len(hist1))}
        dict_hist2 = {edges2[i]: hist2[i] for i in range(len(hist2))}
        return HistPmf(dict_hist1), HistPmf(dict_hist2)
    elif isinstance(one_sample, (np.ndarray, list)):
        # 最小値と最大値を求めるためのサンプリング
        test_samples = 200
        samples = np.empty(test_samples, dtype=object)
        float_max = -np.inf
        float_min = np.inf
        type_set = set()
        for i in range(test_samples):
            sampling_val_memo = {}
            samples[i] = _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo)[0]
            for scalar  in samples[i]:
                if np.isnan(scalar):
                    type_set.add(np.nan)
                elif isinstance(scalar, bool):
                    type_set.add(bool)
                elif isinstance(scalar, (int, np.int64)):
                    type_set.add(int)
                elif isinstance(scalar, (float, np.float64)):
                    float_max = max(float_max, scalar)
                    float_min = min(float_min, scalar)
                    type_set.add(float)
                elif isinstance(scalar, np.nan):
                    type_set.add(np.nan)
                else:
                    raise ValueError("Invalid type")

        # floatを含む場合は、グリッドを作成する
        float_grid, float_grid_labels = [], []
        if float in type_set:
            float_grid = np.linspace(float_min, float_max, GRID_NUM)
            float_grid_labels = [f"[{float_grid[i]:.2f}, {float_grid[i + 1]:.2f})" for i in range(len(float_grid) - 1)]

        outputs1, outputs2 = [], []
        for _ in range(SAMPLING_NUM):
            sampling_val_memo = {}
            output1, output2 = _calc_pdf_by_sampling_rec(var1, var2, sampling_val_memo)
            processed_output1, processed_output2 = [], []
            for val in output1:
                if np.isnan(val):
                    processed_output1.append(val)
                elif isinstance(val, float):
                    bin_index = np.digitize(val, float_grid, right=False) - 1
                    bin_index = max(0, min(bin_index, len(float_grid_labels) - 1))
                    processed_output1.append(float_grid_labels[bin_index])
                else:
                    processed_output1.append(val)
            for val in output2:
                # np.nanはnp.float型であるため、np.nanを含む場合はこちらで処理する
                if np.isnan(val):
                    processed_output2.append(val)
                elif isinstance(val, float):
                    bin_index = np.digitize(val, float_grid, right=False) - 1
                    bin_index = max(0, min(bin_index, len(float_grid_labels) - 1))
                    processed_output2.append(float_grid_labels[bin_index])
                else:
                    processed_output2.append(val)
            outputs1.append(tuple(processed_output1))
            outputs2.append(tuple(processed_output2))
        counts_dict1 = Counter(outputs1)
        counts_dict2 = Counter(outputs2)
        commonkeys = set(counts_dict1.keys()) & set(counts_dict2.keys())
        filtered_counts_dict1 = {k: counts_dict1[k] for k in commonkeys}
        filtered_counts_dict2 = {k: counts_dict2[k] for k in commonkeys}
        return HistPmf(filtered_counts_dict1), HistPmf(filtered_counts_dict2)
    else:
        raise ValueError("output_type is invalid")
