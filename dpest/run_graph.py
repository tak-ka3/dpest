import numpy as np
import copy
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
from collections import Counter

# Laplace, Exp, Uni, HistPmf, ConfigManager, prng などの定義や import は適宜行ってください。


def _calc_pdf_by_sampling_rec_vec(v1, v2, n_samples):
    """
    v1, v2 をまとめて n_samples 回サンプリングし、それぞれ形状 (n_samples, ...) の配列(またはリスト)を返す。
    再帰呼び出しをベクトル化した実装。
    """
    #-------------------------------------------------------------
    # 1) スカラーや文字列なら (n_samples,) にブロードキャストする
    #-------------------------------------------------------------
    if isinstance(v1, (bool, np.bool_)):
        arr1 = np.full((n_samples,), v1, dtype=np.bool_)
        arr2 = np.full((n_samples,), v2, dtype=np.bool_)
        # print(type(arr1[0]))
        # print(arr1)
        return arr1, arr2
    if isinstance(v1, (float, int, np.float64, np.int64, str)):
        arr1 = np.full((n_samples,), v1)
        arr2 = np.full((n_samples,), v2)
        return arr1, arr2
    
    if v1.sample is not None:
        return v1.sample, v2.sample

    #-------------------------------------------------------------
    # 2) Laplace, Exp などで「一度の呼び出しで n_samples 個」取得できる場合
    #-------------------------------------------------------------
    if isinstance(v1, (Laplace, Exp)):
        # ここでは v1.sampling(n_samples) が shape = (n_samples,) を返すと仮定
        arr1 = v1.sampling(n_samples)
        arr2 = v2.sampling(n_samples)
        v1.sample, v2.sample = arr1, arr2
        return arr1, arr2

    #-------------------------------------------------------------
    # 3) Uni の場合: まとめて integers(n_samples)
    #-------------------------------------------------------------
    if isinstance(v1, Uni):
        arr1 = prng.integers(v1.lower, v1.upper, size=n_samples)
        arr2 = prng.integers(v2.lower, v2.upper, size=n_samples)
        v1.sample, v2.sample = arr1, arr2
        return arr1, arr2

    #-------------------------------------------------------------
    # 4) 子をもつ場合（木構造・式構造の再帰）
    #    各子からまとめてサンプリングした結果を func() にベクトル演算で適用
    #-------------------------------------------------------------
    child_arrays1 = []
    child_arrays2 = []
    for c1, c2 in zip(v1.child, v2.child):
        subarr1, subarr2 = _calc_pdf_by_sampling_rec_vec(c1, c2, n_samples)
        child_arrays1.append(subarr1)  # 形状 (n_samples,)
        child_arrays2.append(subarr2)

    # ここで child_arrays1, child_arrays2 はリストになっているが、
    # var1.func(var_list) がベクトル演算に対応するように実装しておくか、
    # あるいは何らかの形で要素ごとに処理して (n_samples,) を作り直す。
    # たとえば下記のように書けるかもしれない：
    arr1 = v1.func(child_arrays1)  # ベクトル演算ができる前提
    arr2 = v2.func(child_arrays2)  # 同上
    return arr1, arr2


def calc_pdf_by_sampling(var1, var2):
    """
    一度のサンプリングで n_samples 個まとめて取得できる実装。
    var1.sampling(n_samples) が (n_samples, …) の形状を返すことを仮定している。
    """
    SAMPLING_NUM = ConfigManager.get("SAMPLING_NUM")
    GRID_NUM     = ConfigManager.get("GRID_NUM")

    #------------------------------------------------------
    # まずはテストサンプリング(例: 200) で min/max を推定
    #------------------------------------------------------
    var1_cp = copy.deepcopy(var1)
    var2_cp = copy.deepcopy(var2)
    test_samples = 200
    test_arr1, test_arr2 = _calc_pdf_by_sampling_rec_vec(var1_cp, var2_cp, test_samples)
    # test_arr1, test_arr2 は形状 (test_samples, ...) の配列またはリスト

    # 先頭要素を見て出力タイプを推定 (スカラーか配列か etc.)
    one_sample = test_arr1[0]
    output_type = type(one_sample)

    #------------------------------------------------------
    # case 1: スカラー (int, float) の場合
    #------------------------------------------------------
    if isinstance(one_sample, (int, np.int64, float, np.float64)):
        # スカラーなので test_arr1, test_arr2 は shape = (test_samples,)
        min_vals, max_vals = np.min(test_arr1), np.max(test_arr1)

        # 本番サンプリング
        arr1, arr2 = _calc_pdf_by_sampling_rec_vec(var1, var2, SAMPLING_NUM)
        # shape = (SAMPLING_NUM,)

        if isinstance(one_sample, (int, np.int64)):
            # 整数の場合
            hist_range = np.arange(min_vals, max_vals + 2)
            hist1, edges1 = np.histogram(arr1, bins=hist_range)
            hist2, edges2 = np.histogram(arr2, bins=hist_range)
        else:
            # float の場合
            hist_range = (min_vals, max_vals)
            hist1, edges1 = np.histogram(arr1, bins=GRID_NUM, range=hist_range)
            hist2, edges2 = np.histogram(arr2, bins=GRID_NUM, range=hist_range)

        dict_hist1 = {edges1[i]: hist1[i] for i in range(len(hist1))}
        dict_hist2 = {edges2[i]: hist2[i] for i in range(len(hist2))}
        return HistPmf(dict_hist1), HistPmf(dict_hist2)

    #------------------------------------------------------
    # case 2: 配列(list, np.ndarray) の場合
    #------------------------------------------------------
    elif isinstance(one_sample, (np.ndarray, list)):
        # test_arr1, test_arr2 は (test_samples, array_dim) という2次元以上の可能性もある
        # あるいは each element がリストという可能性もある。
        float_max = -np.inf
        float_min = np.inf
        type_set = set()

        for row in test_arr1:  # row は一つのサンプリング結果(リスト/配列)
            for val in row:
                if np.isnan(val):
                    type_set.add(np.nan)
                elif isinstance(val, (bool, np.bool_)):
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
        GRID_NUM = int(GRID_NUM)
        if float in type_set:
            float_grid = np.linspace(float_min, float_max, GRID_NUM)
            float_grid_labels = [
                f"[{float_grid[i]:.2f}, {float_grid[i+1]:.2f})"
                for i in range(len(float_grid) - 1)
            ]

        # 本番サンプリング
        arr1, arr2 = _calc_pdf_by_sampling_rec_vec(var1, var2, SAMPLING_NUM)

        arr1 = np.array(arr1).T
        arr2 = np.array(arr2).T

        # arr1, arr2 は形状 (SAMPLING_NUM, array_dim) や要素がリストの可能性あり

        def process_rows(arr, float_grid, float_grid_labels):
            # NaNの判定関数を定義
            def is_nan_safe(val):
                return isinstance(val, float) and np.isnan(val)

            # NaNマスク
            nan_mask = np.vectorize(is_nan_safe)(arr)

            # float型の値マスク
            float_mask = ~nan_mask & np.vectorize(lambda x: isinstance(x, float))(arr)

            # ビン分け
            bins = np.digitize(arr[float_mask], float_grid, right=False) - 1
            bins = np.clip(bins, 0, len(float_grid_labels) - 1)

            # 結果の生成
            result = np.empty(arr.shape, dtype=object)  # オブジェクト型で空配列を作成
            result[nan_mask] = None  # NaNの場所をNoneに
            result[float_mask] = np.array(float_grid_labels)[bins]  # ビン分けしたラベルを割り当て

            # 他の値はそのままコピー
            other_mask = ~(nan_mask | float_mask)
            result[other_mask] = arr[other_mask]

            # 行ごとにタプルに変換
            return [tuple(row) for row in result]

        # サンプリング結果すべてを処理
        outputs1_processed = process_rows(arr1, float_grid, float_grid_labels)
        outputs2_processed = process_rows(arr2, float_grid, float_grid_labels)

        counts_dict1 = Counter(outputs1_processed)
        counts_dict2 = Counter(outputs2_processed)

        common_keys = set(counts_dict1.keys()) & set(counts_dict2.keys())

        filtered_counts_dict1 = {k: counts_dict1[k] for k in common_keys}
        filtered_counts_dict2 = {k: counts_dict2[k] for k in common_keys}

        return HistPmf(filtered_counts_dict1), HistPmf(filtered_counts_dict2)

    else:
        raise ValueError("Unsupported output type in calc_pdf_by_sampling.")
