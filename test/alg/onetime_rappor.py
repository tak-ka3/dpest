import numpy as np
import mmh3
from dpest.distrib import RawPmf
from dpest.input import InputScalarToArray
from dpest.operation import ToArray, Case
from dpest.func import eps_est

def populate_bloom_filter(val):
    filter_size = 20
    n_hashes = 4
    filter = np.zeros(filter_size)
    for i in range(0, n_hashes):
        hashval = mmh3.hash(str(val), seed=i) % filter_size
        filter[hashval] = 1
    return filter

def onetime_rappor():
    filter_size = 20
    rappor_f = 0.95
    val_to_prob = {0: 0.5*rappor_f, 1: 0.5*rappor_f, 2: 1 - rappor_f}
    pmf1, pmf2, pmf3, pmf4, pmf5 = RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob)
    pmf6, pmf7, pmf8, pmf9, pmf10 = RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob)
    pmf11, pmf12, pmf13, pmf14, pmf15 = RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob)
    pmf16, pmf17, pmf18, pmf19, pmf20 = RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob), RawPmf(val_to_prob=val_to_prob)
    case_dict_list = []
    # 計算グラフでInputArrayクラスの要素にアクセスしたと同時に、入力からBloom Filterを生成して、それを再利用する
    Arr = InputScalarToArray(size=filter_size, func=populate_bloom_filter)
    for i in range(filter_size):
        case_dict_list.append({0: 1, 1: 0, 2: Arr[i]})

    Y = ToArray(Case(pmf1, case_dict_list[0]), Case(pmf2, case_dict_list[1]), Case(pmf3, case_dict_list[2]), Case(pmf4, case_dict_list[3]), Case(pmf5, case_dict_list[4]), Case(pmf6, case_dict_list[5]), Case(pmf7, case_dict_list[6]), Case(pmf8, case_dict_list[7]), Case(pmf9, case_dict_list[8]), Case(pmf10, case_dict_list[9]), Case(pmf11, case_dict_list[10]), Case(pmf12, case_dict_list[11]), Case(pmf13, case_dict_list[12]), Case(pmf14, case_dict_list[13]), Case(pmf15, case_dict_list[14]), Case(pmf16, case_dict_list[15]), Case(pmf17, case_dict_list[16]), Case(pmf18, case_dict_list[17]), Case(pmf19, case_dict_list[18]), Case(pmf20, case_dict_list[19]))
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    print(onetime_rappor())