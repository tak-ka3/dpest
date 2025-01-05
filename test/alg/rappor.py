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

filter_size = 20
rappor_f = 0.75
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

p, q = 0.45, 0.55
p_val_to_prob = {0: 1-p, 1: p}
q_val_to_prob = {0: 1-q, 1: q}
pmf1q, pmf2q, pmf3q, pmf4q, pmf5q = RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob)
pmf6q, pmf7q, pmf8q, pmf9q, pmf10q = RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob)
pmf11q, pmf12q, pmf13q, pmf14q, pmf15q = RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob)
pmf16q, pmf17q, pmf18q, pmf19q, pmf20q = RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob), RawPmf(val_to_prob=q_val_to_prob)
pmf1p, pmf2p, pmf3p, pmf4p, pmf5p = RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob)
pmf6p, pmf7p, pmf8p, pmf9p, pmf10p = RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob)
pmf11p, pmf12p, pmf13p, pmf14p, pmf15p = RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob)
pmf16p, pmf17p, pmf18p, pmf19p, pmf20p = RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob), RawPmf(val_to_prob=p_val_to_prob)

Y = ToArray(
    Case(Case(pmf1, case_dict_list[0]), {0: pmf1p, 1: pmf1q}), 
    Case(Case(pmf2, case_dict_list[1]), {0: pmf2p, 1: pmf2q}), 
    Case(Case(pmf3, case_dict_list[2]), {0: pmf3p, 1: pmf3q}), 
    Case(Case(pmf4, case_dict_list[3]), {0: pmf4p, 1: pmf4q}), 
    Case(Case(pmf5, case_dict_list[4]), {0: pmf5p, 1: pmf5q}), 
    Case(Case(pmf6, case_dict_list[5]), {0: pmf6p, 1: pmf6q}), 
    Case(Case(pmf7, case_dict_list[6]), {0: pmf7p, 1: pmf7q}), 
    Case(Case(pmf8, case_dict_list[7]), {0: pmf8p, 1: pmf8q}), 
    Case(Case(pmf9, case_dict_list[8]), {0: pmf9p, 1: pmf9q}), 
    Case(Case(pmf10, case_dict_list[9]), {0: pmf10p, 1: pmf10q}), 
    Case(Case(pmf11, case_dict_list[10]), {0: pmf11p, 1: pmf11q}), 
    Case(Case(pmf12, case_dict_list[11]), {0: pmf12p, 1: pmf12q}), 
    Case(Case(pmf13, case_dict_list[12]), {0: pmf13p, 1: pmf13q}), 
    Case(Case(pmf14, case_dict_list[13]), {0: pmf14p, 1: pmf14q}), 
    Case(Case(pmf15, case_dict_list[14]), {0: pmf15p, 1: pmf15q}), 
    Case(Case(pmf16, case_dict_list[15]), {0: pmf16p, 1: pmf16q}), 
    Case(Case(pmf17, case_dict_list[16]), {0: pmf17p, 1: pmf17q}), 
    Case(Case(pmf18, case_dict_list[17]), {0: pmf18p, 1: pmf18q}), 
    Case(Case(pmf19, case_dict_list[18]), {0: pmf19p, 1: pmf19q}), 
    Case(Case(pmf20, case_dict_list[19]), {0: pmf20p, 1: pmf20q})
)
eps = eps_est(Y)
