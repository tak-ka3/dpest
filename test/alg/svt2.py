import numpy as np
from dpest.distrib import Laplace
from dpest.input import InputArray
from dpest.operation import Br, Case, Add, ToArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

def svt2():
    eps = 0.1
    sens = 1
    eps1 = eps/2
    eps2 = eps - eps1
    c = 1
    th = 1

    INPUT_ARR_SIZE = 10

    Lap = Laplace(th, sens*c/eps1)
    LapArr = list(laplace_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens*c/(eps2/2)))
    result = []
    cnt = 0 # 閾値を超えた数
    cnt_over = 0 # breakの真偽値
    for i in range(INPUT_ARR_SIZE):
        is_over = Br(LapArr[i], Lap, True, False)
        # もし閾値を超えたら閾値のノイズを更新
        new_noise_case_dict = {True: Laplace(th, sens*c/eps1), False: Lap}
        Lap = Case(is_over, new_noise_case_dict)

        # 前回の結果とがと今回閾値を超えたかどうかを用いて、今回の出力を決定
        case_dict = {1: np.nan, "otherwise": is_over}
        output = Case(cnt_over, case_dict)
        result.append(output)

        # breakの処理
        cnt = Add(is_over, cnt)
        cnt_over = Br(cnt, c, 1, 0)

    Y = ToArray(*result)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    print(svt2())