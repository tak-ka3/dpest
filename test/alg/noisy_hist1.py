from dpest.input import InputArray
from dpest.operation import ToArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

def noisy_hist1():
    INPUT_ARR_SIZE = 5
    eps = 0.1
    sens = 1
    th =1
    # 配列要素それぞれにラプラスノイズを加えて取り出す
    LapArr = list(laplace_extract(InputArray(INPUT_ARR_SIZE, adj="1"), sens/eps))
    Y = ToArray(*LapArr)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    print(noisy_hist1())