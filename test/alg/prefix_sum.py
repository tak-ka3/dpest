from dpest.operation import Add, ToArray
from dpest.input import InputArray
from dpest.utils import laplace_extract
from dpest.func import eps_est
from dpest.config import ConfigManager

def prefix_sum():
    INPUT_ARR_SIZE = 10
    eps = 0.1
    sens = 1
    LapArr = list(laplace_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens/eps))

    # 実験結果
    result = [LapArr[0]]
    for i in range(1, INPUT_ARR_SIZE):
        result.append(Add(LapArr[i-1], LapArr[i]))
    Y = ToArray(*result)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    ConfigManager.load_config()
    print(prefix_sum())