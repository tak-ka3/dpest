from dpest.operation import Br, Max
from dpest.input import InputArray
from dpest.utils import exp_extract
from dpest.func import eps_est
from dpest.config import ConfigManager

def noisy_arg_max_exp():
    INPUT_ARR_SIZE = 5
    eps = 0.1
    sens = 1
    ExpArr = list(exp_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens/(eps/2)))

    max_val = ExpArr[0]
    max_ind = 0
    for i in range(INPUT_ARR_SIZE):
        max_ind = Br(max_val, ExpArr[i], max_ind, i)
        max_val = Max(max_val, ExpArr[i])
    Y = max_ind
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    ConfigManager.load_config()
    print(noisy_arg_max_exp())