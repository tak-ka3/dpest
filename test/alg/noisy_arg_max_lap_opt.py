from dpest.operation import ArgMaxArray
from dpest.input import InputArray
from dpest.utils import laplace_extract
from dpest.func import eps_est
from dpest.config import ConfigManager

def noisy_arg_max_lap_opt():
    INPUT_ARR_SIZE = 5
    eps = 0.1
    sens = 1
    LapArr = list(laplace_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens/(eps/2)))
    Y = ArgMaxArray(*LapArr)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    ConfigManager.load_config()
    print(noisy_arg_max_lap_opt())