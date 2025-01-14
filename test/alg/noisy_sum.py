from dpest.operation import Add
from dpest.input import InputArray
from dpest.utils import laplace_extract
from dpest.func import eps_est
from dpest.config import ConfigManager

def noisy_sum():
    eps = 0.1
    sens = 1
    Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5, adj="inf"), sens/eps)

    Y = Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    ConfigManager.load_config()
    print(noisy_sum())