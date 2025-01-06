from dpest.distrib import Laplace
from dpest.input import InputArray
from dpest.operation import Comp, ToArray
from dpest.utils import raw_extract
from dpest.func import eps_est

def svt5():
    eps = 0.1
    eps1 = eps/2
    sens = 1

    INPUT_ARRAY_SIZE = 10
    th = 1
    Lap = Laplace(th, sens/eps1)
    Arr = list(raw_extract(InputArray(INPUT_ARRAY_SIZE, adj="inf")))
    result = []
    for i in range(INPUT_ARRAY_SIZE):
        result.append(Comp(Lap, Arr[i]))
    Y = ToArray(*result)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    print(svt5())