from dpest.distrib import Laplace
from dpest.input import InputArray
from dpest.operation import Comp, ToArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

def svt6():
    eps = 0.1
    eps1 = eps/2
    eps2 = eps - eps1
    sens = 1
    th =1

    INPUT_ARRAY_SIZE = 10
    LapArr = list(laplace_extract(InputArray(INPUT_ARRAY_SIZE, adj="inf"), sens/eps2))
    Lap = Laplace(th, sens/eps1)
    result = []
    for i in range(INPUT_ARRAY_SIZE):
        result.append(Comp(LapArr[i], Lap))
    Y = ToArray(*result)
    eps = eps_est(Y)
    return eps

if __name__ == "__main__":
    print(svt6())