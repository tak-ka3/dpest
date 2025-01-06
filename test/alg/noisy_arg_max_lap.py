from dpest.operation import Br, Max
from dpest.input import InputArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

INPUT_ARR_SIZE = 5
eps = 0.1
sens = 1
LapArr = list(laplace_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens/(eps/2)))

max_val = LapArr[0]
max_ind = 0
for i in range(INPUT_ARR_SIZE):
    max_ind = Br(max_val, LapArr[i], max_ind, i)
    max_val = Max(max_val, LapArr[i])
Y = max_ind
eps = eps_est(Y)
