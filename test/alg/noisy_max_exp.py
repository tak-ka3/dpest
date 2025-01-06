from dpest.operation import Max
from dpest.input import InputArray
from dpest.utils import exp_extract
from dpest.func import eps_est

INPUT_ARR_SIZE = 5
eps = 0.1
sens = 1
ExpArr = list(exp_extract(InputArray(INPUT_ARR_SIZE, adj="inf"), sens/eps))

Y = ExpArr[0]
for i in range(1, INPUT_ARR_SIZE):
    Y = Max(Y, ExpArr[i])
eps = eps_est(Y)
