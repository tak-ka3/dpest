from dpest.operation import Br, Max
from dpest.input import InputArray
from dpest.utils import exp_extract
from dpest.func import eps_est

eps = 0.1
sens = 1
Exp1, Exp2, Exp3, Exp4, Exp5 = exp_extract(InputArray(5), sens/(eps/2))
ExpArr = [Exp1, Exp2, Exp3, Exp4, Exp5]

max_val = ExpArr[0]
max_ind = 0
for i in range(5):
    max_ind = Br(max_val, ExpArr[i], max_ind, i)
    max_val = Max(max_val, ExpArr[i])
Y = max_ind
eps = eps_est(Y)
