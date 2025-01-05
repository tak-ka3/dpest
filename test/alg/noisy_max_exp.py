from dpest.operation import Max
from dpest.input import InputArray
from dpest.utils import exp_extract
from dpest.func import eps_est

eps = 0.1
sens = 1
Exp1, Exp2, Exp3, Exp4, Exp5 = exp_extract(InputArray(5), sens/eps)

Y = Max(Max(Max(Max(Exp1, Exp2), Exp3), Exp4), Exp5)
eps = eps_est(Y)
