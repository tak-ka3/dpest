from dpest.operation import Max
from dpest.input import InputArray
from dpest import utils
from dpest.func import eps_est

eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/eps)

Y = Max(Max(Max(Max(Lap1, Lap2), Lap3), Lap4), Lap5)
eps = eps_est(Y)
