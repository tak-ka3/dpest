from dpest.operation import Add, ToArray
from dpest.input import InputArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5, adj="inf"), sens/eps)

# 実験結果
Y = ToArray(Lap1, Add(Lap1, Lap2), Add(Add(Lap1, Lap2), Lap3), Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5))
eps = eps_est(Y)
