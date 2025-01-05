from dpest.input import InputArray
from dpest.operation import ToArray
from dpest.utils import laplace_extract
from dpest.func import eps_est

eps = 0.1
sens = 1
th =1
# 配列要素それぞれにラプラスノイズを加えて取り出す
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/eps)
Y = ToArray(Lap1, Lap2, Lap3, Lap4, Lap5)
eps = eps_est(Y)
