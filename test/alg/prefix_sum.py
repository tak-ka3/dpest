from dpest.__main__ import laplace_extract, Array, Add, ToArray
eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)

# 実験結果
Y = ToArray(Lap1, Add(Lap1, Lap2), Add(Add(Lap1, Lap2), Lap3), Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5))
eps = Y.eps_est()
