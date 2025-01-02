from dpest.__main__ import laplace_extract, Array, Max

eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)

Y = Max(Max(Max(Max(Lap1, Lap2), Lap3), Lap4), Lap5)
eps = Y.eps_est()
