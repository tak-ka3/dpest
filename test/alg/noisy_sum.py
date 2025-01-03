from dpest.__main__ import laplace_extract, InputArray, Add

eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/eps)

Y = Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5)
eps = Y.eps_est()
