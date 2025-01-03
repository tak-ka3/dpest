from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, Comp

eps = 0.1
sens = 1
th =1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/eps)
Lap = Laplace(th, sens/eps)
Y = ToArray(Comp(Lap1, Lap), Comp(Lap2, Lap), Comp(Lap3, Lap), Comp(Lap4, Lap), Comp(Lap5, Lap))
eps = Y.eps_est()
