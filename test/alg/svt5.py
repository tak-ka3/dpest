from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, raw_extract, Comp

eps = 0.1
sens = 1

Arr = InputArray(5)
th = 1
Lap = Laplace(th, sens/eps)
q1, q2, q3, q4, q5 = raw_extract(InputArray(5))
Y = ToArray(Comp(Lap, q1), Comp(Lap, q2), Comp(Lap, q3), Comp(Lap, q4), Comp(Lap, q5))
eps = Y.eps_est()
