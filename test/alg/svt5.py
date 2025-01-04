from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, raw_extract, Comp

eps = 0.1
eps1 = eps/2
sens = 1

Arr = InputArray(10)
th = 1
Lap = Laplace(th, sens/eps1)
q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = raw_extract(InputArray(10))
Y = ToArray(Comp(Lap, q1), Comp(Lap, q2), Comp(Lap, q3), Comp(Lap, q4), Comp(Lap, q5), Comp(Lap, q6), Comp(Lap, q7), Comp(Lap, q8), Comp(Lap, q9), Comp(Lap, q10))
eps = Y.eps_est()
