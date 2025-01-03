from dpest.__main__ import exp_extract, InputArray, Max

eps = 0.1
sens = 1
Exp1, Exp2, Exp3, Exp4, Exp5 = exp_extract(InputArray(5), sens/eps)

Y = Max(Max(Max(Max(Exp1, Exp2), Exp3), Exp4), Exp5)
eps = Y.eps_est()
