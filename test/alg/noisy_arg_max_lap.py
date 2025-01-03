from dpest.__main__ import laplace_extract, InputArray, Max, Br

eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/(eps/2))
LapArr = [Lap1, Lap2, Lap3, Lap4, Lap5]

max_val = LapArr[0]
max_ind = 0
for i in range(5):
    max_ind = Br(max_val, LapArr[i], max_ind, i)
    max_val = Max(max_val, LapArr[i])
Y = max_ind
eps = Y.eps_est()
