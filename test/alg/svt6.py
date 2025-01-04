from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, Comp

eps = 0.1
eps1 = eps/2
eps2 = eps - eps1
sens = 1
th =1

INPUT_ARRAY_SIZE = 10
LapArr = list(laplace_extract(InputArray(INPUT_ARRAY_SIZE), sens/eps2))
Lap = Laplace(th, sens/eps1)
result = []
for i in range(INPUT_ARRAY_SIZE):
    result.append(Comp(LapArr[i], Lap))
Y = ToArray(*result)
eps = Y.eps_est()
