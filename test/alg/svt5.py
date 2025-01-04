from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, raw_extract, Comp

eps = 0.1
eps1 = eps/2
sens = 1

INPUT_ARRAY_SIZE = 10
th = 1
Lap = Laplace(th, sens/eps1)
Arr = list(raw_extract(InputArray(INPUT_ARRAY_SIZE)))
result = []
for i in range(INPUT_ARRAY_SIZE):
    result.append(Comp(Lap, Arr[i]))
Y = ToArray(*result)
eps = Y.eps_est()
