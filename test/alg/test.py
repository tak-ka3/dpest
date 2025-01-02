from dpest.__main__ import laplace_extract, Array, Add, ToArray, Max, Laplace, raw_extract, Comp

print("SATRT ALL TESTS")
eps = 0.1
sens = 1
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)

print("NoisySum, 0.1")
Y = Max(Max(Max(Max(Lap1, Lap2), Lap3), Lap4), Lap5)
eps = Y.eps_est()

print("NoisyMax, 0.1")
Y = Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5)
eps = Y.eps_est()

print("PrefixSum, 0.1")
Y = ToArray(Lap1, Add(Lap1, Lap2), Add(Add(Lap1, Lap2), Lap3), Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Add(Add(Add(Add(Lap1, Lap2), Lap3), Lap4), Lap5))
eps = Y.eps_est()

print("SVT5, inf")
Arr = Array(5)
Lap = Laplace(0, 1/eps)
q1, q2, q3, q4, q5 = raw_extract(Array(5))
Y = ToArray(Comp(Lap, q1), Comp(Lap, q2), Comp(Lap, q3), Comp(Lap, q4), Comp(Lap, q5))
eps = Y.eps_est()
print("COMPLETE ALL TESTS")
