from dpest.__main__ import laplace_extract, Array, ToArray

eps = 0.1
sens = 1
th =1
# 配列要素それぞれにラプラスノイズを加えて取り出す
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(Array(5), sens/eps)
Y = ToArray(Lap1, Lap2, Lap3, Lap4, Lap5)
