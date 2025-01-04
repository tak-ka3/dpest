from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, raw_extract, Case, Add, Br

eps = 0.1
sens = 1
eps1 = eps/2
eps2 = eps - eps1
c = 1
th = 1

Arr = InputArray(5)
Lap = Laplace(th, sens/eps1)
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens/eps2)
LapArr = [Lap1, Lap2, Lap3, Lap4, Lap5]
result = []
cnt = 0 # 閾値を超えた数
cnt_over = 0 # breakの真偽値
for i in range(5):
    is_over = Br(LapArr[i], Lap, 1, 0)
    case_dict = {1: 2, "otherwise": is_over}
    output = Case(cnt_over, case_dict)
    result.append(output)

    # breakの処理
    cnt = Add(is_over, cnt)
    cnt_over = Br(cnt, c, 1, 0)

Y = ToArray(*result)
eps = Y.eps_est()
