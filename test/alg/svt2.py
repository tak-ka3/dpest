from dpest.__main__ import laplace_extract, InputArray, ToArray, Laplace, raw_extract, Case, Add, Br

eps = 0.1
sens = 1
eps1 = eps/2
eps2 = eps - eps1
c = 1
th = 1

Arr = InputArray(5)
Lap = Laplace(th, sens*c/eps1)
Lap1, Lap2, Lap3, Lap4, Lap5 = laplace_extract(InputArray(5), sens*c/(eps2/2))
LapArr = [Lap1, Lap2, Lap3, Lap4, Lap5]
result = []
cnt = 0 # 閾値を超えた数
cnt_over = 0 # breakの真偽値
for i in range(5):
    is_over = Br(LapArr[i], Lap, 1, 0)
    # もし閾値を超えたら閾値のノイズを更新
    new_noise_case_dict = {1: Laplace(th, sens*c/eps1), 0: Lap}
    Lap = Case(is_over, new_noise_case_dict)

    # 前回の結果とがと今回閾値を超えたかどうかを用いて、今回の出力を決定
    case_dict = {1: 2, "otherwise": is_over}
    output = Case(cnt_over, case_dict)
    result.append(output)

    # breakの処理
    cnt = Add(is_over, cnt)
    cnt_over = Br(cnt, c, 1, 0)

Y = ToArray(*result)
eps = Y.eps_est()
