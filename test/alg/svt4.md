## アルゴリズム概要
- SVT1とはεの値のみが異なる
- TrueやFalseを配列に含めるのが正しいが、SVT3のようにやや複雑になるので、1や0を返すように実装している
    - SVT3のうようにTrueやFalseを返す実装にすることも可能

## 実験結果
input_size: 5, SAMPLING_NUM = 100000
eps= 0.06734855034742351
eps= 0.05289457575452441
eps= 0.0949626675958627
eps= 0.116723312599069
eps= 0.15981062197154686
eps= 0.051898438127651235
eps= 0.1295718167520885

input_size: 5, SAMPLING_NUM: 100000
eps= 0.10185118042440947
eps= 0.11624777929605373
eps= 0.11433983464410624
eps= 0.0959713032296311
eps= 0.19891986063288708
eps= 0.07932354187483091
eps= 0.13697589266393498

## 実験結果考察
- 理論値eps=0.18を大体達成している
    - サンプリングによって答えを求めている
    - breakを2を出力するという方法で実現している
- 既存手法だと、比率の最大値を探索するという方法であるため、基本的には理論値よりも小さくなるが、サンプリングだと理論値を中心とした正規分布となるので、理論値を超えてしまうことがある
