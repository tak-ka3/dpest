## 実験結果
eps= 0.5004172927849141
eps= 0.6005007513418966
eps= 0.5004172927849141
eps= 0.6005007513418966
eps= 0.6005007513418966
eps= 0.5004172927849141
eps= 0.6005007513418966

## 実験結果考察
- DP-Sniperと精度は同じくらいを達成している
    - 特に(1, 2)の組み合わせではなく、(0, 1)の組み合わせでε=0.6を達成している
- Rapporと同様に既存研究よりもかなり短い実行時間を達成している
- fの値によって結果が大きく変わるので、元の実験と揃える必要あり
    - DP-Sniperだとf=0.95
- 入力が左右対称なため、色々な入力セットを試すことで、理論値0.8を達成できるかもしれない
    - (0.5, -0.5)の組み合わせで`ε=0.8006676684558616`を達成
    - さらに実験を進めると、[(0.5, -0.5)~(0.2, -0.8)]ぐらいの範囲で、0.8という上限値を達成する
