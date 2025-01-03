## 設定
```python
prng = np.random.default_rng(seed=42)
LAP_UPPER = 50
LAP_LOWER = -50
LAP_VAL_NUM = 100
EXP_UPPER = 50
EXP_LOWER = 0
EXP_VAL_NUM = 100
INF = 10000
SAMPLING_NUM = 100000
GRID_NUM = 10
```

## 実験結果
eps= 0.05293007773042741
eps= 0.046212504860653404
eps= 0.09681743517779998
eps= 0.0826022692804749
eps= 0.08954698120775982
eps= 0.014456869340966651
eps= 0.07078175265821185

## 実験結果考察
- 理論値ε=0.1を達成
    - eps/2を設定することに注意
- 依存関係があるので、全てサンプリングにより確率を求めている
