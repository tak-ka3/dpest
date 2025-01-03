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
eps= 0.05327452457218238
eps= 0.02911774549908415
eps= 0.08748149366144951
eps= 0.0971729124665691
eps= 0.08669879118439372
eps= 0.020213771984968373
eps= 0.07480731450285076

## 実験結果考察
- 理論値ε=0.1を達成
    - eps/2を設定することに注意
- 依存関係があるので、全てサンプリングにより確率を求めている
