## 実行方法
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -e .
$ cd dpsniper
$ pip install .
# alg_nameは以下を参照

# 推定されたεを標準出力
$ make {alg_name}

# 個別のアルゴリズムを実験する。実験結果はexperiment/out/{datetime}/*.logに格納される
$ make exp_{alg_name}

# 全てのアルゴリズムを実験する。実験結果は上記と同じ場所にある。
$ make dpest-exp
```

## アルゴリズム一覧
上記alg_nameに代入
- noisy_max_lap noisy_max_exp
- noisy_arg_max_lap noisy_arg_max_exp
- prefix_sum
- svt1 svt2 svt3 svt4 svt5 svt6 svt34_parallel
- num_svt
- laplace_parallel, noisy_hist1, noisy_hist2
- onetime_rappor rappor
- truncated_geometric
