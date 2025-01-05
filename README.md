## 実行方法
```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -e .
# alg_nameは以下を参照
# 実験結果はexperiment/{alg_name}/{datetime}.mdに格納される
$ make exp_{alg_name}
```

## アルゴリズム一覧
上記alg_nameに代入
- noisy_max_lap noisy_max_exp
- noisy_arg_max_lap noisy_arg_max_exp
- prefix_sum
- svt1 svt2 svt3 svt4 svt5 svt6 
- num_svt
- laplace_parallel
- onetime_rappor rappor
- truncated_geometric
