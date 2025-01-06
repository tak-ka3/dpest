.PHONY: all $(TASKS) $(EXP_TASKS)

# タスクリスト
TASKS := noisy_sum prefix_sum noisy_max_lap noisy_max_exp noisy_arg_max_lap \
         noisy_arg_max_exp svt1 svt2 svt3 svt4 svt5 svt6 num_svt \
         laplace_parallel onetime_rappor rappor truncated_geometric

# EXPタスクリストを生成
EXP_TASKS := $(addprefix exp_, $(TASKS))

# すべての通常タスクを実行
all: $(TASKS)

all_exp: $(EXP_TASKS)

# 変数で eps を管理
correct_eps_noisy_sum := 0.1
adj_noisy_sum := inf

correct_eps_prefix_sum := 0.1
adj_prefix_sum := inf

correct_eps_noisy_max_lap := 0.5
adj_noisy_max_lap := inf

correct_eps_noisy_max_exp := 0.5
adj_noisy_max_exp := inf

correct_eps_noisy_arg_max_lap := 0.1
adj_noisy_arg_max_lap := inf

correct_eps_noisy_arg_max_exp := 0.1
adj_noisy_arg_max_exp := inf

correct_eps_svt1 := 0.1
adj_svt1 := inf

correct_eps_svt2 := 0.1
adj_svt2 := inf

correct_eps_svt3 := 0.1
adj_svt3 := inf

correct_eps_svt4 := 0.18
adj_svt4 := inf

correct_eps_svt5 := inf
adj_svt5 := inf

correct_eps_svt6 := inf
adj_svt6 := inf

correct_eps_num_svt := 0.1
adj_num_svt := inf

correct_eps_laplace_parallel := 0.1
adj_laplace_parallel := 1

correct_eps_onetime_rappor := 0.8
adj_onetime_rappor := 1

correct_eps_rappor := 0.4
adj_rappor := 1

correct_eps_truncated_geometric := 0.12
adj_truncated_geometric := 1

# 各タスクの共通ルール
%:
	@echo "correct_eps=$(correct_eps_$@), adj=$(adj_$@)"
	python -m test.alg.$@

# EXPタスクのルール
exp_%:
	@echo "Running experiment for $*"
	python experiment/run_alg.py $*
