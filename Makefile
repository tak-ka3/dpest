.PHONY: all $(TASKS) $(EXP_TASKS)

# タスクリスト
TASKS := noisy_sum prefix_sum noisy_max_lap noisy_max_exp noisy_arg_max_lap \
         noisy_arg_max_exp svt1 svt2 svt3 svt4 svt5 svt6 svt34_parallel num_svt \
         noisy_hist1 noisy_hist2 laplace_parallel \
		 onetime_rappor rappor truncated_geometric

# EXPタスクリストを生成
EXP_TASKS := $(addprefix exp_, $(TASKS))

OUT_BASE=$(CURDIR)/experiment/out

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

correct_eps_svt3 := inf
adj_svt3 := inf

correct_eps_svt4 := 0.18
adj_svt4 := inf

correct_eps_svt34_parallel := inf
adj_svt34_parallel := inf

correct_eps_svt5 := inf
adj_svt5 := inf

correct_eps_svt6 := inf
adj_svt6 := inf

correct_eps_num_svt := 0.1
adj_num_svt := inf

correct_eps_noisy_hist1 := 0.1
adj_noisy_hist1 := 1

correct_eps_noisy_hist2 := 10
adj_noisy_hist2 := 1

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

exp_%:
	@echo "Running experiment for $*"
	python -m experiment --output-dir "$(OUT_BASE)" --processes 1 --alg $*


make exp_dpest:
	@echo "Running dpest experiments"
	python -m experiment --output-dir "$(OUT_BASE)" --processes 1
