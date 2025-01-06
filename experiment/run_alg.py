import subprocess
import re
import os
from dpest.config import *
from datetime import datetime
import sys

# 正規表現パターン: {変数名} = 数値（整数または小数）
pattern = r"^([A-Z_]+)\s*=\s*([-+]?\d+\.?\d*)$"

def extract_parameters(input_path, output_path):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        outfile.write("## Parameters\n")
        for line in infile:
            line = line.strip()  # 前後の空白を削除
            match = re.match(pattern, line)
            if match:
                # マッチした場合、変数名と値を取得
                var_name, value = match.groups()
                # 結果を出力
                outfile.write(f"{var_name} = {value}\n")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python run_alg.py <algorithm_name>"
    alg_name = sys.argv[1]
    command = ["make", alg_name]

    start_time = datetime.now()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = datetime.now()
    expr_result = result.stdout

    # コマンドの実行結果がエラーの場合、エラーメッセージを表示して終了
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

    DP_EST_DIR = script_dir = os.getcwd()

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(f"{DP_EST_DIR}/experiment/{alg_name}", exist_ok=True)

    target_file = f"{DP_EST_DIR}/dpest/config.py"
    output_file = f"{DP_EST_DIR}/experiment/{alg_name}/{formatted_now}.md"

    with open(output_file, "w") as outfile:
        outfile.write(f"# {alg_name}\n")
        outfile.write(f"## Execution Time\n{end_time - start_time} sec\n")
    # パラメタを抽出してファイルに書き込む
    extract_parameters(target_file, output_file)

    # 実験結果をファイルに書き込む
    with open(output_file, "a") as outfile:
        outfile.write(f"\n## Experiment Result\n{expr_result}\n")
