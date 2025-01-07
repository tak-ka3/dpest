import os

from test.alg.noisy_arg_max_lap import noisy_arg_max_lap
from test.alg.noisy_arg_max_exp import noisy_arg_max_exp
from test.alg.noisy_max_lap import noisy_max_lap
from test.alg.noisy_max_exp import noisy_max_exp

from test.alg.noisy_hist1 import noisy_hist1
from test.alg.noisy_hist2 import noisy_hist2
from test.alg.laplace_parallel import laplace_parallel

from test.alg.noisy_sum import noisy_sum
from test.alg.prefix_sum import prefix_sum

from test.alg.onetime_rappor import onetime_rappor
from test.alg.rappor import rappor

from test.alg.svt1 import svt1
from test.alg.svt2 import svt2
from test.alg.svt3 import svt3
from test.alg.svt4 import svt4
from test.alg.svt5 import svt5
from test.alg.svt6 import svt6
from test.alg.svt34_parallel import svt34_parallel
from test.alg.num_svt import num_svt
from test.alg.truncated_geometric import truncated_geometric

from dpsniper.utils.my_multiprocessing import initialize_parallel_executor
from dpsniper.utils.paths import get_output_directory, set_output_directory
from dpsniper.utils.my_logging import log
from datetime import datetime
from experiment.run_alg import run_alg_by_dpest
from dpest.config import ConfigManager
from enum import Enum, auto

class SettingType(Enum):
    COMMON = auto()
    SPECIFIC = auto()

def run_with_postprocessing(n_processes: int, out_dir: str, only_mechanism=None, postfix=""):
    log.configure("WARNING")
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_dir, formatted_date)
    os.makedirs(out_dir, exist_ok=True)
    set_output_directory(out_dir)
    logs_dir = get_output_directory("logs")
    log_file = os.path.join(logs_dir, f"dpest{postfix}_log.log")
    data_file = os.path.join(logs_dir, f"dpest{postfix}_data_{formatted_date}.log")

    log.configure("INFO", log_file=log_file, data_file=data_file, file_level="INFO")

    alg_dict = {
        "noisy_arg_max_lap": noisy_arg_max_lap, "noisy_arg_max_exp": noisy_arg_max_exp, "noisy_max_lap": noisy_max_lap, "noisy_max_exp": noisy_max_exp,
        "noisy_hist1": noisy_hist1, "noisy_hist2": noisy_hist2, "laplace_parallel": laplace_parallel,
        "noisy_sum": noisy_sum, "prefix_sum": prefix_sum,
        "onetime_rappor": onetime_rappor, "rappor": rappor,
        "svt1": svt1, "svt2": svt2, "svt3": svt3, "svt4": svt4, "svt5": svt5, "svt6": svt6, "svt34_parallel": svt34_parallel, "num_svt": num_svt,
        "truncated_geometric": truncated_geometric
    }

    with initialize_parallel_executor(n_processes, out_dir):
        alg_list = [
            (noisy_max_lap, SettingType.COMMON),
            (noisy_arg_max_lap, SettingType.COMMON),
            (noisy_arg_max_exp, SettingType.COMMON),
            (noisy_max_exp, SettingType.COMMON),
            (noisy_hist1, SettingType.SPECIFIC),
            (noisy_hist2, SettingType.SPECIFIC),
            (laplace_parallel, SettingType.SPECIFIC),
            (noisy_sum, SettingType.COMMON),
            (prefix_sum, SettingType.SPECIFIC),
            (onetime_rappor, SettingType.COMMON),
            (rappor, SettingType.COMMON),
            (svt1, SettingType.COMMON),
            (svt2, SettingType.COMMON),
            (svt3, SettingType.COMMON),
            (svt4, SettingType.COMMON),
            (svt5, SettingType.COMMON),
            (svt6, SettingType.COMMON),
            (svt34_parallel, SettingType.COMMON),
            (num_svt, SettingType.COMMON),
            (truncated_geometric, SettingType.COMMON)
        ]

        # 特定のアルゴリズムだけを実行する場合
        if only_mechanism is not None:
            alg_list = [(alg_dict[only_mechanism], SettingType.COMMON)]

        for alg in alg_list:
            # ここでcommon.yamlを読み込むことも選択可能
            alg_func = alg[0]
            setting_file_name = "common" if alg[1] == SettingType.COMMON else alg_func.__name__
            ConfigManager.load_config(setting_file_name)
            setting_str = ConfigManager.get_setting_str()
            log.info("setting: %s", setting_str)
            run_alg_by_dpest(alg_func)
    log.info("finished experiments")
