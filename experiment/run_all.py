from dpsniper.utils.my_logging import log, log_context, time_measure
from test.alg.noisy_max_lap import noisy_max_lap
import time
from dpest.config import ConfigManager


def run_alg_by_dpest(alg_func):
    alg_func_name = alg_func.__name__
    with log_context(alg_func_name):
        try:
            log.info("Running dpest...")
            start_time = time.time()
            with time_measure("dpest_time"):
                eps = alg_func()
            print(f"Time: {time.time() - start_time}")
            log.info("dpest result: [eps=%f]", eps)
            log.data("dpest_result", {"eps": eps})
        except Exception:
            log.error("Exception while running dpest on %s", alg_func_name, exc_info=True)

if __name__ == "__main__":
    ConfigManager.load_config()
    run_alg_by_dpest(noisy_max_lap)