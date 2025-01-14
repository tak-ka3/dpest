from dpest.input import InputScalar
from dpest.func import eps_est
from dpest.config import ConfigManager
from dpest.distrib import Laplace

def lap():
    eps = 0.1
    sens = 1
    # 配列要素それぞれにラプラスノイズを加えて取り出す
    Lap  = Laplace(InputScalar(), sens/eps)
    Y = Lap
    eps = eps_est(Y)
    return eps
    
if __name__ == "__main__":
    ConfigManager.load_config()
    print(lap())