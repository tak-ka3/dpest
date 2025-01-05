from dpest.distrib import Laplace, Exp, ArrayItem
from dpest.input import InputArray

def laplace_extract(arr: InputArray, scale: float):
    """
    配列要素それぞれにラプラスノイズを加えて取り出す
    """
    lap_list = []
    for val in arr:
        if isinstance(val, ArrayItem):
            lap_list.append(Laplace(val, scale))
        else:
            raise ValueError("Invalid value")
    return lap_list

def exp_extract(arr: InputArray, scale: float):
    """
    配列要素それぞれに指数分布ノイズを加えて取り出す
    """
    exp_list = []
    for val in arr:
        if isinstance(val, ArrayItem):
            exp_list.append(Exp(val, scale))
        else:
            raise ValueError("Invalid value")
    return exp_list

def raw_extract(arr: InputArray):
    """
    配列要素それぞれにラプラスノイズを加えて取り出す
    """
    lap_list = []
    for val in arr:
        if isinstance(val, ArrayItem):
            lap_list.append(val)
        else:
            raise ValueError("Invalid value")
    return lap_list