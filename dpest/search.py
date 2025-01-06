import numpy as np

def search_scalar_by_threshold(x, y1, y2, th=0.1) -> np.float64:
    """
    二つの確率密度がそれぞれ下位thの割合以下の値である場合は無視して探索する
    """
    y1_threshold = np.sort(y1)[int(y1.size * th)]
    y2_threshold = np.sort(y2)[int(y2.size * th)]
    threshold = max(y1_threshold, y2_threshold)

    max_ratio = 0
    for i in range(x.size):
        if y1[i] > threshold or y2[i] > threshold:
            ratio = y1[i] / y2[i] if y1[i] > y2[i] else y2[i] / y1[i]
            if max_ratio < ratio:
                max_ratio = ratio
    return np.log(max_ratio)

def search_scalar_all(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.float64:
    """
    確率密度の比率の最大値を全探索によって求める
    """
    max_ratio = 0
    for i in range(x.shape[0]):
        if y1[i] == 0 or y2[i] == 0:
            continue
        ratio = y1[i] / y2[i] if y1[i] > y2[i] else y2[i] / y1[i]
        if max_ratio < ratio:
            max_ratio = ratio
    return np.log(max_ratio)

def search_vec_all(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.float64:
    """
    確率密度の比率の最大値を全探索によって求める
    y1,y2はそれぞれ同時確率密度関数の値を持ち、確率変数が次元数である
    もし四次元であって、確率変数の数をnとすると、y1, y2はそれぞれ(n, n, n, n)の形を持つ
    """
    max_ratio = 0
    for pdf1, pdf2 in np.nditer([y1, y2]):
            ratio = pdf1 / pdf2 if pdf1 > pdf2 else pdf2 / pdf1
            if max_ratio < ratio:
                max_ratio = ratio
    return np.log(max_ratio)

def search_hist(hist1, hist2):
    max_ration = 0
    cnt = 0
    for ind in np.ndindex(hist1.shape):
        cnt += 1
        if hist1[ind] == 0 or hist2[ind] == 0:
            continue
        ratio = hist1[ind] / hist2[ind] if hist1[ind] > hist2[ind] else hist2[ind] / hist1[ind]
        if max_ration < ratio:
            max_ration = ratio
    return np.log(max_ration)
