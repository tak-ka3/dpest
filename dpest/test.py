import numpy as np

# 5次元データを生成（各次元にランダムなデータを用意）
data = np.random.uniform(-1, 1, size=(1000, 5))  # 1000サンプル, 5次元

# 各次元10個のビンでヒストグラムを計算
bins = 10  # 各次元のビン数
hist, edges = np.histogramdd(data, bins=bins)

# 総ビン数を計算
total_bins = hist.size

# 結果を表示
print(f"5次元ヒストグラムの形状: {hist.shape}")
print(f"総ビン数: {total_bins}")
# print(hist)
for index in np.ndindex(hist.shape):
    bin_count = hist[index]
    bin_edges = [edges[dim][index[dim]:index[dim]+2] for dim in range(5)]
    print(f"ビンインデックス: {index}, カウント: {bin_count}, 境界: {bin_edges}")