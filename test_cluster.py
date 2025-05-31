# quick_test_threshold_graph.py

import numpy as np
from utils.cluster.graph_clustering import graph_threshold_clustering

# 4 个 3 维嵌入，假设用如下小数组演示
X = np.array([
    [0.1, 0.1, 0.2],
    [0.15, 0.05, 0.25],
    [0.9, 0.8, 0.75],
    [0.85, 0.78, 0.7]
])

# 对应的 IDs
ids = ["A", "B", "C", "D"]

# 阈值 0.9：前两向量之间余弦类似（相似度超阈），后两类似，
# A/B 会连边，C/D 会连边，形成两个连通分量
clusters_dict = graph_threshold_clustering(
    embedding_matrix=X,
    ids=ids,
    similarity_threshold=0.9,
    use_cosine=True
)
print("Threshold graph clusters:", clusters_dict)
# 可能输出 {'A': ['A','B'], 'C': ['C','D']}
