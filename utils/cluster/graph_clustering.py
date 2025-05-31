# utils/cluster/graph_clustering.py

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def graph_threshold_clustering(
    embedding_matrix: np.ndarray,
    ids: list,
    similarity_threshold: float = 0.85,
    use_cosine: bool = True
) -> dict:
    """
    基于阈值的图聚类（连通分量）。构建无向图：若两向量余弦相似度 >= threshold，则在对应节点间添加边。
    最后对图求连通分量，每个连通分量即为一个簇。未与任何点连边的节点也属于自己单独的簇。

    Args:
        embedding_matrix: (N, D) 数组，N 个嵌入向量。
        ids: 长度为 N 的样本 ID 列表，用作图节点标签。
        similarity_threshold: 当相似度 >= threshold 时，在节点间建边。
        use_cosine: 若 True，先对向量做 L2 归一化并计算余弦相似；否则直接用欧氏距离、rbf等。

    Returns:
        clusters: dict，格式 {canonical_id: [member_id, ...], ...}。取每个连通分量中的第一个 ID 作为 canonical。
    """
    N = embedding_matrix.shape[0]
    if use_cosine:
        X = normalize(embedding_matrix, norm='l2')
        sim_matrix = cosine_similarity(X)
    else:
        # 这里也可改为其他相似度 / 距离计算
        sim_matrix = cosine_similarity(embedding_matrix)

    G = nx.Graph()
    # 添加所有节点
    for idx in ids:
        G.add_node(idx)

    # 对上三角矩阵进行遍历，构建边
    for i in range(N):
        for j in range(i+1, N):
            if sim_matrix[i, j] >= similarity_threshold:
                G.add_edge(ids[i], ids[j])

    # 找所有连通分量
    clusters = {}
    for component in nx.connected_components(G):
        component = list(component)
        canonical = component[0]
        clusters[canonical] = component

    return clusters


def graph_kcore_clustering(
    embedding_matrix: np.ndarray,
    ids: list,
    similarity_threshold: float = 0.85,
    k_core: int = 15,
    use_cosine: bool = True
) -> dict:
    """
    在 graph_threshold_clustering 基础上，先构建阈值图，再提取 k-core 子图作为最终聚类簇。
    只有度 >= k_core 的节点才保留到核心子图中；其余节点将被标记为单独簇或噪声。

    Args:
        embedding_matrix: (N, D) 嵌入向量矩阵。
        ids: 样本 ID 列表，与行顺序对应。
        similarity_threshold: 建边阈值。
        k_core: 需要保留的最小节点度数。
        use_cosine: 是否先归一化再计算余弦相似度。

    Returns:
        clusters: dict，格式 {canonical_id: [member_id,...], ...}，k-core 节点组成的簇。
                  其他未进入任何 k-core 子图的节点独立作为 {id: [id]} 键值对。
    """
    # 先构建连通阈值图
    N = embedding_matrix.shape[0]
    if use_cosine:
        X = normalize(embedding_matrix, norm='l2')
        sim_matrix = cosine_similarity(X)
    else:
        sim_matrix = cosine_similarity(embedding_matrix)

    G = nx.Graph()
    for idx in ids:
        G.add_node(idx)

    for i in range(N):
        for j in range(i+1, N):
            if sim_matrix[i, j] >= similarity_threshold:
                G.add_edge(ids[i], ids[j])

    # 提取 k-core 子图
    core_subgraph = nx.k_core(G, k=k_core)

    clusters = {}
    visited = set()

    # 核心子图中每个连通分量作为一个簇
    for component in nx.connected_components(core_subgraph):
        component = list(component)
        canonical = component[0]
        clusters[canonical] = component
        visited |= set(component)

    # 未在核心子图中的节点，各自单独成为一个簇或标记为“噪声簇”
    for idx in ids:
        if idx not in visited:
            clusters[idx] = [idx]

    return clusters
