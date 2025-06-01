# utils/cluster/cluster_algorithms.py

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import hdbscan
from typing import List, Dict

def cluster_hdbscan(
    embedding_matrix: np.ndarray,
    ids: List[str],
    min_cluster_size: int = 2,
    metric: str = 'cosine',
    cluster_selection_method: str = 'eom'
) -> Dict[str, List[str]]:
    """
    使用 HDBSCAN 对嵌入向量进行聚类，并直接返回 { canonical_id: [member_id, ...], ... }。

    Args:
        embedding_matrix: (N, D) 数组，N 个嵌入向量。
        ids: 长度为 N 的样本 ID 列表，用作节点标签。
        min_cluster_size: HDBSCAN 的最小簇大小。
        metric: 度量方式，若为 'cosine' 则先 normalize 后用欧氏距离。
        cluster_selection_method: HDBSCAN 的簇选择策略。

    Returns:
        clusters_dict: { canonical_id: [member_id, ...], ... }，每个簇的第一个成员作为 canonical_id。
    """
    # 1) 如果使用余弦相似度，需要先做 L2 归一化，并把 metric 改为 'euclidean'
    if metric == 'cosine':
        X = normalize(embedding_matrix, norm='l2')
        metric_used = 'euclidean'
    else:
        X = embedding_matrix
        metric_used = metric

    # 2) 调用 HDBSCAN 拿到 labels
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric_used,
        cluster_selection_method=cluster_selection_method
    )
    labels = clusterer.fit_predict(X)  # shape = (N, )

    # 3) 按 label 把 ids 分组
    temp_groups: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        if lab == -1:
            # “-1” 被认为是噪声，把它作为一个单独簇，key 设成 “noise_<id>”
            noise_key = f"noise_{ids[idx]}"
            temp_groups.setdefault(noise_key, []).append(ids[idx])
        else:
            temp_groups.setdefault(int(lab), []).append(ids[idx])

    # 4) 最后把每个簇里的第一个成员设为 canonical_id，并形成最终字典
    clusters_dict: Dict[str, List[str]] = {}
    for lab_key, members in temp_groups.items():
        # 如果 lab_key 是整型（0, 1, 2, …），则 canonical_id = members[0]
        # 如果 lab_key 是 “noise_<id>”，也同样 members[0] 做 canonical
        canonical_id = members[0]
        clusters_dict[canonical_id] = members

    return clusters_dict


def cluster_kmeans(
    embedding_matrix: np.ndarray,
    ids: List[str],
    n_clusters: int = 450,
    random_state: int = 42
) -> Dict[str, List[str]]:
    """
    使用 K-Means 对嵌入向量进行聚类，并直接返回 { canonical_id: [member_id, ...], ... }。
    """
    # 1) 拿到 labels 数组
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding_matrix)  # shape = (N, )

    # 2) 按 label 分组
    temp_groups: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        temp_groups.setdefault(int(lab), []).append(ids[idx])

    # 3) 构造最终字典
    clusters_dict: Dict[str, List[str]] = {}
    for lab_key, members in temp_groups.items():
        canonical_id = members[0]
        clusters_dict[canonical_id] = members

    return clusters_dict


def cluster_agglomerative(
    embedding_matrix: np.ndarray,
    ids: List[str],
    n_clusters: int = 450,
    affinity: str = 'cosine',
    linkage: str = 'average'
) -> Dict[str, List[str]]:
    """
    使用层次聚类（AgglomerativeClustering）进行聚类，并直接返回 { canonical_id: [member_id, ...], ... }。
    """
    # 如果 affinity 是 'cosine'，先做 L2 归一化，实际用欧氏距离
    if affinity == 'cosine':
        X = normalize(embedding_matrix, norm='l2')
        affinity_used = 'euclidean'
    else:
        X = embedding_matrix
        affinity_used = affinity

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=affinity_used,
        linkage=linkage
    )
    labels = clusterer.fit_predict(X)  # shape = (N, )

    # 按 label 把 ids 分组
    temp_groups: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        temp_groups.setdefault(int(lab), []).append(ids[idx])

    # 构造最终字典
    clusters_dict: Dict[str, List[str]] = {}
    for lab_key, members in temp_groups.items():
        canonical_id = members[0]
        clusters_dict[canonical_id] = members

    return clusters_dict


def cluster_spectral(
    embedding_matrix: np.ndarray,
    ids: List[str],
    n_clusters: int = 450,
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 10
) -> Dict[str, List[str]]:
    """
    使用谱聚类（SpectralClustering）对嵌入向量聚类，并直接返回 { canonical_id: [member_id, ...], ... }。
    """
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=42
    )
    labels = clusterer.fit_predict(embedding_matrix)  # shape = (N, )

    # 按 label 分组
    temp_groups: Dict[int, List[str]] = {}
    for idx, lab in enumerate(labels):
        temp_groups.setdefault(int(lab), []).append(ids[idx])

    # 构造最终字典
    clusters_dict: Dict[str, List[str]] = {}
    for lab_key, members in temp_groups.items():
        canonical_id = members[0]
        clusters_dict[canonical_id] = members

    return clusters_dict