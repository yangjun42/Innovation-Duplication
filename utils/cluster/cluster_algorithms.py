# utils/cluster/cluster_algorithms.py

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import hdbscan


def cluster_hdbscan(
    embedding_matrix: np.ndarray,
    min_cluster_size: int = 2,
    metric: str = 'cosine',
    cluster_selection_method: str = 'eom'
) -> np.ndarray:
    """
    使用 HDBSCAN 对嵌入向量进行聚类。
    """
    if metric == 'cosine':
        embedding_matrix = normalize(embedding_matrix, norm='l2')
        metric_used = 'euclidean'
    else:
        metric_used = metric

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric_used,
        cluster_selection_method=cluster_selection_method
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels


def cluster_kmeans(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    random_state: int = 42
) -> np.ndarray:
    """
    使用 K-Means 对嵌入向量进行聚类。
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding_matrix)
    return labels


def cluster_agglomerative(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    affinity: str = 'cosine',
    linkage: str = 'average'
) -> np.ndarray:
    """
    使用层次聚类（Agglomerative Clustering）进行聚类。
    """
    if affinity == 'cosine':
        embedding_matrix = normalize(embedding_matrix, norm='l2')
        affinity_used = 'euclidean'
    else:
        affinity_used = affinity

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=affinity_used,
        linkage=linkage
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels


def cluster_spectral(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 10
) -> np.ndarray:
    """
    使用谱聚类（Spectral Clustering）对嵌入向量聚类。
    """
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=42
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels