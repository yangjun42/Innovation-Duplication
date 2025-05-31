# utils/faiss_utils.py

import os
import numpy as np
import faiss

def build_faiss_index(
    embedding_matrix: np.ndarray,
    ids: np.ndarray,
    index_path: str,
    index_type: str = "Flat",
    **index_kwargs
) -> faiss.Index:
    """
    用给定的 embedding_matrix (N, D) 和对应的 ids (长度 N) 构建 Faiss 索引，并保存到本地。

    Args:
        embedding_matrix (np.ndarray): 形如 (N, D) 的浮点向量矩阵 (dtype=float32)
        ids (np.ndarray): 长度为 N 的 int64 数组，对应每个向量的唯一 ID
        index_path (str): 保存索引文件的路径，例如 "data/faiss_index.idx"
        index_type (str, optional): 索引类型，支持 "Flat"、"HNSW"、"IVF"、"IVF_PQ" 等。默认 "Flat"
        **index_kwargs: 传给具体索引构造的参数，比如：
            - HNSW: m, efConstruction
            - IVF: nlist
            - IVF_PQ: nlist, m_pq, nbits

    Returns:
        idx (faiss.Index): 已经训练并添加好向量的 faiss.Index 对象（同时写文件到 index_path）
    """

    N, D = embedding_matrix.shape

    # 1) 根据 index_type 来构建对应的 Faiss 索引骨架
    _type = index_type.lower()
    if _type == "flat":
        idx_inner = faiss.IndexFlatL2(D)

    elif _type == "hnsw":
        # HNSWFlat (允许 add())
        m = index_kwargs.get("m", 32)
        efC = index_kwargs.get("efConstruction", 200)
        idx_inner = faiss.IndexHNSWFlat(D, m)
        idx_inner.hnsw.efConstruction = efC

    elif _type == "ivf":
        # IVF Flat，需要先 train
        nlist = index_kwargs.get("nlist", 100)
        quantizer = faiss.IndexFlatL2(D)
        idx_inner = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
        idx_inner.train(embedding_matrix)

    elif _type == "ivf_pq":
        # IVF + PQ，需要先 train
        nlist = index_kwargs.get("nlist", 100)
        m_pq  = index_kwargs.get("m_pq", 8)
        nbits = index_kwargs.get("nbits", 8)
        quantizer = faiss.IndexFlatL2(D)
        idx_inner = faiss.IndexIVFPQ(quantizer, D, nlist, m_pq, nbits)
        idx_inner.train(embedding_matrix)

    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    # 2) 为了让 Faiss 返回自定义 ID，这里用 IndexIDMap 包裹一下
    idx = faiss.IndexIDMap(idx_inner)

    # 3) 批量添加向量与其对应的 IDs（IDs 必须是 int64）
    embedding_matrix = embedding_matrix.astype(np.float32)
    ids = ids.astype(np.int64)
    idx.add_with_ids(embedding_matrix, ids)

    # 4) 保存到磁盘
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(idx, index_path)
    print(f">>> Faiss 索引已保存到 {index_path}，共 {idx.ntotal} 个向量。")

    return idx


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    从本地文件加载已构建完成的 Faiss 索引。

    Args:
        index_path (str): .idx 文件路径

    Returns:
        idx (faiss.Index): 加载好的 Faiss 索引
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss 索引文件不存在: {index_path}")
    idx = faiss.read_index(index_path)
    print(f">>> 从磁盘加载 Faiss 索引：{index_path}，共 {idx.ntotal} 向量。")
    return idx


def search_faiss_index(
    idx: faiss.Index,
    query_vectors: np.ndarray,
    top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 Faiss 索引上执行 k-NN 检索。

    Args:
        idx (faiss.Index): 已加载的索引
        query_vectors (np.ndarray): 形如 (Q, D) 的查询向量 (dtype=float32)
        top_k (int): 返回最相似的 k 个结果

    Returns:
        distances (np.ndarray): 形如 (Q, top_k) 的 L2 距离矩阵
        indices   (np.ndarray): 形如 (Q, top_k) 的对应 ID 矩阵
    """
    queries = query_vectors.astype(np.float32)
    distances, indices = idx.search(queries, top_k)
    return distances, indices