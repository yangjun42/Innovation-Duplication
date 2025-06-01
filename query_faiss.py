# query_faiss.py

import json
import numpy as np

from utils.faiss_utils import load_faiss_index, search_faiss_index
from innovation_resolution import get_embedding

FAISS_INDEX_PATH = "data/vector-database-2-Deduplication/faiss_index.idx"
ID2META_PATH    = "data/vector-database-2-Deduplication/id2meta.json"

def main():
    # 1) 加载 Faiss 索引
    idx = load_faiss_index(FAISS_INDEX_PATH)

    # 2) 加载 int_id → metadata 的映射
    with open(ID2META_PATH, "r", encoding="utf-8") as f:
        id2meta = json.load(f)

    # 3) 用户输入要检索的文本
    query_text = "Looking for innovations about nuclear reactor decommissioning"
    q_vec = get_embedding(query_text, model='text-embedding-3-large').astype(np.float32).reshape(1, -1)

    # 4) 在 Faiss 上做 top-5 检索
    top_k = 10
    distances, indices = search_faiss_index(idx, q_vec, top_k)
    # distances: shape (1, top_k)，indices: shape (1, top_k)

    print("=== Faiss 检索 Top-20 结果 ===")
    for rank, (dist, iid) in enumerate(zip(distances[0], indices[0]), start=1):
        if iid == -1:
            continue
        meta = id2meta.get(str(int(iid)), {})
        print(f"Rank {rank}: int_id={iid}, distance={dist:.4f}, metadata={meta}")

if __name__ == "__main__":
    main()