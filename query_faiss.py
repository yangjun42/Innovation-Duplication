# query_faiss.py

import json
import numpy as np

from utils.faiss_utils import load_faiss_index, search_faiss_index
from innovation_resolution import get_embedding

# ----------------- 常量：索引与路径 -----------------
FAISS_INDEX_PATH = "data/faiss_index.idx"
ID2META_PATH    = "data/id2meta.json"

# 2D 与 3D 节点坐标 JSON 路径
NODE2D_JSON_PATH = "results/node_positions_tufte_2d.json"
NODE3D_JSON_PATH = "results/node_positions_tufte_3d.json"


def load_node_positions_2d(json_path: str) -> dict:
    """
    从 JSON 文件加载 2D 坐标，返回 { node_id: (x, y) }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pos2d = {}
    for node in data.get("nodes", []):
        nid = node["id"]
        x = node["x"]
        y = node["y"]
        pos2d[nid] = (x, y)
    return pos2d


def load_node_positions_3d(json_path: str) -> dict:
    """
    从 JSON 文件加载 3D 坐标，返回 { node_id: (x, y, z) }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pos3d = {}
    for node in data.get("nodes", []):
        nid = node["id"]
        x = node["x"]
        y = node["y"]
        z = node.get("z", None)
        pos3d[nid] = (x, y, z)
    return pos3d


def main():
    # 1) 加载 Faiss 索引
    idx = load_faiss_index(FAISS_INDEX_PATH)

    # 2) 加载 int_id → metadata 的映射
    with open(ID2META_PATH, "r", encoding="utf-8") as f:
        id2meta = json.load(f)

    # 2.5) 加载 2D、3D 坐标字典
    pos2d_dict = load_node_positions_2d(NODE2D_JSON_PATH)
    pos3d_dict = load_node_positions_3d(NODE3D_JSON_PATH)

    # 3) 用户输入要检索的文本
    query_text = "Looking for innovations about nuclear reactor decommissioning"
    # get_embedding 内部返回的是 numpy 数组，这里我们假设使用 float32
    q_vec = get_embedding(query_text, model='text-embedding-3-large').astype(np.float32).reshape(1, -1)

    # 4) 在 Faiss 上做 top-10 检索
    top_k = 10
    distances, indices = search_faiss_index(idx, q_vec, top_k)
    # distances: shape (1, top_k)
    # indices:   shape (1, top_k)

    print("=== Faiss 检索 Top-10 结果 ===")
    for rank, (dist, iid) in enumerate(zip(distances[0], indices[0]), start=1):
        if iid == -1:
            # Faiss 里可能返回 -1 表示“空”
            continue

        meta = id2meta.get(str(int(iid)), {})
        # 假设 meta 至少包含一个 key="canonical_id"，值为 networkx 图中的节点 ID 字符串
        node_id = meta.get("canonical_id")
        print("\n", "node_id:", node_id)

        # 默认坐标都用 None 填充
        pos2d = (None, None)
        pos3d = (None, None, None)
        if node_id is not None:
            pos2d = pos2d_dict.get(node_id, (None, None))
            pos3d = pos3d_dict.get(node_id, (None, None, None))

        print(f"Rank {rank}: int_id={iid}, distance={dist:.4f}")
        print(f"    innovation_id = {node_id}")
        print(f"    metadata      = {meta}")
        print(f"    position2D    = {pos2d}")
        print(f"    position3D    = {pos3d}")
        print("-----------------------------------------\n")


if __name__ == "__main__":
    main()