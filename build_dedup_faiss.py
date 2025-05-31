#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_dedup_faiss.py

本脚本将:
  1. 从 company + VTT domain CSV 里加载并合并出 df_relationships
  2. 对所有 Innovation 节点做去重（resolve_innovation_duplicates）
  3. 对每个 dedup 后的簇做“文本合并”，得到一段长文本
  4. 对每段文本做 embedding，得到一个向量矩阵 (num_clusters, D)
  5. 用 Faiss 构建索引，并把结果写到 data/faiss_index.idx
  6. 同时把每个 cluster_id → “元信息” 存到 data/id2meta.json，方便后续检索时拿回原始信息
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 如果项目里已经有一个 innovation_resolution.py，就直接 import 其中的方法
from innovation_resolution import (
    load_and_combine_data,
    resolve_innovation_duplicates,
    get_embedding
)

# 导入我们刚才写好的 Faiss 工具
from utils.faiss_utils import build_faiss_index

# ========== 配置区 ==========
# Faiss 索引要写到的路径
FAISS_INDEX_PATH = "data/faiss_index.idx"
# Metadata 存到哪个 JSON
ID2META_PATH = "data/id2meta.json"

# 聚类时阈值（与 resolve_innovation_duplicates 保持一致）
SIMILARITY_THRESHOLD = 0.85

def main():
    # 1) 加载并合并 CSV，得到一个大 DataFrame
    print("STEP1: Loading and combining raw CSV → DataFrame")
    df_relationships = load_and_combine_data()

    # 2) 调用现有的去重函数，得到 {innovation_id: canonical_id} 映射
    print("\nSTEP2: Deduplicating innovations ...")
    # 这里可以传入 cache_config，如果不需要缓存就直接默认 None
    canonical_mapping = resolve_innovation_duplicates(df_relationships, model=None, cache_config=None)

    # 如果真的没有任何 Innovation，就直接退出
    if not canonical_mapping:
        print("No innovations found; exiting.")
        return

    # 3) 收集“每个 canonical_id 对应的这一簇里所有 member ID”
    clusters = {}
    for inno_id, can_id in canonical_mapping.items():
        clusters.setdefault(can_id, []).append(inno_id)

    # 4) 对每个簇做“文本拼接/聚合”：把 source_description、relationship_description、target_description
    #    这些字段都从 df_relationships 里取出来，再一股脑拼到一个长文本里
    print("\nSTEP3: Building aggregated text for each cluster ...")
    cluster_texts = {}    # { canonical_id: "拼接后的长文本" }
    cluster_meta = {}     # { canonical_id: { "members": [...], "aggregated_fields": {...} } }

    # 先转换为 dict-of-lists 方便 filter 查询
    # df_relationships 里每行对应一个 “node→node 的关系”，
    # 对于属于同一个 innovation 簇的成员，我们都要把所有相关列都拼起来
    for can_id, members in tqdm(clusters.items(), desc="Aggregating cluster contents"):
        # 我要把以下这些列拼成一个“超长文本”：
        #  - 该 innovation 在不同行的 source_english_id, source_description
        #  - 该 innovation 在不同行的 relationship_description, target_english_id, target_description
        agg_parts = []
        meta_info = {
            "members": members,      # 该簇包含的所有 innovation_id
            "data_sources": set(),   # 后面可存簇里所有行的 data_source 值
            "source_urls": set(),    # 存一遍 “Link Source Text”
        }

        for mid in members:
            # 筛出 df_relationships 里所有 source_id == mid 的行
            sub_df = df_relationships[df_relationships["source_id"] == mid]
            for _, row in sub_df.iterrows():
                # 拿出 3 个重要字段
                s_ename = str(row.get("source_english_id", "")).strip()
                s_desc  = str(row.get("source_description", "")).strip()
                r_desc  = str(row.get("relationship description", "")).strip()
                t_ename = str(row.get("target_english_id", "")).strip()
                t_desc  = str(row.get("target_description", "")).strip()
                link_url = str(row.get("Link Source Text", "")).strip()
                ds = str(row.get("data_source", "")).strip()

                # 拼到 agg_parts 里
                if s_ename:
                    agg_parts.append(f"Innovation name: {s_ename}.")
                if s_desc:
                    agg_parts.append(f"Description: {s_desc}.")
                if r_desc and t_ename:
                    # 比如 "DEVELOPED_BY: Fortum..., target_desc"
                    agg_parts.append(f"{r_desc} {t_ename}.")
                if t_desc:
                    agg_parts.append(f"{t_ename} described as: {t_desc}.")
                if link_url:
                    link_url = link_url.strip()
                    agg_parts.append(f"Source link: {link_url}.")
                if ds:
                    meta_info["data_sources"].add(ds)
                if link_url:
                    meta_info["source_urls"].add(link_url)

        # 用空格把所有片段连接起来
        aggregated_text = " ".join(agg_parts)
        cluster_texts[can_id] = aggregated_text

        # 把 set 转成 list 存到 meta 里
        meta_info["data_sources"] = list(meta_info["data_sources"])
        meta_info["source_urls"] = list(meta_info["source_urls"])
        cluster_meta[can_id] = meta_info

    # 5) 对每段聚合好的文本做 embedding → 得到一个矩阵 (num_clusters, D)
    print("\nSTEP4: Computing embeddings for each aggregated cluster text ...")
    cluster_ids = list(cluster_texts.keys())  # 自定义 ID 列表，对应矩阵的每一行
    embedding_list = []
    for can_id in tqdm(cluster_ids, desc="Embedding cluster texts"):
        text = cluster_texts[can_id]
        vec = get_embedding(text, model=None)  # 返回一个 ndarray, 形状 (D,)
        embedding_list.append(vec)
    embedding_matrix = np.vstack(embedding_list).astype(np.float32)  # (num_clusters, D)

    # 6) 调用 Faiss 构建索引，并保存 metadata JSON
    print("\nSTEP5: Building Faiss index ...")
    # Faiss 里 ID 必须是 int64，这里 cluster_ids 里可能是 str，需要转换成 int 或者自定义一套 int 映射。
    # 业务上常见的做法：把 can_id（原本 string）先映射成连续的 int，比如 0,1,2...
    int_ids = np.arange(len(cluster_ids), dtype=np.int64)
    id_map_str2int = {cluster_ids[i]: int_ids[i] for i in range(len(cluster_ids))}

    # 保存 Faiss 索引：embedding_matrix 对应 int_ids
    faiss_idx = build_faiss_index(
        embedding_matrix=embedding_matrix,
        ids=int_ids,
        index_path=FAISS_INDEX_PATH,
        index_type="Flat"        # 可改为 "HNSW" / "IVF" 等
    )

    # 7) 把 “int_id → 原始 canonical_id(str) → meta 信息” 存到一个 JSON
    print("\nSTEP6: Saving ID→metadata mapping ...")
    out_meta = {}
    for can_str_id, meta in cluster_meta.items():
        int_id = int(id_map_str2int[can_str_id])
        out_meta[str(int_id)] = {
            "canonical_id": can_str_id,
            "members": meta["members"],
            "data_sources": meta["data_sources"],
            "source_urls": meta["source_urls"],
            # 我们也可以把“拼接后的长文本”也记下来
            "aggregated_text": cluster_texts[can_str_id]
        }

    # 写到 JSON 文件
    os.makedirs(os.path.dirname(ID2META_PATH), exist_ok=True)
    with open(ID2META_PATH, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)
    print(f">>> Metadata 已保存到 {ID2META_PATH}")

    print("\n=== 完成：已生成去重后的 Faiss 矢量数据库和 metadata 映射 ===")
    print(f"  Faiss Index: {FAISS_INDEX_PATH}")
    print(f"  Metadata JSON: {ID2META_PATH}")


if __name__ == "__main__":
    main()