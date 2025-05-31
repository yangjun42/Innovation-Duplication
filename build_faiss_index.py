# build_faiss_index.py

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1) 从 innovation_resolution.py 导入 load_and_combine_data() 和 get_embedding()
from innovation_resolution import load_and_combine_data, get_embedding

# 2) 从 utils/faiss_utils.py 导入构建与加载索引的函数
from utils.faiss_utils import build_faiss_index

# --- 常量定义 ---
# combined_df 文件路径可不硬编码，因为我们直接在代码里调用 load_and_combine_data()
FAISS_INDEX_PATH = os.path.join("data", "faiss_index.idx")
ID2META_PATH    = os.path.join("data", "id2meta.json")

def main():
    # —— 第 1 步：调用 load_and_combine_data() 得到合并后的 DataFrame —— 
    combined_df = load_and_combine_data()
    # combined_df 应该包含如下列（至少）：
    # ["source_type", "source_id", "source_english_id", "source_description", "relationship_type", 
    #  "target_english_id", "target_description", "Link Source Text", "data_source", ... ]
    
    # 只保留 source_type == "Innovation" 的行，并在 source_id 上去重
    df_inno = combined_df[combined_df["source_type"] == "Innovation"] \
                   .drop_duplicates(subset=["source_id"]) \
                   .reset_index(drop=True)
    print(f"共提取到 {len(df_inno)} 条“Innovation”节点，用于构建向量数据库。")
    
    # 如果没有找到任何 Innovation，则提前返回
    if df_inno.empty:
        print("未发现任何 ‘Innovation’，跳过 Faiss 索引构建。")
        return
    
    # —— 第 2 步：给每个 Innovation 拼接一段文本上下文 (context)，以供后续做 embedding —— 
    # 同时要为每个 Innovation 分配一个唯一的整型 ID(int64)，以便 Faiss 使用
    texts       = []   # 用于后续批量生成 embedding
    faiss_ids   = []   # 与上面 texts 一一对应的 int64 型 ID
    id2meta     = {}   # 最终要 dump 的 “ID(str) → metadata(dict)” 映射（JSON 里 key 必须是 str）
    
    for idx, row in tqdm(df_inno.iterrows(), total=len(df_inno), desc="准备文本 + 元数据"):
        orig_id = row["source_id"]
        # 1) 构造一个整型 ID：如果 source_id 本身就是 int，可以直接用；否则用一个简单的自增索引
        #    这里只以当前行号 idx + 1 作为 int_id（保证从 1 开始，且不会和 0 冲突）
        int_id = np.int64(idx + 1)
        
        # 2) 拼接文本上下文：至少包含 “名称 + 描述” 
        name = str(row.get("source_english_id", "")) or ""
        desc = str(row.get("source_description", "")) or ""
        context = f"Innovation name: {name}. Description: {desc}."
        
        # 3) 如果有 DEVELOPED_BY 关系，就把“组织名称”加进 context 里
        dev_by_list = combined_df[
            (combined_df["source_id"] == orig_id) &
            (combined_df["relationship_type"] == "DEVELOPED_BY")
        ]["target_english_id"] \
        .dropna() \
        .unique() \
        .tolist()
        
        if dev_by_list:
            context += " Developed by: " + ", ".join(dev_by_list) + "."
        
        # 4) （可选）把该 innovation 跟其他 target 的关系描述也塞进去
        related_rows = combined_df[combined_df["source_id"] == orig_id]
        for _, rel_row in related_rows.iterrows():
            rel_desc    = str(rel_row.get("relationship description", "")).strip()
            tgt_name    = str(rel_row.get("target_english_id", "")).strip()
            tgt_desc    = str(rel_row.get("target_description", "")).strip()
            # 只有当 rel_desc、tgt_name、tgt_desc 都非空时，才把它插入
            if rel_desc and tgt_name and tgt_desc:
                context += f" {rel_desc} {tgt_name}, which is described as: {tgt_desc}."
        
        # 记下要做 embedding 的文本
        texts.append(context)
        faiss_ids.append(int_id)
        
        # 记下 metadata，检索时只要拿到 int_id，就能通过 id2meta[int_id str] 拿到这一行的详尽信息
        id2meta[str(int_id)] = {
            "orig_source_id": orig_id,
            "source_english_id": row.get("source_english_id", ""),
            "source_description": row.get("source_description", ""),
            "Link Source Text": row.get("Link Source Text", ""),
            "data_source": row.get("data_source", ""),
            # 如果你还想把 target、relationship_type 等都保留，可以在此补充
        }
    
    # —— 第 3 步：批量生成 embedding —— 
    print("开始批量生成文本 embedding ...")
    embedding_list = []
    for txt in tqdm(texts, desc="Embedding 文本"):
        vec = get_embedding(txt, model=None)   # get_embedding 会返回 np.ndarray，长度例如 1536
        embedding_list.append(vec)
    
    embedding_matrix = np.vstack(embedding_list).astype(np.float32)  # 形状 (N, D)，dtype 必须是 float32
    id_array         = np.array(faiss_ids, dtype=np.int64)          # 形状 (N,)
    
    # —— 第 4 步：调用 Faiss 工具把 embedding_matrix 和 id_array 构成向量索引，并持久化 —— 
    print("调用 faiss_utils.build_faiss_index 来构建索引 ...")
    idx = build_faiss_index(
        embedding_matrix=embedding_matrix,
        ids=id_array,
        index_path=FAISS_INDEX_PATH,
        index_type="Flat"   # 如需更复杂索引，可改成 "HNSW"、"IVF"、"IVF_PQ" 并传入相应参数
        # 如果用 IVF: 在此加上 nlist=xx；如果用 PQ: 在此加上 nlist=xx, m_pq=xx, nbits=xx
    )
    
    # —— 第 5 步：把 ID→metadata 映射存成 JSON —— 
    print(f"正在保存 ID→metadata 映射到 {ID2META_PATH} ...")
    os.makedirs(os.path.dirname(ID2META_PATH), exist_ok=True)
    with open(ID2META_PATH, "w", encoding="utf-8") as f:
        json.dump(id2meta, f, ensure_ascii=False, indent=2)
    print("ID→metadata 映射保存完成。")

    print("Faiss 索引构建流程全部完成！")

if __name__ == "__main__":
    main()