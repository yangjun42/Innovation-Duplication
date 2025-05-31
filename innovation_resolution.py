#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Innovation Resolution Challenge Solution

This script implements solutions for identifying duplicate innovations
and creating a consolidated view of innovation relationships.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional, Union, Protocol
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain_core.documents import Document

from langchain.prompts import PromptTemplate


import math




def chat_loop(llm, embed_model, vector_store, innovation_features):

    chatbot_prompt = PromptTemplate.from_template("""

        You're a smart assistant helping extract insights from VTT innovation relationships.

        Context:
        {context}

        According to the context, answer this question:
        {question}
    """)

    chat_bot = chatbot_prompt | llm

    while True:

        query = input("\nAsk about innovation: ")

        # For testing.
        # query = "nuclear decommissioning"

        if query.lower() == "exit":
            break

        query_embedding = get_embedding(query, embed_model)

        results = vector_store.vector_search(query, k=3)

        context = "\n".join([innovation_features[doc.page_content] for doc in results])

        # print(context)

        llm_result = chat_bot.invoke({"context":context, "question":query})
        answer = llm_result.content

        print("\n --- --- --- [Answer] --- --- ---")
        print(answer)



# Import local modules
from innovation_utils import (
    compute_similarity_matrix,
    find_potential_duplicates,
    calculate_innovation_statistics
)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Set up paths
GRAPH_DOCS_COMPANY = os.path.join(DATA_DIR, 'graph_docs_names_resolved')
GRAPH_DOCS_VTT = os.path.join(DATA_DIR, 'graph_docs_vtt_domain_names_resolved')
DATAFRAMES_DIR = os.path.join(DATA_DIR, 'dataframes')

# Create results directory if it doesn't exist
try:
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    RESULTS_DIR = '.'  # Use current directory as fallback

# Set up plotting style
sns.set_theme(style="whitegrid")

# 添加缓存模块
class CacheBackend(Protocol):
    """缓存后端接口协议"""
    
    def load(self) -> Dict:
        """加载缓存数据"""
        ...
    
    def save(self, data: Dict) -> bool:
        """保存数据到缓存"""
        ...
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项"""
        ...
    
    def update(self, data: Dict) -> None:
        """批量更新缓存"""
        ...
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """获取缓存中缺失的键"""
        ...
    
    def contains(self, key: str) -> bool:
        """检查缓存是否包含指定键"""
        ...


class JsonFileCache:
    """基于JSON文件的缓存实现"""
    
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache_data = {}
        self.loaded = False
    
    def load(self) -> Dict:
        if self.loaded:
            return self.cache_data
            
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
                print(f"Loaded {len(self.cache_data)} items from cache at {self.cache_path}")
                self.loaded = True
                return self.cache_data
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        else:
            print(f"Cache file not found at {self.cache_path}")
            return {}
    
    def save(self, data: Dict) -> bool:
        try:
            # 确保目录存在
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} items to cache at {self.cache_path}")
            self.cache_data = data
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        if not self.loaded:
            self.load()
        return self.cache_data.get(key, None)
    
    def set(self, key: str, value: Any) -> None:
        if not self.loaded:
            self.load()
        self.cache_data[key] = value
    
    def update(self, data: Dict) -> None:
        if not self.loaded:
            self.load()
        self.cache_data.update(data)
        self.save(self.cache_data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        if not self.loaded:
            self.load()
        return [k for k in required_keys if k not in self.cache_data]
    
    def contains(self, key: str) -> bool:
        if not self.loaded:
            self.load()
        return key in self.cache_data


class MemoryCache:
    """基于内存的缓存实现"""
    
    def __init__(self):
        self.cache_data = {}
    
    def load(self) -> Dict:
        return self.cache_data
    
    def save(self, data: Dict) -> bool:
        self.cache_data = data
        return True
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache_data.get(key, None)
    
    def set(self, key: str, value: Any) -> None:
        self.cache_data[key] = value
    
    def update(self, data: Dict) -> None:
        self.cache_data.update(data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        return [k for k in required_keys if k not in self.cache_data]
    
    def contains(self, key: str) -> bool:
        return key in self.cache_data


class EmbeddingCache:
    """
    可插拔的嵌入向量缓存系统，支持不同的存储后端。
    
    当前支持:
    - 文件缓存 (JSON)
    - 内存缓存
    
    可扩展支持:
    - 数据库缓存
    - 分布式缓存
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None, cache_path: str = "./embedding_vectors.json", 
                backend_type: str = "json", use_cache: bool = True):
        """
        初始化嵌入缓存系统。
        
        Args:
            backend: 自定义缓存后端实现
            cache_path: 缓存文件路径 (仅用于文件缓存)
            backend_type: 后端类型 ('json' 或 'memory')
            use_cache: 是否启用缓存
        """
        self.use_cache = use_cache
        
        if not use_cache:
            self.backend = None
            return
            
        if backend is not None:
            self.backend = backend
        elif backend_type == "json":
            self.backend = JsonFileCache(cache_path)
        elif backend_type == "memory":
            self.backend = MemoryCache()
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    def load(self) -> Dict:
        """
        加载缓存数据。
        
        Returns:
            Dict: 加载的缓存数据
        """
        if not self.use_cache or self.backend is None:
            return {}
        return self.backend.load()
    
    def save(self, data: Dict) -> bool:
        """
        保存数据到缓存。
        
        Args:
            data: 要保存的数据
            
        Returns:
            bool: 是否成功保存
        """
        if not self.use_cache or self.backend is None:
            return False
        return self.backend.save(data)
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取值。
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，如不存在返回None
        """
        if not self.use_cache or self.backend is None:
            return None
        return self.backend.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值。
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        if not self.use_cache or self.backend is None:
            return
        self.backend.set(key, value)
    
    def update(self, data: Dict) -> None:
        """
        批量更新缓存。
        
        Args:
            data: 要更新的数据
        """
        if not self.use_cache or self.backend is None:
            return
        self.backend.update(data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """
        获取缓存中缺失的键。
        
        Args:
            required_keys: 需要的键列表
            
        Returns:
            List[str]: 缓存中不存在的键列表
        """
        if not self.use_cache or self.backend is None:
            return required_keys
        return self.backend.get_missing_keys(required_keys)
    
    def contains(self, key: str) -> bool:
        """
        检查缓存是否包含指定键。
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 是否存在
        """
        if not self.use_cache or self.backend is None:
            return False
        return self.backend.contains(key)


class CacheFactory:
    """缓存工厂，用于创建不同类型的缓存实例"""
    
    @staticmethod
    def create_cache(cache_type: str = "embedding", 
                    backend_type: str = "json",
                    cache_path: str = "./embedding_vectors.json",
                    use_cache: bool = True) -> Union[EmbeddingCache, Any]:
        """
        创建缓存实例。
        
        Args:
            cache_type: 缓存类型 ('embedding' 或自定义)
            backend_type: 后端类型 ('json' 或 'memory')
            cache_path: 缓存文件路径
            use_cache: 是否启用缓存
            
        Returns:
            Union[EmbeddingCache, Any]: 缓存实例
        """
        if cache_type == "embedding":
            return EmbeddingCache(
                backend_type=backend_type,
                cache_path=cache_path,
                use_cache=use_cache
            )
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")


def load_and_combine_data() -> pd.DataFrame:
    """
    Load relationship data from both company websites and VTT domain,
    combine them into a single dataframe.
    
    Returns:
        pd.DataFrame: Combined relationship dataframe
    """
    print("Loading data from company websites...")
    
    # Load company domain data
    df_company = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'vtt_mentions_comp_domain.csv'))
    df_company = df_company[df_company['Website'].str.startswith('www.')]
    df_company['source_index'] = df_company.index

    # Extract relationship triplets from company domain
    df_relationships_comp_url = pd.DataFrame()
    
    with tqdm(total=len(df_company), desc="Processing company data") as pbar:
        for i, row in df_company.iterrows():
            try:
                file_path = os.path.join(GRAPH_DOCS_COMPANY, f"{row['Company name'].replace(' ','_')}_{i}.pkl")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        graph_doc = pickle.load(f)[0]
                    
                    node_description = {}
                    node_en_id = {}
                    for node in graph_doc.nodes:
                        node_description[node.id] = node.properties['description']
                        node_en_id[node.id] = node.properties['english_id']

                    relationship_rows = []
                    for j in range(len(graph_doc.relationships)):
                        rel = graph_doc.relationships[j]
                        relationship_rows.append({
                            "Document number": row['source_index'],
                            "Source Company": row["Company name"],
                            "relationship description": rel.properties['description'],
                            "source_id": rel.source,
                            "source_type": rel.source_type,
                            "source_english_id": node_en_id.get(rel.source, None),
                            "source_description": node_description.get(rel.source, None),
                            "relationship_type": rel.type,
                            "target_id": rel.target,
                            "target_type": rel.target_type,
                            "target_english_id": node_en_id.get(rel.target, None),
                            "target_description": node_description.get(rel.target, None),
                            "Link Source Text": row["Link"],
                            "Source Text": row["text_content"],
                            "data_source": "company_website"
                        })

                    df_relationships_comp_url = pd.concat([df_relationships_comp_url, pd.DataFrame(relationship_rows)], ignore_index=True)
            except Exception as e:
                print(f"Error processing {i}: {e}")
            pbar.update(1)
    
    print(f"Processed {len(df_relationships_comp_url)} relationships from company websites")
    
    # Load VTT domain data
    print("Loading data from VTT domain...")
    df_vtt_domain = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'comp_mentions_vtt_domain.csv'))
    
    # Extract relationship triplets from VTT domain
    df_relationships_vtt_domain = pd.DataFrame()
    
    with tqdm(total=len(df_vtt_domain), desc="Processing VTT domain data") as pbar:
        for index_source, row in df_vtt_domain.iterrows():
            try:
                file_path = os.path.join(GRAPH_DOCS_VTT, f"{row['Vat_id'].replace(' ','_')}_{index_source}.pkl")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        graph_doc = pickle.load(f)[0]
                    
                    node_description = {}
                    node_en_id = {}
                    for node in graph_doc.nodes:
                        node_description[node.id] = node.properties['description']
                        node_en_id[node.id] = node.properties['english_id']

                    relationship_rows = []
                    for j in range(len(graph_doc.relationships)):
                        rel = graph_doc.relationships[j]
                        relationship_rows.append({
                            "Document number": index_source,
                            "VAT id": row["Vat_id"],
                            "relationship description": rel.properties['description'],
                            "source_id": rel.source,
                            "source_type": rel.source_type,
                            "source_english_id": node_en_id.get(rel.source, None),
                            "source_description": node_description.get(rel.source, None),
                            "relationship_type": rel.type,
                            "target_id": rel.target,
                            "target_type": rel.target_type,
                            "target_english_id": node_en_id.get(rel.target, None),
                            "target_description": node_description.get(rel.target, None),
                            "Link Source Text": row["source_url"],
                            "Source Text": row["main_body"],
                            "data_source": "vtt_website"
                        })

                    df_relationships_vtt_domain = pd.concat([df_relationships_vtt_domain, pd.DataFrame(relationship_rows)], ignore_index=True)
            except Exception as e:
                print(f"Error processing {index_source}: {e}")
            pbar.update(1)
    
    print(f"Processed {len(df_relationships_vtt_domain)} relationships from VTT domain")
    
    # Rename columns to align dataframes
    df_relationships_vtt_domain = df_relationships_vtt_domain.rename(columns={"VAT id": "Source Company"})
    
    # Combine dataframes
    combined_df = pd.concat([df_relationships_comp_url, df_relationships_vtt_domain], ignore_index=True)
    print(f"Combined dataframe contains {len(combined_df)} relationships")
    
    return combined_df


def initialize_openai_client():
    """
    Initialize the OpenAI client using the API keys.
    
    Returns:
        llm, embedding model
    """
    import json
    
    config_path = os.path.join(DATA_DIR, 'keys', 'azure_config.json')
    
    if not os.path.exists(config_path):
        print(f"API configuration file not found at {config_path}")
        print("Please obtain API keys and create the configuration file as described in the README.md")
        return None, None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Dimensions for embedding
    dim = 3072
    
    # Initialize LLM with gpt-4.1-mini
    model_name = 'gpt-4.1-mini'
    if model_name in config:
        llm = AzureChatOpenAI(
            api_key=config[model_name]['api_key'],
            azure_endpoint=config[model_name]['api_base'].split('/openai')[0],
            azure_deployment=config[model_name]['deployment'],
            api_version=config[model_name]['api_version'],
            temperature=0
        )
        
        # Initialize embedding model
        embedding_model = AzureOpenAIEmbeddings(
            api_key=config[model_name]['api_key'],
            azure_endpoint=config[model_name]['api_base'].split('/openai')[0],
            azure_deployment=config[model_name]['emb_deployment'],
            api_version=config[model_name]['api_version'],
            dimensions=dim
        )

        vector_store_name = 'azure-ai-search'

        vector_store = AzureSearch(
            azure_search_endpoint = config[vector_store_name]['azure_endpoint'],
            azure_search_key = config[vector_store_name]['api_key'],
            index_name = config[vector_store_name]['index_name'],
            embedding_function= embedding_model
        )
        
        return llm, embedding_model, vector_store
    else:
        print(f"Model {model_name} configuration not found")
        return None, None


def get_embedding(text: str, model) -> np.ndarray:
    """
    Get embedding for a text using OpenAI model. Falls back to TF-IDF if model is unavailable.
    
    Args:
        text: Text to embed
        model: OpenAI embedding model
    
    Returns:
        np.ndarray: Embedding vector
    """
    # Try to use OpenAI embedding model first
    if model is not None:
        try:
            embedding = model.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error using OpenAI embedding: {e}")
            print("Falling back to TF-IDF embedding...")
    
    # Fallback to TF-IDF embedding
    try:
        # Create embedding using TF-IDF method
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Using TF-IDF for more meaningful text representation
        vectorizer = TfidfVectorizer(max_features=768)
        
        # Split text into sentences for document collection
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        if len(sentences) < 2:
            sentences = text.split()  # If no sentences, use words
        
        # Ensure sufficient text for vectorization
        if len(sentences) < 2:
            sentences = [text, "placeholder"]
            
        # Vectorize
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Use mean as final representation
        embedding = tfidf_matrix.mean(axis=0).A[0]
        
        # Pad to required dimension (1536)
        if len(embedding) < 1536:
            embedding = np.pad(embedding, (0, 1536-len(embedding)), 'constant')
        
        # Truncate if dimension exceeds 1536
        if len(embedding) > 1536:
            embedding = embedding[:1536]
            
        return embedding
    except Exception as e:
        print(f"Error creating TF-IDF embedding: {e}")
        # Return random embedding as last resort
        return np.random.rand(1536)


def compute_similarity(emb1, emb2) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        float: Cosine similarity score
    """
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


def resolve_innovation_duplicates(df_relationships: pd.DataFrame, model=None, vector_store = None,
                              cache_config: Dict = None) -> Dict[str, str]:
    """
    Identify and cluster duplicate innovations using semantic similarity from textual embeddings.

    Args:
        df_relationships (pd.DataFrame): A relationship dataset containing Innovation nodes and their connections.
        model (callable, optional): Embedding model function that converts text to vector. Required if embeddings need to be generated.
        cache_config: 缓存配置，包含以下选项:
            - type: 缓存类型 ('embedding')
            - backend: 后端类型 ('json' 或 'memory')
            - path: 缓存文件路径
            - use_cache: 是否启用缓存

    Returns:
        Dict[str, str]: A dictionary mapping each innovation ID to its canonical ID (representative of a cluster).
    """
    print("Resolving innovation duplicates...")
    
    # 默认缓存配置
    default_cache_config = {
        "type": "embedding",
        "backend": "json", 
        "path": "./embedding_vectors.json",
        "use_cache": True
    }
    
    # 合并配置
    if cache_config is None:
        cache_config = {}
    
    config = {**default_cache_config, **cache_config}

    # Step 1: Extract all unique Innovation nodes from the relationship table
    innovations = df_relationships[df_relationships['source_type'] == 'Innovation']
    unique_innovations = innovations.drop_duplicates(subset=['source_id'])
    print(f"Found {len(unique_innovations)} unique innovations")

    # Step 2: Construct a detailed textual context for each innovation
    # This includes: name, description, developers, and relationship-based context
    innovation_features = {}
    for _, row in tqdm(unique_innovations.iterrows(), total=len(unique_innovations), desc="Creating innovation features"):
        innovation_id = row['source_id']
        if innovation_id not in innovation_features:
            source_name = str(row.get('source_english_id', ''))
            source_description = str(row.get('source_description', ''))
            context = f"Innovation name: {source_name}. Description: {source_description}."

            # Append developers (organizations linked by DEVELOPED_BY relationship)
            developed_by = df_relationships[
                (df_relationships['source_id'] == innovation_id) &
                (df_relationships['relationship_type'] == 'DEVELOPED_BY')
            ]['target_english_id'].dropna().unique().tolist()

            if developed_by:
                context += f" Developed by: {', '.join(developed_by)}."

            # Add additional relationships and their target descriptions
            related_rows = df_relationships[df_relationships['source_id'] == innovation_id]
            for _, rel_row in related_rows.iterrows():
                rel_desc = str(rel_row.get('relationship description', '')).strip()
                target_name = str(rel_row.get('target_english_id', '')).strip()
                target_desc = str(rel_row.get('target_description', '')).strip()
                if rel_desc and target_name and target_desc:
                    context += f" {rel_desc} {target_name}, which is described as: {target_desc}."

            innovation_features[innovation_id] = context

    # Step 3: Generate embeddings or use text features directly
    print("Generating features for similarity comparison...")
    
    # 初始化缓存系统
    cache = CacheFactory.create_cache(
        cache_type=config["type"],
        backend_type=config["backend"],
        cache_path=config["path"],
        use_cache=config["use_cache"]
    )
    
    # 加载缓存
    embeddings = cache.load()
    
    # 找出未缓存的ID
    missing_ids = cache.get_missing_keys(list(innovation_features.keys()))
    
    # 生成新的embeddings
    if missing_ids:
        print(f"Generating {len(missing_ids)} new embeddings...")
        new_embeddings = {}
        
        for id in tqdm(missing_ids, desc="Generating embeddings"):
            text = innovation_features[id]
            new_embeddings[id] = get_embedding(text, model)
        
        # 更新缓存
        cache.update(new_embeddings)
        embeddings.update(new_embeddings)

    # Step 4: Compute cosine similarity between all embedding vectors
    # Group similar innovations into clusters based on similarity threshold
    print("Clustering similar innovations...")
    threshold = 0.85
    embedding_items = list(embeddings.items())
    innovation_ids = [item[0] for item in embedding_items]
    embedding_matrix = np.array([item[1] for item in embedding_items])
    similarity_matrix = cosine_similarity(embedding_matrix)

    clusters = {}
    processed = set()

    for i, id1 in enumerate(tqdm(innovation_ids, desc="Clustering innovations")):
        if id1 in processed:
            continue

        cluster = [id1]
        for j, id2 in enumerate(innovation_ids):
            if id1 != id2 and id2 not in processed:
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    cluster.append(id2)
                    processed.add(id2)

        canonical_id = id1  # use the first item as the canonical (representative) ID
        clusters[canonical_id] = cluster
        processed.add(id1)

    # Step 5: Build a mapping from each innovation ID to its canonical representative
    canonical_mapping = {}
    for canonical_id, cluster_ids in clusters.items():
        for innovation_id in cluster_ids:
            canonical_mapping[innovation_id] = canonical_id

    print(f"Found {len(clusters)} unique innovation clusters")
    print(f"Reduced from {len(innovation_features)} to {len(clusters)} innovations")

    # Step 6: Upload canonical embeddings to Azure AI Search
    if vector_store is not None:
        print("Uploading embeddings to Azure AI Search...")

        text_embeddings = [(id, embeddings[id]) for id in clusters.keys() if id in embeddings]

        # 1000 时有 error_map 的bug
        batch_size = 500
        total_batches = math.ceil(len(text_embeddings) / batch_size)

        for i in range(total_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size

            batch_text_embeddings = text_embeddings[start_index:end_index]

            try:
                vector_store.add_embeddings(
                    text_embeddings=batch_text_embeddings
                )
                print(f"Successfully uploaded batch {i + 1}/{total_batches}")
            # Batch 失败就一个一个试
            except Exception as e:
                try:
                    print(f"Error batch")
                    for embd in batch_text_embeddings:
                        vector_store.add_embeddings(
                            text_embeddings=[embd]
                        )
                except Exception as e:
                    print(f"Error uploading embedding: {e}")

        print("Uploaded embeddings to Azure AI Search...")


    return canonical_mapping, innovation_features


def create_innovation_knowledge_graph(df_relationships: pd.DataFrame, canonical_mapping: Dict[str, str]) -> Dict:
    """
    Create a consolidated knowledge graph of innovations and their relationships.
    
    Args:
        df_relationships: DataFrame with innovation relationships
        canonical_mapping: Mapping from innovation IDs to canonical IDs
    
    Returns:
        Dict: Consolidated knowledge graph
    """
    print("Creating innovation knowledge graph...")
    
    # Step 1: Create consolidated innovations
    consolidated_innovations = {}
    
    for _, row in tqdm(df_relationships[df_relationships['source_type'] == 'Innovation'].iterrows(), 
                      desc="Consolidating innovations"):
        innovation_id = row['source_id']
        canonical_id = canonical_mapping.get(innovation_id, innovation_id)
        
        if canonical_id not in consolidated_innovations:
            consolidated_innovations[canonical_id] = {
                'id': canonical_id,
                'names': set(),
                'descriptions': set(),
                'developed_by': set(),
                'sources': set(),
                'source_ids': set([innovation_id]),
                'data_sources': set()
            }
        else:
            consolidated_innovations[canonical_id]['source_ids'].add(innovation_id)
        
        consolidated_innovations[canonical_id]['names'].add(str(row['source_english_id']))
        consolidated_innovations[canonical_id]['descriptions'].add(str(row['source_description']))
        consolidated_innovations[canonical_id]['sources'].add(str(row['Link Source Text']))
        consolidated_innovations[canonical_id]['data_sources'].add(str(row['data_source']))
        
        # Add relationship
        if row['relationship_type'] == 'DEVELOPED_BY':
            consolidated_innovations[canonical_id]['developed_by'].add(row['target_id'])
    
    # Step 2: Build consolidated graph
    consolidated_graph = {
        'innovations': consolidated_innovations,
        'organizations': {},
        'relationships': []
    }
    
    # Add organizations
    for _, row in tqdm(df_relationships[df_relationships['target_type'] == 'Organization'].drop_duplicates(subset=['target_id']).iterrows(),
                      desc="Adding organizations"):
        org_id = row['target_id']
        if org_id not in consolidated_graph['organizations']:
            consolidated_graph['organizations'][org_id] = {
                'id': org_id,
                'name': row['target_english_id'],
                'description': row['target_description']
            }
    
    # Add relationships
    for canonical_id, innovation in tqdm(consolidated_innovations.items(), desc="Adding relationships"):
        for org_id in innovation['developed_by']:
            consolidated_graph['relationships'].append({
                'source': canonical_id,
                'target': org_id,
                'type': 'DEVELOPED_BY'
            })
    
    # Add collaboration relationships
    for _, row in tqdm(df_relationships[
        (df_relationships['source_type'] == 'Organization') & 
        (df_relationships['relationship_type'] == 'COLLABORATION')
    ].iterrows(), desc="Adding collaborations"):
        consolidated_graph['relationships'].append({
            'source': row['source_id'],
            'target': row['target_id'],
            'type': 'COLLABORATION'
        })
    
    print(f"Created knowledge graph with {len(consolidated_graph['innovations'])} innovations, " 
          f"{len(consolidated_graph['organizations'])} organizations, and "
          f"{len(consolidated_graph['relationships'])} relationships")
    
    return consolidated_graph


def analyze_innovation_network(consolidated_graph: Dict) -> Dict:
    """
    Analyze the innovation network to extract insights.
    
    Args:
        consolidated_graph: Consolidated knowledge graph
    
    Returns:
        Dict: Analysis results
    """
    print("Analyzing innovation network...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for innovation_id, innovation in consolidated_graph['innovations'].items():
        G.add_node(innovation_id, 
                   type='Innovation', 
                   names=', '.join(innovation['names']),
                   sources=len(innovation['sources']),
                   developed_by=len(innovation['developed_by']))
    
    for org_id, org in consolidated_graph['organizations'].items():
        G.add_node(org_id, 
                   type='Organization', 
                   name=org['name'])
    
    # Add edges
    for rel in consolidated_graph['relationships']:
        G.add_edge(rel['source'], rel['target'], type=rel['type'])
    
    # Basic statistics
    innovation_stats = {
        'total': len(consolidated_graph['innovations']),
        'avg_sources': sum(len(i['sources']) for i in consolidated_graph['innovations'].values()) / max(1, len(consolidated_graph['innovations'])),
        'avg_developers': sum(len(i['developed_by']) for i in consolidated_graph['innovations'].values()) / max(1, len(consolidated_graph['innovations'])),
        'multi_source_count': sum(1 for i in consolidated_graph['innovations'].values() if len(i['sources']) > 1),
        'multi_developer_count': sum(1 for i in consolidated_graph['innovations'].values() if len(i['developed_by']) > 1)
    }
    
    # Find innovations with multiple sources
    multi_source_innovations = {
        k: v for k, v in consolidated_graph['innovations'].items() 
        if len(v['sources']) > 1
    }
    
    # Find organizations with most innovations
    org_innovation_counts = {}
    for rel in consolidated_graph['relationships']:
        if rel['type'] == 'DEVELOPED_BY' and rel['target'] in consolidated_graph['organizations']:
            org_id = rel['target']
            if org_id not in org_innovation_counts:
                org_innovation_counts[org_id] = 0
            org_innovation_counts[org_id] += 1
    
    top_orgs = sorted(
        [(org_id, count) for org_id, count in org_innovation_counts.items()], 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Centrality analysis
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Find key players
        key_orgs = sorted(
            [(node, betweenness_centrality[node]) 
             for node in G.nodes if G.nodes[node].get('type') == 'Organization'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        key_innovations = sorted(
            [(node, eigenvector_centrality[node]) 
             for node in G.nodes if G.nodes[node].get('type') == 'Innovation'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    except:
        # If centrality algorithms fail, provide empty lists
        key_orgs = []
        key_innovations = []
    
    return {
        'graph': G,
        'stats': innovation_stats,
        'multi_source': multi_source_innovations,
        'top_orgs': top_orgs,
        'key_orgs': key_orgs,
        'key_innovations': key_innovations
    }


def visualize_network(analysis_results: Dict, output_dir: str = RESULTS_DIR):
    """
    Visualize the innovation network.
    
    Args:
        analysis_results: Results from network analysis
        output_dir: Directory to save visualizations
    """
    print("Visualizing network...")
    
    G = analysis_results['graph']
    
    # 确保所有节点都有type属性
    for node in G.nodes:
        if 'type' not in G.nodes[node]:
            # 检查是否有其他可以用来推断类型的属性
            if 'names' in G.nodes[node]:
                G.nodes[node]['type'] = 'Innovation'
            elif 'name' in G.nodes[node]:
                G.nodes[node]['type'] = 'Organization'
            else:
                # 如果无法确定，设置为默认类型
                G.nodes[node]['type'] = 'Unknown'
    
    # 颜色映射，增加未知类型的颜色
    color_map = {'Innovation': 'lightblue', 'Organization': 'lightgreen', 'Unknown': 'lightgray'}
    node_colors = [color_map[G.nodes[n].get('type', 'Unknown')] for n in G.nodes]
    
    # 节点大小
    innovation_sizes = [
        G.nodes[n].get('sources', 1) * 50 if G.nodes[n].get('type') == 'Innovation' 
        else 100 for n in G.nodes
    ]
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制网络
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=innovation_sizes,
                          alpha=0.8)
    
    # 绘制边，不同类型的边用不同颜色
    edge_colors = ['red' if G.edges[e].get('type') == 'DEVELOPED_BY' else 'blue' for e in G.edges]
    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors,
                          width=1.0,
                          alpha=0.5)
    
    # 仅为关键节点绘制标签
    key_nodes = [node for node, _ in analysis_results['key_orgs'] + analysis_results['key_innovations']]
    labels = {node: G.nodes[node].get('name', G.nodes[node].get('names', node)) 
              for node in G.nodes if node in key_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # 添加图例
    innovation_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Innovation')
    org_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Organization')
    unknown_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Unknown')
    developed_by = plt.Line2D([0], [0], color='red', lw=2, label='Developed By')
    collaboration = plt.Line2D([0], [0], color='blue', lw=2, label='Collaboration')
    
    plt.legend(handles=[innovation_patch, org_patch, unknown_patch, developed_by, collaboration], loc='best')
    
    plt.title('VTT Innovation Network')
    plt.axis('off')
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'innovation_network.png'), dpi=300)
    plt.close()
    
    # Create 3D visualization with Plotly
    visualize_network_3d(G, analysis_results, output_dir)
    
    # 创建汇总统计可视化
    plt.figure(figsize=(10, 6))
    stats = analysis_results['stats']
    
    # 关键统计数据条形图
    sns.barplot(x=['Total Innovations', 'Multi-Source Innovations', 'Multi-Developer Innovations'],
               y=[stats['total'], stats['multi_source_count'], stats['multi_developer_count']])
    
    plt.title('Innovation Statistics')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'innovation_stats.png'), dpi=300)
    plt.close()
    
    # 可视化按创新计数排名前列的组织
    plt.figure(figsize=(12, 8))
    top_orgs = analysis_results['top_orgs']
    
    if top_orgs:
        # 确保只使用存在于图中的组织
        filtered_top_orgs = [(org_id, count) for org_id, count in top_orgs if org_id in G.nodes]
        
        if filtered_top_orgs:
            org_names = [G.nodes[org_id].get('name', org_id) for org_id, _ in filtered_top_orgs]
            org_counts = [count for _, count in filtered_top_orgs]
            
            # 创建水平条形图
            sns.barplot(y=org_names, x=org_counts, palette='viridis')
            
            plt.title('Top Organizations by Innovation Count')
            plt.xlabel('Number of Innovations')
            plt.ylabel('Organization')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_organizations.png'), dpi=300)
            plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def visualize_network_3d(G: nx.Graph, analysis_results: Dict, output_dir: str = RESULTS_DIR):
    """
    Create a 3D visualization of the innovation network using Plotly.
    
    Args:
        G: NetworkX graph of the innovation network
        analysis_results: Results from network analysis
        output_dir: Directory to save visualizations
    """
    print("Creating 3D network visualization...")
    
    # Use a force-directed layout algorithm in 3D
    pos_3d = nx.spring_layout(G, dim=3, k=0.15, iterations=50)
    
    # Extract node positions
    x_nodes = [pos_3d[node][0] for node in G.nodes]
    y_nodes = [pos_3d[node][1] for node in G.nodes]
    z_nodes = [pos_3d[node][2] for node in G.nodes]
    
    # Node types for coloring
    node_types = [G.nodes[node].get('type', 'Unknown') for node in G.nodes]
    
    # Node sizes
    node_sizes = []
    for node in G.nodes:
        if G.nodes[node].get('type') == 'Innovation':
            # Larger nodes for innovations with more sources
            size = G.nodes[node].get('sources', 1) * 10
        else:
            # Default size for organizations
            size = 10
        node_sizes.append(size)
    
    # Node labels
    key_nodes = [node for node, _ in analysis_results['key_orgs'] + analysis_results['key_innovations']]
    node_labels = []
    for node in G.nodes:
        if node in key_nodes:
            if G.nodes[node].get('type') == 'Innovation':
                label = G.nodes[node].get('names', node)
            else:
                label = G.nodes[node].get('name', node)
        else:
            label = ""
        node_labels.append(label)
    
    # Create a color scale for node types
    color_map = {'Innovation': 'rgb(100, 149, 237)', 'Organization': 'rgb(144, 238, 144)', 'Unknown': 'rgb(211, 211, 211)'}
    node_colors = [color_map[G.nodes[n].get('type', 'Unknown')] for n in G.nodes]
    
    # Create nodes trace
    nodes_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers+text',
        text=node_labels,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(width=0.5, color='rgb(50, 50, 50)')
        ),
        hoverinfo='text',
        hovertext=[f"{G.nodes[node].get('name', G.nodes[node].get('names', node))}<br>Type: {G.nodes[node].get('type', 'Unknown')}" 
                  for node in G.nodes]
    )
    
    # Create edges traces
    edge_traces = []
    
    # Group edges by type
    edge_types = {'DEVELOPED_BY': 'red', 'COLLABORATION': 'blue', 'OTHER': 'gray'}
    
    for edge_type, color in edge_types.items():
        x_edges = []
        y_edges = []
        z_edges = []
        
        for edge in G.edges:
            # Filter edges by type
            edge_data = G.edges[edge]
            current_edge_type = edge_data.get('type', 'OTHER')
            
            if edge_type == 'OTHER' and current_edge_type not in edge_types:
                # This is for edges that don't have a specific type defined in edge_types
                source, target = edge
                x_edges.extend([pos_3d[source][0], pos_3d[target][0], None])
                y_edges.extend([pos_3d[source][1], pos_3d[target][1], None])
                z_edges.extend([pos_3d[source][2], pos_3d[target][2], None])
            elif current_edge_type == edge_type:
                source, target = edge
                x_edges.extend([pos_3d[source][0], pos_3d[target][0], None])
                y_edges.extend([pos_3d[source][1], pos_3d[target][1], None])
                z_edges.extend([pos_3d[source][2], pos_3d[target][2], None])
        
        if x_edges:  # Only add a trace if there are edges of this type
            edge_trace = go.Scatter3d(
                x=x_edges,
                y=y_edges,
                z=z_edges,
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='none',
                name=edge_type
            )
            edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[nodes_trace] + edge_traces)
    
    # Update layout
    fig.update_layout(
        title='VTT Innovation Network - 3D Visualization',
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            title_text='Relationship Types',
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        showlegend=True
    )
    
    # Save interactive HTML file
    fig.write_html(os.path.join(output_dir, 'innovation_network_3d.html'))
    
    # Save static image as well
    fig.write_image(os.path.join(output_dir, 'innovation_network_3d.png'), width=1200, height=900)
    
    print(f"3D visualization saved to {output_dir}")


def export_results(analysis_results: Dict, consolidated_graph: Dict, canonical_mapping: Dict, output_dir: str = RESULTS_DIR):
    """
    Export analysis results and consolidated data to files.
    
    Args:
        analysis_results: Results from network analysis
        consolidated_graph: Consolidated knowledge graph
        canonical_mapping: Mapping from innovation IDs to canonical IDs
        output_dir: Directory to save results
    """
    print("Exporting results...")
    
    # Save canonical mapping
    with open(os.path.join(output_dir, 'canonical_mapping.json'), 'w') as f:
        # Convert sets to lists for JSON serialization
        mapping_for_json = {k: v for k, v in canonical_mapping.items()}
        json.dump(mapping_for_json, f, indent=2)
    
    # Save consolidated graph (need to convert sets to lists for JSON serialization)
    graph_for_json = {
        'innovations': {},
        'organizations': consolidated_graph['organizations'],
        'relationships': consolidated_graph['relationships']
    }
    
    for k, v in consolidated_graph['innovations'].items():
        graph_for_json['innovations'][k] = {
            'id': v['id'],
            'names': list(v['names']),
            'descriptions': list(v['descriptions']),
            'developed_by': list(v['developed_by']),
            'sources': list(v['sources']),
            'source_ids': list(v['source_ids']),
            'data_sources': list(v['data_sources'])
        }
    
    with open(os.path.join(output_dir, 'consolidated_graph.json'), 'w') as f:
        json.dump(graph_for_json, f, indent=2)
    
    # Save innovation statistics
    with open(os.path.join(output_dir, 'innovation_stats.json'), 'w') as f:
        stats_for_json = analysis_results['stats']
        json.dump(stats_for_json, f, indent=2)
    
    # Save multi-source innovations information
    multi_source_for_json = {}
    for k, v in analysis_results['multi_source'].items():
        multi_source_for_json[k] = {
            'names': list(v['names']),
            'descriptions': list(v['descriptions']),
            'developed_by': list(v['developed_by']),
            'sources': list(v['sources']),
            'source_ids': list(v['source_ids']),
            'data_sources': list(v['data_sources'])
        }
    
    with open(os.path.join(output_dir, 'multi_source_innovations.json'), 'w') as f:
        json.dump(multi_source_for_json, f, indent=2)
    
    # Save key organizations and innovations
    key_nodes = {
        'key_organizations': [{
            'id': org_id,
            'centrality': centrality,
            'name': analysis_results['graph'].nodes[org_id].get('name', org_id)
        } for org_id, centrality in analysis_results['key_orgs']],
        'key_innovations': [{
            'id': inno_id,
            'centrality': centrality,
            'names': list(consolidated_graph['innovations'][inno_id]['names'])
        } for inno_id, centrality in analysis_results['key_innovations'] if inno_id in consolidated_graph['innovations']]
    }
    
    with open(os.path.join(output_dir, 'key_nodes.json'), 'w') as f:
        json.dump(key_nodes, f, indent=2)
    
    print(f"Results exported to {output_dir}")


def main():
    """Main function to execute the innovation resolution workflow."""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="VTT Innovation Resolution process")
    parser.add_argument("--cache-type", default="embedding", choices=["embedding"],
                       help="Cache type to use")
    parser.add_argument("--cache-backend", default="json", choices=["json", "memory"],
                       help="Cache backend type")
    parser.add_argument("--cache-path", default="./embedding_vectors.json",
                       help="Path to cache file (for file-based backends)")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable caching")
    
    args = parser.parse_args()
    
    # 缓存配置
    cache_config = {
        "type": args.cache_type,
        "backend": args.cache_backend,
        "path": args.cache_path,
        "use_cache": not args.no_cache
    }
    
    print("Starting VTT Innovation Resolution process...")
    print(f"Cache configuration: {cache_config}")
    
    # Step 1: Load and combine data
    df_relationships = load_and_combine_data()
    
    # Step 2: Initialize OpenAI client
    llm, embed_model, vector_store = initialize_openai_client()
    
    if llm is None:
        print("Warning: Language model not available. Some features may be limited.")
    
    if embed_model is None:
        print("Warning: Embedding model not available. Using TF-IDF embeddings as fallback.")
    
    # Step 3: Resolve innovation duplicates
    canonical_mapping, innovation_features = resolve_innovation_duplicates(
        df_relationships, 
        embed_model,
        vector_store,
        cache_config=cache_config
    )

    chat_loop(llm, embed_model, vector_store, innovation_features)
    
    # Step 4: Create consolidated knowledge graph
    consolidated_graph = create_innovation_knowledge_graph(df_relationships, canonical_mapping)
    
    # Step 5: Analyze innovation network
    analysis_results = analyze_innovation_network(consolidated_graph)
    
    # Step 6: Visualize network
    visualize_network(analysis_results)
    visualize_network_3d(analysis_results['graph'], analysis_results)
    
    # Step 7: Export results
    export_results(analysis_results, consolidated_graph, canonical_mapping)
    
    print("Innovation Resolution process completed successfully!")
    print(f"Results and visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main() 