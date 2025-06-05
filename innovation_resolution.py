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
from vis import visualize_network_tufte

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate

from azure_openai import get_openai_models, \
                            CONFIG_PATH, DEFAULT_DIMS, DEFAULT_INDEX_PATH

from qdrant import VectorStore

import math


# Import local modules
from innovation_utils import (
    compute_similarity_matrix,
    find_potential_duplicates,
    calculate_innovation_statistics
)
from local_entity_processing import Node, Relationship

from utils.cluster.cluster_algorithms import (
    cluster_hdbscan,
    cluster_kmeans,
    cluster_agglomerative,
    cluster_spectral
)
from utils.cluster.graph_clustering import (
    graph_threshold_clustering,
    graph_kcore_clustering
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


# 添加通用的文本过滤器
def is_valid_entity_name(name: str) -> bool:
    """
    检查实体名称是否有效，过滤掉明显无效的实体。
    
    Args:
        name: 实体名称
        
    Returns:
        bool: 是否是有效的实体名称
    """
    if not name or not isinstance(name, str):
        return False
    
    # 过滤掉太短的名称
    if len(name.strip()) < 3:
        return False
    
    # 过滤掉只包含数字或特殊字符的名称
    if all(not c.isalpha() for c in name):
        return False
    
    # 过滤掉常见的占位符名称
    invalid_patterns = [
        'null', 'none', 'undefined', 'n/a', 'unknown', 
        'temp_', 'unknown', 'placeholder', 'example'
    ]
    
    name_lower = name.lower()
    for pattern in invalid_patterns:
        if pattern in name_lower:
            # 特殊处理：如果是以temp_开头但后面有有意义的内容，仍然保留
            if pattern == 'temp_' and len(name) > 10 and any(c.isalpha() for c in name[5:]):
                continue
            return False
    
    return True


def extract_entities_from_document(doc, pred_entities: List[Dict] = None) -> List[Dict]:
    """
    Extract innovation and organization entities from document.
    Also accumulate extracted entities in the pred_entities list if provided.
    
    Args:
        doc: Source document
        pred_entities: Optional list to accumulate extracted entities
        
    Returns:
        List of entity dictionaries in the format {"name": str, "type": str}
    """
    entities = []
    
    # Extract entities from the document (assuming graph_doc format)
    if hasattr(doc, 'nodes'):
        for node in doc.nodes:
            # 获取实体名称，优先使用english_id
            entity_name = node.properties.get('english_id', node.id) if hasattr(node, 'properties') else node.id
            
            # 检查实体名称是否有效
            if not is_valid_entity_name(entity_name):
                continue
                
            entity = {
                "name": entity_name,
                "type": node.type
            }
            
            # 添加描述信息，便于后续过滤和评估
            if hasattr(node, 'properties') and 'description' in node.properties:
                entity["description"] = node.properties['description']
            
            entities.append(entity)
            
            # Accumulate to prediction list if provided
            if pred_entities is not None:
                pred_entities.append(entity)
    
    return entities


def is_valid_relationship(innovation: str, organization: str, relation_type: str) -> bool:
    """
    检查关系是否有效，过滤掉明显无效的关系。
    
    Args:
        innovation: 创新名称
        organization: 组织名称
        relation_type: 关系类型
        
    Returns:
        bool: 是否是有效的关系
    """
    # 检查创新和组织名称是否有效
    if not is_valid_entity_name(innovation) or not is_valid_entity_name(organization):
        return False
    
    # 检查关系类型是否有效
    if relation_type not in ["DEVELOPED_BY", "COLLABORATION"]:
        return False
    
    # 过滤掉创新和组织相同的情况
    if innovation.lower() == organization.lower():
        return False
    
    return True


def extract_relationships_from_document(doc, pred_relations: List[Dict] = None) -> List[Dict]:
    """
    Extract relationships from document.
    Also accumulate extracted relationships in the pred_relations list if provided.
    
    Args:
        doc: Source document
        pred_relations: Optional list to accumulate extracted relationships
        
    Returns:
        List of relationship dictionaries in the format {"innovation": str, "organization": str, "relation": str}
    """
    relationships = []
    
    # Extract relationships from the document (assuming graph_doc format)
    if hasattr(doc, 'relationships'):
        # 首先获取节点的english_id映射
        node_english_id = {}
        if hasattr(doc, 'nodes'):
            for node in doc.nodes:
                if hasattr(node, 'properties') and 'english_id' in node.properties:
                    node_english_id[node.id] = node.properties['english_id']
                else:
                    node_english_id[node.id] = node.id
        
        for rel in doc.relationships:
            # 只包含DEVELOPED_BY和COLLABORATION关系
            if rel.type in ["DEVELOPED_BY", "COLLABORATION"]:
                # 获取源和目标的名称，优先使用english_id
                source_name = node_english_id.get(rel.source, rel.source)
                target_name = node_english_id.get(rel.target, rel.target)
                
                # 确保source/target是正确的创新/组织映射
                if rel.source_type == "Innovation" and rel.target_type == "Organization":
                    innovation_name = source_name
                    organization_name = target_name
                elif rel.source_type == "Organization" and rel.target_type == "Innovation":
                    innovation_name = target_name
                    organization_name = source_name
                else:
                    # 如果关系不是创新-组织之间的关系，跳过
                    continue
                
                # 检查关系是否有效
                if not is_valid_relationship(innovation_name, organization_name, rel.type):
                    continue
                
                relationship = {
                    "innovation": innovation_name,
                    "organization": organization_name,
                    "relation": rel.type
                }
                
                # 添加描述信息，便于后续过滤和评估
                if hasattr(rel, 'properties') and 'description' in rel.properties:
                    relationship["description"] = rel.properties['description']
                
                relationships.append(relationship)
                
                # Accumulate to prediction list if provided
                if pred_relations is not None:
                    pred_relations.append(relationship)
    
    return relationships


def load_and_combine_data() -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Load relationship data from both company websites and VTT domain,
    combine them into a single dataframe, and collect entities and relationships.
    
    Returns:
        Tuple of (combined dataframe, all_pred_entities, all_pred_relations)
    """
    print("Loading data from company websites...")
    
    # Initialize lists to collect predicted entities and relationships
    all_pred_entities = []
    all_pred_relations = []
    
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
                        graph_docs = pickle.load(f)
                        graph_doc = graph_docs[0]
                    
                    # Extract and collect entities
                    extract_entities_from_document(graph_doc, all_pred_entities)
                    
                    # Extract and collect relationships
                    extract_relationships_from_document(graph_doc, all_pred_relations)
                    
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
                        graph_docs = pickle.load(f)
                        graph_doc = graph_docs[0]
                    
                    # Extract and collect entities
                    extract_entities_from_document(graph_doc, all_pred_entities)
                    
                    # Extract and collect relationships
                    extract_relationships_from_document(graph_doc, all_pred_relations)
                    
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
    
    return combined_df, all_pred_entities, all_pred_relations


def initialize_openai_client():
    """初始化OpenAI客户端"""
    import json
    
    # 打印调试信息
    print("="*50)
    print("初始化OpenAI客户端...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"DATA_DIR: {DATA_DIR}")
    
    try:
        # 尝试从streamlit secrets获取配置
        import streamlit as st
        
        print("尝试从st.secrets获取配置...")
        if hasattr(st, 'secrets') and st.secrets:
            print(f"st.secrets可用，包含以下键: {list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else 'No keys method'}")
            if 'default_model' in st.secrets:
                print(f"默认模型信息: {st.secrets['default_model']}")
            if 'gpt-4.1-mini' in st.secrets:
                print(f"模型配置包含键: {list(st.secrets['gpt-4.1-mini'].keys()) if hasattr(st.secrets['gpt-4.1-mini'], 'keys') else 'No keys method'}")
            if 'azure-ai-search' in st.secrets:
                print(f"向量存储配置包含键: {list(st.secrets['azure-ai-search'].keys()) if hasattr(st.secrets['azure-ai-search'], 'keys') else 'No keys method'}")
        else:
            print("st.secrets不可用")
        
        config = st.secrets
        
        # 如果streamlit secrets不可用，尝试从文件读取
        if not config:
            print("st.secrets为空，尝试从文件读取...")
            # ✅ 优先从环境变量读取路径
            config_path = os.environ.get("AZURE_CONFIG", os.path.join(DATA_DIR, 'keys', 'azure_config.json'))
            print(f"尝试读取配置文件: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"API配置文件未找到: {config_path}")
                print("请获取API密钥并按照README.md中的说明创建配置文件")
                return None, None, None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"从文件加载配置成功，包含键: {list(config.keys())}")
        
        dim = 3072
        # 如果使用streamlit secrets，需要适配其结构
        if hasattr(config, 'get'):
            # 从st.secrets中获取
            model_name = config.get('default_model', {}).get('name', 'gpt-4.1-mini')
            model_config = config.get(model_name, {})
            vector_store_name = 'azure-ai-search'
            vector_store_config = config.get(vector_store_name, {})
        else:
            # 从JSON配置中获取
            model_name = 'gpt-4.1-mini'
            model_config = config.get(model_name, {})
            vector_store_name = 'azure-ai-search'
            vector_store_config = config.get(vector_store_name, {})
        
        if model_config:
            api_key = model_config.get('api_key')
            api_base = model_config.get('api_base', '')
            # 移除URL部分（如果存在）
            base_endpoint = api_base.split('/openai')[0] if '/openai' in api_base else api_base
            
            llm = AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=base_endpoint,
                azure_deployment=model_config.get('deployment'),
                api_version=model_config.get('api_version'),
                temperature=0
            )
            
            embedding_model = AzureOpenAIEmbeddings(
                api_key=api_key,
                azure_endpoint=base_endpoint,
                azure_deployment=model_config.get('emb_deployment'),
                api_version=model_config.get('api_version'),
                dimensions=dim
            )
            
            if vector_store_config:
                vector_store = AzureSearch(
                    azure_search_endpoint = vector_store_config.get('azure_endpoint'),
                    azure_search_key = vector_store_config.get('api_key'),
                    index_name = vector_store_config.get('index_name'),
                    embedding_function = embedding_model
                )
                
                return llm, embedding_model, vector_store
            else:
                print(f"Vector store {vector_store_name} configuration not found")
                return llm, embedding_model, None
        else:
            print(f"Model {model_name} configuration not found")
            return None, None, None
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


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


def resolve_innovation_duplicates(
    df_relationships: pd.DataFrame, 
    model=None, 
    vector_store = None,
    using_local_store = True,
    cache_config: Dict = None,
    method: str = "hdbscan",
    **method_kwargs) -> Dict[str, str]:
    """
    Identify and cluster duplicate innovations using semantic similarity from textual embeddings.
    
    Args:
        df_relationships (pd.DataFrame): A relationship dataset containing Innovation nodes.
        model (callable, optional): Embedding model function that converts text -> vector.
        vector_store (callable, optional): Azure AI search function, containing text emb. feat.
        cache_config (Dict, optional): 缓存配置，包含以下字段:
            - type: 缓存类型 ('embedding')
            - backend: 后端类型 ('json' or 'memory')
            - path: 缓存文件路径
            - use_cache: 是否启用缓存
        method (str, optional): Which clustering method to use. One of:
            - "hdbscan"
            - "kmeans"
            - "agglomerative"
            - "spectral"
            - "graph_threshold"
            - "graph_kcore"
          Default: "hdbscan".
        **method_kwargs: Additional keyword args to pass into the chosen clustering function.
          For example:
            - if method="hdbscan", you can pass min_cluster_size=2, metric="cosine", cluster_selection_method="eom".
            - if method="kmeans", you can pass n_clusters=450, random_state=42.
            - if method="agglomerative", you can pass n_clusters=450, affinity="cosine", linkage="average".
            - if method="spectral", you can pass n_clusters=450, affinity="nearest_neighbors", n_neighbors=10.
            - if method="graph_threshold", you can pass similarity_threshold=0.85, use_cosine=True.
            - if method="graph_kcore", you can pass similarity_threshold=0.85, k_core=15, use_cosine=True.

    Returns:
        Dict[str, str]: Mapping from each innovation ID -> its canonical cluster ID.
    """
    print("Resolving innovation duplicates...")
    
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

    # Step 1: 从 DataFrame 中筛选出所有 source_type == "Innovation"，并去重
    innovations = df_relationships[df_relationships["source_type"] == "Innovation"]
    unique_innovations = innovations.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    print(f"Found {len(unique_innovations)} unique innovations.")
    if unique_innovations.empty:
        return {}

    # Step 2: Construct a detailed textual context for each innovation
    # This includes: name, description, developers, and relationship-based context
    # Step 2: 为每个 Innovation 构建一个文本上下文（包含名称、描述、开发组织、其他关系说明）
    innovation_features = {}


    for _, row in tqdm(unique_innovations.iterrows(), total=len(unique_innovations), desc="Creating innovation features"):
        innovation_id = row["source_id"]
        if innovation_id in innovation_features:
            continue

        source_name = str(row.get("source_english_id", "")).strip()
        source_desc = str(row.get("source_description", "")).strip()
        context = f"Innovation name: {source_name}. Description: {source_desc}."

        # 如果有 DEVELOPED_BY 关系，则拼接开发组织
        dev_by = (
            df_relationships[
                (df_relationships["source_id"] == innovation_id) &
                (df_relationships["relationship_type"] == "DEVELOPED_BY")
            ]["target_english_id"]
            .dropna()
            .unique()
            .tolist()
        )
        if dev_by:
            context += " Developed by: " + ", ".join(dev_by) + "."

        # 将其他关系（relationship description + target 英文名 + target 描述）拼接进 context
        related_rows = df_relationships[df_relationships["source_id"] == innovation_id]
        for _, rel_row in related_rows.iterrows():
            rel_desc = str(rel_row.get("relationship description", "")).strip()
            target_name = str(rel_row.get("target_english_id", "")).strip()
            target_desc = str(rel_row.get("target_description", "")).strip()
            if rel_desc and target_name and target_desc:
                context += f" {rel_desc} {target_name}, which is described as: {target_desc}."

        innovation_features[innovation_id] = context

    # Step 3: 生成或加载这些上下文的 embeddings
    print("Generating features for similarity comparison...")
    # 假设 CacheFactory.create_cache 可以根据 config 创建出缓存实例，支持 load()/get_missing_keys()/update() 等接口
    cache = CacheFactory.create_cache(
        cache_type=config["type"],
        backend_type=config["backend"],
        cache_path=config["path"],
        use_cache=config["use_cache"]
    )

    # 从缓存里 load 出已经存在的 embedding
    embeddings: Dict[str, np.ndarray] = cache.load()  # { innovation_id: np.array(...) }
    all_ids = list(innovation_features.keys())
    missing_ids = cache.get_missing_keys(all_ids)

    if missing_ids:
        print(f"Generating {len(missing_ids)} new embeddings...")
        new_embd: Dict[str, np.ndarray] = {}
        for iid in tqdm(missing_ids, desc="Embedding innovations"):
            txt = innovation_features[iid]
            new_embd[iid] = get_embedding(txt, model)
        cache.update(new_embd)
        embeddings.update(new_embd)

    # 确保 embeddings 的顺序和 unique_innovations 一致
    embedding_items = [(iid, embeddings[iid]) for iid in all_ids]
    innovation_ids = [item[0] for item in embedding_items]
    embedding_matrix = np.vstack([item[1] for item in embedding_items])  # shape = (N, D)

    # Step 4: 根据用户指定的 method，调用对应的聚类算法
    print(f"Clustering similar innovations with method='{method}'...")
    canonical_mapping: Dict[str, str] = {}

    method_lower = method.lower()
    if method_lower in {"hdbscan", "kmeans", "agglomerative", "spectral"}:
        # -------- 1) 平面簇算法 --------
        if method_lower == "hdbscan":
            min_cluster_size = method_kwargs.get("min_cluster_size", 2)
            metric = method_kwargs.get("metric", "cosine")
            cluster_selection_method = method_kwargs.get("cluster_selection_method", "eom")
            labels = cluster_hdbscan(
                embedding_matrix=embedding_matrix,
                min_cluster_size=min_cluster_size,
                metric=metric,
                cluster_selection_method=cluster_selection_method
            )
        elif method_lower == "kmeans":
            n_clusters = method_kwargs.get("n_clusters", 450)
            random_state = method_kwargs.get("random_state", 42)
            labels = cluster_kmeans(
                embedding_matrix=embedding_matrix,
                n_clusters=n_clusters,
                random_state=random_state
            )
        elif method_lower == "agglomerative":
            n_clusters = method_kwargs.get("n_clusters", 450)
            affinity = method_kwargs.get("affinity", "cosine")
            linkage = method_kwargs.get("linkage", "average")
            labels = cluster_agglomerative(
                embedding_matrix=embedding_matrix,
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage
            )
        else:  # spectral
            n_clusters = method_kwargs.get("n_clusters", 450)
            affinity = method_kwargs.get("affinity", "nearest_neighbors")
            n_neighbors = method_kwargs.get("n_neighbors", 10)
            labels = cluster_spectral(
                embedding_matrix=embedding_matrix,
                n_clusters=n_clusters,
                affinity=affinity,
                n_neighbors=n_neighbors
            )

        # 把 label -> cluster 成员映射出来
        clusters: Dict[int, List[str]] = {}
        for idx, lab in enumerate(labels):
            if lab == -1:
                # HDBSCAN 的 -1 (噪声) 单独成一簇
                key = f"noise_{innovation_ids[idx]}"
                clusters.setdefault(key, []).append(innovation_ids[idx])
            else:
                clusters.setdefault(int(lab), []).append(innovation_ids[idx])

        # 把每个簇里的第一个成员设为 canonical_id
        for lab_key, members in clusters.items():
            canonical_id = members[0]
            for mid in members:
                canonical_mapping[mid] = canonical_id

    elif method_lower in {"graph_threshold", "graph_kcore"}:
        # -------- 2) 图聚类算法 --------
        sim_threshold = method_kwargs.get("similarity_threshold", 0.85)
        use_cos = method_kwargs.get("use_cosine", True)

        if method_lower == "graph_threshold":
            clusters_dict = graph_threshold_clustering(
                embedding_matrix=embedding_matrix,
                ids=innovation_ids,
                similarity_threshold=sim_threshold,
                use_cosine=use_cos
            )
        else:  # "graph_kcore"
            k_core = method_kwargs.get("k_core", 15)
            clusters_dict = graph_kcore_clustering(
                embedding_matrix=embedding_matrix,
                ids=innovation_ids,
                similarity_threshold=sim_threshold,
                k_core=k_core,
                use_cosine=use_cos
            )

        # clusters_dict 已经是 { canonical_id: [member_id,...], ... }
        for canonical_id, members in clusters_dict.items():
            for mid in members:
                canonical_mapping[mid] = canonical_id

    else:
        raise ValueError(
            f"Unknown clustering method '{method}'.\n"
            "请选择：['hdbscan','kmeans','agglomerative','spectral','graph_threshold','graph_kcore']。"
        )

    print(f"Found {len(set(canonical_mapping.values()))} unique innovation clusters "
          f"(reduced from {len(unique_innovations)}).")
          
    # Step 5: (deleted)
    
    # Step 6: Store embeddings to memory and local.

    if using_local_store and vector_store is not None:

        # embedding_matrix = np.vstack([v for k, v in embeddings.items()])

        innovation_texts = [innovation_features[iid] for iid in innovation_ids]
        embeddings_input = [embeddings[iid] for iid in innovation_ids]

        vector_store.load_cache(embeddings_input, innovation_texts)

        print("Saved embeddings to local vectore store...")

    # Step 6: Upload canonical embeddings to Azure AI Search
    elif vector_store is not None:

        uploaded_id_path = "./uploaded_ids.json"
        if os.path.exists(uploaded_id_path):
            with open(uploaded_id_path, "r") as f:
                uploaded_ids = set(json.load(f))
        else:
            uploaded_ids = set()
        
        to_be_upload_ids = set()
        new_uploaded_ids = set()

        text_embeddings = []
        contents = []
        # print(clusters.values())

        for id, emb in embeddings.items():
            if id in innovation_features and id not in uploaded_ids \
                    and id not in to_be_upload_ids:
                text_embeddings.append((id, emb))
                contents.append(innovation_features[id])
                to_be_upload_ids.add(id)
            else:
                print(f"[警告] {id} 被跳过")

        # 1000 时有 error_map 的bug
        batch_size = 500
        total_batches = math.ceil(len(text_embeddings) / batch_size)
        print(f"将上传 {len(text_embeddings)} 条嵌入向量，总共 {total_batches} 批")


        # 将 id 对应的 description 装入 metadata，不需要再使用 global innovation_features
        for i in range(total_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size

            batch_text_embeddings = text_embeddings[start_index:end_index]
            batch_contents = contents[start_index:end_index]
            metadatas = [{'source': content} for content in batch_contents]

            try:
                vector_store.add_embeddings(
                    text_embeddings=batch_text_embeddings,
                    metadatas=metadatas
                )

                for id,_ in batch_text_embeddings:
                    new_uploaded_ids.add(id)

                print(f"Successfully uploaded batch {i + 1}/{total_batches}")
            # Batch 失败就一个一个试
            except Exception as e:
                try:
                    print(f"Error batch")
                    for embd in batch_text_embeddings:
                        vector_store.add_embeddings(
                            text_embeddings=[embd],
                            metadatas=[{'source':innovation_features[embd[0]]}]
                        )
                        new_uploaded_ids.add(embd[0])

                except Exception as e:
                    print(f"Error uploading embedding: {e}")

        # 所有上传结束后，更新本地记录
        uploaded_ids.update(new_uploaded_ids)
        with open(uploaded_id_path, "w") as f:
            json.dump(sorted(list(uploaded_ids)), f, indent=2)

        print("Uploaded embeddings to Azure AI Search...")
        
    return canonical_mapping



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


def chat_bot(query:str) -> str:

    llm, embedding_model, vector_store, using_local_store = get_openai_models()

    
    if llm is None:
        return "Error: Language model not available. Please check your configuration."
    
    if embedding_model is None:
        return "Error: Embedding model unavailable. Please check your configuration."
    
    if vector_store is None:
        return "Error: Vectore store unavailable. Please check your configuration."
    

    # Set up Chatbot
    chatbot_prompt = PromptTemplate.from_template("""

        You're a smart assistant helping extract insights from VTT innovation relationships.

        Context:
        {context}

        According to the context, answer this question:
        {question}
    """)

    chatbot_llm = chatbot_prompt | llm

    if using_local_store:

        try:
            _, result_texts = vector_store.search([query], limit = 5)
            context = "\n".join(result_texts[0])

        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            context = "No relevant information found."

        try:
            llm_result = chatbot_llm.invoke({"context":context, "question":query})
            answer = llm_result.content
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            answer = f"Sorry, I couldn't process your question. Error: {str(e)}"


    else:
        try:
            results = vector_store.vector_search(query, k = 5)
            context = "\n".join([res.metadata['source'] for res in results])
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            context = "No relevant information found."

        try:
            llm_result = chatbot_llm.invoke({"context":context, "question":query})
            answer = llm_result.content
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            answer = f"Sorry, I couldn't process your question. Error: {str(e)}"
    
    return answer

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

    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--auto-label", action="store_true",
                       help="Automatically label consistency samples and generate gold standard files")
    
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
    
    # Step 1: Load and combine data (modified to also collect predictions)
    df_relationships, all_pred_entities, all_pred_relations = load_and_combine_data()
    
    # Step 2: Initialize OpenAI client

    llm, embed_model, vector_store, using_local_store = get_openai_models()

    
    if llm is None:
        print("Warning: Language model not available. Some features may be limited.")
    
    if embed_model is None:
        print("Warning: Embedding model not available. Using TF-IDF embeddings as fallback.")

    if vector_store is None:
        print("Warning: AI search unavailable.")

    # Step 3: Resolve innovation duplicates
    canonical_mapping = resolve_innovation_duplicates(
        df_relationships, 
        embed_model,
        vector_store,
        using_local_store,
        cache_config=cache_config
    )


    
    # Step 3: Resolve innovation duplicates
    canonical_mapping = resolve_innovation_duplicates(
        df_relationships=df_relationships,
        model=embed_model,
        cache_config=cache_config,
        method="hdbscan",  # 默认使用hdbscan
        min_cluster_size=2,  # 可配置参数
        metric="cosine",
        cluster_selection_method="eom"
    )
        
    # Step 4: Create consolidated knowledge graph
    consolidated_graph = create_innovation_knowledge_graph(df_relationships, canonical_mapping)
    print(consolidated_graph['organizations'].get("FI01120389"))
    
    # Step 5: Analyze innovation network
    analysis_results = analyze_innovation_network(consolidated_graph)
    
    # Step 6: Visualize network

    visualize_network_tufte(analysis_results)
    
    # Save predicted entities and relationships for evaluation
    os.makedirs("evaluation", exist_ok=True)
    
    # Remove duplicates from predicted entities and relations
    unique_pred_entities = []
    seen_entities = set()
    for entity in all_pred_entities:
        entity_key = (entity["name"].lower(), entity["type"])
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            unique_pred_entities.append(entity)
    
    unique_pred_relations = []
    seen_relations = set()
    for relation in all_pred_relations:
        relation_key = (relation["innovation"].lower(), relation["organization"].lower(), relation["relation"])
        if relation_key not in seen_relations:
            seen_relations.add(relation_key)
            unique_pred_relations.append(relation)
    
    # Save to JSON files
    pred_entities_path = os.path.join("evaluation", "pred_entities.json")
    pred_relations_path = os.path.join("evaluation", "pred_relations.json")
    
    with open(pred_entities_path, "w", encoding="utf-8") as f:
        json.dump(unique_pred_entities, f, ensure_ascii=False, indent=2)
    
    with open(pred_relations_path, "w", encoding="utf-8") as f:
        json.dump(unique_pred_relations, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(unique_pred_entities)} unique predicted entities to {pred_entities_path}")
    print(f"Saved {len(unique_pred_relations)} unique predicted relations to {pred_relations_path}")
    
    # Step 7: Export results
    export_results(analysis_results, consolidated_graph, canonical_mapping)
    
    # Step 8: Run evaluation if not skipped
    if not args.skip_eval:
        # Convert consolidated_graph to Node and Relationship objects for evaluation
        from evaluation import run_all_evaluations
        
        # Create Node objects for merged innovations
        merged_innovations = []
        for inno_id, inno_data in consolidated_graph['innovations'].items():
            node = Node(
                id=inno_id,
                type="Innovation",
                properties={
                    "aliases": "|".join(list(inno_data['names'])) if inno_data['names'] else "",
                    "source_docs": "|".join(list(inno_data['descriptions'])) if inno_data['descriptions'] else "",
                    "developed_by": "|".join(list(inno_data['developed_by'])) if inno_data['developed_by'] else "",
                    "sources": "|".join(list(inno_data['sources'])) if inno_data['sources'] else ""
                }
            )
            merged_innovations.append(node)
        
        # Create all nodes
        all_nodes = []
        
        # Add innovations
        for inno_id, inno_data in consolidated_graph['innovations'].items():
            node = Node(
                id=inno_id,
                type="Innovation",
                properties={
                    "name": list(inno_data['names'])[0] if inno_data['names'] else inno_id,
                    "description": list(inno_data['descriptions'])[0] if inno_data['descriptions'] else ""
                }
            )
            all_nodes.append(node)
        
        # Add organizations
        for org_id, org_data in consolidated_graph['organizations'].items():
            node = Node(
                id=org_id,
                type="Organization",
                properties={
                    "name": str(org_data['name']) if org_data['name'] is not None else "",
                    "description": str(org_data['description']) if org_data['description'] is not None else ""
                }
            )
            all_nodes.append(node)
        
        # Create relationships
        all_rels = []
        for rel_data in consolidated_graph['relationships']:
            rel = Relationship(
                source=rel_data['source'],
                source_type="Innovation" if rel_data['source'] in consolidated_graph['innovations'] else "Organization",
                target=rel_data['target'],
                target_type="Organization" if rel_data['target'] in consolidated_graph['organizations'] else "Innovation",
                type=rel_data['type'],
                properties={}
            )
            all_rels.append(rel)
        
        # Run evaluation
        evaluation_results = run_all_evaluations(
            merged_innovations=merged_innovations,
            all_nodes=all_nodes,
            all_rels=all_rels,
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            eval_dir="evaluation",
            auto_label=args.auto_label,
            llm=llm  # 传递语言模型以便自动标注
        )
    
    print("Innovation Resolution process completed successfully!")
    print(f"Results and visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main() 
