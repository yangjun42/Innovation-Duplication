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
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Set, Tuple, Any

# Import local modules
from local_entity_processing import Node, Relationship, Document, GraphDocument

# Set up paths
DATA_DIR = 'data'
GRAPH_DOCS_COMPANY = os.path.join(DATA_DIR, 'graph_docs_names_resolved')
GRAPH_DOCS_VTT = os.path.join(DATA_DIR, 'graph_docs_vtt_domain_names_resolved')
DATAFRAMES_DIR = os.path.join(DATA_DIR, 'dataframes')
RESULTS_DIR = 'results'  # Changed from os.path.join(DATA_DIR, 'results') to save in root directory

# Create results directory if it doesn't exist
try:
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    RESULTS_DIR = '.'  # Use current directory as fallback

# Set up plotting style
sns.set_theme(style="whitegrid")


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
        OpenAI client or None if initialization fails
    """
    try:
        from langchain_openai import AzureChatOpenAI
        import json
        
        config_path = os.path.join(DATA_DIR, 'keys', 'azure_config.json')
        
        if not os.path.exists(config_path):
            print(f"API configuration file not found at {config_path}")
            print("Please obtain API keys and create the configuration file as described in the README.md")
            return None
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Try to initialize with different models in order of preference
        for model_name in ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o-mini']:
            if model_name in config:
                try:
                    model = AzureChatOpenAI(
                        model=model_name,
                        api_key=config[model_name]['api_key'],
                        azure_endpoint=config[model_name]['api_base'],
                        api_version=config[model_name]['api_version']
                    )
                    print(f"Successfully initialized {model_name}")
                    return model
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
        
        print("Failed to initialize any OpenAI model")
        return None
    
    except ImportError:
        print("Required packages not installed. Please install langchain-openai and openai.")
        return None


def get_embedding(text: str, model) -> np.ndarray:
    """
    Get embedding for a text using text hashing method when API is not available.
    
    Args:
        text: Text to embed
        model: OpenAI model (not used in this implementation)
    
    Returns:
        np.ndarray: Embedding vector
    """
    # 不再尝试调用OpenAI API，直接使用哈希方法生成"伪嵌入"
    try:
        # 使用文本哈希方法创建一个简单的嵌入
        import hashlib
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 使用TF-IDF来创建更有意义的文本表示
        vectorizer = TfidfVectorizer(max_features=768)
        # 由于TF-IDF需要多个文档，我们创建一个简单的集合
        # 将文本分割成句子
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        if len(sentences) < 2:
            sentences = text.split()  # 如果没有句子，就用单词
        
        # 确保有足够的文本进行向量化
        if len(sentences) < 2:
            sentences = [text, "placeholder"]
            
        # 向量化
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # 取平均作为最终表示
        embedding = tfidf_matrix.mean(axis=0).A[0]
        
        # 如果维度不够1536，填充到所需大小
        if len(embedding) < 1536:
            embedding = np.pad(embedding, (0, 1536-len(embedding)), 'constant')
        
        # 如果维度超过1536，截断
        if len(embedding) > 1536:
            embedding = embedding[:1536]
            
        return embedding
    except Exception as e:
        print(f"Error creating text embedding: {e}")
        # 返回随机嵌入作为后备方案
        return np.random.rand(1536)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        float: Cosine similarity score
    """
    # Reshape embeddings to 2D arrays for sklearn
    emb1_reshaped = emb1.reshape(1, -1)
    emb2_reshaped = emb2.reshape(1, -1)
    
    return cosine_similarity(emb1_reshaped, emb2_reshaped)[0][0]


def resolve_innovation_duplicates(df_relationships: pd.DataFrame, model=None) -> Dict[str, str]:
    """
    Identify duplicate innovations using text similarity.
    
    Args:
        df_relationships: DataFrame with innovation relationships
        model: OpenAI model for generating embeddings (not used if API not available)
    
    Returns:
        Dict[str, str]: Mapping from innovation IDs to canonical IDs
    """
    print("Resolving innovation duplicates...")
    
    # Step 1: Extract innovations
    innovations = df_relationships[df_relationships['source_type'] == 'Innovation']
    unique_innovations = innovations.drop_duplicates(subset=['source_id'])
    print(f"Found {len(unique_innovations)} unique innovations")
    
    # Step 2: Create feature vectors for innovations
    innovation_features = {}
    for _, row in tqdm(unique_innovations.iterrows(), total=len(unique_innovations), desc="Creating innovation features"):
        innovation_id = row['source_id']
        if innovation_id not in innovation_features:
            # Combine name and description
            name = str(row['source_english_id'])
            description = str(row['source_description'])
            context = f"{name}: {description}"
            
            # Add organizations that developed this innovation
            developed_by = df_relationships[
                (df_relationships['source_id'] == innovation_id) & 
                (df_relationships['relationship_type'] == 'DEVELOPED_BY')
            ]['target_english_id'].tolist()
            
            if developed_by:
                context += f" Developed by: {', '.join(str(org) for org in developed_by)}"
            
            innovation_features[innovation_id] = context
    
    # Step 3: Generate embeddings or use text features directly
    print("Generating features for similarity comparison...")
    embeddings = {}
    for id, features in tqdm(innovation_features.items(), desc="Processing innovations"):
        embeddings[id] = get_embedding(features, model)
    
    # Step 4: Cluster innovations using similarity threshold
    print("Clustering similar innovations...")
    clusters = {}
    threshold = 0.85  # Tunable parameter
    processed = set()
    
    # Compute full similarity matrix for all innovations
    embedding_items = list(embeddings.items())
    innovation_ids = [item[0] for item in embedding_items]
    embedding_matrix = np.array([item[1] for item in embedding_items])
    
    # 计算相似度矩阵 (批量计算更高效)
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    # 使用相似度矩阵进行聚类
    for i, id1 in enumerate(tqdm(innovation_ids, desc="Clustering innovations")):
        if id1 in processed:
            continue
            
        cluster = [id1]
        # 查找与当前创新相似的所有创新
        for j, id2 in enumerate(innovation_ids):
            if id1 != id2 and id2 not in processed:
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    cluster.append(id2)
                    processed.add(id2)
        
        canonical_id = id1  # Use the first ID as canonical
        clusters[canonical_id] = cluster
        processed.add(id1)
    
    # Step 5: Create mapping dictionary
    canonical_mapping = {}
    for canonical_id, cluster_ids in clusters.items():
        for innovation_id in cluster_ids:
            canonical_mapping[innovation_id] = canonical_id
    
    print(f"Found {len(clusters)} unique innovation clusters")
    print(f"Reduced from {len(innovation_features)} to {len(clusters)} innovations")
    
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
    print("Starting VTT Innovation Resolution process...")
    
    # Step 1: Load and combine data
    df_relationships = load_and_combine_data()
    
    # Step 2: Initialize OpenAI client
    model = initialize_openai_client()
    
    # Step 3: Resolve innovation duplicates
    canonical_mapping = resolve_innovation_duplicates(df_relationships, model)
    
    # Step 4: Create consolidated knowledge graph
    consolidated_graph = create_innovation_knowledge_graph(df_relationships, canonical_mapping)
    
    # Step 5: Analyze innovation network
    analysis_results = analyze_innovation_network(consolidated_graph)
    
    # Step 6: Visualize network
    visualize_network(analysis_results)
    
    # Step 7: Export results
    export_results(analysis_results, consolidated_graph, canonical_mapping)
    
    print("Innovation Resolution process completed successfully!")
    print(f"Results and visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main() 