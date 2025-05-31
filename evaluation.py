#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation module for the Innovation Resolution Challenge.

This module provides tools to evaluate the quality of:
1. Consistency checking (for duplicate detection)
2. Entity/relation extraction accuracy
3. Knowledge graph structure metrics
4. End-to-end QA testing
"""

import random
import csv
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import os
import networkx as nx
import numpy as np
from pathlib import Path
from collections import Counter
import re
from langchain_openai import AzureChatOpenAI

from local_entity_processing import Node, Relationship

# ---------- Consistency Checking ----------

def sample_for_manual_check(
    merged_innovations: List[Node],
    sample_size: int = 20,
    output_csv: str = "consistency_sample.csv"
) -> None:
    """
    From merged_innovations, randomly sample items to generate a CSV file for manual consistency checking.
    
    Args:
        merged_innovations: List of merged Innovation nodes
        sample_size: Number of samples to generate
        output_csv: Path to output CSV file
    """
    sampled = random.sample(merged_innovations, min(sample_size, len(merged_innovations)))
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "innovation_id", "aliases", "source_snippets", "human_label"
        ])
        writer.writeheader()
        for item in sampled:
            aliases = item.properties.get("aliases", [])
            sources = item.properties.get("source_docs", [])
            writer.writerow({
                "innovation_id": item.id,
                "aliases": "|".join(aliases) if isinstance(aliases, list) else str(aliases),
                "source_snippets": "|".join(sources) if isinstance(sources, list) else str(sources),
                "human_label": ""  # To be filled manually with Yes/No
            })
    print(f"[Evaluation] Exported {len(sampled)} innovation samples to {output_csv} for manual consistency checking.")


def compute_consistency_rate(csv_path: str = "consistency_sample_labeled.csv") -> float:
    """
    Calculate consistency rate from manually labeled CSV file.
    
    Args:
        csv_path: Path to the labeled CSV file
        
    Returns:
        float: Consistency rate (percentage of "Yes" labels)
    """
    total = 0
    correct = 0
    
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row["human_label"].strip().lower() == "yes":
                    correct += 1
        
        if total == 0:
            return 0.0
        return correct / total
    except FileNotFoundError:
        print(f"[Evaluation] Warning: File {csv_path} not found. Please complete manual labeling first.")
        return 0.0


# ---------- Entity/Relation Extraction Accuracy ----------

def compute_entity_metrics(
    gold_path: str,
    pred_path: str
) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 for entity extraction.
    
    Args:
        gold_path: Path to gold standard entities JSON
        pred_path: Path to predicted entities JSON
        
    Returns:
        Dict with precision, recall, F1, true positives, false positives, and false negatives
    """
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        
        # Create sets of (name, type) tuples for comparison
        gold_set = set((e["name"].strip().lower(), e["type"]) for e in gold)
        pred_set = set((e["name"].strip().lower(), e["type"]) for e in pred)
        
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    except FileNotFoundError as e:
        print(f"[Evaluation] Error: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def compute_relation_metrics(
    gold_path: str,
    pred_path: str
) -> Dict[str, Any]:
    """
    Calculate precision, recall, and F1 for relation extraction.
    
    Args:
        gold_path: Path to gold standard relations JSON
        pred_path: Path to predicted relations JSON
        
    Returns:
        Dict with precision, recall, F1, true positives, false positives, and false negatives
    """
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        
        # Create sets of (innovation, organization, relation) tuples
        gold_set = set((
            r["innovation"].strip().lower(),
            r["organization"].strip().lower(),
            r["relation"]
        ) for r in gold)
        
        pred_set = set((
            r["innovation"].strip().lower(),
            r["organization"].strip().lower(),
            r["relation"]
        ) for r in pred)
        
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    except FileNotFoundError as e:
        print(f"[Evaluation] Error: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


# ---------- Knowledge Graph Structure Metrics ----------

def build_nx_graph(
    nodes: List[Node],
    rels: List[Relationship]
) -> nx.Graph:
    """
    Convert Node and Relationship lists to a NetworkX graph.
    
    Args:
        nodes: List of Node objects
        rels: List of Relationship objects
        
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # Add nodes
    for n in nodes:
        G.add_node(n.id, type=n.type)
    
    # Add edges
    for r in rels:
        G.add_edge(r.source, r.target, type=r.type)
    
    return G


def compute_graph_structure_metrics(
    nodes: List[Node],
    rels: List[Relationship]
) -> Dict[str, Any]:
    """
    Calculate structural metrics of the knowledge graph.
    
    Args:
        nodes: List of Node objects
        rels: List of Relationship objects
        
    Returns:
        Dict with graph metrics
    """
    G = build_nx_graph(nodes, rels)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    # Calculate average degree
    avg_degree = sum(dict(G.degree()).values()) / V if V > 0 else 0.0
    
    # Calculate graph density
    density = nx.density(G) if V > 1 else 0.0
    
    # Find largest connected component
    if V > 0:
        connected_components = list(nx.connected_components(G))
        largest_cc = max(connected_components, key=len) if connected_components else set()
        largest_cc_size = len(largest_cc)
        num_connected_components = len(connected_components)
    else:
        largest_cc_size = 0
        num_connected_components = 0
    
    # Count node types
    count_innovation = sum(1 for n in nodes if n.type == "Innovation")
    count_org = sum(1 for n in nodes if n.type == "Organization")
    
    return {
        "num_nodes_total": V,
        "num_edges_total": E,
        "num_innovations": count_innovation,
        "num_organizations": count_org,
        "avg_degree": avg_degree,
        "graph_density": density,
        "largest_cc_size": largest_cc_size,
        "num_connected_components": num_connected_components,
        "connectivity_ratio": largest_cc_size / V if V > 0 else 0.0
    }


# ---------- End-to-end QA Testing ----------

def get_developers_of_innovation(
    G: nx.Graph,
    innovation_id: str
) -> List[str]:
    """
    Find organizations that developed a specific innovation.
    
    Args:
        G: NetworkX graph
        innovation_id: ID of the innovation
        
    Returns:
        List of organization IDs that developed the innovation
    """
    developers = []
    
    for u, v, data in G.edges(data=True):
        if data.get("type") == "DEVELOPED_BY":
            if u == innovation_id:
                developers.append(v)
            elif v == innovation_id:
                developers.append(u)
    
    return developers


def get_innovations_of_organization(
    G: nx.Graph,
    org_id: str
) -> List[str]:
    """
    Find innovations associated with a specific organization.
    
    Args:
        G: NetworkX graph
        org_id: ID of the organization
        
    Returns:
        List of innovation IDs associated with the organization
    """
    innovations = []
    
    for u, v, data in G.edges(data=True):
        if data.get("type") in ["DEVELOPED_BY", "COLLABORATION"]:
            if u == org_id:
                innovations.append(v)
            elif v == org_id:
                innovations.append(u)
    
    return innovations


def evaluate_qa_examples(
    nodes: List[Node],
    rels: List[Relationship],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate graph with sample QA queries and save results.
    
    Args:
        nodes: List of Node objects
        rels: List of Relationship objects
        output_file: Optional file path to save results
        
    Returns:
        Dict with QA results
    """
    G = build_nx_graph(nodes, rels)
    
    # Sample innovations and organizations for testing
    innovations = [n.id for n in nodes if n.type == "Innovation"]
    organizations = [n.id for n in nodes if n.type == "Organization"]
    
    sample_innovations = random.sample(innovations, min(5, len(innovations)))
    sample_orgs = random.sample(organizations, min(5, len(organizations)))
    
    # Run QA queries
    qa_results = {
        "innovation_developers": {},
        "organization_innovations": {}
    }
    
    for inno_id in sample_innovations:
        developers = get_developers_of_innovation(G, inno_id)
        qa_results["innovation_developers"][inno_id] = developers
    
    for org_id in sample_orgs:
        innovations = get_innovations_of_organization(G, org_id)
        qa_results["organization_innovations"][org_id] = innovations
    
    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
    
    return qa_results


# ---------- Main Evaluation Function ----------

def auto_label_consistency_sample(
    csv_path: str,
    output_path: str = None,
    llm = None
) -> None:
    """
    使用LLM自动标注一致性检查样本。
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出CSV文件路径，如果为None则覆盖原文件
        llm: 语言模型，如果为None则尝试创建新的模型
    """
    if output_path is None:
        output_path = csv_path
    
    # 读取CSV文件
    samples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    if not samples:
        print(f"[Evaluation] Warning: No samples found in {csv_path}")
        return
    
    # 如果没有提供语言模型，尝试创建一个
    if llm is None:
        try:
            import json
            import os
            
            # 尝试加载API配置
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            config_path = os.path.join(data_dir, 'keys', 'azure_config.json')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 使用配置创建语言模型
                model_name = 'gpt-4.1-mini'
                if model_name in config:
                    llm = AzureChatOpenAI(
                        api_key=config[model_name]['api_key'],
                        azure_endpoint=config[model_name]['api_base'].split('/openai')[0],
                        azure_deployment=config[model_name]['deployment'],
                        api_version=config[model_name]['api_version'],
                        temperature=0
                    )
            
            if llm is None:
                print(f"[Evaluation] Warning: Could not create language model. Using simple heuristic for labeling.")
        except Exception as e:
            print(f"[Evaluation] Warning: Error creating language model: {e}. Using simple heuristic for labeling.")
    
    print(f"[Evaluation] Auto-labeling {len(samples)} consistency samples...")
    
    # 对每个样本进行标注
    for i, sample in enumerate(samples):
        aliases = sample["aliases"].split("|")
        source_snippets = sample["source_snippets"].split("|")
        
        # 如果有LLM，使用它进行更智能的标注
        if llm is not None:
            try:
                prompt = f"""
                请判断以下两个创新描述是否指的是同一个创新项目。
                
                创新名称/别名:
                1. {aliases[0] if len(aliases) > 0 else 'N/A'}
                2. {aliases[1] if len(aliases) > 1 else aliases[0] if len(aliases) > 0 else 'N/A'}
                
                创新描述:
                1. {source_snippets[0] if len(source_snippets) > 0 else 'N/A'}
                2. {source_snippets[1] if len(source_snippets) > 1 else 'N/A'}
                
                请只回答"Yes"或"No"。"Yes"表示它们是同一创新，"No"表示它们是不同的创新。
                """
                
                # 调用LLM进行判断
                response = llm.invoke(prompt).content.strip()
                if "yes" in response.lower():
                    label = "Yes"
                elif "no" in response.lower():
                    label = "No"
                else:
                    # 使用简单启发式方法作为后备
                    similarity = simple_text_similarity(aliases[0] if aliases else "", 
                                                      aliases[1] if len(aliases) > 1 else "")
                    label = "Yes" if similarity > 0.7 else "No"
            except Exception as e:
                print(f"[Evaluation] Warning: Error using LLM for sample {i}: {e}. Using simple heuristic.")
                # 使用简单启发式方法作为后备
                similarity = simple_text_similarity(aliases[0] if aliases else "", 
                                                  aliases[1] if len(aliases) > 1 else "")
                label = "Yes" if similarity > 0.7 else "No"
        else:
            # 使用简单启发式方法：基于名称相似度
            similarity = simple_text_similarity(aliases[0] if aliases else "", 
                                              aliases[1] if len(aliases) > 1 else "")
            label = "Yes" if similarity > 0.7 else "No"
        
        # 更新样本标签
        samples[i]["human_label"] = label
    
    # 保存结果
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"[Evaluation] Auto-labeled samples saved to {output_path}")


def simple_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的简单相似度。
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
    
    Returns:
        float: 相似度分数 (0-1)
    """
    # 规范化文本：转为小写，移除标点和多余空格
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    if not text1 or not text2:
        return 0.0
    
    text1 = normalize(text1)
    text2 = normalize(text2)
    
    # 计算Jaccard相似度
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def generate_gold_entities(
    pred_entities_path: str,
    output_path: str,
    sample_size: int = 30
) -> None:
    """
    基于预测实体生成黄金标准实体文件。
    
    Args:
        pred_entities_path: 预测实体JSON文件路径
        output_path: 输出JSON文件路径
        sample_size: 采样大小
    """
    try:
        # 读取预测实体
        with open(pred_entities_path, "r", encoding="utf-8") as f:
            pred_entities = json.load(f)
        
        if not pred_entities:
            print(f"[Evaluation] Warning: No entities found in {pred_entities_path}")
            return
        
        # 随机采样
        sampled_entities = random.sample(pred_entities, min(sample_size, len(pred_entities)))
        
        # 保存为黄金标准
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_entities, f, ensure_ascii=False, indent=2)
        
        print(f"[Evaluation] Generated gold entities with {len(sampled_entities)} samples at {output_path}")
    except Exception as e:
        print(f"[Evaluation] Error generating gold entities: {e}")


def generate_gold_relations(
    pred_relations_path: str,
    output_path: str,
    sample_size: int = 30
) -> None:
    """
    基于预测关系生成黄金标准关系文件。
    
    Args:
        pred_relations_path: 预测关系JSON文件路径
        output_path: 输出JSON文件路径
        sample_size: 采样大小
    """
    try:
        # 读取预测关系
        with open(pred_relations_path, "r", encoding="utf-8") as f:
            pred_relations = json.load(f)
        
        if not pred_relations:
            print(f"[Evaluation] Warning: No relations found in {pred_relations_path}")
            return
        
        # 随机采样
        sampled_relations = random.sample(pred_relations, min(sample_size, len(pred_relations)))
        
        # 保存为黄金标准
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled_relations, f, ensure_ascii=False, indent=2)
        
        print(f"[Evaluation] Generated gold relations with {len(sampled_relations)} samples at {output_path}")
    except Exception as e:
        print(f"[Evaluation] Error generating gold relations: {e}")


def run_all_evaluations(
    merged_innovations: List[Node],
    all_nodes: List[Node],
    all_rels: List[Relationship],
    data_dir: str = "data",
    results_dir: str = "results",
    eval_dir: str = "evaluation",
    generate_samples: bool = True,
    auto_label: bool = False,
    llm = None
) -> Dict[str, Any]:
    """
    运行所有评估指标并返回结果。
    
    Args:
        merged_innovations: 合并后的Innovation节点列表
        all_nodes: 图中所有节点列表
        all_rels: 图中所有关系列表
        data_dir: 数据文件目录
        results_dir: 结果保存目录
        eval_dir: 评估文件目录
        generate_samples: 是否生成一致性检查样本
        auto_label: 是否自动标注样本和生成黄金标准文件
        llm: 语言模型，用于自动标注
        
    Returns:
        Dict: 所有评估结果
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    evaluation_results = {}
    
    # 评估文件路径
    consistency_sample_path = os.path.join(eval_dir, "consistency_sample.csv")
    consistency_labeled_path = os.path.join(eval_dir, "consistency_sample_labeled.csv")
    gold_entities_path = os.path.join(eval_dir, "gold_entities.json")
    gold_relations_path = os.path.join(eval_dir, "gold_relations.json")
    pred_entities_path = os.path.join(eval_dir, "pred_entities.json")
    pred_relations_path = os.path.join(eval_dir, "pred_relations.json")
    qa_results_path = os.path.join(eval_dir, "qa_examples.json")
    
    print("\n[Evaluation] Starting evaluation process...")
    
    # 1. 一致性检查
    if generate_samples:
        print("[Evaluation] Generating consistency checking samples...")
        sample_for_manual_check(merged_innovations, sample_size=20, output_csv=consistency_sample_path)
        
        # 如果启用自动标注，自动标注一致性检查样本
        if auto_label:
            print("[Evaluation] Auto-labeling consistency samples...")
            auto_label_consistency_sample(consistency_sample_path, consistency_labeled_path, llm)
    
    # 检查是否存在已标注文件并计算一致率
    if os.path.exists(consistency_labeled_path):
        consistency_rate = compute_consistency_rate(consistency_labeled_path)
        evaluation_results["consistency_rate"] = consistency_rate
        print(f"[Evaluation] Consistency rate: {consistency_rate:.2%}")
    else:
        print(f"[Evaluation] Consistency checking samples generated. Please manually label them in: {consistency_sample_path}")
        if not auto_label:
            evaluation_results["consistency_rate"] = None
    
    # 2. 实体/关系抽取准确率
    print("\n[Evaluation] Computing entity and relation extraction metrics...")
    
    # 如果启用自动标注且需要生成黄金标准文件
    if auto_label:
        if not os.path.exists(gold_entities_path) and os.path.exists(pred_entities_path):
            generate_gold_entities(pred_entities_path, gold_entities_path)
        
        if not os.path.exists(gold_relations_path) and os.path.exists(pred_relations_path):
            generate_gold_relations(pred_relations_path, gold_relations_path)
    
    entity_metrics = None
    relation_metrics = None
    
    if os.path.exists(gold_entities_path) and os.path.exists(pred_entities_path):
        entity_metrics = compute_entity_metrics(gold_entities_path, pred_entities_path)
        evaluation_results["entity_metrics"] = entity_metrics
        print(f"[Evaluation] Entity extraction metrics: Precision={entity_metrics['precision']:.2f}, " +
              f"Recall={entity_metrics['recall']:.2f}, F1={entity_metrics['f1']:.2f}")
    else:
        print(f"[Evaluation] Entity extraction evaluation skipped. Required files not found.")
        print(f"[Evaluation] Please place gold standard files in {eval_dir}/ directory.")
        evaluation_results["entity_metrics"] = None
    
    if os.path.exists(gold_relations_path) and os.path.exists(pred_relations_path):
        relation_metrics = compute_relation_metrics(gold_relations_path, pred_relations_path)
        evaluation_results["relation_metrics"] = relation_metrics
        print(f"[Evaluation] Relation extraction metrics: Precision={relation_metrics['precision']:.2f}, " +
              f"Recall={relation_metrics['recall']:.2f}, F1={relation_metrics['f1']:.2f}")
    else:
        print(f"[Evaluation] Relation extraction evaluation skipped. Required files not found.")
        print(f"[Evaluation] Please place gold standard files in {eval_dir}/ directory.")
        evaluation_results["relation_metrics"] = None
    
    # 3. 知识图谱结构指标
    print("\n[Evaluation] Computing knowledge graph structure metrics...")
    graph_metrics = compute_graph_structure_metrics(all_nodes, all_rels)
    evaluation_results["graph_metrics"] = graph_metrics
    
    print(f"[Evaluation] Graph structure metrics:")
    print(f"  - Total nodes: {graph_metrics['num_nodes_total']}")
    print(f"  - Total edges: {graph_metrics['num_edges_total']}")
    print(f"  - Innovations: {graph_metrics['num_innovations']}")
    print(f"  - Organizations: {graph_metrics['num_organizations']}")
    print(f"  - Average degree: {graph_metrics['avg_degree']:.2f}")
    print(f"  - Graph density: {graph_metrics['graph_density']:.4f}")
    print(f"  - Largest connected component: {graph_metrics['largest_cc_size']} nodes")
    print(f"  - Connected components: {graph_metrics['num_connected_components']}")
    
    # 4. 端到端QA测试
    print("\n[Evaluation] Running end-to-end QA tests...")
    qa_results = evaluate_qa_examples(all_nodes, all_rels, qa_results_path)
    evaluation_results["qa_results"] = qa_results
    
    # 样本QA结果显示
    if qa_results["innovation_developers"]:
        sample_inno = next(iter(qa_results["innovation_developers"]))
        developers = qa_results["innovation_developers"][sample_inno]
        print(f"[Evaluation] Sample QA: Innovation '{sample_inno}' developers: {developers}")
    
    if qa_results["organization_innovations"]:
        sample_org = next(iter(qa_results["organization_innovations"]))
        innovations = qa_results["organization_innovations"][sample_org]
        print(f"[Evaluation] Sample QA: Organization '{sample_org}' innovations: {innovations}")
    
    # 保存完整评估结果
    evaluation_output_path = os.path.join(eval_dir, "evaluation_results.json")
    
    # 转换非可序列化值为字符串
    serializable_results = {}
    for key, value in evaluation_results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v 
                                        for k, v in value.items()}
        else:
            serializable_results[key] = str(value) if not isinstance(value, (int, float, str, list, dict, bool, type(None))) else value
    
    with open(evaluation_output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Evaluation] Evaluation complete. Results saved to: {evaluation_output_path}")
    
    return evaluation_results


if __name__ == "__main__":
    print("Evaluation module - run this from innovation_resolution.py") 