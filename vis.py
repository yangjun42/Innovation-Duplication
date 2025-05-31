#!/usr/bin/env python

"""
VTT Innovation Resolution Challenge Solution (Tufte-inspired Visualizations)

将 visualize_network() 拆分为：
  - visualize_network_tufte_2D()
  - visualize_network_tufte_3D()
  - visualize_network_tufte_bar()
由 visualize_network_tufte() 统一调用。
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from innovation_utils import (
    compute_similarity_matrix,
    find_potential_duplicates,
    calculate_innovation_statistics
)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 配色方案
COLOR_PALETTE = {
    "Innovation": "#1f77b4",   # 深蓝
    "Organization": "#2ca02c", # 深绿
    "Edge_Developed": "#d62728",       # 红
    "Edge_Collaboration": "#9467bd",   # 紫
    "Unknown": "#bbbbbb"       # 淡灰
}


def visualize_network_tufte(analysis_results: dict):
    """
    Tufte-inspired 总调用函数，依次调用 2D 网络图、3D 交互网络图、以及统计条形图。
    Args:
        analysis_results: 从 analyze_innovation_network() 返回的字典
    """
    visualize_network_tufte_2D(analysis_results)
    visualize_network_tufte_3D(analysis_results)
    visualize_network_tufte_bar(analysis_results)


def visualize_network_tufte_2D(analysis_results: dict):
    """
    Tufte 风格的二维网络图可视化：
      - 去除网格与坐标轴背景
      - 固定布局以保持可重现
      - 节点按 Innovation/Organization 区分颜色与大小
      - 仅标注前 5 位关键节点
    Args:
        analysis_results: analyze_innovation_network() 返回的字典
    """
    G: nx.Graph = analysis_results['graph']

    # 1. 设置画布，纯白背景
    plt.figure(figsize=(14, 10), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.axis('off')

    # 2. 固定布局
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    # 3. 计算节点颜色与大小
    node_colors = []
    node_sizes = []
    node_types = nx.get_node_attributes(G, 'type')
    sources_counts = nx.get_node_attributes(G, 'sources')

    for n in G.nodes():
        ntype = node_types.get(n, 'Unknown')
        if ntype == 'Innovation':
            count = sources_counts.get(n, 1)
            # 将 sources_count 映射到尺寸 [10, 100]
            size = 10 + min(count, 10) * 8
            node_sizes.append(size)
            node_colors.append(COLOR_PALETTE['Innovation'])
        elif ntype == 'Organization':
            node_sizes.append(20)
            node_colors.append(COLOR_PALETTE['Organization'])
        else:
            node_sizes.append(20)
            node_colors.append(COLOR_PALETTE['Unknown'])

    # 4. 计算边颜色
    edge_colors = []
    for u, v, d in G.edges(data=True):
        etype = d.get('type', 'OTHER')
        if etype == 'DEVELOPED_BY':
            edge_colors.append(COLOR_PALETTE['Edge_Developed'])
        elif etype == 'COLLABORATION':
            edge_colors.append(COLOR_PALETTE['Edge_Collaboration'])
        else:
            edge_colors.append(COLOR_PALETTE['Unknown'])

    # 5. 绘制节点与边
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.2,
        edgecolors='black'
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=0.7,
        alpha=0.5
    )

    # 6. 仅标注 top 5 关键节点
    key_orgs = [node for node, _ in analysis_results['key_orgs'][:5]]
    key_innos = [node for node, _ in analysis_results['key_innovations'][:5]]
    key_nodes = set(key_orgs + key_innos)
    labels = {
        n: (G.nodes[n].get('name') if G.nodes[n].get('type') == 'Organization'
            else G.nodes[n].get('names', n))
        for n in key_nodes
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    # 7. 添加精简图例
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_PALETTE['Innovation'], markersize=10,
                   label='Innovation'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_PALETTE['Organization'], markersize=8,
                   label='Organization'),
        plt.Line2D([0], [0], color=COLOR_PALETTE['Edge_Developed'], lw=2,
                   label='Developed By'),
        plt.Line2D([0], [0], color=COLOR_PALETTE['Edge_Collaboration'], lw=2,
                   label='Collaboration'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(RESULTS_DIR, 'innovation_network_tufte_2D.png'), dpi=300)
    plt.close()
    print("2D network visualization saved: innovation_network_tufte_2D.png")


def visualize_network_tufte_3D(analysis_results: dict):
    """
    Tufte 风格的 3D 交互网络图，可视化最核心节点（如只展示前 50 节点，避免过度拥挤）：
      - 使用 Plotly 3D Scatter
      - 仅标记最重要的局部，让评委专注而不过度分散注意力
    Args:
        analysis_results: analyze_innovation_network() 返回的字典
    """
    G: nx.Graph = analysis_results['graph']

    # 如果网络节点超多，可先提取前 50 或前若干度数最高节点的子图
    if len(G.nodes) > 100:
        # 以 degree 排序，取前 50
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:50]
        subG = G.subgraph(top_nodes).copy()
    else:
        subG = G

    # 1. 3D 布局（固定 seed）
    pos_3d = nx.spring_layout(subG, dim=3, k=0.15, iterations=50, seed=42)

    # 2. 提取坐标与属性
    x_nodes = [pos_3d[n][0] for n in subG.nodes()]
    y_nodes = [pos_3d[n][1] for n in subG.nodes()]
    z_nodes = [pos_3d[n][2] for n in subG.nodes()]

    node_types = nx.get_node_attributes(subG, 'type')
    sources_counts = nx.get_node_attributes(subG, 'sources')

    node_colors = []
    node_sizes = []
    node_labels = []
    for n in subG.nodes():
        ntype = node_types.get(n, 'Unknown')
        if ntype == 'Innovation':
            cnt = sources_counts.get(n, 1)
            node_sizes.append(5 + min(cnt, 10) * 2)
            node_colors.append(COLOR_PALETTE['Innovation'])
        elif ntype == 'Organization':
            node_sizes.append(5)
            node_colors.append(COLOR_PALETTE['Organization'])
        else:
            node_sizes.append(5)
            node_colors.append(COLOR_PALETTE['Unknown'])

        # 仅最重要节点显示标签
        if n in [nd for nd, _ in analysis_results['key_innovations'][:3]] \
           or n in [nd for nd, _ in analysis_results['key_orgs'][:3]]:
            label = (subG.nodes[n].get('name') 
                     if subG.nodes[n].get('type') == 'Organization' 
                     else subG.nodes[n].get('names', n))
        else:
            label = ""
        node_labels.append(label)

    # 3. 边 trace（按类型分组）
    edge_traces = []
    edge_types = {
        'DEVELOPED_BY': COLOR_PALETTE['Edge_Developed'],
        'COLLABORATION': COLOR_PALETTE['Edge_Collaboration']
    }
    for etype, color in edge_types.items():
        x_edges, y_edges, z_edges = [], [], []
        for u, v, d in subG.edges(data=True):
            if d.get('type') == etype:
                x_edges += [pos_3d[u][0], pos_3d[v][0], None]
                y_edges += [pos_3d[u][1], pos_3d[v][1], None]
                z_edges += [pos_3d[u][2], pos_3d[v][2], None]
        if x_edges:
            edge_traces.append(
                go.Scatter3d(
                    x=x_edges, y=y_edges, z=z_edges,
                    mode='lines',
                    line=dict(color=color, width=1),
                    hoverinfo='none',
                    name=etype
                )
            )

    # 4. 节点 trace
    hover_texts = [
        f"{subG.nodes[n].get('name', subG.nodes[n].get('names', n))}<br>Type: {subG.nodes[n].get('type','Unknown')}"
        for n in subG.nodes()
    ]
    nodes_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        text=node_labels,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(width=0.2, color='black')
        ),
        hoverinfo='text',
        hovertext=hover_texts,
        name="Nodes"
    )

    # 5. 组合并渲染
    fig = go.Figure(data=[nodes_trace] + edge_traces)
    fig.update_layout(
        title='VTT Innovation Network (3D Subgraph)',
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title_text='Relation Types', x=0, y=1, bgcolor='rgba(255,255,255,0.5)')
    )

    # 6. 保存交互 HTML 和静态图
    fig.write_html(os.path.join(RESULTS_DIR, 'innovation_network_tufte_3D.html'))
    fig.write_image(os.path.join(RESULTS_DIR, 'innovation_network_tufte_3D.png'),
                    width=1200, height=900)
    print("3D network visualization saved: innovation_network_tufte_3D.html & .png")

def visualize_network_tufte_bar(analysis_results: dict):
    """
    Tufte 风格的统计条形图：Innovation 统计 & Top Organizations
    - 去除多余网格，仅在需要时保留轻微网格
    - 统一纯色、高对比度
    - 数值标注提升易读性
    Args:
        analysis_results: analyze_innovation_network() 返回的字典
    """
    stats = analysis_results['stats']
    top_orgs = analysis_results['top_orgs']
    G = analysis_results['graph']

    # --- (1) Innovation Statistics Barplot ---
    plt.figure(figsize=(8, 5), facecolor='white')
    labels = ['Total Innovations', 'Multi-Source Innovations', 'Multi-Developer Innovations']
    values = [stats['total'], stats['multi_source_count'], stats['multi_developer_count']]
    colors = ['#4c72b0', '#55a868', '#c44e52']  # 蓝, 绿, 红

    bars = plt.bar(labels, values, color=colors, edgecolor='gray')
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(values) * 0.01,
            f"{int(height)}",
            ha='center', va='bottom', fontsize=9
        )

    plt.title('Innovation Statistics', fontsize=14, weight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('')
    plt.grid(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'innovation_stats_tufte.png'), dpi=300)
    plt.close()
    print("Innovation statistics bar plot saved: innovation_stats_tufte.png")

    # --- (2) Top Organizations Horizontal Barplot ---
    if top_orgs:
        # 过滤出实际存在于网络中的组织
        filtered = [(oid, cnt) for oid, cnt in top_orgs if oid in G.nodes()]
        if filtered:
            # 按 count 升序，这样最大的在最顶部
            filtered_sorted = sorted(filtered, key=lambda x: x[1])

            # 确保名称不为 None，用 oid 的字符串表示替代
            org_names = []
            org_counts = []
            for oid, cnt in filtered_sorted:
                name = G.nodes[oid].get('name')
                if not name:
                    # 如果节点上没有 'name' 属性，就直接用 oid 转字符串
                    name = str(oid)
                org_names.append(name)
                org_counts.append(cnt)

            # 这里修正了 figsize 括号位置
            plt.figure(figsize=(8, len(org_names) * 0.4 + 1), facecolor='white')
            bars = plt.barh(org_names, org_counts, color='#6b56b5', edgecolor='gray')
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + max(org_counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    va='center', fontsize=9
                )

            plt.title('Top Organizations by Innovation Count', fontsize=14, weight='bold')
            plt.xlabel('Number of Innovations', fontsize=12)
            plt.ylabel('')
            plt.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'top_organizations_tufte.png'), dpi=300)
            plt.close()
            print("Top organizations bar plot saved: top_organizations_tufte.png")
    else:
        print("No top organizations data to plot.")

