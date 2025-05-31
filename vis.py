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
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go

RESULTS_DIR = "results"

# 可区分的配色和线型
COLOR_PALETTE = {
    "Innovation": "#1f77b4",
    "Organization": "#2ca02c",
    "Edge_Developed": "#e15759",
    "Edge_Collaboration": "#4e79a7",
    "Unknown": "#bbbbbb"
}
EDGE_STYLE = {"DEVELOPED_BY": "solid", "COLLABORATION": "dashed"}

def visualize_network_tufte(analysis_results: dict):
    visualize_network_tufte_2D(analysis_results)
    visualize_network_tufte_3D(analysis_results)
    visualize_network_tufte_bar(analysis_results)

def visualize_network_tufte_2D(analysis_results: dict):
    G: nx.Graph = analysis_results['graph']

    # 仅画degree Top-N及其一阶邻居
    MAX_NODES = 64
    deg_top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:MAX_NODES]
    nodes_keep = set(n for n, _ in deg_top)
    for n, _ in deg_top:
        nodes_keep |= set(G.neighbors(n))
    H = G.subgraph(nodes_keep).copy()

    plt.figure(figsize=(16, 11), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.axis('off')

    pos = nx.spring_layout(H, k=0.25, seed=42)

    # 节点样式
    node_colors, node_sizes = [], []
    node_types = nx.get_node_attributes(H, 'type')
    sources_counts = nx.get_node_attributes(H, 'sources')
    for n in H.nodes():
        ntype = node_types.get(n, 'Unknown')
        if ntype == 'Innovation':
            count = sources_counts.get(n, 1)
            size = 10 + min(count, 10) * 8
            node_sizes.append(size)
            node_colors.append(COLOR_PALETTE['Innovation'])
        elif ntype == 'Organization':
            node_sizes.append(22)
            node_colors.append(COLOR_PALETTE['Organization'])
        else:
            node_sizes.append(18)
            node_colors.append(COLOR_PALETTE['Unknown'])

    # 边样式
    for etype, color in [("DEVELOPED_BY", COLOR_PALETTE["Edge_Developed"]),
                         ("COLLABORATION", COLOR_PALETTE["Edge_Collaboration"])]:
        edge_list = [(u, v) for u, v, d in H.edges(data=True) if d.get('type', 'COLLABORATION') == etype]
        nx.draw_networkx_edges(
            H, pos, edgelist=edge_list,
            width=0.5, alpha=0.25, edge_color=color,
            style=EDGE_STYLE.get(etype, 'solid')
        )

    nx.draw_networkx_nodes(
        H, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=0.2,
        edgecolors='black'
    )

    # 不画任何标签，避免混乱

    # 图例移至空白角，去边框
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_PALETTE['Innovation'], markersize=10, label='Innovation'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLOR_PALETTE['Organization'], markersize=8, label='Organization'),
        plt.Line2D([0], [0], color=COLOR_PALETTE['Edge_Developed'], lw=2, label='Developed By'),
        plt.Line2D([0], [0], color=COLOR_PALETTE['Edge_Collaboration'], lw=2, label='Collaboration'),
    ]
    ax.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(RESULTS_DIR, 'innovation_network_tufte_2D.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("2D network visualization saved: innovation_network_tufte_2D.png")

def visualize_network_tufte_3D(analysis_results: dict):
    G: nx.Graph = analysis_results['graph']

    # 只画最多50个节点的子图
    SUB_N = 128
    deg_top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:SUB_N]
    nodes_keep = set(n for n, _ in deg_top)
    for n, _ in deg_top:
        nodes_keep |= set(G.neighbors(n))
    subG = G.subgraph(nodes_keep).copy()

    pos_3d = nx.spring_layout(subG, dim=3, k=0.25, iterations=50, seed=42)
    x_nodes = [pos_3d[n][0] for n in subG.nodes()]
    y_nodes = [pos_3d[n][1] for n in subG.nodes()]
    z_nodes = [pos_3d[n][2] for n in subG.nodes()]

    node_types = nx.get_node_attributes(subG, 'type')
    sources_counts = nx.get_node_attributes(subG, 'sources')
    node_colors, node_sizes, node_labels = [], [], []
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
        node_labels.append("")  # 静态图不显示

    # 边样式
    edge_traces = []
    for etype, color in [("DEVELOPED_BY", COLOR_PALETTE["Edge_Developed"]),
                         ("COLLABORATION", COLOR_PALETTE["Edge_Collaboration"])]:
        x_edges, y_edges, z_edges = [], [], []
        for u, v, d in subG.edges(data=True):
            if d.get('type') == etype:
                x_edges += [pos_3d[u][0], pos_3d[v][0], None]
                y_edges += [pos_3d[u][1], pos_3d[v][1], None]
                z_edges += [pos_3d[u][2], pos_3d[v][2], None]
        if x_edges:
            edge_trace = go.Scatter3d(
                x=x_edges, y=y_edges, z=z_edges,
                mode='lines', line=dict(color=color, width=1),
                hoverinfo='none', name=etype
            )
            edge_trace.update(render_mode='webgl')
            edge_traces.append(edge_trace)

    hover_texts = [
        f"{subG.nodes[n].get('name', subG.nodes[n].get('names', n))}<br>Type: {subG.nodes[n].get('type','Unknown')}"
        for n in subG.nodes()
    ]
    nodes_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        text=[""] * len(subG),
        hovertext=hover_texts,
        marker=dict(size=node_sizes, color=node_colors, opacity=0.9,
                    line=dict(width=0.2, color='black')),
        render_mode='webgl',
        name="Nodes"
    )

    fig = go.Figure(data=[nodes_trace] + edge_traces)
    fig.update_layout(
        title='VTT Innovation Network (3D Subgraph)',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, backgroundcolor='white'),
            yaxis=dict(showgrid=False, showticklabels=False, backgroundcolor='white'),
            zaxis=dict(showgrid=False, showticklabels=False, backgroundcolor='white')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title_text='Relation Types', x=0, y=1, bgcolor='rgba(255,255,255,0.5)')
    )
    fig.write_html(os.path.join(RESULTS_DIR, 'innovation_network_tufte_3D.html'))
    fig.write_image(os.path.join(RESULTS_DIR, 'innovation_network_tufte_3D.png'), width=1200, height=900)
    print("3D network visualization saved: innovation_network_tufte_3D.html & .png")

def visualize_network_tufte_bar(analysis_results: dict):
    stats = analysis_results['stats']
    top_orgs = analysis_results['top_orgs']
    G = analysis_results['graph']

    # (1) Innovation Statistics Barplot
    plt.figure(figsize=(7, 4), facecolor='white')
    labels = ['Total Innovations', 'Multi-Source Innovations', 'Multi-Developer Innovations']
    values = [stats['total'], stats['multi_source_count'], stats['multi_developer_count']]
    colors = ['#4c72b0', '#55a868', '#c44e52']
    bars = plt.bar(labels, values, width=0.6, color=colors)
    if len(values) <= 10:
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f'{int(bar.get_height())}', ha='center', va='bottom')
    plt.title('Innovation Statistics', fontsize=14, weight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('')
    plt.grid(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'innovation_stats_tufte.png'), dpi=300)
    plt.close()
    print("Innovation statistics bar plot saved: innovation_stats_tufte.png")

    # (2) Top Organizations Horizontal Barplot
    if top_orgs:
        filtered = [(oid, cnt) for oid, cnt in top_orgs if oid in G.nodes()]
        if filtered:
            filtered_sorted = sorted(filtered, key=lambda x: x[1])
            org_names, org_counts = [], []
            for oid, cnt in filtered_sorted:
                name = G.nodes[oid].get('name')
                if not name:
                    name = str(oid)
                # 截断长名称
                maxlen = 32
                if len(name) > maxlen:
                    name = name[:maxlen - 2] + "…"
                org_names.append(name)
                org_counts.append(cnt)
            # 梯度色
            palette = sns.light_palette('#6b56b5', n_colors=len(org_counts), reverse=True)
            plt.figure(figsize=(8, len(org_names) * 0.4 + 1), facecolor='white')
            bars = sns.barplot(y=org_names, x=org_counts, palette=palette, edgecolor='none')
            plt.xscale('log') if max(org_counts) > 50 else None
            for p, count in zip(bars.patches, org_counts):
                plt.text(p.get_width() + max(org_counts) * 0.01, p.get_y() + p.get_height() / 2,
                         f"{int(count)}", va='center', fontsize=9)
            plt.title('Top Organizations by Innovation Count', fontsize=14, weight='bold')
            plt.xlabel('Number of Innovations', fontsize=12)
            plt.ylabel('')
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'top_organizations_tufte.png'), dpi=300)
            plt.close()
            print("Top organizations bar plot saved: top_organizations_tufte.png")
    else:
        print("No top organizations data to plot.")
