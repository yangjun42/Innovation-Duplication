#streamlit run app.py 前端全在这个app
#chatbot，项目介绍
#引导用户点击

import os
import streamlit as st
from streamlit.components.v1 import html
from innovation_resolution import chat_bot

# ----------------------------
# Page Config & Title
# ----------------------------
st.set_page_config(page_title="VTT Innovation Platform", layout="wide")
st.title("🔍 VTT Innovation Knowledge Graph Platform")

# ----------------------------
# Introduction
# ----------------------------
st.markdown("""
Welcome to the **VTT Innovation Knowledge Graph Platform**.

This tool helps you explore relationships between innovations and organizations based on publicly available data.
The platform includes:
- A **semantic assistant** that helps you navigate the graph via natural language queries.
- **Interactive resultsizations** of innovation networks in both 2D and 3D.
- **Statistical dashboards** that summarize key patterns and contributors.

Scroll down to explore each module below.
""")
#wanchengle
# --- Display HTML with Expand Button ---
html_path = "results/innovation_network_3d.html"
if os.path.exists(html_path):
    st.subheader("3D Interactive Network (Plotly)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")

# --- Display HTML with Expand Button ---

html_path = "results/innovation_network_tufte_3D.html"
if os.path.exists(html_path):
    st.subheader("3D Interactive Network (Plotly)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")
st.divider()


# ----------------------------
# Network Graph Visualizations
# ----------------------------
st.header("🌐 Network Graph Visualizations")

img_path = "results/innovation_network_tufte_2D.png"
if os.path.exists(img_path):
    st.subheader("🖼️ 2D Network Snapshot (PNG)")
    st.image(img_path, use_column_width=True)
else:
    st.warning("2D PNG image not found.")

# ----------------------------
# Innovation Metrics Dashboard
# ----------------------------
st.header("📈 Innovation Metrics Dashboard")

img_stat = "results/innovation_stats_tufte.png"
if os.path.exists(img_stat):
    st.subheader("Key Innovation Statistics")
    st.image(img_stat, use_column_width=True)
else:
    st.warning("Innovation stats image not found.")

img_top_orgs = "results/top_organizations_tufte.png"
if os.path.exists(img_top_orgs):
    st.subheader("Top Contributing Organizations")
    st.image(img_top_orgs, use_column_width=True)
else:
    st.warning("Top organizations image not found.")


# ----------------------------
# Semantic Graph Assistant
# ----------------------------

#chatbot部分
st.header("🧠 Semantic Graph Assistant")

query = st.text_input("Ask a question (e.g., 'Which organizations developed the most innovations?'):")

if query:
    # TODO: Replace with actual FAISS + LangChain response
    st.info("This is a placeholder response. Semantic search will be available here soon.")
    with st.spinner("Thinking..."):
        reply = chat_bot(query)
    st.success("Response:")
    st.markdown(reply)

st.divider()