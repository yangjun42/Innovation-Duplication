#streamlit run app.py 前端全在这个app
#chatbot，项目介绍
#引导用户点击

import os
import sys
import streamlit as st
from streamlit.components.v1 import html
import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

# 尝试生成配置文件（如果文件存在）
try:
    config_generator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_config_from_toml.py")
    if os.path.exists(config_generator):
        print("尝试生成配置文件...")
        import subprocess
        result = subprocess.run([sys.executable, config_generator], capture_output=True, text=True)
        if result.returncode == 0:
            print("配置文件生成成功！")
            print(result.stdout)
        else:
            print(f"配置文件生成失败: {result.stderr}")
    else:
        print(f"配置生成脚本不存在: {config_generator}")
except Exception as e:
    print(f"尝试生成配置文件时出错: {str(e)}")

# 创建数据目录和密钥目录（如果不存在）
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
keys_dir = os.path.join(data_dir, "keys")
os.makedirs(keys_dir, exist_ok=True)

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
#### Welcome to the **VTT Innovation Knowledge Graph Platform**!

#### This tool helps you explore relationships between innovations and organizations based on publicly available data
#### The platform includes:
- 🌐 **Interactive resultsizations** of innovation networks in both 2D and 3D.
- 🌟 **Statistical dashboards** that summarize key patterns and contributors.
- 🧠 A **semantic assistant** that helps you navigate the graph via natural language queries.

#### Scroll down to explore each module below:
""")
#wanchengle
# --- Display HTML with Expand Button ---
st.header("🌐 Network Graph Visualizations")


#在ein的本地
html_path = "results/innovation_network_3d.html"
if os.path.exists(html_path):
    st.subheader("Interactive Network (Before Dedupulication)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")

# --- Display HTML with Expand Button ---

st.markdown("""
These visualizations represent the relationships between innovations and organizations:

- **Blue nodes**: Innovations
- **Green nodes**: Organizations
- **Red edges**: "Developed By" relationships
- **Blue edges**: Collaborations

Hover or zoom to explore the connections. The layout is generated based on semantic clustering.
""")

html_path = "results/innovation_network_tufte_3D.html"
if os.path.exists(html_path):
    st.subheader("3D Interactive Network (After Dedupulication)")
    with open(html_path, "r", encoding="utf-8") as f:
        html(f.read(), height=600)
else:
    st.warning("3D HTML file not found. Please run the backend script to generate it.")
st.divider()



# ----------------------------
# Innovation Metrics Dashboard
# ----------------------------
st.header(" 🌟 Innovation Metrics Dashboard")
st.markdown("""
These charts summarize statistical patterns in the innovation network:

- Count of innovations
- Proportion of multi-source or multi-developer innovations
- Top contributing organizations
""")

img_path = "results/innovation_network_tufte_2D.png"
if os.path.exists(img_path):
    st.subheader("2D Network Snapshot")
    st.image(img_path, use_container_width=True)
else:
    st.warning("2D PNG image not found.")




# 第二行：两列展示 Statistics 和 Top Organizations
col1, col2 = st.columns(2)

with col2:
    st.subheader("Key Innovation Statistics")
    img_stat = "results/innovation_stats_tufte.png"
    if os.path.exists(img_stat):
        st.image(img_stat, use_container_width=True)
        st.markdown("""
        Summary statistics highlighting:
        - Total innovations in the dataset
        - Innovations sourced from multiple data providers
        - Innovations developed by more than one organization
        """)
    else:
        st.warning("Innovation stats image not found.")

with col1:
    st.subheader("Top Contributing Organizations")
    img_top_orgs = "results/top_organizations.png"
    if os.path.exists(img_top_orgs):
        st.image(img_top_orgs, use_container_width=True)
        st.markdown("""
        - Organizations ranked by the number of innovations they have contributed to.
        - A great way to identify major innovation players in the ecosystem.
        """)
    else:
        st.warning("Top organizations image not found.")
# ----------------------------
# Semantic Graph Assistant
# ----------------------------

#chatbot部分
# st.header("🧠 Semantic Graph Assistant")

# query = st.text_input("💬 free-form questions like 'Who developed nuclear energy innovations?', 'Which organizations developed the most innovations?':")

# if query:
#     with st.spinner("Retrieving relevant information..."):
#         reply = chat_bot(query)
#     st.success("🧠 Answer:")
#     st.markdown(reply)
#     st.info("🔎 This answer is based on the top 3 semantically similar innovation descriptions retrieved from the knowledge graph.")

# st.divider()


with st.sidebar:
    st.header("💬 Ask the AI Assistant")
    user_input = st.chat_input("Ask something...Who developed nuclear energy innovations?, Which organizations developed the most innovations?:")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_bot(user_input)
                st.markdown(response)
