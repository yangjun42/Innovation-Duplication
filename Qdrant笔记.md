

### 🔹Qdrant 中 Collection 的核心编程逻辑解析

#### 0. 初始化 QdrantClient

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

```

`QdrantClient(url=" ... ")` 是 Qdrant 的常用初始化方式，可以用于初始化本地、远程的 Qdrant。例如，连接远程部署的 Qdrant：

```python
client = QdrantClient(
    url="https://your-remote-qdrant.com",
    api_key="your_api_key",  # 若开启身份认证
)
```

| 场景            | 推荐初始化方式                                          |
| ------------- | ------------------------------------------------ |
| 本地开发 + Docker | `QdrantClient("http://localhost:6333")`          |
| 云端部署（含认证）     | `QdrantClient(url="https://...", api_key="...")` |
| 实验/嵌入式工具      | `QdrantClient(path="./data/")`                   |
| 生产搜索（高性能）     | `QdrantClient(grpc_url="...", prefer_grpc=True)` |

注意：gRPC 的高速是靠“HTTP/2 + Protobuf + 流式传输”实现的，但它的代价是调试与集成复杂度上升，不适合所有场景。

#### 1. **什么是 Collection？**

Collection 是 Qdrant 中的顶级数据结构，代表一个向量集合。每个 collection 包含若干 **point（点）**，每个 point 对应一个或多个向量 + payload（附加数据）。这是 Qdrant 的存储与搜索基本单位。

* 所有向量必须有一致的维度（dimensionality）和相同的距离度量方式（如 Cosine、Dot、Euclidean）。
* 支持多个 named vectors（如 image/text）及不同维度和度量方式。

一个 Qdrant Collection 就是一个高效组织的“点集合（Points）”，每个点由：
```text
[向量 Vector(s)] + [payload] + [ID] + [索引结构]
```
组成。


#### 2. **Collection 的创建：编程入口**

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name = "my_collection",
    vectors_config = models.VectorParams(size = 3072, distance = models.Distance.COSINE)
)
```

这里的 `from qdrant_client import QdrantClient, models` 中的 `model` 是 Qdrant SDK 中所有数据结构和 API 请求体的“类型定义集合”，所有操作中涉及的数据结构（如向量参数、别名配置、过滤器、索引策略）就从 models 中调用。

`models.VectorParams(size = 3072, distance = models.Distance.COSINE)` 设置 vector 的维度、距离度量（metric）。

也支持多个 named vectors：

```python
client.create_collection(
    collection_name = "my_collection",
    vectors_config = {
        "image": models.VectorParams(size = 256, distance = models.Distance.DOT),
        "text": models.VectorParams(size = 768, distance= models.Distance.COSINE)
    }
)
```

其他可选参数：

* `on_disk_payload`: 控制是否将 payload 存储在磁盘以节省内存；当 payload 比较大时能节约内存，但也意味着针对 payload 的操作降速；
* `quantization_config`: 配量化压缩配置；
* `hnsw_config`: 控制图索引构建细节；
* `wal_config`: 控制写前日志（WAL）策略。


#### 2.1 补充说明：什么是 Named Vectors？

**Named Vectors** 是指：**一个 point（数据点）中可以存储多个具名的向量**，每个向量都有自己的名称（name）、维度（dimension）和距离度量（metric）。

例如，一个 point 可以包含：

```json
{
  "id": "123",
  "vector": {
    "text": [0.1, 0.2, ...],
    "image": [0.3, 0.7, ...]
  },
  "payload": {
    "title": "Some innovation"
  }
}
```

Qdrant 支持你在同一个 Collection 中，为每个点同时存储 `text` 和 `image` 向量，甚至配置它们使用不同的距离度量。

这是为了解决 **多模态语义表示（multi-modal embedding）和多视角表示（multi-view representation）** 的实际需求：

| 场景            | 原因                                                 |
| ------------- | -------------------------------------------------- |
| 文本 + 图像混合搜索   | 同一条数据既有文本描述，又有图像 embedding                         |
| 文本摘要 vs. 全文搜索 | 一个 `text_short` vector 用于快速摘要检索，`text_full` 用于全文理解 |
| 多语言嵌入         | `zh_embedding` / `en_embedding` 存储不同语言视图           |
| 同一内容多种嵌入模型表示  | 如同时使用 OpenAI 和 BGE 模型的嵌入结果                         |

**Named Vectors 提供灵活、结构化的多向量存储方式，使得 Qdrant 可用于更复杂、更精细的向量搜索任务。**


**如何查询 Named Vector?**

使用 `search` 或 `search_batch` 时，明确指定使用哪个 named vector：P

```python
client.search(
    collection_name="multi_vector_collection",
    query_vector=[0.1, 0.2, ...],
    vector_name="text",  # 👈 关键在此
    limit=5,  # 返回最相似的前 5 个点
)
```

如果你不设置 `vector_name`，但 collection 定义了多个 vector，则会报错。

| 参数                | 说明                         |
| ----------------- | -------------------------- |
| `collection_name` | Collection 名称              |
| `query_vector`    | 查询向量（必须与 collection 的维度匹配） |
| `limit`           | 返回的相似点数量                   |
| `with_payload`    | 是否返回 payload（默认为 `True`）   |
| `with_vectors`    | 是否返回原始向量                   |
| `filter`          | 使用 payload 的过滤条件           |
| `score_threshold` | 最小相似度分数（可选）                |


**如何插入数据到 Named Vectors？**

```python
client.upsert(
    collection_name="multi_vector_collection",
    points=[
        models.PointStruct(
            id=123,
            vector={
                "text": [...],   # 必须与创建时一致
                "image": [...]
            },
            payload={"title": "Multi-modal data"}
        )
    ]
)
```

支持 partial update，例如仅更新 `text` 向量或 `image` 向量。


#### 3. **Collection 的数据结构设计原则**

* 所有 vectors 默认存储在内存中（unless `on_disk=True`）；
* 默认使用 HNSW 索引做 ANN；HNSW:Hierarchical Navigable Small World Graph, 最强的 ANN 实现之一，它构建一个分层图结构，在图中导航（跳跃 + 局部搜索），快速找到近似最近邻; ANN: Approximate Nearest Neighbor，近似最近邻，其目标是在保证较高精度的前提下，比精确最近邻（NN）搜索更快地找到相似向量）；
* 支持稀疏向量与密集向量混合；
* 支持 uint8 类型向量（即将默认的元素类型 float32 转换为 uint8，空间是原来的四分之一，精度略有下降。可适用于大量数据又要求高性能的情况，更多数据可以直接跑在内存中）。

#### 4. **Collection 的核心能力**

| 能力            | 说明                                  |
| ------------- | ----------------------------------- |
| 高性能搜索         | HNSW 索引 + 内存存储 + SIMD               |
| 多向量支持         | 每个 point 可有多个向量                     |
| Payload 存储    | 支持任意 JSON payload，并用于过滤             |
| 向量筛选          | 支持结构化 Filter + 向量 ANN 组合            |
| Collection 更新 | 支持热更新 Index / 配置 / Quantization     |
| 别名机制          | 支持 alias 切换 collection，便于生产环境版本平滑迁移 |

#### 5. **集合管理操作**

* 检查是否存在collection：

```python
client.collection_exists("my_collection")
```

* 删除集合：

```python
client.delete_collection("my_collection")
```

* 更新集合：
运行时热更新 Qdrant Collection 的存储策略（开启磁盘存储）：

```python
client.update_collection(
    collection_name="my_collection",
    vectors_config={
        "": models.VectorParamsDiff(on_disk=True)
    }
)
```

`""` 是 unnamed vector 的占位符，当你创建 collection 时没有使用 named vector（即只有一个默认向量），那么这个默认向量的名字就是空字符串。

`models.VectorParamsDiff(...)` 是一个“部分更新对象”，用于修改已存在的 vector 配置中部分字段（比如 on_disk=True），而不是重新定义整个结构。

这段代码：将默认 unnamed 向量 的存储方式改为 on_disk=True，即把向量从内存迁移到磁盘，以降低内存占用。

#### 6. **关于多租户（Multitenancy）的建议**

* ✅ 推荐：一个大集合，使用 payload 来区分租户（高效、可扩展）；
* ❌ 谨慎：多个 collection，每个租户一个（高资源开销，仅适用于极端隔离需求）。

#### 7. **collection 的 alias**

在 Qdrant 中，`alias` 是指一个逻辑名称，**指向实际的 collection**，可以在不中断服务、不更改用户侧调用逻辑的情况下（无感切换），实现后端 collection 的热切换（热部署）。


生产场景中，当你想“无感切换”向量集合collection，可以使用 alias：

```python
client.update_collection_aliases(
    change_aliases_operations=[
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name="v2_collection", 
                alias_name="active_collection"
            )
        )
    ]
)
```

将 alias `active_collection` 指向真正的 collection `v2_collection`，即：

```
alias: active_collection  -->  collection: v2_collection
```


`client.update_collection_aliases(...)`执行 **批量 alias 更新操作**，支持同时创建、删除、替换多个 alias。

```python
    change_aliases_operations=[
        models.CreateAliasOperation(
```

* 表示你要“创建一个 alias”，即 `CreateAliasOperation`；
* 若 alias 已存在，它将会替换为新的 collection 指向（覆盖行为，原子替换）；

```python
            create_alias=models.CreateAlias(
                collection_name="v2_collection", 
                alias_name="active_collection"
            )
```

* 指定了目标 collection 和别名；
* 后续所有使用 `active_collection` 的查询、写入等请求，都会**自动作用于 `v2_collection`**。


使用场景：

| 场景             | 说明                                        |
| -------------- | ----------------------------------------- |
| ✅ 无感升级 / 滚动发布  | 先创建 `v2_collection`，切换 alias，不用重启或更改客户端配置 |
| ✅ 灰度测试 / AB 实验 | 用不同 alias 同时映射到不同版本 collection，逐步调整比例     |
| ✅ 快速回滚         | 原子切换 alias 回到 `v1_collection`，立刻生效        |
| ✅ 版本解耦         | 用户只知道 alias 名，后端自由管理物理 collection 的生命周期   |


## 另一个 alias 的例子（可批量操作）

```python
client.update_collection_aliases(
    change_aliases_operations=[
        models.DeleteAliasOperation(
            delete_alias=models.DeleteAlias(alias_name="old_active")
        ),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name="new_version",
                alias_name="active_collection"
            )
        )
    ]
)
```

* 原子删除旧别名并创建新别名；
* 全部操作同时生效，无状态不一致问题。

Qdrant 中的 alias 机制就是向量数据库的“DNS 域名系统”——用户只用 alias，后端可以随时无感切换实际 collection。


* 用户调用可以统一写死 alias，如：`search("active_collection")`
* 后端每次部署升级时：
  * 创建 `v2_collection`
  * 预热索引 + 数据加载
  * 切换 alias（原子性）
* 回滚同理，只需切回 `v1_collection`

---

如你希望我帮你：

* 写一个 alias 自动切换函数（检测新旧版本、自动回退）；
* 或封装一个 alias + version 控制系统的最小代码框架；

我都可以继续提供。是否需要？



### 总结与实践建议

* **Qdrant 本质是向量搜索数据库，不是通用文档数据库**，其核心优势是高速 ANN 与高维向量索引优化。
* **Collection 是中心设计单元**，需要设计好结构（是否 named vectors、是否使用 on\_disk、是否用 alias）。
* 若系统数据大、实时要求低，**建议使用 `on_disk=True` 配置减少内存占用**。
* **对于重复数据管理**，Qdrant 不提供自动去重机制，需由应用层控制（如判断 payload 或 vector 是否已存在）。






# Points 的全面笔记


##  什么是 Point？

在 Qdrant 中，**Point 是最核心的实体单元**，它表示一条记录，通常包含以下部分：

```json
{
  "id": 129,                    # 唯一标识符，可为 int 或 UUID
  "vector": [0.1, 0.2, ...],    # 主向量，或多个 named vectors
  "payload": {"color": "red"}   # 可选的结构化元数据
}
```

🔹 每个 Point 属于某个 Collection
🔹 支持按向量相似度搜索（ANN），并**可使用 payload 做结构化筛选**
🔹 操作包括：上传、修改、删除、搜索、滚动、计数等


## Point 的 ID 支持两种格式

| 类型   | 举例                              |
| ---- | ------------------------------- |
| 数值型  | `129`、`1`                       |
| UUID | `"550e8400-e29b-41d4-a716-..."` |

所有 API 都支持两种格式并兼容混用，内部统一处理。



## 向量类型与结构（vector）

每个 Point 可以携带一个或多个 vector，支持：

| 类型            | 说明                        |
| ------------- | ------------------------- |
| Dense Vector  | 密集向量，主流模型输出，如 OpenAI/BGE  |
| Sparse Vector | 稀疏向量，索引-权重对，如 TF-IDF      |
| MultiVector   | 多行向量（如 ColBERT 输出）        |
| Named Vectors | 支持多个具名通道，如 "image"、"text" |

示例：Named Dense Vector

```python
vector={
    "text": [0.1, 0.2, ...],
    "image": [0.3, 0.4, ...]
}
```

## 上传Point（Upsert Points）

多个函数都可上传 Point：

| 方法名                   | 是否批量 | 数据格式          | 使用推荐  | 特点总结                  |
| --------------------- | ---- | ------------- | ----- | --------------------- |
| `upsert()`            | ✅ 支持 | Record/Column | 小规模插入 | 通用插入方法，**立即写入 WAL**   |
| `upload_points()`     | ✅ 支持 | Record 格式     | 大规模上传 | 支持并行上传 + 重试，**更稳定可靠** |
| `upload_collection()` | ✅ 支持 | Column 格式     | 初始化导入 | 批量导入整个 Collection（更快） |

upsert() 是通用型、简单直接的插入方式，upload_points() / upload_collection() 是专为大规模高性能导入设计的批处理方式，二者在底层处理与效率上有所差异。


| 格式类型            | 表示结构        | 优点              | 使用方法                               |
| --------------- | ----------- | --------------- | ---------------------------------- |
| Record-Oriented | 每条记录携带完整字段  | 可读性高，支持多模态、稀疏向量 | `upsert()` / `upload_points()`     |
| Column-Oriented | 各字段拆成列并平行存储 | 更高性能、更适合批量      | `upsert()` / `upload_collection()` |


| 对比项              | `upload_points()`（record） | `upload_collection()`（column） |
| ---------------- | ------------------------- | ----------------------------- |
| 支持 Named Vectors | ✅ 是                       | ❌ 否                           |
| 支持 Sparse Vector | ✅ 是                       | ❌ 否                           |
| 可懒加载（generator）  | ✅ 是                       | ❌ 否                           |
| 支持动态 payload 字段  | ✅ 是                       | ❌ 否（必须结构统一）                   |
| 极端结构化性能          | ❌ 稍慢                      | ✅ 最快                          |



**1. client.upsert(...)**：通用方法
支持全部两种上传方式：

Record-oriented（推荐）：

```python
client.upsert(
    collection_name="my_collection",
    points=[
        models.PointStruct(
            id=1,
            vector=[0.9, 0.1, 0.1],
            payload={"color": "red"}
        )
    ]
)
```

Column-oriented（性能优化）：

```python
client.upsert(
    collection_name="my_collection",
    points=models.Batch(
        ids=[1, 2],
        vectors=[[0.1]*768, [0.2]*768],
        payloads=[{"a": 1}, {"a": 2}]
    )
)
```

* 不提供 ID 时，自动生成 UUID。


**2. client.upload_points(...)**：懒加载 + 自动重试(只能 Record-Oriented)
```python
client.upload_points(
    collection_name="my_collection",
    points=[  # 仅支持 record 格式
        models.PointStruct(...),
        ...
    ],
    parallel=4,
    max_retries=3
)
```

🚀 专为大批量点设计；
✅ 支持并行上传 parallel=n；
✅ 可用于从磁盘分批读取数据上传；
✅ 支持 retry 机制，避免上传失败；
✅ Python SDK ≥1.7.1 推荐方式；
❌ 仅支持 record-oriented（点结构）格式。


**3. client.upload_collection(...)**：专为 Column-Oriented 设计
```python
client.upload_collection(
    collection_name="my_collection",
    ids=[1, 2, 3],
    vectors=[[...], [...], [...]],
    payload=[{...}, {...}, {...}],
    parallel=4
)
```

使用 column-oriented 格式（id 列、vector 列、payload 列）；
🚀 适合一次性导入完整向量集（如从 parquet/csv 读取）；
✅ 支持并行化处理；
✅ 可自动生成 UUID；
❌ 不支持 SparseVector。



## 修改与更新 Points

### 更新 vector（仅替换部分向量）：

```python
client.update_vectors(
    collection_name="...",
    points=[
        models.PointVectors(
            id=1,
            vector={"text": [0.1, 0.2, 0.3]}
        )
    ]
)
```

### 删除 vector：

```python
client.delete_vectors(
    collection_name="...",
    points=[1],
    vectors=["text"]
)
```

### 修改 payload：


每个 Point 可以有一个 **payload（字典）**，用于保存结构化信息，如标签、语言、来源等。

| 操作方法                  | 影响内容         | 是否删除其他字段 | 典型用途        |
| --------------------- | ------------ | -------- | ----------- |
| `set_payload()`       | 设置/更新部分字段    | ❌ 否      | 增量添加字段、轻量更新 |
| `overwrite_payload()` | 替换为新 payload | ✅ 是      | 强制重置，完全替换   |
| `delete_payload()`    | 删除指定字段       | ❌ 否      | 清除部分标签或属性   |
| `clear_payload()`     | 删除所有 payload | ✅ 是      | 完全清空元数据     |

**1. `set_payload()`**：添加或更新 payload 的部分字段

```python
client.set_payload(
    collection_name="my_collection",
    payload={
        "category": "AI",
        "language": "en"
    },
    points=[1, 2, 3]
)
```

* 对指定的 point，添加或更新上述字段；
* **不会清空已有的其他字段**。


原 payload: `{"domain": "science"}`
执行后变成：`{"domain": "science", "category": "AI", "language": "en"}`



**2. `overwrite_payload()`**：**完全覆盖**已有 payload

```python
client.overwrite_payload(
    collection_name="my_collection",
    payload={
        "category": "robotics"
    },
    points=[1]
)
```

* 用新 payload 替换原 payload；
* 原有字段会被全部清除，只保留新字段。


原 payload: `{"domain": "science", "language": "en"}`
执行后变成：`{"category": "robotics"}`（原有的完全被删除）


**3. `delete_payload()`**：删除指定字段

```python
client.delete_payload(
    collection_name="my_collection",
    keys=["category", "language"],
    points=[1, 2]
)
```

* 删除 ID 对应 Point 的指定字段（"category", "language"），保留其余字段内容。


原 payload: `{"category": "AI", "language": "en", "source": "VTT"}`
执行后变成：`{"source": "VTT"}`


**4. `clear_payload()`**：清空整个 payload

```python
client.clear_payload(
    collection_name="my_collection",
    points=[1, 2, 3]
)
```

* 删除所有 payload 字段；


原 payload: `{"category": "AI", "language": "en"}`
执行后变成：`{}`


## 额外补充：Filter + Payload 联合使用

你可以通过 payload 字段进行搜索/筛选，例如：

```python
models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="AI")
        )
    ]
)
```



所以，合理维护 payload 字段对于后续结构化过滤和混合搜索非常重要。



## 删除 Point

### 按 ID 删除：

```python
client.delete(
    collection_name="...",
    points_selector=models.PointIdsList(points=[1, 2, 3])
)
```

### 按条件过滤删除：

```python
client.delete(
    collection_name="...",
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[models.FieldCondition(
                key="color", match=models.MatchValue(value="red")
            )]
        )
    )
)
```


## 7. 检索点（Retrieve Point）

读取特定点的信息。

```python
client.retrieve(
    collection_name="...",
    ids=[1, 2],
    with_vectors=True,
    with_payload=True
)
```

| 参数名            | 说明                              |
| -------------- | ------------------------------- |
| `ids`          | 一组点的 ID（支持 int 或 UUID）          |
| `with_vectors` | 是否包含向量数据（默认为 `True`）            |
| `with_payload` | 是否包含 payload（结构化元数据，默认为 `True`） |

返回一个 list，每一项是：

```python
{
  "id": 1,
  "vector": [...],       # 若 with_vectors=True
  "payload": {...}       # 若 with_payload=True
}
```



## 8. 滚动分页（Scroll）

用于批量获取点的信息。

```python
client.scroll(
    collection_name="...",
    scroll_filter=...,      # 可选：结构化筛选条件
    limit=10,               # 每页最多返回几条
    with_payload=True,
    with_vectors=False
)
```

返回：

```python
points, next_page_offset = client.scroll(...)
```
- points: 当前页的点
- next_page_offset: 下一页起点 ID（或 None 表示结束）


---

## 9. 支持按 Payload 排序（v1.8+）

按 payload 字段（如 timestamp）进行排序滚动：

```python
client.scroll(
    collection_name="...",
    limit=10,
    order_by="timestamp"
)
```

✅ 要求该字段创建索引
⚠️ order\_by 启用后，offset-based 分页不可用

---

## 批量操作（Batch Update）

Qdrant 支持原子批量执行以下操作：

```python
client.batch_update_points(
    collection_name="...",
    update_operations=[
        models.UpsertOperation(...),
        models.UpdateVectorsOperation(...),
        models.DeleteVectorsOperation(...),
        models.SetPayloadOperation(...),
        models.DeleteOperation(...)
    ]
)
```

🔹 推荐用于管道任务、数据同步、消息队列消费等场景。

---

## 设计细节补充

| 特性       | 描述                                    |
| -------- | ------------------------------------- |
| 写前日志 WAL | 所有 point 操作先写入 WAL，确保断电不丢数据           |
| 异步插入     | 可设置 `wait=True` 保证操作完成后返回             |
| 幂等性      | 多次上传相同 ID 的 Point，只保留最后一次（等价于覆盖）      |
| 多向量策略    | 上传部分向量将覆盖已有向量，未指定部分将被置空               |
| 稀疏向量     | 适合 TF-IDF/BM25，使用 indices+values 格式上传 |


## 简写！

Qdrant 的 Python SDK 支持**字典形式的简写（dict-style shorthand）** 来构造 `Filter`、`FieldCondition`、`MatchValue` 等对象，这种方式比 `models.XXX(...)` 更简洁、易读。

比如：
```python
from qdrant_client import models

models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="AI")
        )
    ]
)
```

可以简写为：

```python
{
    "must": [
        {
            "key": "category",
            "match": {
                "value": "AI"
            }
        }
    ]
}
```

### 使用 dict 简写的合法场景

| 用途               | 支持 dict 简写？ | 示例说明                                         |
| ---------------- | ----------- | -------------------------------------------- |
| `Filter`         | ✅           | 如上所示                                         |
| `FieldCondition` | ✅           | `{"key": ..., "match": ...}`                 |
| `MatchValue`     | ✅           | `{"value": ...}`                             |
| `PointStruct`    | ✅           | `{"id": ..., "vector": ..., "payload": ...}` |
| `VectorParams`   | ❌ 不建议       | 必须是 `VectorParams(...)` 实例                   |

---

### 示例：用简写构造 filter 的 `scroll` 查询

```python
uid = "abc123"
result, _ = client.scroll(
    collection_name="my_collection",
    scroll_filter={
        "must": [
            {
                "key": "id",
                "match": {
                    "value": uid
                }
            }
        ]
    },
    limit=1,
    with_payload=True
)
```

---

### ⚠️ 注意事项

* dict 简写本质上是 **兼容 JSON 风格的数据结构**；
* 简写方式在调用 SDK 的 `search`、`scroll`、`upsert`, `recommend`, `retrieve` 等函数时都能正常工作；
* 但如果你使用的是 `models.create_collection()` 这类更结构化的方法，**仍建议用模型对象而非字典**。

---

### ✅ 推荐实践

* 在原型阶段或 notebook 调试时，用简写更方便；
* 在生产代码中，为了类型检查、安全性与补全，推荐用 `models.Filter(...)` 的方式。

---

是否需要我把你已有的上传逻辑中的 filter 全部替换为简写风格？我可以批量帮你替换。



