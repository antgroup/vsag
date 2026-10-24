# LangChain 集成

`pyvsag.langchain.VSAGVectorStore` 基于 `pyvsag.Index` 实现 LangChain 的
`VectorStore` 接口，仅支持稠密向量和同步操作。

## 安装

```bash
pip install "pyvsag[langchain]"
```

该 extra 安装 `langchain-core>=0.3.77,<2` 和 `numpy`；`pip install pyvsag`
不会安装 LangChain。未安装 extra 时导入 `pyvsag.langchain` 会抛出
`ImportError`，并提示上述安装命令。

| Python      | `langchain-core` 0.3.x | `langchain-core` 1.x |
|-------------|------------------------|----------------------|
| 3.9         | 支持                   | 不支持               |
| 3.10 – 3.14 | 支持                   | 支持                 |


## 快速开始

```python
from langchain_core.embeddings import DeterministicFakeEmbedding
from pyvsag.langchain import VSAGVectorStore

store = VSAGVectorStore.from_texts(
    ["HGraph 是图索引", "SQ8 压缩稠密向量"],
    DeterministicFakeEmbedding(size=64),  # 仅演示接口；随机向量无语义
    metadatas=[{"topic": "index"}, {"topic": "quantization"}],
    ids=["doc-1", "doc-2"],
)

store.similarity_search("图索引", k=1)
store.as_retriever(search_kwargs={"k": 1}).invoke("向量压缩")
```

标准 `VectorStore` 方法 `from_texts`、`from_documents`、`add_texts`、
`add_documents`、`get_by_ids`、`delete`、`similarity_search`、
`similarity_search_with_score`、`similarity_search_by_vector`、
`similarity_search_with_score_by_vector` 和 `as_retriever` 均遵循 LangChain
接口约定。

## 配置

```python
store = VSAGVectorStore(
    embedding,                                     # LangChain Embeddings
    index_type="hgraph",                           # hgraph、ivf、brute_force 等
    metric_type="l2",                              # l2、ip、cosine
    index_param={"max_degree": 32},                # 用于构造 pyvsag.Index
    search_params={"hgraph": {"ef_search": 100}},  # 传给 Index.knn_search()
)
```

- `index_param` 和 `search_params` 接受 dict 或 JSON 字符串。
- 索引在第一次非空 `add_texts()` 时创建，维度取自 embedding。
- `hgraph` 和 `brute_force` 有默认搜索参数；其他索引类型必须显式传入
  `search_params`，例如 `{"ivf": {"scan_buckets_count": 16}}`；`lazy_hgraph`
  还需传入嵌套的 `index_param`。
- `hgraph` 的 `ef_search` 可在每次搜索时覆盖，默认值为 `max(k, 100)`。
- `hgraph` 以 `support_remove=true` 构建，`delete()` 和 upsert 依赖该选项。

## ID 映射

LangChain 字符串 ID 映射为 store 内部的 `int64` ID，且永不复用。未提供 ID 时用
`uuid4()` 生成；同一批次内重复 ID 会抛出 `ValueError`。写入已存在的 ID 为
upsert：先加入新向量，再删除旧向量。`delete()` 忽略未知 ID，`delete(ids=None)`
删除全部文档。

## 分数语义

`similarity_search_with_score()` 返回 VSAG 原始距离，按升序排列：

| `metric_type` | 分数                       |
|---------------|----------------------------|
| `l2`          | 平方 L2 距离               |
| `ip`          | `1 - dot(query, document)` |
| `cosine`      | `1 - cosine_similarity`    |

分数未归一化，不能跨 metric 比较。详见[度量语义](../resources/metric_semantics.md)。

## 已知限制

- 仅支持稠密 `float32` embedding，不支持稀疏向量（SINDI）和 FP16/BF16 输入。
- 不支持 metadata 过滤、MMR、范围搜索和原生异步：`filter` 和 `where_document`
  抛出 `NotImplementedError`，异步方法使用 LangChain 的线程池回退，其他未知
  关键字参数忽略。
- 不支持 relevance score 转换；`similarity_search_with_relevance_scores()` 和
  `search_type="similarity_score_threshold"` 使用 LangChain 默认的
  `NotImplementedError`。
- 文档正文、metadata 和 ID 映射保存在内存中；`store.index.save(path)` 只持久化
  VSAG 索引。
- 删除为 tombstone 机制，适配器不做 compaction 或重建。
- 写操作在适配器内部串行化。

## 示例

[`examples/python/111_langchain_vector_store.py`](https://github.com/antgroup/vsag/blob/main/examples/python/111_langchain_vector_store.py)
