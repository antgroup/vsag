# VSAG 索引分析工具 (`analyze_index`)

[\[EN\]](README.md)

`analyze_index` 是一个强大的命令行诊断工具，专为 VSAG 库设计。它允许开发者和高级用户深入了解一个已构建的 VSAG 索引文件的内部状态、元数据和性能特征。

该工具支持两种核心分析模式：

1.  **静态属性分析**：无需查询数据，直接解析索引文件，展示其基本信息（如维度、度量类型）和内部结构健康度指标。
2.  **动态查询分析**：当提供查询数据集时，模拟搜索过程并分析索引在查询时的行为，输出召回率、耗时等动态性能指标。

## 主要功能

-   **显示索引元数据**：展示索引的维度、数据类型、距离度量方法等基本信息。
-   **揭示内部结构属性**：输出如图连通分量、重复向量率、量化偏差等深层健康度指标。
-   **分析查询行为**：在给定查询集上评估索引的查询召回率、平均距离、查询耗时等。
-   **支持多种索引**：目前已支持对 HGraph 和 IVF 等主流索引类型进行分析。
-   **灵活的参数配置**：允许在分析时覆盖索引的构建和搜索参数，用于“假设”分析。

## 命令行参数

| 参数                   | 缩写 | 是否必需 | 描述                                                                                                             |
| ---------------------- | ---- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `--index_path`         | `-i` | **是**   | 要分析的 VSAG 索引文件的路径。                                                                                   |
| `--build_parameter`    | `-bp`| 否       | 用于加载索引的构建参数 (JSON 字符串)。如果未提供，将使用索引文件中存储的原始参数。这允许测试不同构建参数对加载的影响。 |
| `--query_path`         | `-qp`| 否       | 用于动态查询分析的查询数据集路径。如果未提供，则只进行静态分析。                                                 |
| `--search_parameter`   | `-sp`| 否       | 查询时使用的搜索参数 (JSON 字符串)。                                                                             |
| `--topk`               | `-k` | 否       | 查询时要返回的 top K 结果数量 (默认: 100)。                                                                      |

## 使用示例

#### 示例 1：仅进行静态分析

只分析索引文件本身，不进行查询。

```bash
./analyze_index --index_path /path/to/my_index.hnsw
```

**输出示例 (部分):**

```
[info] index name: hnsw
[info] index dim: 128
[info] index data type: float
[info] index metric: l2
[info] index param: {
    "M": 32,
    "ef_construction": 400
}
[info] index inner property: {
    "connect_components": 1,
    "deleted_count": 0,
    ...
}
```

#### 示例 2：进行静态和动态查询分析

分析索引文件，并使用一个查询文件来评估其搜索行为。

```bash
./analyze_index \
    --index_path /path/to/my_index.ivf \
    --query_path /path/to/queries.bin \
    --search_parameter '{"nprobe": 16}' \
    --topk 50
```

**输出示例 (部分):**
除了上述静态信息外，还会额外输出：

```
[info] Search Analyze: {
    "avg_distance_query": 0.45,
    "recall_query": 0.98,
    "time_cost_query": 1.2
    ...
}
```

## 输出指标详解

### 1. 索引配置概览

| 属性            | 描述                       |
| --------------- | -------------------------- |
| **Vector Dimension** | 向量维度                   |
| **Data Type**        | 数据类型 (float32/int8...) |
| **Distance Metric**  | 距离度量 (l2/ip...)        |
| **Index Type**       | 索引类型 (HGraph/IVF...)   |

### 2. 索引内部属性 (`GetStats`)

这些指标反映了索引构建完成后的静态健康状况。

```json
{
    "avg_distance_base": "基础数据集中向量间的平均距离 (建索引前)",
    "connect_components": "索引图结构中的连通分量数量",
    "deleted_count": "索引中被标记为删除的向量数量",
    "duplicate_rate": "数据集中重复向量的比例",
    "proximity_recall_neighbor": "索引中邻居邻近度验证的召回率",
    "quantization_bias_ratio": "压缩向量表示中的量化偏差比率",
    "quantization_inversion_count_rate": "量化导致的距离倒置（顺序错误）率",
    "recall_base": "基础数据集的召回率 (用于比较的基准)",
    "total_count": "索引中的向量总数"
}
```

### 3. 搜索分析 (`AnalyzeIndexBySearch`)

这些指标反映了索引在处理查询请求时的动态表现。

```json
{
    "avg_distance_query": "查询向量与检索到的最近邻之间的平均距离",
    "quantization_bias_ratio": "搜索阶段观察到的量化偏差",
    "quantization_inversion_count_rate": "搜索时量化引起的距离倒置率",
    "recall_query": "搜索的召回率 (检索到的真实最近邻的比例)",
    "time_cost_query": "每次查询的平均耗时 (毫秒)"
}
```

## License

该工具根据 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 授权。
