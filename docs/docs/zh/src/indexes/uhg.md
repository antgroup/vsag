# UHG 混合检索

UHG（Unified Hybrid Graph）在一张固定图上使用查询级权重检索稠密、稀疏向量对。通过 `Factory::CreateIndex("uhg", parameters)` 创建索引，在同一个 Dataset 中同时设置 `Float32Vectors` 和 `SparseVectors`，两种表示共享外部文档 ID。

返回距离为 `alpha * (1 - dense_dot) + (1 - alpha) * (1 - sparse_dot)`，越小越相似。若希望稠密分量等价于余弦距离，请在写入前归一化稠密向量；索引不自动归一化，也不校准两种分数的尺度。

## 建图

```json
{
  "dtype": "float32",
  "metric_type": "ip",
  "dim": 1024,
  "index_param": {
    "max_degree": 64,
    "neighbors_per_alpha": 32,
    "construction_candidates": 200,
    "alpha_intervals": 10,
    "hgraph": {"max_degree": 64, "ef_construction": 200},
    "sindi": {"term_id_limit": 500000, "window_size": 10000}
  }
}
```

HGraph 和 SINDI 离线生成候选。每个文档分别从两种索引取最多 `construction_candidates + 1` 个候选（包含中心点余量），合并去重后计算精确分量距离。在 `a / alpha_intervals`（`a=0,...,alpha_intervals`）的每个权重下选择 `neighbors_per_alpha` 个邻居，按入选频次保留最多 `max_degree` 条出边。频次相同按外部 ID 排序。查询改变权重时无需重建图。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `max_degree` | 64 | UHG 最大出度 |
| `neighbors_per_alpha` | 32 | 每个采样权重保留的邻居数 |
| `construction_candidates` | 200 | 单路候选数，不含中心点余量；不得小于 `neighbors_per_alpha` |
| `alpha_intervals` | 10 | 采样区间数；10 对应 0、0.1、…、1；取值 1–1000 |
| `hgraph` | `{}` | 稠密引导索引配置：`max_degree`、`ef_construction`、`build_thread_count` |
| `sindi` | `{}` | 稀疏引导索引配置：`term_id_limit`、`window_size`、`remap_term_ids` |

HGraph 和 SINDI 保留引导召回职责。UHG 另存 FP32 稠密向量及稀疏正排向量，用于批量距离计算，增加存储以避免逐节点公共接口调用和倒排点距扫描；同时保存邻接表及稀疏范数。为保证精确分量距离，HGraph 固定 `base_quantization_type="fp32"`，SINDI 固定 `use_quantization=false`、`doc_prune_ratio=0`，两者固定 `use_reorder=false`；冲突配置会报错。

## 查询

```cpp
auto index = vsag::Factory::CreateIndex("uhg", build_parameters).value();
auto base = vsag::Dataset::Make()
                ->Owner(false)
                ->NumElements(count)
                ->Dim(1024)
                ->Ids(ids)
                ->Float32Vectors(dense_vectors)
                ->SparseVectors(sparse_vectors);
auto built = index->Build(base);
if (!built.has_value()) {
    throw std::runtime_error(built.error().message);
}
auto query = vsag::Dataset::Make()
                 ->Owner(false)
                 ->NumElements(1)
                 ->Dim(1024)
                 ->Float32Vectors(query_dense)
                 ->SparseVectors(&query_sparse);
auto result = index->KnnSearch(query, 10, R"({
    "uhg": {
        "alpha": 0.3,
        "ef_search": 300,
        "entry_points": 8,
        "guide_ef_search": 100,
        "use_pruning": true,
        "pruning_scale": 1.0
    }
})");
```

`alpha <= 0.5` 时由 SINDI 提供入口，`alpha > 0.5` 时由 HGraph 提供入口，然后按查询指定的混合距离执行图上的束搜索。空稀疏查询改用稠密入口；引导结果为空时从首个顶点开始。图遍历可以发现引导结果之外的文档。

| 查询参数 | 默认值 | 含义 |
| --- | --- | --- |
| `alpha` | 0.5 | 稠密权重，范围 [0,1]，每次查询独立指定 |
| `ef_search` | 100 | 图遍历束宽，不得小于 k |
| `entry_points` | 8 | 引导结果中的入口数，不超过束宽 |
| `guide_ef_search` | 100 | HGraph 束宽或 SINDI 候选预算，至少等于入口数 |
| `use_pruning` | true | 是否启用稠密优先的稀疏距离剪枝 |
| `pruning_scale` | 1.0 | γ，范围 [0,1]；1 使用 Cauchy–Schwarz 下界，更小的值属于启发式剪枝，可能降低召回 |

剪枝估计为 `alpha * dense_distance + (1-alpha) * (1-gamma*query_sparse_norm*document_sparse_norm)`。束已满且估计超过当前最差距离（加上 FP32 舍入余量）时，跳过精确稀疏距离计算。alpha=0 时跳过稠密计算，alpha=1 时跳过稀疏计算；`use_pruning=false` 关闭估计剪枝。

## 支持范围与限制

支持空索引 Build、单查询 KNN、外部 ID 过滤、并发读取、ID 导出、内存统计及 BinarySet/流/ReaderSet 序列化。加载需使用相同建图参数和维度。构建或加载完全成功后才发布索引状态。不支持 Add、Remove、UpdateId/UpdateVector、范围检索、量化向量、extra info 或属性过滤。

基库稀疏向量必须非空，term ID 严格升序且不重复，值为有限非负 FP32；查询稀疏向量允许为空。稠密值必须有限。基库 ID 不得重复或等于 -1。过滤节点仍可作为遍历桥接点；过滤非常严格时可能返回少于 k 个结果，增大 `ef_search` 可改善覆盖。

召回受离线候选池、图出度、查询束宽和剪枝比例影响，实现不使用全库精确扫描兜底。论文中的性能提升不代表本实现的性能保证；请使用业务向量及对应动态权重的精确混合 ground truth 评估 Recall/QPS。

## MS MARCO 对比实验

独立程序 `tools/uhg_benchmark` 对比生产 UHG 查询与 HGraph + SINDI 等候选预算检索。两路分别取 100、150、200、300、500 个候选，取并集、去重，对每条候选重新计算两项完整距离，按混合距离重排取 top-100。计时包含检索、去重和完整重排。UHG 束宽为 100、200、300、500、1000，入口数与束宽相同，剪枝系数为 0.3。两种方法在线稀疏查询均不剪枝。

```sh
cmake -S tools/uhg_benchmark -B build-uhg-benchmark \
  -DVSAG_BUILD="$PWD/build-release" -DVSAG_DEPS=/tmp/vsag-deps
cmake --build build-uhg-benchmark --parallel 8
OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 \
  build-uhg-benchmark/uhg_benchmark \
  /data/datasets/msmarco-1024-8m-ip.hdf5 /data/results/uhg-msmarco-full 0 0 64 3
```

先构建 release 版 VSAG。实验工具还需要 HDF5、BLAS、OpenMP 开发库；`VSAG_DEPS` 指向解压后的 VSAG 依赖源码。末尾参数依次为基库条数上限、查询条数上限（0 表示全部）、离线构建线程数和计时重复次数。查询在一个线程上顺序执行。固定权重为 0.3、0.5、0.7；混合场景将这三种权重近似均分给查询，用种子 42 打乱。

精确答案穷举整个选定基库，采用 FP32 稠密、稀疏内积和双精度加权分数。Recall@100 衡量精确混合邻居的覆盖率，不是 MS MARCO 相关性标注指标。索引与答案缓存在输出目录；更改数据或参数时请使用新目录。

大数据集使用相同的 alpha 采样与邻居频次规则并行构图，然后经生产 UHG 反序列化接口加载。复用两路基线的引导索引，离线稠密束宽为 200、稀疏查询剪枝比例为 0.9；该离线构建过程独立于公共串行 `UHG::Build` 接口。报告分别记录引导索引构建时间、图构建时间和查询耗时。`metadata.json` 保存配置，`results.json` 保存 Recall@100、QPS、延迟分位数和平均并集大小。

## 查询实现与兼容性

查询为每次搜索创建一次稠密、稀疏计算器，复用邻居批次的缓冲区和池化访问标记；先批量计算稠密距离，再对通过剪枝的节点批量计算稀疏距离。无过滤查询直接复用搜索束作为输出，过滤查询保留独立结果堆，使被过滤节点仍可作为桥接节点。批次内使用固定剪枝阈值，启发式剪枝的召回率可能与逐节点实现略有不同。

新索引采用 v2 序列化格式保存正排向量；仍可读取 v1 索引，v1 使用公共距离接口回退路径。实验工具可利用原始数据升级缓存，保留原有图和引导索引。未复用 HGraph 引导搜索内部的中间距离，因此无需修改 HGraph/SINDI 代码。
