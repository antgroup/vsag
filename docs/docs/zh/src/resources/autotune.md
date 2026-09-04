# AutoTune 工具（`autotune`）

`autotune` 用于自动完成 VSAG 构建参数和查询参数的枚举评测。AutoTune V1 通过共享的 eval
核心运行真实 VSAG 索引，按约束过滤 trial，并为一个 KNN workload 选择最优
可行候选。

CLI JSON 请求和共用报告结构见 [AutoTune V1 CLI JSON 契约](autotune_api_v1.md)。

V1 支持在 dense float32 数据上构建和调优 HGraph、IVF，也支持对已有 HGraph、IVF 或
Pyramid 索引调 search 参数。CLI 读取 HDF5；C++ API 接收内存 Dataset 和 Index。C++ API
是实验性接口，只在构建目录中提供，不会随 VSAG 库安装。

## 编译

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON
cmake --build build-release --target autotune -j
```

可执行文件位于 `build-release/tools/autotune/autotune`。

仓库提供了[可运行的 SIFT 请求模板][sift-example]。
它默认读取 `/tmp/sift-128-euclidean.hdf5`；数据集位于其他目录时请修改 `data_path`。

[sift-example]: https://github.com/antgroup/vsag/blob/main/tools/autotune/examples/sift_hgraph_autotune_request.json

## C++ API

CLI 和 C++ 入口复用同一套候选生成、trial 规划、评测、约束过滤和结果选择逻辑，区别仅在
输入适配：

- CLI 解析 JSON 并加载 HDF5；
- `TuneIndex` 接收内存 Dataset，同时调优构建参数和查询参数；
- `TuneSearch` 接收已经构建或加载的 `IndexPtr`，只调查询参数。

链接构建目录中的 `vsag::autotune` target 后，可以直接完成“调优、选择、加载”闭环：

```cpp
vsag::autotune::IndexRequest request;
request.base = base;
request.metric_type = vsag::METRIC_L2;
request.workload = {queries, ground_truth, 10, 48};
request.index_spaces = {
    {"hgraph", create_candidate_space, search_candidate_space},
};
request.constraints = {
    {vsag::autotune::Metric::RECALL_AT_K, 0.95},
    {vsag::autotune::Metric::INDEX_MEMORY_MB, 8192.0},
};
request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;

auto result = vsag::autotune::TuneIndex(request);
if (result.has_value() && result->status == vsag::autotune::TuneStatus::SUCCESS) {
    auto neighbors =
        result->index->KnnSearch(query, 10, result->search_parameters).value();
}
```

`base` 和 `queries` 必须是 dense float32 Dataset，base ID 必须存在且唯一。ground truth 的
`NumElements()` 等于 query 数量，`Dim()` 表示 ground-truth K，`Ids()` 是来自 base ID 空间的
展平数组。当 `RECALL_AT_K` 是约束或目标时必须提供 ground truth。Dataset 的 buffer 需保持有效，
直到这个同步调用返回。

每个索引拥有不同的原生参数模式，因此 `IndexSpace::create_parameter_space` 和
`search_parameter_space` 仍使用 JSON 字符串。`index_spaces` 为空时使用内置 HGraph 和
IVF 候选；显式参数中的数组和 `$range` 与 CLI 契约语义相同。

返回值包括已加载的 `index`、`index_name`、完整具体 `create_parameters`、经过验证的
`search_parameters`、metrics、artifact 路径和完整报告。`metrics` 和 `report` 都是
`JsonType` 对象。typed API 在内存中返回报告，绝不会写报告文件。最终推荐 artifact 会
保留，因此可以用 `index_name + create_parameters` 重新创建空索引并反序列化该文件；
未选中的中间 artifact 默认删除。调用方应在不再需要时删除推荐 artifact。

如果没有候选满足全部约束，调用仍正常完成，并返回
`status=TuneStatus::NO_FEASIBLE_CANDIDATE` 和结构化 `best_effort`；此时 `index` 等推荐
字段无效。请求非法或执行失败时仍然返回 error。

完整示例见
[`examples/cpp/326_feature_create_index_with_constraints.cpp`][factory-example]。同时启用工具和
示例即可编译：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON \
  -DENABLE_EXAMPLES=ON
cmake --build build-release --target 326_feature_create_index_with_constraints -j
```

[factory-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/326_feature_create_index_with_constraints.cpp

## 请求

将下面的请求保存为 `request.json`。`dim`、`dtype` 和 `metric_type` 会从数据集读取，用户
不需要重复填写。

```json
{
  "version": 1,
  "data_path": "/tmp/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "index_param": {
          "base_quantization_type": ["fp32", "sq8_uniform"],
          "max_degree": [16, 32],
          "ef_construction": 200
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": {
            "$range": {"start": 40, "stop": 1000}
          }
        }
      }
    }
  ],
  "workload": {
    "top_k": 10,
    "concurrency": 1
  },
  "constraints": {
    "recall_at_k": 0.95,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "latency_avg_ms"
  },
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  }
}
```

运行方式：

```bash
./build-release/tools/autotune/autotune request.json
```

参数叶子可以是标量、数组，或带有 `start`、`stop`、`step` 的 `$range`。HGraph
`ef_search` 还可以省略 `step`；AutoTune 会从 `start` 倍增，找到首个满足 `recall_at_k`
约束的区间，再在该区间内二分查找最小通过值。该策略假设固定其他配置时 recall 不会随
`ef_search` 增大而降低。每个实际评测点都使用全部 query。用户显式提供的值保持不变；
对于缺失的调优字段，V1 会补充一小组 HGraph 或 IVF 内置候选。省略 `indexes` 时会同时
评测这两种索引。

具体 create 参数相同的候选只构建一次；每组具体 search 参数仍然是独立 trial。自适应
`ef_search` 范围只记录实际评测过的具体参数点。

## 已有索引

设置 `index_path` 并且只提供一个索引规格，可以跳过构建、只调查询参数。其中
`create_params` 必须是序列化该索引时使用的具体参数。该模式不会展开 `create_params`，
因此索引原生的数组字段会保持不变。

程序调用方使用独立的 `SearchRequest` 和 `TuneSearch`：

```cpp
vsag::autotune::SearchRequest request;
request.index = existing_index;
request.workload = {queries, ground_truth, 10, 48};
request.parameter_space = R"({"hgraph":{"ef_search":[40,80,120]}})";
request.constraints = {{vsag::autotune::Metric::RECALL_AT_K, 0.95}};
request.objective = vsag::autotune::Metric::LATENCY_AVG_MS;

auto result = vsag::autotune::TuneSearch(request);
if (result.has_value() && result->status == vsag::autotune::TuneStatus::SUCCESS) {
    auto neighbors = existing_index->KnnSearch(query, 10, result->parameters).value();
}
```

该接口避免序列化和文件 I/O，并让全部 search trial 复用调用方的索引。AutoTune 从
`IndexPtr` 推导索引类型和元素数量，因此 search tuning 只需要 query，不需要 base dataset
或 metric type。使用 recall 时还需要 ground truth：每条 query 的 `recall_at_k` 定义为
返回结果和 ground truth 前 `top_k` 个 ID 的交集大小除以 `top_k`，最终指标是所有 query
的平均值。recall 既不是约束也不是目标时，ground truth 可省略。

typed HGraph 请求可以为每条 query 提供一个 `FilterPtr`，或者一个表示排除集合的
`BitsetPtr`，以复现带过滤的 workload：

```cpp
request.workload = {queries, filtered_ground_truth, 10, 48};
request.workload.query_filters = query_filters;
// 或：request.workload.query_invalid_bitsets = query_invalid_bitsets;
```

两种 vector 互斥。vector 为空表示未使用该种过滤方式；否则数量必须与 query 数量相同，
其中空指针表示该 query 不做过滤。`query_invalid_bitsets` 中置位的 bit 表示排除对应的索引
外部 ID。调优期间不能修改 bitset，并且其中应只包含待评测索引中的 ID。

每个非空 `FilterPtr` 会跨候选重复使用，并且可能被并发调用，因此必须确定、可重复使用且
线程安全。`ValidRatio()` 必须能代表该 query 的实际选择率，因为 HGraph 会用它选择搜索路径。
本版本使用 `CheckValid(int64_t)` 并拒绝 `use_extra_info_filter=true`；`CheckValid` 收到的 ID
是索引的外部 label。

每条 query 必须至少允许 `top_k` 个 ID 通过，ground truth 必须使用对应 filter 或 bitset
生成；推荐参数应继续用于具有同类代表性过滤分布的 workload。`TuneIndex` 同样支持这种
HGraph workload；存在任一种过滤输入且省略 `index_spaces` 时只生成 HGraph 候选。filter 和
bitset 都不会保存到索引中，调用方在最终的每次 `KnnSearch` 中仍需传入对应值。调用方提供的
过滤输入只支持 typed API，不会写入 CLI 请求或报告。本版本不支持 IVF 和 Pyramid 的
filtered tuning。

CLI 的 `index_path` 仍是离线适配：它先使用具体 create 参数创建并反序列化 Index，再进入
同一条 search-only 流程。

Pyramid 通过这个 search-only 模式接入。HDF5 CLI 适配器不提供 query path，因此只能评测
Pyramid 原生的默认/root 搜索。typed 请求直接从 `workload.queries->GetPaths()` 取得 path：
提供 path 时，每条 query 按自己的 path 评测；不提供时沿用 root 搜索语义。V1 只支持默认
未命名 hierarchy。若不同 path 需要不同 `ef_search`，应为每个代表性 path workload 分别
发起一次 typed AutoTune 请求，并提供与该 path 对应的 ground truth。V1 不把多个 path 的
推荐聚合成一条结果。

`FilterPtr` 完整示例见
[`examples/cpp/327_feature_autotune_existing_index.cpp`][existing-index-example]；直接传入每条
query 的排除 bitset 见
[`examples/cpp/330_feature_autotune_existing_index_bitset.cpp`][bitset-example]。
Pyramid path 调优示例见
[`examples/cpp/328_feature_autotune_existing_pyramid.cpp`][pyramid-example]。
它对同一个 Pyramid 索引中的 512-vector 和 4096-vector 叶子子图使用相同 recall 目标分别
调优，用于展示不同 path 可能需要不同的 `ef_search`。

[existing-index-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/327_feature_autotune_existing_index.cpp
[bitset-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/330_feature_autotune_existing_index_bitset.cpp
[pyramid-example]: https://github.com/antgroup/vsag/blob/main/examples/cpp/328_feature_autotune_existing_pyramid.cpp

## 指标

以下指标可以作为约束或优化目标：

| 指标 | 约束比较方式 |
| --- | --- |
| `recall_at_k`、`qps` | 不低于请求值 |
| `latency_avg_ms`、`latency_p99_ms` | 不高于请求值 |
| `index_memory_mb`、`index_size_mb` | 不高于请求值 |
| `build_seconds`、`search_seconds`、`build_and_search_seconds` | 不高于请求值 |

已有索引模式不提供 `build_seconds`、`index_size_mb` 和 `build_and_search_seconds`。
`index_memory_mb` 可以作为约束，但不能作为目标，因为同一个已有索引的所有 search 候选
内存相同，无法据此排序。目标方向由指标本身决定，因此不需要 `direction` 字段。

AutoTune 使用独立的 latency/QPS pass 和 recall/statistics pass，因此同一条逻辑 query 在
每个 trial 中会执行多次。`search_seconds` 包含两个内存 pass 和指标采集，但不包含索引
反序列化。候选按固定顺序运行，不做 warm-up 或随机排序。如果 latency 或 QPS 决定最终
推荐，应在有代表性且受控的机器上测量，并通过重复实验确认。

## 阅读输出

标准输出是便于阅读和复制的短 summary，字段顺序如下：

- `recommendation`：最终选中的具体参数、指标和 trial 证据。build-and-search 结果还包含
  create 参数和 artifact/build 证据；search-only 结果不包含这些字段。
- `best_effort`：只有在没有 trial 满足全部约束时出现。
- `failure`：请求失败或所有评测失败时出现。
- `report_path`：完整可复现报告的位置。

完整报告还包含每个 build 和 trial、约束 violation、补齐后的请求，以及分阶段的结构化
失败。search-only 模式的 `builds` 为空，trial 不包含 artifact 或 build 字段。CLI 和 typed
`TuneIndex` 设置 `keep_intermediate=false` 时都只保留最终推荐索引，并删除未选中的
artifact；设置为 `true` 时保留所有生成索引。

请求通过初始校验后，CLI 和 `RunAutoTune` 会持久化完整报告：显式 `output.result_path`
会覆盖输出路径，否则在 workspace 下生成默认路径。早期校验和请求文件错误不会写报告。
typed `TuneIndex` 和 `TuneSearch` 绝不会写报告文件，而是通过返回值提供完整的 `JsonType`
报告。

## V1 边界

V1 评测一个 KNN workload；除已支持的 HGraph `ef_search` 自适应搜索外，其他候选仍完整遍历。
typed API 支持 HGraph 的逐 query ID filter；CLI 和其他索引类型暂不支持 filtered workload。
V1 仍不支持范围查询 workload、query sampling、跨请求 build cache，以及基于模型的候选生成。
