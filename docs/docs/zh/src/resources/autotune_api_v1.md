# AutoTune V1 CLI JSON 输入输出契约

本文档定义 `autotune` 命令行工具使用的离线 JSON/HDF5 适配契约。实验性的 C++ API 只在
构建目录中提供，接收 typed `IndexRequest` 和 `SearchRequest`，不会随 VSAG 库安装。所有
入口完成输入转换后使用同一个规范化请求和调优引擎。AutoTune V1 支持 dense float32 数据、
HGraph/IVF 的构建和查询调优，以及一个使用全量 query 的 KNN workload。

请求、索引规格、workload、objective、`tuning_config` 和 `output` 中的未知字段会被拒绝。
`create_params` 和 `search_params` 内部的原生参数会传给具体 VSAG 索引，在创建或查询时由
索引校验。

## 请求对象

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `version` | integer | 是 | 必须为 `1`。 |
| `data_path` | string | 是 | 可读的 HDF5 评测数据集。 |
| `index_path` | string | 否 | 用于 search-only 调优的已有序列化索引。 |
| `indexes` | array | 否 | 索引规格；普通模式默认使用 HGraph 和 IVF。 |
| `workload` | object | 是 | 要优化的 KNN workload。 |
| `constraints` | object | 是 | 非空的“指标到阈值”映射。 |
| `objective` | object | 是 | 对可行候选排序的目标指标。 |
| `tuning_config` | object | 否 | workspace、产物保留和 trial 上限。 |
| `output` | object | 否 | 报告路径和原始 eval 输出开关。 |

### 数据集规则

`data_path` 必须指向 VSAG eval 工具可读取的普通 HDF5 文件。V1 要求：

- dense vector；
- base 和 query 均为 float32；
- 至少包含一个 base、一个 query 和一个 ground-truth neighbor；
- 使用 recall 指标时，`workload.top_k` 不超过 ground truth 的宽度；
- distance 属性能够映射为 `l2`、`ip` 或 `cosine`。

AutoTune 从数据集读取 `dim`、`dtype` 和 `metric_type`，并注入每组具体
`create_params`。请求可以重复提供这些字段，但值必须与数据集完全一致。

## 索引规格

每个 `indexes[]` 元素的形态如下：

```json
{
  "name": "hgraph",
  "create_params": {
    "index_param": {
      "max_degree": {"$choices": [16, 32]}
    }
  },
  "search_params": {
    "hgraph": {
      "ef_search": {"$range": {"start": 40, "stop": 1000}}
    }
  }
}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `name` | string | 是 | `hgraph`、`ivf`，或 search-only 的 `pyramid`；不区分大小写。 |
| `create_params` | object | 否 | VSAG 创建参数或候选表达式。 |
| `search_params` | object | 否 | 查询参数，key 必须是同一个索引名。 |

普通 build-and-search 模式省略 `indexes` 时，AutoTune 会同时评测 HGraph 和 IVF。设置
`index_path` 时，必须提供 `indexes`，并且只能包含一个元素。
Pyramid 只接受已有的内存索引或序列化索引。

### 候选表达式

`create_params` 和 `search_params` 中的每个字段都可以写成：

- 具体 JSON 值（包括数组）：固定为该完整值；
- `$choices` 对象：其中包含非空数组，用于提供离散候选值；
- `$range`：必须且只能包含 `start`、`stop` 和非零 `step`，范围包含端点。

示例：

```json
{
  "fixed": 32,
  "literal_array": [0, 1],
  "choices": {"$choices": [16, 32, 48]},
  "array_choices": {"$choices": [[], [0], [0, 1]]},
  "integer_range": {"$range": {"start": 40, "stop": 120, "step": 40}},
  "float_range": {"$range": {"start": 0.1, "stop": 0.3, "step": 0.1}}
}
```

`$choices` 中的每一项都是一个完整的具体值，其中不再解释嵌套的候选表达式。重复项会被
去除；`$choices` 不能为空，也不能和其他 key 混用。

HGraph `ef_search` 还支持一种特殊写法：

```json
{"$range": {"start": 40, "stop": 1000}}
```

上下界必须是正整数并满足 `start <= stop`。这种写法要求存在 `recall_at_k` 约束，且目标为
`latency_avg_ms`、`latency_p99_ms`、`qps`、`search_seconds` 或
`build_and_search_seconds`；不能同时设置 `timeout_ms` 或 `hops_limit`。对于其他参数的每个
固定组合，AutoTune 从 `start` 开始倍增 `ef_search`（不超过 `stop`），直到实测 recall 满足
约束；随后在最后一个失败/通过区间内二分，查找最小通过值。分支方向只由 recall 决定；
latency 和 QPS 仍作为实测指标参与最终约束过滤和结果选择。该策略假设其他参数和 workload
固定时，recall 不会随 `ef_search` 增大而降低。

每个探测点都会评测完整 query workload。如果某个评测点执行失败或没有 recall，AutoTune
会停止该自适应范围，因为此时无法安全选择下一个区间。

V1 会把参数对象中的所有 `$choices` 和带 `step` 的范围解释成候选集合，计算完整笛卡尔积
并去除完全相同的候选。裸 JSON 数组会保持为索引原生参数值。其他参数不接受省略 `step`
的范围。

计划的最坏评测次数超过 `tuning_config.max_trials` 时，请求失败。普通具体候选占一个
trial；自适应 `ef_search` 在 `start == stop` 时占一个。否则，planner 会为每个可能的首次
通过探测点计算“到达该点的探测次数 + 前一区间的二分次数”，并取其中最大值。例如
`40..1000` 最坏预留 15 个 trial。

### 缺失字段的内置候选

用户显式字段始终优先。只有字段缺失时才补充以下候选：

| 索引 | 缺失字段 | V1 候选 |
| --- | --- | --- |
| HGraph | `base_quantization_type` | `fp32`、`sq8_uniform` |
| HGraph | `max_degree` | `16`、`32` |
| HGraph | `ef_construction` | `100`、`200` |
| HGraph | `ef_search` | 根据 `top_k` 生成三个值，下限为 `40`、`80`、`120` |
| Pyramid，search-only | `ef_search` | 根据 `top_k` 生成三个值，下限为 `40`、`80`、`120` |
| IVF | `base_quantization_type` | `fp32`、`sq8_uniform` |
| IVF | `buckets_count` | 不超过 base 数量的 `1024`、`2048` |
| IVF，build-and-search | `scan_buckets_count` | 根据具体 `buckets_count` 生成至多四个值 |
| IVF，search-only | `scan_buckets_count` | `1`、`4`、`16`、`64` |

HGraph 和 Pyramid 查询候选具体为 `max(40, top_k)`、`max(80, 2 * top_k)` 和
`max(120, 4 * top_k)`，重复值会删除。build-and-search 模式的 IVF 查询候选不会超过具体
bucket 数量。AutoTune 无法推断已有 IVF 的 bucket 数量，因此 search-only 模式使用上面的
保守通用值；被已加载索引拒绝的值会记录为 failed trial，调用方可以显式提供
`scan_buckets_count` 候选以避免这些 trial。

规范化索引名和具体 `create_params` 相同的候选属于同一个 build group。AutoTune 对该组
只构建一次、序列化一次作为证据，再使用同一个内存索引实例执行所有关联的 search 候选。

### 已有索引模式

设置 `index_path` 后：

- 路径必须指向可读普通文件；
- 必须且只能提供一个索引规格；
- 必须提供 `create_params.index_param`；
- `create_params` 整体视为一份具体的索引原生配置，不再做候选展开；
- AutoTune 不补充 create 候选，也不会执行 build；
- 缺失的 search 字段仍可使用内置查询候选；
- `build_seconds`、`index_size_mb` 和 `build_and_search_seconds` 不能作为约束或目标；
- `index_memory_mb` 可以作为约束，但不能作为目标；
- 不会删除调用方提供的索引。

V1 反序列化前需要使用具体创建参数实例化正确的 VSAG 索引，目前不会从序列化文件推导
这些参数。

对于 typed `SearchRequest`，索引已经完成实例化和加载。AutoTune 从 `IndexPtr` 推导索引
类型和元素数量；请求只提供 index、workload、search `parameter_space`、constraints、
objective 和 config，不需要 create 参数、base dataset 或 metric type。

HDF5 适配器不提供 Pyramid query path，因此 CLI 只能评测 Pyramid 原生的默认/root 查询。
按 path 调优 Pyramid 需要使用 typed `SearchRequest`，并通过 query Dataset 的
`Dataset::Paths()` 提供 path。V1 只转发默认的未命名 hierarchy，不支持 Pyramid named
hierarchy。

## Workload

```json
{
  "workload": {
    "top_k": 10,
    "concurrency": 48
  }
}
```

| 字段 | 类型 | 必填 | 默认值 | 校验 |
| --- | --- | --- | --- | --- |
| `top_k` | positive integer | 是 | — | 不超过 base 数量；使用 recall 指标时不超过 ground truth 宽度；并能用原生 `int` 表示。 |
| `concurrency` | positive integer | 否 | `1` | 最大为 `200`。 |

V1 始终评测 HDF5 文件中的全量 query，并且只支持 KNN。benchmark 机器、系统负载和运行
环境都属于 workload 的一部分；延迟和 QPS 不能直接移植到不同机器规格或负载条件。
对于 typed `SearchRequest`，`top_k` 会和 `IndexPtr::GetNumElements()` 比较，不依赖 base
dataset。只有 recall 是约束或目标时才必须提供 ground truth。

## 约束和优化目标

`constraints` 直接把指标名映射到阈值；`objective.metric` 指定一个目标指标。比较方式和
目标方向由指标决定，用户不能覆盖。

```json
{
  "constraints": {
    "recall_at_k": 0.95,
    "latency_p99_ms": 2.0,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "latency_avg_ms"
  }
}
```

| 指标 | 约束 | 目标方向 | 已有索引用法 |
| --- | --- | --- | --- |
| `recall_at_k` | actual ≥ threshold | 最大化 | 是 |
| `qps` | actual ≥ threshold | 最大化 | 是 |
| `latency_avg_ms` | actual ≤ threshold | 最小化 | 是 |
| `latency_p99_ms` | actual ≤ threshold | 最小化 | 是 |
| `index_memory_mb` | actual ≤ threshold | 最小化 | 仅约束 |
| `index_size_mb` | actual ≤ threshold | 最小化 | 否 |
| `build_seconds` | actual ≤ threshold | 最小化 | 否 |
| `search_seconds` | actual ≤ threshold | 最小化 | 是 |
| `build_and_search_seconds` | actual ≤ threshold | 最小化 | 否 |

阈值必须是有限非负数，`recall_at_k` 还必须不大于 `1.0`。

V1 指标口径：

- 每条 query 的 `recall_at_k` 是返回结果和 ground truth 前 `top_k` 个 ID 的交集大小除以
  `top_k`，最终指标是所有 query 的平均值。它需要 ground truth，但不需要 base vector 或
  metric type。
- `build_seconds` 是 eval 工具报告的索引 `Build` 操作耗时。
- 它不包含索引序列化、数据集加载和候选编排。
- `search_seconds` 是完整内存 search eval trial 的墙钟时间，不包含索引反序列化，但包含
  所有 search pass 和指标采集；线上查询性能目标应使用 latency 或 QPS。
- `build_and_search_seconds` 是一次 build-and-search trial 中前两项之和。
- `index_memory_mb` 来自已加载索引大于零的 `GetMemoryUsage()` 结果。索引报告零时，
  AutoTune 会把该指标视为不可用，而不是零内存；相关约束会记录 missing-metric violation。
- `index_size_mb` 是生成的序列化索引文件大小，在 search-only 模式不可用。
- AutoTune 使用独立的 latency/QPS pass 和 recall/statistics pass。latency 用
  `steady_clock` 包围每次 `KnnSearch`；QPS 是成功查询数除以并发性能 pass 的墙钟时间。
  因此，同一条逻辑 query 在每个 trial 中会执行多次。

`search_seconds` 仍表示评测成本，因为它包含两个 pass 和 monitor 处理；它不是一次线上
逻辑 query batch 的执行时间。候选会按固定顺序在复用的索引上运行，不做 warm-up 或随机
排序，因此 cache 状态和无关系统负载会影响性能指标。如果 latency 或 QPS 决定结果，应在
有代表性且受控的环境中运行，并通过重复实验确认。

## Tuning 和输出配置

```json
{
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  },
  "output": {
    "result_path": "/tmp/vsag_autotune/report.json",
    "include_raw_eval": false
  }
}
```

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `tuning_config.workspace_path` | `/tmp/vsag_autotune` | run 产物和 CLI 默认报告目录。 |
| `tuning_config.keep_intermediate` | `false` | 是否保留所有生成索引。 |
| `tuning_config.max_trials` | `1000` | 计划最坏 trial 数上限；硬上限为 `100000`。 |
| `output.result_path` | `<workspace>/run-<id>.json` | CLI 完整报告路径。 |
| `output.include_raw_eval` | `false` | 在 build/trial 中包含原生 eval JSON。 |

`output.result_path` 不能与 `data_path` 或 `index_path` 指向同一文件。通过初始校验后，
离线 JSON/CLI 入口会写完整报告：显式 `output.result_path` 会覆盖输出位置，否则 AutoTune
在 `workspace_path` 下生成报告路径。早期校验和请求文件错误不会写报告。

typed `TuneIndex` 和 `TuneSearch` 不接收报告路径，也绝不会写报告文件，而是通过结果对象
返回完整的 `JsonType` 报告。CLI 和 typed `TuneIndex` 设置
`keep_intermediate=false` 时都只保留推荐索引并删除未选中 artifact；设置为 `true` 时保留
所有生成索引。没有 recommendation 时，默认删除所有生成索引。`TuneSearch` 不生成
artifact，因此 `workspace_path` 和 `keep_intermediate` 对它没有影响。

## 完整报告

完成评测后会返回以下顶层结构；CLI 还会把它写入磁盘：

```json
{
  "version": 1,
  "status": "success",
  "recommendation": {},
  "best_effort": null,
  "builds": [],
  "trials": [],
  "request": {},
  "elapsed_seconds": 84.2,
  "report_path": "/tmp/vsag_autotune/run-....json"
}
```

| 字段 | 含义 |
| --- | --- |
| `version` | 报告契约版本，固定为 `1`。 |
| `status` | `success`、`no_feasible_candidate` 或 `failed`。 |
| `recommendation` | 最优可行 trial；不存在时为 `null`。 |
| `best_effort` | 约束不可行时最接近的成功 trial，否则为 `null`。 |
| `builds` | 每组具体生成 build 一条记录；search-only 模式为空数组。 |
| `trials` | 每组已执行的具体 search 候选一条记录。 |
| `request` | 调优引擎实际使用的有效规范化请求。 |
| `elapsed_seconds` | AutoTune 墙钟时间包含 artifact 清理，不含最终报告写入。 |
| `report_path` | 持久化完整报告路径；只在 CLI/JSON 适配器中存在。 |
| `failure` | 整体执行失败时出现。 |

早期校验失败不包含 `builds`、`trials` 或 `report_path`，也不会写报告。如果所有候选评测
失败，CLI 只要报告路径仍然可用，就会保留尝试过的 build/trial 记录并写报告。typed API
的报告绝不会包含 `report_path`。

规范化 `request` 包含推导出的 dataset 元数据、workload 默认值、constraints、objective、
config 和 output。build-and-search 模式还包含 `index_spaces`；search-only 模式包含
`index_name` 和 `parameter_space`。CLI 报告还会在这个规范化对象顶层保留离线
`data_path`，存在已有索引输入时也保留 `index_path`。

### Status 语义

- `success`：至少一个成功 trial 满足全部约束，设置 `recommendation`。
- `no_feasible_candidate`：存在成功 trial，但没有任何一个满足全部约束；设置
  `best_effort` 用于解释，它不是有效推荐。typed 调用会返回带
  `TuneStatus::NO_FEASIBLE_CANDIDATE` 的值，而不是 error。
- `failed`：请求非法、执行或报告阶段失败，或者没有成功且带目标指标的 trial。

### Recommendation 和 Best Effort

build-and-search 模式下，`recommendation` 包含：

```json
{
  "index_name": "hgraph",
  "create_params": {},
  "search_params": {},
  "workload": {"top_k": 10, "concurrency": 1},
  "metrics": {},
  "artifacts": {},
  "evidence": {"build_id": "build-0", "trial_id": "trial-0"}
}
```

search-only 模式会省略 `create_params`、`artifacts` 和 `evidence.build_id`，只包含具体
查询参数、workload、metrics 和 `evidence.trial_id`。`best_effort` 使用相同的模式相关
字段，并额外包含 `constraint_evaluation`。它首先按违反或缺失约束数量选择，再比较归一化
violation 大小，只用于解释。

### Build 记录

search-only 模式的 `builds` 为空。build-and-search 模式下，每个 `builds[]` 元素包含：

| 字段 | 含义 |
| --- | --- |
| `build_id` | 被 trial 引用的稳定 ID。 |
| `index_name` | 具体索引类型。 |
| `create_params` | 具体创建参数。 |
| `status` | `success` 或 `failed`。 |
| `metrics` | 可用的 build 共享指标。 |
| `artifacts` | `source`、`index_path`、`use_existing_index` 和 `retained`。 |
| `failure` | 结构化失败或 `null`。 |
| `elapsed_seconds` | build group 准备耗时。 |
| `raw_eval_result` | 请求原始输出且真实 build eval 成功时出现。 |

### Trial 记录

每个 `trials[]` 元素包含：

| 字段 | 含义 |
| --- | --- |
| `trial_id` | 稳定 trial ID。 |
| `build_id` | 关联的 build group；仅 build-and-search 模式存在。 |
| `index_name` | 具体索引类型。 |
| `create_params` | 具体创建参数；仅 build-and-search 模式存在。 |
| `search_params` | 具体查询参数。 |
| `status` | `success` 或 `failed`。 |
| `metrics` | 用于约束评估和选择的可用指标。 |
| `constraint_evaluation` | `satisfied` 和 `violations` 数组。 |
| `artifacts` | 从 build group 复制的产物证据；仅 build-and-search 模式存在。 |
| `failure` | 结构化失败或 `null`。 |
| `elapsed_seconds` | search trial 墙钟时间。 |
| `raw_eval_result` | 请求原始输出且原生 search eval 成功时出现。 |

对于自适应 HGraph `ef_search` 范围，报告会为每个实际评测点记录一条 trial。每条记录中的
`search_params` 都是具体值；`$range` 表达式只保留在规范化请求中。每条记录都会评测完整
query workload。

每条 constraint violation 包含 `metric`、`comparison`、`expected` 和 `actual`。指标缺失或
非有限数时，`actual` 为 `null`。

同一 build group 的 search trial 复用同一个已加载索引实例。新生成的索引只构建和序列化
一次。search-only 模式直接为所有 trial 复用调用方索引，或复用 CLI 适配器反序列化得到的
索引。

### Artifact 语义

V1 只在 build-and-search 记录中提供 artifact 字段。`artifacts.source` 为 `generated`；
`artifacts.index_path` 用于说明被评测索引曾存放在哪里，不保证响应返回时路径仍然存在。
需要检查 `artifacts.retained`：

- `true`：这是最终推荐产物，或者请求保留所有生成索引；
- `false`：AutoTune 计划删除或已经删除生成产物。

### 结构化失败

所有失败使用统一结构：

```json
{
  "stage": "validation",
  "code": "invalid_request",
  "message": "request.workload.top_k is required"
}
```

常见 stage 包括 `cli`、`validation`、`candidate_generation`、`build`、`search`、
`evaluation`、`selection` 和 `report`。build/trial 失败保留在对应记录中，其他候选可以
继续执行。

| Stage | Code | 含义 |
| --- | --- | --- |
| `cli` | `request_file_error` | CLI 无法读取或解析请求文件。 |
| `validation` | `invalid_request` | 请求校验失败。 |
| `candidate_generation` | `invalid_request` | 候选表达式或数量非法。 |
| `build` | `build_evaluation_failed` | 一组 build 失败。 |
| `search` | `build_failed` | 关联 build 失败，search 被跳过。 |
| `search` | `search_evaluation_failed` | 一组 search trial 失败。 |
| `evaluation` | `all_trials_failed` | 所有候选 trial 都执行失败。 |
| `selection` | `objective_metric_unavailable` | 存在成功 trial，但都没有产生目标指标。 |
| `evaluation`、`report` | `execution_failed` | 顶层执行或报告写入失败。 |

## CLI Summary

CLI 按以下顺序输出完整报告的紧凑子集：

1. `recommendation`（存在时）；
2. `best_effort`（存在时）；
3. `failure`（存在时）；
4. `status`；
5. `elapsed_seconds`；
6. `report_path`（存在时）；
7. `version`。

标准输出会省略值为 null 的结果分支和详细 build/trial 数组。完整证据请读取
`report_path`。

CLI 在 `status=failed` 或命令行/请求文件出错时返回 `1`。`success` 和
`no_feasible_candidate` 都返回 `0`；调用方必须检查 `status`，不能只用退出码判断是否存在
recommendation。

## Build-tree C++ 入口

实验性的可选 CMake target `vsag::autotune` 暴露 `tools/autotune/autotune.h` 中的接口。
该 target 和头文件只在构建目录中提供，不会安装：

```cpp
tl::expected<IndexResult, Error> TuneIndex(const IndexRequest& request);
tl::expected<SearchResult, Error> TuneSearch(const SearchRequest& request);
JsonType RunAutoTune(const JsonType& request);  // CLI adapter
```

JSON 入口只负责离线适配：加载 `data_path`；存在 `index_path` 时创建并反序列化索引；随后
构造与 typed 入口相同的内部请求。V1 不安装该 target 和头文件。由于底层 eval 路径会配置
进程级 OpenMP 状态，同一进程内的多次调用会串行执行。

`TuneIndex` 同时调优构建参数和查询参数：

```cpp
IndexRequest request;
request.base = base;
request.metric_type = METRIC_L2;
request.workload = {queries, ground_truth, 10, 48};
request.index_spaces = {{"hgraph", create_candidate_space, search_candidate_space}};
request.constraints = {{Metric::RECALL_AT_K, 0.95}};
request.objective = Metric::LATENCY_AVG_MS;
auto result = TuneIndex(request);
```

返回值包含已加载、可查询的推荐索引，以及具体 create 和 search 参数。`metrics` 和完整
`report` 都以 `JsonType` 返回；typed 调用绝不会持久化报告。`TuneSearch` 对已经构建或
加载的索引调查询参数：

```cpp
SearchRequest request;
request.index = existing_index;
request.workload = {queries, ground_truth, 10, 48};
request.parameter_space = search_candidate_space;
request.constraints = {{Metric::RECALL_AT_K, 0.95}};
request.objective = Metric::LATENCY_AVG_MS;
auto result = TuneSearch(request);
```

AutoTune 从 `existing_index` 推导类型和元素数量，因此 `TuneSearch` 不需要 base vector 或
metric type。只有 recall 是约束或目标时才需要 ground truth。`TuneSearch` 不提供构建阶段
指标和 `index_size_mb`。`index_memory_mb` 可以作为约束，但不能作为目标，因为它不会随
search 候选变化。

两个 typed 调用仅在请求非法或执行失败时返回 `tl::unexpected<Error>`。如果评测正常完成但
没有候选满足全部约束，则返回 `status=TuneStatus::NO_FEASIBLE_CANDIDATE` 和结构化
`best_effort`；该状态下推荐字段无效。
