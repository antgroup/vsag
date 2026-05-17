# 升级到 VSAG 1.0

本页汇总了从 **VSAG 0.18.x**（及更早版本）平滑升级到 **VSAG 1.0** 所需了解的
全部内容。请在重新编译或重新部署前先阅读本页。

> 进度跟踪在
> [issue #2069](https://github.com/antgroup/vsag/issues/2069) /
> [PR #2070](https://github.com/antgroup/vsag/pull/2070)。如有勘误或
> "升级途中踩到的坑"反馈，欢迎提 issue。

> 1.0 之后版本之间的兼容性规则将由独立的 *API 稳定性* 页面说明，作为后续
> PR 跟踪于 [#2069](https://github.com/antgroup/vsag/issues/2069)；
> 本页只覆盖 0.18 → 1.0 的一次性迁移。

## 一图速览

| 主题 | 1.0 中的状态 | 操作 |
|------|-------------|------|
| `hnsw` 索引 | 已弃用，仍可用 | 新增部署改用 [HGraph](../indexes/hgraph.md) |
| `diskann` 索引 | 已弃用，仍可用 | 新增部署改用 [IVF](../indexes/ivf.md) 或 [内存-磁盘混合索引](../advanced/hybrid_index.md) |
| `Index::KnnSearch(query, k, SearchParam&)` | 已弃用的重载 | 改用 `Index::SearchWithRequest(SearchRequest)` |
| `SearchParam::allocator` | 已弃用字段 | 改用 `SearchRequest::search_allocator_` |
| `Index::CalDistanceById`（批量） | 保留（拼写错误的名字） | 继续使用；正确拼写的 `CalcDistancesById` 在规划中（见 [#2068](https://github.com/antgroup/vsag/issues/2068)） |
| 0.18.x 序列化产物 | 1.0 可读 | 升级后建议重新序列化以采用新的布局优化 |
| 公共 C ABI | 稳定 | 无需操作 |

下文逐行展开，给出可直接套用的代码片段。

## 已弃用的索引

### `hnsw` → `hgraph`

`hnsw` 是从 hnswlib 继承下来的图索引。1.0 中为了向后兼容仍然保留，但
**已标记为弃用**；新增部署请使用 [HGraph](../indexes/hgraph.md)，它是
`hnsw` 的超集：

- 同样的分层图拓扑，同样的 `max_degree` / `ef_construction` /
  `ef_search` 调参。
- 统一的 `index_param` 构建参数 schema，量化选项更丰富（`fp32`、
  `fp16`、`bf16`、`sq8`、`sq8_uniform`、`sq4_uniform`、`pq`、`pqfs`、
  `rabitq`）。
- 可选的重排（`use_reorder` + `precise_quantization_type`）、去重、
  `Remove()`、以及基于 ELP 的运行期自动调参。

构建期映射：

```diff
- auto index = vsag::Factory::CreateIndex("hnsw", R"({
-     "dim": 768,
-     "dtype": "float32",
-     "metric_type": "ip",
-     "hnsw": {
-         "max_degree": 32,
-         "ef_construction": 400
-     }
- })").value();
+ auto index = vsag::Factory::CreateIndex("hgraph", R"({
+     "dim": 768,
+     "dtype": "float32",
+     "metric_type": "ip",
+     "index_param": {
+         "base_quantization_type": "fp32",
+         "max_degree": 32,
+         "ef_construction": 400
+     }
+ })").value();
```

搜索期映射：

```diff
- auto result = index->KnnSearch(query, k, R"({"hnsw": {"ef_search": 100}})").value();
+ auto result = index->KnnSearch(query, k, R"({"hgraph": {"ef_search": 100}})").value();
```

两个易错点：

1. 构建子对象的 key 从 `"hnsw"` 变成 `"index_param"`，并且
   `base_quantization_type` 是必填字段。
2. 搜索子对象的 key 也从 `"hnsw"` 变成 `"hgraph"`。

### `diskann` → `ivf` 或 内存-磁盘混合索引

`diskann` 提供了内存放 PQ、磁盘放原始向量的混合检索能力。1.0 中
**已弃用**；请按以下顺序选择替代项：

- [IVF](../indexes/ivf.md) —— 适合大规模分区检索；当数据可以完全放入
  内存时，是 `diskann` 的自然替代。
- [内存-磁盘混合索引](../advanced/hybrid_index.md) —— 当确实需要把
  部分索引下沉到 NVMe（语料巨大、内存预算紧张）时再使用。

优先尝试 IVF；只有在确实测量到内存是瓶颈时，再退回磁盘混合配置。

### `hnsw` / `diskann` 不再作为首选示例

网站文档中 [创建索引](../guide/create_index.md)、
[索引参数](index_parameters.md)、[序列化](../advanced/serialization.md)
等页面将在后续 PR 中改为默认使用 `hgraph` 示例，整体进度跟踪于
[#2069](https://github.com/antgroup/vsag/issues/2069)。原有的示例代码
`examples/cpp/101_index_hnsw.cpp`、`examples/cpp/102_index_diskann.cpp`
仍保留以供参考。

## 已弃用的检索 API：`SearchParam` → `SearchRequest`

历史上 VSAG 累积了多个 `Index::KnnSearch` 重载。1.0 的公开 API 收敛到
单一入口，把**所有**检索选项都通过一个 struct 传递：

```cpp
[[nodiscard]] tl::expected<DatasetPtr, Error>
SearchWithRequest(const SearchRequest& request) const;
```

`SearchRequest`（声明在 [`include/vsag/search_request.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/search_request.h)）
同时支持 KNN 与范围检索、属性过滤、回调过滤、bitset 过滤、迭代器检索、
每次检索独立的 allocator，以及 expected-labels 召回归因 —— 全部由一个
struct 承载。旧的 `Index::KnnSearch(query, k, SearchParam&)` 重载
**已弃用**，将在未来某个 major 版本中移除。

### 字段映射

| `SearchParam`（旧） | `SearchRequest`（新） | 说明 |
|---------------------|-----------------------|------|
| `parameters` (`const std::string&`) | `params_str_` (`std::string`) | JSON 参数字符串（如 `{"hgraph": {"ef_search": 200}}`）。 |
| `filter` | `filter_` + `enable_filter_ = true` | 回调式 `Filter` 对象，需要显式开启。 |
| `allocator` | `search_allocator_` | 每次检索使用的 arena allocator，见 [搜索路径 Allocator](../advanced/search_allocator.md)。 |
| `iter_ctx` | `p_iter_ctx_` + `enable_iterator_search_ = true` | 注意指针层级 —— `SearchRequest` 接收 `IteratorContext**`。 |
| `is_iter_filter` | 由 `enable_iterator_search_` 承担 | 迭代器检索改为一个布尔开关。 |
| `is_last_search` | `is_last_search_` | 语义不变。 |

`SearchRequest` 还额外暴露了 `SearchParam` 不具备的能力：

- `mode_`（`SearchMode::KNN_SEARCH` / `SearchMode::RANGE_SEARCH`）、
  `topk_`、`radius_`、`limited_size_` —— KNN 与范围检索共用同一 struct。
- `enable_attribute_filter_` + `attribute_filter_str_` —— SQL 风格的
  属性过滤，见 [属性过滤](../advanced/attribute_filter.md)。
- `enable_bitset_filter_` + `bitset_filter_` —— bitset 过滤。
- `expected_labels_` —— 用于召回调试 / 归因分析。

### 代码迁移

迁移前：

```cpp
vsag::SearchParam param(
    /*iter_filter_flag=*/false,
    R"({"hgraph": {"ef_search": 200}})",
    /*filter=*/my_filter,
    /*allocator=*/my_arena);
auto result = index->KnnSearch(query, /*k=*/10, param).value();
```

迁移后：

```cpp
vsag::SearchRequest req;
req.query_              = query;
req.mode_               = vsag::SearchMode::KNN_SEARCH;
req.topk_               = 10;
req.params_str_         = R"({"hgraph": {"ef_search": 200}})";
req.enable_filter_      = static_cast<bool>(my_filter);
req.filter_             = my_filter;
req.search_allocator_   = my_arena;
auto result = index->SearchWithRequest(req).value();
```

范围检索只需切换 `mode_`：

```cpp
req.mode_         = vsag::SearchMode::RANGE_SEARCH;
req.radius_       = 0.42F;
req.limited_size_ = 1000;   // -1 表示不限制
auto result = index->SearchWithRequest(req).value();
```

> **提示**：`SearchRequest` 是带默认值的 POD struct，包一层小型
> builder/helper 通常比旧的多参数 `SearchParam` 构造函数更清晰。

## `CalDistanceById` 拼写问题与 `CalcDistancesById` 迁移路径

`Index` 上有两种按 ID 计算距离的 API：

- **单条** ID，拼写正确：`CalcDistanceById(...)`。
- **批量** IDs，历史上**拼写有误**：`CalDistanceById(...)`（少了
  `Calc` 中的 `c`）。

该命名不一致在 [按 ID 计算距离](../advanced/calc_distance_by_id.md)
里有说明，并由 [#2068](https://github.com/antgroup/vsag/issues/2068)
跟踪。

**1.0 的处理：**

- 两个名字都继续可用；批量方法**不会**在 1.0 中改名。
- 未来某个版本会把批量方法重命名为 `CalcDistancesById`，旧名字会以
  弃用别名的形式至少保留一个 minor 版本。

**现在应该怎么做：**

- 批量调用继续使用 `CalDistanceById`。
- 在自己的代码中包一层 thin wrapper，未来重命名时只改 wrapper：

  ```cpp
  // wrappers/vsag_calc_distance.h
  inline auto CalcDistances(const vsag::IndexPtr& index,
                            const float* query,
                            const int64_t* ids,
                            int64_t count,
                            bool precise = true) {
      // 当前：转发到拼写错误的旧名字。
      return index->CalDistanceById(query, ids, count, precise);
  }
  ```

## 序列化兼容性

VSAG 1.0 通过三种序列化接口（`BinarySet` / `ReaderSet`、文件流、
自定义 `WriteFuncType`）均可**读取** 0.18.x 序列化产物；磁盘布局与
元数据格式在前向方向上兼容。

建议：

- 升级完成后**重新序列化一次**，让新产物使用 1.0 的布局改进。
- 反向兼容（1.0 → 0.18.x）**不支持**。升级窗口期内，每个生产集群应固定
  在单一 reader 版本上。
- `Deserialize` 仍要求目标索引为空，且构建配置（`dim`、`dtype`、
  `metric_type` 等）与原索引一致；详见
  [序列化](../advanced/serialization.md)。
- DiskANN 的磁盘文件仍独立管理；如果你正在从 `diskann` 迁出，把这些
  磁盘文件当作可丢弃数据，在新的索引类型上重建即可。

之后版本之间的兼容性合约将由独立的 *API 稳定性* 页面规范，作为后续 PR
跟踪于 [#2069](https://github.com/antgroup/vsag/issues/2069)。

## 默认值与行为变化

升级 1.0 后建议确认：

- **MKL 默认关闭。** `VSAG_ENABLE_INTEL_MKL`（CMake：
  `ENABLE_INTEL_MKL`）默认 `OFF`。在原本期望开启 MKL 的 Intel CPU
  环境，请在构建时显式 `VSAG_ENABLE_INTEL_MKL=ON`。
  [标准环境性能参考](performance.md) 的数据是在 MKL 关闭下采集的。
- **HGraph 默认值。** `max_degree` 默认 `64`，`ef_construction` 默认
  `400`，`graph_type` 默认 `"nsw"`。构建子对象的 key 为
  `index_param`；`base_quantization_type` 是必填字段。
- **`support_remove` / `support_duplicate` 默认关闭。** 如果你依赖
  `Remove()` 或之前实验分支上的去重能力，请在 `index_param` 中显式
  开启。
- **`store_raw_vector`** 默认关闭，只在确实需要在构建后访问原始向量
  时再开启（例如基础表征已被量化、需要 `cosine` 重排）。

本页未覆盖到的行为变化欢迎提 issue 反馈。

## 构建系统与打包说明

- **工具链版本约束不变。** `clang-format` / `clang-tidy` 必须**严格
  等于 15**；GCC ≥ 9.4，Clang ≥ 13.0，CMake ≥ 3.18。
- **ABI 变体不变。** 根据下游工具链选择对应的发行包：
  - `make dist-pre-cxx11-abi` —— GCC `_GLIBCXX_USE_CXX11_ABI=0`。
  - `make dist-cxx11-abi` —— GCC `_GLIBCXX_USE_CXX11_ABI=1`。
  - `make dist-libcxx` —— Clang libc++。
- **Python wheel。** 继续支持 `pip install pyvsag`；源码构建用
  `make pyvsag PY_VERSION=3.10` 或 `make pyvsag-all`。
- **Node.js / TypeScript。** `npm install vsag`。

## 升级操作清单

驱动从 0.18.x 升级到 1.0 的一个简短有序清单：

1. **通读本页**，并速览 [版本日志](release_notes.md)。
2. **盘点代码中的弃用用法**：
   - `vsag::Factory::CreateIndex("hnsw", ...)` / `("diskann", ...)`。
   - `Index::KnnSearch(query, k, SearchParam&)` 以及直接构造
     `vsag::SearchParam` 的代码。
   - 直接调用 `CalDistanceById`（批量重载）的位置；现在包一层
     wrapper，未来改名只需改 wrapper。
3. **规划替换**，优先选 HGraph 与 `SearchRequest`。
4. **预发环境验证。** 用同样的 `dim` / `metric_type` 构建 HGraph
   （或 IVF），通过 [`eval_performance`](eval.md) 对比召回与延迟。
5. **序列化往返验证。** 用 1.0 二进制加载 0.18.x 产物，重新序列化后
   再次加载。
6. **灰度滚动。** 旧版本集群作为回滚池保留，直到新集群在某个 1.0.x
   小版本上稳定一段时间。
7. **更新 CI/CD 版本约束。** `pip install pyvsag==1.0.*`、
   `npm install vsag@^1.0.0`、C++ 发行包固定到匹配的 ABI 变体。

升级完成后，欢迎提 issue 或贡献一段"实战记录"，帮助本页持续完善。

## 参考

- [版本日志](release_notes.md)
- *API 稳定性*（规划中，见 [#2069](https://github.com/antgroup/vsag/issues/2069)）
- [HGraph](../indexes/hgraph.md)
- [IVF](../indexes/ivf.md)
- [搜索路径 Allocator](../advanced/search_allocator.md)
- [序列化](../advanced/serialization.md)
- 序列化格式兼容性声明。
- 默认值与行为变化。
- 构建系统 / 打包相关说明。
- 升级操作清单。
