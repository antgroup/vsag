# 内存管理

VSAG 在关键路径上大量使用自定义 `Allocator` 与 `Resource`，允许用户：

- 接入业务侧已有的内存池；
- 对索引内存占用进行度量与上限控制；
- 在多进程 / NUMA 环境下精细分配内存来源。

## 自定义 Allocator

```cpp
class MyAllocator : public vsag::Allocator {
public:
    std::string Name() override { return "my_allocator"; }
    void* Allocate(size_t size) override;
    void Deallocate(void* p) override;
    void* Reallocate(void* p, size_t size) override;
    // ...
};

auto allocator = std::make_shared<MyAllocator>();
auto resource = std::make_shared<vsag::Resource>(allocator, /*thread_pool=*/nullptr);
auto engine = vsag::Engine(resource);

auto index = engine.CreateIndex("hgraph", build_params).value();
```

完整示例参见 `examples/cpp/201_custom_allocator.cpp`。

## 搜索路径上的临时 Allocator

`KnnSearch` / `RangeSearch` 支持为单次搜索注入临时 `Allocator`，用于在线程局部的 arena 中分配工作区，
避免与全局堆竞争：

```cpp
vsag::SearchParam search_param;
search_param.allocator = thread_local_allocator.get();
auto result = index->KnnSearch(query, k, search_param);
```

示例：`examples/cpp/313_feature_search_allocator.cpp`、`examples/cpp/314_feature_hgraph_search_allocator.cpp`。

## 估算与查询内存占用

### `EstimateMemory(data_num)`

`Index::EstimateMemory(data_num)` 返回索引在插入 `data_num` 条向量后预期占用的字节数。它仅基于
构建参数（dim、量化方式、`max_degree` 等）推算，不会分配任何向量存储，因此可以在空索引上安全
调用，是入库前评估节点规格的推荐方式：

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
    uint64_t estimated = index->EstimateMemory(1'000'000);  // 字节
}
```

完整示例：`examples/cpp/308_feature_estimate_memory.cpp`。

### `EstimateBuildMemory(num_elements)`

`Index::EstimateBuildMemory(num_elements)` 返回构建 `num_elements` 条向量的索引时**构建过程中**
所需的预估内存（字节数）。与 `EstimateMemory`（估算最终索引的稳态大小）不同，该接口考虑了构建
过程中仅临时存在的缓冲区与中间数据结构。构建期间的峰值内存通常高于构建完成后的内存占用：

```cpp
uint64_t peak = index->EstimateBuildMemory(1000000);  // 字节
```

目前仅 DiskANN 提供了有效实现，其他索引类型默认抛出异常。

### `GetMemoryUsage()`

`Index::GetMemoryUsage()` 返回索引**当前**占用的字节数：

```cpp
uint64_t bytes = index->GetMemoryUsage();
```

特性：

- 所有索引类型均实现了该方法，但只有通过 `CheckFeature` 公布 `vsag::SUPPORT_GET_MEMORY_USAGE`
  的索引才保证返回有意义的数值。HGraph、IVF、BruteForce、Pyramid、WARP 均声明了该能力
  （见 `src/algorithm/{hgraph,ivf,brute_force,pyramid,warp}.cpp`）；SINDI 出于
  接口纯虚函数的要求实现了该方法，但当前未设置该 feature flag，请仅把返回值视为参考信息。
- 线程安全；可与搜索并发轮询。
- 延迟在微秒量级 —— 适合生产环境的实时内存监控。
- 统计的是索引自身占用的内存（向量、图、量化器状态）。该值通常小于操作系统层面观察到的 RSS：
  RSS 还包含 allocator 的开销、临时 scratch buffer、以及索引外部持有的数据（例如用户自有的输入
  向量缓冲）。SINDI 索引尤其建议在构建完成**之后**调用 `GetMemoryUsage()` 才能拿到具有代表性的
  数值。

可运行示例：`examples/cpp/319_feature_get_memory_usage.cpp`，其中包含一个辅助函数将接口值与进程
驻留内存进行对照。

### `GetMemoryUsageDetail()`

`Index::GetMemoryUsageDetail()` 返回索引**当前**内存占用按组件的细分：

```cpp
std::unordered_map<std::string, uint64_t> detail = index->GetMemoryUsageDetail();
for (const auto& [component, bytes] : detail) {
    std::cout << component << ": " << bytes << " bytes\n";
}
```

返回的 map 的 key 为组件名，value 为对应内存字节数。该接口有助于了解索引内部的内存分布。

目前仅 HGraph 提供了有效实现，返回的组件包括 `basic_flatten_codes`、`bottom_graph`、
`route_graph`、`neighbors_mutex`、`pool`、`label_table`、`high_precise_codes`、
`extra_infos` 和 `raw_vector`。SINDI 返回空 map，其他索引类型默认抛出异常。

### 能力标志

| 标志                              | 含义                                  |
|-----------------------------------|---------------------------------------|
| `vsag::SUPPORT_ESTIMATE_MEMORY`   | 支持 `EstimateMemory(data_num)`。     |
| `vsag::SUPPORT_GET_MEMORY_USAGE`  | 支持 `GetMemoryUsage()`。             |

两个标志均可通过 `index->CheckFeature(...)` 查询 —— 参见
[索引自省](introspection.md)。

## 线程池

`Resource` 也接受用户提供的 `ThreadPool`，与 Allocator 配合可完全托管并行度与资源归属。见
`examples/cpp/203_custom_thread_pool.cpp`。

## 注意事项

- 自定义 Allocator 必须是线程安全的。
- `Allocator` 生命周期必须覆盖所有引用它的索引与结果对象。
- 若未显式指定，VSAG 会创建一个默认的基于 `malloc` 的 allocator。
