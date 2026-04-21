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

- `Index::EstimateMemory(data_num)`：在构建前估算索引将占用的内存（示例：`examples/cpp/308_feature_estimate_memory.cpp`）。
- `Index::GetMemoryUsage()`：查询当前实际占用的字节数（示例：`examples/cpp/319_feature_get_memory_usage.cpp`）。

## 线程池

`Resource` 也接受用户提供的 `ThreadPool`，与 Allocator 配合可完全托管并行度与资源归属。见
`examples/cpp/203_custom_thread_pool.cpp`。

## 注意事项

- 自定义 Allocator 必须是线程安全的。
- `Allocator` 生命周期必须覆盖所有引用它的索引与结果对象。
- 若未显式指定，VSAG 会创建一个默认的基于 `malloc` 的 allocator。
