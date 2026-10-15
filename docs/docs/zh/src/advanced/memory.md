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
    uint64_t estimated = index->EstimateMemory(1'000'000);  // bytes
}
```

对于 HGraph，count 为 0 时返回 0；向上取整后的元素数或字节总量无法用 `uint64_t` 表示时，
抛出派生自 `std::exception` 的异常，而不是静默回绕。

完整示例：`examples/cpp/308_feature_estimate_memory.cpp`。

### `EstimateBuildMemory(num_elements)`

`Index::EstimateBuildMemory(num_elements)` 返回在索引当前状态下，成功 `Build` `num_elements`
条向量期间归属于 VSAG 的预估峰值内存（字节数）。与估算最终索引稳态大小的 `EstimateMemory`
不同，该估算覆盖构造期保留内存以及 `Build` 创建的存储：采样与训练缓冲区、预处理行、构建
scratch 和调度器对象。`EstimateMemory` 作为下界使用，因此结果不会低于最终占用量的估算值。

该估算计入两条分配路径：经由索引所配置 `Allocator` 发起的请求，以及经由全局 `operator new`
的标准库分配（共享状态控制块、`std::function` 目标、容器内部结构）。每个分配族只计一次，
包括 vector 与哈希表增长期间新旧存储并存的重叠部分。按约定排除：调用方自有的输入向量、
已有的 `Engine`/`Resource`/线程池归属、共享池上的无关任务、allocator 内部开销、线程栈与
运行时状态、以及任意失败路径上的异常载荷。结果是针对下列受支持配置的有文档约定的模型估算，
而不是通用的进程堆上界：

```cpp
if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_BUILD_MEMORY)) {
    uint64_t peak = index->EstimateBuildMemory(1000000);  // bytes
}
```

**前置条件。** 文档保证假定被估算的 `Build` 独占该索引运行 —— 不与其他操作并发，包括可能
租借 visited list 或分配索引自有 scratch 的并发 const 查询。另外，调用方不得在同索引变更或
关联资源扩容的同时调用 `EstimateBuildMemory` 或其 `CheckFeature` 判断。结果描述当前状态；
状态或资源变化之后，它不保证后续 `Build` 的数值。

**支持的配置。** 仅当 HGraph 当前配置与状态符合文档化内存模型时，才会公布
`SUPPORT_ESTIMATE_BUILD_MEMORY`：float32 标量 NSW 图、扁平的内存块式 code 与图存储；
`fp32`、`sq8` 或 `sq8_uniform` 基础量化；以及默认 `pg` 标签映射下的普通数值标签
（`label_remap_type: "robin"` 不受支持）。公开的默认 `Engine`、普通内部线程池以及已知的
内置外供线程池方式 —— 包括 `examples/cpp/308_feature_estimate_memory.cpp` 所用的配置 ——
均在覆盖范围内。资格还与编译期相关：标准库分配模型仅在 libc++ 15.0.7 或 21.1.6
（`_LIBCPP_VERSION` 为 15007 或 210106，ABI v1，且未启用优化版 `std::function` ABI）或
libstdc++ 上编译，因此在其他 C++ 标准库下无论配置如何该 flag 均为 false。峰值对照测量已在
原生 Apple Clang 21 / SDK 26.5 / libc++ 210106 构建以及 aarch64 Linux 的 GCC 11.4 头文件 /
libstdc++ 构建上完成；在其他工具链上具备资格只表示模型可用，不代表其运行时经过验证 ——
lint 所用的 LLVM 15 clang-tidy 解析不构成运行时验证。不支持的配置与状态会以 false 返回该
flag，并抛出派生自 `std::exception` 的类型化“不支持操作”错误，而不是返回数值。这包括
高级 transform、RaBitQ、MCI、删除或反向边、reorder/raw 存储、duplicate 或 deduplicated
存储、ELP，以及 attribute 或 extra-info 路径，还包括任何非空索引、immutable 索引、持有
导入或保留的 build cache、或已进入 `Merge`、`Tune`、`Deserialize` 或
`ParallelDeserialize` 的索引 —— 包括失败的调用，也包括 count 为 0 的 `Tune`，它即使在
空索引上也可能替换量化器与存储。

**规模与错误。** 内部实现将元素数向上取整到配置的容量步长，推导各组件包络，并以固定存储、
O(N) 时间重放 route-level 生成器 —— 不分配与 `num_elements` 成比例的内存。当
`num_elements` 向上取整后的容量超出内部 ID 范围，或任何中间结果无法用 `uint64_t` 表示时，
抛出类型化的 invalid-argument 错误；不支持的配置与状态即使在 `num_elements` 为 0 时也产生
不支持操作错误。在受支持的空索引上，`EstimateBuildMemory(0)` 返回正的保留基线估算 ——
已构造的索引仍持有内存 —— 而 `EstimateMemory(0)` 返回 0。

**模型局限。** 反向标签映射使用名义上的质数表增长规则，仅在至多 8193 个不同标签以及非负
稠密 ID `0..N-1` 的情形下有论证依据；其他标签模式可能触发模型未覆盖的额外位移增长，而
大规模 route-map 的键值域上限是保守的、可能偏宽松。建模的输入是带数值标签的向量；
`Build` 复制进索引自有表的额外 source-ID 或 metadata 载荷不在保证范围内 —— 它们仍归
VSAG 所有，也不属于调用方自有的排除项。对于不透明的自定义线程池或额外的封装层，超出文档
所述方式的 `Build` 产生的排队与包装开销未建模 —— 不予保证，也不会被重新标注为既有归属。
估算的对照目标是实测的 allocator 阶段峰值加上推导出的标准库分配包络，覆盖有文档约定的
24 格矩阵（`fp32`/`sq8`/`sq8_uniform` × 维度 16/128 × `max_degree` 8/32 × 串行与
宽度 8 线程池 / 4 个请求 worker 两种模式；N=1537，`ef_construction` 64，固定 256 KiB 块）。

**并行插入说明。** 在并行 `Build`/`Add` 期间，行插入最多分发给
`min(max(build_thread_count, 1), rows)` 个 worker，因此比有效 worker 数更宽的外供线程池
不会再增加行并发，每次 `Build` 的调度器对象也随该上界伸缩。使用外供线程池时，即使
`build_thread_count` 为 0 或 1，非空操作仍走并行路径 —— 运行单个 worker，但采用并行
路径的错误语义（重抛行号最小的失败行异常）。这适用于 `Add` 以及所有其他并行插入调用方，
不限于 `Build`；若提交 worker 失败发生在部分 worker 已启动之后，已启动的 worker 可能
排空所有剩余行，因此提交失败与行失败混合时的排序可能与旧版本不同。

示例 `examples/cpp/308_feature_estimate_memory.cpp` 会同时检查两个 flag。

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

| 标志                                | 含义                                                            |
|-------------------------------------|-----------------------------------------------------------------|
| `vsag::SUPPORT_ESTIMATE_MEMORY`     | 支持 `EstimateMemory(data_num)`。                               |
| `vsag::SUPPORT_ESTIMATE_BUILD_MEMORY` | `EstimateBuildMemory(num_elements)` 在索引当前配置与状态下可用。 |
| `vsag::SUPPORT_GET_MEMORY_USAGE`    | 支持 `GetMemoryUsage()`。                                       |

这些标志均可通过 `index->CheckFeature(...)` 查询 —— 参见
[索引自省](introspection.md)。

## 线程池

`Resource` 也接受用户提供的 `ThreadPool`，与 Allocator 配合可完全托管并行度与资源归属。见
`examples/cpp/203_custom_thread_pool.cpp`。

## 注意事项

- 自定义 Allocator 必须是线程安全的。
- `Allocator` 生命周期必须覆盖所有引用它的索引与结果对象。
- 若未显式指定，VSAG 会创建一个默认的基于 `malloc` 的 allocator。
