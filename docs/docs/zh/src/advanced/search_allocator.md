# 搜索路径 Allocator

VSAG 提供一个与索引自身 allocator 解耦的 **per-call** `Allocator` 注入点，适合：

- 把单次查询的内存与索引长期持有的堆隔离开；
- 在高并发在线场景下，每个线程绑一个 thread-local arena，彼此之间没有原子争用；
- 独立于索引地核算或限制每次查询的内存占用。

这个 Allocator 通过两个入口暴露：`SearchRequest::search_allocator_`（推荐）和旧版
`SearchParam::allocator`。**但具体有多少搜索路径真正消费这个 allocator，取决于索引与入口的实现。**
HGraph、IVF、Pyramid 与 SINDI 会把非空的 `SearchRequest::search_allocator_` 用于非空结果缓冲
和搜索临时状态。BruteForce（包括 WARP 模式）只把它用于部分临时工作，结果 `Dataset` 仍使用
索引 allocator。详见下文[与索引 Allocator 的关系](#与索引-allocator-的关系)。

> **适用范围。** Allocator 注入通过 `KnnSearch`（`SearchParam` 重载）和
> `SearchWithRequest` 暴露。专用的 `RangeSearch` 重载没有 allocator 参数，但 range 模式的
> `SearchRequest` 可以携带它，支持该入口的索引也会在范围搜索路径读取 `search_allocator_`。

## 推荐 API —— `SearchRequest::search_allocator_`

```cpp
#include "vsag/search_request.h"

vsag::SearchRequest req;
req.query_ = query;
req.mode_ = vsag::SearchMode::KNN_SEARCH;
req.topk_ = 10;
req.params_str_ = R"({"hgraph":{"ef_search":100}})";
req.search_allocator_ = thread_local_allocator.get();  // 可选，可为 nullptr

auto result = index->SearchWithRequest(req).value();
```

`SearchRequest`（`include/vsag/search_request.h`）是当前未废弃、推荐用来驱动单次搜索的入口。
`search_allocator_` 字段是可选的，留空时索引会回退到它所属 `Resource` 上的 allocator。

> **可用性。** `Index::SearchWithRequest` 默认实现会返回 *不支持* 错误。目前只有 HGraph、
> IVF、BruteForce、WARP、SINDI 和 Pyramid 实现了它。Pyramid 支持 KNN 请求搜索，并可通过
> `expected_labels_` 生成 reasoning 报告；携带 expected labels 的 Range 请求暂不支持。对于
> 尚未 override 的索引（SINDI_V2），请使用下文的旧版
> `SearchParam` 路径。

## 旧版 API —— `SearchParam::allocator`（已弃用）

```cpp
#include "vsag/search_param.h"

nlohmann::json search_params = {{"hgraph", {{"ef_search", 100}}}};
std::string param_str = search_params.dump();

vsag::SearchParam search_param(/*iter_filter=*/false,
                               param_str,
                               /*filter=*/nullptr,
                               /*allocator=*/thread_local_allocator.get());
auto result = index->KnnSearch(query, /*k=*/10, search_param).value();
```

`SearchParam` 在 `include/vsag/search_param.h` 中以文档注释的形式标注为已弃用
（"Use SearchRequest instead"），仅为源码兼容保留。注意当前只是注释层面的弃用 —— struct
本身并没有 C++ `[[deprecated]]` 属性，编译器不会发出弃用告警；但新代码如果所用索引已支持
`SearchRequest`/`SearchWithRequest`，仍应优先使用该路径。
`examples/cpp/314_feature_hgraph_search_allocator.cpp`（HGraph）展示了旧版形式。

## 结果所有权

结果 `Dataset` 的所有权契约取决于具体实现 `SearchWithRequest` 的索引：

- **HGraph / IVF / Pyramid / SINDI** 会把选中的 query allocator 传给非空结果构造逻辑。
  `request.search_allocator_` 非空时，返回的 `Dataset` 通过该 allocator 持有 `ids` / `distances`
  缓冲，并在析构时释放它们。
- **BruteForce / WARP** 当前调用不带 allocator 参数的
  `pack_knn_result_with_extra_info(heap)`，因此结果使用索引 allocator；
  `request.search_allocator_` 仍用于部分临时工作。
- **空结果** 使用共享的 empty-dataset 路径，不会通过 per-search allocator 分配结果缓冲。

实际意义：

- **不要手动 `Deallocate` 结果缓冲。** 让 `Dataset` 离开作用域即可；同时手动 `Deallocate(...)`
  与析构器释放会触发双重释放，属于未定义行为。
- **持有结果的那个 allocator 必须比结果 `Dataset` 活得更久。** HGraph、IVF、Pyramid、
  SINDI 上可能是 per-search allocator；BruteForce / WARP 上则是索引 allocator。

最简单的安全模式是「一线程一 allocator，批与批之间 reset」：

```cpp
ArenaAllocator arena;       // thread-local，足以容纳一批

for (const auto& q : batch) {
    vsag::SearchRequest req;
    req.query_ = q;
    req.topk_ = topk;
    req.params_str_ = params;
    req.search_allocator_ = &arena;
    auto result = index->SearchWithRequest(req).value();
    consume(result);
    // result Dataset 在这里析构；arena 通过自己的 Deallocate 释放 ids/distances。
}
arena.reset();              // 一次性回收本批所有 per-query 缓冲
```

## 与索引 Allocator 的关系

| 场景 | 使用的 allocator |
|---|---|
| 索引构建、插入、持久状态 | `Resource` 的 allocator（未传入则使用默认 allocator）。 |
| HGraph / IVF / Pyramid / SINDI 的 `SearchWithRequest` 非空结果 | 已设置 `search_allocator_` 时使用它，否则使用索引 allocator。 |
| BruteForce / WARP 的 `SearchWithRequest` 结果 | 使用索引 allocator；`search_allocator_` 只用于部分临时工作。 |
| `SearchWithRequest` 临时状态 | 由索引实现决定；上述实现至少会在部分查询路径读取 `search_allocator_`。 |
| `KnnSearch(query, k, SearchParam)`（旧版） | 支持该参数的旧版重载使用 `SearchParam::allocator`，否则使用索引 allocator。 |
| `KnnSearch(query, k, parameters_str)` | 没有 per-search allocator 入口，使用索引 allocator。 |
| 专用 `RangeSearch(...)` 重载 | 没有 allocator 参数，使用索引 allocator。 |
| Range 模式的 `SearchWithRequest` | 与 KNN 模式遵循相同的索引特定规则。 |

设置 per-search Allocator 不会影响索引的永久数据结构。它只是收窄了某一次搜索调用所触碰内存的
生命周期 —— 且仅限于索引/入口实际消费它的那部分（详见各行说明）。

## 约束

- allocator 只有在跨线程共享时才必须线程安全；thread-local arena 不需要内部同步。
- allocator 的生命周期必须超过它产生的每一个结果 `Dataset`。
- `Reallocate(nullptr, size)` 必须等价于 `Allocate(size)`。VSAG 的内部容器依赖该契约。

## 可运行示例

- `examples/cpp/314_feature_hgraph_search_allocator.cpp` —— HGraph（`sq8`）+ 自定义 allocator。

参见 [内存管理](memory.md) 了解索引级 `Allocator` / `Resource` 的设置，以及
[过滤搜索](filtered_search.md) 了解如何在 `SearchRequest` 中同时使用 per-search Allocator 与
自定义过滤器。
