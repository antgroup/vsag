# 搜索路径 Allocator

VSAG 提供一个与索引自身 allocator 解耦的 **per-call** `Allocator` 注入点，适合：

- 把单次查询的内存与索引长期持有的堆隔离开；
- 在高并发在线场景下，每个线程绑一个 thread-local arena，彼此之间没有原子争用；
- 独立于索引地核算或限制每次查询的内存占用。

这个 Allocator 通过两个入口暴露：`SearchRequest::search_allocator_`（推荐）和旧版
`SearchParam::allocator`。**但具体有多少搜索路径真正消费这个 allocator，取决于索引与入口的实现。**
目前只有 `HGraph::SearchWithRequest` 把 `search_allocator_` 端到端贯通了（既用于临时缓冲，也用
于结果 `Dataset`）；其它 `SearchWithRequest` 实现（IVF / BruteForce / WARP）只在部分临时
状态上使用 `search_allocator_`，结果 `Dataset` 仍由索引自身的 allocator 分配。详见下文
[与索引 Allocator 的关系](#与索引-allocator-的关系)。

> **适用范围。** Allocator 注入目前只通过 `KnnSearch`（`SearchParam` 重载）和
> `SearchWithRequest` 暴露。`RangeSearch` 没有携带 Allocator 的重载；
> `SearchRequest::search_allocator_` 也不会被 range-search 路径读取。

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
> IVF、BruteForce、WARP 实现了它（`src/algorithm/{hgraph,ivf,brute_force,warp}.cpp`）。对于
> 尚未 override 的索引（HNSW、DiskANN、SINDI、Pyramid），请使用下文的旧版
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

- **HGraph** 是目前唯一把 `request.search_allocator_` 贯通到 `create_fast_dataset` 的索引
  （见 `src/algorithm/hgraph.cpp` 中 `ctx.alloc = request.search_allocator_`）。其结果 `Dataset`
  被标记为 `Owner(true, allocator)`，析构时会自动用该 allocator 释放 `ids` / `distances`。
- **IVF / BruteForce / WARP** 当前用 `create_fast_dataset(..., allocator_)` 构造结果，即索引
  自身的 allocator（`src/algorithm/ivf/ivf.cpp`、`src/algorithm/bruteforce/bruteforce.cpp`；
  WARP 使用 BruteForce 的 WARP 模式实现）。这些路径上 `request.search_allocator_` 只会被部分
  临时缓冲读取，结果缓冲仍由索引 allocator 持有。在这些索引上请把结果 `Dataset` 的生命周期
  视为绑定到索引 allocator。

实际意义：

- **不要手动 `Deallocate` 结果缓冲。** 让 `Dataset` 离开作用域即可；同时手动 `Deallocate(...)`
  与析构器释放会触发双重释放，属于未定义行为。
- **持有结果的那个 allocator 必须比结果 `Dataset` 活得更久。** HGraph 上是 per-search
  allocator；IVF / BruteForce / WARP 上是索引 allocator（索引活着它就活着）。
- **`examples/cpp/314_feature_hgraph_search_allocator.cpp` 目前显式地 Deallocate。** 这是早期
  API 迭代遗留的写法；针对当前 owner-tracking 行为的新代码应改为依赖 `Dataset` 析构器。

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

| 场景                                                                       | 使用的 allocator                                                                                                       |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 索引构建、插入、持久状态                                                     | `Resource` 的 allocator（未传入则使用默认 allocator）。                                                                  |
| `HGraph::SearchWithRequest` 的临时缓冲与结果 `Dataset`                        | 已设置 `search_allocator_` 时使用它，否则使用 `Resource` 的 allocator。HGraph 是目前唯一把 `search_allocator_` 贯通到结果的索引。 |
| `IVF` / `BruteForce` / `WARP` `SearchWithRequest` 的结果 `Dataset`            | 始终使用索引自身的 allocator（`allocator_`）。目前**不**消费 `search_allocator_`。                                       |
| `IVF` / `BruteForce` / `WARP` `SearchWithRequest` 的部分临时状态              | 设置 `search_allocator_` 时会用它分配部分临时缓冲，否则使用索引 allocator。                                              |
| `KnnSearch(query, k, SearchParam)`（旧版）                                   | 在支持 `SearchParam::allocator` 的索引上（如 HGraph 示例）使用该 allocator，否则使用 `Resource` allocator。           |
| `KnnSearch(query, k, parameters_str)`                                       | 无 per-search Allocator 入口，统一使用 `Resource` 的 allocator。                                                         |
| `RangeSearch(...)`（所有形态）                                               | 使用 `Resource` 的 allocator；没有 per-search Allocator 入口。                                                           |

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
