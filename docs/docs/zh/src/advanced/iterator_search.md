# 迭代式搜索

VSAG 支持**迭代式搜索**（Iterator Search）：调用方无需一次性请求 top-`k`，而是可以分多次、增量地
拉取结果，VSAG 在调用之间保留内部搜索状态。后续调用会从上一次结束的位置继续，返回**不重叠**的
新结果。

适用场景：

- 上层应用有外部 rerank 或后过滤逻辑，需要边拉取边判断，直到攒够通过条件的结果。
- 结果消费是惰性 / 流式的（如分页 UI、服务器端游标）。
- 最终需要的 `k` 不确定，需按需扩展。

## 工作原理

迭代式搜索依赖一个生命周期较长的 `IteratorContext` 对象，其中保存：

- 当前的候选堆与已访问位图；
- 在底层图 / 倒排链上的游标。

首次调用时，如果传入的指针为 `nullptr`，索引会在内部创建一个 `IteratorContext`；后续调用复用它，
搜索因此可以"继续"而不是"重新开始"。**调用方完成后需要自行 `delete` 这个 `IteratorContext`**——
迭代器持有的内部状态由 `delete` 释放。

`is_last_search` 标记是**可选**的：当置为 `true` 时，索引会把上下文里仍缓存的候选（"discard heap"
中尚未对外返回的部分）作为该次调用的结果一次性输出。如果你需要这部分尾部候选，就发起一次
`is_last_search=true` 的调用；如果不需要，直接 `delete` 上下文即可，无需"收尾调用"。注意返回结果
仍会被 `k` 截断，想拿到全部尾部候选时需要把 `k` 设得足够大。

## 基本用法（`SearchParam` API）

```cpp
#include <vsag/vsag.h>

// 1. 构造索引（以 HGraph 为例）
auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_params).value();
index->Build(dataset);

// 2. 准备查询
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec)->Owner(false);

// 3. 以迭代模式配置 SearchParam
nlohmann::json search_parameters = {
    {"hgraph", {{"ef_search", 100}}},
};
std::string param_str = search_parameters.dump();

vsag::SearchParam search_param(
    /*iter_filter_flag=*/true,   // 开启迭代模式
    param_str,
    /*filter=*/nullptr,
    /*allocator=*/&allocator,
    /*iter_ctx=*/nullptr,        // 首次调用：内部自动创建上下文
    /*last_search_flag=*/false);

// 4. 第一页
auto page1 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 5. 后续页：上下文延续，结果与 page1 不重叠
auto page2 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 6. （可选）取出上下文中仍缓存的候选；如果不需要，可跳过本步，
//    清理只依赖第 7 步的 delete。
search_param.is_last_search = true;
auto page3 = index->KnnSearch(query, /*k=*/10, search_param).value();

// 7. 由调用方销毁上下文——这才是真正释放资源的地方。
delete search_param.iter_ctx;
```

> 参考示例：`examples/cpp/313_feature_search_allocator.cpp`、
> `examples/cpp/314_feature_hgraph_search_allocator.cpp`。

## 另一种写法：显式传入 `IteratorContext`

更底层的 `KnnSearch` 重载允许直接传入 `IteratorContext*&`，VSAG 自身的测试用例
`tests/test_index/test_index_search.cpp` 即采用这种形式连续调用：

```cpp
vsag::IteratorContext* iter_ctx = nullptr;

auto r1 = index->KnnSearch(query, k1, param_str, filter, iter_ctx, /*is_last_search=*/false);
auto r2 = index->KnnSearch(query, k2, param_str, filter, iter_ctx, /*is_last_search=*/false);
auto r3 = index->KnnSearch(query, k3, param_str, filter, iter_ctx, /*is_last_search=*/false);

delete iter_ctx;
```

每次调用都会推进 `iter_ctx`；多次结果的并集就是按距离顺序、不重叠的延续序列。如果还想取出
上下文中仍缓存的尾部候选，可以在最后再加一次 `is_last_search=true` 的调用。

> **`SearchRequest` API。** `SearchRequest` 中定义了 `enable_iterator_search_` /
> `p_iter_ctx_` / `is_last_search_` 三个字段，但仓库内当前的 `SearchWithRequest` 实现尚未
> 读取这些字段，无法通过 `SearchWithRequest` 触发迭代式搜索。在这部分接入完成之前，请使用
> 上面两种 `KnnSearch` 形式。

## 与过滤器组合

迭代式搜索可以与常规过滤器（label filter、attribute filter、bitset filter）组合，典型场景是
"持续迭代直到外部检查通过的结果攒够"：

```cpp
size_t needed = 50;
std::vector<int64_t> kept;
vsag::IteratorContext* ctx = nullptr;

while (kept.size() < needed) {
    auto page = index->KnnSearch(query, 32, param_str, filter, ctx, /*is_last_search=*/false);
    if (!page.has_value() || page.value()->GetDim() == 0) break;

    for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
        if (external_check(page.value()->GetIds()[i])) {
            kept.push_back(page.value()->GetIds()[i]);
        }
    }
}

// 释放迭代器内部状态；不需要"收尾调用"，
// 仅当还想取出上下文里仍缓存的候选时，再加一次 is_last_search=true 的调用。
delete ctx;
```

HGraph 图索引在迭代模式下还支持一个额外的运行期参数 `skip_ratio`，用于控制延续搜索时跳过已探索区域
的力度，详见 `examples/cpp/314_feature_hgraph_search_allocator.cpp`。

## 支持情况

通过 `Index::CheckFeature` 查询 `SUPPORT_KNN_ITERATOR_FILTER_SEARCH` 是否被支持：

| 索引类型 | 是否支持迭代搜索 |
|---------|----------------|
| hgraph     | 是 |
| ivf        | 否 |
| brute_force| 否 |
| sindi      | 否 |

使用前请在运行时通过 `index->CheckFeature(vsag::SUPPORT_KNN_ITERATOR_FILTER_SEARCH)` 检查，后续版本
中支持范围可能会扩大。

## 注意事项

- **所有权。** `IteratorContext` 由调用方持有，忘记 `delete` 会泄漏内部搜索状态（堆、已访问位图、
  allocator 临时分配）。资源释放完全依赖 `delete`，与 `is_last_search` 无关。
- **最后一次调用是可选的。** `is_last_search = true` 不是清理步骤，唯一作用是让索引把上下文里
  仍缓存的候选作为该次调用的结果输出（仍受 `k` 截断）。仅当你需要这些尾部候选时再发起这次调用，
  并把 `k` 设得足够大以避免截断。
- **参数一致性。** 同一个上下文复用期间，不要更换查询向量、距离度量或过滤器——只有保持逻辑上的
  同一次查询，迭代结果才有意义。
- **每次调用的 `k`。** `k` 只作用于单次调用；多次结果互不重叠，每次最多增加 `k` 条（不足则表示
  索引候选已耗尽）。
- **线程安全。** 单个 `IteratorContext` 不能在多线程间并发使用；不同查询应各自持有独立上下文。

## 原生 HGraph SearchSession（实验性）

`OpenSearchSession(query, allocator = nullptr)` 仅固定查询、索引和资源，返回单消费者、有所有权的会话，
通过 `SUPPORT_CONTINUE_SEARCH_SESSION` 检查支持情况。此 API 独立于 `IteratorContext`，旧迭代器实现保持不变。

Open 深拷贝查询，不创建计算器或执行路由/距离计算。首次 `Next(SearchSessionNextOptions)`
创建可复用计算器并执行上层路由，后续调用从保存的底层图 frontier 继续扩展，复用距离缓存、访问/扩展状态和计算器。
每次至少扩展 `max(ef_search, max_candidates)` 个剩余顶点，或耗尽 frontier；
过滤条件严格时可继续扩展以填满当前页。这不是预先计算大 KNN 结果后分页。
页内按距离排序，不保证跨页全局顺序；耗尽指可达图耗尽，不保证枚举不连通顶点。
过滤前 `HasMore()` 可保守返回 true。

`Next` 的 `max_candidates`、`filter`（nullptr 表示全部接受）和 `search_parameters` 仅对本次调用生效。
可在调用之间更换过滤器或修改同一过滤器对象，但调用期间条件必须稳定。每次重新检查所有已发现、未交付的 ID，
包括此前被过滤器或 threshold 拒绝的候选及普通重复向量别名。threshold 可收紧或放宽；精排可开关，
每个 ID 的精确距离延迟计算一次并单独缓存。改变 ef、需求量或条件不会重新路由或重复计算已有距离，已交付 ID 不再返回。

`HasMore()` 独立于当前过滤条件：尚未初始化、frontier 非空或存在已发现未交付 ID 时为 true。
空批次不关闭会话；即使遍历已耗尽，之后放宽过滤器仍可得到结果。当前请求必须在空批次时停止，
不能在无法满足的过滤条件下仅依赖 `while (HasMore())`，否则可能无限循环。
统计 `session_traversal_exhausted`、`session_undelivered_nodes` 区分遍历耗尽与保留的拒绝候选。
永久无法返回的 NaN ID 在不存在可挽救它的其他输出 cell 时不再计入 `HasMore`：没有独立精排 cell，或两份已缓存距离均为 NaN；仍保留为遍历桥梁。尚未计算的其他精排 cell 会保守保留该 ID。
参数是完整替换，不是 JSON 增量补丁。此 POC 每次扫描 O(N) 稠密 ID 域并对合格候选排序，持有 O(N) 距离和状态数组。
非法参数或需求量可重试，不破坏会话；遍历异常则关闭会话。

```cpp
auto opened = index->OpenSearchSession(query);
vsag::SearchSessionNextOptions options{10, R"({"hgraph":{"ef_search":64}})", nullptr};
if (opened.has_value()) {
    auto session = std::move(opened.value());
    while (session->HasMore()) {
        auto page = session->Next(options);
        if (!page.has_value() || page.value()->GetDim() == 0) break;
        // 消费本批次；下次调用前可修改 options。
    }
}
```

旧的 `OpenSearchSession(query, k_per_call, parameters, filter, allocator)` 和 `Next(max)` 仍作为便捷接口，
使用 Open 指定的默认条件，最小扩展量为 `max(k_per_call, ef_search)`。
显式 `Next(options)` 仅覆盖本次条件，不修改后续旧接口调用的默认值。

会话打开期间不能修改索引、查询或距离度量，也不能并发调用会话方法。Open 后调用方可以销毁查询和
公开索引对象，会话会保留后端资源。自定义 allocator 必须活得比会话久。返回结果独立拥有缓冲区，
在 Close、会话析构和索引析构后仍有效。Close 可重复调用；耗尽或关闭后的 Next 返回空结果，保留累计工作统计，本轮工作量为零，且不校验请求数量。
Close 释放遍历状态存储。

支持普通标签/extra-info 过滤、返回 extra info、输出距离 threshold 和普通精排。被过滤顶点仍作为
图遍历桥梁。不支持并明确拒绝：共享重复向量存储、强制删除、启用的 MCI/共轭图搜索、特殊 RaBitQ 搜索、
并行/超时/factor/暴力回退/跳数限制及显式跳过策略。frontier 和分数缓存使用标准库分配，
可选 allocator 用于查询拷贝和邻居临时存储，并不覆盖所有会话分配。
累计统计 `session_routing_runs`、`session_computers`、`session_scored`、`session_reordered`、
`session_expanded` 可用于检查真实遍历工作量。
