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

HNSW 图索引在迭代模式下还支持一个额外的运行期参数 `skip_ratio`，用于控制延续搜索时跳过已探索区域
的力度，详见 `examples/cpp/313_feature_search_allocator.cpp` 中的 HNSW 部分。

## 支持情况

通过 `Index::CheckFeature` 查询 `SUPPORT_KNN_ITERATOR_FILTER_SEARCH` 是否被支持：

| 索引类型 | 是否支持迭代搜索 |
|---------|----------------|
| hnsw       | 是 |
| hgraph     | 是 |
| ivf        | 否 |
| diskann    | 否 |
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
