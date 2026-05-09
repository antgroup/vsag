# 属性过滤（混合搜索）

属性过滤（Attribute Filter），又称**混合搜索**（Hybrid Search）或**带结构化谓词的近邻搜索**，
让 `KnnSearch` / `RangeSearch` 只返回结构化标签满足某个 SQL 风格表达式的向量。相比
[带过滤的搜索](filtered_search.md) 中基于 id 的过滤方式，它能直接表达类似下面的谓词：

```
category = "electronics" AND price <= 1000 AND multi_in(tag, "promo|new", "|")
```

而无需写回调代码。VSAG 在向量索引旁额外构建一份属性倒排索引；表达式只解析一次，并在图遍历
过程中完成判定，从而尽早剪除不可能满足条件的候选。

> 本文中的“混合搜索”指的是**向量 + 结构化属性**的混合检索；DiskANN 的“**内存 + 磁盘**”
> 存储混合请参见 [内存-磁盘混合索引](hybrid_index.md)。

## 何时选择哪种过滤 API

| 需求 | 推荐 |
|---|---|
| 排除一组已知 id（例如墓碑） | [位图 / 函数过滤](filtered_search.md) |
| 在 id 上跑用户自定义逻辑 | [`Filter` 对象](filtered_search.md) |
| 在图内基于每条向量的字节负载过滤 | [Extra Info](extra_info.md) |
| 在**命名、有类型的字段**上做 AND/OR/IN 判定 | **本文** |

三者可以同时放进同一个 `SearchRequest`，按 AND 组合。

## 索引支持情况

| 索引             | 构建时启用 `use_attribute_filter` | `SearchWithRequest` + 属性表达式 | `UpdateAttribute` |
|------------------|:---------------------------------:|:--------------------------------:|:-----------------:|
| HGraph           |                支持               |               支持               |        支持       |
| IVF              |                支持               |               支持               |        支持       |
| BruteForce       |                支持               |               支持               |        支持       |
| WARP（稀疏）     |                支持               |               支持               |        支持       |
| HNSW / DiskANN / SINDI / Pyramid | — | 仅支持基于 id 的过滤，详见 [带过滤的搜索](filtered_search.md) | — |

启用 `use_attribute_filter` 后，BruteForce 暂不支持 `Remove`（如需删除请重建索引）。

## 属性数据模型

属性按向量定义，组织成 `AttributeSet`（`include/vsag/attribute.h`）。每个属性包含：

- **名称**（字符串）；
- **值类型**（`AttrValueType` 枚举）；
- **值列表**——所有字段都是多值字段，因此 `IN` 风格的成员判定能自然适用于标签类字段。

支持的值类型：

```cpp
enum AttrValueType {
    INT8 = 5,  INT16 = 7,  INT32 = 1,  INT64  = 3,
    UINT8 = 6, UINT16 = 8, UINT32 = 2, UINT64 = 4,
    STRING = 9,
};
```

字段的 (名称, 类型) 在首次构建/插入时被锁定；后续插入必须保持一致。

### 构造一个 `AttributeSet`

```cpp
auto* category = new vsag::AttributeValue<std::string>();
category->name_ = "category";
category->GetValue() = { "electronics" };

auto* tags = new vsag::AttributeValue<std::string>();
tags->name_ = "tag";
tags->GetValue() = { "promo", "new" };       // 多值字段

auto* price = new vsag::AttributeValue<int32_t>();
price->name_ = "price";
price->GetValue() = { 899 };

vsag::AttributeSet set;
set.attrs_ = { category, tags, price };
```

`Attribute*` 的生命周期取决于承载该 `AttributeSet` 的 `Dataset` 的 `Owner(...)` 标志：

- `Owner(true)`（默认）：`DatasetImpl` 析构时会 `delete` 每个 `Attribute*` 并
  `delete[]` `AttributeSet` 数组，调用方**不要**再自行释放。
- `Owner(false)`（下文示例所用）：调用方保留所有权，需在 `Build` / `Add` 返回后自行释放
  `Attribute*`（以及若为堆分配的 `AttributeSet` 数组）。

同一个 dataset 请只选一种策略，避免双重释放或泄漏。

## 构建支持属性过滤的索引

把 `index_param.use_attribute_filter` 设为 `true`，可选地在 `attr_params` 下调整属性
倒排索引参数。

```cpp
std::string build_params = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "use_attribute_filter": true,
        "attr_params": {
            "has_buckets": false
        }
    }
}
)";
auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
```

`has_buckets` 控制倒排索引中倒排链的存储布局，不同索引的默认值不同：

| 索引       | `has_buckets` 默认 |
|------------|:------------------:|
| HGraph     |       `false`      |
| IVF        |       `true`       |
| BruteForce |       `true`       |

如果没有性能数据明确指向需要修改，建议保留默认值。

## 在 Build / Add 时附加属性

`Dataset::AttributeSets` 接收一个长度等于向量数的 `AttributeSet` 数组
（`include/vsag/dataset.h`）：

```cpp
std::vector<vsag::AttributeSet> sets(num_vectors);
for (int64_t i = 0; i < num_vectors; ++i) {
    sets[i] = build_attrs_for_row(i);
}

auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)
    ->Dim(dim)
    ->Ids(ids)
    ->Float32Vectors(vectors)
    ->AttributeSets(sets.data())
    ->Owner(false);

index->Build(base);     // 或 index->Add(base)
```

## 通过 `SearchRequest` 查询

属性过滤目前仅通过 `SearchWithRequest` 暴露
（`include/vsag/search_request.h`）：

```cpp
vsag::SearchRequest req;
req.query_                    = query;
req.mode_                     = vsag::SearchMode::KNN_SEARCH;
req.topk_                     = 10;
req.params_str_               = R"({ "hgraph": { "ef_search": 200 } })";
req.enable_attribute_filter_  = true;
req.attribute_filter_str_     =
    "category = \"electronics\" AND price <= 1000 "
    "AND multi_in(tag, \"promo|new\", \"|\")";

auto result = index->SearchWithRequest(req).value();
for (int64_t i = 0; i < result->GetDim(); ++i) {
    std::cout << result->GetIds()[i] << " " << result->GetDistances()[i] << "\n";
}
```

可同时启用 `enable_filter_`（提供 `FilterPtr`）和 `enable_bitset_filter_`（提供 `BitsetPtr`），
所有启用的过滤项按**逻辑 AND**组合。

## 过滤表达式语法

文法定义见 `src/attr/grammar/FC.g4`。语法虽然紧凑，但已经能覆盖结构化过滤的常见需求。

### 逻辑运算符

| 形式  | 别名                  |
|-------|-----------------------|
| AND   | `AND`、`and`、`&&`    |
| OR    | `OR`、`or`、`\|\|`    |
| NOT   | `!(expr)`             |
| 分组  | `(...)`               |

`NOT` 仅支持前缀写法 `!(...)`。

### 比较运算符

数值字段：`=`、`!=`、`>`、`<`、`>=`、`<=`。
字符串字段：仅 `=` 和 `!=`。

数值比较的左侧可以包含算术运算（`+ - * /`）：

```
(price - discount) <= 100
```

### 列表成员判定

提供两种写法。它们使用**同一组关键字**（`IN` 与 `NOT_IN`，含下方别名），但**参数形态不同**。

**方括号中缀形式**——配合字面量列表使用：

```
id IN [1, 2, 3, 4]
category NOT_IN ["electronics", "clothing"]
```

列表元素必须是 `INTEGER` 字面量或**双引号**字符串；文法不接受单引号。

**函数式竖线形式**——上游已经把候选值拼接成字符串时使用。第二个参数必须是单个用 `|`
分隔的字符串字面量，第三个（可选）参数是分隔符，必须为 `"|"`：

```
multi_in(category, "electronics|clothing", "|")
multi_notin(uid, "1961|8669|9090", "|")
```

函数形式**不接受**方括号列表（`multi_in(field, [...])` 是文法错误）；中缀形式也**不接受**
竖线分隔字符串。

两种形式的别名：`IN` / `in` / `MULTI_IN` / `multi_in`，
`NOT_IN` / `not_in` / `NOTIN` / `notin` / `MULTI_NOTIN` / `multi_notin`。

对多值字段而言，只要其中**任一个值**出现在列表中，成员谓词即为真。

### 字面量

| 类型  | 示例                          |
|-------|-------------------------------|
| 整数  | `42`、`-7`                    |
| 浮点  | `3.14`、`1.5e-3`              |
| 字符串 | `"electronics"`、`"new"`（始终双引号） |
| 引号包裹的整型字符串 | `"123"`（在 `multi_in` 中按字符串处理） |

标识符匹配 `[a-zA-Z_][a-zA-Z0-9_]*`，可以含 `.`（即 `namespace.field` 视为同一个标识符）。

注释以 `#` 开头，到行尾。

### 示例

```sql
# 等值
category = "electronics"

# 数值范围 + 多值字段
price >= 100 AND price <= 1000 AND tag IN ["promo", "new"]

# 取反
!(status = "archived") AND multi_notin(region, "us-east|us-west", "|")

# 比较左侧的算术运算
(end_ts - start_ts) > 3600 AND charge_type = 5
```

## 更新属性

调用 `index->UpdateAttribute(id, new_attrs)`（或同时传入旧属性的重载，可让倒排索引更新更高效）：

```cpp
vsag::AttributeSet new_attrs = build_new_attrs();
auto status = index->UpdateAttribute(/*id=*/123, new_attrs);
```

向量本身不会改变，只更新倒排索引，后续搜索立即可见新属性值。

## 性能要点

- 属性倒排索引的内存占用大致与「字段平均值数量 × 向量数」成正比；字符串字段还要额外占用
  与「不同值数量」成正比的字典空间。
- 谓词越严格，候选越早被剪除，搜索越快；不严格的谓词大致等于无过滤搜索的成本加一个常数开销。
- 对图索引，谓词非常严格时应同步增大 `ef_search`，否则可能因存活候选不足而无法收敛。
- 优先使用 `multi_in` / `IN`，避免冗长的 `OR` 链——倒排索引可以一次扫描完成成员判定。

## 测试用例参考

最完整的使用示例在测试套件中：

- `tests/test_index.cpp` 中的 `TestIndex::TestWithAttr`：构建属性、用 `SearchRequest` 查询，
  以及 `UpdateAttribute` 后再次查询。
- `tests/fixtures/data/vector_generator.cpp` 中的 `generate_attributes`：演示如何按程序化方式
  构造混合类型的 `AttributeSet*` 数组。
- `src/attr/expression_visitor_test.cpp`：穷举式的语法用例，可作为 DSL 的参考实现。

## Python 状态

属性 / 混合搜索 API 目前仅 C++ 可用，`pyvsag` 暂未提供绑定，
`examples/python/todo_examples/301_feature_filter.py` 是一个空占位文件。
