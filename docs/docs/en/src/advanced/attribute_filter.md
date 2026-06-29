# Attribute Filter (Hybrid Search)

Attribute filtering — sometimes called *hybrid search* or *filtered ANN with structured
predicates* — restricts a `KnnSearch` / `RangeSearch` to vectors whose structured tags
satisfy an SQL-like expression. Compared to the id-based filters in
[Filtered Search](filtered_search.md), it lets you express predicates like:

```
category = "electronics" AND price <= 1000 AND multi_in(tag, "promo|new", "|")
```

without writing a callback. VSAG builds an attribute inverted index alongside the vector
index; the predicate is parsed once and evaluated during graph traversal, so candidates that
cannot satisfy the predicate are pruned early.

> "Hybrid search" on this page means **vector + structured attributes** (not a
> storage-layout hybrid).

## When to Use Each Filter API

| You want to … | Use |
|---|---|
| Exclude a known set of ids (e.g. tombstones) | [Bitset / function filter](filtered_search.md) |
| Run user-defined logic over an id | [`Filter` object](filtered_search.md) |
| Filter on opaque per-vector bytes inside the graph | [Extra Info](extra_info.md) |
| Filter on **named, typed fields** with AND/OR/IN | **This page** |

All three can be combined inside a single `SearchRequest`; they are ANDed together.

## Index Support

| Index            | Build with `use_attribute_filter` | `SearchWithRequest` + attribute string | `UpdateAttribute` |
|------------------|:---------------------------------:|:--------------------------------------:|:-----------------:|
| HGraph           |                Yes                |                   Yes                  |        Yes        |
| IVF              |                Yes                |                   Yes                  |        Yes        |
| BruteForce       |                Yes                |                   Yes                  |        Yes        |
| WARP (sparse)    |                Yes                |                   Yes                  |        Yes        |
| SINDI / Pyramid | — | id-based filters only (see [Filtered Search](filtered_search.md)) | — |

When `use_attribute_filter` is enabled, BruteForce currently rejects `Remove` calls
(re-add the index to delete entries).

## Attribute Data Model

Attributes are defined per vector and grouped into an `AttributeSet`
(`include/vsag/attribute.h`). Each attribute has:

- a **name** (string),
- a **value type** (`AttrValueType` enum),
- a **list of values** — every field is multi-valued by design, so `IN`-style membership
  works naturally for tag-like fields.

Supported value types:

```cpp
enum AttrValueType {
    INT8 = 5,  INT16 = 7,  INT32 = 1,  INT64  = 3,
    UINT8 = 6, UINT16 = 8, UINT32 = 2, UINT64 = 4,
    STRING = 9,
};
```

The schema is auto-discovered from the first build/add: the (name, type) pair seen for each
field is locked. Subsequent inserts must match.

### Building an `AttributeSet`

```cpp
auto* category = new vsag::AttributeValue<std::string>();
category->name_ = "category";
category->GetValue() = { "electronics" };

auto* tags = new vsag::AttributeValue<std::string>();
tags->name_ = "tag";
tags->GetValue() = { "promo", "new" };       // multi-valued

auto* price = new vsag::AttributeValue<int32_t>();
price->name_ = "price";
price->GetValue() = { 899 };

vsag::AttributeSet set;
set.attrs_ = { category, tags, price };
```

Lifetime of the `Attribute*` entries depends on the `Dataset::Owner(...)` flag passed to the
dataset that carries the `AttributeSet`:

- `Owner(true)` (the default): `DatasetImpl`'s destructor will `delete` each `Attribute*` and
  `delete[]` the `AttributeSet` array; do **not** free them yourself.
- `Owner(false)` (used in the example below): the caller retains ownership and must free the
  `Attribute*` entries (and the `AttributeSet` array, if heap-allocated) after `Build`/`Add`
  returns.

Pick one and stick with it for a given dataset to avoid double-free or leaks.

## Building an Index with Attribute Support

Set `index_param.use_attribute_filter` to `true` and (optionally) tune the
attribute-inverted-index parameters under `attr_params`.

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

`has_buckets` controls how the inverted index lays out posting lists. Defaults differ by
index:

| Index      | Default `has_buckets` |
|------------|:---------------------:|
| HGraph     |        `false`        |
| IVF        |        `true`         |
| BruteForce |        `true`         |

Leave the defaults unless profiling indicates otherwise.

## Attaching Attributes During Build / Add

`Dataset::AttributeSets` accepts a contiguous array of `AttributeSet`, one per vector
(`include/vsag/dataset.h`):

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

index->Build(base);     // or index->Add(base)
```

## Querying with `SearchRequest`

Attribute filtering is only exposed via `SearchWithRequest`
(`include/vsag/search_request.h`):

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

You can simultaneously enable `enable_filter_` (with a `FilterPtr`) and
`enable_bitset_filter_` (with a `BitsetPtr`); all enabled filters are combined with
**AND**.

## Filter Expression Language

The expression grammar is defined in `src/attr/grammar/FC.g4`. It is small but covers the
common needs of structured filtering.

### Logical operators

| Form        | Aliases                |
|-------------|------------------------|
| AND         | `AND`, `and`, `&&`     |
| OR          | `OR`,  `or`,  `\|\|`   |
| NOT         | `!(expr)`              |
| Grouping    | `(...)`                |

`NOT` is only available in the prefixed form `!(...)`.

### Comparison operators

For numeric fields: `=`, `!=`, `>`, `<`, `>=`, `<=`.
For string fields: only `=` and `!=`.

Numeric comparands may include arithmetic (`+`, `-`, `*`, `/`):

```
(price - discount) <= 100
```

### List membership

Two forms are supported. They use the **same set of keywords** (`IN` and `NOT_IN`, with the
aliases listed below) but **different argument shapes**.

**Infix bracket form** — use this with a literal list:

```
id IN [1, 2, 3, 4]
category NOT_IN ["electronics", "clothing"]
```

The list members must be `INTEGER` literals or **double-quoted** strings. Single quotes are
not accepted by the grammar.

**Function pipe form** — use this when the candidate values are produced by string
concatenation upstream. The second argument must be a single pipe-delimited string literal,
and the third (optional) argument is the separator and must be `"|"`:

```
multi_in(category, "electronics|clothing", "|")
multi_notin(uid, "1961|8669|9090", "|")
```

Bracket lists are **not** accepted in the function form (`multi_in(field, [...])` is a
syntax error). Pipe strings are **not** accepted in the infix form.

Aliases for both forms: `IN` / `in` / `MULTI_IN` / `multi_in`,
`NOT_IN` / `not_in` / `NOTIN` / `notin` / `MULTI_NOTIN` / `multi_notin`.

A field with multiple values matches the membership predicate if **any** of its values is
contained in the literal list.

### Literals

| Kind   | Examples                       |
|--------|--------------------------------|
| Integer | `42`, `-7`                    |
| Float   | `3.14`, `1.5e-3`              |
| String  | `"electronics"`, `"new"` (always double-quoted) |
| Quoted integer (string) | `"123"` (treated as a string in `multi_in`) |

Identifiers match `[a-zA-Z_][a-zA-Z0-9_]*` and may contain dots
(`namespace.field` is one identifier).

Comments start with `#` and run to end of line.

### Examples

```sql
# simple equality
category = "electronics"

# numeric range, multi-valued field
price >= 100 AND price <= 1000 AND tag IN ["promo", "new"]

# negation
!(status = "archived") AND multi_notin(region, "us-east|us-west", "|")

# arithmetic on the left side of the comparison
(end_ts - start_ts) > 3600 AND charge_type = 5
```

## Updating Attributes

Use `index->UpdateAttribute(id, new_attrs)` (or the overload that also takes the previous
attribute set for cheaper inverted-index updates):

```cpp
vsag::AttributeSet new_attrs = build_new_attrs();
auto status = index->UpdateAttribute(/*id=*/123, new_attrs);
```

The vector itself is unchanged; only the inverted index is updated. Subsequent searches see
the new attribute values immediately.

## Performance Notes

- The attribute inverted index adds memory roughly proportional to the average number of
  values per field times the number of vectors. For string fields, the dictionary cost is
  proportional to the number of distinct values.
- Highly selective predicates accelerate search (more candidates pruned early); very
  unselective predicates approach the cost of unfiltered search plus a constant overhead.
- For graph indexes, increase `ef_search` when predicates are very selective so the search
  has enough surviving candidates to converge.
- Use `multi_in` / `IN` instead of long `OR` chains; the inverted index can resolve list
  membership in a single pass.

## Tests as Reference

The most complete usage sample lives in the test suite:

- `tests/test_index.cpp` — `TestIndex::TestWithAttr` (build attributes, search via
  `SearchRequest`, then `UpdateAttribute` and re-search).
- `tests/fixtures/data/vector_generator.cpp` — `generate_attributes` shows how to construct
  `AttributeSet*` arrays of mixed types programmatically.
- `src/attr/expression_visitor_test.cpp` — exhaustive grammar coverage; useful as a working
  reference for the DSL.

## Python Status

The attribute / hybrid-search API is currently C++-only. There is no `pyvsag` binding yet,
and the placeholder example at `examples/python/todo_examples/301_feature_filter.py` is
intentionally empty.
