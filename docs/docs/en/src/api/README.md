# API Reference

This chapter is a curated reference for VSAG's **public C++ API** — the headers installed under
`include/vsag/`. It documents the classes, structs, enums, and free functions an application links
against, grouped by responsibility. The installed headers remain the authoritative source of
truth; the pages here explain intent, ownership, and how the pieces fit together.

> Looking for how to *configure* an index (the JSON `index_param` / search keys)? That is covered
> in [Index Parameters](../resources/index_parameters.md) and each [index page](../indexes/README.md).
> This chapter covers the *code* surface (types and methods), not the JSON schema.

## Include and namespace

A single umbrella header pulls in the whole public API, and every symbol lives in the `vsag`
namespace:

```cpp
#include <vsag/vsag.h>   // includes factory.h, index.h, dataset.h, engine.h, ...

int main() {
    vsag::init();                       // one-time process initialization
    std::string ver = vsag::version();  // git-derived version string
}
```

| Free function | Header | Description |
|---------------|--------|-------------|
| `bool vsag::init()` | `vsag/vsag.h` | Initializes the library. Call once before other APIs. Always returns `true`. |
| `std::string vsag::version()` | `vsag/vsag.h` | Returns the build version derived from the git revision. |

## Error-handling model

Almost every fallible call returns `tl::expected<T, Error>` (a `std::expected`-style type shipped in
`vsag/expected.hpp`) instead of throwing. A handful of legacy statistics accessors still throw
`std::runtime_error` when unsupported; those are called out on the [Index](index_class.md) page.

```cpp
auto result = vsag::Factory::CreateIndex("hgraph", params);
if (not result.has_value()) {
    const vsag::Error& err = result.error();
    std::cerr << "create failed: " << static_cast<int>(err.type) << " " << err.message << "\n";
    return;
}
std::shared_ptr<vsag::Index> index = result.value();
```

`Error` carries a machine-readable `type` and a human-readable `message`:

```cpp
struct Error {
    ErrorType type;
    std::string message;
};
```

### `ErrorType`

Defined in `vsag/errors.h`. Values start at `1` (`0` is reserved).

| Category | Value | Meaning |
|----------|-------|---------|
| Common | `UNKNOWN_ERROR` | Unknown error. |
| Common | `INTERNAL_ERROR` | Internal algorithm error. |
| Common | `INVALID_ARGUMENT` | An argument was invalid. |
| Behavior | `WRONG_STATUS` | Index is in the wrong state for the call. |
| Behavior | `BUILD_TWICE` | The index was already built and cannot be built again. |
| Behavior | `INDEX_NOT_EMPTY` | Deserializing onto a non-empty index. |
| Behavior | `UNSUPPORTED_INDEX` | Requested an index type that does not exist. |
| Behavior | `UNSUPPORTED_INDEX_OPERATION` | This index does not implement the called method. |
| Behavior | `DIMENSION_NOT_EQUAL` | Request dimension differs from the index dimension. |
| Behavior | `INDEX_EMPTY` | Index is empty; cannot search or serialize. |
| Runtime | `NO_ENOUGH_MEMORY` | Memory allocation failed. |
| Runtime | `READ_ERROR` | Failed to read from a binary. |
| Runtime | `MISSING_FILE` | A required file is missing (e.g. DiskANN deserialization). |
| Runtime | `INVALID_BINARY` | Serialized binary content is invalid. |

Because most index methods are `virtual` with a default body that returns
`UNSUPPORTED_INDEX_OPERATION`, an "unsupported" result is normal and expected: it means the concrete
index does not implement that optional capability. Use
[`Index::CheckFeature`](index_class.md#checkfeature) to probe support ahead of time.

## Header map

| Header | Primary symbols | Reference page |
|--------|-----------------|----------------|
| `factory.h`, `engine.h`, `vsag.h` | `Factory`, `Engine`, `init`, `version` | [Factory & Engine](factory_engine.md) |
| `index.h` | `Index`, `IndexType`, `RemoveMode`, `MergeUnit` | [Index](index_class.md) |
| `dataset.h` | `Dataset`, `SparseVector`, `MultiVector` | [Dataset](dataset.md) |
| `search_request.h`, `filter.h`, `bitset.h`, `search_param.h`, `iterator_context.h` | `SearchRequest`, `Filter`, `Bitset` | [Search Request & Filters](search.md) |
| `binaryset.h`, `readerset.h` | `BinarySet`, `Binary`, `Reader`, `ReaderSet` | [Serialization Types](serialization.md) |
| `resource.h`, `allocator.h`, `thread_pool.h`, `options.h`, `logger.h` | `Resource`, `Allocator`, `ThreadPool`, `Options`, `Logger` | [Resource Management](resource.md) |
| `attribute.h`, `index_features.h`, `index_detail_info.h`, `utils.h`, `constants.h` | `Attribute`, `IndexFeature`, `IndexDetailInfo` | [Auxiliary Types](types.md) |

## In this chapter

- [Factory & Engine](factory_engine.md) — create indexes and readers; own resources with `Engine`.
- [Index](index_class.md) — the core index interface: build, search, update, serialize, inspect.
- [Dataset](dataset.md) — the builder-pattern container for vectors, ids, and metadata.
- [Search Request & Filters](search.md) — `SearchRequest`, `Filter`, `Bitset`, iterator context.
- [Serialization Types](serialization.md) — `BinarySet` / `Binary` and `Reader` / `ReaderSet`.
- [Resource Management](resource.md) — allocator, thread pool, engine resources, options, logger.
- [Auxiliary Types](types.md) — attributes, feature flags, index detail info, and utility helpers.
