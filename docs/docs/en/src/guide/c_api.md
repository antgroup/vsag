# C API

VSAG exports a C ABI in `include/vsag/vsag_c_api.h`. It wraps the same index implementations as
the C++ API and is useful for language bindings and applications that need opaque handles and
C-style functions.

The current installed header is not directly includable from a C translation unit: it contains an
unguarded `extern "C"` block and uses C++ `bool`. Compile the translation unit that includes this
header as C++ (for example, use a `.cpp` file). If the rest of the application is written in C,
expose the required operations through a small application-owned C wrapper.

## Lifecycle

Include the installed header from a C++ translation unit and link the normal VSAG library:

```cpp
#include <vsag/vsag_c_api.h>
```

The basic lifecycle is:

1. Create an opaque `vsag_index_t` with `vsag_index_factory`.
2. Build or train/add vectors.
3. Search, update, serialize, or query the index.
4. Release the handle with `vsag_index_destroy`.

`vsag_index_factory` returns `nullptr` when creation fails. Other functions return `Error_t`;
`code == VSAG_SUCCESS` indicates success and `message` contains error details otherwise.

```cpp
const char* build_params =
    "{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":128,"
    "\"index_param\":{\"base_quantization_type\":\"fp32\","
    "\"max_degree\":32,\"ef_construction\":200}}";

vsag_index_t index = vsag_index_factory("hgraph", build_params);
if (index == nullptr) {
    /* handle creation failure */
}

Error_t error = vsag_index_build(index, base, ids, 128, count);
if (error.code != VSAG_SUCCESS) {
    /* inspect error.message */
}

vsag_index_destroy(index);
```

## Search result buffers

The caller owns the `ids` and `dists` buffers in `SearchResult_t`. Allocate at least `k` entries
for both buffers before KNN or range search. VSAG writes no more than `k` results and stores the
actual number in `count`.

```cpp
int64_t result_ids[10];
float result_dists[10];
SearchResult_t result{};
result.dists = result_dists;
result.ids = result_ids;

Error_t error =
    vsag_index_knn_search(index, query, 128, 10, "{\"hgraph\":{\"ef_search\":100}}", &result);
```

The filter variants accept a `FilterFunc_t`. Returning `true` means the ID is valid and may be
returned.

## Range search result cap

Both C range-search functions require an explicit positive result cap:

```cpp
Error_t
vsag_index_range_search(vsag_index_t index,
                        const float* query,
                        uint64_t dim,
                        float radius,
                        int64_t k,
                        const char* parameters,
                        SearchResult_t* result);
```

`k` must be at least `1`. It is forwarded to the C++ range-search `limited_size` argument and
also protects the caller-owned buffers: `result.count` never exceeds `k`. Applications upgrading
from the older signature must add this argument to both `vsag_index_range_search` and
`vsag_index_range_search_with_filter`.

## API groups

- **Create and destroy:** `vsag_index_factory`, `vsag_index_destroy`.
- **Populate:** `vsag_index_build`, `vsag_index_train`, `vsag_index_add`.
- **Search:** `vsag_index_knn_search`, `vsag_index_range_search`, and their filter variants.
- **Read and update:** `vsag_index_calculate_distance_by_ids`,
  `vsag_index_get_vector_by_ids`, `vsag_index_update_ids`, `vsag_index_update_vector`, and
  `vsag_index_update_vector_force`.
- **Copy and reuse:** `vsag_index_clone`, `vsag_index_export_model`.
- **Persistence:** `vsag_serialize_file`, `vsag_deserialize_file`,
  `vsag_serialize_write_func`, and `vsag_deserialize_read_func`.

Support still depends on the selected index. An unsupported operation returns
`VSAG_UNSUPPORTED_INDEX_OPERATION`.

## Persistence errors

`vsag_serialize_file` and `vsag_deserialize_file` validate that the target file can be opened.
Check the returned `Error_t` for `VSAG_READ_ERROR`, `VSAG_MISSING_FILE`, or another non-success
code instead of assuming a file operation completed.

The callback-based persistence functions are available when the application manages its own
storage. Their callback offsets and sizes use the `OffsetType` and `SizeType` aliases declared in
the header.

## Reference

`include/vsag/vsag_c_api.h` is the complete source-level reference for signatures, error codes,
callbacks, and structures. Search parameters use the same JSON objects documented on the
corresponding [index pages](../indexes/README.md).
