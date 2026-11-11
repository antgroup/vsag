# C API

VSAG exports a C ABI in `include/vsag/vsag_c_api.h`. It wraps the same index implementations as
the C++ API and is the base for language bindings (`pyvsag` is built on it) and for applications
that want opaque handles and plain C functions. The header can be included from C and C++.

`libvsag_c` (shared library, built with `ENABLE_PYBINDS=ON`) exports only `vsag_*` symbols.
The C API is versioned with `VSAG_C_API_VERSION`; a binding should compare it with
`vsag_c_api_version()` when it loads the library.

## Conventions

- **Opaque handles:** `vsag_index_t`, `vsag_dataset_t`, `vsag_filter_t`, `vsag_search_result_t`
  and `vsag_error_t` are opaque. Each has exactly one `_destroy` function, and destroying `NULL`
  is a no-op.
- **Errors:** every fallible function returns a `vsag_error_t*`: `NULL` on success, otherwise an
  error object the caller reads with `vsag_error_code` and `vsag_error_message` and releases with
  `vsag_error_destroy`. No exception crosses the boundary and no error state is kept per thread.
- **Borrowed buffers:** vectors, ids and filters are never copied or freed by the library. A
  buffer passed to a call only has to stay valid for that call; a buffer attached to a dataset or
  bitmap filter has to stay valid while that handle is in use.
- **Returned pointers:** pointers read from a handle stay valid until that handle is destroyed.
- **Types:** signatures use fixed-width integers only. Flags are `int32_t` (0 is false).
- **Parameters:** index and search parameters are JSON strings, as in the C++ API.
- **Threads:** an index can be searched concurrently. A dataset, filter or search result may be
  read concurrently but must not be modified or destroyed while another thread uses it.

## Lifecycle

```c
#include <vsag/vsag_c_api.h>

const char* params =
    "{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":128,"
    "\"index_param\":{\"base_quantization_type\":\"fp32\","
    "\"max_degree\":32,\"ef_construction\":200}}";

vsag_index_t* index = NULL;
vsag_error_t* error = vsag_index_create("hgraph", params, &index);
if (error != NULL) {
    /* vsag_error_code(error), vsag_error_message(error) */
    vsag_error_destroy(error);
    return;
}

vsag_dataset_t* base = NULL;
vsag_dataset_create(&base);
vsag_dataset_set_num_elements(base, count);
vsag_dataset_set_dim(base, 128);
vsag_dataset_set_ids(base, ids);
vsag_dataset_set_float32_vectors(base, vectors);
error = vsag_index_build(index, base);
vsag_dataset_destroy(base);

vsag_index_destroy(index);
```

## Datasets

A dataset is a view over caller-owned memory and describes both base data and queries. Set the
element count, then the vectors in one of these forms:

- `vsag_dataset_set_float32_vectors`, `vsag_dataset_set_float16_vectors` (also used for bfloat16
  as raw `uint16` bits) and `vsag_dataset_set_int8_vectors`, with `vsag_dataset_set_dim`.
- `vsag_dataset_set_sparse_vectors`: CSR rows, where row `i` owns
  `indices[index_pointers[i] .. index_pointers[i + 1])` and the matching values.

`vsag_dataset_set_ids` is required for build and add. Dense and sparse data go through the same
`vsag_index_build`, `vsag_index_add` and `vsag_index_knn_search` functions.

## Search

`vsag_index_knn_search` and `vsag_index_range_search` take a single-query dataset and return a
`vsag_search_result_t`. Read it with `vsag_search_result_count`, `vsag_search_result_ids`,
`vsag_search_result_distances` and `vsag_search_result_statistics` (a JSON string), then release
it with `vsag_search_result_destroy`. The pointers can be wrapped without copying.
`vsag_index_range_search` takes a `limit` that caps the number of results; `-1` means no cap.

For dense float32 queries, `vsag_index_knn_search_float32` is the low-overhead path: one call, no
handles, and caller-allocated `out_ids`/`out_distances` buffers of `num_queries * k` entries.

## Filters

Filters are plain data, so no call crosses the language boundary per candidate:

- `vsag_filter_create_id_list`: keep (`VSAG_FILTER_ALLOW`) or drop (`VSAG_FILTER_DENY`) a list of ids.
- `vsag_filter_create_bitmap`: keep id `i` when bit `i` of a caller-owned bitmap is set.
- `vsag_filter_create_id_range`: keep ids in an inclusive range.

Pass a filter, or `NULL`, to the search functions. A per-candidate callback is available separately for hosts (see Callback filters), but `pyvsag`
does not expose it: it would cost a GIL acquisition per candidate.

## Logging

`vsag_set_log_level` sets the minimum level (`VSAG_LOG_*`). A host can route library log records
to its own logger with `vsag_set_logger(log_fn, user_data)`; passing `NULL` restores the built-in
logger. The callback may be invoked concurrently from any thread, including threads created by the
library, so it must be thread-safe, must not throw or unwind into the library, and must not keep
the message pointer after it returns. `user_data` must stay valid until the logger is replaced. The first call with a non-`NULL` callback
installs a forwarding logger and should happen before other threads log; later calls only swap the
callback and are safe while other threads log.

## API groups

- **Create and destroy:** `vsag_index_create`, `vsag_index_destroy`.
- **Populate:** `vsag_index_build`, `vsag_index_train`, `vsag_index_add`.
- **Modify:** `vsag_index_remove`, `vsag_index_update_ids`, `vsag_index_update_vector`.
- **Search:** `vsag_index_knn_search`, `vsag_index_range_search`, `vsag_index_knn_search_float32`.
- **Read:** `vsag_index_calc_distances_by_ids`, `vsag_index_get_vectors_by_ids`,
  `vsag_index_get_num_elements`, `vsag_index_get_memory_usage`, `vsag_index_check_id_exist` and
  `vsag_index_get_min_max_id`.
- **Copy and reuse:** `vsag_index_clone`, `vsag_index_export_model`.
- **Persistence:** `vsag_index_serialize_file`, `vsag_index_deserialize_file`, and, for hosts that
  keep index bytes in their own storage, `vsag_index_serialize_writer` and
  `vsag_index_deserialize_reader`.

Support still depends on the selected index. An unsupported operation returns
`VSAG_UNSUPPORTED_INDEX_OPERATION`. Check the returned error code for `VSAG_READ_ERROR`,
`VSAG_MISSING_FILE`, or another non-success value instead of assuming a file operation completed.

## Host storage callbacks

`vsag_index_serialize_writer(index, write_fn, user_data)` passes the serialized bytes to
`write_fn(user_data, offset, size, data)`. `vsag_index_deserialize_reader(index, read_fn, size_fn,
user_data)` reads them back through `read_fn(user_data, offset, size, dest)` and
`size_fn(user_data)`; both callbacks are required because deserialization seeks to the end of the input. The
callbacks return `0` on success and any other value on failure, which makes the call return an
error: `VSAG_INTERNAL_ERROR` for a failing write callback (there is no dedicated write code) and
`VSAG_READ_ERROR` for a failing read callback. They run synchronously on the calling thread, must not throw or unwind into the library,
and must not keep the pointers they receive. `user_data` only has to stay valid for the call.

## Host allocator and thread pool

An engine creates indexes that use memory and threads supplied by the host:

```c
vsag_engine_options_t* options = NULL;
vsag_engine_options_create(&options);
vsag_engine_options_set_allocator(options, "host", allocate, deallocate, reallocate, host);
vsag_engine_options_set_thread_pool(options, submit, host);

vsag_engine_t* engine = NULL;
vsag_engine_create(options, &engine);   /* options may be NULL for the defaults */
vsag_engine_options_destroy(options);   /* the engine copies what it needs */

vsag_index_t* index = NULL;
vsag_engine_create_index(engine, "hgraph", params, &index);
vsag_engine_destroy(engine);            /* indexes keep the resources alive */
```

The allocator callbacks mirror `malloc`, `free` and `realloc`; returning `NULL` for a non-zero size is
reported to the caller as `VSAG_NO_ENOUGH_MEMORY`. `submit(user_data, task,
task_data)` must run `task(task_data)` exactly once on any thread and return `0`, or return a
non-zero value to decline, in which case the library runs the task on the calling thread. The pool
is owned by the host, so the library never resizes it. All of these callbacks may run concurrently
from any thread, must not throw or unwind into the library, and `user_data` has to stay valid until
the engine and every index created from it are destroyed.

## Callback filters

`vsag_filter_create_callback(filter_fn, user_data, &filter)` is the host-side alternative to the
data filters: `filter_fn(user_data, id)` returns non-zero to keep an id. It is called for every
candidate a search considers, possibly concurrently, so it must be thread-safe and cheap and must
not call back into VSAG. Prefer a data filter whenever the valid ids are known up front.

## Reference

`include/vsag/vsag_c_api.h` is the complete source-level reference for signatures and error
codes. Search parameters use the same JSON objects documented on the corresponding
[index pages](../indexes/).
