# C API

VSAG 在 `include/vsag/vsag_c_api.h` 中导出 C ABI。它封装了与 C++ API 相同的索引实现，是各语言
绑定（`pyvsag` 即基于它构建）的基础，也适合需要不透明句柄和纯 C 函数的应用。该头文件可同时被 C
和 C++ 包含。

`libvsag_c`（共享库，需 `ENABLE_PYBINDS=ON` 构建）只导出 `vsag_*` 符号。C API 通过
`VSAG_C_API_VERSION` 标识版本，绑定在加载库时应将其与 `vsag_c_api_version()` 比较。

## 约定

- **不透明句柄：** `vsag_index_t`、`vsag_dataset_t`、`vsag_filter_t`、`vsag_search_result_t` 和
  `vsag_error_t` 都是不透明类型，每种恰有一个 `_destroy` 函数，销毁 `NULL` 不做任何事。
- **错误：** 所有可能失败的函数返回 `vsag_error_t*`：成功为 `NULL`，否则返回错误对象，调用方用
  `vsag_error_code` 和 `vsag_error_message` 读取，并用 `vsag_error_destroy` 释放。异常不会越过
  边界，也不保存线程局部的错误状态。
- **借用缓冲区：** 向量、id 和过滤器数据不会被库复制或释放。传给某次调用的缓冲区只需在该调用期间
  有效；挂到 dataset 或 bitmap 过滤器上的缓冲区需在该句柄使用期间保持有效。
- **返回的指针：** 从句柄读取的指针在该句柄销毁前有效。
- **类型：** 签名中只使用定宽整数；标志位为 `int32_t`（0 表示假）。
- **参数：** 索引参数和搜索参数与 C++ API 一样是 JSON 字符串。
- **线程：** 索引可被并发搜索。dataset、过滤器和搜索结果可并发读取，但在其他线程使用时不得修改
  或销毁。

## 生命周期

```c
#include <vsag/vsag_c_api.h>

const char* params =
    "{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":128,"
    "\"index_param\":{\"base_quantization_type\":\"fp32\","
    "\"max_degree\":32,\"ef_construction\":200}}";

vsag_index_t* index = NULL;
vsag_error_t* error = vsag_index_create("hgraph", params, &index);
if (error != NULL) {
    /* vsag_error_code(error)、vsag_error_message(error) */
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

## Dataset

dataset 是对调用方内存的视图，既可描述底库数据，也可描述查询。先设置元素个数，再用下列形式之一
设置向量：

- `vsag_dataset_set_float32_vectors`、`vsag_dataset_set_float16_vectors`（bfloat16 也使用它，
  传入原始 `uint16` 位）和 `vsag_dataset_set_int8_vectors`，并配合 `vsag_dataset_set_dim`。
- `vsag_dataset_set_sparse_vectors`：CSR 行，第 `i` 行拥有
  `indices[index_pointers[i] .. index_pointers[i + 1])` 及对应的值。

构建和添加需要 `vsag_dataset_set_ids`。稠密与稀疏数据使用相同的 `vsag_index_build`、
`vsag_index_add` 和 `vsag_index_knn_search`。

## 搜索

`vsag_index_knn_search` 和 `vsag_index_range_search` 接收只含一条查询的 dataset，返回
`vsag_search_result_t`。用 `vsag_search_result_count`、`vsag_search_result_ids`、
`vsag_search_result_distances` 和 `vsag_search_result_statistics`（JSON 字符串）读取，再用
`vsag_search_result_destroy` 释放，指针可零拷贝包装。`vsag_index_range_search` 的 `limit` 限制
结果数量，`-1` 表示不限。

对于稠密 float32 查询，`vsag_index_knn_search_float32` 是低开销路径：一次调用、无需句柄，输出
缓冲区 `out_ids`/`out_distances` 由调用方分配，大小为 `num_queries * k`。

## 过滤器

过滤器是纯数据，因此每个候选点都不会跨越语言边界：

- `vsag_filter_create_id_list`：保留（`VSAG_FILTER_ALLOW`）或排除（`VSAG_FILTER_DENY`）一组 id。
- `vsag_filter_create_bitmap`：当调用方持有的 bitmap 第 `i` 位为 1 时保留 id `i`。
- `vsag_filter_create_id_range`：保留闭区间内的 id。

将过滤器（或 `NULL`）传给搜索函数。面向宿主的逐候选点回调另有提供（见“回调过滤器”），但 `pyvsag` 不暴露它：
每个候选点都会产生一次 GIL 获取。

## 日志

`vsag_set_log_level` 设置最低日志级别（`VSAG_LOG_*`）。宿主可通过
`vsag_set_logger(log_fn, user_data)` 将库的日志转发到自己的日志系统，传入 `NULL` 则恢复内置
日志器。回调可能在任意线程并发调用（包括库创建的线程），因此必须线程安全，不得抛出异常或展开
到库内，返回后不得保留消息指针。`user_data` 需在日志器被替换前保持有效。第一次传入非 `NULL` 回调时会安装转发日志器，应在其他线程
开始打日志之前完成；之后的调用只替换回调，在其他线程打日志时也是安全的。

## API 分组

- **创建与销毁：** `vsag_index_create`、`vsag_index_destroy`。
- **写入：** `vsag_index_build`、`vsag_index_train`、`vsag_index_add`。
- **修改：** `vsag_index_remove`、`vsag_index_update_ids`、`vsag_index_update_vector`。
- **搜索：** `vsag_index_knn_search`、`vsag_index_range_search`、`vsag_index_knn_search_float32`。
- **读取：** `vsag_index_calc_distances_by_ids`、`vsag_index_get_vectors_by_ids`、
  `vsag_index_get_num_elements`、`vsag_index_get_memory_usage`、`vsag_index_check_id_exist` 和
  `vsag_index_get_min_max_id`。
- **复制与复用：** `vsag_index_clone`、`vsag_index_export_model`。
- **持久化：** `vsag_index_serialize_file`、`vsag_index_deserialize_file`；对于把索引字节放在自有
  存储中的宿主，还有 `vsag_index_serialize_writer` 和 `vsag_index_deserialize_reader`。

是否支持仍取决于所选索引，不支持的操作返回 `VSAG_UNSUPPORTED_INDEX_OPERATION`。请检查返回的
错误码是否为 `VSAG_READ_ERROR`、`VSAG_MISSING_FILE` 等非成功值，不要假定文件操作一定完成。

## 宿主存储回调

`vsag_index_serialize_writer(index, write_fn, user_data)` 把序列化后的字节交给
`write_fn(user_data, offset, size, data)`。`vsag_index_deserialize_reader(index, read_fn,
size_fn, user_data)` 通过 `read_fn(user_data, offset, size, dest)` 和 `size_fn(user_data)` 读回；
两个回调都必须提供，因为反序列化会定位到输入的末尾。回调成功返回 `0`，其他值表示失败，此时调用
会返回错误：写回调失败返回 `VSAG_INTERNAL_ERROR`（目前没有专门的写错误码），读回调失败返回
`VSAG_READ_ERROR`。回调在调用线程上同步执行，不得抛出异常或展开到库内，也不得保留收到的指针。
`user_data` 只需在该调用期间保持有效。

## 宿主 allocator 与线程池

engine 创建的索引会使用宿主提供的内存和线程：

```c
vsag_engine_options_t* options = NULL;
vsag_engine_options_create(&options);
vsag_engine_options_set_allocator(options, "host", allocate, deallocate, reallocate, host);
vsag_engine_options_set_thread_pool(options, submit, host);

vsag_engine_t* engine = NULL;
vsag_engine_create(options, &engine);   /* options 可为 NULL，使用默认实现 */
vsag_engine_options_destroy(options);   /* engine 已复制所需内容 */

vsag_index_t* index = NULL;
vsag_engine_create_index(engine, "hgraph", params, &index);
vsag_engine_destroy(engine);            /* 索引会让这些资源继续存活 */
```

allocator 回调对应 `malloc`、`free` 和 `realloc`；对非零大小返回 `NULL` 时，调用方会收到
`VSAG_NO_ENOUGH_MEMORY`。`submit(user_data, task, task_data)` 必须在
任意线程上恰好执行一次 `task(task_data)` 并返回 `0`；返回非零表示拒绝，此时库会在调用线程上
执行该任务。线程池由宿主持有，库不会调整其大小。这些回调都可能在任意线程并发调用，不得抛出
异常或展开到库内，且 `user_data` 需在 engine 及其创建的所有索引销毁之前保持有效。

## 回调过滤器

`vsag_filter_create_callback(filter_fn, user_data, &filter)` 是数据型过滤器之外、面向宿主的选择：
`filter_fn(user_data, id)` 返回非零表示保留该 id。搜索考察的每个候选点都会调用它，且可能并发，
因此必须线程安全、足够轻量，并且不得回调 VSAG。只要事先知道有效 id，就优先使用数据型过滤器。

## 参考

`include/vsag/vsag_c_api.h` 是签名和错误码的完整源码参考。搜索参数使用对应
[索引页面](../indexes/)中记录的 JSON 对象。
