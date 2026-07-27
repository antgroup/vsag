# C API

VSAG 在 `include/vsag/vsag_c_api.h` 中导出 C ABI。它封装了与 C++ API 相同的索引实现，
适合需要不透明句柄和 C 风格函数的应用及语言绑定。

当前安装的头文件不能直接被 C 编译单元包含：其中使用了未加条件保护的 `extern "C"`
代码块和 C++ `bool`。请将包含该头文件的编译单元按 C++ 编译（例如使用 `.cpp` 文件）。
如果应用的其余部分使用 C，请在这个 C++ 编译单元中封装所需操作，再导出应用自己的
C 接口。

## 生命周期

在 C++ 编译单元中包含安装后的头文件，并链接普通 VSAG 库：

```cpp
#include <vsag/vsag_c_api.h>
```

基本生命周期如下：

1. 使用 `vsag_index_factory` 创建不透明的 `vsag_index_t`。
2. 构建索引，或先训练再增量添加向量。
3. 执行搜索、更新、序列化或读取操作。
4. 使用 `vsag_index_destroy` 释放句柄。

创建失败时，`vsag_index_factory` 返回 `nullptr`。其他函数返回 `Error_t`；
`code == VSAG_SUCCESS` 表示成功，否则可从 `message` 读取错误详情。

```cpp
const char* build_params =
    "{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":128,"
    "\"index_param\":{\"base_quantization_type\":\"fp32\","
    "\"max_degree\":32,\"ef_construction\":200}}";

vsag_index_t index = vsag_index_factory("hgraph", build_params);
if (index == nullptr) {
    /* 处理创建失败 */
}

Error_t error = vsag_index_build(index, base, ids, 128, count);
if (error.code != VSAG_SUCCESS) {
    /* 检查 error.message */
}

vsag_index_destroy(index);
```

## 搜索结果缓冲区

`SearchResult_t` 中的 `ids` 与 `dists` 缓冲区由调用方持有。执行 KNN 或范围
搜索前，两个缓冲区都至少分配 `k` 个元素。VSAG 最多写入 `k` 条结果，并把实际
数量写入 `count`。

```cpp
int64_t result_ids[10];
float result_dists[10];
SearchResult_t result{};
result.dists = result_dists;
result.ids = result_ids;

Error_t error =
    vsag_index_knn_search(index, query, 128, 10, "{\"hgraph\":{\"ef_search\":100}}", &result);
```

带过滤器的重载接收 `FilterFunc_t`。回调返回 `true` 表示该 ID 有效，可以出现
在结果中。

## 范围搜索结果上限

两个 C 范围搜索接口都要求显式传入正数结果上限：

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

`k` 必须至少为 `1`。它会传递给 C++ 范围搜索的 `limited_size` 参数，同时保护
调用方缓冲区：`result.count` 不会超过 `k`。从旧签名升级时，需要为
`vsag_index_range_search` 和 `vsag_index_range_search_with_filter` 同时补上该参数。

## 接口分组

- **创建与释放：** `vsag_index_factory`、`vsag_index_destroy`。
- **写入数据：** `vsag_index_build`、`vsag_index_train`、`vsag_index_add`。
- **搜索：** `vsag_index_knn_search`、`vsag_index_range_search` 及其过滤器重载。
- **读取与更新：** `vsag_index_calculate_distance_by_ids`、
  `vsag_index_get_vector_by_ids`、`vsag_index_update_ids`、`vsag_index_update_vector`
  及 `vsag_index_update_vector_force`。
- **复制与复用：** `vsag_index_clone`、`vsag_index_export_model`。
- **持久化：** `vsag_serialize_file`、`vsag_deserialize_file`、
  `vsag_serialize_write_func` 及 `vsag_deserialize_read_func`。

具体接口是否可用仍取决于索引类型。不支持的操作返回
`VSAG_UNSUPPORTED_INDEX_OPERATION`。

## 持久化错误

`vsag_serialize_file` 与 `vsag_deserialize_file` 会校验目标文件是否成功打开。
不要假定文件操作已经完成；应检查返回的 `Error_t` 是否为 `VSAG_READ_ERROR`、
`VSAG_MISSING_FILE` 或其他非成功状态。

应用自行管理存储时，可以使用基于回调的持久化接口。回调中的 offset 和 size
使用头文件里声明的 `OffsetType` 与 `SizeType`。

## 参考

`include/vsag/vsag_c_api.h` 是函数签名、错误码、回调和结构体的完整源码级
参考。搜索参数 JSON 与对应[索引页面](../indexes/README.md)中的 C++ API 相同。
