# 用户自定义 IO

用户自定义 IO（`user_defined_io`）允许调用方注入自己的读、写、扩容回调，让 VSAG 在构建索引时将精确码直接写入外部存储，无需额外的序列化步骤。当精确码需要存放在 VSAG 原生不支持的受管存储（如分布式文件系统、对象存储或自定义块设备）中时，此功能非常有用。

## 支持的配置

- **索引**：HGraph（需启用 `use_reorder: true`）。
- **Base 编码**：`sq8` 置于 `block_memory_io`。
- **Precise 编码**：`fp32` 置于 `user_defined_io`。
- **图**：`block_memory_io`。

当前版本不支持其他图 / base / bucket 存储组合。

## 回调约定

需要提供三个同步回调：

| 回调 | 签名 | 职责 |
|------|------|------|
| Read | `void(uint64_t offset, uint64_t len, void* dest)` | 从 `offset` 处复制 `len` 字节到 `dest`。越界时抛出异常。 |
| Write | `void(uint64_t offset, uint64_t len, const void* src)` | 在 `offset` 处消费 `src` 的 `len` 字节。如果 `offset + len` 超出当前容量，需扩展底层存储。 |
| Resize | `void(uint64_t new_size)` | 将底层存储截断或扩展至 `new_size`，同时保留未截断的前缀。 |

约束：

- 三个回调均由互斥锁串行化，不得重入同一个适配器。
- 写入完成后须立即可供后续 Read 读取。
- `Factory::CreateUserDefinedIO` 的 `initial_size` 参数必须与现有存储一致（全新存储使用 `0`）。
- 失败须抛异常；部分写入不会回滚。
- 每个独立的 `user_defined_io` 实例需要独立的底层字节空间；同一存储用两个名字注册不会产生隔离。

## 示例

```cpp
auto store = std::make_shared<ByteStore>();

auto read_func = [store](uint64_t offset, uint64_t len, void* dest) {
    std::lock_guard lock(store->mutex);
    if (offset > store->bytes.size() || len > store->bytes.size() - offset)
        throw std::runtime_error("read out of bounds");
    if (len > 0)
        std::memcpy(dest, store->bytes.data() + offset, len);
    ++store->reads;
};

auto write_func = [store](uint64_t offset, uint64_t len, const void* src) {
    std::lock_guard lock(store->mutex);
    if (offset > UINT64_MAX - len)
        throw std::runtime_error("write overflow");
    if (offset + len > store->bytes.size())
        store->bytes.resize(offset + len);
    if (len > 0)
        std::memcpy(store->bytes.data() + offset, src, len);
    ++store->writes;
};

auto resize_func = [store](uint64_t size) {
    std::lock_guard lock(store->mutex);
    store->bytes.resize(size);
    ++store->resizes;
};

auto storage = vsag::Factory::CreateUserDefinedIO(read_func, write_func, resize_func, 0);

vsag::UserDefinedIOSet storages;
storages.Set("hgraph_precise", storage.reader, storage.writer);

auto index = engine.CreateIndex("hgraph", R"({
    "dtype": "float32", "metric_type": "l2", "dim": 128,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 26, "ef_construction": 100,
        "use_reorder": true,
        "precise_quantization_type": "fp32",
        "precise_io_type": "user_defined_io",
        "precise_user_defined_io": "hgraph_precise"
    }
})", storages).value();

index->Build(base);
auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 50}})");
```

可运行版本位于 `examples/cpp/200_user_defined_io.cpp`。

## 索引参数

| 参数 | 取值 | 说明 |
|------|------|------|
| `precise_io_type` | `"user_defined_io"` | 必填 |
| `precise_user_defined_io` | **字符串** | 必须与传入 `CreateIndex` 的 `UserDefinedIOSet` 中的名称匹配。非空、唯一。 |
| `use_reorder` | `true` | 必填 |
| `precise_quantization_type` | `"fp32"` | 必填 |
| `base_quantization_type` | `"sq8"` | 必填 |
| `graph_io_type` | `"block_memory_io"` | 隐式默认值 |
| `base_io_type` | `"block_memory_io"` | 隐式默认值 |

## 限制

- 图、base 和 bucket 存储必须保留在内存中。
- 不提供异步写、崩溃持久化、事务或经过验证的可写存储恢复能力。
- 未测量通过网络实时构建的性能；ODescent 可能发起大量小读取。
- 后端内存估算不计用户回调底层存储的驻留内存。

## 参见

- [Extensibility](extensibility.md) — 其他 VSAG 扩展点。
- [示例 `200_user_defined_io`](https://github.com/antgroup/vsag/blob/main/examples/cpp/200_user_defined_io.cpp)（可运行）。
- [`include/vsag/readerset.h`](https://github.com/antgroup/vsag/blob/main/include/vsag/readerset.h) — `UserDefinedIOSet` API。
- [磁盘索引最佳实践](../resources/disk_index.md) — 使用 `user_defined_io` 作为磁盘分层后端。