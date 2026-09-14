# 用户自定义 IO

用户自定义 IO（`user_defined_io`）允许调用方注入自己的读、写、扩容回调，让 VSAG 在构建索引时将精确码直接写入外部存储，无需额外的序列化步骤。当精确码需要存放在 VSAG 原生不支持的受管存储（如分布式文件系统、对象存储或自定义块设备）中时，此功能非常有用。

## 支持的配置

- **索引**：HGraph（需启用 `use_reorder: true`）。
- **Base 编码**：`sq8` 置于 `block_memory_io`。
- **Precise 编码**：`fp32` 置于 `user_defined_io`。
- **图**：`block_memory_io`。

当前版本不支持其他图 / base / bucket 存储组合。

## `precise_user_defined_io` 是什么

`precise_io_type` 决定 **HGraph 精确码**使用哪种 IO 后端；设置为
`"user_defined_io"` 时，`precise_user_defined_io` 则是用于查找 Reader/Writer 配对的**注册名**。
它不是文件路径、连接地址或另一种 IO 类型，也不会创建底层存储。名称只在本次
`CreateIndex` 调用传入的 `UserDefinedIOSet` 中解析，不是在进程范围内全局注册。

按以下顺序绑定：

1. 准备访问同一份底层字节空间的 Reader 和 Writer。可通过
   `Factory::CreateUserDefinedIO(read_func, write_func, resize_func, initial_size)`
   将同步回调封装成配对，也可自行实现 Reader/Writer。
2. 用 `storages.Set("hgraph_precise", storage.reader, storage.writer)` 将配对注册到
   `UserDefinedIOSet`。这个名称由应用选择，须非空，且在**同一个 set** 中不能重复；
   一个名字对应一对句柄，不代表独立的磁盘文件。
3. 在 HGraph 的 `index_param` 中同时指定 `"precise_io_type": "user_defined_io"` 和
   `"precise_user_defined_io": "hgraph_precise"`，再调用
   `engine.CreateIndex("hgraph", parameters, storages)`（或相应的 Factory 重载）。
   VSAG 按名称取出该配对，用于精确码的构建写入与后续读取。

两个字符串必须**完全相同**；例如注册名是 `"hgraph_precise"`，却把参数写成
`"other_store"`，索引创建时会报“user defined IO not found”。选择 `user_defined_io`
时，名称缺失、不是字符串或为空也会被拒绝；没有提供配对 set 同样会失败。反过来，
如果 `precise_io_type` **没有**选 `user_defined_io`，即使预先填写
`precise_user_defined_io`，该名称也不会启用自定义存储，配置会按所选的 IO 类型执行。
注册和参数是两步，单独填写任意一步都不够。

此设置只影响 `fp32` **precise codes**：`sq8` base codes 与图仍为
`block_memory_io`。当前还必须设置 `use_reorder: true`；不能用该参数将整个索引或
base/graph/bucket 存储改为自定义 IO。`CreateIndex` 返回后，索引会保留已绑定配对的
共享所有权；修改或销毁用于注册的 set 不会把已有索引重新绑定到另一个名称。

## 回调约定

需要提供三个同步回调：

| 回调 | 签名 | 职责 |
|------|------|------|
| Read | `void(uint64_t offset, uint64_t len, void* dest)` | 从 `offset` 处复制 `len` 字节到 `dest`。越界时抛出异常。 |
| Write | `void(uint64_t offset, uint64_t len, const void* src)` | 在 `offset` 处消费 `src` 的 `len` 字节。如果 `offset + len` 超出当前容量，需扩展底层存储。 |
| Resize | `void(uint64_t new_size)` | 将底层存储截断或扩展至 `new_size`，同时保留未截断的前缀。 |

约束：

- 三个回调均由互斥锁串行化，不得重入同一个适配器。
- 在索引或应用层串行执行写入与扩缩容操作。单次回调持锁不保证“检查大小后调整大小”这一多步操作的原子性。自定义 Reader/Writer 必须访问同一份底层字节空间并提供一致的同步机制；索引使用期间不得绕过配对修改底层数据。
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
// "hgraph_precise" 是注册名，不是路径；必须与下面的 JSON 参数一致。
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
| `precise_io_type` | `"user_defined_io"` | 要启用自定义精确码存储，必须选择此 IO 类型；否则下面的名称不生效。 |
| `precise_user_defined_io` | **非空字符串** | 启用上述 IO 类型时必填；必须与本次 `CreateIndex` 传入的 `UserDefinedIOSet::Set` 注册名完全一致。同一 set 内名称不得重复。 |
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
