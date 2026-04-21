# 序列化格式

VSAG 索引可通过多种方式序列化与反序列化，便于持久化、跨进程共享及分布式部署。

## 三种接口

### 1. `BinarySet` / `ReaderSet`

最灵活的方式，把索引拆分为多个命名二进制段。适合用户自己管理存储介质（例如对象存储、KV、分片上传）。

```cpp
// 保存
vsag::BinarySet bs = index->Serialize().value();
for (const auto& key : bs.GetKeys()) {
    auto binary = bs.Get(key);
    // 写入存储介质
}

// 加载
vsag::BinarySet bs_loaded;
// 从介质中读取每个 key 对应的 Binary 放入 bs_loaded
auto empty = vsag::Factory::CreateIndex("hnsw", build_params).value();
empty->Deserialize(bs_loaded);
```

`ReaderSet` 与 `BinarySet` 类似，但通过用户自定义的 `Reader` 按需读取，避免一次性加载全部数据，
常用于内存受限或部分反序列化场景（例如 DiskANN 的磁盘部分）。

### 2. 文件流（`std::ostream` / `std::istream`）

最简单的方式，将索引整体写入文件或内存流：

```cpp
std::ofstream out("index.bin", std::ios::binary);
index->Serialize(out);

std::ifstream in("index.bin", std::ios::binary);
empty->Deserialize(in);
```

### 3. 自定义写函数（`WriteFuncType`）

对于流式/分块写入的后端，可传入写回调：

```cpp
index->Serialize([&](const void* buf, uint64_t offset, uint64_t size) {
    // 将 [buf, buf+size) 写入 offset 位置
});
```

## 注意事项

- `Deserialize` 要求目标索引为**空**索引，并且参数配置与序列化时一致（如 `dim`、`metric_type`）。
- 跨大版本升级时请关注 [版本日志](../resources/release_notes.md) 中的兼容性说明。
- DiskANN 的磁盘索引文件独立管理，`Serialize` 返回的是内存侧元信息。
- 示例参考：`examples/cpp/318_feature_tune.cpp`、`examples/cpp/401_persistent_kv.cpp`、
  `examples/cpp/402_persistent_streaming.cpp`。
