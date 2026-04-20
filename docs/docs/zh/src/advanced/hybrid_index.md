# 内存-磁盘混合索引（DiskANN）

在海量向量场景下，将整个图索引放入内存既昂贵又浪费。VSAG 提供的 `diskann` 索引将：

- **压缩后的向量（PQ）**保留在内存中，用于快速剪枝；
- **完整向量**与**图结构**存储在磁盘，按查询路径异步读取。

从而在有限内存预算下支撑 10 亿级向量的近邻查询。

## 构建 DiskANN

```cpp
std::string build_params = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "diskann": {
        "max_degree": 32,
        "ef_construction": 400,
        "pq_sample_rate": 0.1,
        "pq_dims": 32,
        "use_async_io": true
    }
}
)";
auto index = vsag::Factory::CreateIndex("diskann", build_params).value();
index->Build(dataset);
```

完整例子：`examples/cpp/102_index_diskann.cpp`。

## 异步 IO（libaio）

Linux 下可在构建参数中开启 `use_async_io`，搜索路径会通过 libaio 并发发起读请求。
需要在编译时打开 `VSAG_ENABLE_LIBAIO=ON`（详见 [编译构建](../development/building.md)）。

## 文件布局

`diskann` 在磁盘上产生两类文件：

- `*.index`：图结构；
- `*.data`：完整精度向量。

反序列化时需要同时可访问这两份数据。

## 注意事项

- 磁盘介质建议使用 NVMe SSD；HDD 下查询延迟会显著退化。
- 内存中 PQ 的压缩率与精度由 `pq_dims` 控制，过小会导致召回下降。
- 冷启动时建议对索引文件做预热（随机读几兆）以建立 page cache。
- 当前 DiskANN 不支持在线插入 / 删除；如需增删请重建。
