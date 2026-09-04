# VSAG Lite

VSAG Lite 是面向稠密向量检索的精简构建版本。它保留公开 C++ API，
只使用保留索引的应用可以在 Full 与 Lite 之间切换而无需修改业务代码。

## 功能范围

| 功能 | Lite 状态 |
| --- | --- |
| HGraph、LazyHGraph、BruteForce、WARP | 保留 |
| 添加、构建、搜索、更新、删除 | 保留 |
| 序列化与反序列化 | 保留 |
| 标记删除状态持久化 | 保留 |
| IVF、Pyramid、SINDI、SINDI v2 | 移除 |
| SINDI 与稀疏向量专用 DataCell | 移除 |

通过 `Factory::CreateIndex` 创建已移除索引会返回错误。

## 构建与测试

```bash
make lite
make test-lite
```

等价的 CMake 命令：

```bash
cmake -S . -B build-lite -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_LITE=ON -DENABLE_LITE_TESTS=ON
cmake --build build-lite --parallel
ctest --test-dir build-lite --output-on-failure
```

`ENABLE_TESTS` 是 Full 完整测试集，不能与 Lite 同时启用。
Lite 使用 `ENABLE_LITE_TESTS` 独立冒烟测试。

## 兼容性与持久化

Lite 保留相同的头文件、Factory、Dataset、结果对象和 BinarySet API。
标记删除的内部 ID 会写入 HGraph 标签元数据的可选区块；读取器仍兼容
不含该区块的旧文件，但旧文件从未保存的删除信息无法恢复。

HGraph 运行时主要保存稠密编码、图邻接表、标签表、访问列表和可选高精度
重排数据。LazyHGraph 先使用扁平索引，达到阈值后切换为 HGraph。Lite
移除不用的工厂和专用 DataCell，不改变保留索引的运行布局或精度。

## 基准测试

```bash
make benchmark-lite
./build-lite-release/benchs/lite/vsag_lite_benchmark 10000 64 100
```

参数依次为向量数、维度和查询数。JSON 输出包含构建/序列化/加载耗时、
索引字节数、QPS、P50/P99 和 Recall@10。Full/Lite 对比应使用相同的
Release 参数并多次运行取中位数。峰值内存可用 `/usr/bin/time -v`
外部测量。稀疏向量和 SINDI 场景必须使用 Full。
