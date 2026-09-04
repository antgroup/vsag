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
| IVF、Pyramid、SINDI、SINDI v2、SIMQ | 移除 |
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
  -DENABLE_LITE=ON -DENABLE_TESTS=ON
cmake --build build-lite --parallel
./build-lite/tests/unittests -d yes --allow-running-no-tests
./build-lite/tests/functests -d yes --allow-running-no-tests
```

Lite 使用仓库现有的 Catch2 测试二进制。依赖已裁剪索引实现的测试文件
会在构建时排除。`make test-lite` 会在编译保留测试集后运行聚焦的
`[lite]` 用例；如需更广泛的回归，可按上面的命令直接运行测试二进制。

## 兼容性与持久化

Lite 保留相同的头文件、Factory、Dataset、结果对象和 BinarySet API。
标记删除的内部 ID 会写入 HGraph 标签元数据的可选区块；读取器仍兼容
不含该区块的旧文件，但旧文件从未保存的删除信息无法恢复。

没有标记删除时，writer 不写入可选区块，因此保持旧版字节布局。包含
删除状态的新文件必须使用理解该扩展的新 reader；旧 reader 无法安全读取。

HGraph 运行时主要保存稠密编码、图邻接表、标签表、访问列表和可选高精度
重排数据。LazyHGraph 先使用扁平索引，达到阈值后切换为 HGraph。Lite
移除不用的工厂和专用 DataCell，不改变保留索引的运行布局或精度。

## 基准测试

使用仓库已有的 `tools/eval` 框架，以相同的 Full/Lite YAML 配置运行
标准 ANN 数据集。包体应测量 strip 后的动态库和最终链接部署程序，而不是
带调试信息的静态归档。保留每轮原始数据，并报告多轮运行的中位数和离散
程度。稀疏向量和 SINDI 场景必须使用 Full。
