# HGraph 构建缓存使用与测试说明

## 1. 当前状态

VSAG 项目的主体实现是 C++。当前缓存阶段接口也是以 C++ 形式实现的，核心入口位于以下文件：

- include/vsag/index.h
- include/vsag/dataset.h
- src/algorithm/hgraph.h
- src/algorithm/hgraph.cpp

项目里同时存在一层 C API，它的作用不是把项目“改成 C 项目”，而是提供一组稳定的 C ABI 入口，方便其他语言或运行时通过 FFI 调用，例如 Python、Go、Rust、Java 等。对应文件是：

- include/vsag/vsag_c_api.h
- src/vsag_c_api.cpp

当前 Build Cache 能力已经在 C++ 核心层实现并通过单测验证，但尚未暴露到 C API、Python 绑定和 Node 绑定。

## 2. 当前缓存接口能力范围

目前只有 HGraph 支持构建缓存接口。调用前可以先通过 SupportsBuildCache 判定能力。

已支持的 C++ 接口：

- SupportsBuildCache()
- ExportBuildCache(std::ostream&)
- BuildWithCache(const DatasetPtr&, std::istream&, const BuildCacheOptions&)
- GetBuildCacheStats()

Dataset 也已经新增了 FeatureIds 字段，用于承载跨构建周期稳定的 FeatureID。

## 3. 当前是否可以运行测试

可以。

需要注意两点：

1. make release 只做 release 编译，默认不会编译测试。
2. 运行缓存相关测试时，需要使用 ENABLE_TESTS=ON 的构建目录。

当前已经验证通过的缓存相关测试为：

1. HGraph Build Cache Round Trip
2. HGraph Build Cache Survives Serialize Round Trip

对应测试文件：

- src/algorithm/hgraph_build_cache_test.cpp

## 4. 如何启动缓存相关测试

### 4.1 方式一：使用单独的测试构建目录

如果你希望单独验证缓存能力，推荐新建一个测试构建目录，例如：

```bash
cmake -S /home/gubaoyuan.gby/vsag-main \
  -B /home/gubaoyuan.gby/vsag-main/build-cache-test \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_TESTS=ON \
  -DENABLE_MOCKIMPL=ON \
  -DENABLE_ASAN=OFF \
  -DENABLE_CCACHE=ON

cmake --build /home/gubaoyuan.gby/vsag-main/build-cache-test --target unittests --parallel 6

/home/gubaoyuan.gby/vsag-main/build-cache-test/tests/unittests "[BuildCache]" --allow-running-no-tests
```

上面这条命令只运行带 BuildCache 标签的单测。

### 4.2 方式二：运行全部单测

如果你希望跑整个测试集合，可以在项目根目录执行：

```bash
cd /home/gubaoyuan.gby/vsag-main
make test
```

这会生成 Debug 测试构建，并执行：

- build/tests/unittests
- build/tests/functests
- build/mockimpl/tests_mockimpl

### 4.3 只运行缓存相关测试

如果已经有测试构建目录，也可以只运行缓存相关标签：

```bash
/home/gubaoyuan.gby/vsag-main/build-cache-test/tests/unittests "[BuildCache]"
```

## 5. 当前缓存测试覆盖了什么

### 5.1 HGraph Build Cache Round Trip

测试流程：

1. 构造带 FeatureIds 的小型 Dataset
2. 创建 HGraph 索引
3. 调用 Build
4. 调用 ExportBuildCache 导出缓存到内存流
5. 创建新的 HGraph 索引
6. 调用 BuildWithCache 从缓存流构建
7. 调用 GetBuildCacheStats 校验命中统计

### 5.2 HGraph Build Cache Survives Serialize Round Trip

测试流程：

1. 普通 Build 建索引
2. Serialize 到内存流
3. Deserialize 恢复索引
4. 再次调用 ExportBuildCache
5. 验证序列化回读后 FeatureID 元数据仍可用于导出缓存

## 6. 输入输出文件如何设置

### 6.1 当前缓存接口的输入输出形式

缓存接口本身不绑定固定路径，而是使用流接口：

- ExportBuildCache 接收 std::ostream
- BuildWithCache 接收 std::istream

这意味着你可以把缓存写到：

- 本地文件
- 内存流
- 管道
- 自定义流对象

最常见的是本地二进制文件。

### 6.2 推荐的路径组织方式

以你当前的数据目录为例，可以按下面方式组织：

```text
/home/tianlan.lht/data/
  whole-30m-768-euclidean-day1/
    id_map
    label.bin
    vector_data.bin
  whole-30m-768-euclidean-day2/
    id_map
    label.bin
    vector_data.bin

/home/gubaoyuan.gby/cache/
  day1.hgraph.cache
  day2.hgraph.cache

/home/gubaoyuan.gby/indexes/
  day1.hgraph.index
  day2.hgraph.index
```

建议含义：

- day1 目录作为旧数据集输入
- day2 目录作为新数据集输入
- day1.hgraph.cache 作为 day1 导出的缓存文件
- day2.hgraph.index 作为 day2 构建完成后的索引文件

### 6.3 数据集输入路径说明

当前项目还没有内置一个直接读取下列三文件并自动组装 Dataset 的通用加载器：

- id_map
- label.bin
- vector_data.bin

所以当前使用方式是：调用方自己读取这些文件，再填充到 C++ Dataset 中。

其中建议对应关系如下：

- id_map -> FeatureIds
- label.bin -> Ids
- vector_data.bin -> Float32Vectors

同时要先做数据对齐校验：

1. id_map 有无尾部空行
2. id_map 记录数是否与 label.bin/vector_data.bin 条数一致
3. FeatureID 是否为空或重复

## 7. C++ 使用示例

下面是缓存阶段的一个最小使用范式。

```cpp
#include <fstream>
#include <string>
#include <vector>

#include "vsag/dataset.h"
#include "vsag/factory.h"

using namespace vsag;

DatasetPtr LoadDatasetFromFiles(const std::string& dataset_root);

int main() {
    const std::string day1_root = "/home/tianlan.lht/data/whole-30m-768-euclidean-day1";
    const std::string day2_root = "/home/tianlan.lht/data/whole-30m-768-euclidean-day2";
    const std::string cache_path = "/home/gubaoyuan.gby/cache/day1.hgraph.cache";
    const std::string index_path = "/home/gubaoyuan.gby/indexes/day2.hgraph.index";

    auto day1 = LoadDatasetFromFiles(day1_root);
    auto day2 = LoadDatasetFromFiles(day2_root);

    std::string params = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 768,
        "hgraph": {
            "max_degree": 64,
            "ef_construction": 200
        }
    }
    )";

    auto old_index = Factory::CreateIndex("hgraph", params);
    if (!old_index.has_value() || !old_index.value()->SupportsBuildCache()) {
        return -1;
    }

    auto build_result = old_index.value()->Build(day1);
    if (!build_result.has_value()) {
        return -2;
    }

    {
        std::ofstream cache_out(cache_path, std::ios::binary);
        auto export_result = old_index.value()->ExportBuildCache(cache_out);
        if (!export_result.has_value()) {
            return -3;
        }
    }

    auto new_index = Factory::CreateIndex("hgraph", params);
    BuildCacheOptions options;
    options.enable_warm_start = true;
    options.hit_refine_rounds = 2;
    options.missed_refine_rounds = 4;
    options.enable_parallel_refine = true;
    options.refine_parallelism = 16;
    options.drop_invalid_neighbors = true;

    {
        std::ifstream cache_in(cache_path, std::ios::binary);
        auto cached_build_result = new_index.value()->BuildWithCache(day2, cache_in, options);
        if (!cached_build_result.has_value()) {
            return -4;
        }
    }

    {
        std::ofstream index_out(index_path, std::ios::binary);
        auto serialize_result = new_index.value()->Serialize(index_out);
        if (!serialize_result.has_value()) {
            return -5;
        }
    }

    auto stats = new_index.value()->GetBuildCacheStats();
    if (stats.has_value()) {
        // 可打印 stats.value().cache_hit_rate 等指标
    }

    return 0;
}
```

## 8. Dataset 需要包含哪些字段

缓存构建场景下，Dataset 至少应包含：

- Dim
- NumElements
- Ids
- Float32Vectors
- FeatureIds

最小构造示意：

```cpp
auto dataset = Dataset::Make()
    ->Owner(true)
    ->Dim(dim)
    ->NumElements(count)
    ->Ids(ids)
    ->Float32Vectors(vectors)
    ->FeatureIds(feature_ids);
```

其中：

- ids 是 int64_t 数组
- vectors 是 float 数组
- feature_ids 是 std::string 数组

## 9. 缓存文件与索引文件的区别

两者不是一回事。

### 9.1 缓存文件

由 ExportBuildCache 生成，用于下一轮 BuildWithCache。

内容主要是：

- FeatureID 索引信息
- FeatureID 字节区
- 底层图邻居关系

### 9.2 索引文件

由 Serialize 生成，用于后续加载整个索引对象。

内容主要是完整索引状态，而不是只服务于 warm start 的缓存子集。

## 10. 当前限制

当前实现有以下边界：

1. 只有 C++ 层支持 Build Cache，C API 暂未暴露这些能力。
2. Python/Node 绑定尚未提供 Build Cache 接口。
3. 项目内部还没有提供现成的 day1/day2 三文件数据加载器。
4. 运行缓存构建前，调用方需要自己完成 id_map、label.bin、vector_data.bin 的对齐检查与 Dataset 组装。

## 11. Refine 语义与并行化

当前 BuildWithCache 的 Warm Start 与 Refine 语义如下：

- `hit_nodes` 会用旧缓存邻居映射结果初始化 `bottom_graph_`。
- `missed_nodes` 也会初始化邻居列表，但初始化值是空邻居列表，不会隐式补边。
- `missed_nodes` 主要依赖后续 `missed_refine_rounds` 在 `bottom_graph_` 上逐轮补边和互连。

并行 Refine 的开关位于 `BuildCacheOptions`：

- `enable_parallel_refine`：是否启用轮内跨节点并行 refine。
- `refine_parallelism`：请求的 refine 线程数，`0` 表示复用 `build_thread_count`。

当前实现遵循“轮间栅栏、轮内并行”的策略：

- 同一轮内，不同节点的 refine 可以并发执行。
- 每一轮结束后会等待所有节点完成，再进入下一轮。
- `missed_nodes` 与 `hit_nodes` 仍按两个 phase 顺序执行，不会混合调度。

BuildCacheStats 新增了以下可观测字段：

- `hit_refine_parallelism`
- `missed_refine_parallelism`

这两个字段表示各 phase 实际使用的并行度，已经过 `thread_pool` 容量和节点数量裁剪。

## 12. 推荐的实际操作顺序

如果你要在当前仓库里继续验证缓存构建，建议顺序如下：

1. 先用 build-cache-test 跑 `[BuildCache]` 单测，确认本地环境正常。
2. 编写一个本地小工具或示例程序，负责把 day1/day2 三文件读取成 Dataset。
3. 用 day1 执行普通 Build 并导出缓存文件。
4. 用 day2 调用 BuildWithCache，并输出 BuildCacheStats。
5. 最后把新索引 Serialize 到索引文件，供后续加载和查询使用。