---
title: "使用固定种子正态分布数据解决测试随机性问题"
status: in_progress
priority: high
created_by: "opencode (glm-5)"
assigned_to: "lht"
created_at: "2026-04-20"
updated_at: "2026-04-20"
target_repo: "antgroup/vsag"
related_files:
  - "tests/fixtures/core/random.h"
  - "tests/fixtures/data/vector_generator.h"
  - "tests/fixtures/data/vector_generator.cpp"
  - "tests/fixtures/framework/test_dataset.cpp"
  - "tests/test_index.cpp"
  - "tests/test_hnsw.cpp"
  - "tests/test_hgraph.cpp"
  - "tests/test_ivf.cpp"
pr_url: ""
---

# 使用固定种子正态分布数据解决测试随机性问题

## 背景

在VSAG项目的CI流程中，偶现测试失败问题。这些失败并非代码bug，而是由于测试数据生成使用了完全随机的种子，导致每次测试运行时的数据分布不同，进而影响召回率等测试指标。召回率测试在随机数据下存在天然波动，当波动幅度超过阈值时，即使算法实现正确，CI也会判定为失败。

当前测试框架使用`std::random_device`初始化随机数生成器（`tests/fixtures/core/random.h:32-33`），每次运行产生不同的随机种子，测试数据完全不可重复。在`test_dataset.cpp:51`中，生成向量时调用`fixtures::RandomValue(0, 564)`获取随机种子，数据每次都不同。测试中对召回率有硬性要求（如`test_multi_thread.cpp:113`中的`REQUIRE(recall >= 0.99)`），在随机数据下，召回率可能因数据分布偶然不理想而低于阈值，导致CI偶现失败。

部分测试已意识到随机数据的局限性，如`test_ivf.cpp:81-82`注释说明"IVF just can't achieve high recall on random datasets"，并主动降低了召回率期望值。但这只是权宜之计，未从根本上解决数据随机性问题。

## 问题描述

当前测试随机性问题体现在以下几个方面：

1. **种子不可控**：`fixtures::RandomValue`函数（`tests/fixtures/core/random.h:31-36`）使用`thread_local std::random_device`初始化`std::mt19937`，每次测试运行都产生新的随机种子，数据生成完全不可预测。

2. **数据分布不稳定**：`generate_vectors`函数（`tests/fixtures/data/vector_generator.h:55-69`）使用均匀分布`std::uniform_real_distribution<T>(0.1, 0.9)`生成数据。均匀分布在高维空间中，点与点之间的距离分布较为集中，不利于测试最近邻搜索的真实性能。正态分布数据在高维空间中距离分布更分散，更贴近真实场景，且固定种子后数据分布稳定可控。

3. **召回率波动大**：随机数据导致召回率波动，测试结果不可重复。CI可能在某次运行中因数据分布偶然不佳而判定失败，即使算法实现本身无误。例如`test_hnsw.cpp`、`test_hgraph.cpp`中的召回率测试依赖随机数据，偶现低于阈值的情况。

4. **调试困难**：当CI失败时，因每次数据都不同，无法在本地复现失败场景，难以定位问题根源是数据分布偶然不佳还是真实的代码bug。

## 改进方案

### 1. 新增固定种子正态分布数据生成函数

在`tests/fixtures/data/vector_generator.h`和`vector_generator.cpp`中新增：

```cpp
std::vector<float>
generate_normal_vectors(uint64_t count, 
                        uint32_t dim, 
                        bool need_normalize = true,
                        int seed = 47,
                        float mean = 0.0f,
                        float stddev = 1.0f);
```

使用`std::mt19937(seed)`初始化随机数生成器，通过`std::normal_distribution<float>(mean, stddev)`生成正态分布数据。正态分布在高维空间中距离分布更分散，更接近真实业务场景（如Embedding向量通常服从某种分布而非均匀分布）。

### 2. 替换随机种子生成方式

修改`tests/fixtures/core/random.h`中的`RandomValue`模板函数，增加可选种子参数：

```cpp
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
RandomValue(const T& min, const T& max, int seed = -1) {
    if (seed >= 0) {
        thread_local std::mt19937 gen(seed);
        std::uniform_int_distribution<T> dis(min, max);
        return dis(gen);
    }
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);
    return dis(gen);
}
```

默认`seed = -1`保持原行为（完全随机），但允许测试框架传入固定种子以获得可重复数据。

### 3. 修改测试数据生成调用

在`tests/fixtures/framework/test_dataset.cpp`的`GenerateRandomDataset`函数（第40-87行）中：

- 将`fixtures::generate_vectors(count, dim, need_normalize, fixtures::RandomValue(0, 564))`改为使用固定种子的`generate_normal_vectors`
- 或为`generate_vectors`传入固定种子（如47），不再使用`RandomValue(0, 564)`随机种子

可选方案：
1. **全局固定种子**：所有测试使用同一固定种子（如47），数据完全一致，召回率波动降至最低
2. **测试级固定种子**：每个测试用例可指定不同种子，但同一用例多次运行数据一致

推荐使用方案1（全局固定种子），简单统一，CI稳定性最佳。

### 4. 保留随机测试选项

对于需要验证算法在极端数据分布下鲁棒性的测试，可保留随机数据生成路径。通过测试参数或环境变量控制是否使用随机数据，默认使用固定种子数据以保证CI稳定。

在`TestDataset::CreateTestDataset`函数中增加可选参数：

```cpp
static std::shared_ptr<TestDataset>
CreateTestDataset(uint64_t dim,
                  uint64_t count,
                  std::string metric_str = "l2",
                  bool with_path = false,
                  float valid_ratio = 0.8,
                  std::string vector_type = "dense",
                  uint64_t extra_info_size = 0,
                  bool has_duplicate = false,
                  int64_t id_shift = 16,
                  bool use_random_seed = false);  // 新增参数
```

### 5. 调整召回率阈值

使用固定种子正态分布数据后，可根据稳定的数据分布重新评估和调整召回率阈值。正态分布数据可能使某些索引类型的召回率表现更稳定，可适当提高阈值以确保算法质量。同时移除`test_ivf.cpp`等文件中因随机数据而刻意降低召回率期望的权宜代码。

### 6. 测试验证

在`tests/fixtures/data/vector_generator_test.cpp`（如不存在则新建）中增加测试：

| 测试用例 | 验证内容 |
|----------|----------|
| 固定种子生成一致性 | 相同种子多次调用`generate_normal_vectors`生成相同数据 |
| 正态分布统计特性 | 生成的数据均值、方差符合预期参数 |
| 可重复性验证 | 使用固定种子运行完整测试流程，多次运行召回率一致 |
| CI稳定性验证 | 固定种子下CI运行多次不偶现召回率失败 |

## 预期效果

1. **CI稳定性大幅提升**：固定种子消除数据随机性，召回率测试波动降至最低，CI偶现失败问题基本消除。

2. **测试结果可重复**：本地开发可复现CI失败场景（若仍有失败），便于定位问题。调试时可通过调整种子观察不同数据分布下的算法表现。

3. **数据分布更贴近真实场景**：正态分布在高维空间中的距离分布特性更符合真实Embedding向量场景，测试结果更能反映算法实际性能。

4. **召回率阈值可合理设置**：稳定数据下可科学设定召回率阈值，无需为随机数据波动预留过宽容差，提高测试对算法质量的把控能力。

5. **测试框架可扩展**：保留随机数据选项，未来可用于压力测试、鲁棒性验证等场景，不牺牲CI日常稳定性。

## 参考资料

- 当前随机数据生成：`tests/fixtures/core/random.h:31-45`
- 向量生成函数：`tests/fixtures/data/vector_generator.h:55-69`、`vector_generator.cpp`
- 测试数据集构建：`tests/fixtures/framework/test_dataset.cpp:40-87`
- 召回率测试示例：`tests/test_index.cpp:475`、`tests/test_multi_thread.cpp:113`
- IVF召回率权宜代码：`tests/test_ivf.cpp:81-82`、`test_ivf.cpp:428-431`
- 正态分布特性：在高维空间中，正态分布数据的点间距离服从Chi分布，比均匀分布更分散，更利于测试最近邻搜索算法的区分能力
