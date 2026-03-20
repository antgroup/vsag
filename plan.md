# 实施计划: 支持 FP16 和 BF16 输入向量格式

## 1. 目标概述
- **任务目标**: 为 VSAG 添加 FP16 和 BF16 输入向量格式支持
- **预期成果**: HGraph 和 IVF 索引能够直接处理 FP16/BF16 格式的 base 向量
- **完成标准**: 
  - 所有单元测试通过
  - HGraph/IVF 功能测试通过（FP16/BF16）
  - 代码检查通过

## 2. 当前状态分析

### 已存在的实现
| 组件 | 状态 | 说明 |
|------|------|------|
| `DATA_TYPE_FP16` 枚举 | ✅ 已存在 | `src/data_type.h` |
| `DATATYPE_FLOAT16` 常量 | ✅ 已存在 | `constants.h/cpp` |
| FP16 SIMD 函数 | ✅ 已存在 | `fp16_simd.h` |
| BF16 SIMD 函数 | ✅ 已存在 | `bf16_simd.h` |
| `FP16ToFloat/BF16ToFloat` | ✅ 已存在 | 类型转换函数 |
| `QuantizerAdapter<int8_t>` | ✅ 已存在 | 可参考实现 |

### 需要新增的部分
| 组件 | 文件 | 说明 |
|------|------|------|
| `DATA_TYPE_BF16` 枚举 | `src/data_type.h` | 新增枚举值 |
| `DATATYPE_BFLOAT16` 常量 | `include/vsag/constants.h` | 新增常量 |
| `FLOAT16_VECTORS` 常量 | `include/vsag/constants.h` | Dataset key |
| `Float16Vectors` API | `include/vsag/dataset.h` | getter/setter |
| Dataset 实现 | `src/dataset_impl.h/cpp` | uint16_t variant 支持 |
| QuantizerAdapter<uint16_t> | `src/quantization/` | FP16/BF16 适配器 |

## 3. 技术方案

### 核心思路
1. **Dataset 层**: 添加 `Float16Vectors` API，使用 `uint16_t*` 作为数据载体
2. **量化器层**: 扩展 `QuantizerAdapter` 支持 `uint16_t`，通过 `data_type_` 区分 FP16/BF16
3. **索引层**: HGraph/IVF 根据 `data_type_` 选择正确的 getter

### 关键设计

#### 数据流
```
用户输入 (uint16_t*) 
  → Dataset::Float16Vectors() 
  → GetFloat16Vectors() 
  → QuantizerAdapter 转换为 float 
  → 内部量化器
```

#### FP16/BF16 区分
```cpp
// QuantizerAdapter 中
if (data_type_ == DataTypes::DATA_TYPE_FP16) {
    vec[i] = FP16ToFloat(data[i]);
} else if (data_type_ == DataTypes::DATA_TYPE_BF16) {
    vec[i] = BF16ToFloat(data[i]);
}
```

### 接口定义

#### Dataset API
```cpp
// dataset.h
virtual DatasetPtr Float16Vectors(const uint16_t* vectors) = 0;
virtual const uint16_t* GetFloat16Vectors() const = 0;
```

#### 常量
```cpp
// constants.h
extern const char* const DATATYPE_BFLOAT16;  // "bfloat16"
extern const char* const FLOAT16_VECTORS;    // "float16_vectors"
```

## 4. 实施步骤

| 序号 | 步骤 | 涉及文件 | 详细说明 |
|------|------|----------|----------|
| 1 | 添加常量声明 | `include/vsag/constants.h` | 添加 `DATATYPE_BFLOAT16`, `FLOAT16_VECTORS` |
| 2 | 添加常量定义 | `src/constants.cpp` | 定义 `"bfloat16"`, `"float16_vectors"` |
| 3 | 添加枚举值 | `src/data_type.h` | 添加 `DATA_TYPE_BF16 = 4` |
| 4 | 添加 Dataset API | `include/vsag/dataset.h` | 添加 `Float16Vectors`/`GetFloat16Vectors` |
| 5 | 扩展 variant | `src/dataset_impl.h` | 添加 `const uint16_t*` 类型 |
| 6 | 实现 Dataset | `src/dataset_impl.cpp` | 实现新方法、内存管理 |
| 7 | 参数解析 | `src/index_common_param.cpp` | 支持 `"float16"`/`"bfloat16"` |
| 8 | 扩展量化器 | `src/quantization/quantizer_adapter.h` | 支持 `uint16_t`，添加 `data_type_` |
| 9 | 实现量化器转换 | `src/quantization/quantizer_adapter.cpp` | FP16/BF16 → FP32 转换逻辑 |
| 10 | HGraph 支持 | `src/algorithm/hgraph.cpp` | 根据 `data_type_` 选择 getter |
| 11 | IVF 支持 | `src/algorithm/ivf.cpp` | 类似 HGraph 处理 |
| 12 | 文档更新 | `docs/dataset_format.md` | 添加 FP16/BF16 格式说明 |
| 13 | 示例代码 | `examples/cpp/` | 添加 FP16/BF16 示例 |
| 14 | 单元测试 | `tests/` | Dataset、量化器测试 |
| 15 | 功能测试 | `tests/` | HGraph/IVF FP16/BF16 测试 |

## 5. 测试计划

### 单元测试
| 测试项 | 测试文件 | 验证内容 |
|--------|----------|----------|
| Dataset Float16Vectors | `test_dataset.cpp` | 设置/获取、DeepCopy、Append |
| QuantizerAdapter<uint16_t> | `test_quantizer_adapter.cpp` | Train/Encode/Decode |
| 类型转换 | `test_type_conversion.cpp` | FP16ToFloat/BF16ToFloat 正确性 |

### 功能测试
| 测试项 | 测试文件 | 验证内容 |
|--------|----------|----------|
| HGraph FP16 | `test_hgraph_fp16.cpp` | 构建和搜索 |
| HGraph BF16 | `test_hgraph_bf16.cpp` | 构建和搜索 |
| IVF FP16 | `test_ivf_fp16.cpp` | 构建和搜索 |
| IVF BF16 | `test_ivf_bf16.cpp` | 构建和搜索 |

### 测试命令
```bash
# 构建
make debug

# 运行单元测试
make test

# 运行功能测试
./build/debug/tests/functest --gtest_filter="*fp16*"
./build/debug/tests/functest --gtest_filter="*bf16*"
```

## 6. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 内存管理问题 | 高 | 参考 Int8Vectors 实现，确保正确释放 |
| 类型转换精度 | 中 | 使用已验证的 FP16ToFloat/BF16ToFloat |
| 索引兼容性 | 中 | 保持向后兼容，新增功能不影响现有代码 |
| 性能影响 | 低 | 批量转换优化，减少逐元素转换开销 |

## 7. 验收标准

- [ ] 编译通过：`make debug` 成功
- [ ] 单元测试通过：`make test` 成功
- [ ] 代码风格检查通过：`make lint` 成功
- [ ] HGraph FP16 构建和搜索正确
- [ ] HGraph BF16 构建和搜索正确
- [ ] IVF FP16 构建和搜索正确
- [ ] IVF BF16 构建和搜索正确
- [ ] 文档更新完整

## 8. 相关资源

- **需求文档**: `/home/tianlan.lht/code/workspace/fp16_and_bf16_support.md`
- **任务文件**: `/home/tianlan.lht/code/workspace/agent-hive/tasks/2026-03-20-支持-fp16-bf16-输入向量格式.md`
- **参考实现**: 
  - `Int8Vectors` 实现模式
  - `QuantizerAdapter<int8_t>` 实现模式
- **SIMD 参考**: `src/simd/fp16_simd.h`, `src/simd/bf16_simd.h`