---
title: "添加 Python 绑定 FP16/BF16 API"
status: in_progress
priority: medium
created_by: "opencode"
assigned_to: "opencode"
created_at: "2026-04-28"
updated_at: "2026-04-29"
target_repo: "antgroup/vsag"
related_files:
  - "python_bindings/"
  - "src/data_type.h"
  - "include/vsag/constants.h"
pr_url: ""
---

# 添加 Python 绑定 FP16/BF16 API

## 背景

VSAG 的 C++ 层已完整支持 FP16 和 BF16 数据类型的索引构建与搜索，但 Python 绑定层尚未暴露任何 FP16/BF16 相关 API。这使得 Python 用户无法使用半精度向量数据与 VSAG 交互，限制了 FP16/BF16 功能的实际可用性。

## 问题描述

`python_bindings/` 目录中没有任何 FP16/BF16 相关代码。具体缺失：

1. Dataset 构造缺少 FP16/BF16 向量数据设置方法
2. 索引构建和搜索接口无法接受 FP16/BF16 格式的输入
3. 无法从 Python 端指定 `data_type` 为 `"float16"` 或 `"bfloat16"`

## 改进方案

1. **扩展 Dataset Python 接口**：
   - 添加 `set_float16_vectors()` 和 `set_bfloat16_vectors()` 方法
   - Python 端接受 numpy 的 `float16` / `bfloat16` 数组，内部转为 `uint16_t*` 传递给 C++ 层

2. **扩展索引构建/搜索接口**：
   - 确保构建（`Build`）和搜索（`KnnSearch`/`RangeSearch`）接口能接受 FP16/BF16 向量数据
   - Dataset 从 Python 传入 FP16/BF16 数据时应正确设置 `data_type` 字段

3. **添加 Python 示例和测试**：
   - 提供使用 FP16/BF16 数据构建和搜索索引的 Python 示例
   - 编写 pytest 测试覆盖 FP16/BF16 的基本功能路径

## 预期效果

- Python 用户可以直接使用 numpy float16/bfloat16 数组与 VSAG 交互
- FP16/BF16 功能在 Python 层面的可用性与 C++ 层保持一致
- 降低半精度向量搜索的使用门槛

## 参考资料

- `src/data_type.h` — DATA_TYPE_FP16/DATA_TYPE_BF16 枚举定义
- `include/vsag/constants.h` — DATATYPE_FLOAT16/DATATYPE_BFLOAT16 字符串常量
- 现有 Python 绑定代码结构