# Pyramid 重复向量检测测试方案

## 测试目标
验证 Pyramid 索引在 GRAPH 模式下的重复向量检测与跨路径/跨层级召回机制。

## 测试数据设计

### 基础数据
- **基础向量数量**: 1000 个（ID: 0-999）
- **维度**: 128
- **路径分布**: 随机分配到多个路径（确保 nodes 进入 GRAPH 状态）

### 重复向量（5种场景 × 10对 = 50个）

| 场景 | ID范围 | 路径 | 复制来源 | 测试目的 |
|------|--------|------|---------|---------|
| **A: 同路径** | 1000-1009 | a/b/c | ID 0-9 | 同 node 内重复 |
| **B: 跨路径同父** | 1010-1019 | a/b/d | ID 0-9 | 跨兄弟 node 重复 |
| **C: 跨路径不同分支** | 1020-1029 | x/y/z | ID 0-9 | 跨完全不同分支重复 |
| **D: 跨层级子→父** | 1030-1039 | a/b | ID 0-9 | 子路径向量在父 node 重复 |
| **E: 跨层级父→子** | 1040-1049 | a/b/c/d | ID 0-9 | 父路径向量在子 node 重复 |

**总计**: 1050 个向量

## 索引配置

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "alpha": 1.2,
        "graph_iter_turn": 15,
        "neighbor_sample_rate": 0.2,
        "no_build_levels": [],
        "use_reorder": false,
        "graph_type": "nsw",
        "build_thread_count": 4,
        "index_min_size": 5,
        "support_duplicate": true,
        "ef_construction": 200
    }
}
```

## 预期结果

### Build 阶段
- 重复检测应识别出 50 对重复
- 重复向量不应插入 graph（通过 GetNumElements 验证应为 ~1000）

### Search 阶段
- 查询原始向量应同时召回原始 ID 和重复 ID
- 跨路径/跨层级召回应生效

## 关键机制分析

### 跨层级重复检测
```
ID=0:   path="a/b/c"  -> 插入到 root, "a", "a/b", "a/b/c"
ID=1030: path="a/b"    -> 插入到 root, "a", "a/b"

在 node "a/b" 中：
1. ID=0 已存在于 ids_（从 Build 阶段）
2. ID=1030 插入时搜索 graph
3. 如果找到相似向量（ID=0），检测为重复
```

### 跨路径召回
```
搜索 "a/b/c" 时：
1. 在 node "a/b/c" 搜索
2. 访问 ID=0 时展开重复 -> 得到 ID=1000, ID=1040
3. 跨路径/层级召回通过 label_table 全局维护
```

## 超时问题分析

### 可能原因
1. **ef_construction=200 太大**: NSW 建图时搜索深度过大
2. **max_degree=32**: 图密度高导致搜索慢
3. **graph_iter_turn=15**: ODescent 迭代次数多
4. **代码bug**: 无限循环或内存问题

### 排查步骤
1. 减少 ef_construction 到 50
2. 减少 max_degree 到 16
3. 添加 Build 阶段日志
4. 单独测试小数据集（100向量）

## 测试验证矩阵

| 查询路径 | 查询ID | 应召回ID | 场景验证 |
|-----------|--------|----------|---------|
| a/b/c | 0 | 0, 1000, 1040? | A + E |
| a/b | 0 | 0, 1030 | D |
| a/b/d | 0 | 0, 1010 | B |
| x/y/z | 0 | 0, 1020 | C |
| a/b/c/d | 0 | 0, 1040 | E |

## 关键观察点

1. **Build 日志**: DUPLICATE DETECTED 是否输出
2. **Graph 大小**: GetNumElements() 是否为 ~1000（非1050）
3. **召回完整性**: 50对重复是否都能被召回
4. **距离一致性**: 原始和重复向量距离是否相同

## 修复建议

如果超时，尝试：
```json
{
    "ef_construction": 50,  // 从200减少到50
    "max_degree": 16,       // 从32减少到16
    "graph_iter_turn": 5    // 从15减少到5
}
```

## 日志添加位置

### Build 阶段
```cpp
// pyramid.cpp: Build()
std::cout << "[BUILD] Starting build with " << data_num << " vectors" << std::endl;

// pyramid.cpp: add_one_point()
std::cout << "[BUILD] Processing id=" << inner_id << " at level=" << node->level_ << std::endl;
if (search_param.duplicate_id >= 0) {
    std::cout << "[BUILD] DUPLICATE: " << inner_id << " is dup of " << search_param.duplicate_id << std::endl;
}
```

### Search 阶段
```cpp
// pyramid.cpp: search_node()
std::cout << "[SEARCH] Searching node level=" << node->level_ << " status=" << node->status_ << std::endl;
```
