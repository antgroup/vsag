# 性能评估工具（eval_performance）

`eval_performance` 是 VSAG 自带的命令行性能评估工具，位于 `tools/eval/`，编译后二进制路径为
`build-release/tools/eval/eval_performance`。它可以用于对比不同索引、不同参数组合的吞吐、延迟与召回率。

## 构建

`tools/` 默认不会编译，需要显式开启：

```bash
# 通过项目 Makefile
VSAG_ENABLE_TOOLS=ON make release
# 或：make dev

# 也可直接通过 CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
# 产物：./build-release/tools/eval/eval_performance
```

需要系统安装 HDF5（Ubuntu: `apt install libhdf5-dev`；CentOS: `yum install hdf5-devel`）。

## 两种模式

### 1. 命令行模式（适合单次快速测试）

```bash
./build-release/tools/eval/eval_performance \
    --datapath /tmp/sift-128-euclidean.hdf5 \
    --index_name hgraph \
    --type search \
    --create_params '{"dim":128,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}' \
    --search_params '{"hgraph":{"ef_search":60}}' \
    --topk 10
```

### 2. 配置文件模式（适合批量对比）

```bash
./build-release/tools/eval/eval_performance --config my_eval.yaml
```

参考模板 `tools/eval/eval_template.yaml`，一份配置可包含多个 `eval_caseN` 并分别使用不同参数。

## 支持的评估维度

- **效率**：QPS、TPS
- **效果**：平均召回率、分位召回率（P0/P10/P50/P90...）
- **延迟**：平均延迟、P50/P95/P99 延迟
- **资源**：峰值内存占用

## 搜索模式

`search_mode` 支持 `knn`、`range`、`knn_filter`、`range_filter` 四种。

## 结果导出

- `table`：输出到 stdout 或文件；
- `json`：便于自动化处理；
- `line_protocol`：直接写入 InfluxDB（支持 token 认证）；
- `markdown`：便于贴到 issue / 文档。

## 数据集

可使用 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 提供的 HDF5 格式数据集
（如 `sift-128-euclidean.hdf5`、`gist-960-euclidean.hdf5`）。

## 参考

- 源码：`tools/eval/`
- 英文说明：`tools/eval/README.md`
- 标准机型的基准结果见 [标准环境性能参考](performance.md)。
