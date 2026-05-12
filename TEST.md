# SINDI 测试速查

## 进入容器

```bash
DOCKER_HOST= docker exec -it -w /root/project/vsag vsag-dev bash
```

## 编译（容器内）

```bash
# 编译（含测试二进制），不跑测试
VSAG_ENABLE_TESTS=ON make debug COMPILE_JOBS=4

# 或者编译+只跑指定 tag 的测试（推荐）
make test COMPILE_JOBS=4 CASE="[SINDI]"
```

## SINDI 测试（容器内）

```bash
# 跑所有 SINDI 相关单元测试（逗号分隔 = OR）
./build/tests/unittests "[SINDI],[SINDIParameter],[TermIdMapper],[ProximityScorer]" -d yes

# 单独跑某个 tag
./build/tests/unittests "[SINDI]" -d yes
./build/tests/unittests "[SINDIParameter]" -d yes
./build/tests/unittests "[TermIdMapper]" -d yes
./build/tests/unittests "[ProximityScorer]" -d yes

# 功能测试
./build/tests/functests "[sindi]" -d yes
```

> **Catch2 语法**：空格分隔 = AND（同时匹配），逗号分隔 = OR（任一匹配）。

## 全量测试（容器内）

```bash
# 全量
make test COMPILE_JOBS=4

# AddressSanitizer
make test_asan COMPILE_JOBS=4

# ThreadSanitizer
make test_tsan COMPILE_JOBS=4
```

## 格式化 & Lint（容器内）

```bash
make fmt
make release COMPILE_JOBS=4 && make lint
```

## 注意事项

- **不要用 `make test` 跑全量**：`BucketDataCell Basic Test` 有已知内存泄漏 bug 会 SIGABRT（见 DEBUG.md），每次都会卡住。开发期间只跑相关 tag 即可。
- QEMU 模拟 x86，编译速度慢，COMPILE_JOBS 建议 ≤ 4（避免 OOM）
- 容器名 `vsag-dev`，源码挂载在 `/root/project/vsag`
- `DOCKER_HOST=` 前缀是因为宿主机有 Colima 的环境变量干扰

## 离线 Eval（容器内）

### 第一次：安装 Python 依赖

容器内：
```bash
pip install h5py numpy
```

容器外（macOS）：
```bash
# 方案 1: 系统 Python + 强制安装
pip install --break-system-packages h5py numpy

# 方案 2: venv
cd /Users/chenhua/claudecode_workspace/research_monorepo/xuanji/remote/vsag-sindi
python3 -m venv .venv
source .venv/bin/activate
pip install h5py numpy
```

> **大数据集（10K+ docs）建议容器外跑 Python**：QEMU 模拟 x86 下 Python 跑大量循环会 segfault 或极慢。容器外跑 H5 生成快很多，文件在 vsag-sindi/ 目录下容器能直接看到。

### 1. 生成压测数据

```bash
mkdir -p bench  # 容器外，挂载到容器的 /root/project/vsag/bench

# 小数据集（容器内可跑）
python3 scripts/gen_sindi_bench.py \
  --output bench/sindi_bench.hdf5 \
  --num_base 1000 --num_query 20 \
  --doc_len 100 --vocab_size 500 --max_terms 50 \
  --proximity_weight 0.3 --multiplicative --topk 10

# 中等数据集（容器外建议）
python3 scripts/gen_sindi_bench.py \
  --output bench/sindi_bench_m.hdf5 \
  --num_base 5000 --num_query 100 \
  --doc_len 200 --vocab_size 1000 --max_terms 80 \
  --proximity_weight 0.3 --multiplicative --topk 10

# 大数据集（必须容器外）
python3 scripts/gen_sindi_bench.py \
  --output bench/sindi_bench_large.hdf5 \
  --num_base 50000 --num_query 200 \
  --doc_len 200 --vocab_size 2000 --max_terms 100 \
  --proximity_weight 0.3 --multiplicative --topk 10
```

参数：
- `--proximity_weight 0.3 --multiplicative`：用乘法 boost 算 GT
- `--proximity_weight 0`：纯 IP GT（baseline 对比用）
- `--proximity_ordered`：有序模式

### 验证 HDF5 文件

```bash
python3 -c "import h5py; f=h5py.File('/root/project/vsag/bench/sindi_bench.hdf5','r'); print('keys:',list(f.keys())); print('attrs:',dict(f.attrs)); print('train shape:',f['train'].shape); print('test shape:',f['test'].shape); print('token_seq:',f['train_token_sequences'].shape if 'train_token_sequences' in f else 'none')"
```

应输出：keys 含 train_token_sequences，attrs 含 type='sparse', distance='ip'

### 2. 编译 eval_performance

```bash
# 注意环境变量是 VSAG_ENABLE_TOOLS（不是 ENABLE_TOOLS）
VSAG_ENABLE_TOOLS=ON make release COMPILE_JOBS=4
```

编出来的二进制在 `build-release/tools/eval/eval_performance`

### 3. Build 索引（看 TPS、内存）

```bash
./build-release/tools/eval/eval_performance \
  --datapath /root/project/vsag/bench/sindi_bench.hdf5 \
  --type build \
  --index_name sindi \
  --create_params '{"dim":500,"dtype":"sparse","metric_type":"ip","index_param":{"term_id_limit":2001,"window_size":50000,"store_positions":true,"max_positions_per_term":64}}' \
  --index_path /tmp/sindi_index \
  --search-query-count 100
```

(`dtype` 必须是 `"sparse"`，`dim` 是词表上限/sparse vector 的命名维度)
(`--search-query-count 100` 是必要的，避开 argparse 默认值 bug)

### 4. 搜索压测（baseline vs proximity）

```bash
# Baseline (无 proximity)
./build-release/tools/eval/eval_performance \
  --datapath /root/project/vsag/bench/sindi_bench.hdf5 \
  --type search \
  --index_name sindi \
  --create_params '同 build' \
  --search_params '{"sindi":{"n_candidate":50,"proximity_weight":0.0}}' \
  --index_path /tmp/sindi_index --topk 10 \
  --search-query-count 100

# With proximity boost
./build-release/tools/eval/eval_performance \
  --datapath /root/project/vsag/bench/sindi_bench.hdf5 \
  --type search \
  --index_name sindi \
  --create_params '同 build' \
  --search_params '{"sindi":{"n_candidate":50,"proximity_weight":0.3,"proximity_boost_multiplicative":true,"proximity_candidates":10000}}' \
  --index_path /tmp/sindi_index --topk 10 \
  --search-query-count 100
```

**输出关键指标**：QPS、avg latency、recall@10、内存。proximity 的 recall 用对应 GT 评估才准。

### 5. 跑 sindi_test.cpp 里的 benchmark（无需数据文件）

```bash
build/tests/unittests "SINDI Proximity Benchmark" --success
```

输出 brute-force GT vs SINDI 搜索的 recall、QPS、rank changes、内存估算。

### 已知坑

- **eval_performance 必须传 `--search-query-count`**（即使是 build）：上游 argparse 有 bug，`default_value(100000)` 是 int 但 `.scan<'i', uint64_t>()`，命令行不传时 `parser.get<uint64_t>()` 会抛 `bad_any_cast` 然后 SIGABRT。
- **dtype 必须是 `"sparse"`**（不是 `"float32"`）。
- **QEMU 下 Python 跑大量循环可能 segfault**：容器外生成 HDF5 更稳。
- **小数据集内存对比失真**：1000 docs 的 444KB→2.09MB 看起来增量 +370%，但基数中很大一部分是固定开销（指针数组、label table 等），不能直接外推。真实场景（50K docs/window）摊销固定开销后，位置数据增量回归到 issue 估算的 +150% 量级。
