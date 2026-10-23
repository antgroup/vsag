# VSAG Lite 基准复现

本页说明构建、数据准备、测量和校验流程；[完整参数与历史重放命令](../../../en/src/development/lite_benchmarks.md)与英文指南共用同一组程序。
[验收入口](../../../../../lite/benchmark/ACCEPTANCE.md)对应四项产出要求，[综合报告](../../../../../lite/benchmark/ACCEPTANCE_REPORT.md)解释全部正负结果；不能把不同版本的最好数字拼成当前版本矩阵。

## 目录与边界

- `lite/benchmark/`：数据准备、Lite 消费程序、运行器、校验器和回归测试。
- `lite/benchmark/full/`：独立构建原生 Full 消费程序，不能用旧 Lite 代替。
- `lite/benchmark/results/`：当前 E10、E11、E12 的报告、原始压缩包和校验清单。
- [历史归档入口](../../../../../lite/benchmark/HISTORY.md)：E1、E9 的全部原件、固定提交和恢复步骤；不是删掉较差数据。
- `ENABLE_BENCHMARKS` 默认关闭；支持能力、架构和 API 由[Lite 功能文档](lite_first.md)解释。

## 构建与测试

在 Linux x86_64 仓库根目录运行。需要 C++17、CMake、Ninja、Python3/pip、固定版本的 Catch2 源码；CPU/RSS 测量还需要 `/proc`、`/dev/shm`、`taskset` 和 `/usr/bin/time`。编译和输出使用仓库之外的新目录。

```bash
BENCH_ROOT=$(pwd -P)
BENCH_WORK=$(mktemp -d /tmp/vsag-lite-check.XXXXXX)
printf 'Validation workspace: %s\n' "$BENCH_WORK"
if ! python3 -m venv "$BENCH_WORK/venv"; then
  python3 -m pip install --target "$BENCH_WORK/venv-bootstrap" virtualenv==20.26.6
  PYTHONPATH="$BENCH_WORK/venv-bootstrap" python3 -m virtualenv --no-download \
    --app-data "$BENCH_WORK/virtualenv-cache" "$BENCH_WORK/venv"
fi
. "$BENCH_WORK/venv/bin/activate"
python3 -m pip install numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0
BENCH_BUILD="$BENCH_WORK/lite-build"
cmake -S "$BENCH_ROOT/lite" -B "$BENCH_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_TESTS=ON -DENABLE_RABITQ_LITE_BACKEND=ON -DENABLE_BENCHMARKS=ON
cmake --build "$BENCH_BUILD" -j2
ctest --test-dir "$BENCH_BUILD" --output-on-failure
export VSAG_MIXED_PROBE="$BENCH_BUILD/lite_acceptance_mixed_cpu_acceptance"
test -x "$VSAG_MIXED_PROBE"
python3 -c 'import numpy, h5py, pyarrow'
python3 -m unittest discover -s lite/benchmark -p 'test_*.py'
```

系统缺少 `ensurepip` 时，上述 fallback 只在外部工作目录安装 virtualenv；不修改系统 Python。离线时可给 CMake 指定与仓库固定版本匹配的 `FETCHCONTENT_SOURCE_DIR_CATCH2`。缺依赖或探针导致的 skip 不能算成功。

## SIFT、GIST、Cohere 数据准备

三者都保留。SIFT 输入为 ANN-Benchmarks HDF5 的 train/test；GIST 输入为记录来源的 little-endian fbin，不能把 fvecs 改名冒充；Cohere 输入为含 id 和 768 维 emb 的 train/test Parquet。原始数据本体不提交，来源、准备参数、归一化和输入哈希必须保留。

```bash
BENCH_SIFT_SOURCE=/absolute/path/to/sift-128-euclidean.hdf5
BENCH_GIST_BASE=/absolute/path/to/base-100k.fbin
BENCH_GIST_QUERY=/absolute/path/to/queries.fbin
BENCH_COHERE_SOURCE=/absolute/path/to/cohere-parquet-directory
python3 "$BENCH_ROOT/lite/benchmark/prepare_sift.py" \
  "$BENCH_SIFT_SOURCE" "$BENCH_WORK/sift" --counts 10000 100000 --queries 100 --k 10
c++ -std=c++17 -O3 -I"$BENCH_ROOT/src" \
  "$BENCH_ROOT/lite/benchmark/prepare_gist.cpp" -L"$BENCH_BUILD" \
  -Wl,-rpath,"$BENCH_BUILD" -lvsag-lite -o "$BENCH_WORK/prepare-gist"
"$BENCH_WORK/prepare-gist" "$BENCH_GIST_BASE" "$BENCH_GIST_QUERY" \
  "$BENCH_WORK/gist" 100000 100
python3 "$BENCH_ROOT/lite/benchmark/prepare_cohere.py" \
  "$BENCH_COHERE_SOURCE" "$BENCH_WORK/cohere" --counts 10000 100000 --queries 100 --k 10
```

每个规模重新生成真值，不能复制完整库真值。SIFT、GIST、Cohere 的历史算术协议不同；Cohere 先 FP64 归一化再存 FP32。混合 CRUD 程序另外计算变值后初末状态的 FP64 真值。重复查询同一批向量不等于新增独立查询。

## 当前 Lite 真实改值 CRUD 与原始归档

使用干净功能检出、实际提交和新编译库；不要给旧库贴上新 SHA。下面会记录三种存储、每次调用 wall/user/system CPU、初末质量、每轮 RSS、快照大小/哈希和运行环境。

```bash
BENCH_FEATURE=/absolute/path/to/clean-feature-checkout
BENCH_FEATURE_SHA=$(git -C "$BENCH_FEATURE" rev-parse HEAD)
BENCH_FEATURE_BUILD="$BENCH_WORK/feature-build"
python3 "$BENCH_ROOT/lite/benchmark/run_mixed_cpu_acceptance.py" \
  --prepare-build --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD"
BENCH_CPU=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
BENCH_RESULT="$BENCH_WORK/mixed-results"
python3 "$BENCH_ROOT/lite/benchmark/run_mixed_cpu_acceptance.py" \
  --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD" \
  --dataset "SIFT=$BENCH_WORK/sift/scale-10000" \
  --dataset "GIST=$BENCH_WORK/gist/scale-10000" \
  --dataset "Cohere=$BENCH_WORK/cohere/scale-10000" \
  --trials 3 --rounds 20 --cycles 250 --cpu "$BENCH_CPU" --resident \
  --output "$BENCH_RESULT"
python3 "$BENCH_ROOT/lite/benchmark/verify_mixed_cpu_acceptance.py" "$BENCH_RESULT"
python3 "$BENCH_ROOT/lite/benchmark/pack_mixed_cpu_acceptance.py" "$BENCH_RESULT"
BENCH_UNPACKED="$BENCH_WORK/unpacked-mixed"
test ! -e "$BENCH_UNPACKED"
mkdir "$BENCH_UNPACKED"
PYTHONPATH="$BENCH_ROOT/lite/benchmark" python3 -c \
  'from pack_mixed_cpu_acceptance import unpack; from pathlib import Path; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  "$BENCH_RESULT/raw.tar.gz" "$BENCH_UNPACKED"
python3 "$BENCH_ROOT/lite/benchmark/verify_mixed_cpu_acceptance.py" "$BENCH_UNPACKED"
```

`--prepare-build` 拒绝已有构建目录，并在成功编译及源码复核后生成 build-source receipt；测量再次检查源码、CMake、实际加载库和哈希。这是防误用的本地来源记录，不是签名认证。

正式参数是 degree16/build ef128/query ef512/k10，三次进程按存储模式轮换。`--resident` 保留原索引，加载副本只用于持久化校验。快照二进制是工具临时产物，大小和哈希保留在原始审计记录。吞吐只计对应 API 调用，不包含真值、验证和记录开销。

小规模工具链 smoke 可准备 128/256 条 base、5 条 query，使用 `--trials 1 --rounds 2 --cycles 3`。它验证运行、校验、打包与解包，不替代 10k/100k 正式性能结果。

## 原生 Full、包体与历史质量重放

精确对照使用 `lite_dataset_benchmark` 与 `full_dataset_benchmark`；Full SDK 必须包含头文件、原生库及 `lib/cmake/vsag`，仅有运行时库的 receipt 安装目录不是完整 SDK。另有 Full HGraph/RaBitQ 消费程序；相同 ef 不等于相同质量。

当前 Full 需要显式 `--source-sha`、新构建目录、匹配依赖源码和 ANTLR 归档。`--strict-fp` 是记录过的 O3 隔离 overlay，不是原生默认 Ofast；不得冒称整套 Full 测试通过或相同执行 ISA。包体工具需要当前 Lite/Full receipts、完整运行时依赖及受信消费程序源码归档，保存清单和迁移 smoke。

历史 Full、包体、质量重放各有固定源码／归档／库身份，必须按[英文完整命令](../../../en/src/development/lite_benchmarks.md#guarded-historical-full-package-and-quality-replay)获取匹配依赖，不能去掉保护来运行当前版本。冻结 .990 是历史诊断目标，不是自行新增的项目验收门槛。

## 指标与验收解释

- Save/Load 默认是未控制文件缓存的 warm 测量，不能称严格冷启动。
- 进程 peak RSS 包含输入、真值和加载副本，不等于索引独占内存；累计分配量也不等于 RSS。
- P50/P99、吞吐、recall、CPU user/system 必须注明样本、单位、版本与聚合方式，保留全部重复。
- 包体结论只适用于报告限定的应用＋ELF 运行时＋loader＋许可文本，不推广到 SDK、wheel 或容器。
- 当前 E12 中 GIST 召回下降必须保留；原始审计 PASS 不等于质量稳定、小时耐久、并发安全或整个项目完成。

## 可选的 CRUD 退化诊断

向 mixed runner 传入 `--resident --checkpoint-every 10`，记录初始、每十轮及最终的质量／重建检查点。旧命令及初末 quality 格式不变。每个检查点对当前 FP32 输入向量重新计算 FP64 真值，用完全相同的当前向量和 ID 新建图，与原地维护索引比较；存储、degree16、构建预算128及检索预算512保持一致。`checkpoint-*-neighbors.csv` 保留有序 ID、距离及真值 ID；queries 文件记录命中及诊断延迟，state 文件记录累计修改 API 次数与重建耗时；raw 验证及打包／解包覆盖这些文件。

Linux 工具在单线程子进程中进行重建和真值分配，父进程等待完成再继续 CRUD；不替换原对象，子进程重建缓冲不计入父对象分配器／RSS。fork、调度和缓存预热仍会扰动时间，不能把本组时延和内存与旧协议拼成一次结果，也不能把检查点计时当作严格的维护图／重建图性能比较。每次重复独立保留。RaBitQ 新建还会重新训练模型，单看差距不能区分图维护与量化模型因素。不据此设置最低召回门槛。

既有逐轮 Save/Load 检查验证显式快照查询结果一致，不代表逐操作落盘、原子替换、fsync 或崩溃恢复。容量回收与图质量修复是不同问题；RSS 平稳既不能证明零碎片，也不能证明召回稳定。

## 隔离 Update 与 Remove/Add

`--mutation-mode update|replace|reinsert` 必须配合 `--resident --checkpoint-every N`。
三者分别为只Update改值、Remove/Add改值、Remove/Add原值；前两组变值序列相同，第三组逻辑数据始终不变。
默认 `all` 旧协议兼容。每cycle分别1/2/2次修改，不能把调用数差异当作性能收益。
三组保持相同预触达样本缓冲；正式检查点固定ef512，末轮另存ef=N诊断 `ceiling.csv`，不冒称调参优化。
校验器检查操作纯度、顺序、ID、次数、真值交集、重建对照及全预算结果。
全部曲线、来源、独立真值复算和原始归档见
[GIST九组隔离实验](../../../../../lite/benchmark/results/crud-isolation-20261022/)。不改生产修复或承诺逐操作落盘。
