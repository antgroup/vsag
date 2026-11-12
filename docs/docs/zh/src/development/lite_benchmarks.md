# VSAG Lite 基准复现

[验收协议](../../../../../lite/benchmark/ACCEPTANCE.md)、[综合报告](../../../../../lite/benchmark/ACCEPTANCE_REPORT.md)和[证据清单](../../../../../lite/benchmark/results/manifest.json)共同说明条件与结论。功能与 API 见[Lite 指南](lite_first.md)。测量绑定记录中的源码、库、编译配置和输入；集成提交通过测试不等于已有数据被重新测量。

阅读综合报告时，先看四项产出的要求、证据、结果与结论；第 6 章提供规定指标到原始归档成员的索引，
第 4–5 章解释资源取舍与尚未闭环的动态质量。包体分解来自已有 inventory 复算，不是新增测量。
Update-only、Remove/Add-only 和同最终向量重建是下一步核对方法，不是已完成修复或全部验收通过的声明。

## 构建与回归

在干净的 Linux x86_64 仓库运行，需要 C++17、CMake、Ninja、Python3/pip 和仓库固定版本的 Catch2。资源测量需要 `/proc`、`/dev/shm`、`taskset` 和 `/usr/bin/time`。构建和输出放在仓库外的新目录。

工具版本遵循当前仓库的 `AGENTS.md` 与构建说明：格式检查和静态检查使用 clang-format 20、clang-tidy 20；独立 Lite 和消费程序仍使用 C++17。当前工程验证环境与历史性能测量环境分别记录，不能互相替代。

```bash
BENCH_ROOT=$(pwd -P)
BENCH_WORK=$(mktemp -d /tmp/vsag-lite-accept.XXXXXX)
python3 -m venv "$BENCH_WORK/venv"
. "$BENCH_WORK/venv/bin/activate"
python3 -m pip install numpy==1.26.4 h5py==3.11.0 pyarrow==17.0.0
BENCH_BUILD="$BENCH_WORK/consumer-build"
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

离线构建可指定匹配的 `FETCHCONTENT_SOURCE_DIR_CATCH2`。缺依赖或缺探针造成的 skip 不能计为通过。`ENABLE_BENCHMARKS` 默认关闭，显式启用后才构建测量程序。

## 输入与真值

准备合法可用的 SIFT HDF5、GIST little-endian fbin 和 Cohere Parquet。显式指定路径，不能把 fvecs 改名冒充 fbin。完整命令见[英文输入步骤](../../../en/src/development/lite_benchmarks.md#inputs-and-truth)，两种语言使用同一组程序与参数。

每个规模重新计算子集真值。保留来源、预处理、输入哈希和查询选择；更换查询集合产生的是新测量。报告中的 GIST 1000 行只有 998 个不同查询。动态质量另由 mixed probe 按当前 FP32 向量计算 FP64 穷举平方 L2 真值。Cohere 初始归一化，改值后不再归一化，不能将其动态平方 L2 质量写成持续余弦检索质量。

## 源码绑定的重复 CRUD

测量源码必须在显式指定的提交上且工作区干净。准备步骤只接受新构建目录，并生成 `build-source-receipt.json`，核对源码、翻译单元、参数和实际库。收据用于一致性核对，不是签名证明。

```bash
BENCH_FEATURE="$BENCH_ROOT"
BENCH_FEATURE_SHA=$(git -C "$BENCH_FEATURE" rev-parse HEAD)
BENCH_FEATURE_BUILD="$BENCH_WORK/source-bound-build"
BENCH_DATASET=/absolute/path/to/prepared/scale-10000
BENCH_RESULT="$BENCH_WORK/resident-result"
BENCH_CPU=0
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --prepare-build --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD"
python3 lite/benchmark/run_mixed_cpu_acceptance.py \
  --feature-repo "$BENCH_FEATURE" --feature-sha "$BENCH_FEATURE_SHA" \
  --build "$BENCH_FEATURE_BUILD" --dataset SIFT="$BENCH_DATASET" \
  --resident --trials 3 --rounds 20 --cycles 250 --cpu "$BENCH_CPU" --output "$BENCH_RESULT"
python3 lite/benchmark/verify_mixed_cpu_acceptance.py "$BENCH_RESULT"
python3 lite/benchmark/pack_mixed_cpu_acceptance.py "$BENCH_RESULT"
```

使用可用 CPU 并记录实际配置。各数据集分别传入正确标签和目录，输出不得复用。固定存活数量稳定性使用同配方 GIST、`--rounds 100`，不因修改次数多就称为长期服务验证。

`--checkpoint-every` 和 `--mutation-mode` 是可选质量诊断；额外真值、重建和 fork 会改变计时范围。功能检查、诊断曲线与正式性能不能混为同一结果。

当前 mixed probe 保留既有混合公式；若 donor 相同或 FP32 舍入使结果未变，便将第 0 维设为 1（当前绝对值不超过 0.5 时），否则设为 0。该确定性回退保证重复向量和小数据集也发生真实修改；刻意保持原向量的 `reinsert` 对照不使用此回退。校验器核对所有模式（包括默认 `all`）的操作、轮次、ID 和查询顺序。这些工具修复不改写历史测量或其源码身份。

## Full 与运行时包

新编 Full 工具显式清空 CMake 的 C/C++ compiler launcher 并关闭 ccache，阻断环境变量继承的编译缓存。运行时包 smoke 等待探针和全部打包 ELF 依赖出现在 `/proc/PID/maps`；只看到加载早期的可执行文件不足以证明依赖完整。映射文件按 ELF 文件头识别，普通输入文件的映射不计入，预期闭包之外的 ELF 映射会被拒绝。Full RaBitQ 输入读取同时拒绝不完整的维度头和向量数据。

Full 消费程序在 `lite/benchmark/full` 独立构建。比较运行器还需要显式匹配的 helper 仓库和源码引用，包体程序需要固定哈希的 consumer 源码包；这些都是必须记录的输入，不能替换后绕过校验。

[英文 Full/包体命令](../../../en/src/development/lite_benchmarks.md#native-full-and-runtime-package)说明 `build_fresh_full_reference.py`、`run_pinned_full_acceptance.py`、`measure_runtime_package.py` 及对应校验器。新源码比较要求匹配的新编 Full/Lite 收据；固定引用复核在没有 fresh Full 收据时显式传入 `--reference-provenance`，仍核对库和头文件哈希。对于历史功能版本 dbb2fe0，运行脚本和离线校验器都要求 Lite 库的 SHA-256 与固定值一致。

Full O3 实验配置与默认 Ofast 不能混用；相同 ef 不保证同质量。运行时包仅涵盖应用、递归 ELF 依赖、loader、launcher 和 notices；结果归档大小不代表产品包大小。SDK、Python、容器、系统、数据和快照不在该包体范围内。

## 原始证据复核

Full SDK 保留 CMake 实际生成的版本化 SONAME 别名，不假定固定版本。Bundle 解包仅支持普通文件及同目录、最终指向归档普通文件的别名链；父目录跳转、以文件或别名作父路径、悬空/循环链接均在写入前拒绝。

正式结果使用三个稳定目录：`resident-crud`、`full-comparison`、`dynamic-stability`。清单记录测量身份、字节数和 SHA256。先核对外层成员，再在新目录安全解开内部归档；不能改写原始数据来适应另一份输入。

```bash
BENCH_EVIDENCE="$BENCH_WORK/evidence"
PYTHONPATH=lite/benchmark python3 -c \
  'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
  lite/benchmark/results/resident-crud/raw.tar.gz "$BENCH_EVIDENCE"
for BENCH_NAME in SIFT GIST Cohere; do
  PYTHONPATH=lite/benchmark python3 -c \
    'from pathlib import Path; from pack_mixed_cpu_acceptance import unpack; import sys; unpack(Path(sys.argv[1]), Path(sys.argv[2]))' \
    "$BENCH_EVIDENCE/$BENCH_NAME/raw.tar.gz" "$BENCH_EVIDENCE/$BENCH_NAME/raw"
  python3 lite/benchmark/verify_mixed_cpu_acceptance.py "$BENCH_EVIDENCE/$BENCH_NAME/raw"
done
```

动态稳定性另解开 `measurements/raw.tar.gz`，运行 mixed verifier；Full 解开 `full/raw.tar.gz` 后传入匹配数据路径，再对 `package` 目录运行包体验证。独立 FP64 oracle 在计时之外执行；固定快照质量校准需要显式输入与 Full 收据，其诊断阈值不是项目验收门槛。

该校准工具清除继承的动态加载环境变量，并在计时前校验 Lite/Full 构建与查询程序实际解析的库路径及哈希；新结果保留纳入 manifest 的链接收据，供离线复核。

## 稳定性与内存解释

功能一致、检索质量、内存平稳和并发安全分别判断。重复有限查询集合不产生同等数量的独立长尾样本。修改后质量必须使用当前真值；显式 Save/Load 一致性不是每操作 WAL/fsync 或断电恢复。测量调用在外部串行执行。

报告保留实际召回下降。后半窗口 RSS 平稳不能抵消质量退化，也不能证明零碎片、索引独占峰值或无限期稳定。RSS 包含输入、真值/采样缓冲及分配器保留页；getrusage 高水位可能受启动父进程影响。新 VmHWM 字段不能回填未采集的结果。暖加载也不能称为严格冷启动。

合成数据 `main.cpp` 稳定性 CSV 的 `resident_rss_kib` 在 CRUD/查询后、Save/Load 前采样；稍后采集的 `peak_rss_kib` 是进程累计高水位，两次高水位相减不是单轮峰值。保留原字段以兼容旧结果，也不与真实数据 resident 协议混用。SIFT 准备过程每条查询有一个 N×维度的差分缓冲，使用有界子集，不作为无界数据准备工具。

## 工具职责

混合 CRUD 的耗时和 CPU 合计显式沿用原记录的按顺序 binary64 逐项累加规则。
Python 3.12 改变了内置浮点 `sum`；记录与离线复核共同使用固定规则，保留精确比较和原始数据。

| 文件 | 职责 |
| --- | --- |
| `prepare_sift.py` | 准备 SIFT HDF5 子集、查询、真值和输入身份。 |
| `prepare_gist.cpp` | 读取 little-endian GIST fbin，准备子集输入和真值。 |
| `prepare_cohere.py` | 校验 Cohere Parquet 向量与 ID，归一化初始数据并生成子集/真值。 |
| `main.cpp` | 合成数据的 Lite 生命周期、图/稳定性和独立进程加载消费程序。 |
| `dataset_main.cpp` | 真实准备数据的 Lite 生命周期与查询消费程序。 |
| `mixed_cpu_acceptance.cpp` | 驻留/隔离 CRUD、当前 FP64 真值、查询与进程资源采样。 |
| `run_mixed_cpu_acceptance.py` | 生成源码绑定构建收据并调度重复 CRUD，要求显式输入输出。 |
| `verify_mixed_cpu_acceptance.py` | 独立核查完整运行、清单、操作和质量记录。 |
| `pack_mixed_cpu_acceptance.py` | 校验、打包原始证据，并安全解包到新目录。 |
| `full/CMakeLists.txt` | 针对显式安装的 Full SDK 构建独立消费程序。 |
| `full_main.cpp` | 原生 Full BruteForce 合成数据生命周期基线。 |
| `full_dataset_main.cpp` | 原生 Full BruteForce 真实准备数据基线。 |
| `full_rabitq_dataset_main.cpp` | 原生 HGraph FP32/FP16/RaBitQ 真实数据消费程序。 |
| `build_fresh_full_reference.py` | 从干净的匹配源码编译 Full，记录编译器、库与头文件身份。 |
| `run_pinned_full_acceptance.py` | 依据匹配源码与构建收据，执行重复 Full/Lite 对照。 |
| `verify_pinned_full_acceptance.py` | 复算对照汇总并校验输入和工件。 |
| `measure_runtime_package.py` | 测量限定原生运行时依赖闭包，并验证包迁移后的执行。 |
| `verify_runtime_package.py` | 校验包清单、字节、动态链接、哈希和测量记录。 |
| `query_fixed_snapshot.cpp` | 查询固定 Lite/Full 快照，输出有序邻居和逐查询耗时。 |
| `run_quality_target.py` | 可选的固定历史源码质量校准，不是默认当前源码验收入口。 |
| `verify_quality_target.py` | 审计该校准的查询、真值、时间及来源记录。 |
| `quality_target_fp64.cpp` | 固定快照质量核对使用的独立 FP64 穷举平方 L2 oracle。 |
| `check_quality_target_oracle.py` | 在正式计时外运行独立 oracle 并比较结果。 |
| `test_prepare_cohere.py` | 预处理校验、归一化、ID 和损坏输入回归。 |
| `test_prepare_sift.py` | 有限 SIFT 子集、重复规模、错误形状与输出前预检查回归。 |
| `test_mixed_cpu_acceptance.py` | 构建身份、资源/生命周期、隔离修改和核心 CRUD 证据回归。 |
| `test_pack_mixed_cpu_acceptance.py` | 安全打包、成员校验及损坏档案回归。 |
| `test_pinned_full_acceptance.py` | Full/Lite 收据、输入预检查及核心对照证据回归。 |
| `test_runtime_package.py` | 运行时闭包、安全解包、新源码预检查和包体验证回归。 |
| `test_quality_target.py` | 可选校准/oracle、来源身份和无效输入回归。 |
| `test_benchmark_contracts.py` | 消费程序行为、必需依赖、构建/CI、文档和证据导航回归。 |
