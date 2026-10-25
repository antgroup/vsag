# 官方模板复用：保持求和顺序的修复距离批量试验

这是E17之后的独立私有overlay，不修改#2904、默认生产源码、API或持久化格式。
本组只有纯修改计时，没有新增真实混合CRUD/QPS/P50/P99或召回率测量。
控制是已有r2邻接重选候选，不是默认生产策略；不能与E17最快进程拼成一组成绩。

本轮没有证明显著或全面降本：改值Remove/Add三对略快，Update只有1/3对更快，原值重插基本持平；不采纳为默认实现。

## 1. 复用方法与边界

参考[官方共享batch4模板](https://github.com/antgroup/vsag/blob/73eaa98bdafb67881af3702040d10150eadec40b/src/simd/kernels/compute_batch4.h)。
原生batch4与Lite的AVX512单点距离使用不同累加展开，不能原样替换后声称数值不变。
本候选用四组累加器的Traits适配，保持原单点的每组FMA和最终归并顺序，批量计算首个已选邻居与四个候选的距离。
只在FP32、已选AVX512分派、持有连续向量且维度为64倍数时启用；FP16、其他ISA/维度和借用build源走原距离路径。
候选、排序、选边、回填、入边保障和事务回滚规则不改；查询路径不改。
达到边数上限前提前停止时可能预先计算少量原本不会用到的距离，因此不假定工作量必然减少。

## 2. 正确性与安全验证

- Release与ASan/UBSan各4352个float位模式对照：17个维度、64组输入、每组四距离；包含整块/尾部、零值、subnormal和有限输入导致平方距离溢出的情况。
- 两种配置各通过既有5项Lite测试，实际ldd绑定均验证为候选库；format/tidy15通过。
- 三种960维真实短修改序列分别在ASan/UBSan下运行，末轮快照与r2逐字节身份相同；原始源码、命令、日志和SHA收据归档。
- 18个纯修改进程的全部初始/最终快照SHA与原r2一致。此处没有重新计算召回，不把图等价的推断包装成新的召回实测。

验证只覆盖本组输入/配置，不承诺任意ISA、所有数据或无限长期行为。

## 3. 三种负载、三对重复的完整成本

GIST10000×960、FP32、平方L2，degree16、build ef128/query ef512/k10；CPU0单线程。
每进程100轮×250cycles；Update每cycle一个改值调用，replace为改值Remove/Add，reinsert为原值Remove/Add。
各负载三对新进程交替先后顺序，共750000次修改；无交错查询或每轮Save/Load。
循环wall/CPU包括配方与记账，不是API分位数或端到端业务吞吐。

| 负载 | r2 wall中位数（s） | batch wall中位数（s） | 变化 | r2 CPU（s） | batch CPU（s） | batch更快对数 |
|---|---:|---:|---:|---:|---:|---:|
| update | 7.843173 | 7.812856 | -0.39% | 7.842944 | 7.812620 | 1/3 |
| replace | 12.046940 | 11.899776 | -1.22% | 12.046560 | 11.899455 | 3/3 |
| reinsert | 12.951521 | 12.951807 | +0.00% | 12.951120 | 12.951376 | 2/3 |

上表“变化”是两个wall中位数之比，不是逐对变化的中位数。逐对百分比变化中位数：update +0.22%；replace -1.16%；reinsert -0.15%。尤其Update有两对变慢，不能据独立中位数之比宣称稳定改善。

三对每次的原值：

| 负载 | 对次 | r2 wall（s） | batch wall（s） | 变化 |
|---|---|---:|---:|---:|
| update | 1 | 7.843173 | 7.812856 | -0.39% |
| update | 2 | 7.784465 | 7.803085 | +0.24% |
| update | 3 | 7.844082 | 7.861077 | +0.22% |
| replace | 1 | 12.051608 | 11.899776 | -1.26% |
| replace | 2 | 11.976358 | 11.837752 | -1.16% |
| replace | 3 | 12.046940 | 11.997392 | -0.41% |
| reinsert | 1 | 12.971908 | 12.951807 | -0.15% |
| reinsert | 2 | 12.934248 | 13.001850 | +0.52% |
| reinsert | 3 | 12.951521 | 12.862711 | -0.69% |

以上均为观察值，不承诺统计显著性或跨负载稳定加速；不取最快一次作为验收成绩。
raw里沿用旧探针的side=cache通用候选标签，identity的candidate_kind明确为exact-order-first-neighbor-batch，本组没有距离缓存。
无论表中变化正负，均不将本候选采纳为默认生产代码；实际混合CRUD与多数据集成本还需独立验证。

## 4. 实现成本、失败记录与复现

候选共享库423000 bytes，控制库418408 bytes。
这是共享库编译产物大小，不是整包体；新增临时距离数组仅在一次repair内保留，不是持久化索引字段。
没有新增进程峰值/RSS测量，不借用E17的内存表充作本候选成绩。

初次静态检查拒绝空fallback和Traits契约命名；已提供真实累加兜底，命名标注仅限模板所要求的成员名，不禁用分析器或全局检查。
恢复控制器曾因自有构建目录存在而停止；数值探针的异常边界/花括号/循环规范也经修正后重新验证。
所有首次源码和失败日志完整保存在first-attempt/、resume-attempt/、numeric-attempt/及controllers/，不是静默删除失败记录。

先按archive.json校验raw.tar.gz，再用安全解包函数核对manifest每个成员；公开回归测试复算全部时间表并核对数值/安全/快照收据。
在独立的ccf8252检出和外部build目录复现，保持默认源码、编译选项不变，仅在隔离构建替换graph_backend.cpp并额外编译repair_batch_avx512.cpp；
实际编译/链接参数在commands.json，模板/traits/控制源在reproduction/。先跑数值和既有Release/SAN测试，再用probe.cpp计时。
换机需显式映射控制脚本中的绝对路径，不应删除版本检查。不能把实验库冒充已有干净生产构建收据。

```bash
LD_LIBRARY_PATH="$R2_BUILD" taskset -c 0 "$PROBE" "$DATASET" "$R2_OUTPUT" replace 100 250 fp32
LD_LIBRARY_PATH="$BATCH_BUILD" taskset -c 0 "$PROBE" "$DATASET" "$BATCH_OUTPUT" replace 100 250 fp32
python3 -m unittest discover -s lite/benchmark -p 'test_*.py' -v
```

[E17真实混合CRUD及此前否定结果](../repair-cost-reuse-20261025/README.md)仍独立保留。
本组不代表默认长期召回退化已经解决，也不代表RaBitQ、FP16、ARM或社区全部CI通过。
