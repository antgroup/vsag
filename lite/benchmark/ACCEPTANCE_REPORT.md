# VSAG Lite / OSPP 综合产出与性能验收报告

本次发布验证：Cohere synthetic Parquet回归独立执行通过，覆盖归一化、真值/ties、
sourceID、offset、重复ID/非法向量/范围和已有输出拒绝；非真实Cohere性能重跑。
临时依赖NumPy1.26.4/h5py3.11.0/PyArrow17.0.0，无系统环境修改。默认Python套件
19pass/3skip(其中Cohere可选依赖缺失)；有依赖时19pass/2skip，并额外执行Cohere
import-time fixture，不能将fixture说成新增一组正式unittest计数。固定报告目标
gitblob、原始包解包及production parity/diffcheck通过；没有新增C++编译、SAN、
coverage、Full性能或社区CI结论。

证据汇总日期：2026-10-14。本文汇总已有实验，不是一次新的性能测量。
结论：功能、架构、复现工具和主要指标已具备；定义范围内的包体、部分加载和
进程内存收益明确。但动态高维图质量、统一同质量原生 Full 对照、长时耐久及
社区 CI 尚未完全闭环，不宣称整个项目完成。

## 1. 版本与比较口径

功能归 PR #2904，源码 `74a9420c771e75dea5e56324775d8db5d1472459`；验收工具与
证据归 #2926。最终 Lite 的15个生产编译单元仓库依赖哈希与 `dbb2fe0` 全部一致，
新库SHA256 `3171d0d7e86866f02096db130f1937eb13c5115cbe993d4a6feb2e9987774330`，
412512bytes，与历史被测 Lite 库一致。这是工件身份关联，不是重跑历史 Full。

| 证据组 | 被测版本/范围 | 可支持的结论 |
| --- | --- | --- |
| 20261014 final resident | 当前74a9420库，SIFT/GIST10k | 最新真实交错CRUD、CPU、延迟、质量、持久化 |
| 20261012 fresh Full | 两边源dbb2fe0；Full空目录243核心对象重新构建 | 同degree/build/query预算的原生Full对照 |
| 20261011 三数据集600queries | Lite a492b6f/记录3298616，Full d18c82a增量安装 | SIFT/GIST/Cohere历史对照，不能伪称最新Full |
| 100k与优化专项 | 各报告自己的parent/candidate/source/library | 历史规模覆盖或特定真实代码修改收益 |
| 图候选 | 隔离候选，不是当前默认实现 | 初始收益及改值后失败，不能作生产成绩 |

旧Lite不是Full；同ef不等于同质量；重复执行数不是独立查询数；原始审计PASS不等于
质量PASS；候选最好成绩不等于交付成绩。表中分位数通常是三个进程分位数的中位数，
不是池化分位数。不同报告参数、查询、缓存、版本不同，不能拼接或相加优化百分比。

## 2. 四项产出与功能范围

| 产出要求 | 已有交付 | 仍需说明/验收 |
| --- | --- | --- |
| 可评审PR、独立构建/配置、CRUD/Search/SaveLoad、配置测试说明 | #2904独立CMake、LiteAPI、回归测试、中英文指南 | 社区相关编译/测试最终执行与评审 |
| 架构：保留/裁剪、索引布局、内存、API兼容 | canonical lite_first.md/lite_rabitq_design.md | 独立API/格式，不完全兼容Full ABI/快照 |
| >=2数据集/规模、全部性能指标、复现及raw | SIFT/GIST/Cohere10k/100k，分版本报告/哈希/校验器 | 各指标存在但尚非统一最新版本全矩阵 |
| 对原生Full至少包体/加载/内存一项明显改善且正确/效果稳定 | 定义应用包收益、部分加载/RSS收益、功能与质量验证 | 高维动态ANN质量及同质量成本未闭环 |

支持Add、Remove、真实Update、Search、过滤及Save/Load；有FP32/FP16单层图与可选
RaBitQ8，默认关闭RaBitQ，固定训练模型；分配失败有回滚测试。默认BruteForce是精确
检索，但ANN图模式不能按精确算法的效果承诺。需外部串行，不支持此证据下的同时
读写保证。单层图不是HGraph；不宣称mmap/零拷贝/原子Save、零泄漏或零碎片。

## 3. 数据集与实验覆盖

| 数据 | 维度 | 已有规模 | 主要证据 |
| --- | ---: | --- | --- |
| SIFT |128|10k/100k|三存储、Full、真实CRUD、I/O/路由优化|
| GIST |960|10k/100k|高维Full、冻结质量、真实CRUD、图维护候选|
| Cohere |768|10k/100k|归一化、三存储Full、RaBitQ/反向邻接/SaveLoad|

Cohere原始metric为cosine：FP64范数归一化后存FP32，按存储后的向量用FP64平方L2
计算真值，按ID打破平局。源ID与索引行号区分保存，并审计归一化舍入对cosine TopK
的影响。不是把cosine真值直接混用为任意L2真值。最新270k resident矩阵仅SIFT/GIST；
Cohere已有历史证据，但尚未做这一版同协议复测。准备脚本见prepare_cohere.py。

## 4. 最新真实交错CRUD：条件与正确性

SIFT/GIST各10000base/100已有queries；degree16/build128/query512/k10；CPU0，
单线程，OMP/OpenBLAS/MKL1；FP32/FP16/RaBitQ8各三个新进程，共18进程，循环轮换模式。
每进程20rounds*250cycles，原索引不替换为Load副本。Update→query→Remove→query→
Add新值→query；新值FP32(.875*current+.125*original donor)，要求与旧FP32输入不同。
量化表示可能不变，不声称每次编码必变。每进程5000update/5000remove/5000add/
15000query/20save/20load；总270000mutation及270000mixedquery。

liveID/数量/唯一/顺序/finite距离检查，初始最终FP64真值，每轮SaveLoad有序ID/距离
一致；原始次数/哈希/分位数/重复快照/线程状态审计PASS。仅100distinctqueries重复，
不是270k独立查询。每API getrusage扰动计时，CPU微秒分辨率；吞吐排除真值/验证/记账。

### 4.1 查询：us，三进程中位数

| 数据/存储 |P50|P99|调用吞吐/s|初始→最终Recall@10|
| --- | ---: | ---: | ---: | --- |
|SIFT FP32|168.767|203.985|5852|.993→.997|
|SIFT FP16|163.817|193.596|6046|.993→.997|
|SIFT RaBitQ|198.285|237.685|4979|.987→.994|
|GIST FP32|488.239|683.158|2025|.971→.970|
|GIST FP16|372.233|544.429|2633|.971→.969|
|GIST RaBitQ|443.992|598.468|2227|.970→.963|

RaBitQ P50比FP16慢SIFT约21.0%、GIST约19.3%；比FP32 SIFT慢17.5%，GIST快9.1%。
FP16是本负载最快。GIST RaBitQ质量下降0.7百分点。SIFT召回升高是数据变化下的结果，
不是维护算法被证明更好。同参数并不使图/质量相同。

### 4.2 CRUD：us P50/P99

| 数据/存储 |Add|Remove|Update|
| --- | ---: | ---: | ---: |
|SIFT FP32|69.249/85.279|12.670/19.599|76.358/106.078|
|SIFT FP16|66.619/81.788|12.290/19.150|73.788/103.057|
|SIFT RaBitQ|101.527/119.128|22.580/34.450|109.308/178.307|
|GIST FP32|176.446/254.444|20.459/49.050|222.696/434.291|
|GIST FP16|154.308/204.795|16.660/44.969|183.236/374.832|
|GIST RaBitQ|288.745/343.792|36.059/101.169|338.504/1194.750|

| 数据/存储 |Add/s|Remove/s|Update/s|
| --- | ---: | ---: | ---: |
|SIFT FP32|14566|74682|13029|
|SIFT FP16|15096|76916|13457|
|SIFT RaBitQ|9936|44167|8878|
|GIST FP32|5594|45020|4311|
|GIST FP16|6461|54094|5174|
|GIST RaBitQ|3452|24345|2501|

整批秒级不等于单次秒级：GIST RaBitQ Update P50 .339ms/P99 1.195ms。Remove更快，
Add/Update要维护邻居和编码，成本较高；以上观察不代替完整因果profile。

### 4.3 Build/Save/Load：ms

| 数据/存储 |Build|Save P50/P99|Load P50/P99|
| --- | ---: | ---: | ---: |
|SIFT FP32|544.480|3.774/4.072|2.934/4.034|
|SIFT FP16|523.526|3.206/3.414|2.612/3.299|
|SIFT RaBitQ|602.295|4.728/5.691|3.478/3.890|
|GIST FP32|1386.653|16.100/16.698|16.741/17.388|
|GIST FP16|1336.927|10.161/11.343|6.807/10.536|
|GIST RaBitQ|1679.405|7.564/8.800|5.229/5.700|

同进程未控制缓存/无fsync，非冷启动或耐久落盘。RaBitQ高维持久化较快，但SIFT
并不优于FP16；构建均更慢。Build每进程只有一次，不能把其P99解释为丰富构建尾样本。

## 5. 最新CPU资源与内存

整进程包括构建、真值、持久化副本及记账。各列独立取三次中位数，不要求相加严格
等于wall。平均GNUtime CPU约99–100%单核，不是全服务器100%或瞬时峰值。

| 数据/存储 |user s|system s|wall s|进程peak KiB|query user/system ms(15000calls)|
| --- | ---: | ---: | ---: | ---: | ---: |
|SIFT FP32|5.43|.07|5.51|40328|2556.202/17.901|
|SIFT FP16|5.25|.04|5.30|32872|2479.734/6.206|
|SIFT RaBitQ|6.48|.03|6.52|30644|3007.702/5.842|
|GIST FP32|19.91|.53|20.48|166328|7390.116/22.028|
|GIST FP16|17.44|.24|17.66|146860|5694.645/9.533|
|GIST RaBitQ|20.51|.15|20.64|144552|6738.520/15.478|

自愿切换0–2，非自愿13–96；major faults多数0，首SIFT FP32有2。不是完整硬件
cycles/instructions/IPC/cachemiss/branchmiss采集。历史perf权限受限；未改权限设置。
peak含输入/原始向量/真值/预触碰记录/构建和两份索引，不是index-owned内存。

## 6. 同源码新构建原生Full对照

dbb2fe0两边；Full空目录243核心对象，ccache OFF；SIFT/GIST10k各100queries，
degree16/build128/query512/top10、CPU0/thread1；36builders/108loaders/3600queries。
Full为HGraph小配置，Lite单层图；Full Ofast/g、Lite O3，非全flag/ISA对齐或同质量。
以下所有斜杠为Lite/Full。warm fresh-process Load不是冷磁盘。

| 数据/存储 |Recall|P50 us|P99 us|Load ms|loaded进程RSS KiB|
| --- | ---: | ---: | ---: | ---: | ---: |
|SIFT FP32|.993/.987|168.475/274.703|196.215/440.798|3.830/5.549|12592/32860|
|SIFT FP16|.993/.987|165.645/274.803|188.634/442.968|3.005/3.515|10128/30352|
|SIFT RaBitQ|.987/.979|201.824/335.451|354.951/383.320|3.979/2.709|8540/29824|
|GIST FP32|.971/.992|446.936/529.685|604.513/704.161|17.722/29.785|45156/66112|
|GIST FP16|.971/.992|357.390/434.567|442.128/517.226|10.903/16.019|26368/46928|
|GIST RaBitQ|.970/.989|439.427/520.206|695.172/648.962|7.125/9.536|16700/38184|

| 数据/存储 |Build ms|Save ms|snapshot bytes|builder peak KiB|
| --- | ---: | ---: | ---: | ---: |
|SIFT FP32|546.987/791.412|3.759/3.220|6560064/5766017|27724/42340|
|SIFT FP16|535.058/811.327|3.269/2.499|4000064/3144565|22684/37072|
|SIFT RaBitQ|605.148/930.795|2.667/1.989|2960672/2203866|21620/38256|
|GIST FP32|1330.220/1528.252|16.428/14.110|39840064/39808753|125344/108884|
|GIST FP16|1321.432/1494.708|10.661/8.465|20640064/20148057|103840/85404|
|GIST RaBitQ|1607.813/1384.545|4.891/4.671|11284416/10690982|103856/102252|

loaded总RSS六组更低、Load五组更快、P50六组更低；但snapshot六组更大/Save六组更慢，
GIST builderpeak三组更高、GIST召回三组更低、SIFTRQ Load更慢/GISTRQ P99更慢。
loaded-minus-before-load增量RSS六组Lite更高，不把Full运行时基线误算成索引布局收益。

## 7. 包体：明确收益与部署边界

可运行C++应用包包含consumer、递归ELF/loader、aliases、launcher及已记录notices，
双方同样strip；打包/解包/异目录三storage共六次smoke，maps显示包内依赖。
Lite dbb库对历史Full d18c82a增量库，不是新构建Full，也非SDK/wheel/container。

| 指标 |Lite|Full|减少|
| --- | ---: | ---: | ---: |
|普通文件payload bytes|6689348|99246762|93.260%|
|实际tar.gz bytes|2664623|26932509|90.106%|
|ELF数含consumer/loader|7|17|—|

新构建Full实验另测共享库闭包6147776/98428088bytes，减少93.754%；不含应用/launcher。
包体与共享闭包是不同指标。不是正式发布或完整许可认证，redistribution需单独审查。

## 8. 包含Cohere的三数据集历史Full对照

Lite a492b6f/记录3298616，Full d18c82a原生增量安装；10k/600observedqueries，
query来源[400,1000)，degree16/build128、Liteef512/Fullef128，54builders/162loaders。
不是盲测/同budget/同recall。以下按Lite/Full列出，与源报告Full/Lite顺序相反。

| 数据/存储 |Recall|P50 us|P99 us|
| --- | ---: | ---: | ---: |
|SIFT FP32|.997500/.975833|179.876/101.168|229.834/129.657|
|SIFT FP16|.997500/.975833|175.045/99.098|215.044/120.428|
|SIFT RaBitQ|.992167/.965833|221.495/124.947|293.924/152.906|
|GIST FP32|.965500/.971000|640.474/228.754|788.971/269.954|
|GIST FP16|.965500/.970667|478.488/185.136|625.675/216.866|
|GIST RaBitQ|.961000/.967000|514.677/206.014|680.253/263.992|
|Cohere FP32|.987333/.981333|634.125/213.963|723.671/250.993|
|Cohere FP16|.987000/.981167|451.840/176.255|549.106/209.554|
|Cohere RaBitQ|.983500/.978500|480.107/197.964|629.573/245.394|

九组Lite查询均慢，但预算/质量不同，不是严格同质量慢1.77–2.96倍的证明。
Cohere资源Lite/Full如下；历史FP16 Save劣势不能冒称当前未测最终库数值。

| Cohere10k |FP32|FP16|RaBitQ|
| --- | ---: | ---: | ---: |
|Load ms|14.130/23.918|9.081/13.434|6.804/8.506|
|loaded RSS KiB|37612/58384|22616/43104|14760/36208|
|Build ms|1466.238/1705.426|1346.881/1701.127|1764.078/1534.054|
|Save ms|14.889/11.282|36.004/6.893|5.813/4.783|
|snapshot bytes|32160064/31957337|16800064/16228737|9363552/8737006|
|builder peak KiB|104528/95124|85880/76320|85480/87376|

## 9. 100k历史真实改值

SIFT a126a53，每storage一次新进程/固定顺序，非统计重复。30k真替换/90kAPIcalls，
三storage270kcalls；查询仅前后，不交错。当前FP64真值/SaveLoad结果验证PASS。

| SIFT100k |FP32|FP16|RaBitQ|
| --- | ---: | ---: | ---: |
|initial/final Recall|.988/.989|.988/.989|.975/.983|
|Build ms|12968.672|11300.322|14053.185|
|CRUD ms /calls per s|10801.636/8332|9340.009/9636|12542.166/7176|
|query P50/P99 us|539.707/685.103|508.177/645.244|559.947/693.763|
|Save/Load ms|53.870/55.963|123.550/48.467|94.032/68.997|
|snapshot bytes|65600064|40000064|29600672|
|peak KiB|252600|202436|206688|

RaBitQ文件比FP16约小26%/FP32约小55%，但查询CRUD较慢、召回低.6百分点，peak略高
于FP16。GIST/Cohere100k反向邻接优化为单before/after配对，90kAPIcalls/dataset：

| 数据 |CRUD old/new ms|降幅|mutation/s old/new|initial→final Recall|
| --- | ---: | ---: | ---: | --- |
|GIST|329487.429/35222.392|89.31%|273/2555|.859→.853(两版本一致)|
|Cohere|337975.452/30192.778|91.07%|266/2981|.937→.953(两版本一致)|

反向邻接避免全图行扫描，输出/快照一致，但GIST质量未修。Save GIST76.121→151.724ms/
Cohere71.148→138.516；Load77.177→126.204/72.444→114.930；build也略慢。
这些是旧Lite/新Lite，不是Full，也不是长期交错；一配对不能宣称普遍统计保证。

## 10. 真实结构优化账本

| 修改 | 有据成绩 | 限制 |
| --- | --- | --- |
|原生RaBitQ SIMD补充码、query sum/coarse IP复用|S/G/C10k ef128 P50 163.546→134.206/468.390→231.125/422.981→237.605us，即-17.9/-50.7/-43.8%|3交替trial，recall不变；舍入变化可能存在；非Full/真实CRUD |
|数组I/O与heap准入|SIFT100k7pairs load54.624→37.929ms(-30.6%)，query731.655→557.188us(-23.8%)，P99949.149→737.253，recall.975|热缓存/100旧queries，非Full |
|反向一致性连续缓冲且保留全部验证|GIST/Cohere100k5pairs Load104.207→93.943/95.253→87.440ms；Save71.492→62.060/62.051→53.993|loaded-stateSave非postCRUD整流程；snapshots哈希一致 |
|路由仅算消费字段/入口特化|GIST100k7triples均值1092.762→1072.303us(-1.87%)，P501100.791→1082.083，P991281.466→1271.118，recall.859|21kexecutions仅100distinctqueries，未修质量 |

阶段插桩SIFT100k约transform<1%/route85%/rerank15%，不是完整精确分割或所有分布
结论。测量扰动与未计开销保留。各项百分比不可相加；起点/预算不同。

## 11. 质量失败和未采用候选

GIST10k前100选参/后900验证，诊断.990非OSPP合同目标；27krecordedqueries不是
27kdistinct。独立FP64 Top10全部匹配。表为frozen ef/Recall/P50/P99us：

| mode |Lite|Full|
| --- | --- | --- |
|FP32|2048/.989556/1522.077/1728.693|512/.992889/529.496/645.763|
|FP16|2048/.989222/1056.080/1261.224|512/.992444/440.098/495.046|
|RaBitQ|2048/.985111/1309.674/1373.472|2048/.991222/1463.168/1549.347|

Lite三模式校准过/验证失败；RQ较快但质量不达目标，不是同质量胜利。该900queries
已观察，不能再调参后称盲集。严格O3/实际AVX512诊断候选GIST新cohort初始：
原Liteef512 .9676/P50428.238，candidate256 .9946/333.120，Full512 .9931/505.537。
候选初始快34.11%，但3whole-index真实变更后GIST .992→.973；加ef1024恢复.997/
P50约1046us。SIFTcandidate1.000→.995。候选未作为生产默认通过，不用最好静态
成绩掩盖动态失败；FullCRUD未在此pilot重跑。

## 12. 耐久、内存与读写边界

20round resident证明秒级窗口非小时级。历史同库RSS before-persistence轨迹KiB：

| mode |initial|round5|round10|round20|
| --- | ---: | ---: | ---: | ---: |
|SIFT FP32|27408|40400|40400|40400|
|SIFT FP16|24872|32904|32904|32904|
|SIFT RaBitQ|24784|30140|30404|30668|
|GIST FP32|125188|128636|128788|128836|
|GIST FP16|106408|146876|146876|146876|
|GIST RaBitQ|98224|111484|111748|112012|

RQ晚期264KiB增长各repeat复现，不直接证明泄漏也不证明零泄漏；allocator可保留
副本页面，public无capacity指标。尚无小时/天、多线程同时读写、多比例/业务排队、
最新三数据集100k长时交错或大量独立queries稳定P99证据。

## 13. 测试、门禁与下一步

当前精简ReleaseON5/OFF4、3Fullconsumer编译、Python19pass2skip、raw解包PASS；
本次添加Cohere测试单独验证，不冒称新Full性能。历史最终Lite行覆盖3072/3233=
95.0201%，是Lite生产并集非全仓/branchcoverage；历史ASan/UBSan/rollback记录按
各自版本归属。coverage编译通过≠coverage统计通过。最近核实社区Actions需审批、
2926缺kind标签；本文不实时监控，不称全绿或以后不会失败。

优先：补全三数据集证据入口→诊断并修复GIST初始/动态图质量→冻结最终版本进行
同质量/编译/选定ISA的生产Lite与原生Full矩阵→100k/Cohere长时resident真实CRUD→
针对路由/精排/维护/持久化做有因果profile与配对验证→处理真实社区CI日志和门禁。

## 14. 固定证据索引与复现

以下报告保留每组完整条件、原始结果/commands/hashes/负面结果；不是所有历史探索
逐项复审清单。RAW_MANIFEST.json提供800a822历史文件与c072a42归档工具路径/哈希。

- [当前resident及raw](results/final-version-20261014/README.md)、[当前全API汇总](results/final-version-20261014/summary.json)。raw4483066bytes/152files，SHA256 `6dd27f9d041df80d619602f1a494de39f4c8d4f778fc4719a6d025fc21e67391`；解包后运行verify_mixed_cpu_acceptance.py raw。
- [Fresh原生Full](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/full-fresh-equal-r1-20261012/README.md)。raw489003bytes/702members，SHA `13f08607e3efb79a1e752f460085795a85f0ebb56eeb23aba5e89b14ed019b81`。
- [运行包](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/runtime-package-r1-20261012/README.md)。inventory/maps/license/manifest各有记录。
- [历史resident/RSS](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/resident-mixed-cpu-pr2904-20261012/README.md)。raw4390665bytes/152members。
- [冻结质量失败](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/quality-target-r2-20261012/README.md)。raw3729401bytes/597members，SHA `bac805acedd7032ba2580be616634ca64207078657687ddbc348facc12997498`。
- [严格O3候选与CRUD失败](https://github.com/nw9663644-eng/vsag/blob/800a82232da3a91dfc90229c3a8fdbf4fa9b428a/lite/benchmark/results/aligned-fp32-full-r2-20261012/README.md)。raw22397608bytes/228members，SHA `c38e5a8405cdfda0cd20fd2ef9d2ea9b41f4d11cd70f796b267f6344bd4784e4`。
- [三数据集600queries历史Full](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/current-full-10k600-20261011/README.md)。含全部Build/Save/Load/snapshot/peak与1041成员原始记录。
- [SIFT100kCRUD](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/sift-production-100k-crud-20261011/README.md)。270k三storage调用/当前真值/操作分位数。
- [GIST/Cohere100k reverseCRUD](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-production-100k-crud-20261011/README.md)。单pair/90kcalls数据集，baseline库hash缺失也披露。
- [共享原生内核](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/shared-rabitq-kernels-20261009/README.md)。原生SIMD舍入/FP32FP16负面结果也保留。
- [I/O/route stages](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-stages-20261009/README.md)。诊断插桩与未插桩配对分开。
- [连续校验缓冲](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-validation-scratch-20261011/README.md)。SaveLoad40rows/43commands及lint边界。
- [GIST100k重复查询](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-routing-repeat-20261011/README.md)。21kexecutions/100distinct，recall未修。

正式脚本保留source/hash guards；历史重放使用对应固定checkout和外部数据，不能删除
guard后冒充最新结果。一般依赖/构建说明见README.md；原始数据不因评审精简而删除。
本报告没有新增性能结果，不认证许可、所有平台并发能力或未来CI必过。
