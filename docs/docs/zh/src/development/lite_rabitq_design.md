# Lite 8bit RaBitQ 设计与交付边界

状态（2026-10-09）：按个人开发分支 `bcaf9c31e89abe811de84ffad3e1f63dadaa6882` 的真实实现核对。公开候选接口已经接通；性能验收和上游正式合入尚未完成。本文替代早期 RABITQ 枚举、VSAGLT01 v4 的提案，不代表导师或维护者已批准候选格式。

## 已定设计

| 项目 | 首次交付的选择 |
| --- | --- |
| 接入 | 非空 BruteForce 显式调用 BuildGraph(VectorStorage::RABITQ8, max_degree, ef_search)；ENABLE_RABITQ_LITE_BACKEND 默认 OFF，使用时开启 |
| 算法 | 平方 L2 估计；3bit 过滤 + 5bit 补充；4轮变换、6轮坐标调整、error rate 1.9、训练 seed 47 |
| 模型 | 一个固定质心与翻转模型，存编码和外部 ID；不在线重训，不常驻原始 FP32 向量 |
| 查询 | 沿用 Search、外部 ID 过滤，并支持 SearchWithOptions 单次预算；不支持并发调用 |
| 增删改 | 预备编码与局部回滚日志；验证后完整编码相同则不改图 |
| 持久化 | 保留独立 VSAGLQ01 v1 候选格式；不混用 VSAGLT01 v1/v2/v3 或实验 VSLRBQ01 |
| 上游提交 | 完成剩余验收后再获得提交授权；功能 PR 不带实验原始归档 |

这些是当前候选的工程设计，不是新增的官方验收指标。不加入按数据量自动切换后端：由调用方显式决定何时承担建图成本。

## 源码依据与复用范围

| 源码 | 依据 |
| --- | --- |
| include/vsag/lite/index.h、src/lite/index.cpp | Index 所有权、tl::expected、构建成功才替换后端；未开启 RaBitQ 返回 UNSUPPORTED_INDEX_OPERATION |
| src/quantization/rabitq_quantization/rabitq_quantizer.cpp | 官方 FastEncodeRaBitQ 坐标调整、中心化 split IP 和距离公式 |
| src/simd/kernels/rabitq_pack.h、rabitq_compute.h | 官方分平面打包与独立 ISA 过滤内核；不引入 Full allocator/datacell 依赖 |
| src/lite/rabitq_codec.h | 模型、正逆变换、预备编码、连续编码容器、调用方所有的解码缓冲 |
| src/lite/rabitq_filter_ip*.cpp | Generic 回退；完整 CPU 特性检查与逐文件 ISA 编译 |
| src/lite/rabitq_graph_state.h | 图查询、过滤与事务式 CRUD |
| src/lite/rabitq_backend.cpp、rabitq_snapshot.h | API 异常转换、实际格式及严格恢复 |
| src/lite/rabitq_backend_test.cpp、lite/example/rabitq_example.cpp | 生命周期、故障、损坏快照回归及公开使用示例 |

复用的是官方简单算法和内核，不声称完整复用了 Full 的16簇融合模型、图实现或存储 ABI。量化距离和重建向量都不是精确原始向量。

## 构建、模型及内存

当前安全限制：1 <= N <= 1000000、0 < D <= 2^20、2 <= degree <= 64、ef >= degree，同时检查容器容量及有限算术。Add 不能超过候选的一百万条限制。安全上限不代表已验证的实际规模能力。

现有构建按行读取源索引训练质心和翻转模型并编码，再调用 FP32 图构建取得初始拓扑，最后生成 RaBitQ 状态并成功发布。训练保留原始行顺序，分别执行有限值检查和质心累加；至多使用单行解码 scratch，不复制整个 N×D 矩阵。后续 Add/Update 使用冻结模型和量化邻居发现。初始构建保留原始源向量和临时浮点邻接表，仅用于构建的图按行读取源向量，不再拥有第二份 FP32 矩阵；峰值仍需独立测量；后续数据分布漂移仍需质量验收。

训练行指针在读取下一行 scratch 前用完。拓扑构建复用现有 FP32 图算法，同步只读源向量；成对距离使用两个独立行缓冲。借用图和行指针不逃出工厂函数，仅返回自有邻接表并转换成拓扑，在可变 RaBitQ 状态构造前释放临时邻接表。原始源索引保留至成功发布，中间分配失败时仍保持完整。

每条编码平面占 8*ceil(D/8) 字节，六个 float 元数据占24字节；模型占4D字节质心及4*ceil(D/8)字节翻转信息。运行时还包含外部 ID、哈希槽位表、邻接表容量及查询/日志缓冲。8bit 不能直接推导整个进程 RSS；构建峰值和长期 CRUD 内存必须分开测。

重建使用调用方缓冲，产生近似原空间向量。Save 必须直接保存精确模型与编码，不能解码再编码。seed 可复现性限于记录的实现/工具链；跨平台 Load 依靠已保存的翻转和质心，不承诺标准库随机分布完全相同。

## 查询路径

1. 检查 FP32 输入的维度和有限值，按固定模型变换、归一化。
2. 用3bit估计遍历，保留均匀入口和隐式槽位环；visited 防重复。
3. 无过滤时对保留候选用完整3+5bit重排；有过滤时为访问到且允许的外部 ID 计算完整估计，被拒 ID 仍可用于路由。
4. 按估计距离、外部 ID 排序返回。启发式 lower bound 不构成普适召回保证。

SearchWithOptions 的零预算沿用默认，预算受活跃数量和所需结果数约束；不改变建图、CRUD 和持久化默认配置。k=0 或删空后的索引返回空结果，过滤可能使结果少于k。

每查询变换归一化为 O(D)，每候选过滤/补充计算也与D相关，visited空间与N相关。图查询每次缓存 query_sum，完整3+5bit评分复用粗排过滤内积。每个四记录邻接批次评分前，参考原生 HGraph RaBitQ，使用只读缓存提示预取过滤码缓存行和元数据。不改变评分、遍历顺序、visited、过滤或持久化，也不分配共享查询缓冲。收益依赖维度、缓存及硬件，不承诺统一延迟改善。提高ef会增加工作量。

## CRUD、失败语义和限制

| 操作 | 当前实现 | 限制 |
| --- | --- | --- |
| Add | 验证、预备编码、预留容量、记录修改行、发布；异常回滚 | 扩容、邻居发现、日志仍有成本 |
| Update | 验证存在的ID、预备一次编码、比较平面和全部元数据，相同则跳过，否则记录编码/行后重连 | 编码相同不等于原输入完全相同；真实大幅更新可能影响模型和图质量 |
| Remove | 物理末尾搬槽；修复不对称入边；记录编码/ID/行并保留移除的map节点供回滚 | 默认不建反向边索引，可能 O(total_edges) 扫描，不能说 O(degree) |
| 内部反向边实验 | 单独可选路径 | 事务仍可能使用完整clone，不属于这次公开默认交付 |

Add/Update/Search 返回 tl::expected。分配/容量错误映射 NO_ENOUGH_MEMORY，普通输入/ID错误 INVALID_ARGUMENT，受保护的其他操作异常 INTERNAL_ERROR；损坏格式 INVALID_BINARY，写流沿用 READ_ERROR。Remove 的 bool 无法区分缺失ID与事务失败，false时原状态保留；必须说明限制，不偷偷改签名。

Build失败保留原后端，事务失败保留逻辑内容、拓扑和模型。Save失败可能留下部分写入流；原子文件替换需调用方临时文件写完后替换。没有新增并发和文件原子保证；已有分配故障测试不代表覆盖所有外部平台分配器。

## 候选快照的实际布局

除8字节magic外整数为little-endian u64；float是little-endian IEEE float32位模式，ID存int64原位模式。

| 顺序 | 内容 |
| --- | --- |
| 1 | VSAGLQ01、version=1、D、N、centroid_count=D、flip_count=4*ceil(D/8) |
| 2 | D个质心float、flip_count个字节 |
| 3 | 每条六float：norm/code_norm/error/filter_norm/filter_error/lower_bound_error；随后3*ceil(D/8)过滤字节和5*ceil(D/8)补充字节 |
| 4 | max_degree、ef_search、ID count=N、N个外部ID |
| 5 | offset_count=N+1、neighbor_count=E、N+1个offset及E个邻居槽位 |

总大小：96 + 4D + 4*ceil(D/8) + N*(40 + 8*ceil(D/8)) + 8E 字节，不包括运行时哈希/容量和文件系统开销。文件按记录交错写入，内存则为独立连续过滤/补充数组。

v1隐式固定算法常量，并没有独立保存seed/error rate/bit字段；旧提案说法已废止。Load用精确模型恢复，不重训。未来改变固定编解码/变换语义必须新增可识别版本及兼容测试，不能重新解释v1。不会仅为统一magic而迁移到VSAGLT01 v4。

加载在分配编码前检查剩余可寻址流长度，验证维度/数量/模型布局、元数据有限性和范围、ID唯一、offset、度数、邻居范围、自环/重复、截断和尾部字节。恢复的是编码和拓扑，不是原向量。浮点格式保持不变；关闭后端和错误magic的分类按Index::Load实际逻辑处理。

## 已有证据与未完成验收

以下数字引用已提交实验，并非本次重新跑实验。

| 记录（lite/benchmark/results下） | 已有结果 | 不能证明 |
| --- | --- | --- |
| integrated-load-final-20261009，10k | 三分布fresh-process loaded RSS较FP32低约41–66% | 冷IO、构建峰值、长期真实更新、普遍延迟优势 |
| integrated-100k-pilot-20261009 | 固定degree16/ef128 RaBitQ召回：SIFT .946、GIST .748、Cohere .889；loaded RSS更小 | 默认预算下质量达标 |
| integrated-100k-budget-grid-20261009 | 已观察查询选择：SIFT ef512 .975；GIST ef2048 .924；Cohere ef2048 .967 | 盲验收、精确等召回对照、重复稳定耗时、最优 |
| CRUD journal回归 | 既有公开生命周期、roundtrip和有限分配故障回滚通过 | 验收配置下长期真实整向量变化的性能质量 |

SIFT/Cohere .95、GIST .90是实验预声明floor，不是社区统一最终要求。增大ef能达到这些已观察floor，但查询比浮点更慢。RAM暖缓存加载不是冷启动磁盘实验；100k预算网格仅聚合质量，最终需要保存逐查询返回邻居，不能拿旧ef128原始结果替代高ef证据。

## 接下来按顺序完成

1. 完成可审计验收runner：复用缓存SIFT128/GIST960/Cohere768输入，10k与100k固定单核，冻结源码/配置；保存逐查询ID/距离/hits和CPU/wall，按真实变更的活跃向量集合独立重算GT。
2. 长混合Add/真实Update/Remove，覆盖整向量变化、固定模型漂移、反复ID churn、过滤及变更后持久化；记录吞吐/P50/P99、阶段召回、事务错误及fallback成本，不在运行中增预算；并发不属于本期保证。
3. 同单核、声明质量目标下比较Lite FP32/FP16/RABITQ8与Full；分别看初建和单条增删改、构建/查询CPU、包体/快照、load-only RSS及峰值，耗时用重复fresh-process。1M作为后续容量刻画可选，不替代小规模10k/100k必需验证。
4. 只优化明确成本：先缓存查询query_sum和重排所需filter值，再做标量/SIMD差分及三分布对照；若构建峰值突出，再验证不保留重复图的编码/拓扑转移，同时保留失败后发布语义。反向边另评估内存和Remove收益。这里是计划，没有冒称已实现。
5. 固定文档/安装示例/格式兼容/default-OFF；重跑Release、ASan/UBSan、format/tidy15、候选库覆盖率>=90%、损坏/故障及依赖检查；明确授权后才提取功能到PR。

## 与导师确认的边界

- 最低Recall@10、允许P99退化及内存降幅是多少？以质量对齐比较，不能只看RSS。
- 第一版本是否按单线程、固定模型和bool Remove限制交付？
- 独立候选格式及default-OFF是否适合上游，还是需要正式维护格式决策后再提交？

已有CRUD正确性、单核证据和query context优化可继续执行，不必为等待指标新增无关功能。暂不加入IP/cosine、PCA/MRQ、磁盘补充、mmap/零拷贝、在线重训、原向量精确重排、融合多簇和ARM SIMD。不在验收完成前宣称完成日期或全局最优。

### 2026-10-10 CRUD 独占阶段诊断

基于bc3e9bd在单线程RAM隔离副本插桩，生产不变。固定10k/600历史查询、degree16/维护128/查询512、三遍全ID真实UpdateRemoveAdd，GIST/Cohere CPU独占占比：nearest（含路由/评分）29.20/30.91%，入边扫描19.08/22.21%，邻居评分排序20.15/17.90%，变换后码重建13.77/11.93%，编码6.28/5.97%。时钟有开销，仅诊断，非生产收益或Full对照。有序结果/快照SHA与历史NONE control一致，Recall仍.811167/.908167。证据`lite/benchmark/results/rabitq-mutation-profile-20261010`含源码、绑定、校准/作用域夹具、180000操作/2400真值独立审计。生产测试覆盖率继承，非新跑。下一固定质量预算优先路由/扫描；100k/交错CRUD/freshFull对齐/coldIO未验收。

### 2026-10-10 拒绝批量变换码解码候选

真实生产候选复用原生RecoverOrderSQ移位掩码恢复思路，每字节批量重建8维；原浮点除法、图政策、预算、API/model/snapshot不变。256编码/尾维/未对齐/四norm逐位回归和candidate Release6/SAN5通过。固定10k GIST/Cohere600查询、NONE/NONE、三交替pair、三遍全ID真实UpdateRemoveAdd，CRUD中位+0.18484%/+1.57367%，样本内全部pair变慢，无收益，不推断普遍回退或具体CPU原因。header/test撤回为675aed7逐字节一致，restoredRelease6/SAN5、最终format/tidy15通过，默认库SHA1a18379a不变。无fresh coverage/OFF/Full全套/Node成绩。证据`lite/benchmark/results/rabitq-block-decode-20261010`独立审计1080000操作/14400真值、paired有序结果/快照SHA、183raw成员30852726B及10验证退出码。默认Recall.811167/.908167未修复。下一检索已有原生Batch4内积评分在图维护中的复用，尚未实现/测量。终验未完成，PR2904/2926未改。


### 图维护 Batch4 复用（2026-10-10）

link/repair 合格邻居复用已有原生三位码 centered-IP Batch4 分派，每四个粗排内积合并，尾部仍标量；补充码距离公式、候选顺序、tie breaking、图预算/政策、API、快照及事务回滚不变。新增标量黄金对照覆盖0..65候选和维度尾部。固定预算10k GIST/Cohere各三组交替配对，CRUD中位耗时下降1.17%/1.95%，六pair均改善，快照SHA/有序ID/hex距离/hits完全相同。这只是小幅工作负载内观察，不是普遍或统计显著收益：GIST Update/Remove P50略升，最终Recall .811167/.908167仍低研究质量floor；不宣称查询或Full优势。Release6、ASan/UBSan5、fresh scoped coverage5、默认关闭4及format/tidy15通过，Lite行覆盖2862/3007=95.18%，不是整个Full覆盖率。首次仅测试float memcmp的lint失败改为显式字节表示后，全部验证重跑；生产不变、未重复计时。证据：`lite/benchmark/results/rabitq-maintenance-batch4-20261010`。100k、真实改值交错CRUD、fresh原生Full配置/SIMD/质量对齐及cold-I/O/完整包体终验仍未完成。


### 撤回入边分组扫描候选（2026-10-10）

隔离的真实生产候选仅把Update/Remove的逐项成员判定改为四项分组整数比较。固定预算10k GIST/Cohere各三组交替配对，CRUD中位耗时退化11.50%/14.05%，六pair均慢；header/test已byteexact恢复194bf86，上一轮Batch4优化保留。初始/最终ID、hex距离、hits和snapshotSHA相同，Recall .811167/.908167未修复。候选及恢复版Release6/SAN5通过，最终恢复源码format/tidy15通过；无freshCov/OFF/Full/Nodechecker声明。新test编译/预期错误在计时前修正，原日志保留。核实219个已缺失对象的编译清单/源码/include/编译器后，才清理闲置旧Full静态归档1.30GB，动态库/data/raw保留。证据：`lite/benchmark/results/rabitq-grouped-scan-20261010`。下一真正结构方向是反向行的局部journal及其内存成本，而非直接开启现有全图copy事务或重复此分组比较候选。

### 内部反向 Update 局部事务日志（2026-10-10）

反向邻接 Update 不再复制整个图，只备份被替换的编码及受影响出边、入边行。在修改入度计数或入边前
先备份；异常时以 vector swap 恢复，并从恢复的入边长度还原有效计数，无分配回滚。
CACHED 首次冷重建仍用整图复制隔离，反向 Add/Remove 也仍保留复制回退。公开构造默认关闭反向邻接，
没有新增公开参数，也不改变评分、预算或快照格式。已知内存账目补入反向向量的有效载荷及容量；
不是包含分配器开销或临时日志的 RSS。失败恢复保证逻辑状态，不保证 vector 容量完全相同。

固定 10k、600 查询、degree16/维护128/查询512、CPU0 单线程，三遍全 ID 真实整向量改值 Update，
每数据集三组交替配对：相对同样启用反向邻接、NONE 策略的整图复制对照，GIST/Cohere 的 Update 块
中位耗时降低 82.92%/85.15%。有序 ID、hex 距离、hits 和快照 SHA 全部配对一致。
这不是公开默认路径、混合 CRUD、查询或原生 Full 提速；Update-only 最终召回仍为 .858667/.935500，
研究质量门槛未满足。Release6、Sanitizer5、fresh scoped coverage5、默认关闭4、format/tidy15 通过。
内部故障注入覆盖四种状态各500位置，共347次实际分配失败，无状态、缓存重建或重试不一致。
证据目录 `lite/benchmark/results/rabitq-reverse-update-20261010` 保留可复现配置、原始结果和独立审计。
反向增删局部日志、公开路径内存/时间权衡、100k、交错 CRUD、fresh Full 对齐、cold I/O 和完整包验收仍待完成。

### 内部反向 Add 局部事务日志（2026-10-10）

反向 Add 复用新增/截断事务，只备份受影响出边、入边行，row_limit 排除新增槽位。失败时先恢复原入边及
有效计数，再截断新槽位。冷 CACHED Add 仍复制隔离，反向 Remove 本轮未改。公开默认仍关闭反向邻接，
评分、模型、预算、API 和快照不变；回滚保证逻辑状态，不保证容器容量完全一致。

固定10k/600查询、degree16/维护128/查询512、CPU0单线程，每个进程三遍全ID Remove→真实整向量改值Add，
两库各三组交替配对。整个增删块中位耗时：GIST156625.917→65534.431ms（-58.16%），
Cohere204033.785→62390.095ms（-69.42%）；Add P50中位2667.221→560.127us、3437.696→541.119us。
六对均快，但对照绝对耗时波动明显。Remove 实现相同，其计时变化不当成独立算法收益。
快照、有序ID/hex距离/hits全部配对一致；最终召回 .8495/.923333 属于 replace-only 工作负载，
不是质量修复，也不与 Update-only 混作改善；不代表公开默认或原生 Full 优势。

Release6、Sanitizer5、fresh scoped coverage5、默认关闭4、format/tidy15通过；内部增改四种状态各500位置，
共4000注入位置，Add/Update实际失败306/363次，无状态/重建/重试不一致。本轮还覆盖编码准备分配和保护标志。
Lite2906/3047=95.37%为限定源码覆盖率，非Whole Full。
证据 `lite/benchmark/results/rabitq-reverse-add-20261010` 独立审计720000操作/14400真值状态。
下一步反向Remove局部日志、公开路径内存/时间权衡及固定预算质量；100k、交错CRUD、fresh Full对齐、
cold I/O和完整部署验收仍待完成。

### 内部反向 Remove 局部事务日志（2026-10-10）

反向删除仅备份受影响出入边，不再整图复制；删除槽/末槽/删除目标先保存入边再改计数，擦除改号前备份行。
异常利用原容量恢复数组大小、原行、有效计数，并归还map节点。冷CACHED仍复制隔离，公开reverse=false/NONE、
模型、评分、顺序、degree/ef、API、快照不变。官方HGraph出入边定位仅作依据，槽位语义不同，未移植删除实现。
回滚保证逻辑状态，不保证容量一致。10k/600查询、degree16/维护128/查询512、CPU0单线程，各三组交替配对、
每进程三遍全ID删除/真实改值新增：增删块GIST65928.356→12563.964ms(-80.94%)、Cohere62249.974→10340.707ms(-83.39%)。
Remove P50/P99中位1571.716/2730.809→38.520/1169.654us、1489.659/2506.485→30.089/1042.167us。
六对均快；Add未改，计时变化非独立算法收益。快照、有序ID/hex距离/hits全一致，replace-only召回仍.8495/.923333，
质量未达标，非公开或Full收益。Release6/SAN5/freshCov5/OFF4/format-tidy15通过，Lite2924/3065=95.40%限定覆盖率。
Remove6000注入位置覆盖中间/末槽/单节点和四缓存状态，692实际失败，无状态/重建/重试/后续增改不一致。
测试整数转换lint在计时前修复，失败日志保留。证据lite/benchmark/results/rabitq-reverse-remove-20261010保存199原始成员，
独立审计720000操作/14400真值状态。下一公开路径内存/时间权衡、质量、100k/交错CRUD、fresh Full对齐、cold I/O/完整包验收。

### 当前反向邻接关闭/开启的验收权衡（2026-10-10）

同06ab591生产状态和当前public adapter、NONE保护，仅隔离overlay改变构造默认反向开关；本轮未改生产代码或默认。
10k/600查询、degree16/维护128/查询512、CPU0单线程，各三组交替配对、三遍全ID真实Update/Remove/Add，1080000操作：
混合块中位GIST25007.966→20511.163ms(-17.98%)、Cohere22422.324→17637.928ms(-21.34%)。
Add P50略涨267.354→270.933us、238.555→240.845us；所有配对初始/最终有序ID/hex距离/hits完全一致，
召回仍.811167/.908167，未满足质量门槛。

72次新进程内部加载对每份相同快照开关各测三次：10000节点/160000边额外入边logical/capacity1520000B。
已知容量GIST11444320→12964320B、Cohere9523456→11043456B，不含map节点/分配器/临时量。
减进程基线的resident KiB中位13766→16776、11970→14900；暖内部加载6.122→8.637ms、5.355→7.978ms。
RAM/page-cache暖数据非cold I/O；probe ru_maxrss可能含launcher继承HWM。完整builder峰值含输入矩阵和双索引，
微小变化不足证明维护内存更低。因此保留公开reverse=false，不只因提速就启用。

证据lite/benchmark/results/rabitq-reverse-tradeoff-20261010独立审计1080000操作/14400真值/72加载。
重跑Release6、新probe format/tidy15、语法/错误输入通过；生产SAN/coverage/OFF继承06ab591，未重跑。
模式拼写在计时前修正；计时后RAM probe不存在，仅重建作错误输入检查，无性能重跑。
下一调查入边分配/布局并保留journal回滚与快照兼容；quality、100k/交错CRUD、fresh Full对齐、cold I/O/完整包待验收。

### 内部入边槽位改为有检查的32位（2026-10-11）

参考官方src/basic_types.h的uint32_t InnerIdType，入边源槽位与回滚备份改uint32_t；外部ID/出边/计数/
评分/模型/预算/API/持久化格式不变。新增反向槽前检查溢出，max/max+1单测不需海量分配。公开reverse仍false。
对照5d633c9为64位、候选32位，两边reverse/NONE；10k/600查询、degree16/维护128/查询512、CPU0单线程，
各三组交替配对、三遍全ID真实UpdateRemoveAdd：GIST中位20327.354→20116.230ms(-1.04%)，
Cohere17470.224→17346.610ms(-0.71%)。六对均快，但小幅变化不作普遍/显著结论。有序ID/hex距离/hits/
snapshotSHA全pair exact，mixed召回仍.811167/.908167，质量未达标，非Full优势。
72次新进程同snapshot暖加载：160000边载荷1280000→640000B(-50%)，vector头240000B不变，
总incoming1520000→880000B(-42.11%)；knowncapacity GIST12964320→12324320B、Cohere11043456→10403456B，
不含map节点/分配器/临时量。减基线resident16808→15532KiB、14880→13662KiB，暖load8.949→8.434ms、
8.341→7.805ms；非coldIO，probeHWM可能继承launcher，builderpeak含输入矩阵及双索引，不算维护峰值改善。
快照大小不变。Release6/SAN5/freshCov5/OFF4/format-tidy15通过，Lite2936/3077=95.42%限定覆盖率非WholeFull。
已有故障注入/重试/继续增改及48混合缓存配置仍通过。证据lite/benchmark/results/rabitq-incoming32-20261011
独立审计1080000操作/14400真值/72加载/fresh覆盖及hash。下一入边分配/布局及固定预算quality；公开默认继续关闭。
100k/交错CRUD、fresh原生Full配置SIMD质量对齐、coldIO/完整包终验待完成，PR冻结。

### 大规模RaBitQ查询的独立访问标记

可变RaBitQ路由在每次查询入口选择visited表示：少于65536槽位保留字节数组，达到阈值使用64位字和官方VisitedList的TestAndSet位掩码逻辑。编译期特化消除逐节点的表示策略分支。临时缓冲只属于本次调用，不引入epoch池或常驻可变workspace；FP32/FP16生产路径、距离、图维护、公开参数和快照不变。

100k的visited载荷由100000降为12504字节；记录的每查询累计普通new申请量由170160降为82664字节（约-51.42%），不等于加载RSS或整进程峰值。固定预算有序结果全等，但最终探索性P50变化不能证明查询提速。全存储位图及逐节点动态策略原型未采用。原始配对结果、限制、fresh限定覆盖率和阈值跨越CRUD/过滤/SaveLoad消毒器回归见`lite/benchmark/results/query-visited-specialized-20261011`。

### 小图查询堆替换

字节visited的RaBitQ路由特化在best堆满时，将获准候选直接替换堆顶并一次下沉，避免先push再pop；保留原比较器与candidate队列入队顺序。位图visited和reference特化保持原操作。这是对Lite既有堆的扩展，不是移植原生StandardHeap的堆顶替换实现；FP32/FP16查询、公开参数及快照字节不变。

三个10k输入的公共API固定预算七交替fresh-process配对，SIFT/GIST/Cohere P50约降低3.15%/1.80%/2.40%，每查询少一次16384字节申请。每进程100历史query，不是通用尾延迟验收。GIST100k虽保持旧位图路径堆操作，P50仍回退约2.20%；原因未验证，不能用噪声否定。此改动针对小图CPU和临时申请成本，未修复100k召回，也不证明优于原生Full。精确数据、否定候选及验证范围见`lite/benchmark/results/rabitq-root-public-20261011`。

### 路由专用评分与大图无过滤特化

RaBitQ图路由只请求粗排距离和centered内积，不再计算未使用的误差下界；消费下界的过滤全扫描等调用仍默认保留原计算。达到65536槽位时，空用户过滤器在调用入口选择编译期无过滤路由，非空过滤器继续原回调和排序语义。距离公式、图、搜索预算、快照格式及FP32/FP16路径不变。

GIST100k在当前parent9c44640/新库/referencec55466b三库七轮轮换顺序中，每进程将100历史query重复十遍，平均查询延迟1092.762→1072.303us（-1.87%，7/7配对更快），P501100.791→1082.083us（-1.70%，6/7）。同轮新P50比c55466b低2.38%（7/7）。这支持在该工作负载上消除此前小幅回退，不代表所有100k性能已经验收；原100query短窗口P50仅改善0.21%，重复不是1000独立query或新质量样本。Recall仍为.859，图质量问题未解决。共享库文件395648→412512字节，本轮不声明peak/RSS/Full/完整包收益。全部数据和限制见`lite/benchmark/results/rabitq-routing-only-public-20261011`与`lite/benchmark/results/rabitq-routing-repeat-20261011`。

## Promoted implementation and historical evidence

The implementation promoted on 2026-10-12 is derived from personal source `a62e62e08db3f0d7da2e4439785feed6fcf90257`. Experimental diversity switches and diagnostic benchmark targets are excluded. Benchmark commands and `lite/benchmark/` paths below refer to that pinned experimental repository, not targets shipped in this standalone PR. The reports retain each measured source version: numbers from different stages are not a single latest-version acceptance matrix. See [the pinned reproducibility sources](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark) and [promotion evidence](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results). Current CPU-time/utilization, quality-matched native Full comparisons, cold I/O and hours-long interleaved changed-vector CRUD remain incomplete. Externally concurrent calls require caller serialization.

本轮移植的是已验证生产实现，不是整条实验分支。历史报告必须按各自源码版本读取；最新版本 CPU 时间、同质量 Full 对照、冷加载和小时级真实改值交错 CRUD 尚未完成，不宣称项目全部验收通过。
