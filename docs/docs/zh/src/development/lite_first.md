# 独立 VSAG Lite v0.1

> 第二阶段分支说明（2026-09-14）：默认 Index::Create(dim) 仍是精确 BruteForce，
> v1 快照字节格式不变。显式调用 BuildGraph(degree, ef_search) 成功后才切换为
> 独立的单层近似图；失败不修改原索引。图模式继续使用同一套 Add/Update/Remove/
> Search 接口；FP32 图 Save 写入 v2，FP16 图写入 v3，均包含向量、ID、参数和
> 邻接关系；Load 接受 v1/v2/v3。图快照与 Full VSAG 不兼容，不提供并发、校验和、
> SQ8 或 mmap。
> BuildGraph 后同样支持按外部 ID 过滤；图搜索可经过不允许返回的节点以保持连通性，
> 但这些节点不会出现在结果中。
> 本页余下内容说明首版默认 BruteForce 行为。


这是实验性的独立构建入口，以 FP32、L2 平方距离的精确 BruteForce 起步，
选择性复用源码，而不是条件裁剪完整库。目前面向 Linux x86_64、C++17 和项目
既定编译器基线，不改变默认 Full 构建。从仓库根目录运行：

```sh
cmake -S lite -B build-lite-first -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite-first -j2
ctest --test-dir build-lite-first --output-on-failure
cmake --install build-lite-first --prefix "$PWD/install-lite-first"
cmake -S lite/example -B build-lite-consumer -DCMAKE_PREFIX_PATH="$PWD/install-lite-first"
cmake --build build-lite-consumer -j2
./build-lite-consumer/lite_example "$PWD/new-example.lite"
```

安装后的示例只包含 `vsag/lite/index.h` 并链接 `vsag::lite`，不链接
`libvsag.so`。它展示 Add、Search、Update、Remove、Save 和 Load。每次运行
请使用新的快照路径。预期输出：

```text
added id=42 squared_l2=0
filtered id=7 squared_l2=48
updated id=42 squared_l2=0
removed id=42 remaining=1
loaded id=7 squared_l2=0
graph query id=7 budget=64
```

与 Full 的 make 入口不同，`cmake -S lite` 刻意隔离依赖。测试复用仓库固定版本
Catch2 v3.7.1，离线时可指定 `-DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/catch2-v3.7.1`。
现有 Catch2 用例已注册到 CTest，名称为 `lite_unit` 和 `lite_scale`。关闭测试时不下载 Catch2。
构建同时生成 `libvsag-lite.so` 和 `libvsag-lite.a`。共享库目标仍为 `vsag::lite`；
静态归档也会安装，供选择显式静态链接的嵌入式或离线使用方使用。

## 接口和所有权

`vsag::lite::Index::Create(dim)` 创建固定维度索引。Add、Update、Search、Save 和静态
Load 采用现有 `tl::expected` / `vsag::Error`，不调用 Full 全局日志；Remove 返回 bool。
输入必须是指定数量的有限 float32 值。调用需要外部串行化。结果拥有内存，不暴露内部槽位。

- 单条 Add 拒绝重复 ID；Update 拒绝不存在的 ID。
- 图模式 Update 在完成原有校验后，若 FP32 字节或编码后的 FP16 位模式与已存表示相同，直接返回，不改变邻接和快照。FP16 范围检查仍先执行；FP32 正零和负零的字节表示不同。同值 Update 不是图修复请求。BuildGraph 仅支持 BruteForce 到 Graph 的转换，不会原地重建已经成图的索引。
- Remove 对不存在的 ID 返回 false；删除后的外部 ID 可以重新新增。
- Add 失败不改变逻辑记录，但预留容量可能已经增加。
- Search 最多返回 min(k, Size()) 条记录，按 L2 平方距离、ID 升序排列。
  BruteForce 在计算距离前按外部 ID 过滤。图搜索为保持连通性，仍可对被拒绝的
  节点计算距离并遍历，但不会返回它们。空过滤器接受全部 ID；结果可能少于 k 条。
  合法查询遇 k=0 或空库返回空结果。
- 极端有限输入的 FP32 累加可能溢出为正无穷，此时按 ID 排序，不承诺任意精度 L2。
- 即使 k=0，NaN/Inf、空指针和维度不匹配仍报错。
- 已捕获的分配失败转为 Error，但错误对象本身仍可能分配内存，不承诺 OOM 下绝不抛异常。

## 数据组织和复用

私有实现统一拥有连续 FP32 数据、外部 ID 和 ID→槽位映射。BruteForce 扫描与图的构建、搜索共用官方
ComputeL2SqrImpl 内核及运行时距离分派。x86_64 GNU/Clang 构建通过小型运行时分派器按 CPU/OS 支持
选择 AVX512、AVX2、SSE4.1 或 Generic；维度小于 16 时保留 Generic 路径，其他支持的
工具链使用 Generic。删除参考 Full BruteForce 的末尾填洞原则，但保留容量供复用，
不承诺 RSS 立即下降。不引入 Full Factory、InnerIndexInterface、属性及多向量依赖。

BruteForce 仍是默认后端。`BuildGraph` 构建独立的私有图后端，成功后才发布；
图阶段保留 CRUD 和过滤搜索。BruteForce 使用 v1 快照，FP32 图使用 v2，显式选择的
FP16 图使用 v3。公开 Lite 索引尚未集成 SQ8 或 mmap。

## 架构设计与兼容边界

独立 Lite 的边界刻意保持精简：

```text
应用程序
    |
    v
vsag::lite::Index             公开所有权和错误契约
    |
    v
detail::Backend               私有 CRUD/Search/存储抽象
    |-------------------|
    v                   v
BruteForceBackend       GraphBackend
FP32 行优先             FP32 或 FP16 行优先 + 邻接行
    |                   |
    +---------+---------+
              v
    运行时 L2 分派            按环境使用 Generic/SSE/AVX2/AVX512
              |
              v
    版本化持有式快照           v1 flat / v2 FP32 图 / v3 FP16 图
```

`lite/CMakeLists.txt` 只把上述源码闭包构建为共享和静态 `vsag-lite` 库。
Lite 复用仓库的 `Error`/`tl::expected` 契约和官方 L2 实现 traits，但不链接
Full Factory、HGraph、线程池、分配器编排、日志运行时、IO 框架、属性系统、
稀疏或多向量索引。安装面保持为 `vsag/lite/index.h` 和仅头文件的错误契约依赖。

索引选择围绕小数据部署目标：

- BruteForce 是默认项，因为它结果精确、无需构建阶段，并且对小集合保持最少状态。
- 单层图必须显式构建，因为它以构建时间、快照字节和近似召回为代价换取更快查询。
  `BuildGraph` 先完整创建替代后端，只有成功后才发布。
- FP16 是显式图存储选项。它减少向量和快照字节，但调用输入和结果仍为 FP32；
  它可能改变距离与召回，因此图默认仍为 FP32。

BruteForce 用一个行优先 `std::vector<float>` 保存向量，另用并行 ID vector 和
`unordered_map` 保存外部 ID 到槽位映射。Graph 为每个槽位增加一个有界出邻接 vector；
FP16 将向量载荷换成行优先 binary16。容量按几何方式增长。Remove 把最后一条记录
物理搬入空洞并更新槽位引用，同时保留已分配容量供复用，不承诺 RSS 立即下降。
图裁剪和旧快照可能含非对称邻接，因此删除会扫描已存边，不能假设每条入边都出现在
被删节点的出邻接行中。

建图期间 flat 源后端和替代后端会暂时同时存在，所以峰值内存可能高于任一稳态。
每次 Search 自己持有 visited 位图和候选队列。FP16 解码 scratch 在实现内部由调用方
持有，并在序列化迭代间复用；不存在共享可变解码缓冲、后台压缩或内部同步。

兼容目标是行为和迁移成本，不是 ABI 等价：

| 关注点 | Lite 行为 | 与 Full 的差异 |
| --- | --- | --- |
| 错误 | 适用接口返回 `tl::expected<..., vsag::Error>` | 复用契约但不依赖 Full 运行时 |
| ID 和向量 | int64 外部 ID、固定维度、调用输入 FP32 | 仅单向量稠密 L2 平方距离 |
| CRUD | Add、Update、物理 Remove、Search | 无删除标记、属性和多向量操作 |
| 过滤 | 按外部 ID 回调 | 独立的最小回调类型 |
| 结果 | 持有式 `std::vector<Neighbor>` | 不是 Full `Dataset` 结果 ABI |
| 持久化 | 可定位 C++ 流和 Lite v1/v2/v3 | 不兼容 Full 序列化 |
| 并发 | 调用方必须串行化 | 无内部线程池或并发调用保证 |

因此，即使 CRUD 命名和错误处理较熟悉，迁移代码仍需调整构造、结果处理和持久化；
本项目不宣称可直接替换 Full API/ABI。

## 快照 v1

格式与 Full 独立，数值字段均为小端：

| 偏移 | 字段 |
| --- | --- |
| 0 | 8 字节 VSAGLT01 |
| 8 | uint64 版本 = 1 |
| 16 | uint64 维度 |
| 24 | uint64 记录数 |
| 32 | uint64 载荷长度 = 记录数 * (8 + 4 * 维度) |
| 40 | uint64 表示 = 1（IEEE754 FP32、L2 平方距离） |
| 48 | 记录数个 int64 ID，按二进制补码位模式编码 |
| 48 + 8 * 记录数 | 连续行优先 FP32 向量 |

Load 要求可定位流，读取当前位置到结尾；拒绝截断、尾随字节、错误尺寸/版本、重复 ID 和
非有限值。实现将连续 ID、向量以及每一行图邻接表批量读入最终持有内存；v3 binary16
向量也直接读入最终 FP16 容器，通过指数位检查有限值，不产生 FP32 中间副本。非小端主机在原位
转换载荷数值。成功后返回新索引，不修改已有索引；该路径仍为持有内存加载，不是 mmap 或
零拷贝。没有校验和，不能检测所有仍合法的位损坏。
Save 从当前输出位置写入；flush/close、文件权限、原子替换及崩溃持久性由调用者负责，
失败可能留下部分输出。现有 ErrorType 无写错误码，因此写失败暂用 READ_ERROR。

## 验证和限制

`[lite]` 覆盖 CRUD、独立参考搜索与损坏快照；显式选择的 `[lite-scale]` 运行十万条128维
合成数据流程，仅为正确性冒烟测试，不代表真实检索 Benchmark 或性能收益。
独立的 [实验 PR #2926](https://github.com/antgroup/vsag/pull/2926) 在同一机器上，
以 10k/100k × 128 的相同 FP32 精确 BruteForce 负载，交替运行 Full/Lite 各 7 次。
下表来自提交 `d729426` 的生成数据对照：动态库在 `strip` 后测量；RSS 是
`getrusage(RUSAGE_SELF)` 报告的进程生命周期峰值在 7 次中的中位数。

| 指标 | Full | Lite |
| --- | ---: | ---: |
| strip 后动态库（字节） | 40,463,344 | 39,488 |
| 10k 进程峰值 RSS（KiB） | 164,416 | 12,112 |
| 100k 进程峰值 RSS（KiB） | 214,200 | 72,912 |
| 100k Search P50（微秒） | 2,401 | 4,392 |

RSS 包含 benchmark 的数据缓冲、CRUD、查询和临时分配，不是纯索引对象内存。
在 100k 下，Lite 的 Search、Save 和同进程 warm Load 慢于 Full。完整配置、
原始结果流程和限制见 #2926；严格冷加载与图索引对照仍待完成。

后续在同一台 AMD EPYC 服务器上，将标量 Lite 与运行时 SIMD 分派版本交替运行 7 次。
抽查的 640 条 Top-10 结果中，ID 和顺序全部一致；最大距离绝对误差为 `9.54e-6`。

| 数据集 | 标量 Search P50（微秒） | SIMD Search P50（微秒） | 标量峰值 RSS（KiB） | SIMD 峰值 RSS（KiB） |
| --- | ---: | ---: | ---: | ---: |
| 10k × 128 | 436.0 | 100.1 | 12,108 | 12,120 |
| 100k × 128 | 4,344.0 | 3,331.8 | 72,908 | 72,924 |

strip 后 Lite 动态库由 39,488 增至 47,704 字节。这些数据只是该机器、该负载下的
中位数，不代表普遍性能结论。

Debug 构建可加 `-DENABLE_COVERAGE=ON`，运行测试后用 gcov 收集源码覆盖率；
只报告实际结果，不代表 Full 覆盖率。安装后的外部示例验证不依赖 libvsag.so。
当前不承诺 Full API/ABI 替换、语言绑定、稀疏向量、WARP、SQ8、mmap 或并发调用。

## FP16 图存储

调用 `BuildGraph(VectorStorage::FP16, max_degree, ef_search)` 可让图向量使用 IEEE binary16 存储，同时保持现有 FP32 输入与查询接口。原有 `BuildGraph(max_degree, ef_search)` 仍默认使用 FP32；`ActiveVectorStorage()` 可查询当前表示。FP16 图使用 v3 快照，既有 v1/v2 字节与加载行为不变。v3 采用可移植的小端 binary16，加载不依赖保存机器的指令集。加载器将 v3 向量批量读入最终 FP16 存储，通过 binary16 指数位检查有限值，并在需要时原地转换字节序。建图或更新时会拒绝超出有限 FP16 范围的值。


## 每次查询的图搜索预算

`SearchWithOptions(query, dim, k, SearchOptions{ef_search}, filter)` 为当前图查询
单独设置预算。0 沿用配置默认值；正值最多限制到当前记录数量，必要时提高到请求结果数量。
BruteForce 忽略该选项，仍执行精确搜索。可选过滤器与 Search 使用相同的外部 ID 语义。
既有 Search 调用，包括空过滤器，保持兼容。

```cpp
vsag::lite::SearchOptions options{512};
auto result = index->SearchWithOptions(query, dim, 10, options);
```

覆盖值不会改变 BuildGraph/Add/Update/Remove 的预算，也不会改变 Save 输出；
快照仍保存配置默认值。消费者示例以预算8建图，再以预算64查询。
调用仍须由外部串行化，本功能不增加并发调用保证。更大的查询预算会增加工作量，
可能改善召回，但不保证达到某个质量目标。

## 可选8bit RaBitQ候选后端

独立构建时设置 `-DENABLE_RABITQ_LITE_BACKEND=ON`，即可调用
`BuildGraph(VectorStorage::RABITQ8, degree, ef_search)`。开关默认关闭；关闭时
选择该存储返回 `UNSUPPORTED_INDEX_OPERATION` 并保留原平面索引。候选要求非空训练集，
维度不超过1,048,576，活动记录不超过1,000,000，degree为2–64、ef不小于degree。
当前训练种子固定为47。

输入仍为有限FP32 L2向量；模型包含中心、四轮FHT/Kac符号掩码及3+5位平面编码。
新增与更新使用固定模型，不重新训练。检索使用3bit过滤估计遍历，用8bit估计排序；
返回距离是量化估计，不是原向量的精确平方L2。过滤回调接收外部ID，true表示保留，
拒绝节点仍可用于路由；单次预算不改变持久化默认值。即使过滤全部允许，过滤和未过滤
两条近似路径也不承诺始终返回相同行。

保存直接写独立小端 `VSAGLQ01` v1模型/编码/ID/图状态，不经近似重建再编码；加载后
保持查询结果且可继续增删改，包括删除所有记录后再新增。只有开启此功能的构建可加载。
原 `VSAGLT01` v1/v2/v3继续按旧路径处理，实验 `VSLRBQ01` 不能冒充此候选格式。
输入流需可定位，较大分配前按剩余字节校验声明大小；没有校验和、原子文件替换或并发支持。
模型或变换内部算术溢出的极端有限输入会失败，不替换原索引。

重复新增、缺失ID更新/删除在事务复制之前检查。更新先编码并比较两组位平面和全部六项元数据；
完整编码相同则保持拓扑与快照字节，不能仅根据编码位相同判断。编码本身仍有分配和内部算术成本。真实改值时，已准备的归一化查询及完整编码移动进事务，
后续维护不再对该输入重复归一化或编码。

**默认公开CRUD均采用局部回滚事务**。Remove备份被删/末尾槽位编码与ID、保留被删映射节点并记录受影响行；
失败时在原容量内恢复长度和行，重新插入原映射节点且不超过旧负载，不需要回滚分配。
公开Add先准备编码/邻居并预留几何增长的容器空间，再记录受影响行；
失败时撤销新ID、截回追加长度并恢复旧行。容量/桶增长可能保留，逻辑内容不变；正常扩容仍可能搬移已有数据。
公开Update改为记录旧编码与首次写入前的受影响邻接行，
失败时无分配地恢复编码并交换旧行；Update不改变ID、模型及容器尺寸。仍需扫描有向邻接寻找入边，
高入度情况下可能记录较多行；实验反向邻接模式的Add/Update/Remove仍回退全复制。不能据此称全部CRUD恒定时间或没有临时内存成本。Remove的bool无法区分缺失ID与分配失败，
两者均保留状态；构造错误本身仍可能分配内存。构建阶段保留临时FP32数据与FP32图，
构建峰值不能当量化稳态内存。三个数据分布的质量、性能及长期CRUD验收仍待执行，
不得直接套用旧独立探针成绩。可构建 `lite_rabitq_example` 查看公共接口生命周期示例。

正式接口10k SIFT/GIST/Cohere先导结果见 `lite/benchmark/results/integrated-storage-pilot-20261009`。测得快照更小，但查询和整进程峰值内存劣于FP32；历史查询及短维护不能替代100k、独立加载RSS和长期真实改值验收。数据和版本边界以该实验报告为准。

新进程10k独立加载结果见 `lite/benchmark/results/integrated-load-final-20261009`：无原始矩阵驻留时，RaBitQ常驻RSS比FP32低约41–66%；加载/首次查询代价仍存在。RAM暖缓存不能称冷启动，也不替代100k或长期验收。报告分别记录当前进程VmHWM及可能受启动器影响的getrusage峰值。

100k集成先导显示RaBitQ加载RSS更低，但固定degree16/ef128质量不足。另一个历史查询网格中，RaBitQ在SIFT ef512、GIST/Cohere ef2048达到本轮研究门槛，查询仍慢于浮点。见 `lite/benchmark/results/integrated-100k-pilot-20261009` 及 `integrated-100k-budget-grid-20261009`；这不是盲测、精确等召回性能或新默认值，长期真实改值及Full对照仍待完成。


[RaBitQ 架构、实际快照布局与剩余验收](lite_rabitq_design.md)。

## 修改失败回滚与 RaBitQ 距离计算

已发布的 FP32、FP16 图使用局部修改日志。Add、Update 分配失败时保持逻辑记录和
邻接关系不变；Remove 修复图时若分配失败，返回 false 并保留原有内容。预留容量可能
增加。日志只备份受影响的邻接行和被覆盖的记录，不复制全图。BuildGraph 的私有构建
失败时直接丢弃替代图，完整构建成功后才启用日志并发布。

RaBitQ 每个查询或维护查询只计算一次向量求和。5-bit 补充码直接复用 Full 的共享标量、SIMD 解码及尾部处理内核，按 CPU 能力分派 AVX2/AVX512，并保留标量回退。SIMD 使用原生融合乘加和向量归约，距离末位可能与旧标量逐维累加不同。测试覆盖独立标量估计、非对齐输入、位平面尾部、查询质量和修改回滚。编码、持久化格式和搜索预算保持一致。相对旧 Lite 实现的加速不能证明已与 Full VSAG 或 FP32/FP16 性能对齐。

## 扩大查询样本的原生对照验收与 ELF 范围

每数据集600查询的当前Lite/原生HGraph对照见 `lite/benchmark/results/acceptance-full-expanded-r1-20261009`。Full为固定已安装原生版本，不是旧Lite，源/头文件/二进制身份均记录；查询预算Lite512/Full128不同，不是精确等召回或等预算。结果显示暖加载及整进程常驻RSS较小，但查询更慢、快照更大、构建峰值更高。已观察查询只能作为回归数据，不是盲测，也不替代长期真实改值CRUD验收。

`python3 lite/benchmark/measure_elf_closure.py --lite LITE_SO --full FULL_SO --output NEW_OUTPUT --scratch /dev/shm` 用于可信共享库的 ELF DT_NEEDED 依赖闭包。递归解析依赖，按真实路径去重（包含共同系统运行库），只对scratch副本使用相同 `strip --strip-unneeded`，不改变原文件。范围不含可执行文件、头文件、Python绑定、容器及动态加载插件，不能称完整wheel/容器/SDK包体。运行 `python3 lite/benchmark/test_measure_elf_closure.py` 可检查依赖解析、去重及仅修改副本的夹具。

查询局部堆保持既有候选准入和图遍历顺序。无过滤 FP32/FP16 查询在遍历结束后接管堆拥有的候选缓冲区，选择前 k 项，仅对这些结果按距离及外部 ID 排序。RaBitQ 直接对同一组保留候选评分，避免逐项弹出粗排堆；评分公式和最终 ID 排序不变，过滤路径仍保持原实现。这借鉴原生 DistanceHeap 直接访问保留缓冲区的方式，不改变 ef、拓扑、公共 API 或持久化格式。配对证据见 `lite/benchmark/results/query-selection-expanded-20261009`；对照是旧 Lite，不是 Full VSAG。

长期真实改值基准可设置 `VSAG_GRAPH_REPLACEMENTS` 为按操作顺序排列的 fvecs，并传入非空 QUERY_RESULTS；替换行必须有限且相对当前值确实改变，行数必须等于 ROUNDS×CRUD_OPS。使用 `changed-groundtruth.ivecs` 验证最终状态，初始真值仍为 `groundtruth.ivecs`；Update 后 Remove/Add 使用同一新行，不恢复旧值。工具增加 `.initial` 查询证据和 `.operations.csv`（operation 0/1/2 为 Update/Remove/Add）。CLI 及未启用时的 CSV 列不变。准备脚本 `lite/benchmark/prepare_persistent_crud.py` 固定生成三遍全ID改值并以 FP64 直接差分重算真值。结果 `persistent-whole-crud-20261009` 表明维护后质量明显低于同最终数据新建图，尤其 RaBitQ；不能以更短查询时间宣称收益，也不能把新建图重训模型与图维护效应混为一谈。该研究为串行初始/最终查询、暖缓存、每组单次，不是逐操作交错查询、100k或完整终验。

RaBitQ 无反向边的删除仍扫描全部邻接行。现在一次检查被删槽位或搬迁末槽位，未受影响的行跳过后续 erase/remap；受影响行的顺序、事务备份、修复候选和持久化结果不变。这不是渐进复杂度改善，也不修维护质量。固定重放证据及旧扫描黄金路径见 `lite/benchmark/results/rabitq-remove-skip-20261009`；对照为旧 Lite，非原生 Full，长期 CRUD 后召回失败仍明确保留。

仅在真实改值重放中，VSAG_GRAPH_CRUD_MODE=update 只调用 Update，replace 只调用 Remove/Add，all 保持组合协议。显式设置时在可选 CSV 末尾追加 mutation_mode，默认列不变。同最终数据/真值的隔离结果见 lite/benchmark/results/persistent-crud-isolation-20261010；两条 RaBitQ 路径仍低于研究召回门槛。这分离了操作协议，尚未确定或修复唯一根因。

VSAG_GRAPH_DIAGNOSTIC_EF 可在持续改值基准中设置为全节点数，输出独立初始/最终诊断证据，不改变正式 ef。lite/benchmark/results/rabitq-model-ceiling-20261010 的固定模型全候选改值后召回约99.55–99.65%，正式维护质量仍较低。隔离 Update 入边保护候选提高召回，但修改成本增加约33%且GIST仍低研究门槛，未进入生产默认实现。参见相邻候选结果；全预算诊断召回不等于正式验收成绩。

## 实验性运行时入度计数维护

RaBitQ 内部新增 NONE/RECOUNT/CACHED 入边保护策略，公共适配器仍为 NONE。CACHED 在首次需要时为每槽位建立 uint64_t 入度，并随邻接变化增量维护，避免每次 Update 全图重算。成功 Add 追加零计数并应用换边差分，Remove 扣除被删出边并随末槽搬迁计数；失败的日志事务令缓存失效，下次 Update 重建；反向邻接的副本事务失败保留原缓存。计数占 8N 逻辑字节、计入内部内存账目，不改变快照/API/模型/预算；关闭后保留容量。10k 连续 Update 单配对实验中，相对受保护但逐次重算的原型，GIST/Cohere 修改块耗时降低15.79%/18.50%，快照及有序结果逐字节相同。GIST Recall .897667 仍低研究目标，因此不默认启用；混合重放的维护质量仍未达标，100k 及原生 Full 收益尚未证实。原始数据、黄金测试、范围明确的覆盖率及限制见 lite/benchmark/results/rabitq-incoming-count-cache-20261010。

结构混合重放见 lite/benchmark/results/rabitq-mixed-count-cache-20261010：三遍真实改值 Update/Remove/Add 后，相同保护策略的快照及有序结果不变。相对每次成功增删都会失效的旧缓存，GIST/Cohere10k 修改块耗时降低10.05%/11.61%，但最终召回 .811667/.908000 仍低研究目标。这只是内部维护账目降成本，不是图维护质量修复、公共默认提速或原生 Full 对照；各操作并非统一变快，证据为单配对串行重放。

内部 ConfigureIncomingProtection 增加可选 Add/Remove 阶段开关，默认均false：Add 在 link 后保护新槽位，Remove 在搬槽及修复后保护排序后的受影响孤点；继续复用不增加度数的局部保护及事务日志，公共adapter仍NONE。固定因子实验 lite/benchmark/results/rabitq-structural-guard-20261010 中，同时启用将混合GIST/Cohere召回 .811667/.908000 提高到 .847500/.922000，修改块代价 +6.39%/+3.30%，查询延迟也增加，Add阶段贡献更大。仍未达研究目标；这是内部实验开关，不是公共配置/默认验收，借鉴现有Lite浮点GraphBackend而非原生Full HGraph。

进一步的内部 Add 被淘汰目标保护会记录反向连边裁剪丢弃的旧目标，并在现有事务中修复局部入边。固定预算10k真实改值混合CRUD中，GIST召回 .847500→.855167、Cohere .922000→.924500，修改块成本+2.86%/+1.52%；查询单pair点估计改善，但不能据此宣称稳定延迟收益。仍未达研究目标，public策略保持NONE；回归证明旧目标遗漏，不保证全局连通性。证据：`lite/benchmark/results/rabitq-add-displaced-20261010`。

RaBitQ反向连边排名的栈缓冲候选已撤回：每数据集三组固定预算真实改值CRUD配对未证明稳定收益（修改块中位耗时GIST+1.77%、Cohere+1.07%），完整快照和有序结果全等。生产源码恢复，仅保留degree64饱和/并列编码回归和负面证据。见`lite/benchmark/results/rabitq-link-scratch-20261010`；恢复后Lite范围覆盖率95.16%，不是新增性能收益。

## RaBitQ 加载路径 ID 校验优化（2026-10-11）

[同快照加载对照](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-load-id-map-20261011/README.md)移除临时重复 ID 集合，构造函数必需的槽位表仍拒绝重复 ID。默认 reverse=false/NONE、快照字节不变。每数据集15组交替10k对照，GIST/Cohere 暖加载中位数6.483486→6.243492ms（-3.70%）、5.721933→5.503448ms（-3.82%），扣除进程基线的加载后 RSS 少472/416KiB，60次原生加载再保存哈希完全一致。这是分配器页驻留与暖加载证据，不是索引容量、查询、CRUD、冷 I/O、Full 或100k优势。重复 ID 坏输入会在图解析后更晚拒绝，仍为 INVALID_BINARY，但校验顺序和错误文本不保证保持。质量及终验仍待完成。

## RaBitQ 入边安全连接裁剪实验（2026-10-11）

[固定预算配对报告](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-safe-link-20261011/README.md)将现有浮点 Lite 的安全删边规则适配到 RaBitQ 缓存入度：若存在其他可安全删除的排序边，优先保留旧目标的唯一入边；没有安全候选时仍回退到原最远边。这不是完整多样性裁剪或连通性保证。内部新增末尾 `protect_link` 参数要求 CACHED，默认 false；公开适配器仍为 NONE/reverse=false，API、模型、degree/ef、快照格式不变。

对照两边都采用 CACHED，仅候选开启新规则。每数据集3组交替10k对照，GIST/Cohere各600条已观察查询，3轮全 ID 真实整向量 Update/Remove/Add，共108万次调用。最终召回率 .811667→.899167 / .908→.947，提高8.75/3.90个百分点，CRUD中位耗时增加1.20/2.29%；维护后查询P50也从407.651→439.810 /403.991→428.569us上升。这是质量与成本权衡，不是等召回提速；既有 .90/.95 实验门槛仍未通过，同一确定性观察查询集不构成独立或盲质量复现。

策略仅保留为内部可选实验，等待进一步维护质量优化及100k/交错验收。Release6、ASan/UBSan5、新鲜覆盖率5、RaBitQ默认关闭4、format/tidy15及3万个故障注入位置通过，Lite范围行覆盖率95.44%，非Full全库覆盖率。保留源码/依赖/适配器及逐查询、逐操作原始证据，不声称公开默认、Full、冷I/O或完整包收益。

## RaBitQ 构建邻接表所有权交接（2026-10-11）

[成对构建诊断](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-owned-topology-20261011/README.md)移除逐行邻接表转 CSR 再展开的拷贝。构建逐行收缩备用容量后直接移动邻接表；加载保留 CSR 构造路径。API、图边、量化、查询参数、入边默认策略和格式不变。反向入边构建在访问计数数组前校验目标范围。

17维合成数据，10k/100k各三对交替测量：100k构建峰值中位数87628→65460KiB（-25.30%），10k均18856KiB；耗时中位数分别增加0.47%/1.60%，不是稳定提速。累计普通new请求字节分别减少4760184/38653816，不代表常驻内存。六对快照及八条查询的ID/距离顺序一致。不代表真实高维数据、长期CRUD、冷加载、召回验收或Full对照；10k峰值可能被启动器高水位覆盖，真实数据及最新Full验证仍待补齐。

## 真实100k RaBitQ 所有权验收片段（2026-10-11）

[真实数据实验](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-owned-real100k-20261011/README.md)比较生产4f73870与0444bf9，两者都是Lite公开默认NONE/reverse=false，不是原生Full。GIST960和归一化Cohere768各100k/100条已观察查询/k10，degree16/构建128/查询512，三对交替，共12builders/36独立loader。

整进程峰值中位数928420→903892KiB（-2.64%）、759556→734956（-3.24%），包括输入、建图、查询及原生回读，不能当作独立建图阶段峰值。构建耗时基本不变。六对原生快照与有序查询结果一致；召回.859/.937不变且仍不足，快照112804416/93603552字节不变。查询P50中位数1088.753→1115.092us（+2.42%）、1068.973→1134.902（+6.17%）；保存47.454→65.928ms（+38.93%）、38.101→60.065（+57.65%）退化。独立暖加载62.696→60.714/53.161→53.228ms，不是冷加载或普遍提速。

这是小幅真实内存权衡证据，不是全面速度验收、Full优势或长期真实改值CRUD通过；100条已观察查询仅提供探索性尾延迟。邻接行分配/缓存局部性、Save的GetGraph整图CSR物化是后续假设，尚非实测根因。Release6及首轮驱动fixture通过；4f73870消毒器/OFF/coverage仅继承，未重跑。末次收集时RAM对照库已不存在，重复fixture提前失败，保留日志并从已校验归档恢复首轮成功日志，不伪称末次重跑成功。最新Full配置/SIMD/质量对齐、真实长期CRUD及更大查询集仍待补齐。

## RaBitQ 有界分块保存（2026-10-11）

[流式保存对照](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-stream-save-20261011/README.md)将GetGraph整图CSR物化改为只读邻接行按原offsets/edges顺序分块写入，暂存最多64KiB。保留Validate、小端回退、短写错误及格式；API、图边、量化、参数及默认策略不变。64KiB仅指邻接暂存，不是所有Save工作内存。

GIST960/Cohere768各100k，各7对独立进程交替保存，共28次，先Load再计时open/Save/close；CPU0/thread1/RAM暖缓存，非coldIO/fdatasync。Save中位52.819至45.774ms(-13.34%)、44.151至37.441(-15.20%)。保存后VmHWM151216至134964/132460至116216KiB，少16252/16244KiB，加载高水位可能遮蔽Save专项峰值。28次SHA与输入一致。

Release6/SAN5/freshCov5/OFF4/format-tidy15通过，Lite2983/3125=95.456%非wholeFull。跨块、bulk/scalar、短写通过；首次命名lint失败计时前修复并保留日志。这是已加载状态Save优化，不是上轮建图后Save同协议复测；查询退化、长期CRUD质量和Full验收仍待完成。

## 构建邻接行容量候选未采用（2026-10-11）

[隔离实验](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results/rabitq-row-capacity-20261011/README.md)仅在候选适配器取消构建邻接行shrink_to_fit；两边共享bf50173生产头文件、对象和流式Save，未改业务源码、公开选项、图边或参数。GIST960/Cohere768各100k/100已观察query，degree16/build128/query512/CPU0/thread1，三对交替，共12builders/36独立warmload。六对快照SHA和有序结果一致，Recall .859/.937不变。

查询P50中位数GIST1081.373至1084.913us（+0.33%）、Cohere1100.393至1146.963（+4.23%），P99分别+3.46%/+0.94%；构建+0.60%/+0.02%，整进程峰值+0.005%/0%，没有明确收益。Save/load小幅变化不视作稳定算法收益；保留构建容量不能据此解释为查询更快，也未证明此前退化根因。决定不接入生产，继续保留shrinking默认与流式Save。原始参数/源码/输入/延迟/GT/命令归档并独立复核。Fresh Release6通过；SAN/OFF/coverage只继承bf50173，未做候选消毒器或故障验收。不是Full、长期CRUD或盲P99验收，下一步优先路由工作区与堆分配诊断。

## Promoted implementation and historical evidence

The implementation promoted on 2026-10-12 is derived from personal source `a62e62e08db3f0d7da2e4439785feed6fcf90257`. Experimental diversity switches and diagnostic benchmark targets are excluded. Benchmark commands and `lite/benchmark/` paths below refer to that pinned experimental repository, not targets shipped in this standalone PR. The reports retain each measured source version: numbers from different stages are not a single latest-version acceptance matrix. See [the pinned reproducibility sources](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark) and [promotion evidence](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results). Current CPU-time/utilization, quality-matched native Full comparisons, cold I/O and hours-long interleaved changed-vector CRUD remain incomplete. Externally concurrent calls require caller serialization.

本轮移植的是已验证生产实现，不是整条实验分支。历史报告必须按各自源码版本读取；最新版本 CPU 时间、同质量 Full 对照、冷加载和小时级真实改值交错 CRUD 尚未完成，不宣称项目全部验收通过。
