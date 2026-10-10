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
