# 独立 VSAG Lite

本文是 PR #2904 的当前受支持契约，更新于 2026-10-12。历史实验说明不能覆盖当前行为。

## 构建、配置与安装

```bash
cmake -S lite -B build-lite -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite -j2
ctest --test-dir build-lite --output-on-failure
cmake --install build-lite --prefix "$PWD/install-lite"
```

默认后端是 FP32 BruteForce。`ENABLE_RABITQ_LITE_BACKEND` 默认 OFF，显式添加
`-DENABLE_RABITQ_LITE_BACKEND=ON` 才编译 RaBitQ8。
`ENABLE_COVERAGE=ON` 对库和测试编译单元一起插桩。构建动态和静态 `libvsag-lite`，
安装后 CMake 消费方使用 `find_package(vsag-lite CONFIG REQUIRED)` 和 `vsag::lite`。
`lite/example` 提供安装消费示例；`rabitq_example.cpp` 演示可选后端生命周期。
Lite 不链接 Full 的 `libvsag.so`。

## 功能—源码—测试对照

| 支持功能 | 公开入口及实现 | 回归证据 |
| --- | --- | --- |
| 空索引、拥有 FP32 数据及外部 int64 ID | `Index::Create`、`Add`；`index.cpp`、`bruteforce_backend.cpp` | `index_test.cpp`：校验、重复 ID、精确检索、规模测试 |
| 显式构建图 | `BuildGraph(storage, degree, ef)`；`graph_backend.cpp`、`rabitq_backend.cpp` | 浮点图/RaBitQ API 测试，失败保留旧后端 |
| 单向量新增、真实改值、物理删除 | `Add`、`Update`、`Remove` | 三种存储、搬槽引用、入边不变量、加载后继续修改 |
| L2 查询、外部 ID 过滤、每次搜索预算 | `Search`、`SearchWithOptions`、图搜索辅助实现 | 距离相同/非单调 ID、空集/过滤/大图阈值/参考实现对照 |
| 保存加载、损坏输入与流失败处理 | `index.cpp`、`rabitq_snapshot.h` | 截断、重复 ID、非法拓扑/元数据、短写、往返 |
| 分配失败回滚 | 浮点图及 RaBitQ 事务日志 | `graph_allocation_test.cpp`、`rabitq_backend_test.cpp`，恢复逻辑状态 |
| 原生 SIMD 复用与 Generic 回退 | `fp32_distance*`、`fp16_distance*`、`rabitq_filter_ip*` | ISA/标量对照、短维度和尾部 |
| 有界查询临时空间、分块快照 | `query_heap.h`、`query_visited.h`、分块/流式 I/O | 堆参考、位图边界、大小端/分块/字节一致性 |

图 degree 为 2–64，配置 ef 不小于 degree。SearchOptions 为零时沿用保存的预算；
单次预算不改变构建、CRUD 或持久化配置。图检索是近似检索，过滤可能少于 k 个结果；
被拒绝的 ID 仍可用于连接路由。RaBitQ 返回量化 L2 估计，不是精确 FP32 距离。

## 架构、保留和裁剪

公开 `vsag::lite::Index` 拥有私有后端，默认连续存储 BruteForce 做精确平方 L2 查询。
显式 BuildGraph 在发布前构建替代单层图；它不是 Full HGraph/LazyHGraph，也不按数量自动切图。
输入保持 FP32，可显式选择 FP16 图或可选 RaBitQ8。未提供 SQ8、mmap、稀疏向量、语言绑定、
完整 Full 工厂或自动后端选择。

浮点存储使用连续向量/ID 数组、外部 ID 到槽位映射、出边和入边邻接。
删除把最后一个槽搬进空洞并修复引用；容量可能保留，不承诺把全部闲置内存归还操作系统。
局部事务日志备份修改的行/记录；失败后可能保留扩大的容量，但逻辑状态保持不变。
FP16 Save 直接借用已保存的半精度行，不再解码后重新编码。

RaBitQ 保存固定训练的中心/旋转模型、3bit 过滤码、5bit 补充码、每条元数据、ID 和图，
不保留精确 FP32 原始矩阵。当前公开构建/加载启用 32 位反向邻接和 `CACHED` 入度保护，
并启用安全连接裁边；额外 Add/Remove/被挤出目标 guard 仍为 false。
CACHED 同时启用 Update 保护。若存在其他可安全删除边，就保护旧目标最后一条入边；
没有替代边时仍使用原最远边回退，不是全局连通或召回保证。
详见[RaBitQ 设计](lite_rabitq_design.md)。

## 兼容、持久化和失败边界

Lite 是独立 API，不是 Full Index 的 ABI 可替换实现。同一实例必须由调用方串行化，
不支持并发查询或读写混合并发。输入仅在调用期间借用，结果和持久状态由索引拥有。

FP32 BruteForce 使用 `VSAGLT01` v1、FP32 图 v2、FP16 图 v3；可选 RaBitQ 独立使用
`VSAGLQ01` v1，浮点旧格式不变。全部小端，不兼容 Full 快照。
RaBitQ Load 恢复模型/编码/拓扑，不重新训练，也不能还原精确输入向量。
Load 需要可定位输入流并验证长度及内容。Save 不提供校验和、fsync 或原子文件替换，
失败可能留下部分输出；需要持久原子替换时，调用方负责临时文件和替换。

Add/Update/Search/Save/Load 使用现有 `tl::expected`/Error。
Remove 为 bool，false 无法区分 ID 缺失和图分配失败。失败回滚恢复逻辑状态，不恢复分配器容量。
完整存储表示相同的 Update 不改变图；RaBitQ 编码相同不代表原始向量相同。

## 项目产出要求对照

| 要求 | 已有交付 | 剩余验收 |
| --- | --- | --- |
| 可评审独立 Lite PR、CRUD/持久化/配置/测试 | #2904、上述源码及示例、ON/OFF 回归 | 社区 CI 和维护者评审，不自动合并 |
| 架构、裁剪、布局、内存与 API 边界 | 本文及中英文 RaBitQ 设计 | 行为变化后持续同步 |
| 至少两个数据集/规模的可复现实验及原始结果 | [版本化性能证据](../../../../../lite/PERFORMANCE_EVIDENCE.md)，SIFT/GIST/Cohere 10k/100k 的固定脚本/配置/raw | 最新统一版本 CPU/峰值/CRUD/质量矩阵、冷 I/O、小时级交错真实改值稳定性 |
| 相比真实 Full 至少一项优化且效果稳定 | 带 Full 来源的历史暖加载/RSS/ELF 依赖闭包结果 | 同质量原生 Full 对照、完整部署包；GIST 维护质量仍未完整验收 |

`.github/workflows/lite.yml` 定义默认关闭、RaBitQ 开启、ASan/UBSan、覆盖率矩阵，
覆盖全部 Lite 格式、已配置编译单元/头文件静态检查、安装和消费示例。
覆盖率脚本对生产 Lite 行并集设 90% 门槛，不是整个 Full 或分支覆盖率。
远程本地通过不代表在线 CI 已通过。

历史与否定实验保留在[固定实验仓库版本](https://github.com/nw9663644-eng/vsag/tree/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark)。
不同被测版本的最好值不能拼成虚构“最新统一矩阵”。功能可用不代表质量、性能、长期稳定性全部验收通过，
项目尚未宣布完成。
