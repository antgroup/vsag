# 流式序列化方案 Proposal（Streaming Serialization · TLV）

> Author: OpenCode (Assisted)
> Status: Draft / Proposal
> Scope: 在 VSAG 中**新增**一套序列化/反序列化方法，与既有格式**并存、不互转**。

---

## 1. TL;DR

新增一套序列化格式，特性：

1. **写出完全流式**：全程只向前写，无任何 `Seek`/`seekp`。
2. **元信息在头部**（Head），而非尾部 Footer。
3. **读取完全流式**：全程只向前读，无任何 `Seek`/`Length()`（允许**前向空读/丢弃**，不算 seek）。
4. **前向兼容**：低版本代码读高版本索引，自动跳过不认识的新数据块。
5. **后向兼容**：高版本代码读低版本索引，缺失块按低版本默认，参数保持一致。
6. **能力可选加载**：写出可含多种能力（如高/低精度），加载时按参数只取子集。

核心手段：**头部 `StreamHeader`（承载全部语义元信息）+ body 采用 TLV（Tag–Length–Value）自描述分段**。`Length` 让“无 seek 前向跳过”成为可能，从而一举满足兼容性与能力选择。

---

## 2. 背景：为什么现有格式不是“流式”

现有抽象层中，**写端 `StreamWriter`（`src/storage/stream_writer.h:26`）本就无 seek**；真正的障碍是**元信息被放在产物尾部的 `Footer`**（`src/storage/serialization.h:133`），而 `Footer::Parse`（`src/storage/serialization.cpp:47`）必须先 `Seek` 到流尾才能读到它：

![现有格式：元信息在尾部，反序列化必须 Seek 到流尾才能读到 Footer](assets/streaming-serialization/01-footer-problem.svg)

此外，**IVF/DiskANN** 的 body 还用 footer 里的偏移做随机读（`PushSeek`/`seekg`）；而 **HGraph 的 body 本就是纯顺序读**，唯一非流式点也只是开头的 `Footer::Parse`。

> 结论：把**尾部 Footer 改成头部 Head**，并把 body 改成**可顺序读、可跳过**的分段，即可实现完全流式。

---

## 3. 设计目标

| # | 目标 | 落地 |
| --- | --- | --- |
| G1 | 写端无 Seek | 仅 `StreamWriter::Write`，不回写 |
| G2 | 元信息在头部 | 语义元信息全部进 Head；body 自描述 |
| G3 | 读端无 Seek | 新增禁 Seek 顺序读 reader；允许前向 read-discard |
| G4 | 前向兼容（低读高） | 每块自带 Length → 未知块前向跳过；`CRITICAL` 位决定可忽略/报错 |
| G5 | 后向兼容（高读低） | tag 驱动派发；缺块/缺字段取默认（按 `_version`） |
| G6 | 能力选择加载 | 多能力各为独立块；加载参数 → 跳过不需要的块 |
| G7 | 独立自洽 | 独立 magic `vsagstm0`；与旧格式并存、不互转 |
| G8 | 遵循仓库规范 | `include/vsag/` 公共 API、`src/` 实现、`vsag` 命名空间；优先 `uint64_t`；`.cpp`；4 空格/100 列；clang-format/tidy **15**；测试 + 覆盖率 ≥ 90% |

---

## 4. 总体方案：Head + TLV Body

整体布局如下——**所有语义元信息集中在头部 Head**，其后是一串 **TLV 记录**，最后用 `SECTION_END` 哨兵收尾（不依赖 EOF，便于嵌入更大文件）：

![新格式总体布局：Head 头部元信息 + TLV Body + SECTION_END，写读全程只向前](assets/streaming-serialization/02-overall-layout.svg)

---

## 5. 为什么用 TLV（对“跳过”的刚需）

需求 4（低读高跳未知块）与需求 6（加载只取一种精度）的本质都是“**跳过一个块**”。在“无 seek、纯前向”前提下要跳过一个块，唯一办法是**该块自带长度**，读端据此做**前向 read-discard**。这正是 TLV。

> 修订背景：最初草案曾推荐“**纯顺序、分段不带长度**”，但它无法“跳过一个不认识/不需要的块”（不知字节数就无法前向空读跳过），在引入需求 4/6 后被**否决** → 每块必须自描述长度，即 TLV。

业界同构先例可借鉴其思想：**PNG chunk**（`length+type+data+crc`，大小写标识 critical/可忽略）、**Matroska EBML**、**RIFF/WAV**、**Protobuf**（未知字段可跳过）。

由此得到**两层可扩展性**：

![两层可扩展性：分段级 TLV 增删跳过整块 + 字段级 JSON 增删标量参数](assets/streaming-serialization/03-two-layer-extensibility.svg)

- **分段级（TLV）**：增删/跳过整块（datacell、某精度、某能力）。靠 `Tag + Length + flags.CRITICAL`。
- **字段级（JSON）**：增删标量参数。靠 JSON 的“未知键忽略、缺键默认”。

---

## 6. TLV 记录与 Head 的字节布局

![TLV 记录字节布局：tag(u32) | flags(u16) | value_len(u64) | value](assets/streaming-serialization/04-tlv-record-layout.svg)

**Head**（顺序结构）：`magic[8]="vsagstm0"` → `format_version:u32` → `meta_len:u64` → `metadata(JSON)` → `crc32:u32`（CRC 覆盖 metadata 字节）。metadata 关键键：`_version`（驱动后向默认）、`_min_reader_version`（可选全局闸门）、`_empty`、`basic_info`、`_manifest`（本产物包含的 tag 清单）。

---

## 7. 写端：无 Seek 下如何拿到 Length

TLV 要求“先写 Length 再写 Value”，而无 seek 写端不能回填，故每块长度必须**先于写出而知**。三种实现（可混用），推荐 **T1**：

![T1 推荐方案：先整体测量预扫描得到每块尺寸，再一趟流式真正写出（零额外内存）](assets/streaming-serialization/05-writer-t1-prescan.svg)

> T1 的代价是 body 序列化逻辑跑约 2×（多为 memcpy，非热点），但**零额外内存**且复用既有模式（`inner_index_interface.cpp:290`）。对超大 datacell 改用 **T3 分块编码**彻底避免双趟。

三种实现对比（读端跳过语义统一：要么按 `value_len` 跳，要么按 chunk 读到 0）：

| 方案 | 机制 | 额外内存 | 额外 CPU | 适用 |
| --- | --- | --- | --- | --- |
| **T1 整体测量预扫描（推荐）** | 先用空写计数器跑一遍 body 记录每块尺寸（复用 `CalSerializeSize` 的 no-op writer），再真正写 `[tag][flags][len][value]` | 0 | 约 2×（多为 memcpy） | 尺寸可测的内存索引（HGraph/IVF/…） |
| **T2 逐段缓冲** | 每块先写入可增长内存 buffer 得长度，再写 `[tag][flags][len][bytes]` | 单块峰值 | 1× | 段不大、或不便重复执行序列化 |
| **T3 分块编码（`value_len=U64_MAX`）** | value 写成 `[chunk_len][chunk]…[chunk_len=0]`，无需预知总长 | 单 chunk | 1× | 超大段或尺寸天然不可测；跳过时读 chunk 到 0 |

---

## 8. 读端：TLV 驱动循环（无 Seek）

读端解析 Head 后进入 TLV 循环；对每条记录决定**派发**、**跳过**或**报错**。跳过用**前向 read-discard**实现（“向后空读”不算 seek）：

![反序列化读循环（全程前向）：解析 Head 后进入 TLV 循环，对每条记录派发/跳过/报错](assets/streaming-serialization/06-reader-loop.svg)

- `BoundedForwardReader`：把子反序列化器限制在本块 `value_len` 内（可用现有**单参** `Slice(len)`，`stream_reader.cpp:218`，不 seek），读完后 `SkipForward` 跳过块内未读尾部 → **段内前向兼容**（新字段追加块尾时旧读者自动跳过）。

---

## 9. 兼容性 & 能力选择（同一套“跳过”机制）

前向兼容（低读高的未知块）与能力选择（加载方主动跳过不需要的块）复用**同一套** TLV 前向跳过，差别只在 `should_skip(tag)` 的来源。下图同一产物被两类读者处理：

![同一产物被两类读者处理：低版本读者前向兼容跳过未知块，选择性读者按加载参数空读跳过不需要的块](assets/streaming-serialization/07-compat-capability.svg)

**后向兼容（高读低）**：读取是 tag 驱动的，低版本没写的块其 tag 不出现；读完后高版本对“期望但缺失”的块/字段按 Head 的 `_version` **施加低版本默认**（标量参数走 JSON 的 `Contains()` 判定，IVF 已有先例 `ivf.cpp:655`），从而参数与写出低版本时一致。

**能力选择的索引侧依赖**：被跳过能力对应的搜索路径必须可降配。VSAG 已有相关开关可复用：HGraph 的 `has_precise_reorder()`/`ignore_reorder_`、IVF 的 `use_reorder_`。映射关系：HGraph `basic_flatten_codes_`(低) / `high_precise_codes_`(高)，IVF `reorder_codes_`(高)。

---

## 10. 组件与对外 API

**新增/改造组件**

| 组件 | 文件 | 说明 |
| --- | --- | --- |
| 常量 | `include/vsag/constants.h`、`src/constants.cpp` | `SERIAL_STREAM_MAGIC="vsagstm0"`、`SERIAL_STREAM_VERSION`、`SECTION_END=0` |
| tag 注册表 | 新增 `src/storage/serialization_tags.h` | 数值 enum + 默认 `CRITICAL` + 引入版本注释，只增不复用 |
| `StreamHeader` | `src/storage/serialization.{h,cpp}` | `Write` 顺序写头；`Parse` 顺序读头（**禁** `Length/Seek`）；复用 `Metadata` + CRC |
| 禁 Seek 顺序读 reader | `src/storage/stream_reader.{h,cpp}` | `SequentialReadStreamReader`：仅前向 `Read`，`Seek/Length` 抛异常；不在构造期量长度 |
| 有界前向读 + 跳过 | 同上 | `BoundedForwardReader` + `SkipForward(reader,len)` |
| TLV helper | 新增 `src/storage/tlv_section.{h,cpp}` | 写/读 records、处理 CRITICAL、skip 集合、分块编码 |
| 编排 | `src/algorithm/inner_index_interface.{h,cpp}` | 新增 `SerializeStream/DeserializeStream`；建议**新增独立 body 纯虚接口**，与旧路径隔离，回归风险最低 |

> 注意（已核对源码）：`BufferStreamReader` 传 `max_size=UINT64_MAX` 会触发 `reader->Length()`（`stream_reader.cpp:171`），装饰流式源时**必须传显式上界**；`IOStreamReader` 构造即 seek 量长度（`:97`），不可用于不可 seek 流，故须新增顺序读 reader。

**对外 API**（不改既有签名）

```cpp
// include/vsag/index.h，经 IndexImpl SAFE_CALL 透传
virtual tl::expected<void, Error> SerializeStream(std::ostream& out) const;
virtual tl::expected<void, Error> DeserializeStream(std::istream& in,
                                                    const std::string& load_params = "");
// 回调/流式信道版（新增只前向读的 ReadFuncType）
virtual tl::expected<void, Error> SerializeStream(WriteFuncType write) const;
virtual tl::expected<void, Error> DeserializeStream(ReadFuncType read,
                                                    const std::string& load_params = "");
```

`load_params`（JSON）承载能力选择，如 `{"load":{"precision":"low"}}` 或通用 `{"skip_tags":[...]}`；缺省加载全部。

---

## 11. 各索引适配

| 索引 | 现状 | 改造 | 难度 |
| --- | --- | --- | --- |
| HGraph | body 已顺序 | 元信息移头部；body 几乎照搬（去 `Footer::Parse`）；各 datacell 分配 tag | 低 |
| IVF | 偏移随机读 | 改顺序 TLV 读（参考旧 v0.14 顺序路径 `ivf.cpp:628`） | 中 |
| DiskANN | 偏移 + seek + 按需 IO | **仅支持单流整体加载**；按需随机 IO 不在本方案范围 | 中-高 |
| BruteForce/SINDI/Pyramid/WARP/Sparse | 顺序自终止 | 接入 Head + TLV，body 逻辑几乎不变 | 低 |
| ConjugateGraph | 旧式 4KB Footer + 头尾跳转 | 作为可选 TLV 块（`CRITICAL=0`），去跳转 | 中 |

---

## 12. 测试与覆盖率（C++ ≥ 90%）

1. **禁 Seek 强约束**：`NoSeekStreamReader` 替身，任何 `Seek/Length/PushSeek` 调用即失败；包裹产物反序列化，机制层证明无 seek。
2. **头部定位**：断言 offset 0 起 8B == `vsagstm0`。
3. **不可 seek 信道**：自定义不可 `seekg` 的 `streambuf` 端到端。
4. **round-trip**：各索引 build→序列化→反序列化，结果一致（含空索引/reorder/attr filter/各 data_type/metric）。
5. **前向兼容**：注入未知 `CRITICAL=0` 块→跳过且结果正确；注入 `CRITICAL=1`→报错。
6. **后向兼容**：缺块/缺字段产物→按 `_version` 默认、参数一致。
7. **能力选择**：写高/低精度，加载只取低→搜索可用、结构更小。
8. **分块编码(T3)** 与**破坏性**（篡改 magic/版本/crc/value_len）。
9. 工具链：`make fmt`、`make lint`（clang-tidy **15**）、`make test`、`make cov`。

---

## 13. 里程碑

- **M0** 设计确认（wire format / tag 注册表 / API 命名 / DiskANN 范围 / T1·T3）。
- **M1** 基础设施：常量 + `serialization_tags.h` + `StreamHeader` + 禁 Seek reader + `BoundedForwardReader`/`SkipForward` + TLV helper（含单测）。
- **M2** 接口编排 + 公共 API（含 `load_params`）。
- **M3** HGraph 打通（端到端 + 前/后向兼容 + NoSeek）。
- **M4** 能力选择样例（高/低精度分块 + 选择加载）。
- **M5** 扩展索引（BruteForce/SINDI/Pyramid/WARP/Sparse → IVF）。
- **M6** ConjugateGraph + DiskANN（单流整体加载）。
- **M7** 文档（`docs/docs/{en,zh}/src/` 序列化章节）+ 覆盖率收尾。

---

## 14. 风险与边界

1. **DiskANN 按需随机 IO vs 无 seek 读**：天然冲突，首版仅单流整体加载。
2. **T1 双趟幂等性**：内存索引无虞；从流读取的组件需幂等，否则用 T3。
3. **CRITICAL 错标**：错标可忽略会让低版本静默忽略关键块 → 注册表 review 严格把关。
4. **块新字段必须追加块尾**：否则段内前向兼容失效。
5. **CRC 仅覆盖 Head metadata**：body 完整性未校验（与现状一致）；如需可为每块加块尾 CRC。
6. **能力选择的降配自洽**：被跳过能力的搜索路径须可降配，须补测试。
7. **字节序/对齐**：沿用既有 `WriteObj/ReadObj` 的原生字节序假设，不额外处理跨架构（与现状一致）。
8. **EOF/截断**：纯前向读对 `istream` 不足须健壮处理，抛 `READ_ERROR`；`SECTION_END` 哨兵缺失也应报错，而非依赖 EOF。

---

## 15. 提交规范提示（实现阶段）

- Conventional Commits（`feat:` 为主）。**AI 不得自加 `Signed-off-by:`、不得 `git commit -s`**；commit body 先写人类 `Signed-off-by:`，紧接（无空行）`Assisted-by: OpenCode:<model-version>`。
- PR 需 `kind/feature` + `version/*` 两标签；`kind/feature` 用 `Fixes/Closes/Resolves #N` 关联 issue。
- 提交前：`make fmt`、`make lint`、`make test`、`make cov`（≥ 90%）。

---

### 一句话总结

> 尾部 `Footer` → **头部 `StreamHeader`**；body → **TLV 自描述分段**。`Tag` 标识块、`Length` 支撑“无 seek 前向跳过”、`flags.CRITICAL` 区分可忽略/必须报错——由此同时拿下**前向兼容**、**后向兼容**与**能力选择加载**。写端用 T1 测量预扫描/T3 分块在无 seek 下取得长度，读端用有界前向读 + 前向 read-discard 全程不 seek。
