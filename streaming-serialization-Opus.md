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

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 270" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ah-bad" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#d33"/>
    </marker>
  </defs>
  <text x="20" y="28" font-size="17" font-weight="700" fill="#222">现有格式：元信息在尾部 → 反序列化必须 Seek 到流尾</text>

  <rect x="20"  y="120" width="150" height="62" rx="6" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="95"  y="156" font-size="13" text-anchor="middle" fill="#1c3d7a">数据块 1</text>
  <rect x="170" y="120" width="150" height="62" rx="6" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="245" y="156" font-size="13" text-anchor="middle" fill="#1c3d7a">数据块 2 …</text>
  <rect x="320" y="120" width="160" height="62" rx="6" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="400" y="156" font-size="13" text-anchor="middle" fill="#1c3d7a">数据块 N</text>
  <rect x="480" y="120" width="380" height="62" rx="6" fill="#fde8e8" stroke="#d33"/>
  <text x="670" y="146" font-size="12" text-anchor="middle" fill="#a11">Footer（尾部元信息）</text>
  <text x="670" y="167" font-size="11" text-anchor="middle" fill="#a11" font-family="monospace">magic | metadata | crc | len | cigam</text>

  <path d="M 770 118 C 770 66, 520 66, 500 116" fill="none" stroke="#d33" stroke-width="2" marker-end="url(#ah-bad)"/>
  <text x="610" y="58" font-size="12" fill="#d33" text-anchor="middle">① Seek(Length−16) 先读尾部拿到 footer 长度</text>

  <path d="M 700 184 C 700 236, 360 236, 340 184" fill="none" stroke="#d33" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#ah-bad)"/>
  <text x="500" y="256" font-size="12" fill="#d33" text-anchor="middle">② Seek(Length−len) 再读整个 Footer　✗ 纯前向流（管道/分块下载）做不到</text>
</svg>

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

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 360" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ah-fwd" markerWidth="12" markerHeight="12" refX="9" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 Z" fill="#34a853"/>
    </marker>
  </defs>
  <text x="20" y="28" font-size="17" font-weight="700" fill="#222">新格式总体布局：Head（头部元信息）+ TLV Body + END</text>

  <!-- Head -->
  <rect x="20" y="60" width="250" height="210" rx="8" fill="#e8f0fe" stroke="#4a78d6" stroke-width="1.5"/>
  <text x="145" y="84" font-size="14" font-weight="700" text-anchor="middle" fill="#1c3d7a">Head（固定结构，顺序解析）</text>
  <rect x="36" y="98" width="218" height="24" rx="4" fill="#fff" stroke="#9bb6e6"/>
  <text x="46" y="115" font-size="12" fill="#1c3d7a" font-family="monospace">magic[8] = "vsagstm0"</text>
  <rect x="36" y="126" width="218" height="24" rx="4" fill="#fff" stroke="#9bb6e6"/>
  <text x="46" y="143" font-size="12" fill="#1c3d7a" font-family="monospace">format_version : u32</text>
  <rect x="36" y="154" width="218" height="24" rx="4" fill="#fff" stroke="#9bb6e6"/>
  <text x="46" y="171" font-size="12" fill="#1c3d7a" font-family="monospace">meta_len : u64</text>
  <rect x="36" y="182" width="218" height="52" rx="4" fill="#fff" stroke="#9bb6e6"/>
  <text x="46" y="200" font-size="12" fill="#1c3d7a" font-family="monospace">metadata (JSON):</text>
  <text x="52" y="216" font-size="11" fill="#39527f" font-family="monospace">_version / _empty /</text>
  <text x="52" y="229" font-size="11" fill="#39527f" font-family="monospace">basic_info / _manifest</text>
  <rect x="36" y="238" width="218" height="24" rx="4" fill="#fff" stroke="#9bb6e6"/>
  <text x="46" y="255" font-size="12" fill="#1c3d7a" font-family="monospace">crc32 : u32</text>

  <!-- TLV records -->
  <rect x="290" y="90" width="150" height="150" rx="8" fill="#e6f4ea" stroke="#34a853" stroke-width="1.5"/>
  <text x="365" y="112" font-size="13" font-weight="700" text-anchor="middle" fill="#1e6b34">TLV 记录 1</text>
  <text x="365" y="135" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">tag|flags|len</text>
  <text x="365" y="152" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">+ value</text>
  <text x="365" y="186" font-size="11" text-anchor="middle" fill="#3a7a52">(如 base_codes)</text>

  <rect x="450" y="90" width="150" height="150" rx="8" fill="#e6f4ea" stroke="#34a853" stroke-width="1.5"/>
  <text x="525" y="112" font-size="13" font-weight="700" text-anchor="middle" fill="#1e6b34">TLV 记录 2</text>
  <text x="525" y="135" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">tag|flags|len</text>
  <text x="525" y="152" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">+ value</text>
  <text x="525" y="186" font-size="11" text-anchor="middle" fill="#3a7a52">(如 graph)</text>

  <text x="630" y="170" font-size="22" fill="#34a853">…</text>

  <rect x="660" y="90" width="150" height="150" rx="8" fill="#e6f4ea" stroke="#34a853" stroke-width="1.5"/>
  <text x="735" y="112" font-size="13" font-weight="700" text-anchor="middle" fill="#1e6b34">TLV 记录 N</text>
  <text x="735" y="135" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">tag|flags|len</text>
  <text x="735" y="152" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">+ value</text>

  <!-- END -->
  <rect x="822" y="90" width="58" height="150" rx="8" fill="#ede7f6" stroke="#7e57c2" stroke-width="1.5"/>
  <text x="851" y="160" font-size="12" font-weight="700" text-anchor="middle" fill="#4a328a" transform="rotate(90 851 160)">SECTION_END</text>

  <!-- forward arrow -->
  <line x1="20" y1="300" x2="858" y2="300" stroke="#34a853" stroke-width="2.5" marker-end="url(#ah-fwd)"/>
  <text x="440" y="290" font-size="13" font-weight="700" text-anchor="middle" fill="#1e6b34">写 / 读：只向前，绝不回看（no seek）</text>
  <text x="440" y="330" font-size="12" text-anchor="middle" fill="#555">空索引：metadata._empty = true，读完 Head 即返回，不读 body</text>
</svg>

---

## 5. 为什么用 TLV（对“跳过”的刚需）

需求 4（低读高跳未知块）与需求 6（加载只取一种精度）的本质都是“**跳过一个块**”。在“无 seek、纯前向”前提下要跳过一个块，唯一办法是**该块自带长度**，读端据此做**前向 read-discard**。这正是 TLV。

> 修订背景：最初草案曾推荐“**纯顺序、分段不带长度**”，但它无法“跳过一个不认识/不需要的块”（不知字节数就无法前向空读跳过），在引入需求 4/6 后被**否决** → 每块必须自描述长度，即 TLV。

业界同构先例可借鉴其思想：**PNG chunk**（`length+type+data+crc`，大小写标识 critical/可忽略）、**Matroska EBML**、**RIFF/WAV**、**Protobuf**（未知字段可跳过）。

由此得到**两层可扩展性**：

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 230" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <text x="20" y="26" font-size="16" font-weight="700" fill="#222">两层可扩展性：分段级（TLV）+ 字段级（JSON）</text>

  <rect x="20" y="46" width="840" height="74" rx="8" fill="#e6f4ea" stroke="#34a853"/>
  <text x="34" y="68" font-size="13" font-weight="700" fill="#1e6b34">① 分段级 · TLV：增删/跳过整块数据</text>
  <rect x="40" y="78" width="150" height="30" rx="4" fill="#fff" stroke="#34a853"/>
  <text x="115" y="98" font-size="12" text-anchor="middle" fill="#1e6b34">base_codes</text>
  <rect x="200" y="78" width="150" height="30" rx="4" fill="#fff" stroke="#34a853"/>
  <text x="275" y="98" font-size="12" text-anchor="middle" fill="#1e6b34">high_precise_codes</text>
  <rect x="360" y="78" width="120" height="30" rx="4" fill="#fff" stroke="#34a853"/>
  <text x="420" y="98" font-size="12" text-anchor="middle" fill="#1e6b34">graph</text>
  <rect x="490" y="78" width="150" height="30" rx="4" fill="#fff" stroke="#34a853" stroke-dasharray="4 3"/>
  <text x="565" y="98" font-size="12" text-anchor="middle" fill="#3a7a52">新能力块(可忽略)</text>
  <text x="700" y="98" font-size="12" fill="#3a7a52">→ 未知/不要则跳过</text>

  <rect x="20" y="132" width="840" height="78" rx="8" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="34" y="154" font-size="13" font-weight="700" fill="#1c3d7a">② 字段级 · JSON：增删标量参数（unknown 键忽略、缺键取默认）</text>
  <text x="40" y="180" font-size="12" fill="#1c3d7a" font-family="monospace">basic_info = { "dim":128, "metric":2, "use_reorder":true, "index_param":"…",</text>
  <text x="52" y="198" font-size="12" fill="#39527f" font-family="monospace">"new_field_in_v0.17": 42  ← 旧版本读者直接忽略此键 }</text>
</svg>

- **分段级（TLV）**：增删/跳过整块（datacell、某精度、某能力）。靠 `Tag + Length + flags.CRITICAL`。
- **字段级（JSON）**：增删标量参数。靠 JSON 的“未知键忽略、缺键默认”。

---

## 6. TLV 记录与 Head 的字节布局

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 250" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <text x="20" y="26" font-size="16" font-weight="700" fill="#222">TLV 记录字节布局</text>

  <!-- fields -->
  <rect x="20"  y="56" width="120" height="56" rx="5" fill="#fef0e3" stroke="#f29d49"/>
  <text x="80"  y="80" font-size="13" text-anchor="middle" fill="#b5651d" font-weight="700">tag</text>
  <text x="80"  y="99" font-size="11" text-anchor="middle" fill="#b5651d" font-family="monospace">u32 (4B)</text>

  <rect x="140" y="56" width="100" height="56" rx="5" fill="#fef0e3" stroke="#f29d49"/>
  <text x="190" y="80" font-size="13" text-anchor="middle" fill="#b5651d" font-weight="700">flags</text>
  <text x="190" y="99" font-size="11" text-anchor="middle" fill="#b5651d" font-family="monospace">u16 (2B)</text>

  <rect x="240" y="56" width="150" height="56" rx="5" fill="#fef0e3" stroke="#f29d49"/>
  <text x="315" y="80" font-size="13" text-anchor="middle" fill="#b5651d" font-weight="700">value_len</text>
  <text x="315" y="99" font-size="11" text-anchor="middle" fill="#b5651d" font-family="monospace">u64 (8B)</text>

  <rect x="390" y="56" width="470" height="56" rx="5" fill="#e6f4ea" stroke="#34a853"/>
  <text x="625" y="80" font-size="13" text-anchor="middle" fill="#1e6b34" font-weight="700">value</text>
  <text x="625" y="99" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">value_len 字节（块内容；可递归嵌套 TLV）</text>

  <!-- offsets -->
  <text x="20"  y="128" font-size="11" fill="#888" font-family="monospace">0</text>
  <text x="140" y="128" font-size="11" fill="#888" font-family="monospace">4</text>
  <text x="240" y="128" font-size="11" fill="#888" font-family="monospace">6</text>
  <text x="390" y="128" font-size="11" fill="#888" font-family="monospace">14</text>

  <!-- notes -->
  <text x="20" y="166" font-size="12" fill="#444">• flags.bit0 = <tspan font-weight="700">CRITICAL</tspan>：读者不认识此 tag 时<tspan font-weight="700"> 必须报错</tspan>（否则可安全跳过）。</text>
  <text x="20" y="188" font-size="12" fill="#444">• <tspan font-family="monospace">SECTION_END</tspan> 哨兵：<tspan font-family="monospace">tag=0, flags=0, value_len=0</tspan>，标记 body 结束。</text>
  <text x="20" y="210" font-size="12" fill="#444">• 超大/不可测块：<tspan font-family="monospace">value_len = U64_MAX</tspan> 表示<tspan font-weight="700">分块编码</tspan> → value = <tspan font-family="monospace">[chunk_len][chunk]…[chunk_len=0]</tspan>。</text>
  <text x="20" y="232" font-size="12" fill="#444">• tag 取自<tspan font-weight="700">集中注册表</tspan>（数值稳定、只增不复用）；<tspan font-family="monospace">_manifest</tspan> 里可附人类可读名便于调试。</text>
</svg>

**Head**（顺序结构）：`magic[8]="vsagstm0"` → `format_version:u32` → `meta_len:u64` → `metadata(JSON)` → `crc32:u32`（CRC 覆盖 metadata 字节）。metadata 关键键：`_version`（驱动后向默认）、`_min_reader_version`（可选全局闸门）、`_empty`、`basic_info`、`_manifest`（本产物包含的 tag 清单）。

---

## 7. 写端：无 Seek 下如何拿到 Length

TLV 要求“先写 Length 再写 Value”，而无 seek 写端不能回填，故每块长度必须**先于写出而知**。三种实现（可混用），推荐 **T1**：

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 300" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ah-g2" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#5b6b7a"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="16" font-weight="700" fill="#222">T1（推荐）：整体测量预扫描 → 真正写出（零额外内存）</text>

  <!-- pass 1 -->
  <text x="20" y="62" font-size="13" font-weight="700" fill="#5b6b7a">① 测量预扫描（空写计数器，复用 CalSerializeSize 的 no-op writer）</text>
  <rect x="40" y="74" width="130" height="44" rx="5" fill="#f1f3f4" stroke="#9aa0a6"/>
  <text x="105" y="101" font-size="12" text-anchor="middle" fill="#5b6b7a">块1 → size₁</text>
  <rect x="180" y="74" width="130" height="44" rx="5" fill="#f1f3f4" stroke="#9aa0a6"/>
  <text x="245" y="101" font-size="12" text-anchor="middle" fill="#5b6b7a">块2 → size₂</text>
  <rect x="320" y="74" width="130" height="44" rx="5" fill="#f1f3f4" stroke="#9aa0a6"/>
  <text x="385" y="101" font-size="12" text-anchor="middle" fill="#5b6b7a">块N → sizeₙ</text>
  <text x="470" y="101" font-size="12" fill="#9aa0a6">（输出全部丢弃，仅累加字节数）</text>

  <line x1="245" y1="128" x2="245" y2="158" stroke="#5b6b7a" stroke-width="2" marker-end="url(#ah-g2)"/>
  <text x="300" y="148" font-size="12" fill="#5b6b7a">得到每块 size</text>

  <!-- pass 2 -->
  <text x="20" y="186" font-size="13" font-weight="700" fill="#1e6b34">② 真正写出（一趟流式）</text>
  <rect x="40" y="198" width="120" height="50" rx="5" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="100" y="228" font-size="12" text-anchor="middle" fill="#1c3d7a">Head</text>
  <rect x="170" y="198" width="200" height="50" rx="5" fill="#e6f4ea" stroke="#34a853"/>
  <text x="270" y="220" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">tag|flags|len=size₁</text>
  <text x="270" y="238" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">| value₁</text>
  <rect x="380" y="198" width="200" height="50" rx="5" fill="#e6f4ea" stroke="#34a853"/>
  <text x="480" y="220" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">tag|flags|len=size₂</text>
  <text x="480" y="238" font-size="11" text-anchor="middle" fill="#1e6b34" font-family="monospace">| value₂</text>
  <text x="600" y="228" font-size="18" fill="#34a853">… END</text>

  <text x="20" y="284" font-size="12" fill="#444">T2 逐段缓冲（单块峰值内存，单趟）；T3 分块编码（超大/不可测块，单趟、无需预知总长）。三者读端跳过语义统一。</text>
</svg>

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

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 470" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ah-f" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#555"/>
    </marker>
  </defs>
  <text x="20" y="26" font-size="16" font-weight="700" fill="#222">反序列化读循环（全程前向）</text>

  <!-- nodes -->
  <rect x="250" y="44" width="280" height="40" rx="6" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="390" y="69" font-size="13" text-anchor="middle" fill="#1c3d7a">解析 Head（magic/版本/JSON/crc 校验）</text>

  <rect x="300" y="108" width="180" height="44" rx="6" fill="#e6f4ea" stroke="#34a853"/>
  <text x="390" y="135" font-size="13" text-anchor="middle" fill="#1e6b34">读一条 TLV 头</text>

  <polygon points="390,176 510,210 390,244 270,210" fill="#fff8e1" stroke="#f2b705"/>
  <text x="390" y="206" font-size="12" text-anchor="middle" fill="#8a6d00">tag == END ?</text>
  <text x="390" y="223" font-size="11" text-anchor="middle" fill="#8a6d00">是 → 完成</text>

  <polygon points="390,266 540,304 390,342 240,304" fill="#fff8e1" stroke="#f2b705"/>
  <text x="390" y="298" font-size="12" text-anchor="middle" fill="#8a6d00">需要跳过此块？</text>
  <text x="390" y="315" font-size="11" text-anchor="middle" fill="#8a6d00">(未知 tag 或 调用方不需要)</text>

  <rect x="300" y="372" width="180" height="44" rx="6" fill="#e6f4ea" stroke="#34a853"/>
  <text x="390" y="392" font-size="12" text-anchor="middle" fill="#1e6b34">有界派发子反序列化</text>
  <text x="390" y="408" font-size="11" text-anchor="middle" fill="#1e6b34">(BoundedForwardReader)</text>

  <!-- right branch: skip / critical -->
  <polygon points="690,266 800,300 690,334 580,300" fill="#fde8e8" stroke="#d33"/>
  <text x="690" y="296" font-size="12" text-anchor="middle" fill="#a11">未知 且 CRITICAL?</text>
  <text x="690" y="313" font-size="11" text-anchor="middle" fill="#a11">是 → 报错</text>

  <rect x="600" y="372" width="200" height="44" rx="6" fill="#f1f3f4" stroke="#9aa0a6"/>
  <text x="700" y="392" font-size="12" text-anchor="middle" fill="#5b6b7a">SkipForward(value_len)</text>
  <text x="700" y="408" font-size="11" text-anchor="middle" fill="#5b6b7a">前向丢弃 / 分块读到 0</text>

  <rect x="640" y="44" width="180" height="40" rx="6" fill="#fde8e8" stroke="#d33"/>
  <text x="730" y="69" font-size="12" text-anchor="middle" fill="#a11">报错：需更高版本 vsag</text>

  <rect x="600" y="108" width="220" height="40" rx="6" fill="#ede7f6" stroke="#7e57c2"/>
  <text x="710" y="133" font-size="12" text-anchor="middle" fill="#4a328a">完成（缺块按 _version 取默认）</text>

  <!-- edges -->
  <line x1="390" y1="84" x2="390" y2="108" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <line x1="390" y1="152" x2="390" y2="176" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <line x1="510" y1="210" x2="600" y2="148" stroke="#7e57c2" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <line x1="390" y1="244" x2="390" y2="266" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <text x="404" y="262" font-size="11" fill="#555">否</text>
  <line x1="390" y1="342" x2="390" y2="372" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <text x="404" y="362" font-size="11" fill="#555">否</text>
  <line x1="540" y1="304" x2="580" y2="301" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <text x="548" y="294" font-size="11" fill="#555">是</text>
  <line x1="690" y1="334" x2="700" y2="372" stroke="#555" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <text x="710" y="356" font-size="11" fill="#a11">否(可忽略)</text>
  <line x1="690" y1="266" x2="730" y2="84" stroke="#d33" stroke-width="1.5" marker-end="url(#ah-f)"/>
  <text x="744" y="200" font-size="11" fill="#d33">是</text>

  <!-- loopback -->
  <path d="M 300 394 C 120 394, 120 130, 300 130" fill="none" stroke="#555" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#ah-f)"/>
  <path d="M 700 372 C 700 360, 180 360, 150 360 C 120 360, 120 130, 298 130" fill="none" stroke="#9aa0a6" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#ah-f)"/>
  <text x="92" y="262" font-size="11" fill="#555" transform="rotate(-90 92 262)">循环读下一条</text>
</svg>

- `BoundedForwardReader`：把子反序列化器限制在本块 `value_len` 内（可用现有**单参** `Slice(len)`，`stream_reader.cpp:218`，不 seek），读完后 `SkipForward` 跳过块内未读尾部 → **段内前向兼容**（新字段追加块尾时旧读者自动跳过）。

---

## 9. 兼容性 & 能力选择（同一套“跳过”机制）

前向兼容（低读高的未知块）与能力选择（加载方主动跳过不需要的块）复用**同一套** TLV 前向跳过，差别只在 `should_skip(tag)` 的来源。下图同一产物被两类读者处理：

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 330" font-family="-apple-system, Segoe UI, Roboto, sans-serif">
  <defs>
    <marker id="ah-ok" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#34a853"/></marker>
    <marker id="ah-sk" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#9aa0a6"/></marker>
  </defs>
  <text x="20" y="26" font-size="16" font-weight="700" fill="#222">同一产物，两类读者：consume（实读）vs skip（前向空读）</text>

  <!-- stream -->
  <rect x="20"  y="50" width="110" height="52" rx="6" fill="#e8f0fe" stroke="#4a78d6"/>
  <text x="75"  y="80" font-size="12" text-anchor="middle" fill="#1c3d7a">Head</text>
  <rect x="135" y="50" width="160" height="52" rx="6" fill="#e6f4ea" stroke="#34a853"/>
  <text x="215" y="73" font-size="12" text-anchor="middle" fill="#1e6b34">base_codes</text>
  <text x="215" y="91" font-size="11" text-anchor="middle" fill="#3a7a52" font-family="monospace">tag=11</text>
  <rect x="300" y="50" width="190" height="52" rx="6" fill="#e6f4ea" stroke="#34a853"/>
  <text x="395" y="73" font-size="12" text-anchor="middle" fill="#1e6b34">high_precise_codes</text>
  <text x="395" y="91" font-size="11" text-anchor="middle" fill="#3a7a52" font-family="monospace">tag=12 (可选)</text>
  <rect x="495" y="50" width="200" height="52" rx="6" fill="#e6f4ea" stroke="#34a853" stroke-dasharray="5 3"/>
  <text x="595" y="73" font-size="12" text-anchor="middle" fill="#1e6b34">新能力块(v0.17)</text>
  <text x="595" y="91" font-size="11" text-anchor="middle" fill="#3a7a52" font-family="monospace">tag=99, CRITICAL=0</text>
  <rect x="700" y="50" width="60" height="52" rx="6" fill="#ede7f6" stroke="#7e57c2"/>
  <text x="730" y="80" font-size="11" text-anchor="middle" fill="#4a328a">END</text>

  <!-- reader A: low version -->
  <text x="20" y="150" font-size="13" font-weight="700" fill="#444">读者 A：低版本代码（不认识 tag=99）→ 前向兼容</text>
  <line x1="215" y1="160" x2="215" y2="106" stroke="#34a853" stroke-width="2" marker-end="url(#ah-ok)"/>
  <text x="215" y="176" font-size="11" text-anchor="middle" fill="#1e6b34">实读 ✓</text>
  <line x1="395" y1="160" x2="395" y2="106" stroke="#34a853" stroke-width="2" marker-end="url(#ah-ok)"/>
  <text x="395" y="176" font-size="11" text-anchor="middle" fill="#1e6b34">实读 ✓</text>
  <line x1="595" y1="160" x2="595" y2="106" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah-sk)"/>
  <text x="595" y="176" font-size="11" text-anchor="middle" fill="#777">空读跳过（不认识）</text>

  <!-- reader B: selective -->
  <text x="20" y="236" font-size="13" font-weight="700" fill="#444">读者 B：加载参数 {"precision":"low"} → 能力选择</text>
  <line x1="215" y1="246" x2="215" y2="246" stroke="#34a853" stroke-width="2"/>
  <line x1="215" y1="262" x2="215" y2="246" stroke="#34a853" stroke-width="2" marker-end="url(#ah-ok)"/>
  <text x="215" y="278" font-size="11" text-anchor="middle" fill="#1e6b34">实读 ✓</text>
  <line x1="395" y1="262" x2="395" y2="246" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah-sk)"/>
  <text x="395" y="278" font-size="11" text-anchor="middle" fill="#777">空读跳过（不需要高精度）</text>
  <line x1="595" y1="262" x2="595" y2="246" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah-sk)"/>
  <text x="595" y="278" font-size="11" text-anchor="middle" fill="#777">空读跳过</text>

  <text x="20" y="312" font-size="12" fill="#a11">若 tag=99 标了 CRITICAL=1，则读者 A 必须报错“需更高版本 vsag”，而不会静默忽略。</text>
</svg>

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
