# 分块序列化格式

分块序列化把索引 body 写成一串可独立读取的帧，并把它们的物理位置记录在 footer 中。由于每个帧都能被
单独定位和还原，写出的文件可以并发加载，见[并行反序列化](parallel_deserialization.md)。

本文介绍 `Serialize(SerializeWriter&)` 产生的分块格式。它是[序列化格式](serialization.md)所述
footer 格式的一种变体；[新序列化格式](new_serialization.md)中的 header-first 格式是另一套独立机制，
两者不可互换。

分块格式目前仅 HGraph 实现，其他索引类型返回不支持该操作的错误。

## SerializeWriter

`SerializeWriter` 是一个可选地感知帧边界的字节汇（sink）。编解码方式由实现决定：普通 writer 原样写出
body，压缩 writer（例如 zstd）则为每个分块产生一个独立的压缩帧。

```cpp
class MyWriter : public vsag::SerializeWriter {
public:
    void Write(const char* data, uint64_t size) override { /* 追加字节 */ }

    // 压缩是可选的。不压缩的 writer 返回空名字，帧钩子留空即可。
    std::string GetCompressorName() const override { return "zstd"; }
    void BeginCompressedFrame() override { /* 重置编解码会话 */ }
    uint64_t EndCompressedFrame() override { /* flush，返回写出字节数 */ }
};
```

每个帧都是自包含的：`BeginCompressedFrame` 会重置编解码会话，因此读侧可以在不触碰相邻帧的情况下解压
任意单个帧。

## 写出分块文件

```cpp
MyWriter writer(out);
index->Serialize(writer).value();                    // chunk_size 默认 128 MiB
index->Serialize(writer, 64 * 1024 * 1024).value();  // 也可以指定帧大小
```

body 被切分为逻辑字节数不超过 `chunk_size` 的帧。`chunk_size` 是**文件属性而非运行期开关**：它被写入
footer，读侧始终以文件中记录的值为准。因此调整该值只影响新写出的文件，不会让已有文件失效。

每个组件按两种粒度之一写出：

- **whole**——组件占用单个帧。小组件、以及未暴露并行填充钩子的组件按此方式写出。
- **chunked**——组件由 head / io 数据 / tail 三段构成。只有 io 数据会被切分成帧，head 和 tail 保持明文。

## Footer 布局

footer 中的 `chunked_layout` 键描述了每个组件的物理位置：

```json
{
  "chunked_layout": {
    "version": 1,
    "codec": "zstd",
    "chunk_size": 33554432,
    "components": [
      { "name": "label_table", "type": "whole",
        "offset": 0, "csize": 489597, "lsize": 2002832 },
      { "name": "base_codes", "type": "chunked", "io_size": 102760448,
        "head": { "offset": 489597, "size": 20 },
        "chunks": [ { "offset": 489617, "csize": 33555209 } ],
        "tail": { "offset": 102892065, "size": 8229 } }
    ]
  }
}
```

该布局是读侧的唯一事实来源：读侧从数据源末尾解析 footer，并据此驱动全部读取，无需预先知道组件顺序。
解析阶段有两道校验——记录的分块必须精确覆盖组件区间，且未知组件类型会被拒绝——因此比格式扩展更旧的
读侧会快速失败，而不会错误解析 body。

## 兼容性

| 文件形态 | `Deserialize(istream)` | `ParallelDeserialize` |
| --- | --- | --- |
| 既有全合一、未压缩 | 支持 | 支持（探测路径） |
| 分块、未压缩 | 支持（body 字节顺序与既有格式一致） | 支持（布局路径） |
| 分块、已压缩 | 明确报错拒绝 | 支持（布局路径） |

未压缩的分块文件仍可被顺序路径读取，因为其 body 字节保持与既有格式相同的顺序，而多出的
`chunked_layout` 键会被旧读侧忽略。压缩后的分块 body 是一串独立帧而非单一数据流，因此顺序读侧会直接
拒绝，而不会错误解析。

反方向，完全不带布局的未压缩文件——包括 `Serialize(ostream)` 写出的全合一文件——仍可以并发加载，见
[加载不带布局的文件](parallel_deserialization.md#加载不带布局的文件)。

## 注意事项

- 帧大小是 footer 体积与加载并发度之间的权衡：帧越小并行任务越多，但 footer 中每帧约增加一条记录。
- 编解码器名字记录在布局中；读侧若无法提供匹配的解压器，会得到明确的报错。
