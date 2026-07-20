# 序列化类型

VSAG 可以用两种形态持久化索引：内存中的 [`BinarySet`](#binaryset)（一组带名字的字节块），或在磁盘 /
流式场景下由惰性 [`Reader`](#reader) 对象组成的 [`ReaderSet`](#readerset)。这些类型正是传给
[`Index::Serialize` / `Index::Deserialize`](index_class.md#序列化) 的负载。

端到端流程与基于流的序列化见 [序列化格式](../advanced/serialization.md) 与
`examples/cpp/401_persistent_kv.cpp` / `402_persistent_streaming.cpp`。

## `Binary`

声明于 `vsag/binaryset.h`。一块带长度的、有名字的字节缓冲区。

```cpp
struct Binary {
    std::shared_ptr<int8_t[]> data;  // 字节数据
    uint64_t size;                   // 字节数
};
```

`shared_ptr` 拥有该缓冲区，因此 `Binary` 可以自由拷贝与存储，无需担心生命周期。

## `BinarySet`

声明于 `vsag/binaryset.h`。一个以字符串为键的 `Binary` 块映射 —— 标准的内存序列化容器。索引会把自身
序列化为若干带名字的部分（图、向量、量化器等），全部汇集在一个 `BinarySet` 中。

```cpp
class BinarySet {
public:
    void Set(const std::string& name, Binary binary);   // 存入一个块
    Binary Get(const std::string& name) const;          // 不存在时返回 {nullptr, 0}
    std::vector<std::string> GetKeys() const;            // 所有已存名字
    bool Contains(const std::string& key) const;
};
```

| 方法 | 说明 |
|------|------|
| `Set(name, binary)` | 以 `name` 存入 `binary`，覆盖任何已有条目。 |
| `Get(name)` | 返回该块；名字不存在时返回空的 `Binary{nullptr, 0}`。 |
| `GetKeys()` | 返回每一个已存名字。 |
| `Contains(key)` | `key` 下是否存有块。 |

```cpp
// 序列化为 BinarySet，然后按你喜欢的方式持久化每个部分。
auto serialized = index->Serialize();
if (serialized.has_value()) {
    vsag::BinarySet bs = serialized.value();
    for (const auto& key : bs.GetKeys()) {
        vsag::Binary part = bs.Get(key);
        // 以 `key` 为键，把 part.data[0 .. part.size) 写入你的存储
    }
}
```

要恢复，从你的存储重建 `BinarySet`，并在一个全新（空）索引上调用
[`Deserialize(const BinarySet&)`](index_class.md#序列化)。

## `Reader`

声明于 `vsag/readerset.h`。一个抽象的字节来源，索引按需从中读取 —— 这是在不把全部内容载入内存的前提下
反序列化大型磁盘常驻索引的基础。可从
[`Factory::CreateLocalFileReader`](factory_engine.md#createlocalfilereader) 获取本地文件 reader，或为
自定义后端（对象存储、mmap 等）实现 `Reader`。通过 `ReaderPtr`（`std::shared_ptr<Reader>`）持有它。

```cpp
class Reader {
public:
    virtual void Read(uint64_t offset, uint64_t len, void* dest) = 0;                  // 同步
    virtual void AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack cb) = 0; // 异步
    virtual bool MultiRead(uint8_t* dests, const uint64_t* lens,
                           const uint64_t* offsets, uint64_t count);                    // 批量
    virtual uint64_t Size() const = 0;
};
```

| 方法 | 说明 |
|------|------|
| `Read(offset, len, dest)` | 同步地从 `offset` 拷贝 `len` 字节到 `dest`。线程安全。 |
| `AsyncRead(offset, len, dest, callback)` | 异步读；完成时以 [`IOErrorCode`](#ioerrorcode) 和 message 调用 `callback`。 |
| `MultiRead(dests, lens, offsets, count)` | 在一次调用中执行 `count` 次同步读；任一失败返回 `false`。 |
| `Size()` | 底层来源的总字节数。 |

### `IOErrorCode`

```cpp
enum class IOErrorCode {
    IO_SUCCESS = 0,  // 操作成功
    IO_ERROR = 1,    // 一般 I/O 错误
    IO_TIMEOUT = 2,  // 操作超时
};
```

### `CallBack`

```cpp
using CallBack = std::function<void(IOErrorCode code, const std::string& message)>;
```

`AsyncRead` 的完成回调。

## `ReaderSet`

声明于 `vsag/readerset.h`。一个以字符串为键的 `Reader` 对象映射 —— `BinarySet` 的流式对应物。序列化
索引的每个带名字的部分都映射到一个按需拉取该部分的 `Reader`。把一个填充完整的 `ReaderSet` 传给
[`Deserialize(const ReaderSet&)`](index_class.md#序列化)。

```cpp
class ReaderSet {
public:
    void Set(const std::string& name, ReaderPtr reader);
    ReaderPtr Get(const std::string& name) const;   // 不存在时为 nullptr
    std::vector<std::string> GetKeys() const;
    bool Contains(const std::string& key) const;
};
```

其方法语义与 [`BinarySet`](#binaryset) 一致，只是值为 `ReaderPtr` 而非 `Binary`。

```cpp
vsag::ReaderSet readers;
readers.Set("graph", vsag::Factory::CreateLocalFileReader("index.graph", 0, graph_size));
readers.Set("vectors", vsag::Factory::CreateLocalFileReader("index.vectors", 0, vec_size));

auto fresh = vsag::Factory::CreateIndex("hgraph", params).value();
fresh->Deserialize(readers);
```

## 参见

- [Index](index_class.md#序列化) —— `Serialize` / `Deserialize` 方法族。
- [Factory 与 Engine](factory_engine.md#createlocalfilereader) —— 创建基于文件的 reader。
- [序列化格式](../advanced/serialization.md) —— 完整的持久化指南。
