# 并行反序列化

加载大索引通常受 I/O 制约：顺序 `Deserialize` 路径用单线程读取并解压文件，多核机器和高带宽存储都没有被
充分利用。`ParallelDeserialize` 从[分块文件](chunked_serialization.md)还原索引，每帧一个任务，由调用方
提供的定位读 reader 和绑定到索引的线程池驱动。

并行入口目前仅 HGraph 实现，其他索引类型返回不支持该操作的错误。

## DeserializeReader

数据源由调用方提供。VSAG 自身从不打开文件，因此对象存储、分布式文件系统和本地文件都能收敛到同一个接口：

```cpp
class MyReader : public vsag::DeserializeReader {
public:
    uint64_t Size() const override { /* 索引数据总大小 */ }

    void Read(uint64_t offset, uint64_t len, void* dest) override {
        /* 定位读；短读和越界必须抛异常 */
    }

    void ReadDecompressed(uint64_t offset, uint64_t compressed_size,
                          const std::function<void(std::istream&)>& consume) override {
        /* 把 [offset, offset + compressed_size) 处的帧解压成顺序流交给 consume；
           仅当文件由压缩 writer 写出时才需要实现 */
    }
};
```

该接口分两个层次：只实现 `Size` 和 `Read` 的普通数据源可以加载未压缩的分块文件；额外实现
`ReadDecompressed` 的数据源可以加载压缩文件。

`Read` 与 `ReadDecompressed` 会被工作线程在互不重叠的区间上并发调用，因此实现必须对定位读**线程安全**。
如果句柄的读取共享游标、或在内部锁上串行化，即使线程池很大，加载也会被静默降级为单条 I/O 流；应优先使用
每次调用独立的定位读（`pread` 语义）或句柄池。每个解压流只会被一个线程消费。

## 加载索引

```cpp
MyReader reader;                   // 包装文件 / 对象存储句柄
auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();

index->ParallelDeserialize(reader).value();        // 使用内部默认线程池
```

若要在自己的执行器上加载，把线程池绑定到创建索引所用的 `Resource`：

```cpp
auto allocator = vsag::Engine::CreateDefaultAllocator();
auto pool = vsag::Engine::CreateThreadPool(16).value();

vsag::Resource resource(allocator, pool);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", build_params).value();

index->ParallelDeserialize(reader).value();        // 运行在绑定的线程池上
```

绑定线程池可以让调用方复用已有的执行器，并约束整个进程的并发度。线程池的生命周期必须覆盖整个调用。

## 加载阶段

1. **准备（主线程）**：解析 footer，并为每个 chunked 组件读取 head、预分配 io 区间。
2. **填充（线程池）**：每帧一个任务，压缩时先解压，再按记录的偏移写入预分配区间。各任务写入互不相交的
   字节范围，因此无需加锁。
3. **收尾（主线程）**：读取各组件 tail，执行与顺序路径相同的加载后初始化。

阶段 1 的预分配正是阶段 2 能够无锁的前提：工作线程写入期间区间不会再扩容。派发任务之前会先用 body 范围
校验记录的布局：组件名必须唯一、每帧都必须落在 body 内且不溢出、帧之间不得重叠、未压缩 codec 记录的物理
大小必须等于逻辑大小。此外每帧字节数都会与布局做校验，每个 whole 组件也必须恰好消费其记录的逻辑大小，
因此损坏或被篡改的帧会明确报错，而不会污染内存。

## 加载不带布局的文件

在分块格式出现之前写出的索引，footer 里没有 `chunked_layout`，但只要其 body 未压缩就仍可并发加载——
包括 `Serialize(ostream)` 写出的全合一文件。`ParallelDeserialize` 检测到布局缺失后会切换到探测路径：

1. 主线程按序列化顺序走一遍 body。对每个暴露了并行钩子的组件，预分配其 io 区间、记录该区间的文件偏移、
   跳过 io 字节、读取 tail；没有钩子的组件则就地反序列化。
2. 随后由线程池填充这些已记录的区间，按 `DEFAULT_SERIALIZE_CHUNK_SIZE` 字节切分成任务。

这条路径要求 body 未压缩：没有记录布局时无法定位压缩 body 中的各个帧，因此只有 body 字节按原样存放的
文件才能被探测。

由于每个组件的 head 和 tail 仍由主线程顺序读取，探测路径的扩展性弱于布局路径，但它无需对既有文件做任何
重新序列化。

峰值内存由各组件所选的 io 类型决定，而非取决于是否记录了布局：`SkipDeserialize` 的 io（如 `reader_io`）
在两条路径下都只记录区间并跳过字节，file-backed io 写入其 backing 文件，只有内存型 io 才把组件驻留在
堆上。recorded layout 换来的是并发度与压缩帧支持。

## 并发与 IO 注意事项

- **Allocator**：工作线程可能通过索引 allocator 分配内存，自定义 allocator 必须线程安全。
- **内存**：帧是流式处理而非整块攒起来的。每个任务只持有一个有界读窗口和一个编解码上下文，因此峰值内存
  随线程数增长，而与 `chunk_size` 无关。
- **whole 组件**每个由单个任务处理。对压缩文件它们只能前向读取，因此组件的反序列化不得在帧内回退定位。
- **文件型 io**：内存、mmap、buffered、异步 io 都会对大组件做并发填充；`reader_io` 除外——它的写路径是
  空实现——其组件回落为 whole 帧，各由一个任务加载。
- **线程数**：收益主要来自 I/O 并发。由于工作线程大部分时间阻塞在读取上，按存储带宽和独立帧数量来设置
  线程池，通常比对齐 CPU 核数更有效。

## 性能

100 万 × 1024 维向量，`sq8` base + `fp32` reorder，32 MiB 帧，zstd，索引文件 4.6 GiB，在 48 核机器上从
page cache 加载。以顺序 `Deserialize` 为基准的加速比，基准的墙钟耗时列在第二列：

| io 类型 | 顺序 | 4 线程 | 8 线程 | 16 线程 |
| --- | --- | --- | --- | --- |
| block_memory_io | 12.53 s | 1.70x | 2.98x | 4.64x |
| memory_io | 13.19 s | 1.83x | 3.25x | 5.19x |
| mmap_io | 14.51 s | 1.75x | 3.05x | 3.50x |
| buffer_io | 11.83 s | 1.67x | 2.74x | 2.68x |

召回不受影响：上述所有配置返回的结果与内存中的索引逐条查询完全一致。

对于分块格式出现之前的文件，探测路径也能带来可观但略小的加速。同一个索引改用 `Serialize(ostream)` 写出
（5.0 GiB，未压缩，block_memory_io），以单参数 `Deserialize(istream)` 为基准：

| 模式 | 墙钟耗时 | 加速比 |
| --- | --- | --- |
| `Deserialize(istream)` | 7.42 s | 1.00x |
| `ParallelDeserialize`，4 线程 | 3.79 s | 1.96x |
| `ParallelDeserialize`，8 线程 | 3.06 s | 2.43x |
| `ParallelDeserialize`，16 线程 | 2.77 s | 2.67x |

注意单线程的并行加载**比顺序路径更慢**。每个帧都要建立自己的编解码会话、发起自己的定位读，而只有一个
工作线程时这些开销无法与任何东西重叠。约在 4 线程处反超。

## 参见

- [分块序列化格式](chunked_serialization.md)——本路径消费的文件格式。
- [序列化格式](serialization.md)——顺序接口与 footer 格式。
