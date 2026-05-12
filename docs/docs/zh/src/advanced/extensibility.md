# 可扩展性

VSAG 暴露了一组稳定的 C++ 扩展点，方便应用接入自有基础设施而无需 fork 库本身。
本页梳理 **哪些可以扩展**、**哪些不可以**，并给出可运行示例的链接。

## 公开扩展点

| 扩展点 | 头文件 | 用途 |
|---|---|---|
| `vsag::Allocator` | `vsag/allocator.h` | 自定义内存分配策略。 |
| `vsag::Logger` | `vsag/logger.h` | 把 VSAG 日志重定向到你的日志体系。 |
| `vsag::ThreadPool` | `vsag/thread_pool.h` | 复用外部线程池执行 build 和 IO。 |
| `vsag::Filter` | `vsag/filter.h` | 为 `KnnSearch` / `RangeSearch` 提供自定义预过滤器。 |
| `vsag::Reader`（含 `ReaderSet`） | `vsag/readerset.h` | 自定义反序列化的 IO 后端。 |

这五个都是抽象基类。每个至少声明一个必须实现的纯虚方法；部分还声明了带默认实现的非纯虚方法
（例如 `Filter::CheckValid(const char*)`、`Filter::ValidRatio()`、
`Filter::FilterDistribution()`、`Filter::GetValidIds()`，以及 `Reader::MultiRead()`），只在需
要自定义行为时才需要 override。实现必须的方法、用 `std::shared_ptr` 包装（或在 API 要求时
直接传裸指针），然后交给 VSAG 即可。

## 把扩展接入索引

主要有两条接入路径。

### 1. 通过 `Engine` 注入按索引生效的资源

`vsag::Engine`（`vsag/engine.h`）是绑定自定义 `Allocator` 与 `ThreadPool` 的
推荐方式，绑定后由它创建的每个索引都会共享这些资源：

```cpp
auto allocator   = std::make_shared<MyAllocator>();
auto thread_pool = std::make_shared<MyThreadPool>();
vsag::Resource resource(allocator, thread_pool);
vsag::Engine engine(&resource);

auto index = engine.CreateIndex("hgraph", parameters).value();
// ... 使用索引 ...
engine.Shutdown();
```

`Engine(Resource*)` 接收的是 **non-owning** 裸指针（见
`include/vsag/engine.h:38-42`）：调用者必须保证 `Resource`（连同它持有的
allocator 与 thread pool）的生命周期长于 engine 以及 engine 创建的所有索引。
`Engine::Shutdown()` 释放 engine 内部资源，但不会销毁外部的 `Resource`。
`Resource` 提供两个构造器（`include/vsag/resource.h:45,59-60`）：既可以传裸
`Allocator*` / `ThreadPool*`（生命周期由调用者管理），也可以传 `shared_ptr`
重载，让 `Resource` 共享所有权。完整的所有权模型见 [内存管理](memory.md)，
把 allocator 收敛到单次搜索调用的用法见 [搜索路径 Allocator](search_allocator.md)。

如果只是想快速跑通，`Engine::CreateDefaultAllocator()` 与
`Engine::CreateThreadPool(num_threads)` 会返回开箱即用的实现。

### 2. 通过 `Factory::CreateIndex` 传裸 allocator

`vsag::Factory::CreateIndex(name, params, allocator)`（`vsag/factory.h`）接受
一个可选的 `Allocator*`。这条路径不接受线程池，新代码建议改用 `Engine`。

## Filter

实现 `vsag::Filter`，通过 `SearchRequest::filter_`（或已弃用的
`SearchParam::filter`）传入 `FilterPtr` 即可。使用 `SearchRequest` 时，必须
同时把 `enable_filter_` 设为 `true`，filter 才会真正生效
（见 `include/vsag/search_request.h:113,123`）。只有 `CheckValid(int64_t id)`
是必须实现的，其他都是可选的优化钩子：

- `CheckValid(const char* data)`：基于向量 extra info 过滤。
- `ValidRatio()`：向规划器提示选择度。
- `FilterDistribution()`：返回 `NONE`（默认）或 `RELATED_TO_VECTOR`，声明有效
  id 的分布是否与向量在底层存储中的位置相关
  （见 `include/vsag/filter.h:27-30`）。
- `GetValidIds(...)`：对于选择度极低的过滤器，提供预先计算好的有效 id 列表。

可运行示例：`examples/cpp/301_feature_filter.cpp`。过滤接入的细节见
[过滤搜索](filtered_search.md)。

## Reader / ReaderSet

`Index::Deserialize(const ReaderSet&)`（`include/vsag/index.h:810`）允许通过
per-stream 的 `Reader` 从任意存储后端（本地文件、对象存储、远程文件系统…）
反序列化索引。至少实现 `Read`、`AsyncRead`、`Size` 三个方法；`MultiRead` 是
可选的，当底层支持批量 IO 时能显著提升吞吐。`vsag::Factory::CreateLocalFileReader`
是本地文件的参考实现。

可运行示例：`examples/cpp/102_index_diskann.cpp`（DiskANN 的反序列化基于
`ReaderSet`）。完整的序列化/反序列化矩阵见 [序列化](serialization.md)。

## Logger

VSAG 使用全局唯一的 logger，通过 `Options` 单例配置：

```cpp
class MyLogger : public vsag::Logger { /* 实现 Trace/Debug/Info/... */ };
static MyLogger my_logger;
vsag::Options::Instance().set_logger(&my_logger);
```

logger 指针的所有权 **不** 归 VSAG —— 必须在所有 VSAG 调用期间保持其存活。
传入 `nullptr` 则回退到内置 logger。

可运行示例：`examples/cpp/202_custom_logger.cpp`。

## 通过 `Options` 进行全局调参

`vsag::Options::Instance()`（`vsag/options.h`）是进程级单例，承载与具体索引
无关的设置：

| 接口 | 默认值 | 备注 |
|---|---|---|
| `set_num_threads_io(n)` | `8` | 搜索时磁盘索引的 IO 线程数，取值范围 `[1, 200]`。 |
| `set_num_threads_building(n)` | `4` | 构建磁盘索引使用的线程数。 |
| `set_block_size_limit(bytes)` | `128 MiB` | 单次分配 block 的最大值，必须 ≥ 256 KiB（见 `src/options.cpp:53-57`）。 |
| `set_direct_IO_object_align_bit(bits)` | `9` | Direct-IO 对齐位数，必须 ≤ 21（见 `src/options.cpp:40-46`）。 |
| `set_logger(Logger*)` | 内置 | 见上文 [Logger](#logger)。 |

这些 option 对进程内所有索引生效，建议启动时设置一次。它们 **不会** 覆盖
HGraph 的 `build_thread_count` 等具体索引参数。

## 哪些 *不* 提供公开扩展接口

以下能力目前 **没有** 稳定的公开扩展接口：

- **量化器（Quantizer）。** 具体量化类型（SQ8、PQ、RaBitQ…）通过索引参数选择，
  不支持用户代码继承扩展。
- **距离计算器 / 距离类型。** 每个索引的可选 metric 固定为 `l2`、`ip`、
  `cosine`。
- **索引内部的 DataCell / IO / 存储后端。** 这些都是实现细节。若需要自定义
  IO，请在反序列化边界使用 `Reader` 接口。

如果你的场景需要上述任一能力，请提 issue 描述使用场景。

## 关于 `vsag::ext`

`vsag/vsag_ext.h` 提供了一组基于 handle 的精简 API（`IndexHandler`、
`DatasetHandler`、`BitsetHandler` ……），用于 **语言绑定 / FFI**，并不是面向最终用户的扩展层。
C++ 应用应直接使用标准的 `vsag::Index` API。

## 相关示例

- `examples/cpp/201_custom_allocator.cpp`
- `examples/cpp/202_custom_logger.cpp`
- `examples/cpp/203_custom_thread_pool.cpp`
- `examples/cpp/301_feature_filter.cpp`
- `examples/cpp/102_index_diskann.cpp`
