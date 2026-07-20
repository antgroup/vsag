# 资源管理

VSAG 允许你掌控它所使用的内存与线程。本页介绍 [`Allocator`](#allocator)（自定义内存管理）、
[`ThreadPool`](#threadpool)（自定义并发）、[`Resource`](#resource)（由
[`Engine`](factory_engine.md#engine) 共享的两者打包）、进程级的 [`Options`](#options) 单例，以及可插拔的
[`Logger`](#logger)。

可运行示例：`examples/cpp/201_custom_allocator.cpp`、`202_custom_logger.cpp`、
`203_custom_thread_pool.cpp`。另见 [内存管理](../advanced/memory.md) 与
[可扩展性](../advanced/extensibility.md)。

## `Allocator`

声明于 `vsag/allocator.h`。用于自定义内存管理的抽象接口。实现它即可把索引的全部分配路由到你自己的池、
arena 或记账层，然后把它传给 [`Factory::CreateIndex`](factory_engine.md#createindex) 或一个
[`Resource`](#resource)。

```cpp
class Allocator {
public:
    virtual std::string Name() = 0;
    virtual void* Allocate(uint64_t size) = 0;
    virtual void Deallocate(void* p) = 0;
    virtual void* Reallocate(void* p, uint64_t size) = 0;

    template <typename T, typename... Args> T* New(Args&&... args);  // 分配 + 构造
    template <typename T> void Delete(T* p);                          // 析构 + 释放
};
```

| 成员 | 说明 |
|------|------|
| `Name()` | allocator 实现的标识（用于诊断）。 |
| `Allocate(size)` | 返回至少 `size` 字节的内存块。 |
| `Deallocate(p)` | 释放先前由本 allocator 返回的内存块。 |
| `Reallocate(p, size)` | 调整内存块大小，保留其内容。 |
| `New<T>(args...)` | 辅助方法：分配并构造一个 `T`；若构造函数抛异常则释放并重新抛出。 |
| `Delete<T>(p)` | 辅助方法：析构 `*p` 并释放其存储（对 null 安全）。 |

传给索引的 allocator 必须比该索引活得更久。VSAG 内置 allocator 可通过
[`Engine::CreateDefaultAllocator`](factory_engine.md#静态资源辅助方法) 获取。

## `ThreadPool`

声明于 `vsag/thread_pool.h`。一个抽象的任务执行器。提供你自己的实现，即可让 VSAG 共享你应用的线程，而
不是自行创建。

```cpp
class ThreadPool {
public:
    virtual void WaitUntilEmpty() = 0;
    virtual void SetQueueSizeLimit(std::uint64_t limit) = 0;
    virtual void SetPoolSize(std::uint64_t limit) = 0;
    virtual std::future<void> Enqueue(std::function<void(void)> task) = 0;
};
```

| 成员 | 说明 |
|------|------|
| `WaitUntilEmpty()` | 阻塞直到所有已入队任务完成。 |
| `SetQueueSizeLimit(limit)` | 限制待处理任务队列；超过上限后的行为由实现定义。 |
| `SetPoolSize(limit)` | 限制工作线程数量。 |
| `Enqueue(task)` | 提交一个任务；返回用于其完成状态的 `std::future<void>`。 |

现成的线程池可通过 [`Engine::CreateThreadPool`](factory_engine.md#静态资源辅助方法) 创建。

## `Resource`

声明于 `vsag/resource.h`。`Resource` 把一个 [`Allocator`](#allocator) 与一个
[`ThreadPool`](#threadpool) 打包，使一个 [`Engine`](factory_engine.md#engine) —— 及其创建的每个索引 ——
都能共享它们。

```cpp
class Resource {
public:
    explicit Resource(Allocator* allocator, ThreadPool* thread_pool);
    explicit Resource(const std::shared_ptr<Allocator>& allocator,
                      const std::shared_ptr<ThreadPool>& thread_pool);
    explicit Resource();  // 默认 allocator，无线程池

    std::shared_ptr<Allocator> GetAllocator() const;
    std::shared_ptr<ThreadPool> GetThreadPool() const;
};
```

| 构造函数 / 方法 | 说明 |
|-----------------|------|
| `Resource(Allocator*, ThreadPool*)` | 使用你拥有的裸指针。allocator 为 null 表示“创建并拥有一个默认的”；线程池为 null 表示“无线程池”。 |
| `Resource(shared_ptr, shared_ptr)` | 同上，采用共享所有权。 |
| `Resource()` | 默认 allocator，无线程池。 |
| `GetAllocator()` | 该资源的 allocator（若未提供则为默认的）。 |
| `GetThreadPool()` | 该资源的线程池，若未提供则为 null。 |

```cpp
auto alloc = vsag::Engine::CreateDefaultAllocator();
auto pool = vsag::Engine::CreateThreadPool(4).value();
vsag::Resource resource(alloc, pool);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", params);
```

## `Options`

声明于 `vsag/options.h`。用于全局配置的进程级单例，通过 `Options::Instance()` 访问。线程安全。`Option`
是 `Options` 的类型别名。

```cpp
vsag::Options::Instance().set_num_threads_building(8);
vsag::Options::Instance().set_logger(&my_logger);
```

| 配置项 | 访问器 | 默认值 | 含义 |
|--------|--------|--------|------|
| IO 线程 | `num_threads_io()` / `set_num_threads_io(n)` | `8` | 搜索期间磁盘索引 IO 的线程数（1–200）。 |
| 构建线程 | `num_threads_building()` / `set_num_threads_building(n)` | `4` | 构建索引的线程数。 |
| 块大小上限 | `block_size_limit()` / `set_block_size_limit(bytes)` | `128 MB` | 每个分配块的最大字节数（必须 > 2 MB）。 |
| Direct-IO 对齐 | `direct_IO_object_align_bit()` / `set_direct_IO_object_align_bit(bits)` | `9` | Direct-IO 对象对齐，以位为单位（< 21）。 |
| Logger | `logger()` / `set_logger(Logger*)` | `nullptr` | 当前 [`Logger`](#logger)；设置成功返回 `true`。 |

## `Logger`

声明于 `vsag/logger.h`。一个抽象的日志汇。实现它并通过 `Options::set_logger` 注册，即可把 VSAG 的日志
输出路由到你应用的日志系统。

内置 logger 默认使用 `info`。在内置 logger 创建前设置 `VSAG_LOG_LEVEL`，可选择 `trace`、
`debug`、`info`、`warn`/`warning`、`error`、`critical` 或 `off`。无效值会被忽略，并保留默认等级。
显式调用 `SetLevel` 仍会覆盖从环境变量得到的等级。

```cpp
class Logger {
public:
    enum Level : int {
        kTRACE = 0, kDEBUG = 1, kINFO = 2, kWARN = 3, kERR = 4, kCRITICAL = 5, kOFF = 6, kN_LEVELS
    };

    virtual void SetLevel(Level log_level) = 0;
    virtual void Trace(const std::string& msg) = 0;
    virtual void Debug(const std::string& msg) = 0;
    virtual void Info(const std::string& msg) = 0;
    virtual void Warn(const std::string& msg) = 0;
    virtual void Error(const std::string& msg) = 0;
    virtual void Critical(const std::string& msg) = 0;
};
```

| 成员 | 说明 |
|------|------|
| `SetLevel(level)` | 仅发出等级不低于 `level` 的消息。`kOFF` 关闭日志。 |
| `Trace` / `Debug` / `Info` / `Warn` / `Error` / `Critical` | 以相应严重级别发出一条消息。 |

见 `examples/cpp/202_custom_logger.cpp`。

## 参见

- [Factory 与 Engine](factory_engine.md) —— `Engine` 如何消费 `Resource`。
- [Index](index_class.md) —— 取回与搜索方法上的逐次调用 allocator。
- [搜索请求与过滤器](search.md#资源与迭代器字段) —— 逐次搜索 allocator 字段。
