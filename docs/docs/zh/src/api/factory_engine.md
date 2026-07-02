# Factory 与 Engine

每个 VSAG 工作流都从获取一个 `Index` 开始。这里有两个入口：

- [`Factory`](#factory) —— 创建索引或文件 reader 的最简单方式。它使用默认（或调用方提供）的
  allocator，并在内部管理资源。
- [`Engine`](#engine) —— 显式持有共享资源（allocator + 线程池）。当你希望多个索引共享同一个内存
  allocator / 线程池，或需要对资源生命周期进行确定性控制时，使用它。

本页还介绍进程级的[库初始化](#库初始化)，以及用于参数生成与校验的[顶层辅助函数](#顶层辅助函数)。

## 库初始化

```cpp
#include <vsag/vsag.h>

vsag::init();                       // 在调用其他任何 API 之前调用一次
std::string ver = vsag::version();  // 例如由 git 派生的构建版本字符串
```

| 函数 | 签名 | 说明 |
|------|------|------|
| `vsag::init` | `bool init()` | 进程级一次性初始化。返回 `true`。 |
| `vsag::version` | `std::string version()` | 由 git 版本号派生的构建版本。 |

## Factory

声明于 `vsag/factory.h`。`Factory` 是一个无状态的工具类，只有静态方法，无法被实例化。

### `CreateIndex`

```cpp
static tl::expected<std::shared_ptr<Index>, Error>
CreateIndex(const std::string& name,
            const std::string& parameters,
            Allocator* allocator = nullptr);
```

创建给定类型的索引。

| 参数 | 说明 |
|------|------|
| `name` | 索引类型名，例如 `"hgraph"`、`"ivf"`、`"diskann"`、`"brute_force"`、`"sindi"`、`"pyramid"`。 |
| `parameters` | 描述索引配置的 JSON 字符串（dtype、dim、metric，以及各索引特有的键）。见 [索引参数](../resources/index_parameters.md)。 |
| `allocator` | 可选的自定义 [`Allocator`](resource.md#allocator)。为 `nullptr` 时，VSAG 使用内置的默认 allocator。调用方必须在返回索引的整个生命周期内保持 allocator 有效。 |

成功时返回 `std::shared_ptr<Index>`，失败时返回 `Error`（`name` 未知时通常是 `UNSUPPORTED_INDEX`，
`parameters` 非法时是 `INVALID_ARGUMENT`）。

```cpp
auto index = vsag::Factory::CreateIndex("hgraph", R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": { "base_quantization_type": "sq8" }
})");
if (not index.has_value()) {
    std::cerr << index.error().message << std::endl;
    return;
}
std::shared_ptr<vsag::Index> hgraph = index.value();
```

### `CreateLocalFileReader`

```cpp
static std::shared_ptr<Reader>
CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size);
```

创建一个 [`Reader`](serialization.md#reader)，它从本地文件的 `base_offset` 处开始读取 `size` 字节的
窗口。最常见的用法是构建 [`ReaderSet`](serialization.md#readerset) 以流式反序列化磁盘上的索引。与上面
的方法不同，它返回一个普通的 `std::shared_ptr`（没有可失败的 `Error` 通道）。

## Engine

声明于 `vsag/engine.h`。`Engine` 绑定一个 [`Resource`](resource.md#resource)（allocator + 线程池），
并让你创建共享它的索引。engine 不会接管传入的 `Resource*` 的所有权；其生命周期由你控制。

```cpp
vsag::Resource resource(vsag::Engine::CreateDefaultAllocator().get(), nullptr);
vsag::Engine engine(&resource);

auto index = engine.CreateIndex("hgraph", params);
// ... 使用 index ...

engine.Shutdown();   // 释放 engine 持有的状态；若存在悬挂引用会发出告警
```

### 构造与生命周期

| 成员 | 签名 | 说明 |
|------|------|------|
| 构造函数 | `explicit Engine(Resource* resource)` | 绑定一个外部持有的 `Resource`。该 `Resource` **不**由 engine 管理。 |
| `Shutdown` | `void Shutdown()` | 优雅地拆除 engine 持有的状态。若仍存在对 engine 资源的外部引用，会发出告警，以防悬挂引用。 |

### `CreateIndex`

```cpp
[[nodiscard]] tl::expected<std::shared_ptr<Index>, Error>
CreateIndex(const std::string& name, const std::string& parameters);
```

语义与 [`Factory::CreateIndex`](#createindex) 相同，区别在于索引是基于 engine 的共享 `Resource`
（allocator 与线程池）创建的，而不是每次调用各自的 allocator。

### 静态资源辅助方法

| 成员 | 签名 | 说明 |
|------|------|------|
| `CreateDefaultAllocator` | `static std::shared_ptr<Allocator> CreateDefaultAllocator()` | 创建 VSAG 内置的 allocator。失败时返回空指针 —— 需检查是否为 null。 |
| `CreateThreadPool` | `static tl::expected<std::shared_ptr<ThreadPool>, Error> CreateThreadPool(uint32_t num_threads)` | 创建含 `num_threads` 个工作线程的线程池。数量非法时返回 `Error`。 |

关于 `Resource`、`Allocator` 与 `ThreadPool` 如何协作，见 [资源管理](resource.md)；可运行示例见
`examples/cpp/201_custom_allocator.cpp` / `203_custom_thread_pool.cpp`。

## 顶层辅助函数

这些自由函数（声明于 `vsag/index.h`）帮助你在创建索引前生成并校验配置字符串。它们都返回
`tl::expected<..., Error>`。

### `generate_build_parameters`

```cpp
tl::expected<std::string, Error>
generate_build_parameters(std::string metric_type,
                          int64_t num_elements,
                          int64_t dim,
                          bool use_conjugate_graph = false);
```

*（实验性。）* 根据数据集形状（`metric_type`、`num_elements`、`dim`）生成一份建议的构建参数 JSON
字符串。传入 `use_conjugate_graph = true` 可启用[共轭图增强](../advanced/enhance_graph.md)。

### `estimate_search_time`

```cpp
tl::expected<float, Error>
estimate_search_time(const std::string& index_name,
                     int64_t data_num,
                     int64_t data_dim,
                     const std::string& parameters);
```

估算给定索引类型与配置下的单次查询搜索时间（毫秒）。

### `check_diskann_hnsw_build_parameters` / `check_diskann_hnsw_search_parameters`

```cpp
tl::expected<bool, Error>
check_diskann_hnsw_build_parameters(const std::string& json_string);

tl::expected<bool, Error>
check_diskann_hnsw_search_parameters(const std::string& json_string);
```

分别校验 DiskANN/HNSW 的构建与搜索参数 JSON。成功时值为 `true`；失败时 `Error` 的 message 会说明问题
所在。此类校验的命令行封装见[兼容性检查工具](../resources/check_compatibility.md)。

## 参见

- [Index](index_class.md) —— 索引创建之后你能用它做什么。
- [资源管理](resource.md) —— allocator、线程池与 `Resource` 细节。
- [索引参数](../resources/index_parameters.md) —— `CreateIndex` 接受的 JSON 模式。
