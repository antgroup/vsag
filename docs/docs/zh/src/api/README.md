# API 参考

本章是 VSAG **公有 C++ API** 的参考手册，即安装在 `include/vsag/` 下的头文件。它按职责分组，记录应用
程序需要链接的类、结构体、枚举和自由函数。已安装的头文件始终是权威来源；本章的页面负责解释设计意图、
所有权，以及各部分之间如何协作。

> 想了解如何*配置*索引（JSON `index_param` / 搜索键）？相关内容请见
> [索引参数](../resources/index_parameters.md) 与各 [索引页面](../indexes/README.md)。本章覆盖的是
> *代码*层面的接口（类型与方法），而不是 JSON 配置模式。

## 头文件与命名空间

单个总入口头文件即可引入全部公有 API，所有符号都位于 `vsag` 命名空间中：

```cpp
#include <vsag/vsag.h>   // 引入 factory.h、index.h、dataset.h、engine.h ...

int main() {
    vsag::init();                       // 进程级一次性初始化
    std::string ver = vsag::version();  // 由 git 版本号派生的版本字符串
}
```

| 自由函数 | 头文件 | 说明 |
|----------|--------|------|
| `bool vsag::init()` | `vsag/vsag.h` | 初始化库。在调用其他 API 前调用一次。总是返回 `true`。 |
| `std::string vsag::version()` | `vsag/vsag.h` | 返回由 git 版本号派生的构建版本。 |

## 错误处理模型

几乎所有可能失败的调用都返回 `tl::expected<T, Error>`（一个 `std::expected` 风格的类型，定义在
`vsag/expected.hpp`），而不是抛出异常。少数遗留的统计访问器在不支持时仍会抛出 `std::runtime_error`；
这些会在 [Index](index_class.md) 页面明确标注。

```cpp
auto result = vsag::Factory::CreateIndex("hgraph", params);
if (not result.has_value()) {
    const vsag::Error& err = result.error();
    std::cerr << "create failed: " << static_cast<int>(err.type) << " " << err.message << "\n";
    return;
}
std::shared_ptr<vsag::Index> index = result.value();
```

`Error` 携带一个机器可读的 `type` 和一段人类可读的 `message`：

```cpp
struct Error {
    ErrorType type;
    std::string message;
};
```

### `ErrorType`

定义在 `vsag/errors.h`。取值从 `1` 开始（`0` 保留）。

| 类别 | 取值 | 含义 |
|------|------|------|
| 通用 | `UNKNOWN_ERROR` | 未知错误。 |
| 通用 | `INTERNAL_ERROR` | 算法内部错误。 |
| 通用 | `INVALID_ARGUMENT` | 参数非法。 |
| 行为 | `WRONG_STATUS` | 索引处于不允许该调用的状态。 |
| 行为 | `BUILD_TWICE` | 索引已构建，无法再次构建。 |
| 行为 | `INDEX_NOT_EMPTY` | 在非空索引上执行反序列化。 |
| 行为 | `UNSUPPORTED_INDEX` | 请求了不存在的索引类型。 |
| 行为 | `UNSUPPORTED_INDEX_OPERATION` | 该索引未实现所调用的方法。 |
| 行为 | `DIMENSION_NOT_EQUAL` | 请求维度与索引维度不一致。 |
| 行为 | `INDEX_EMPTY` | 索引为空，无法搜索或序列化。 |
| 运行时 | `NO_ENOUGH_MEMORY` | 内存分配失败。 |
| 运行时 | `READ_ERROR` | 从二进制读取失败。 |
| 运行时 | `MISSING_FILE` | 缺少必需的文件（如 DiskANN 反序列化）。 |
| 运行时 | `INVALID_BINARY` | 序列化的二进制内容非法。 |

由于大多数索引方法都是 `virtual`，其默认实现返回 `UNSUPPORTED_INDEX_OPERATION`，因此“不支持”是正常
且预期的结果：它表示具体索引未实现该可选能力。可用
[`Index::CheckFeature`](index_class.md#checkfeature) 提前探测支持情况。

## 头文件映射

| 头文件 | 主要符号 | 参考页面 |
|--------|----------|----------|
| `factory.h`、`engine.h`、`vsag.h` | `Factory`、`Engine`、`init`、`version` | [Factory 与 Engine](factory_engine.md) |
| `index.h` | `Index`、`IndexType`、`RemoveMode`、`MergeUnit` | [Index](index_class.md) |
| `dataset.h` | `Dataset`、`SparseVector`、`MultiVector` | [Dataset](dataset.md) |
| `search_request.h`、`filter.h`、`bitset.h`、`search_param.h`、`iterator_context.h` | `SearchRequest`、`Filter`、`Bitset` | [搜索请求与过滤器](search.md) |
| `binaryset.h`、`readerset.h` | `BinarySet`、`Binary`、`Reader`、`ReaderSet` | [序列化类型](serialization.md) |
| `resource.h`、`allocator.h`、`thread_pool.h`、`options.h`、`logger.h` | `Resource`、`Allocator`、`ThreadPool`、`Options`、`Logger` | [资源管理](resource.md) |
| `attribute.h`、`index_features.h`、`index_detail_info.h`、`utils.h`、`constants.h` | `Attribute`、`IndexFeature`、`IndexDetailInfo` | [辅助类型](types.md) |

## 本章内容

- [Factory 与 Engine](factory_engine.md) —— 创建索引和 reader；用 `Engine` 持有资源。
- [Index](index_class.md) —— 核心索引接口：构建、搜索、更新、序列化、自省。
- [Dataset](dataset.md) —— 用于承载向量、id 和元数据的 builder 模式容器。
- [搜索请求与过滤器](search.md) —— `SearchRequest`、`Filter`、`Bitset`、迭代上下文。
- [序列化类型](serialization.md) —— `BinarySet` / `Binary` 与 `Reader` / `ReaderSet`。
- [资源管理](resource.md) —— allocator、线程池、engine 资源、options、logger。
- [辅助类型](types.md) —— 属性、能力标志、索引细节信息与工具函数。
