# C++ 编码指南

VSAG 以 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html) 为基础，并针对
代码格式、命名和静态检查制定了项目规则。仓库根目录下的配置文件是这些规则的准确定义：

- [`.clang-format`](https://github.com/antgroup/vsag/blob/main/.clang-format) 定义自动格式化规则；
- [`.clang-tidy`](https://github.com/antgroup/vsag/blob/main/.clang-tidy) 定义命名与静态检查规则。

VSAG 严格要求使用 **clang-format 15** 和 **clang-tidy 15**。其他版本可能产生不同的格式化
结果或诊断信息，无法通过 CI。

## 代码组织

- 公开 API 放在 `include/vsag/`，实现代码放在 `src/`；
- C++ 源文件使用 `.cpp` 后缀，不使用 `.cc`；
- 除非文件有明确的特殊要求，代码应放在 `vsag` 命名空间中；
- 除非明确计划进行破坏性变更，否则应保持 API 兼容；
- 修改公开 API 时，应添加或更新 Doxygen 注释；
- 新代码优先使用 `uint64_t` 而不是 `size_t`，避免 macOS 兼容性问题；
- 除非修改目标明确涉及第三方依赖，否则不要修改 `extern/`；
- 所有提交的文本文件都应以换行符结尾。

## 格式化规则

格式化配置以 Google 风格为基础，并应用以下 VSAG 规则：

| 规则 | VSAG 风格 |
| --- | --- |
| 缩进 | 4 个空格 |
| C++ 行宽 | 最多 100 列 |
| 访问控制符 | 相对成员代码减少一级缩进 |
| 返回类型 | 与函数名分行书写 |
| 简短代码块、函数和 `if` 语句 | 不压缩到单行 |
| 指针和引用 | `*`、`&` 与类型结合 |
| 头文件引用 | 自动排序 |
| 换行后的参数和实参 | 每行一个，不进行 bin-pack |
| 行尾注释 | 连续成组时自动对齐 |
| 已有注释文本 | 不自动重新换行 |

例如，clang-format 15 会将类声明格式化为：

```cpp
class Example {
public:
    Result<DatasetPtr>
    Build(const DatasetPtr& base, const std::string& parameters);

private:
    void
    reset_state();

    DatasetPtr dataset_;
};
```

100 列限制只适用于 C++ 源文件和头文件，不要求 Markdown 或其他非 C++ 文本遵循这一行宽。

## 命名规则

`.clang-tidy` 中的 `readability-identifier-naming` 会检查以下命名：

| 对象 | 风格 | 示例 |
| --- | --- | --- |
| 命名空间 | `lower_case` | `vsag::storage` |
| 类 | `CamelCase` | `HGraphIndex` |
| 结构体 | `CamelCase` | `SearchResult` |
| 类型模板参数 | `CamelCase` | `DataType` |
| 值模板参数 | `lower_case` | `max_degree` |
| 自由函数或 C 风格函数 | `lower_case` | `normalize_vector` |
| 公有方法 | `CamelCase` | `Build` |
| 私有方法 | `lower_case` | `reset_state` |
| 变量 | `lower_case` | `search_result` |
| 私有或受保护的数据成员 | `lower_case_` | `search_result_` |
| 宏定义 | `UPPER_CASE` | `CHECK_ARGUMENT` |
| 枚举值 | `UPPER_CASE` | `ROUTING` |
| 全局常量 | `UPPER_CASE` | `DEFAULT_RESIZE_INCREASE_COUNT_BIT` |

`make lint` 还会运行 `scripts/linters/check-struct-names.py`，检查所有已被 Git 跟踪的项目 C/C++
文件中的具名结构体，包括 clang-tidy 常规检查范围之外的文件。因此，新建的具名结构体都必须使用
`CamelCase`。

对于没有项目专用规则的对象，请遵循 Google C++ 风格以及所在模块的现有约定。

## 选择 `class` 还是 `struct`

当类型是被动、值语义的数据载体时，使用 `struct`：

- 主要用途是将一组相关数据组织在一起；
- 数据成员有意保持公开，调用方可以彼此独立地修改它们；
- 不需要保护对象不变量，也不需要隐藏内部表示；
- 不负责需要受控访问或生命周期管理的资源。

例如 `Binary`、`SparseVector` 和 `MultiVector`。`struct` 仍然可以包含成员默认值、构造函数或简单的
辅助函数。是否存在成员函数并不是选择 `class` 的决定因素，关键在于该类型是否仍是透明的数据值。

当类型表示行为或必须控制自身状态时，使用 `class`：

- 通过公开方法维护对象不变量；
- 使用私有或受保护成员隐藏实现细节；
- 持有或管理需要控制生命周期的资源；
- 提供行为接口或多态接口，尤其是包含虚函数的接口。

例如 `BinarySet`、`Dataset`、`Filter` 和 `Index`。新建接口和多态基类时，优先使用 `class`。

无法确定时，可以判断调用方是否应当以任意顺序直接修改每一个数据成员。如果允许，使用 `struct`；如果
类型必须校验、协调或限制这些修改，使用 `class`。不要仅仅为了省略 `public:` 而使用 `struct`，也不要
为了支持聚合初始化而暴露可变的实现细节。

类和结构体都使用 `CamelCase`。除非有意进行 API 变更，否则应保留公开类型当前使用的形式，不要把
`class` 与 `struct` 之间的转换作为纯风格清理。

## 静态检查

所有已启用的 clang-tidy 诊断都按错误处理。当前配置启用了以下检查组：

- `bugprone-*`：发现可能的逻辑缺陷和可疑写法；
- `clang-analyzer-*`：进行路径敏感的正确性和生命周期分析；
- `clang-diagnostic-*`：通过 clang-tidy 报告编译器诊断；
- `modernize-*`：检查项目支持的现代 C++ 写法；
- `openmp-*`：检查 OpenMP 使用的正确性；
- `performance-*`：发现不必要的复制和低效操作；
- `readability-*`：检查可维护性和命名一致性。

部分检查会与 VSAG 的底层、高性能代码冲突，或者在当前代码库中产生过多噪声，因此被明确禁用：

| 禁用的检查 | 原因 |
| --- | --- |
| `clang-analyzer-deadcode.DeadStores` | “写入但未读取”的诊断噪声过多 |
| `clang-diagnostic-deprecated-builtins` | 工具链或平台兼容仍需要部分已弃用的 builtin |
| `bugprone-easily-swappable-parameters` | 误报过多 |
| `modernize-use-trailing-return-type` | 与项目的返回类型风格冲突 |
| `modernize-avoid-c-arrays` | SIMD 和其他底层代码会有意使用 C 数组 |
| `readability-identifier-length` | 上下文清晰时允许使用短名称 |
| `readability-magic-numbers` | 数值密集的向量索引代码会产生过多噪声 |
| `readability-function-cognitive-complexity` | 在解决现有问题前暂不启用 |
| `readability-redundant-access-specifiers` | 允许重复访问控制符来划分方法组 |

禁用检查并不意味着可以忽略它所关注的问题。在能够提升可读性的情况下，仍应使用含义明确的参数名和
具名常量。

### 抑制诊断

应优先修复诊断指出的问题。如果诊断属于误报，或者相关写法是兼容性或性能所必需的，应使用范围最小的
抑制，并说明这样做安全的原因：

```cpp
// NOLINTNEXTLINE(performance-no-int-to-ptr): platform API 将该句柄存储为整数。
auto* handle = reinterpret_cast<Handle*>(raw_handle);
```

在 `NOLINT`、`NOLINTNEXTLINE` 或范围严格受控的 `NOLINTBEGIN`/`NOLINTEND` 中写出具体检查名。
除非无法采用更小的范围，否则不要使用不带检查名的 `NOLINT` 或对整个文件禁用检查。

## 运行检查

首先安装指定版本的工具：

```bash
# Ubuntu 或 Debian
sudo apt-get install clang-format-15 clang-tidy-15 clang-tools-15

# macOS
brew install llvm@15
```

格式化所有项目 C++ 文件并检查修改：

```bash
make fmt
git diff --check
```

clang-tidy 需要 `build-release/` 中的编译数据库，因此应先完成发布模式构建：

```bash
make release
make lint
```

默认 lint 目标以 `src/` 下的生产环境 `.cpp` 文件作为 clang-tidy 入口，并排除 `*_test.cpp`；同时还会
运行覆盖整个仓库的结构体命名检查。测试和其他目录不会在该命令中作为独立的 clang-tidy 入口。

`make fix-lint` 会直接应用 clang-tidy 建议的修改，可能产生较大范围的变更。请在干净或已安全备份的
工作区中使用，检查完整 diff，然后重新运行格式化、静态检查和相关测试：

```bash
make fix-lint
make fmt
make lint
make test
```

## 提交修改前

- 使用 clang-format 15 格式化所有修改过的 C++ 文件；
- 使用 clang-tidy 15 检查修改过的生产环境源码；
- 确认命名符合上述规则，尤其是结构体使用 `CamelCase`；
- 确保每个 `NOLINT` 都指定具体检查，并说明原因；
- 新功能包含测试，bug 修复包含回归测试；
- 保持 C++ 库的单元测试覆盖率不低于 90%；
- 运行 `git diff --check` 并检查完整 diff。
