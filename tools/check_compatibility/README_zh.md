# VSAG 索引兼容性检查工具 (Index Compatibility Check Tool)

[\[EN\]](README.md)

`check_compatibility` 是一个命令行工具，旨在验证当前版本的 VSAG 库是否能够正确加载并使用由旧版本库生成的索引文件。这对于确保库的向后兼容性至关重要，常用于持续集成 (CI) 流程中，以防止破坏性更改。

## 工作原理

该工具执行以下一系列操作来验证兼容性：

1.  接收一个格式为 `<tag>_<algo_name>` 的字符串作为输入参数。
    -   `<tag>`: 通常是旧版本的标签或标识符（例如 `v1.0.0`）。
    -   `<algo_name>`: 索引算法的名称（例如 `hnsw`）。
2.  根据输入参数，在 `/tmp/` 目录下查找预先准备好的三个文件：
    -   一个索引文件 (`.index`)
    -   一个构建配置文件 (`_build.json`)
    -   一个搜索配置文件 (`_search.json`)
3.  使用**当前**编译的 VSAG 库版本和 `_build.json` 中的参数，创建一个新的索引实例。
4.  尝试将旧的 `.index` 文件**反序列化**（加载）到这个新创建的实例中。
5.  如果加载成功，会从 `/tmp/random_512d_10K.bin` 加载一个小型测试数据集，并执行 KNN 搜索进行健全性检查，以确保索引不仅结构兼容，而且功能正常。
6.  如果整个过程（加载和搜索验证）都成功，则在标准输出打印 `success` 信息；否则，打印 `failed` 信息并退出。

## 环境准备（Prerequisites）

在运行此工具之前，您必须确保以下文件已存在于 `/tmp/` 目录中。这些文件通常由一个较早版本的测试脚本生成。

假设您的输入参数是 `v1.0.0_hnsw`，则需要准备：

1.  **索引文件**: `/tmp/v1.0.0_hnsw.index`
    -   这是由 `v1.0.0` 版本的 VSAG 库生成的 `hnsw` 索引文件。
2.  **构建配置文件**: `/tmp/v1.0.0_hnsw_build.json`
    -   包含生成上述索引时使用的构建参数（JSON 格式）。
3.  **搜索配置文件**: `/tmp/v1.0.0_hnsw_search.json`
    -   包含用于健全性检查的搜索参数（JSON 格式）。
4.  **测试数据**: `/tmp/random_512d_10K.bin`
    -   一个二进制文件，包含用于搜索验证的向量数据。

## 使用方法

编译 `check_compatibility` 可执行文件后，通过命令行运行它，并提供目标标识符作为唯一参数。

```bash
# 编译工具 (假设在 build 目录下)
# make check_compatibility

# 运行兼容性检查
./check_compatibility <tag>_<algo_name>
```

#### 示例

检查当前代码是否兼容由 `v1.0.0` 版本生成的 `hnsw` 索引：

```bash
./check_compatibility v1.0.0_hnsw
```

## 输出

-   **成功**: 如果索引加载和搜索验证均通过，工具将输出：
    ```
    v1.0.0_hnsw success
    ```
-   **失败**: 如果任何步骤失败，工具将在标准错误流输出：
    ```
    v1.0.0_hnsw failed
    ```

## License

该工具根据 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 授权。
