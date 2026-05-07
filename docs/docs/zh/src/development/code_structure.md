# 代码目录结构

VSAG 项目代码处于快速迭代中，目录组织并不完美，这里仅对当前目录的功能划分做简要介绍。

## 项目结构

- `.circleci/`：CircleCI 配置文件；
- `.github/`：GitHub 配置文件，包括 CI、Issue 模版、代码 Owner 等；
- `cmake/`：CMake 工具函数，例如检测编译平台的指令集支持；
- `docker/`：构建 CI 的 Dockerfile 以及用于二进制分发的 Dockerfile；
- `docs/`：设计文档、用户文档（含本站点源）和博客文章；
- `examples/`：C++、Python、TypeScript 的示例代码；
- `extern/`：第三方库，以 CMake 的方式从 GitHub 下载和集成；
- `include/`：公开头文件，对外稳定 API 都位于此目录；
- `mockimpl/`：接口的 Mock 实现，可以用于简单的接口测试；
- `python/`：pyvsag 打包和安装工具；
- `python_bindings/`：基于 pybind11 的 Python 绑定实现；
- `typescript/`：Node.js / TypeScript 绑定及对应 npm 包源代码；
- `scripts/`：一些有用的工具脚本，例如安装依赖、计算代码覆盖率等；
- `src/`：核心源代码和单元测试（`*_test.cpp`）；
- `tests/`：功能测试用例；
- `tools/`：相关工具，包括索引性能测试和兼容性检查工具。

## 核心源代码

- `src/*.cpp`：各种公共功能代码实现，包括内存分配器、线程池等；
- `src/algorithm/`：索引算法目录；
- `src/data_cell/`：data cell 是数据的逻辑单元，索引算法依赖于 data cell；
- `src/impl/`：一些功能和算子的实现，例如图结构增强、k-means 聚类等；
- `src/index/`：索引层实现，和 algorithm 目录相互配合；
- `src/io/`：数据 IO 实现，包括基于内存访问数据和基于磁盘访问数据的方法；
- `src/quantization/`：量化方法，当前支持 SQ4/SQ8、PQ、RaBitQ 等量化方式；
- `src/simd/`：指令集加速模块，根据运行平台自动选择最快的距离计算方法；
- `src/utils/`：工具函数目录。
