# 新索引接入检查清单

新增 VSAG 索引实现时使用这份检查清单。第一版应保持范围可控：先让索引能通过公开
factory 创建，支持它声明的生命周期方法；只有在行为已经实现并经过测试后，再开启对应的
feature flag。

## 必做项

- [ ] 选择公开索引名称和类型。
  - 如果新索引需要公开名称常量，请在 `include/vsag/constants.h` 或
    `src/inner_string_params.h` 中添加。
  - 当调用方需要通过 `Index::GetIndexType()` 区分该索引时，在
    `include/vsag/index.h` 中添加 `IndexType` 枚举值。
  - 保持公开名称稳定。`src/factory/index_registry.cpp` 会在查找前把 factory 名称
    归一化为小写。

- [ ] 在公开 `Index` API 背后实现索引。
  - 对新的内存索引，优先沿用 `src/index/index_impl.h` 中现有的 `IndexImpl<T>` 模式：
    在 `src/algorithm/<name>/` 下实现一个 `InnerIndexInterface` 子类 `T`。
  - 实现 `static CheckAndMappingExternalParam(const JsonType&, const IndexCommonParam&)`，
    让 `IndexImpl<T>` 能校验外部 JSON 并构造内部参数对象。
  - 按内部索引契约实现 `GetName()`、`GetIndexType()`、`GetNumElements()`、`Add()`、
    `KnnSearch()`、`Serialize(StreamWriter&)` 和 `Deserialize(StreamReader&)`；索引支持
    `Build()` 时再实现它。`InnerIndexInterface::Add()` 是纯虚函数，因此每个子类都必须重写，
    即使索引只支持 `Build()`；不支持时应抛出 `UNSUPPORTED_INDEX_OPERATION`，且不要开启对应
    的 feature flag。
  - 对不支持的操作保留基类默认行为，不要把未实现能力声明为已支持。

- [ ] 接入 factory 和 engine 创建路径。
  - 在 `src/factory/index_creators.cpp` 中添加 creator。
  - 在 `register_all_index_creators()` 中注册。
  - 对共享字段使用 `src/index_common_param.cpp` 中的
    `IndexCommonParam::CheckAndCreate()`：`dtype`、`metric_type`、`dim`、可选 `repr`、
    可选 `extra_info_size`、allocator、thread pool，以及旧序列化格式兼容信息。
  - 在 `src/factory/factory_test.cpp` 或实现附近的专项测试中，覆盖可接受名称、非法参数
    和不支持的参数形态。

- [ ] 添加构建系统接入。
  - 添加 `src/algorithm/<name>/CMakeLists.txt`，并从 `src/algorithm/CMakeLists.txt` 引入。
  - 将新源码加入最近的现有 target，不要创建平行构建路径。
  - 文件后缀保持 `.cpp`；除非变更本身涉及第三方依赖，否则不要修改 `extern/`。

- [ ] 定义并校验索引参数。
  - 当索引有自己的 schema 时，把实现参数放在 `<name>_parameter.{h,cpp}` 中。
  - 为序列化/重建参数校验实现 JSON 解析、`ToJson()` 和 `CheckCompatibility()`。
  - 对非法维度、metric/data type 组合、缺失的必需配置块和未知模式，通过已有的
    `CHECK_ARGUMENT` / `VsagException` 流程返回 `ErrorType::INVALID_ARGUMENT`。
  - 如果参数对用户可见，更新 `docs/docs/{en,zh}/src/resources/index_parameters.md` 和对应
    索引文档。

- [ ] 明确实现生命周期行为。
  - 决定索引是否支持 `Train()`、`Build()`、`ContinueBuild()`、build 后 `Add()`、空索引
    `Add()`。
  - 决定是否支持 `Remove()`、`UpdateId()`、`UpdateVector()`、`UpdateAttribute()` 和
    `UpdateExtraInfo()`。
  - 对每一种支持的 mutation，测试空数据集、重复 ID（如适用）、缺失 ID、不可变索引行为以及
    mutation 后的搜索正确性。
  - 保持 `InitFeatures()` 与已经实现的操作一致。

- [ ] 实现搜索行为和结果打包。
  - 支持该索引要求的公开 `KnnSearch()` 重载，包括声明支持时的 `BitsetPtr`、
    `std::function<bool(int64_t)>` 和 `FilterPtr` 过滤路径。
  - 如果索引支持新的 request 路径，实现 `SearchWithRequest()`。
  - 一致地返回 `Dataset` 字段：ID、距离、`num_elements`、结果维度和可选结果统计信息。
  - 按 HGraph 等现有索引使用的嵌套索引名约定解析搜索参数。

- [ ] 保持序列化兼容。
  - 同时实现 `Serialize(StreamWriter&)` 和 `Deserialize(StreamReader&)`；基类
    `InnerIndexInterface` 会把它们适配到 `BinarySet`、`ReaderSet` 和 stream。
  - 保存足够的元数据以拒绝不兼容二进制，包括参数兼容性，以及存在 extra info 时的
    `extra_info_size`。
  - 当索引同时支持 `BinarySet` 和 `ReaderSet` 时，添加两条路径的 round-trip 测试。
  - 如果既有索引的二进制格式发生变化，更新兼容性测试并记录迁移路径。

- [ ] 在声明 feature 前补齐测试。
  - 单元测试应覆盖参数解析、build/add/search、序列化、feature flag、已实现的内存估算以及
    错误路径。
  - `tests/` 下的功能测试应覆盖用户能通过 `Factory::CreateIndex()` 触达的公开 API 行为。
  - 保持 `src/` 和 `include/` 的 C++ 单元测试覆盖率不低于项目阈值。

## 可选适配点

仅当新索引确实实现对应行为时才添加这些能力。实现后，在 `InitFeatures()` 中开启匹配的
`IndexFeature`，并添加专项测试。

- [ ] Extra info（`extra_info` / `extrainfo`）。
  - 通过 `IndexCommonParam` 解析 `extra_info_size`。
  - 保存来自 `Dataset::GetExtraInfos()` 的定长逐向量 payload，并在 `Build()`、`Add()` 和
    `UpdateExtraInfo()` 中校验 `Dataset::GetExtraInfoSize()`。
  - 实现 `GetExtraInfoByIds()`，并在支持该 feature 时填充搜索结果中的 extra info。
  - 如果索引支持 extra-info 过滤，记录并测试会切换到 `Filter::CheckValid(const char*)` 的
    搜索参数。
  - 参考 `docs/docs/zh/src/advanced/extra_info.md` 和
    `examples/cpp/320_feature_extra_info.cpp`。

- [ ] 统计与分析。
  - 为能帮助运维理解索引的静态结构数据实现 `GetStats()`。
  - 只有在需要基于 query 分析时，才实现 `AnalyzeIndexBySearch(const SearchRequest&)`。
  - 当 search-time 指标有价值时，通过 `Dataset::Statistics()` 附带结果统计。
  - 保持工具输出兼容 `tools/analyze_index` 和
    `docs/docs/zh/src/resources/analyze_index.md`。

- [ ] 范围搜索。
  - 必须重写 `InnerIndexInterface` 要求的纯虚主重载
    `RangeSearch(..., const FilterPtr&, ...)`，即使算法不支持范围搜索；不支持时应抛出
    `UNSUPPORTED_INDEX_OPERATION`，且不要开启对应的 feature flag。
  - 只有当算法能遵守 radius 语义和 `limited_size` 时，才实现其他 `RangeSearch()` 重载。
  - 测试不限量、限量、带过滤和空结果场景。
  - 参考 `docs/docs/zh/src/advanced/range_search.md`。

- [ ] 过滤器与属性。
  - 只有当各路径已接入搜索时，才支持 `BitsetPtr`、`std::function<bool(int64_t)>` 或
    `FilterPtr`。
  - 如果支持属性过滤，实现属性存储/更新路径，并记录可接受的属性 schema。
  - 测试 bitset invalidation 语义和 `Filter::CheckValid()` keep 语义之间的差异。

- [ ] Allocator、resource 和线程集成。
  - 使用 `IndexCommonParam::allocator_` 或派生的 allocator-aware 组件分配长期结构。
  - 当 build/search 工作并行化时，使用 `Resource` thread pool。
  - 确认自定义 allocator 和自定义 thread-pool 示例仍准确描述行为。
  - 只有在 add/search/delete/update 交互已测试后，才标记并发相关 feature。

- [ ] 内存和自省 API。
  - 当索引能报告有意义数字时，实现 `EstimateMemory()`、`EstimateBuildMemory()`、
    `GetMemoryUsage()` 和 `GetMemoryUsageDetail()`。
  - 只有在底层存储支持时，才实现 `GetMinAndMaxId()`、`CheckIdExist()`、`ExportIDs()`、
    `GetVectorByIds()`、`GetDataByIds()`、`GetIndexDetailInfos()` 或
    `GetDetailDataByName()`。

- [ ] 模型导出、clone、merge、tune、feedback 和 cache 导入/导出。
  - 当索引可以在不错误共享可变存储的情况下复制时，实现 `Clone()` 和 `ExportModel()`。
  - 只有在参数兼容性、ID 重映射和删除语义清晰时，才实现 `Merge()`。
  - 只有带有明确参数解析和测试时，才实现 `Tune()`、`Feedback()`、`ExportCache()` 和
    `ImportCache()`。

- [ ] 绑定、示例、benchmark 和文档。
  - Python 绑定通常只在公开 API surface 变化时需要更新；当前 `pyvsag` 用户通过名称和 JSON
    参数创建索引。
  - 当索引引入新的用户工作流时，在 `examples/cpp/` 下添加 C++ 示例。
  - 如果行为可从 `pyvsag` 触达，在 `tests/python/` 下添加 Python 示例/测试。
  - 当 reviewer 需要可重复性能数据时，在 `benchs/` 下添加 benchmark YAML。
  - 对用户可见的索引或参数，在 `docs/docs/{en,zh}/src/` 下添加英文和中文网站文档。

## Review Checklist

- [ ] `Factory::CreateIndex()` 和 `Engine::CreateIndex()` 能按文档名称创建索引。
- [ ] `CheckFeature()` 只对已实现且已测试的行为返回 true。
- [ ] 不支持的操作通过现有 wrapper 返回 `UNSUPPORTED_INDEX_OPERATION`。
- [ ] 序列化 round trip 保留索引声明支持的 ID、向量或压缩码、参数、删除状态、属性和
  extra info。
- [ ] 每个支持的生命周期转换后，搜索结果仍然有效。
- [ ] 文档列出用户可见参数、支持的 metric/data type 和不支持的操作。
- [ ] 已运行实际验证：变更代码对应的单元/功能测试，以及文档-only 变更需要的格式检查或
  `git diff --check`。
