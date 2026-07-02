# Index

`vsag::Index`（声明于 `vsag/index.h`）是本库的核心抽象。每一种具体索引 —— HGraph、IVF、DiskANN、
BruteForce、SINDI、Pyramid 等 —— 都实现这一接口。你从不直接实例化 `Index`，而是通过
[`Factory::CreateIndex`](factory_engine.md#createindex) 或
[`Engine::CreateIndex`](factory_engine.md#createindex-1) 获取，并用 `IndexPtr`
（`std::shared_ptr<Index>`）持有它。

```cpp
using IndexPtr = std::shared_ptr<Index>;
```

## 如何阅读本参考

`Index` 暴露了许多可选能力。基类为几乎每个方法都提供了**默认实现**：

- 当具体索引未实现某方法时，大多数方法返回
  `tl::unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, ...))`。
- 少数统计访问器则会**抛出 `std::runtime_error`**（下文会明确标注）。若在可能不支持它们的索引上调用，
  请用 `try/catch` 包裹。

由于“不支持”是正常且预期的结果，请用 [`CheckFeature`](#checkfeature) 提前探测能力，而不要假设某个方法一定
可用。标注为 *（纯虚函数）* 的方法必须由每种索引实现，调用它们总是安全的。

本页通篇用到的指针/句柄类型：`DatasetPtr`（[Dataset](dataset.md)）、`FilterPtr`
（[Filter](search.md#filter)）、`BitsetPtr`（[Bitset](search.md#bitset)）、`BinarySet` /
`ReaderSet`（[序列化类型](serialization.md)）。

## 枚举与辅助类型

### `IndexType`

```cpp
enum class IndexType {
    HNSW, DISKANN, HGRAPH, IVF, PYRAMID, BRUTEFORCE, SPARSE, SINDI, WARP, LAZY_HGRAPH, SIMQ
};
```

由 [`GetIndexType`](#getindextype) 返回。

### `RemoveMode`

```cpp
enum class RemoveMode {
    MARK_REMOVE = 0,   // 标记删除；不收缩/不修复 —— 快
    FORCE_REMOVE = 1,  // 物理删除并修复图 —— 重
};
```

传入 [`Remove`](#remove)。

### MergeUnit 与 IdMapFunction

```cpp
using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

struct MergeUnit {
    IndexPtr index = nullptr;         // 要合并进来的源子索引
    IdMapFunction id_map_func = nullptr;  // 逐 id 的过滤 + 重映射
};
```

对每个源 id，`id_map_func` 返回 `{keep, new_id}`：`keep == true` 表示将该向量以目标 id `new_id` 纳入。
由 [`Merge`](#merge) 使用。

### `Checkpoint`

```cpp
struct Index::Checkpoint {
    BinarySet data;       // 中间状态
    bool finish = false;  // 构建完成后为 true
};
```

由 [`ContinueBuild`](#continuebuild) 返回，用于驱动增量构建。

### 数据选择标志

用于 [`GetDataByIdsWithFlag`](#getdatabyidswithflag) 的位标志，可通过按位或组合：

| 宏 | 值 | 选取 |
|----|----|------|
| `DATA_FLAG_FLOAT32_VECTOR` | `0x01` | float32 向量 |
| `DATA_FLAG_INT8_VECTOR` | `0x02` | int8 向量 |
| `DATA_FLAG_SPARSE_VECTOR` | `0x04` | 稀疏向量 |
| `DATA_FLAG_EXTRA_INFO` | `0x10` | extra info 数据块 |
| `DATA_FLAG_ATTRIBUTE` | `0x20` | 属性 |
| `DATA_FLAG_ID` | `0x40` | id |

### `WriteFuncType`

```cpp
using OffsetType = uint64_t;
using SizeType = uint64_t;
using WriteFuncType = std::function<void(OffsetType, SizeType, const void*)>;
```

用于流式 [`Serialize`](#serialize) 的落盘回调。每次调用要求你把 `SizeType` 字节（位于给定源指针处）持久化
到输出中逻辑偏移 `OffsetType` 的位置。

## 构建与训练

| 方法 | 签名 | 说明 |
|------|------|------|
| `Build` | `tl::expected<std::vector<int64_t>, Error> Build(const DatasetPtr& base)` | *（纯虚函数）* 从全部向量构建索引。返回插入失败的 id。 |
| `Train` | `tl::expected<void, Error> Train(const DatasetPtr& data)` | 训练索引（如 IVF 聚类中心、量化器）而不插入数据。 |
| `Tune` | `tl::expected<bool, Error> Tune(const std::string& parameters, bool disable_future_tuning = false)` | 应用运行期调优。见 [优化器](../advanced/optimizer.md)。 |
| `ContinueBuild` | `tl::expected<Checkpoint, Error> ContinueBuild(const DatasetPtr& base, const BinarySet& binary_set)` | 为无法增量插入的索引提供动态性；用返回的 [`Checkpoint`](#checkpoint) 驱动。 |
| `Add` | `tl::expected<std::vector<int64_t>, Error> Add(const DatasetPtr& base)` | 向已构建的索引插入新向量。返回插入失败的 id。 |

见 [索引构建与训练](../advanced/build_and_train.md) 与 `examples/cpp/311_feature_train.cpp`。

## 更新与删除

| 方法 | 签名 | 说明 |
|------|------|------|
| `Remove` | `tl::expected<uint32_t, Error> Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE)` | 删除多个 id；返回被删除的数量。 |
| `Remove` | `tl::expected<uint32_t, Error> Remove(int64_t id, RemoveMode mode = RemoveMode::MARK_REMOVE)` | 单 id 便捷重载。 |
| `UpdateId` | `tl::expected<bool, Error> UpdateId(int64_t old_id, int64_t new_id)` | 为一个基础点重新打标签。 |
| `UpdateVector` | `tl::expected<bool, Error> UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false)` | 替换 `id` 对应的向量。`force_update = false` 会执行连通性检查。 |
| `UpdateExtraInfo` | `tl::expected<bool, Error> UpdateExtraInfo(const DatasetPtr& new_base)` | 更新存储的 extra-info 数据块。 |
| `UpdateAttribute` | `tl::expected<void, Error> UpdateAttribute(int64_t id, const AttributeSet& new_attrs)` | 替换 `id` 的属性。 |
| `UpdateAttribute` | `tl::expected<void, Error> UpdateAttribute(int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs)` | 同上，但提供旧属性以便更快地原地更新。 |

见 `examples/cpp/303_feature_remove.cpp`。

## 搜索

推荐的入口是 [`SearchWithRequest`](#searchwithrequest)，它接收单个
[`SearchRequest`](search.md#searchrequest)，其中携带查询、模式、top-k / 半径以及各类过滤器。较旧的
逐参数 `KnnSearch` / `RangeSearch` 重载为兼容性保留。

每次搜索都返回一个 `DatasetPtr`：对 KNN，`num_elements == 1`，`ids` / `distances` 长度为 `k`；对范围
搜索，结果长度即命中数。如何读取结果见 [Dataset](dataset.md)。

### `SearchWithRequest`

```cpp
[[nodiscard]] tl::expected<DatasetPtr, Error>
SearchWithRequest(const SearchRequest& request) const;
```

由 [`SearchRequest`](search.md#searchrequest) 驱动的统一 KNN 或范围搜索。这是新代码首选的 API；它通过
一个结构体即可支持属性过滤、回调过滤、bitset 过滤、逐次搜索 allocator 以及迭代式搜索。

### KnnSearch 重载

```cpp
// (1) bitset 预过滤 —— 纯虚函数
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          BitsetPtr invalid = nullptr) const;

// (2) 回调预过滤 —— 纯虚函数
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const std::function<bool(int64_t)>& filter) const;

// (3) Filter 对象
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const FilterPtr& filter) const;

// (4) Filter + 迭代上下文
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, const std::string& parameters,
          const FilterPtr& filter, IteratorContext*& iter_ctx, bool is_last_search) const;

// (5) SearchParam —— [[deprecated]]，请改用 SearchWithRequest
tl::expected<DatasetPtr, Error>
KnnSearch(const DatasetPtr& query, int64_t k, SearchParam& search_param) const;
```

关于 filter 参数的说明：

- 在重载 (1)/(2) 中，谓词/bitset 标记的是被**过滤掉**的向量。对 `bitset`，`Test(id) == true` 表示该 id
  被排除；对 `std::function` 谓词，返回 `true` 表示该 id 被排除。
- 重载 (3)/(4) 接收 [`Filter`](search.md#filter) 对象，其 `CheckValid(id)` 采用相反约定
  （`true` 表示*保留*）。完整语义见 [带过滤的搜索](../advanced/filtered_search.md) 与
  `examples/cpp/301_feature_filter.cpp`。
- 重载 (4) 支撑[迭代式搜索](../advanced/iterator_search.md)；跨调用传入同一个 `iter_ctx`，并在最后一次
  调用时设置 `is_last_search`。

### `RangeSearch` 重载

```cpp
// (1) 普通 —— 纯虚函数
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            int64_t limited_size = -1) const;

// (2) bitset 预过滤 —— 纯虚函数
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            BitsetPtr invalid, int64_t limited_size = -1) const;

// (3) 回调预过滤 —— 纯虚函数
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            const std::function<bool(int64_t)>& filter, int64_t limited_size = -1) const;

// (4) Filter 对象
tl::expected<DatasetPtr, Error>
RangeSearch(const DatasetPtr& query, float radius, const std::string& parameters,
            const FilterPtr& filter, int64_t limited_size = -1) const;
```

`radius` 限定距离上界；`limited_size` 限制结果数量（`<= 0` 表示不限，`0` 为错误）。见
[范围搜索](../advanced/range_search.md) 与 `examples/cpp/302_feature_range_search.cpp`。

## 按 id 计算距离

| 方法 | 签名 | 说明 |
|------|------|------|
| `CalcDistanceById` | `tl::expected<float, Error> CalcDistanceById(const float* vector, int64_t id, bool calculate_precise_distance = true) const` | 稠密查询到已存向量 `id` 的距离。 |
| `CalcDistanceById` | `tl::expected<float, Error> CalcDistanceById(const DatasetPtr& vector, int64_t id, bool calculate_precise_distance = true) const` | 同上，接收 `DatasetPtr`（适用于 SINDI 等稀疏索引）。 |
| `CalDistanceById` | `tl::expected<DatasetPtr, Error> CalDistanceById(const float* query, const int64_t* ids, int64_t count, bool calculate_precise_distance = true) const` | 批量版本；结果中的 `-1` 表示无效距离。 |
| `CalDistanceById` | `tl::expected<DatasetPtr, Error> CalDistanceById(const DatasetPtr& query, const int64_t* ids, int64_t count, bool calculate_precise_distance = true) const` | 接收 `DatasetPtr` 查询的批量版本。 |

`calculate_precise_distance = true` 时可能会加载全精度向量（可能来自磁盘）而非量化编码。见
[按 ID 计算距离](../advanced/calc_distance_by_id.md) 与
`examples/cpp/306_feature_calculate_distance_by_id.cpp`。

## 共轭图增强

| 方法 | 签名 | 说明 |
|------|------|------|
| `Pretrain` | `tl::expected<uint32_t, Error> Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters)` | 通过检索生成的查询来增强选定的基础向量。返回成功插入数。 |
| `Feedback` | `tl::expected<uint32_t, Error> Feedback(const DatasetPtr& query, int64_t k, const std::string& parameters, int64_t global_optimum_tag_id = INT64_MAX)` | 把已知最优解反馈到共轭图中。 |

见 [图索引增强](../advanced/enhance_graph.md)。

## 数据获取

| 方法 | 签名 | 说明 |
|------|------|------|
| `GetMinAndMaxId` | `tl::expected<std::pair<int64_t, int64_t>, Error> GetMinAndMaxId() const` | 索引中最小与最大的 id。 |
| `GetExtraInfoByIds` | `tl::expected<void, Error> GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const` | 把 `ids` 的 extra-info 数据块拷贝到调用方提供的缓冲区。 |
| `GetRawVectorByIds` | `tl::expected<DatasetPtr, Error> GetRawVectorByIds(const int64_t* ids, int64_t count, Allocator* specified_allocator = nullptr) const` | 返回已存向量。其值*接近*原始值，但不保证逐位一致（量化/精度）。 |
| `GetDataByIds` | `tl::expected<DatasetPtr, Error> GetDataByIds(const int64_t* ids, int64_t count) const` | 返回 `ids` 的全部已存数据（向量、属性、extra info）。 |
| `GetDataByIdsWithFlag` | `tl::expected<DatasetPtr, Error> GetDataByIdsWithFlag(const int64_t* ids, int64_t count, uint64_t selected_data_flag) const` | 类似 `GetDataByIds`，但通过 [`DATA_FLAG_*`](#数据选择标志) 选择字段。 |
| `GetIndexDetailInfos` | `tl::expected<std::vector<IndexDetailInfo>, Error> GetIndexDetailInfos() const` | 列出可自省的细节字段。见 [`IndexDetailInfo`](types.md#索引细节信息)。 |
| `GetDetailDataByName` | `tl::expected<DetailDataPtr, Error> GetDetailDataByName(const std::string& name, IndexDetailInfo& info) const` | 按名称获取一份细节数据负载。 |

见 [索引自省](../advanced/introspection.md) 与 `examples/cpp/317_feature_get_detail_data.cpp`。

## 能力探测、合并、克隆与导出

| 方法 | 签名 | 说明 |
|------|------|------|
| `CheckFeature` | `bool CheckFeature(IndexFeature feature) const` | 探测某个可选能力是否受支持。见 [`IndexFeature`](types.md#indexfeature)。 |
| `Merge` | `tl::expected<void, Error> Merge(const std::vector<MergeUnit>& merge_units)` | 合并同类型子索引并进行 id 重映射。见 [`MergeUnit`](#mergeunit-与-idmapfunction)。 |
| `Clone` | `tl::expected<IndexPtr, Error> Clone(const std::shared_ptr<Allocator>& allocator = nullptr) const` | 深拷贝索引。 |
| `ExportModel` | `tl::expected<IndexPtr, Error> ExportModel() const` | 返回一个只携带已训练模型的空索引。 |
| `ExportIDs` | `tl::expected<DatasetPtr, Error> ExportIDs() const` | 以 dataset 形式返回全部 id。 |
| `SetImmutable` | `tl::expected<void, Error> SetImmutable()` | 冻结索引；后续的增/删将被拒绝。 |

见 `examples/cpp/309_feature_clone.cpp`、`310_feature_export_model.cpp`、
`315_feature_hgraph_merge.cpp`，以及 [索引生命周期管理](../advanced/index_lifecycle.md)。

## 序列化

| 方法 | 签名 | 说明 |
|------|------|------|
| `Serialize` | `tl::expected<BinarySet, Error> Serialize() const` | *（纯虚函数）* 序列化为内存中的 [`BinarySet`](serialization.md#binaryset)。 |
| `Serialize` | `tl::expected<void, Error> Serialize(WriteFuncType write_func) const` | 通过 [`WriteFuncType`](#writefunctype) 落盘回调流式输出序列化结果。 |
| `Serialize` | `tl::expected<void, Error> Serialize(std::ostream& out_stream)` | 序列化到一个已打开的输出流。 |
| `Deserialize` | `tl::expected<void, Error> Deserialize(const BinarySet& binary_set)` | *（纯虚函数）* 从 `BinarySet` 恢复。索引非空时失败。 |
| `Deserialize` | `tl::expected<void, Error> Deserialize(const ReaderSet& reader_set)` | *（纯虚函数）* 从 [`ReaderSet`](serialization.md#readerset)（如磁盘 reader）恢复。 |
| `Deserialize` | `tl::expected<void, Error> Deserialize(std::istream& in_stream)` | 从一个已打开的输入流恢复。 |

在非空索引上反序列化会得到 `INDEX_NOT_EMPTY`。见 [序列化格式](../advanced/serialization.md) 与
`examples/cpp/401_persistent_kv.cpp` / `402_persistent_streaming.cpp`。

## 缓存（构建加速）

| 方法 | 签名 | 说明 |
|------|------|------|
| `ExportCache` | `tl::expected<void, Error> ExportCache(std::ostream& out_stream) const` | 写出构建期缓存（如图邻居），可加速后续的 `Build`。 |
| `ImportCache` | `tl::expected<void, Error> ImportCache(std::istream& in_stream)` | 加载之前导出的缓存；下一次 `Build` 会复用它。 |

## 统计与自省

除非另有说明，这些方法直接返回值。**标注为“抛出”的方法在索引不支持时会抛出 `std::runtime_error`**
（而非 `tl::expected`）。

| 方法 | 签名 | 说明 |
|------|------|------|
| `GetIndexType` | `IndexType GetIndexType() const` | 不支持时**抛出**。 |
| `GetNumElements` | `int64_t GetNumElements() const` | *（纯虚函数）* 存活元素数。 |
| `GetNumberRemoved` | `int64_t GetNumberRemoved() const` | 不支持时**抛出**。已删除元素数。 |
| `GetMemoryUsage` | `int64_t GetMemoryUsage() const` | *（纯虚函数）* 索引占用的字节数。 |
| `GetMemoryUsageDetail` | `std::string GetMemoryUsageDetail() const` | 不支持时**抛出**。各组件内存的 JSON。 |
| `EstimateMemory` | `uint64_t EstimateMemory(uint64_t num_elements) const` | 不支持时**抛出**。`num_elements` 的预估字节数。 |
| `GetEstimateBuildMemory` | `int64_t GetEstimateBuildMemory(int64_t num_elements) const` | 不支持时**抛出**。预估构建峰值内存。 |
| `GetStats` | `std::string GetStats() const` | 不支持时**抛出**。运行期统计的 JSON。 |
| `AnalyzeIndexBySearch` | `std::string AnalyzeIndexBySearch(const SearchRequest& request)` | 不支持时**抛出**。一次探测搜索的分析 JSON。 |
| `CheckIdExist` | `bool CheckIdExist(int64_t id) const` | 不支持时**抛出**。`id` 是否存在。 |

见 `examples/cpp/308_feature_estimate_memory.cpp`、`319_feature_get_memory_usage.cpp`，以及
[索引分析工具](../resources/analyze_index.md)。

## 参见

- [Dataset](dataset.md) —— 构造查询/基础输入并读取搜索结果。
- [搜索请求与过滤器](search.md) —— `SearchRequest` 字段与过滤器类型。
- [序列化类型](serialization.md) —— `BinarySet`、`Binary`、`Reader`、`ReaderSet`。
- [辅助类型](types.md) —— `IndexFeature`、`IndexDetailInfo`、`AttributeSet`。
