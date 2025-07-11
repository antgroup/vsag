
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "data_cell/attribute_bucket_inverted_datacell.h"
#include "data_cell/bucket_datacell.h"
#include "data_cell/flatten_interface.h"
#include "impl/basic_searcher.h"
#include "impl/heap/distance_heap.h"
#include "index/index_common_param.h"
#include "inner_index_interface.h"
#include "ivf_parameter.h"
#include "ivf_partition/ivf_partition_strategy.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

// IVF index was introduced since v0.14
class IVF : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param);

    explicit IVF(const ParamPtr& param, const IndexCommonParam& common_param)
        : IVF(std::dynamic_pointer_cast<IVFParameter>(param), common_param){};

    ~IVF() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_IVF;
    }

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<IVF>(this->create_param_ptr_, param);
    }

    void
    InitFeatures() override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    void
    Train(const DatasetPtr& data) override;

    InnerIndexPtr
    ExportModel(const IndexCommonParam& param) const override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Merge(const std::vector<MergeUnit>& merge_units) override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    int64_t
    GetNumElements() const override;

private:
    InnerSearchParam
    create_search_param(const std::string& parameters, const FilterPtr& filter) const;

    template <InnerSearchMode mode = KNN_SEARCH>
    DistHeapPtr
    search(const DatasetPtr& query, const InnerSearchParam& param) const;

    DatasetPtr
    reorder(int64_t topk, DistHeapPtr& input, const float* query) const;

    void
    merge_one_unit(const MergeUnit& unit);

    void
    check_merge_illegal(const MergeUnit& unit) const;

private:
    BucketInterfacePtr bucket_{nullptr};

    IVFPartitionStrategyPtr partition_strategy_{nullptr};
    BucketIdType buckets_per_data_;

    int64_t total_elements_{0};

    bool use_reorder_{false};

    bool is_trained_{false};
    bool use_residual_{false};
    bool use_attribute_filter_{false};

    FlattenInterfacePtr reorder_codes_{nullptr};

    AttrInvertedInterfacePtr attr_filter_index_{nullptr};
};
}  // namespace vsag
