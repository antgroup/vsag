
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

#include "algorithm/inner_index_interface.h"
#include "datacell/flatten_interface.h"
#include "index_common_param.h"
#include "utils/visited_list.h"

namespace vsag {

class UHGParameter : public InnerIndexParameter {
public:
    void
    FromJson(const JsonType& json) override;
    JsonType
    ToJson() const override;
    uint64_t max_degree{64};
    uint64_t neighbors_per_alpha{32};
    uint64_t construction_candidates{200};
    uint64_t alpha_intervals{10};
    JsonType hgraph;
    JsonType sindi;
};

// A fixed alpha-cover topology with query-specific dense/sparse IP distances.
// HGraph/SINDI provide guides; UHG also owns forward vectors for batched graph distances.
class UHG : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external, const IndexCommonParam& common);
    UHG(const ParamPtr& param, const IndexCommonParam& common);
    void
    InitFeatures() override;
    bool
    UpdateId(int64_t old_id, int64_t new_id) override;
    std::string
    GetName() const override {
        return "uhg";
    }
    IndexType
    GetIndexType() const override {
        return IndexType::UHG;
    }
    int64_t
    GetNumElements() const override;
    bool
    CheckIdExist(int64_t id) const override;
    uint64_t
    GetMemoryUsage() const override;
    std::vector<int64_t>
    Add(const DatasetPtr& base) override;
    std::vector<int64_t>
    Build(const DatasetPtr& base) override;
    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;
    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;
    void
    Serialize(StreamWriter& writer) const override;
    void
    Deserialize(StreamReader& reader) override;

private:
    void
    validate(const DatasetPtr& data, bool base) const;
    std::pair<InnerIndexPtr, InnerIndexPtr>
    make_guides() const;
    std::pair<FlattenInterfacePtr, FlattenInterfacePtr>
    make_distance_cells() const;
    std::shared_ptr<UHGParameter> param_;
    IndexCommonParam common_;
    InnerIndexPtr dense_;
    InnerIndexPtr sparse_;
    FlattenInterfacePtr dense_codes_;
    FlattenInterfacePtr sparse_codes_;
    std::shared_ptr<VisitedListPool> visited_pool_;
    std::vector<int64_t> ids_;
    std::vector<double> norms_;
    std::vector<std::vector<uint32_t>> graph_;
    std::unordered_map<int64_t, uint32_t> positions_;
    mutable std::shared_mutex mutex_;
};

}  // namespace vsag
