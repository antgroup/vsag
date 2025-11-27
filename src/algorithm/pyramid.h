
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

#include <utility>

#include "datacell/graph_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/distance_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "index_feature_list.h"
#include "inner_index_interface.h"
#include "io/memory_io_parameter.h"
#include "pyramid_zparameters.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "utils/lock_strategy.h"

namespace vsag {

class IndexNode;
using SearchFunc = std::function<DistHeapPtr(const IndexNode* node, const VisitedListPtr& vl)>;

std::vector<std::string>
split(const std::string& str, char delimiter);

class IndexNode {
public:
    IndexNode(IndexCommonParam* common_param, GraphInterfaceParamPtr graph_param);

    void
    BuildGraph(ODescent& odescent);

    void
    InitGraph();

    void
    SearchGraph(const SearchFunc& search_func,
                const VisitedListPtr& vl,
                const DistHeapPtr& search_result,
                int64_t ef_search) const;

    void
    AddChild(const std::string& key);

    std::shared_ptr<IndexNode>
    GetChild(const std::string& key, bool need_init = false);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

public:
    GraphInterfacePtr graph_{nullptr};
    InnerIdType entry_point_{0};
    uint32_t level_{0};
    mutable std::shared_mutex mutex_;

    Vector<InnerIdType> ids_;
    bool has_index_{false};

private:
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> children_;
    IndexCommonParam* common_param_{nullptr};
    GraphInterfaceParamPtr graph_param_{nullptr};
};

// Pyramid index was introduced since v0.14
class Pyramid : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    Pyramid(const PyramidParamPtr& pyramid_param, const IndexCommonParam& common_param)
        : InnerIndexInterface(pyramid_param, common_param),
          pyramid_param_(pyramid_param),
          common_param_(common_param),
          alpha_(pyramid_param->alpha) {
        base_codes_ =
            FlattenInterface::MakeInstance(pyramid_param_->base_codes_param, common_param_);
        root_ = std::make_shared<IndexNode>(&common_param_, pyramid_param_->graph_param);
        points_mutex_ = std::make_shared<PointsMutex>(max_capacity_, allocator_);
        searcher_ = std::make_unique<BasicSearcher>(common_param_, points_mutex_);
        if (use_reorder_) {
            precise_codes_ =
                FlattenInterface::MakeInstance(pyramid_param_->precise_codes_param, common_param_);
            reorder_ = std::make_shared<FlattenReorder>(precise_codes_, allocator_);
        }
    }

    explicit Pyramid(const ParamPtr& param, const IndexCommonParam& common_param)
        : Pyramid(std::dynamic_pointer_cast<PyramidParameters>(param), common_param){};

    ~Pyramid() = default;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<Pyramid>(this->create_param_ptr_, param);
    }

    IndexType
    GetIndexType() const override {
        return IndexType::PYRAMID;
    }

    std::string
    GetName() const override {
        return INDEX_PYRAMID;
    }

    int64_t
    GetNumElements() const override;

    void
    InitFeatures() override;

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
    Train(const vsag::DatasetPtr& base) override;

private:
    void
    resize(int64_t new_max_capacity);

    DatasetPtr
    search_impl(const DatasetPtr& query,
                int64_t limit,
                const SearchFunc& search_func,
                int64_t ef_search) const;

    bool
    is_update_entry_point(uint64_t total_count) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double rand_value = distribution(level_generator_);
        return static_cast<double>(total_count) * rand_value < 1.0;
    }

    std::vector<int64_t>
    build_by_odescent(const DatasetPtr& base);

    void
    add_one_point(const std::shared_ptr<IndexNode>& node,
                  InnerIdType inner_id,
                  const float* vector);

    static std::vector<std::vector<std::string>>
    parse_path(const std::string& path);

private:
    IndexCommonParam common_param_;
    PyramidParamPtr pyramid_param_{nullptr};
    std::shared_ptr<IndexNode> root_{nullptr};
    FlattenInterfacePtr base_codes_{nullptr};
    FlattenInterfacePtr precise_codes_{nullptr};
    std::unique_ptr<VisitedListPool> pool_ = nullptr;

    MutexArrayPtr points_mutex_{nullptr};
    std::unique_ptr<BasicSearcher> searcher_ = nullptr;
    int64_t max_capacity_{0};
    int64_t cur_element_count_{0};
    float alpha_{1.0F};

    std::shared_mutex resize_mutex_;
    std::mutex cur_element_count_mutex_;
    std::string graph_type_{GRAPH_TYPE_VALUE_NSW};

    std::mutex entry_point_mutex_;
    std::default_random_engine level_generator_{2021};
    ReorderInterfacePtr reorder_{nullptr};
};

}  // namespace vsag
