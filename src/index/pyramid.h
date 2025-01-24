
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

#include "base_filter_functor.h"
#include "data_cell/graph_interface.h"
#include "impl/basic_searcher.h"
#include "impl/odescent_graph_builder.h"
#include "io/memory_io_parameter.h"
#include "logger.h"
#include "pyramid_zparameters.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "safe_allocator.h"

namespace vsag {

class IndexNode;
using SearchFunc = std::function<MaxHeap(const std::shared_ptr<IndexNode>&)>;

class IndexNode {
public:
    IndexNode(IndexCommonParam* common_param, GraphInterfaceParamPtr graph_param);

    void
    BuildGraph(ODescent& odescent);

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

    Vector<InnerIdType> ids_;  //

private:
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> children_;
    IndexCommonParam* common_param_{nullptr};
    GraphInterfaceParamPtr graph_param_{nullptr};
};

class Pyramid : public Index {
public:
    Pyramid(PyramidParameters pyramid_param, const IndexCommonParam& common_param)
        : pyramid_param_(std::move(pyramid_param)),
          common_param_(std::move(common_param)),
          labels_(common_param_.allocator_.get()) {
        searcher_ = std::make_unique<BasicSearcher>(common_param_);
        flatten_interface_ptr_ =
            FlattenInterface::MakeInstance(pyramid_param_.flatten_data_cell_param, common_param_);
        root_ = std::make_shared<IndexNode>(&common_param_, pyramid_param_.graph_param);
    }

    ~Pyramid() = default;

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid);)
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter);)
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, limited_size);)
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid, limited_size);)
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, filter, limited_size);)
    }

    tl::expected<BinarySet, Error>
    Serialize() const override;

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override;

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override;

    int64_t
    GetNumElements() const override;

    int64_t
    GetMemoryUsage() const override;

private:
    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               BitsetPtr invalid = nullptr) const;

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 int64_t limited_size = -1) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 BitsetPtr invalid,
                 int64_t limited_size = -1) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const std::function<bool(int64_t)>& filter,
                 int64_t limited_size = -1) const;

    tl::expected<DatasetPtr, Error>
    search_impl(const DatasetPtr& query, int64_t limit, const SearchFunc& search_func) const;

    uint64_t
    cal_serialize_size() const;

private:
    IndexCommonParam common_param_;
    PyramidParameters pyramid_param_;
    std::shared_ptr<IndexNode> root_{nullptr};
    FlattenInterfacePtr flatten_interface_ptr_{nullptr};
    Vector<LabelType> labels_;
    std::unique_ptr<VisitedListPool> pool_ = nullptr;
    std::unique_ptr<BasicSearcher> searcher_ = nullptr;
};

}  // namespace vsag
