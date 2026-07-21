
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
#include "algorithm/sindi/term_id_mapper.h"
#include "algorithm/sindi_v2/sindi_v2_parameter.h"
#include "datacell/disk_sindi_term_datacell.h"
#include "datacell/flatten_interface.h"
#include "datacell/immutable_sindi_term_datacell.h"
#include "datacell/mutable_sindi_term_datacell.h"
#include "index_common_param.h"
#include "vsag/allocator.h"

namespace vsag {

class SINDIV2 : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    explicit SINDIV2(const SINDIV2ParameterPtr& param, const IndexCommonParam& common_param);

    SINDIV2(const ParamPtr& param, const IndexCommonParam& common_param)
        : SINDIV2(std::dynamic_pointer_cast<SINDIV2Parameter>(param), common_param){};

    ~SINDIV2() override = default;

    std::string
    GetName() const override {
        return "sindi_v2";
    }

    void
    InitFeatures() override;

    std::unordered_map<std::string, uint64_t>
    GetMemoryUsageDetail() const override {
        return {};
    }

    std::string
    GetStats() const override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    bool
    UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update = false) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter,
              Allocator* allocator) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(const BinarySet& binary_set) override;

    void
    Deserialize(std::istream& in_stream) override;

    void
    Deserialize(StreamReader& reader) override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    IndexType
    GetIndexType() const override {
        return IndexType::SINDI_V2;
    }

    int64_t
    GetNumElements() const override {
        return cur_element_count_;
    }

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    float
    CalcDistanceById(const DatasetPtr& vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    DatasetPtr
    CalDistanceById(const DatasetPtr& query,
                    const int64_t* ids,
                    int64_t count,
                    bool calculate_precise_distance = true) const override;

    std::pair<int64_t, int64_t>
    GetMinAndMaxId() const override;

    void
    SetImmutable() override;

    void
    SetIO(const std::shared_ptr<Reader> reader) override;

private:
    template <InnerSearchMode mode>
    DatasetPtr
    search_impl(const SparseTermComputerPtr& computer,
                const InnerSearchParam& inner_param,
                Allocator* allocator,
                bool use_term_lists_heap_insert,
                SindiQueryContext& query_context,
                const SparseVector* original_query = nullptr) const;

    std::pair<int64_t, int64_t>
    get_min_max_window_id(const FilterPtr& filter) const;

    void
    cal_memory_usage();

    SparseVector
    remap_sparse_vector_for_build(const SparseVector& input, Vector<uint32_t>& tmp_ids);

    SparseVector
    remap_sparse_vector_for_query(const SparseVector& input,
                                  Vector<uint32_t>& tmp_ids,
                                  Vector<float>& tmp_vals) const;

    std::vector<int64_t>
    build_immutable(const DatasetPtr& base);

    [[nodiscard]] MutableSindiTermDataCellPtr
    get_mutable_term_datacell() const;

    [[nodiscard]] uint32_t
    get_term_dict_count() const;

    void
    serialize_term_layout(StreamWriter& writer) const;

private:
    mutable std::shared_mutex global_mutex_;

    uint32_t term_id_limit_{0};
    uint32_t window_size_{0};

    SindiTermDataCellPtr term_datacell_;

    int64_t cur_element_count_{0};

    bool use_reorder_{false};
    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};
    float doc_retain_ratio_{0};

    FlattenInterfacePtr rerank_flat_{nullptr};

    QuantizationParamsPtr quantization_params_;
    uint32_t avg_doc_term_length_{100};

    bool remap_term_ids_{false};
    std::shared_ptr<TermIdMapper> term_id_mapper_{nullptr};

    bool immutable_enabled_{false};

    std::string rerank_layout_{"none"};
    uint32_t rerank_layout_top_terms_{16};

    SINDIV2ParameterPtr param_;
    IndexCommonParam common_param_;

    bool is_deserialized_{false};
};

}  // namespace vsag
