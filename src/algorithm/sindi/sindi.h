
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

#include <optional>

#include "algorithm/inner_index_interface.h"
#include "algorithm/sindi/term_id_mapper.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_term_datacell.h"
#include "vsag/allocator.h"

namespace vsag {

struct ImmutableSINDIWindow {
    explicit ImmutableSINDIWindow(Allocator* allocator)
        : sorted_global_to_local_terms(allocator),
          offsets(allocator),
          id_payloads(allocator),
          value_payloads(allocator) {
    }

    Vector<std::pair<uint32_t, uint32_t>> sorted_global_to_local_terms;
    Vector<uint32_t> offsets;
    Vector<uint16_t> id_payloads;
    Vector<uint8_t> value_payloads;
};

struct ImmutableSINDIData {
    explicit ImmutableSINDIData(Allocator* allocator) : windows(allocator) {
    }

    uint32_t value_code_size{0};
    SparseValueQuantizationType sparse_value_quant_type{SparseValueQuantizationType::FP32};
    Vector<ImmutableSINDIWindow> windows;
};

/**
 * @brief SINDI: Sparse INverted Index with windowed term lists.
 *
 * SINDI is designed for high-dimensional sparse vectors (e.g. learned sparse
 * representations such as SPLADE).  The index partitions the internal-id
 * space into fixed-size windows; each window maintains per-term inverted
 * lists so that a query only touches the windows that overlap the query's
 * non-zero dimensions.
 *
 * Optional features:
 *  - Quantization: compress stored term values.
 *  - Reranking: use an embedded sparse-vector data cell for precise re-scoring.
 *  - Term-id remapping: remap external dim-ids to a dense [0,N) range to
 *    save memory when the vocabulary is sparse.
 *
 * Fork() is intentionally unsupported (returns nullptr) because the window
 * layout depends on the insertion order and cannot be trivially cloned.
 */
class SINDI : public InnerIndexInterface {
public:
    using ImmutableMappedQueryTerms = Vector<std::pair<uint32_t, uint32_t>>;

    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    friend class SINDIAnalyzer;

    explicit SINDI(const SINDIParameterPtr& param, const IndexCommonParam& common_param);

    SINDI(const ParamPtr& param, const IndexCommonParam& common_param)
        : SINDI(std::dynamic_pointer_cast<SINDIParameter>(param), common_param){};

    ~SINDI() override = default;

    std::string
    GetName() const override {
        return "sindi";
    }

    void
    InitFeatures() override;

    std::unordered_map<std::string, uint64_t>
    GetMemoryUsageDetail() const override {
        return {};
    }

    std::string
    GetStats() const override;

    std::string
    AnalyzeIndexBySearch(const SearchRequest& request) override;

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
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const vsag::FilterPtr& filter,
              vsag::Allocator* allocator) const override;

    DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return nullptr;
    };

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    GetSparseVectorByInnerId(InnerIdType inner_id,
                             SparseVector* data,
                             Allocator* specified_allocator) const override;

    IndexType
    GetIndexType() const override {
        return IndexType::SINDI;
    }

    int64_t
    GetNumElements() const override {
        auto total = cur_element_count_.load();
        auto deleted = delete_count_.load();
        return total > deleted ? total - deleted : 0;
    }

    int64_t
    GetNumberRemoved() const override {
        return delete_count_.load();
    }

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode) override;

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

private:
    static constexpr float K_TERM_LISTS_HEAP_INSERT_PRUNE_THRESHOLD = 0.1F;

    /**
     * @brief Core search implementation shared by KnnSearch / RangeSearch.
     *
     * @tparam mode   KNN_SEARCH or RANGE_SEARCH.
     * @param computer  evaluates partial IP contributions per window.
     * @param inner_param  top-k / radius / ef parameters.
     * @param allocator  scratch allocator for the result heap.
     * @param use_term_lists_heap_insert  when true, accumulate candidates via
     *                                    per-term heap insertion (faster for
     *                                    very sparse queries).
     * @param original_query  non-null only when reranking is needed.
     */
    template <InnerSearchMode mode>
    DatasetPtr
    search_impl(const SparseTermComputerPtr& computer,
                const InnerSearchParam& inner_param,
                Allocator* allocator,
                bool use_term_lists_heap_insert,
                const SparseVector* original_query = nullptr) const;

    template <InnerSearchMode mode>
    DatasetPtr
    immutable_search_impl(const SparseTermComputerPtr& computer,
                          const InnerSearchParam& inner_param,
                          Allocator* allocator,
                          bool use_term_lists_heap_insert,
                          const SparseVector* original_query = nullptr) const;

    bool
    UseTermListsHeapInsert(const SINDISearchParameter& search_param) const;

#ifdef VSAG_SINDI_TEST_ACCESS
    friend class SINDITestAccess;
#endif

    /**
     * @brief Derive the [min_window_id, max_window_id] range that could
     *        contain vectors passing @p filter.
     */
    std::pair<int64_t, int64_t>
    get_min_max_window_id(const FilterPtr& filter) const;

    void
    deserialize_immutable_window(StreamReader& reader_ref, ImmutableSINDIWindow& window) const;

    std::optional<uint32_t>
    get_immutable_local_term(const ImmutableSINDIWindow& window, uint32_t term) const;

    void
    map_immutable_query_terms(const ImmutableSINDIWindow& window,
                              const SparseTermComputerPtr& computer,
                              ImmutableMappedQueryTerms& mapped_terms) const;

    void
    scan_immutable_window_by_mapped_terms(float* dists,
                                          const ImmutableSINDIWindow& window,
                                          const SparseTermComputerPtr& computer,
                                          const ImmutableMappedQueryTerms& mapped_terms) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    immutable_insert_candidate_into_heap(uint32_t id,
                                         float& dist,
                                         float& cur_heap_top,
                                         MaxHeap& heap,
                                         uint32_t offset_id,
                                         float radius,
                                         int range_search_limit_size,
                                         const FilterPtr& filter) const;

    template <InnerSearchType type>
    bool
    immutable_fill_heap_initial(uint32_t id,
                                float& dist,
                                float& cur_heap_top,
                                MaxHeap& heap,
                                uint32_t offset_id,
                                uint32_t n_candidate,
                                const FilterPtr& filter) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    immutable_insert_heap_by_mapped_terms(float* dists,
                                          const ImmutableSINDIWindow& window,
                                          const SparseTermComputerPtr& computer,
                                          const ImmutableMappedQueryTerms& mapped_terms,
                                          MaxHeap& heap,
                                          const InnerSearchParam& param,
                                          uint32_t offset_id) const;

    template <InnerSearchMode mode, InnerSearchType type>
    void
    immutable_insert_heap_by_dists(float* dists,
                                   uint32_t dists_size,
                                   MaxHeap& heap,
                                   const InnerSearchParam& param,
                                   uint32_t offset_id) const;

    /// Recalculate and cache the memory-usage counter.
    void
    cal_memory_usage();

    /**
     * @brief Compact a sparse vector's dim-ids into the remapped space
     *        during Build.  Uses @p tmp_ids as scratch buffer.
     */
    SparseVector
    remap_sparse_vector_for_build(const SparseVector& input, Vector<uint32_t>& tmp_ids);

    /**
     * @brief Map a query's dim-ids into the remapped space.
     *
     * Unlike the Build variant this does not mutate the index and uses
     * @p tmp_ids / @p tmp_vals as scratch buffers.
     */
    SparseVector
    remap_sparse_vector_for_query(const SparseVector& input,
                                  Vector<uint32_t>& tmp_ids,
                                  Vector<float>& tmp_vals) const;

private:
    mutable std::shared_mutex global_mutex_;  // protects structural mutations

    uint32_t term_id_limit_{0};  // max number of distinct terms per window
    uint32_t window_size_{0};    // number of vectors per window

    Vector<SparseTermDataCellPtr> window_term_list_;  // one inverted list per window

    std::atomic<int64_t> cur_element_count_{0};  // total inserted vectors
    std::atomic<int64_t> delete_count_{0};       // soft-deleted vectors

    bool use_reorder_{false};  // enable reranking stage

    float doc_prune_ratio_{0};   // ratio of docs pruned during build
    float doc_retain_ratio_{0};  // ratio of docs kept after pruning

    FlattenInterfacePtr rerank_flat_{nullptr};  // re-rank back-end

    SparseValueQuantizationType sparse_value_quant_type_{SparseValueQuantizationType::FP32};

    bool deserialize_without_footer_{false};  // backward-compat: old format lacks footer
    bool deserialize_without_buffer_{false};  // backward-compat: old format lacks buffer

    std::shared_ptr<QuantizationParams> quantization_params_;
    uint32_t avg_doc_term_length_{100};  // average non-zero terms per doc (estimation)

    bool remap_term_ids_{false};                             // enable dense dim-id remapping
    std::shared_ptr<TermIdMapper> term_id_mapper_{nullptr};  // maps external->internal ids

    bool immutable_enabled_{false};
    std::unique_ptr<ImmutableSINDIData> immutable_data_{nullptr};
};

}  // namespace vsag
