
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

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "algorithm/sindi/proximity_scorer.h"
#include "algorithm/sindi/term_id_mapper.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_term_datacell.h"
#include "vsag/allocator.h"

namespace vsag {

struct ImmutableSINDIWindow {
    explicit ImmutableSINDIWindow(Allocator* allocator)
        : sorted_global_terms(allocator),
          offsets(allocator),
          id_payloads(allocator),
          value_payloads(allocator),
          pos_offsets(allocator),
          pos_pool(allocator) {
    }

    Vector<uint32_t> sorted_global_terms;
    Vector<uint32_t> offsets;
    Vector<uint16_t> id_payloads;
    Vector<uint8_t> value_payloads;
    // Position storage (only populated when store_positions_ is enabled). Flat
    // per-window layout aligned to `offsets`: for a flat posting index
    // p = offsets[local_term] + within, its positions are
    // pos_pool[pos_offsets[p] .. pos_offsets[p+1]). pos_offsets has a trailing
    // sentinel so pos_offsets[p+1] is always valid; size == offsets.back()+1.
    Vector<uint32_t> pos_offsets;
    Vector<uint16_t> pos_pool;
};

struct ImmutableSINDIData {
    explicit ImmutableSINDIData(Allocator* allocator) : windows(allocator) {
    }

    uint32_t value_code_size{0};
    SparseValueQuantizationType sparse_value_quant_type{SparseValueQuantizationType::FP32};
    Vector<ImmutableSINDIWindow> windows;
};

/**
 * @brief Aggregated proximity/phrase parameters passed through the search path.
 *
 * Bundles the proximity/phrase scalars plus two derived values that are NOT raw
 * SINDISearchParameter fields: `query_term_count` (the remapped query length)
 * and `phrase_terms` (the remapped phrase terms). Built once at the KNN/RANGE
 * dispatch site and forwarded by const-ref to search_impl / immutable_search_impl.
 */
struct ProximityPhraseContext {
    float proximity_weight{0.0f};
    bool proximity_ordered{false};
    uint32_t proximity_candidates{10000};
    bool proximity_boost_multiplicative{true};
    bool proximity_all_pairs{false};
    uint32_t query_term_count{0};
    std::vector<uint32_t> phrase_terms;
    uint32_t phrase_slop{0};
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
                const SparseVector* original_query = nullptr,
                const ProximityPhraseContext* ctx = nullptr) const;

    /**
     * @brief Position accessor over the mutable per-term datacell layout.
     *
     * Nested so it can reach SparseTermDataCell's public accessors while keeping
     * the type private to SINDI.
     */
    struct DatacellPositionBackend {
        const SparseTermDataCellPtr& term_list;
        float term_retain_ratio{1.0f};

        // Read-only view of one query term's posting list, produced by
        // ResolveTerm and reused across candidate docs by LookupPositions.
        //   valid       : false when the term is absent (or stores no positions);
        //                 such a view yields empty spans for any doc.
        //   id_begin    : (*term_ids_[global]).data(), start of the doc-id list.
        //   retained    : term_sizes_[global] * term_retain_ratio, the prefix to
        //                 binary-search.
        //   global_term : needed by GetPositionsView on the hit path.
        struct TermPostingListView {
            bool valid{false};
            const uint16_t* id_begin{nullptr};
            uint32_t retained{0};
            uint32_t global_term{0};
        };

        // Resolve one query term to a reusable posting-list view: guard checks +
        // capture the doc-id list pointer and retained prefix length.
        TermPostingListView
        ResolveTerm(uint32_t global_term) const {
            TermPostingListView view;
            if (global_term >= term_list->term_capacity_ ||
                term_list->term_sizes_[global_term] == 0 || !term_list->term_ids_[global_term] ||
                !term_list->term_pos_offsets_[global_term]) {
                return view;  // valid stays false
            }
            const auto& ids = *term_list->term_ids_[global_term];
            view.id_begin = ids.data();
            view.retained = static_cast<uint32_t>(
                static_cast<float>(term_list->term_sizes_[global_term]) * term_retain_ratio);
            view.global_term = global_term;
            view.valid = true;
            return view;
        }

        // Look up positions for a doc using a pre-resolved posting-list view.
        // Binary-searches doc_id in the view's retained posting prefix; the
        // retained prefix mirrors the scan's term_retain_ratio pruning, so a doc
        // present only in the pruned tail reads as absent. Returns {nullptr, 0}
        // when the term is absent from the doc (or has no stored positions).
        PosSpan
        LookupPositions(const TermPostingListView& view, uint16_t doc_id) const {
            if (!view.valid) {
                return PosSpan{nullptr, 0};
            }
            const auto* begin = view.id_begin;
            const auto* end = begin + view.retained;
            const auto* found = std::lower_bound(begin, end, doc_id);
            if (found == end || *found != doc_id) {
                return PosSpan{nullptr, 0};
            }
            return GetPositionsView(view.global_term, static_cast<uint32_t>(found - begin));
        }

    private:
        // Return the position list stored at a known slot in a term's posting.
        //   global_term : the term whose posting we index into.
        //   within      : the slot index within that posting (0-based rank).
        PosSpan
        GetPositionsView(uint32_t global_term, uint32_t within) const {
            auto [ptr, size] = term_list->GetPositionsView(global_term, within);
            return PosSpan{ptr, size};
        }
    };

    /**
     * @brief Position accessor over the immutable flat per-window layout.
     *
     * Nested so it can call SINDI's private get_immutable_local_term (global ->
     * local term translation, honoring remap).
     */
    struct ImmutableWindowPositionBackend {
        const SINDI* self;
        const ImmutableSINDIWindow& window;
        float term_retain_ratio{1.0f};

        // Read-only view of one query term's inverted list in this window,
        // produced by ResolveTerm and reused across candidate docs by
        // LookupPositions. Resolving once outside the candidate loop pays the
        // global->local translation query_len times instead of
        // candidates*query_len times.
        //   valid    : false when the term is absent from this window (or the
        //              window stores no positions); such a view yields empty spans.
        //   id_begin : id_payloads.data() + begin_doc, start of the doc-id list.
        //   retained : length of the retained posting prefix to binary-search.
        //   flat_base: offsets[local], so a doc at prefix rank r maps to flat
        //              posting index flat_base + r.
        struct TermPostingListView {
            bool valid{false};
            const uint16_t* id_begin{nullptr};
            uint32_t retained{0};
            uint32_t flat_base{0};
        };

        // Resolve one query term to a reusable posting-list view (one
        // global->local translation + posting-range capture). See
        // TermPostingListView.
        TermPostingListView
        ResolveTerm(uint32_t global_term) const {
            TermPostingListView view;
            if (window.pos_offsets.empty()) {
                return view;  // valid stays false
            }
            auto local = self->get_immutable_local_term(window, global_term);
            if (!local.has_value()) {
                return view;  // term absent from this window
            }
            const uint32_t begin_doc = window.offsets[local.value()];
            const uint32_t end_doc = window.offsets[local.value() + 1];
            view.retained =
                static_cast<uint32_t>(static_cast<float>(end_doc - begin_doc) * term_retain_ratio);
            view.id_begin = window.id_payloads.data() + begin_doc;
            view.flat_base = begin_doc;
            view.valid = true;
            return view;
        }

        // Look up positions for a doc using a pre-resolved posting-list view.
        // Only the doc binary search + position read remain in the candidate
        // loop; the global->local translation was already paid by ResolveTerm.
        // Binary-searches doc_id in the view's retained posting prefix; the
        // retained prefix mirrors the scan's term_retain_ratio pruning, so a doc
        // present only in the pruned tail reads as absent. Returns {nullptr, 0}
        // when the term is absent from the doc (or has no stored positions).
        PosSpan
        LookupPositions(const TermPostingListView& view, uint16_t doc_id) const {
            if (!view.valid) {
                return PosSpan{nullptr, 0};
            }
            const auto* begin = view.id_begin;
            const auto* end = begin + view.retained;
            const auto* found = std::lower_bound(begin, end, doc_id);
            if (found == end || *found != doc_id) {
                return PosSpan{nullptr, 0};
            }
            return positions_at_flat_index(view.flat_base + static_cast<uint32_t>(found - begin));
        }

    private:
        // Return the position list stored at a flat posting index (no
        // global->local translation).
        PosSpan
        positions_at_flat_index(uint32_t p) const {
            const uint32_t start = window.pos_offsets[p];
            const uint32_t end = window.pos_offsets[p + 1];
            return PosSpan{window.pos_pool.data() + start, end - start};
        }
    };

    /**
     * @brief Phrase filter: for candidates in this window with dists < 0, apply
     *        the sloppy phrase constraint; those failing are set to dists = 0.
     *
     * Phrase terms are resolved once via pos.ResolveTerm before the doc loop;
     * per-doc positions are then fetched via pos.LookupPositions (binary search
     * over the retained posting prefix), so no reverse offset map is needed.
     * phrase_spans / norm_buffer / count_buffer are reused buffers hoisted out
     * of the window loop by the caller.
     *
     * @tparam PosBackend  supplies ResolveTerm(global_term) -> TermPostingListView
     *                     and LookupPositions(view, doc_id) -> PosSpan; a
     *                     {nullptr, 0} span means the term has no positions in
     *                     that doc. Backends wrap either the mutable datacell or
     *                     an immutable window.
     */
    template <class PosBackend>
    void
    phrase_filter(float* dists,
                  const PosBackend& pos,
                  const std::vector<uint32_t>& phrase_terms,
                  uint32_t phrase_slop,
                  std::vector<PosSpan>& phrase_spans,
                  std::vector<NormEntry>& norm_buffer,
                  std::vector<uint32_t>& count_buffer) const;

    /**
     * @brief Proximity boost: adjust dists for the top-N candidates in this
     *        window by their term proximity. scored_candidates / position_lists
     *        are reused buffers hoisted out of the window loop by the caller.
     *        Query terms are resolved once via pos.ResolveTerm, then positions
     *        are fetched per candidate via pos.LookupPositions.
     *
     * @tparam PosBackend  see phrase_filter.
     */
    template <class PosBackend>
    void
    proximity_boost(float* dists,
                    const SparseTermComputerPtr& computer,
                    const PosBackend& pos,
                    std::vector<std::pair<float, uint32_t>>& scored_candidates,
                    std::vector<PosSpan>& position_lists,
                    const ProximityPhraseContext& ctx) const;

    /**
     * @brief Build the proximity/phrase context for a search, remapping phrase
     *        terms and capturing the (post-remap) query length. Shared by the
     *        KNN and RANGE dispatch sites so the mutable and immutable paths get
     *        identical derived values.
     */
    ProximityPhraseContext
    make_proximity_phrase_context(const SINDISearchParameter& search_param,
                                  uint32_t query_term_count) const;

    template <InnerSearchMode mode>
    DatasetPtr
    immutable_search_impl(const SparseTermComputerPtr& computer,
                          const InnerSearchParam& inner_param,
                          Allocator* allocator,
                          bool use_term_lists_heap_insert,
                          const SparseVector* original_query = nullptr,
                          const ProximityPhraseContext* ctx = nullptr) const;

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

    MetadataPtr
    collect_streaming_header() const override;

    void
    serialize_streaming_body(StreamWriter& writer) const override;

    void
    deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) override;

    void
    load_streaming_body(StreamReader& reader,
                        const MetadataPtr& metadata,
                        const LoadParameters& parameters) override;

    void
    read_streaming_body(StreamReader& reader, const MetadataPtr& metadata);

    void
    serialize_windows(StreamWriter& writer) const;

    void
    deserialize_windows(StreamReader& reader_ref);

    void
    deserialize_immutable_window(StreamReader& reader_ref, ImmutableSINDIWindow& window) const;

    void
    serialize_immutable_window(StreamWriter& writer, const ImmutableSINDIWindow& window) const;

    void
    serialize_immutable_window_positions(StreamWriter& writer,
                                         const ImmutableSINDIWindow& window) const;

    void
    deserialize_immutable_window_positions(StreamReader& reader_ref,
                                           ImmutableSINDIWindow& window) const;

    void
    compact_window_to_immutable(const SparseTermDataCell& term_list,
                                ImmutableSINDIWindow& window) const;

    std::vector<int64_t>
    build_immutable(const DatasetPtr& base);

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

    bool store_positions_{false};
    uint32_t max_positions_per_term_{64};
};

}  // namespace vsag
