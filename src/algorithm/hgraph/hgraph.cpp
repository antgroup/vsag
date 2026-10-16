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

#include "hgraph.h"

#include <datacell/compressed_graph_datacell_parameter.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "analyzer/hgraph_analyzer.h"
#include "attr/argparse.h"
#include "common.h"
#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/flatten_interface.h"
#include "datacell/graph_datacell.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/hgraph_rabitq_fused_datacell.h"
#include "datacell/rabitq_split_datacell.h"
#include "datacell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "impl/filter/filter_headers.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "impl/reasoning/reasoning_context.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/hgraph_rabitq_searcher.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "io/memory_block_io/memory_block_io.h"
#include "io/memory_block_io/memory_block_io_parameter.h"
#include "io/reader_io/reader_io_parameter.h"
#include "quantization/computer.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "quantization/scalar_quantization/scalar_quantizer_parameter.h"
#include "quantization/scalar_quantization/sq8_uniform_quantizer.h"
#include "quantization/scalar_quantization/sq8_uniform_quantizer_parameter.h"
#include "typing.h"
#include "utils/filter_search_skip_strategy.h"
#include "utils/linear_congruential_generator.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/filter.h"
#include "vsag/options.h"

namespace vsag {

class HGraphAnalyzer;

void
HGraph::check_fused_mutation_supported(std::string_view operation) const {
    if (this->rabitq_fused_datacell_ != nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            fmt::format("fused RaBitQ HGraph does not support {}", operation));
    }
}

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      cache_(std::make_unique<BuildCache>(common_param.allocator_.get())),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      reorder_by_base_(hgraph_param->reorder_source == HGRAPH_REORDER_SOURCE_BASE),
      ef_construct_(hgraph_param->ef_construction),
      alpha_(hgraph_param->alpha),
      duplicate_distance_threshold_(hgraph_param->duplicate_distance_threshold),
      support_force_remove_(hgraph_param->support_force_remove),
      odescent_param_(hgraph_param->odescent_param),
      pipnn_param_(hgraph_param->pipnn_param),
      graph_type_(hgraph_param->graph_type),
      hierarchical_datacell_param_(hgraph_param->hierarchical_graph_param),
      mci_parameters_(hgraph_param->mci_parameters),
      use_old_serial_format_(common_param.use_old_serial_format_),
      use_conjugate_graph_(hgraph_param->use_conjugate_graph) {
    this->support_duplicate_ = hgraph_param->support_duplicate;
    this->deduplicate_storage_ = hgraph_param->deduplicate_storage;
    const bool is_dense_vector = common_param.repr_ == RecordRepr::DENSE &&
                                 common_param.data_type_ != DataTypes::DATA_TYPE_SPARSE;
    if (this->deduplicate_storage_ && not is_dense_vector) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "HGraph deduplicate_storage only supports dense vectors");
    }
    if (this->graph_type_ == GRAPH_TYPE_VALUE_PIPNN) {
        const bool valid_adaptive_metric = not this->pipnn_param_.adaptive_pruning or
                                           common_param.metric_ == MetricType::METRIC_TYPE_L2SQR;
        CHECK_ARGUMENT(valid_adaptive_metric, "PiPNN adaptive pruning only supports L2");
        if (common_param.repr_ != RecordRepr::DENSE or
            common_param.data_type_ != DataTypes::DATA_TYPE_FLOAT) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "HGraph PiPNN only supports dense float32 indexes");
        }
    }
    if (this->deduplicate_storage_ && this->graph_type_ != GRAPH_TYPE_VALUE_NSW) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "HGraph deduplicate_storage only supports nsw graph");
    }
    this->persist_source_id_ = hgraph_param->persist_source_id;
    if (this->using_dedup_storage()) {
        this->code_slot_map_ = std::make_shared<CodeSlotMap>(allocator_);
    }
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (has_precise_reorder()) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);
    this->mci_searcher_ = std::make_shared<MCISearcher>(common_param);
    this->hybrid_mci_searcher_ = std::make_shared<HybridMCISearcher>(common_param);
    if (this->mci_parameters_.enabled) {
        this->mci_cliques_ = std::make_shared<CliqueDataCell>(common_param.allocator_.get());
    }
    if (this->use_conjugate_graph_) {
        this->conjugate_graph_ = std::make_shared<ConjugateGraph>(common_param.allocator_.get());
    }

    if (hgraph_param->rabitq_fused_datacell) {
        auto split_codes =
            std::dynamic_pointer_cast<RaBitQSplitDataCellInterface>(basic_flatten_codes_);
        CHECK_ARGUMENT(split_codes != nullptr,
                       "rabitq_fused_datacell requires in-memory RaBitQ split codes");
        rabitq_split_codes_ = split_codes;
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(hgraph_param->bottom_graph_param);
        CHECK_ARGUMENT(graph_param != nullptr, "rabitq_fused_datacell requires flat graph storage");
        rabitq_fused_searcher_ =
            std::make_shared<HGraphRaBitQSearcher>(common_param, neighbors_mutex_);
        rabitq_fused_datacell_ =
            std::make_shared<HGraphRaBitQFusedDataCell>(graph_param,
                                                        split_codes->OneBitCodeSize(),
                                                        split_codes->SupplementCodeSize(),
                                                        common_param);
        split_codes->AttachFusedCodeStorage(rabitq_fused_datacell_.get());
        this->bottom_graph_ = rabitq_fused_datacell_;
    } else {
        this->bottom_graph_ =
            GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    }
    if (this->support_duplicate_) {
        this->label_table_->SetDuplicateTracker(this->bottom_graph_->GetDuplicateTracker());
    }
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    this->resize_increase_count_bit_ = hgraph_param->resize_increase_count_bit;
    init_resize_bit_and_reorder();

    this->parallel_searcher_ =
        std::make_shared<ParallelSearcher>(common_param, thread_pool_, neighbors_mutex_);

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    check_and_init_raw_vector(hgraph_param->raw_vector_param, common_param);
    if (this->using_dedup_storage()) {
        this->basic_flatten_codes_ = MakeCodeSlotFlattenAdapter(
            this->basic_flatten_codes_, this->code_slot_map_, allocator_, &this->total_count_);
        if (this->high_precise_codes_ != nullptr) {
            this->high_precise_codes_ = MakeCodeSlotFlattenAdapter(
                this->high_precise_codes_, this->code_slot_map_, allocator_, &this->total_count_);
        }
        if (this->create_new_raw_vector_ && this->raw_vector_ != nullptr) {
            this->raw_vector_ = MakeCodeSlotFlattenAdapter(
                this->raw_vector_, this->code_slot_map_, allocator_, &this->total_count_);
        }
    }
    resize(bottom_graph_->max_capacity_);
    this->build_estimate_initial_capacity_ = this->max_capacity_.load();
    this->build_estimate_fresh_cache_buckets_ = this->cache_->neighbors_.bucket_count();
    if (common_param.thread_pool_ == nullptr and this->thread_pool_ != nullptr) {
        this->build_estimate_owned_pool_threads_ = std::max<uint64_t>(
            this->build_thread_count_, Options::Instance().num_threads_building());
    }
}

bool
HGraph::Tune(const std::string& parameters, bool disable_future_tuning) {
    // Invalidate before acquiring add_mutex_: even failed or count-zero Tune attempts
    // conservatively end eligibility. This does not permit concurrent capability queries.
    this->build_estimate_lifecycle_clean_.store(false, std::memory_order_release);
    this->check_fused_mutation_supported("Tune");
    std::scoped_lock lock(this->add_mutex_);
    if (this->immutable_.load(std::memory_order_acquire) or
        not this->index_feature_list_->CheckFeature(IndexFeature::SUPPORT_TUNE)) {
        return false;
    }

    // parse
    auto parsed_params = JsonType::Parse(parameters);
    JsonType hgraph_json;
    if (parsed_params.Contains(INDEX_PARAM)) {
        hgraph_json = parsed_params[INDEX_PARAM];
    }

    // map
    auto inner_json = map_hgraph_param(hgraph_json);
    ValidateMRLEDim(hgraph_json, this->dim_);
    const bool requires_raw_vector = RequiresRawVectorForMRLERaBitQSplit(inner_json);
    if (requires_raw_vector) {
        inner_json[STORE_RAW_VECTOR_KEY].SetBool(true);
    }

    // construct param obj
    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->FromJson(inner_json);
    auto inner_parameter = std::make_shared<InnerIndexParameter>();
    inner_parameter->FromJson(inner_json);

    // init new_basic_code obj
    auto common_param = this->basic_flatten_codes_->ExportCommonParam();
    auto new_basic_code =
        FlattenInterface::MakeInstance(hgraph_parameter->base_codes_param, common_param);
    FlattenInterfacePtr new_precise_code;
    const bool new_use_reorder = inner_parameter->use_reorder;
    const bool new_reorder_by_base = inner_parameter->reorder_source == HGRAPH_REORDER_SOURCE_BASE;
    const bool need_precise_codes = new_use_reorder and not new_reorder_by_base;
    if (need_precise_codes) {
        new_precise_code =
            FlattenInterface::MakeInstance(hgraph_parameter->precise_codes_param, common_param);
    }
    const auto current_count = total_count_.load(std::memory_order_acquire);
    auto covers_active_ids = [current_count](const FlattenInterfacePtr& codes) {
        return codes != nullptr and codes->TotalCount() >= current_count;
    };

    // Check which codes need to be rebuilt.
    bool is_tune_base_code = false;
    bool is_tune_precise_code = false;
    const bool is_tune_raw_code = requires_raw_vector and not covers_active_ids(raw_vector_);
    FlattenInterfacePtr new_raw_code;
    if (is_tune_raw_code) {
        new_raw_code =
            FlattenInterface::MakeInstance(hgraph_parameter->raw_vector_param, common_param);
    }
    const bool drop_precise_codes = not need_precise_codes;
    if (basic_flatten_codes_->GetQuantizerName() != new_basic_code->GetQuantizerName()) {
        is_tune_base_code = true;
    }
    if (need_precise_codes and
        (not covers_active_ids(high_precise_codes_) or
         high_precise_codes_->GetQuantizerName() != new_precise_code->GetQuantizerName())) {
        is_tune_precise_code = true;
    }

    FlattenInterfacePtr tune_source;
    if (is_tune_base_code or is_tune_precise_code or is_tune_raw_code) {
        if (covers_active_ids(raw_vector_)) {
            tune_source = raw_vector_;
        } else if (covers_active_ids(high_precise_codes_) and
                   high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
            tune_source = high_precise_codes_;
        } else if (covers_active_ids(basic_flatten_codes_) and
                   basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
            tune_source = basic_flatten_codes_;
        } else {
            return false;
        }
    }

    auto decode_tune_source = [&](InnerIdType inner_id, float* data) {
        auto buffer = tune_source->AcquireCodesById(inner_id);
        if (not buffer) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("failed to get vector by inner id {}", inner_id));
        }
        tune_source->Decode(buffer.Data(), data);
    };

    auto train_count = std::min(this->train_sample_count_, this->GetNumElements());
    Vector<float> train_data(train_count * dim_, 0, allocator_);
    if (is_tune_base_code or is_tune_precise_code or is_tune_raw_code) {
        for (InnerIdType i = 0; i < train_count; i++) {
            decode_tune_source(i, train_data.data() + i * dim_);
        }
    }

    auto wrap_new_code = [this](FlattenInterfacePtr code) -> FlattenInterfacePtr {
        if (not this->using_dedup_storage() or code == nullptr) {
            return code;
        }

        auto physical_capacity = this->physical_code_capacity_.load(std::memory_order_acquire);
        if (physical_capacity > 0) {
            code->Resize(physical_capacity);
        }
        return MakeCodeSlotFlattenAdapter(
            std::move(code), this->code_slot_map_, this->allocator_, &this->total_count_);
    };

    auto tune_and_rebuild =
        [&](bool need_tune, FlattenInterfacePtr old_code, FlattenInterfacePtr new_code) {
            if (not need_tune) {
                return old_code;
            }

            new_code = wrap_new_code(new_code);
            new_code->Train(train_data.data(), train_count);

            Vector<float> insert_buffer(dim_, 0, allocator_);
            for (int64_t i = 0; i < total_count_; ++i) {
                decode_tune_source(i, insert_buffer.data());
                new_code->InsertVector(static_cast<const void*>(insert_buffer.data()), i);
            }
            return new_code;
        };

    auto new_basic = tune_and_rebuild(is_tune_base_code, basic_flatten_codes_, new_basic_code);
    auto new_precise =
        tune_and_rebuild(is_tune_precise_code, high_precise_codes_, new_precise_code);
    auto new_raw = tune_and_rebuild(is_tune_raw_code, raw_vector_, new_raw_code);

    // Acquire exclusive global lock to atomically swap flatten codes,
    // preventing concurrent searches from accessing partially updated state.
    {
        std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
        auto param = std::dynamic_pointer_cast<HGraphParameter>(create_param_ptr_);
        basic_flatten_codes_ = new_basic;
        if (drop_precise_codes) {
            high_precise_codes_.reset();
            param->precise_codes_param.reset();
        } else {
            high_precise_codes_ = new_precise;
            if (is_tune_precise_code) {
                param->precise_codes_param = hgraph_parameter->precise_codes_param;
            }
        }
        if (is_tune_base_code) {
            param->base_codes_param = hgraph_parameter->base_codes_param;
        }
        use_reorder_ = new_use_reorder;
        reorder_by_base_ = new_reorder_by_base;
        param->use_reorder = new_use_reorder;
        param->reorder_source = inner_parameter->reorder_source;

        if (requires_raw_vector) {
            raw_vector_ = new_raw;
            has_raw_vector_ = true;
            create_new_raw_vector_ = true;
            param->store_raw_vector = true;
            param->raw_vector_param = hgraph_parameter->raw_vector_param;
        } else {
            check_and_init_raw_vector(param->raw_vector_param, common_param, false);
        }
        init_resize_bit_and_reorder();

        // set status
        if (disable_future_tuning) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_TUNE, false);
            if (not requires_raw_vector) {
                this->raw_vector_.reset();
                has_raw_vector_ = false;
                create_new_raw_vector_ = false;
            }
        }
    }
    return true;
}

namespace {

uint64_t
estimate_checked_mul(uint64_t lhs, uint64_t rhs) {
    const bool product_fits = lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs;
    CHECK_ARGUMENT(product_fits, "memory estimate overflows uint64_t");
    return lhs * rhs;
}

uint64_t
estimate_checked_add(uint64_t lhs, uint64_t rhs) {
    CHECK_ARGUMENT(rhs <= std::numeric_limits<uint64_t>::max() - lhs,
                   "memory estimate overflows uint64_t");
    return lhs + rhs;
}

// Requires positive block_size: Options rejects sizes below 256 KiB; modeled blocks are checked.
uint64_t
estimate_block_memory_ceil(uint64_t memory, uint64_t block_size) {
    const bool valid_size = block_size > 0;
    CHECK_ARGUMENT(valid_size, "memory estimate block size must be positive");
    const auto remainder = memory % block_size;
    if (remainder == 0) {
        return memory;
    }
    return estimate_checked_add(memory, block_size - remainder);
}

}  // namespace

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto add_block_memory = [&](uint64_t record_size) {
        const auto memory = estimate_checked_mul(record_size, element_count);
        estimate_memory =
            estimate_checked_add(estimate_memory, estimate_block_memory_ceil(memory, block_size));
    };

    if (this->rabitq_fused_datacell_ != nullptr) {
        add_block_memory(this->rabitq_fused_datacell_->RecordSize());
    } else {
        if (this->basic_flatten_codes_->InMemory()) {
            add_block_memory(this->basic_flatten_codes_->code_size_);
        }
        if (bottom_graph_->InMemory()) {
            const auto degree = static_cast<uint64_t>(this->bottom_graph_->maximum_degree_);
            add_block_memory(estimate_checked_mul(degree + 1, sizeof(InnerIdType)));
        }
    }

    if (has_precise_reorder() && this->high_precise_codes_->InMemory() &&
        not this->ignore_reorder_) {
        add_block_memory(this->high_precise_codes_->code_size_);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        add_block_memory(this->extra_infos_->ExtraInfoSize());
    }

    auto label_map_memory = estimate_checked_mul(
        element_count, sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory = estimate_checked_add(estimate_memory, label_map_memory);

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    const bool sparse_in_range = sparse_graph_memory >= 0.0 && sparse_graph_memory < 0x1p64;
    CHECK_ARGUMENT(sparse_in_range, "sparse graph memory estimate is out of uint64_t range");
    estimate_memory =
        estimate_checked_add(estimate_memory, static_cast<uint64_t>(sparse_graph_memory));

    auto other_memory = estimate_checked_mul(
        element_count,
        sizeof(LabelType) + sizeof(std::shared_mutex) + sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory = estimate_checked_add(estimate_memory, other_memory);

    if (this->mci_parameters_.enabled) {
        const auto mci_membership_memory = estimate_checked_mul(
            estimate_checked_mul(element_count, this->mci_parameters_.clique_max),
            2 * sizeof(InnerIdType));
        const auto mci_offset_memory = estimate_checked_mul(element_count, 2 * sizeof(InnerIdType));
        const auto mci_delta_row_memory =
            estimate_checked_mul(element_count, 2 * sizeof(Vector<InnerIdType>));
        estimate_memory = estimate_checked_add(estimate_memory, mci_membership_memory);
        estimate_memory = estimate_checked_add(estimate_memory, mci_offset_memory);
        estimate_memory = estimate_checked_add(estimate_memory, mci_delta_row_memory);
    }

    return estimate_memory;
}

namespace {

// Allocation/SBO rules are tied to the inspected headers, not ABI version alone.
#if defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION == 15007 || _LIBCPP_VERSION == 210106) && \
    defined(_LIBCPP_ABI_VERSION) && _LIBCPP_ABI_VERSION == 1 &&                            \
    !defined(_LIBCPP_ABI_OPTIMIZED_FUNCTION)
#define VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP 1
constexpr bool K_BUILD_ESTIMATE_STDLIB_MODELED = true;
#elif defined(__GLIBCXX__)
#define VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX 1
constexpr bool K_BUILD_ESTIMATE_STDLIB_MODELED = true;
#else
constexpr bool K_BUILD_ESTIMATE_STDLIB_MODELED = false;
#endif

// Five helpers below (estimate_shared_request, estimate_function_target_request,
// estimate_packaged_task_request, estimate_scheduler_task_request and
// estimate_sampled_dataset_request) static_assert their fallback #else, so a new modeled
// profile cannot silently return 0. estimate_deque_request has no separate fallback: its
// #else is the libstdc++ model. A new modeled profile must add branches to all six.

// The level replay keeps per-level counts in this fixed array, so replay storage does not
// depend on N. With the 1/ln 4 replay multiplier, finite draws stayed below about 15.8 in
// the G0 exhaustive probe, verified only for the recorded AppleClang/SDK 26.5 and
// GCC 11.4/glibc binaries; that observed bound is not universal and the cap stays 64.
// A draw at or beyond the cap is rejected with a typed error rather than indexing the
// array out of range. If the multiplier, distribution or engine changes, re-derive the
// bound before changing the cap.
constexpr uint64_t K_BUILD_ESTIMATE_MAX_ROUTE_LEVELS = 64;
// Distinct-key count up to which the tsl displacement limit cannot trigger an extra rehash.
constexpr uint64_t K_BUILD_ESTIMATE_NOMINAL_HASH_KEYS = 8193;
// ResourceObjectPool::kSubPoolCount and PointsMutex::kMutexesPerBlock.
constexpr uint64_t K_BUILD_ESTIMATE_SUB_POOLS = VisitedListPool::kSubPoolCount;
constexpr uint64_t K_BUILD_ESTIMATE_MUTEXES_PER_BLOCK = PointsMutex::kMutexesPerBlock;
// Simultaneous Acquire leases in FlattenDataCell::query (four-way unrolled loop).
constexpr uint64_t K_BUILD_ESTIMATE_QUERY_LEASES = 4;
// Distinct StandardHeap creation sites live in one insertion: search result and frontier,
// bottom probe filter, route filter, outer pruning queue, neighbor candidates and inner
// pruning queue.
constexpr uint64_t K_BUILD_ESTIMATE_HEAP_SITES = 7;
// libstdc++ _Prime_rehash_policy fast table ends at 13 buckets; libc++ stays below for the
// four keys a sampled DatasetImpl stores (num_elements, dim, f32_vectors, ids).
constexpr uint64_t K_BUILD_ESTIMATE_DATASET_KEYS = 4;
constexpr uint64_t K_BUILD_ESTIMATE_DATASET_BUCKETS = 13;
// ScalarQuantizationTrainer::MAX_DEFAULT_SAMPLE and sample_train_data's minimum.
constexpr uint64_t K_BUILD_ESTIMATE_TRAINER_SAMPLE_CAP = 100000;
constexpr uint64_t K_BUILD_ESTIMATE_MIN_TRAIN_SAMPLE = 512;

uint64_t
estimate_checked_sum(std::initializer_list<uint64_t> values) {
    uint64_t total = 0;
    for (auto value : values) {
        total = estimate_checked_add(total, value);
    }
    return total;
}

// Requires a positive divisor: callers use validated blocks, positive constants, or sizeof values.
uint64_t
estimate_ceil_div(uint64_t value, uint64_t divisor) {
    const bool valid_size = divisor > 0;
    CHECK_ARGUMENT(valid_size, "memory estimate divisor must be positive");
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

// Reallocation peak of a geometrically grown vector: retained old array plus new array.
uint64_t
estimate_grown_vector(uint64_t count, uint64_t element_size) {
    const bool valid_size = element_size > 0;
    CHECK_ARGUMENT(valid_size, "memory estimate element size must be positive");
    const uint64_t retained_count = estimate_checked_mul(count, 2);
    const uint64_t max_count =
        static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / element_size;
    CHECK_ARGUMENT(retained_count <= max_count, "memory estimate exceeds vector capacity");
    return estimate_checked_mul(estimate_checked_add(retained_count, count), element_size);
}

uint64_t
estimate_string_request(const std::string& value) {
    // +1 for the null terminator stored past capacity().
    return value.capacity() > std::string().capacity() ? estimate_checked_add(value.capacity(), 1)
                                                       : 0;
}

template <typename T>
constexpr uint64_t
estimate_shared_request() {
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
    return sizeof(std::__shared_ptr_emplace<T, std::allocator<T>>);
#elif defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    return sizeof(
        std::_Sp_counted_ptr_inplace<T, std::allocator<T>, __gnu_cxx::__default_lock_policy>);
#else
    static_assert(!K_BUILD_ESTIMATE_STDLIB_MODELED,
                  "estimate_shared_request: add the new stdlib profile");
    return 0;
#endif
}

// Heap target bytes of a std::function<void()> constructed from F (0 when stored inline).
template <typename F>
constexpr uint64_t
estimate_function_target_request() {
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
#if _LIBCPP_VERSION == 210106
    using Func = std::__function::__func<F, void()>;
#else
    using Func = std::__function::__func<F, std::allocator<F>, void()>;
#endif
    constexpr bool local =
        sizeof(Func) <= 3 * sizeof(void*) && std::is_nothrow_copy_constructible<F>::value;
    return local ? 0 : sizeof(Func);
#elif defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    constexpr bool local = std::__is_location_invariant<F>::value &&
                           sizeof(F) <= std::_Function_base::_M_max_size &&
                           alignof(F) <= std::_Function_base::_M_max_align &&
                           std::_Function_base::_M_max_align % alignof(F) == 0;
    return local ? 0 : sizeof(F);
#else
    static_assert(!K_BUILD_ESTIMATE_STDLIB_MODELED,
                  "estimate_function_target_request: add the new stdlib profile");
    return 0;
#endif
}

// make_shared<packaged_task<void()>>(Fn) plus the task's callable and shared state requests.
template <typename Fn>
constexpr uint64_t
estimate_packaged_task_request() {
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
    using Callable = std::__packaged_task_func<Fn, std::allocator<Fn>, void()>;
    constexpr uint64_t callable = sizeof(Callable) <= 3 * sizeof(void*) ? 0 : sizeof(Callable);
    // Apple21's out-of-line promise<void> constructor was checked against this SDK/runtime:
    // one 144-byte request, matching this state size, with no get_future request. LLVM15's
    // inspected headers retain the same source allocation model for formatting/tidy.
    return estimate_shared_request<std::packaged_task<void()>>() + callable +
           sizeof(std::__assoc_sub_state);
#elif defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    // libstdc++ __create_task_state defaults to allocator<int> and rebinds it for State;
    // retain that allocator type when modeling the task and its shared control block.
    using State = std::__future_base::_Task_state<Fn, std::allocator<int>, void()>;
    return estimate_shared_request<std::packaged_task<void()>>() +
           sizeof(std::_Sp_counted_ptr_inplace<State,
                                               std::allocator<int>,
                                               __gnu_cxx::__default_lock_policy>) +
           sizeof(std::__future_base::_Result<void>);
#else
    static_assert(!K_BUILD_ESTIMATE_STDLIB_MODELED,
                  "estimate_packaged_task_request: add the new stdlib profile");
    return 0;
#endif
}

// Deque block and map envelope for the given occupancy, including map reallocation overlap.
// Requires positive element_size; callers pass sizeof values, keeping per_block positive.
uint64_t
estimate_deque_request(uint64_t element_size, uint64_t occupancy) {
    const bool valid_size = element_size > 0;
    CHECK_ARGUMENT(valid_size, "memory estimate element size must be positive");
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
    const uint64_t per_block = element_size < 256 ? 4096 / element_size : 16;
    // libc++ may retain a front spare block; ceil(Q/B) + 2 covers it and partial end blocks.
    const uint64_t blocks = estimate_checked_add(estimate_ceil_div(occupancy, per_block), 2);
    const uint64_t map_slots = std::max<uint64_t>(1, estimate_checked_mul(2, blocks));
#else
    const uint64_t per_block = element_size < 512 ? 512 / element_size : 1;
    // libstdc++ releases consumed front blocks. With start offset s < B, live nodes are
    // floor((s + Q) / B) + 1 <= floor(Q / B) + 2, including the finish node.
    const uint64_t blocks = estimate_checked_add(occupancy / per_block, 2);
    const uint64_t map_slots =
        std::max<uint64_t>(8, estimate_checked_add(estimate_checked_mul(4, blocks), 2));
#endif
    return estimate_checked_add(
        estimate_checked_mul(estimate_checked_mul(blocks, per_block), element_size),
        estimate_checked_mul(map_slots, 2 * sizeof(void*)));
}

// Source capture storage bounds for the inspected LP64 targets: one shared_ptr or one
// std::function member. These are not claims about arbitrary compiler closure layouts.
struct BuildEstimateTaskInvoker {
    std::shared_ptr<std::packaged_task<void()>> task;
    void
    operator()() const {
    }
};

struct BuildEstimateSafeWrapper {
    std::function<void()> task;
    void
    operator()() const {
    }
};

// Target storage bound for ResourceObjectPool's pool-pointer and two constructor arguments.
struct BuildEstimatePoolConstructor {
    void* pool;
    uint64_t max_size;
    Allocator* allocator;
    void
    operator()() const {
    }
};

// One known SafeThreadPool over the built-in DefaultThreadPool (wrapper depth one).
template <typename Worker>
constexpr uint64_t
estimate_scheduler_task_request() {
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
    // bind(F&&) returns __bind<F>; the backend forwards its function as an lvalue.
    using Bound = std::__bind<Worker>;
    using BackendBound = std::__bind<std::function<void()>&>;
#elif defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    // libstdc++'s _Bind_helper decays F before forming the callable signature.
    using Bound = std::_Bind<Worker()>;
    using BackendBound = std::_Bind<std::function<void()>()>;
#endif
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP) || defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    constexpr uint64_t queued = estimate_function_target_request<BuildEstimateTaskInvoker>();
    constexpr uint64_t wrapper =
        estimate_function_target_request<BuildEstimateSafeWrapper>() + queued;
    return estimate_packaged_task_request<Bound>() + queued + 2 * wrapper +
           estimate_packaged_task_request<BackendBound>() + queued;
#else
    static_assert(!K_BUILD_ESTIMATE_STDLIB_MODELED,
                  "estimate_scheduler_task_request: add the new stdlib profile");
    return 0;
#endif
}

using BuildEstimateDatasetValue = std::variant<int64_t,
                                               const float*,
                                               const char*,
                                               const int8_t*,
                                               const uint16_t*,
                                               const int64_t*,
                                               const std::string*,
                                               const SparseVector*,
                                               const AttributeSet*,
                                               const uint32_t*,
                                               const MultiVector*,
                                               std::vector<std::vector<std::string>>>;

uint64_t
estimate_sampled_dataset_request() {
    using Value = std::pair<const std::string, BuildEstimateDatasetValue>;
#if defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBCPP)
    using Node =
        std::__hash_node<std::__hash_value_type<std::string, BuildEstimateDatasetValue>, void*>;
    constexpr uint64_t node = sizeof(Node);
#elif defined(VSAG_HGRAPH_BUILD_ESTIMATE_LIBSTDCXX)
    using Node =
        std::__detail::_Hash_node<Value,
                                  std::__cache_default<std::string, std::hash<std::string>>::value>;
    constexpr uint64_t node = sizeof(Node);
#else
    static_assert(!K_BUILD_ESTIMATE_STDLIB_MODELED,
                  "estimate_sampled_dataset_request: add the new stdlib profile");
    constexpr uint64_t node = 0;
#endif
    return estimate_shared_request<DatasetImpl>() + K_BUILD_ESTIMATE_DATASET_KEYS * node +
           2 * K_BUILD_ESTIMATE_DATASET_BUCKETS * sizeof(void*);
}

struct BuildEstimateTable {
    uint64_t buckets{0};
    uint64_t previous{0};
};

// tsl prime growth policy: the first prime whose load threshold admits every key.
// The table itself is tsl::rh::detail::PRIMES from the pinned tsl in extern/tsl/tsl.cmake;
// a rename fails at compile time, and when that happens the prime and 0.75-threshold
// model must be re-derived.
BuildEstimateTable
estimate_prime_table(uint64_t keys, float max_load) {
    if (keys == 0) {
        return {};
    }
    const auto& primes = tsl::rh::detail::PRIMES;
    for (uint64_t i = 1; i < primes.size(); ++i) {
        const auto threshold = static_cast<uint64_t>(static_cast<float>(primes[i]) * max_load);
        if (threshold >= keys) {
            return {primes[i], i >= 2 ? primes[i - 1] : 0};
        }
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT, "label map estimate exceeds tsl capacity");
}

// tsl power-of-two growth from an empty table at the default 0.5 load factor.
BuildEstimateTable
estimate_power_of_two_table(uint64_t keys) {
    if (keys == 0) {
        return {};
    }
    uint64_t buckets = 2;
    while (static_cast<uint64_t>(static_cast<float>(buckets) * 0.5F) < keys) {
        buckets = estimate_checked_mul(buckets, 2);
    }
    return {buckets, buckets > 2 ? buckets / 2 : 0};
}

uint64_t
estimate_next_power_of_two(uint64_t value) {
    uint64_t result = 1;
    while (result < value) {
        result = estimate_checked_mul(result, 2);
    }
    return result;
}

template <MetricType metric>
uint64_t
estimate_flatten_fixed_request() {
    using Layout = FixedLayout<MemoryBlockIO>;
    const uint64_t flatten =
        std::max({estimate_shared_request<FlattenDataCell<FP32Quantizer<metric>, Layout>>(),
                  estimate_shared_request<FlattenDataCell<SQ8Quantizer<metric>, Layout>>(),
                  estimate_shared_request<FlattenDataCell<SQ8UniformQuantizer<metric>, Layout>>()});
    const uint64_t quantizer = std::max({estimate_shared_request<FP32Quantizer<metric>>(),
                                         estimate_shared_request<SQ8Quantizer<metric>>(),
                                         estimate_shared_request<SQ8UniformQuantizer<metric>>()});
    return flatten + quantizer + estimate_shared_request<Layout>() +
           estimate_shared_request<MemoryBlockIO>();
}

template <MetricType metric>
uint64_t
estimate_computer_request() {
    return std::max({estimate_shared_request<Computer<FP32Quantizer<metric>>>(),
                     estimate_shared_request<Computer<SQ8Quantizer<metric>>>(),
                     estimate_shared_request<Computer<SQ8UniformQuantizer<metric>>>()});
}

bool
estimate_is_block_memory_io(const IOParamPtr& param) {
    return param != nullptr and
           std::dynamic_pointer_cast<MemoryBlockIOParameter>(param) != nullptr and
           std::dynamic_pointer_cast<MemoryBlockIOParameter>(param)->block_size_ > 0;
}

bool
estimate_is_modeled_quantizer_param(const QuantizerParamPtr& param) {
    return std::dynamic_pointer_cast<FP32QuantizerParameter>(param) != nullptr or
           std::dynamic_pointer_cast<ScalarQuantizerParameter<8>>(param) != nullptr or
           std::dynamic_pointer_cast<SQ8UniformQuantizerParameter>(param) != nullptr;
}

}  // namespace

double
HGraph::random_level_draw(std::default_random_engine& engine, double mult) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double probability = distribution(engine);
    const bool valid_probability =
        std::isfinite(probability) and probability > 0.0 and probability <= 1.0;
    CHECK_ARGUMENT(valid_probability, "HGraph random level probability is not representable");
    const double level = -log(probability) * mult;
    const bool valid_level = std::isfinite(level) and level >= 0.0 and
                             level < static_cast<double>(std::numeric_limits<int>::max()) + 1.0;
    CHECK_ARGUMENT(valid_level, "HGraph random level is not representable");
    return level;
}

void
HGraph::validate_random_levels(uint64_t count) const {
    // O(count) time, constant storage; advances a copy only. No per-row storage or live
    // RNG advance. Duplicate labels only shorten the consumed
    // prefix, so checking all input rows is conservative under exceptional RNG conditions.
    auto engine = this->level_generator_;
    for (uint64_t row = 0; row < count; ++row) {
        (void)random_level_draw(engine, this->mult_);
    }
}

uint64_t
HGraph::build_estimate_checked_capacity(uint64_t num_elements, uint64_t resize_bit) {
    const bool valid_bit = resize_bit >= 1 && resize_bit <= 30;
    CHECK_ARGUMENT(valid_bit, "build memory estimate requires resize bit in [1, 30]");
    const uint64_t max_elements = (uint64_t{1} << 31) - (uint64_t{1} << resize_bit);
    CHECK_ARGUMENT(num_elements <= max_elements,
                   fmt::format("build memory estimate supports at most {} elements, got {}",
                               max_elements,
                               num_elements));
    return next_multiple_of_power_of_two(num_elements, resize_bit);
}

bool
HGraph::build_memory_estimate_index_supported(const HGraphParameter& param) const {
    const bool modeled_index =
        this->data_type_ == DataTypes::DATA_TYPE_FLOAT and
        this->graph_type_ == GRAPH_TYPE_VALUE_NSW and param.graph_type == GRAPH_TYPE_VALUE_NSW and
        (this->metric_ == MetricType::METRIC_TYPE_L2SQR or
         this->metric_ == MetricType::METRIC_TYPE_IP or
         this->metric_ == MetricType::METRIC_TYPE_COSINE) and
        this->rabitq_fused_datacell_ == nullptr and not param.rabitq_fused_datacell and
        not this->mci_parameters_.enabled and not this->use_elp_optimizer_ and
        not this->support_duplicate_ and not this->deduplicate_storage_ and
        not this->support_force_remove_ and not this->use_conjugate_graph_ and
        not this->use_attribute_filter_ and this->extra_info_size_ == 0 and
        this->extra_infos_ == nullptr and not this->use_reorder_ and
        not this->has_precise_reorder() and this->high_precise_codes_ == nullptr and
        this->raw_vector_ == nullptr and not this->create_new_raw_vector_ and
        this->reorder_ == nullptr and this->optimizer_ == nullptr and
        this->code_slot_map_ == nullptr and this->optimized_build_codes_ == nullptr and
        this->conjugate_graph_ == nullptr and this->mci_cliques_ == nullptr and
        param.label_remap_type == LabelRemapType::PG and param.precise_codes_param == nullptr and
        param.raw_vector_param == nullptr and param.attr_inverted_interface_param == nullptr and
        param.odescent_param == nullptr;
    return modeled_index;
}

bool
HGraph::build_memory_estimate_codec_supported(const HGraphParameter& param) const {
    const auto codes_param =
        std::dynamic_pointer_cast<FlattenDataCellParameter>(param.base_codes_param);
    const bool modeled_codes =
        codes_param != nullptr and codes_param->name == FLATTEN_DATA_CELL and
        estimate_is_modeled_quantizer_param(codes_param->quantizer_parameter) and
        estimate_is_block_memory_io(codes_param->io_parameter) and
        codes_param->supplement_io_parameter == nullptr and
        this->basic_flatten_codes_ != nullptr and this->basic_flatten_codes_->InMemory() and
        (this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32 or
         this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SQ8 or
         this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM);
    const auto graph_param =
        std::dynamic_pointer_cast<GraphDataCellParameter>(param.bottom_graph_param);
    const auto route_param = param.hierarchical_graph_param;
    const bool modeled_graph =
        graph_param != nullptr and estimate_is_block_memory_io(graph_param->io_parameter_) and
        not graph_param->support_remove_ and not graph_param->use_reverse_edges_ and
        not graph_param->support_duplicate_ and
        std::dynamic_pointer_cast<GraphDataCell<MemoryBlockIO>>(this->bottom_graph_) != nullptr and
        route_param != nullptr and route_param == this->hierarchical_datacell_param_ and
        not route_param->support_delete_ and not route_param->use_reverse_edges_ and
        not route_param->support_duplicate_;
    const bool modeled_extra_info =
        param.extra_info_param == nullptr or
        estimate_is_block_memory_io(param.extra_info_param->io_parameter);
    return modeled_codes and modeled_graph and modeled_extra_info;
}

bool
HGraph::build_memory_estimate_state_supported() const {
    const auto initial_capacity = this->build_estimate_initial_capacity_;
    const bool valid_capacity =
        this->resize_increase_count_bit_ >= 1 and this->resize_increase_count_bit_ <= 30 and
        initial_capacity > 0 and
        initial_capacity <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    // The model assumes a fresh cache: source_ids_ never allocated (capacity 0) and the
    // neighbor-map bucket count equal to the fresh count recorded at construction
    // (build_estimate_fresh_cache_buckets_). A cleared cache keeps its capacity.
    const bool pristine_cache =
        this->cache_ != nullptr and this->cache_->source_ids_.empty() and
        this->cache_->source_ids_.capacity() == 0 and this->cache_->neighbors_.empty() and
        this->cache_->neighbors_.bucket_count() == this->build_estimate_fresh_cache_buckets_;
    return valid_capacity and pristine_cache and
           this->build_estimate_lifecycle_clean_.load(std::memory_order_acquire) and
           not this->immutable_.load(std::memory_order_acquire) and
           this->total_count_.load(std::memory_order_acquire) == 0 and
           static_cast<uint64_t>(this->max_capacity_.load(std::memory_order_acquire)) ==
               initial_capacity;
}

bool
HGraph::build_memory_estimate_supported() const {
    if (not K_BUILD_ESTIMATE_STDLIB_MODELED) {
        return false;
    }
    const auto param = std::dynamic_pointer_cast<HGraphParameter>(this->create_param_ptr_);
    if (param == nullptr) {
        return false;
    }
    return this->build_memory_estimate_index_supported(*param) and
           this->build_memory_estimate_codec_supported(*param) and
           this->build_memory_estimate_state_supported();
}

bool
HGraph::CheckFeature(IndexFeature feature) const {
    if (feature == IndexFeature::SUPPORT_ESTIMATE_BUILD_MEMORY) {
        return this->build_memory_estimate_supported();
    }
    return InnerIndexInterface::CheckFeature(feature);
}

// Per-call fixed storage only: no HGraph object members or proportional allocations.
// Stages follow the original evaluation order, including checked arithmetic and replay.
struct HGraph::BuildMemoryEstimateContext {
    const HGraph& index;
    uint64_t n{};
    uint64_t c0{};
    uint64_t c{};
    uint64_t steady{};
    std::shared_ptr<HGraphParameter> param{};
    std::shared_ptr<FlattenDataCellParameter> codes_param{};
    std::shared_ptr<GraphDataCellParameter> graph_param{};
    std::shared_ptr<MemoryBlockIOParameter> code_io{};
    std::shared_ptr<MemoryBlockIOParameter> graph_io{};
    uint64_t code_block{};
    uint64_t graph_block{};
    uint64_t dim{};
    uint64_t code_size{};
    uint64_t degree{};
    uint64_t route_degree{};
    uint64_t graph_line{};
    uint64_t ef{};
    bool nonempty{};
    bool has_pool{};
    uint64_t workers{};
    uint64_t visited_lists{};
    uint64_t sample_floor{};
    uint64_t sample{};
    uint64_t quant_sample{};
    std::array<uint64_t, K_BUILD_ESTIMATE_MAX_ROUTE_LEVELS> route_counts{};
    uint64_t route_levels{0};
    uint64_t route_headers{0};
    bool pool_replaced{false};

    explicit BuildMemoryEstimateContext(const HGraph& index_ref, uint64_t num_elements)
        : index(index_ref) {
        n = num_elements;
        c0 = index.build_estimate_initial_capacity_;
        c = std::max(c0,
                     HGraph::build_estimate_checked_capacity(num_elements,
                                                             index.resize_increase_count_bit_));
        steady = index.EstimateMemory(num_elements);

        // Sole caller EstimateBuildMemory validated these parameter types before construction.
        // That validation remains valid under the required absence of concurrent mutation.
        param = std::dynamic_pointer_cast<HGraphParameter>(index.create_param_ptr_);
        if (param == nullptr) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "HGraph parameter types do not support EstimateBuildMemory");
        }
        codes_param = std::dynamic_pointer_cast<FlattenDataCellParameter>(param->base_codes_param);
        graph_param = std::dynamic_pointer_cast<GraphDataCellParameter>(param->bottom_graph_param);
        if (codes_param == nullptr or graph_param == nullptr) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "HGraph parameter types do not support EstimateBuildMemory");
        }
        code_io = std::dynamic_pointer_cast<MemoryBlockIOParameter>(codes_param->io_parameter);
        graph_io = std::dynamic_pointer_cast<MemoryBlockIOParameter>(graph_param->io_parameter_);
        if (code_io == nullptr or graph_io == nullptr) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "HGraph parameter types do not support EstimateBuildMemory");
        }
        code_block = MemoryBlockIOParameter::NearestPowerOfTwo(code_io->block_size_);
        graph_block = MemoryBlockIOParameter::NearestPowerOfTwo(graph_io->block_size_);

        dim = static_cast<uint64_t>(index.dim_);
        code_size = index.basic_flatten_codes_->code_size_;
        degree = static_cast<uint64_t>(index.bottom_graph_->MaximumDegree());
        route_degree = static_cast<uint64_t>(index.hierarchical_datacell_param_->max_degree_);
        graph_line = estimate_checked_mul(estimate_checked_add(degree, 1), sizeof(InnerIdType));
        ef = index.ef_construct_;
        nonempty = n > 0;
        has_pool = index.thread_pool_ != nullptr;
        workers = insert_worker_count(has_pool, index.build_thread_count_, n);
        visited_lists = nonempty ? estimate_checked_mul(K_BUILD_ESTIMATE_SUB_POOLS,
                                                        std::max<uint64_t>(1, workers))
                                 : 1;
        sample_floor = static_cast<uint64_t>(std::max<int64_t>(
            static_cast<int64_t>(K_BUILD_ESTIMATE_MIN_TRAIN_SAMPLE), index.train_sample_count_));
        sample = std::min(n, sample_floor);
        quant_sample = std::min(sample, K_BUILD_ESTIMATE_TRAINER_SAMPLE_CAP);
    }

    void
    ReplayRoutes() {
        // Degree-4 level replay through the Build draw expression; fixed storage, O(N) time.
        // Build uses the configured degree; the minimum supported degree (4) conservatively
        // covers route levels for larger configured degrees.
        route_counts = {};
        route_levels = 0;
        {
            // Copying preserves index RNG state; it provides no synchronization.
            auto engine = index.level_generator_;
            const double mult = 1 / log(1.0 * static_cast<double>(4));
            for (uint64_t i = 0; i < n; ++i) {
                const double draw = HGraph::random_level_draw(engine, mult);
                const bool representable =
                    std::isfinite(draw) and draw >= 0.0 and
                    draw < static_cast<double>(K_BUILD_ESTIMATE_MAX_ROUTE_LEVELS);
                if (not representable) {
                    throw VsagException(ErrorType::INVALID_ARGUMENT,
                                        "build memory estimate level replay is not representable");
                }
                const auto levels = static_cast<uint64_t>(draw);
                if (levels > 0) {
                    ++route_counts[levels - 1];
                    route_levels = std::max(route_levels, levels);
                }
            }
            for (uint64_t level = K_BUILD_ESTIMATE_MAX_ROUTE_LEVELS - 1; level > 0; --level) {
                route_counts[level - 1] =
                    estimate_checked_add(route_counts[level - 1], route_counts[level]);
            }
        }
    }

    void
    RouteAllocations(uint64_t& route_resident, uint64_t& route_transient) {
        using RouteBucket = tsl::detail_robin_hash::
            bucket_entry<std::pair<InnerIdType, std::unique_ptr<Vector<InnerIdType>>>, false>;
        route_headers = 0;
        for (uint64_t level = 0; level < route_levels; ++level) {
            const uint64_t keys = route_counts[level];
            uint64_t buckets = estimate_power_of_two_table(keys).buckets;
            if (keys > K_BUILD_ESTIMATE_NOMINAL_HASH_KEYS) {
                buckets = std::max(buckets, estimate_next_power_of_two(estimate_checked_mul(2, c)));
            }
            const uint64_t bucket_bytes = estimate_checked_mul(buckets, sizeof(RouteBucket));
            const uint64_t neighbor_bytes = estimate_checked_mul(
                estimate_checked_mul(keys, estimate_checked_mul(2, route_degree)),
                sizeof(InnerIdType));
            route_resident = estimate_checked_sum({route_resident, bucket_bytes, neighbor_bytes});
            route_transient = estimate_checked_add(route_transient, bucket_bytes);
            route_headers =
                estimate_checked_sum({route_headers,
                                      estimate_shared_request<SparseGraphDataCell>(),
                                      estimate_checked_mul(keys, sizeof(Vector<InnerIdType>))});
        }
    }

    [[nodiscard]] uint64_t
    WorkerScratch() const {
        // Per-worker allocator scratch: one insertion's search, filter, pruning and IO sites.
        const uint64_t record = sizeof(std::pair<float, InnerIdType>);
        const uint64_t worker_scratch = estimate_checked_sum({
            estimate_grown_vector(c, record),  // visited-guarded frontier is not capped by ef
            estimate_checked_mul(3, estimate_grown_vector(estimate_checked_add(ef, 1), record)),
            estimate_grown_vector(ef, record),
            estimate_checked_mul(2, estimate_grown_vector(estimate_checked_add(degree, 1), record)),
            estimate_checked_mul(
                2, estimate_grown_vector(degree, sizeof(std::pair<float, InnerIdType>))),
            estimate_checked_mul(estimate_checked_mul(4, degree), sizeof(InnerIdType)),
            estimate_checked_mul(degree, sizeof(InnerIdType)),
            estimate_checked_mul(2, estimate_grown_vector(degree, sizeof(InnerIdType))),
            estimate_checked_mul(estimate_checked_mul(3, route_degree), sizeof(InnerIdType)),
            std::max(estimate_checked_mul(dim, sizeof(float)), code_size),
            estimate_checked_mul(K_BUILD_ESTIMATE_QUERY_LEASES, code_size),
            code_size,
        });

        return worker_scratch;
    }

    uint64_t
    AllocatorPayload() {
        // Allocator-routed resident (R_A) and transient/overlap (T_A) payloads.
        const uint64_t code_bytes = estimate_checked_mul(c, code_size);
        const uint64_t graph_bytes = estimate_checked_mul(c, graph_line);
        const uint64_t code_blocks = estimate_ceil_div(code_bytes, code_block);
        const uint64_t graph_blocks = estimate_ceil_div(graph_bytes, graph_block);
        const uint64_t block_pointers =
            estimate_checked_mul(estimate_checked_add(code_blocks, graph_blocks), sizeof(uint8_t*));
        const uint64_t label_bytes = estimate_checked_mul(c, sizeof(LabelType));
        using LabelBucket =
            tsl::detail_robin_hash::bucket_entry<std::pair<LabelType, InnerIdType>, false>;
        using PaddingBucket = tsl::detail_robin_hash::bucket_entry<InnerIdType, false>;
        const auto label_table = estimate_prime_table(n, 0.75F);
        const uint64_t mutex_blocks = estimate_ceil_div(c, K_BUILD_ESTIMATE_MUTEXES_PER_BLOCK);
        const uint64_t mutex_block_bytes =
            K_BUILD_ESTIMATE_MUTEXES_PER_BLOCK * sizeof(std::shared_mutex);
        // Each MutexBlockPtr owns a block pointer and an allocator-pointer deleter
        // on these targets.
        const uint64_t mutex_pointers = estimate_checked_mul(mutex_blocks, 2 * sizeof(void*));
        const uint64_t visited_list_bytes =
            estimate_checked_mul(estimate_ceil_div(c, VisitedList::kBitsPerWord),
                                 sizeof(VisitedList::WordType) + sizeof(VisitedList::TagType));
        const uint64_t visited_deques =
            estimate_checked_mul(K_BUILD_ESTIMATE_SUB_POOLS,
                                 estimate_deque_request(sizeof(std::shared_ptr<VisitedList>),
                                                        std::max<uint64_t>(1, workers)));
        pool_replaced = c > c0;

        uint64_t route_resident = 0;
        uint64_t route_transient = 0;
        RouteAllocations(route_resident, route_transient);
        const uint64_t route_pointer_vector =
            estimate_grown_vector(route_levels, sizeof(GraphInterfacePtr));

        const uint64_t resident_payload = estimate_checked_sum({
            estimate_block_memory_ceil(code_bytes, code_block),
            estimate_block_memory_ceil(graph_bytes, graph_block),
            estimate_checked_mul(block_pointers, 2),
            estimate_checked_mul(label_bytes, 2),
            estimate_checked_mul(label_table.buckets, sizeof(LabelBucket)),
            estimate_checked_mul(mutex_blocks, mutex_block_bytes),
            mutex_pointers,
            estimate_checked_mul(visited_lists, visited_list_bytes),
            visited_deques,
            route_resident,
            nonempty ? 2 * sizeof(PaddingBucket) : 0,
        });

        const uint64_t worker_scratch = WorkerScratch();

        const uint64_t transient_payload = estimate_checked_sum({
            block_pointers,
            label_bytes,
            estimate_checked_mul(label_table.previous, sizeof(LabelBucket)),
            mutex_pointers,
            pool_replaced ? visited_list_bytes : 0,
            pool_replaced ? estimate_checked_mul(
                                K_BUILD_ESTIMATE_SUB_POOLS,
                                estimate_deque_request(sizeof(std::shared_ptr<VisitedList>), 1))
                          : 0,
            route_transient,
            route_pointer_vector,
            nonempty
                ? estimate_checked_add(estimate_checked_mul(estimate_checked_mul(8, sample), dim),
                                       estimate_checked_mul(24, sample))
                : 0,
            estimate_grown_vector(n, sizeof(AddRow)),
            nonempty ? sizeof(InnerIdType) : 0,
            estimate_checked_mul(workers, worker_scratch),
        });
        const uint64_t allocator_model =
            estimate_checked_add(std::max(steady, resident_payload), transient_payload);

        return allocator_model;
    }

    [[nodiscard]] uint64_t
    FixedObjects() const {
        // Standard-library allocations (G_model).
        uint64_t fixed_objects = estimate_checked_sum({
            estimate_shared_request<IndexImpl<HGraph>>(),
            estimate_shared_request<HGraph>(),
            estimate_shared_request<LabelTable>(),
            sizeof(PGUnorderedMap<LabelType, InnerIdType>),
            sizeof(IndexFeatureList),
            (static_cast<uint64_t>(IndexFeature::INDEX_FEATURE_COUNT) + 7) / 8,
            sizeof(BuildCache),
            estimate_shared_request<PointsMutex>(),
            estimate_shared_request<BasicSearcher>(),
            estimate_shared_request<MCISearcher>(),
            estimate_shared_request<HybridMCISearcher>(),
            estimate_shared_request<ParallelSearcher>(),
            estimate_shared_request<GraphDataCell<MemoryBlockIO>>(),
            estimate_shared_request<MemoryBlockIO>(),
            estimate_shared_request<HGraphParameter>(),
            estimate_shared_request<FlattenDataCellParameter>(),
            std::max({estimate_shared_request<FP32QuantizerParameter>(),
                      estimate_shared_request<ScalarQuantizerParameter<8>>(),
                      estimate_shared_request<SQ8UniformQuantizerParameter>()}),
            estimate_shared_request<GraphDataCellParameter>(),
            estimate_shared_request<SparseGraphDatacellParameter>(),
            2 * estimate_shared_request<MemoryBlockIOParameter>(),
            estimate_string_request(param->graph_type),
            estimate_string_request(param->name),
            estimate_string_request(param->reorder_source),
            estimate_string_request(param->mci_parameters.knng_source),
            estimate_string_request(param->mci_parameters.knng_path),
            estimate_checked_mul(param->pipnn_param.fanout.capacity(), sizeof(uint64_t)),
            estimate_string_request(codes_param->name),
            estimate_string_request(codes_param->quantizer_parameter->GetTypeName()),
            estimate_string_request(code_io->GetTypeName()),
            estimate_string_request(graph_io->GetTypeName()),
            estimate_string_request(index.graph_type_),
            estimate_string_request(index.mci_parameters_.knng_source),
            estimate_string_request(index.mci_parameters_.knng_path),
            estimate_checked_mul(index.pipnn_param_.fanout.capacity(), sizeof(uint64_t)),
        });
        {
            // RemoveListFilter is file-local: Filter base followed by two reference members.
            constexpr uint64_t filter_align = std::max(alignof(Filter), alignof(void*));
            struct alignas(filter_align) FilterLayoutBound {
                unsigned char
                    bytes[(sizeof(Filter) + filter_align - 1) / filter_align * filter_align +
                          2 * sizeof(void*)];
            };
            fixed_objects =
                estimate_checked_add(fixed_objects, estimate_shared_request<FilterLayoutBound>());
        }
        if (param->extra_info_param != nullptr) {
            fixed_objects = estimate_checked_sum(
                {fixed_objects,
                 estimate_shared_request<ExtraInfoDataCellParameter>(),
                 estimate_shared_request<MemoryBlockIOParameter>(),
                 estimate_string_request(param->extra_info_param->io_parameter->GetTypeName())});
        }
        if (index.build_estimate_owned_pool_threads_ > 0) {
            const uint64_t threads =
                std::max(index.build_estimate_owned_pool_threads_, index.build_thread_count_);
            fixed_objects = estimate_checked_sum(
                {fixed_objects,
                 estimate_shared_request<SafeThreadPool>(),
                 sizeof(DefaultThreadPool),
                 sizeof(progschj::ThreadPool),
                 estimate_grown_vector(threads, sizeof(std::thread)),
                 nonempty ? 0 : estimate_deque_request(sizeof(std::function<void()>), 0)});
        }

        return fixed_objects;
    }

    static uint64_t
    WorkerObjects(uint64_t computer_request) {
        // The two implementation-local skip strategies derive from this complete base. Preserve
        // its padding, then bound either two doubles or an aligned generator followed by a double.
        constexpr uint64_t strategy_align = std::max({alignof(FilterSearchSkipStrategy),
                                                      alignof(LinearCongruentialGenerator),
                                                      alignof(double)});
        constexpr uint64_t strategy_base = (sizeof(FilterSearchSkipStrategy) + strategy_align - 1) /
                                           strategy_align * strategy_align;
        constexpr uint64_t generator_slot =
            (sizeof(LinearCongruentialGenerator) + strategy_align - 1) / strategy_align *
            strategy_align;
        constexpr uint64_t strategy_fields =
            std::max<uint64_t>(2 * sizeof(double), generator_slot + sizeof(double));
        constexpr uint64_t strategy_request =
            (strategy_base + strategy_fields + strategy_align - 1) / strategy_align *
            strategy_align;
        const uint64_t worker_objects = estimate_checked_sum({
            K_BUILD_ESTIMATE_HEAP_SITES * estimate_shared_request<StandardHeap<true, false>>(),
            computer_request,
            strategy_request,
        });

        return worker_objects;
    }

    [[nodiscard]] uint64_t
    Training() const {
        uint64_t training = 0;
        if (nonempty) {
            // Trainer sample copy, uniform upper/lower vectors and the two truncation heaps.
            const auto ignore = std::max<uint64_t>(
                1, static_cast<uint64_t>(static_cast<double>(quant_sample - 1) * 0.001));
            training = estimate_checked_sum({
                estimate_checked_mul(estimate_checked_mul(quant_sample, dim), sizeof(float)),
                estimate_checked_mul(estimate_checked_mul(2, dim), sizeof(float)),
                estimate_checked_mul(2, estimate_grown_vector(ignore + 1, sizeof(float))),
            });
        }

        return training;
    }

    [[nodiscard]] uint64_t
    Scheduler() const {
        uint64_t scheduler = 0;
        if (has_pool and nonempty) {
            scheduler = estimate_checked_sum({
                estimate_checked_mul(workers,
                                     estimate_scheduler_task_request<InsertAddBatchWorker>()),
                estimate_checked_mul(workers, sizeof(std::future<void>)),
                estimate_deque_request(sizeof(std::function<void()>), workers),
            });
        }

        return scheduler;
    }

    [[nodiscard]] uint64_t
    StandardAllocations(uint64_t fixed_objects) const {
        uint64_t flatten_fixed = 0;
        uint64_t computer_request = 0;
        if (index.metric_ == MetricType::METRIC_TYPE_L2SQR) {
            flatten_fixed = estimate_flatten_fixed_request<MetricType::METRIC_TYPE_L2SQR>();
            computer_request = estimate_computer_request<MetricType::METRIC_TYPE_L2SQR>();
        } else if (index.metric_ == MetricType::METRIC_TYPE_IP) {
            flatten_fixed = estimate_flatten_fixed_request<MetricType::METRIC_TYPE_IP>();
            computer_request = estimate_computer_request<MetricType::METRIC_TYPE_IP>();
        } else {
            flatten_fixed = estimate_flatten_fixed_request<MetricType::METRIC_TYPE_COSINE>();
            computer_request = estimate_computer_request<MetricType::METRIC_TYPE_COSINE>();
        }
        // Common retained SQ bound arrays (lower_bound_, diff_) charged to every quantizer.
        const uint64_t quantizer_arrays =
            estimate_checked_mul(estimate_checked_mul(2, dim), sizeof(float));

        const uint64_t pool_objects =
            estimate_shared_request<VisitedListPool>() +
            K_BUILD_ESTIMATE_SUB_POOLS * sizeof(Deque<std::shared_ptr<VisitedList>>) +
            estimate_function_target_request<BuildEstimatePoolConstructor>();
        const uint64_t pool_model = estimate_checked_sum({
            pool_objects,
            estimate_checked_mul(visited_lists, estimate_shared_request<VisitedList>()),
            pool_replaced ? pool_objects + estimate_shared_request<VisitedList>() : 0,
        });

        const uint64_t worker_objects = WorkerObjects(computer_request);

        const uint64_t training = Training();

        const uint64_t scheduler = Scheduler();

        const uint64_t standard_model = estimate_checked_sum({
            fixed_objects,
            flatten_fixed,
            quantizer_arrays,
            pool_model,
            route_headers,
            estimate_checked_mul(workers, worker_objects),
            training,
            nonempty ? estimate_sampled_dataset_request() : 0,
            estimate_grown_vector(n, sizeof(int64_t)),
            scheduler,
        });
        return standard_model;
    }
};

uint64_t
HGraph::EstimateBuildMemory(uint64_t num_elements) const {
    if (not this->build_memory_estimate_supported()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "HGraph configuration or state does not support EstimateBuildMemory");
    }
    BuildMemoryEstimateContext context(*this, num_elements);
    context.ReplayRoutes();
    const uint64_t allocator_model = context.AllocatorPayload();
    const uint64_t fixed_objects = context.FixedObjects();
    const uint64_t standard_model = context.StandardAllocations(fixed_objects);
    return estimate_checked_add(allocator_model, standard_model);
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(hierarchical_datacell_param_, this->allocator_);
}

float
HGraph::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    CHECK_ARGUMENT(data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   "non-float32 HGraph distance requires a native Dataset query");
    CHECK_ARGUMENT(query != nullptr, "distance query must not be null");
    return calc_native_distance_by_id(query, id, calculate_precise_distance);
}

float
HGraph::CalcDistanceById(const DatasetPtr& query,
                         int64_t id,
                         bool calculate_precise_distance) const {
    CHECK_ARGUMENT(query != nullptr, "distance query must not be null");
    CHECK_ARGUMENT(query->GetNumElements() == 1, "single-ID distance requires one query");
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(query->GetDim() == dim_, "distance query dimension mismatch");
    }
    const auto* native_query = get_data(query);
    CHECK_ARGUMENT(native_query != nullptr, "distance query representation must match index dtype");
    if (data_type_ == DataTypes::DATA_TYPE_SPARSE) {
        const auto& sparse = query->GetSparseVectors()[0];
        const bool valid_sparse =
            sparse.len_ == 0 || (sparse.ids_ != nullptr && sparse.vals_ != nullptr);
        CHECK_ARGUMENT(valid_sparse, "sparse query requires term IDs and values");
    }
    return calc_native_distance_by_id(native_query, id, calculate_precise_distance);
}

float
HGraph::calc_native_distance_by_id(const void* native_query,
                                   int64_t id,
                                   bool calculate_precise_distance) const {
    const auto* query = static_cast<const float*>(native_query);
    FlattenInterfacePtr flat;
    std::shared_lock<std::shared_mutex> lock;
    if (!this->immutable_.load(std::memory_order_acquire)) {
        lock = this->acquire_global_read_lock();
    }
    flat = this->basic_flatten_codes_;
    if (has_precise_reorder() && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    const bool reads_fused_codes =
        this->rabitq_fused_datacell_ != nullptr and flat == this->basic_flatten_codes_;
    // FusedComputer resolves raw code views through the fused block directory. Keep the
    // graph-wide read lock until distance evaluation completes so concurrent growth cannot
    // mutate that directory while code views are being acquired.
    if (lock.owns_lock() and not this->using_dedup_storage() and not reads_fused_codes) {
        lock.unlock();
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
HGraph::CalcDistancesById(const float* query,
                          const int64_t* ids,
                          int64_t count,
                          bool calculate_precise_distance,
                          int64_t topk) const {
    CHECK_ARGUMENT(data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   "non-float32 HGraph distance requires a native Dataset query");
    const bool valid_topk = topk == -1 || topk > 0;
    CHECK_ARGUMENT(valid_topk, "distance topk must be -1 or positive");
    FlattenInterfacePtr flat;
    std::shared_lock<std::shared_mutex> lock;
    if (!this->immutable_.load(std::memory_order_acquire)) {
        lock = this->acquire_global_read_lock();
    }
    flat = this->basic_flatten_codes_;
    if (has_precise_reorder() && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    const bool reads_fused_codes =
        this->rabitq_fused_datacell_ != nullptr and flat == this->basic_flatten_codes_;
    // FusedComputer resolves raw code views through the fused block directory. Keep the
    // graph-wide read lock until distance evaluation completes so concurrent growth cannot
    // mutate that directory while code views are being acquired.
    if (lock.owns_lock() and not this->using_dedup_storage() and not reads_fused_codes) {
        lock.unlock();
    }
    std::vector<bool> validity;
    auto result = InnerIndexInterface::calc_distance_by_id(query, ids, count, flat, &validity);
    if (topk == -1) {
        return result;
    }
    return ApplyTopkWithValidity(result->GetDistances(), ids, count, 1, topk, validity, allocator_);
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (this->label_table_->IsRemoved(i)) {
            continue;
        }
        auto label = this->label_table_->GetLabelById(i);
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    this->check_fused_mutation_supported("ExportModel");
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (has_precise_reorder()) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}

InnerIndexPtr
HGraph::Clone(const IndexCommonParam& param) {
    this->check_fused_mutation_supported("Clone");
    return InnerIndexInterface::Clone(param);
}

void
HGraph::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    std::shared_lock<std::shared_mutex> lock;
    if (this->using_dedup_storage() && !this->immutable_.load(std::memory_order_acquire)) {
        lock = this->acquire_global_read_lock();
    }
    if (raw_vector_ != nullptr) {
        raw_vector_->GetCodesById(inner_id, data);
        return;
    }

    if (this->has_precise_reorder()) {
        this->high_precise_codes_->GetCodesById(inner_id, data);
        return;
    }
    if (this->rabitq_fused_datacell_ != nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "fused RaBitQ codes do not expose a global merged code");
    }
    this->basic_flatten_codes_->GetCodesById(inner_id, data);
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    // resize() precedes later merge validation; even a failed call can retain changed storage.
    // Invalidate at entry, as for Tune, rather than certify a pristine state after failure.
    this->build_estimate_lifecycle_clean_.store(false, std::memory_order_release);
    this->check_fused_mutation_supported("Merge");
    CHECK_ARGUMENT(not this->using_dedup_storage(),
                   "HGraph deduplicate_storage does not support Merge");
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        CHECK_ARGUMENT(this->support_duplicate_ == other_index->support_duplicate_,
                       "cannot merge HGraph with different support_duplicate settings");
        CHECK_ARGUMENT(not other_index->using_dedup_storage(),
                       "HGraph deduplicate_storage does not support Merge");

        auto logical_bias = this->total_count_.load(std::memory_order_acquire);
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, logical_bias);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (has_precise_reorder()) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, logical_bias);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, logical_bias);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < std::min(other_index->route_graphs_.size(), route_graphs_.size());
             ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], logical_bias);
        }
        this->total_count_ += other_index->GetNumElements();
        // Merged ids are fully built before being handed over, so they are immediately visible.
        this->PublishThroughTotalCount();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (has_precise_reorder() and not build_by_base_) ? this->high_precise_codes_
                                                                     : this->basic_flatten_codes_;
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        Vector<InnerIdType> neighbors(this->allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        neighbors.resize(neighbors.size() / 2);
        this->bottom_graph_->InsertNeighborsById(inner_id, neighbors);
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
    if (this->mci_parameters_.enabled) {
        this->build_mci_clique_index();
    }
}

void
HGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    std::shared_lock<std::shared_mutex> lock;
    if (this->using_dedup_storage() && !this->immutable_.load(std::memory_order_acquire)) {
        lock = this->acquire_global_read_lock();
    }
    auto codes = (has_precise_reorder()) ? high_precise_codes_ : basic_flatten_codes_;
    codes = (create_new_raw_vector_) ? raw_vector_ : codes;
    auto buffer = codes->AcquireCodesById(inner_id);
    if (not buffer) {
        if (rabitq_fused_datacell_ != nullptr and codes.get() == basic_flatten_codes_.get() and
            rabitq_split_codes_ != nullptr and
            rabitq_split_codes_->DecodeFusedById(inner_id, data)) {
            return;
        }
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("failed to get vector by inner id {}", inner_id));
    }
    codes->Decode(buffer.Data(), data);
}

void
HGraph::SetImmutable() {
    if (this->immutable_.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock<std::shared_mutex> add_lock(this->add_mutex_);
    std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
    auto empty_mutex = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(empty_mutex);
    this->parallel_searcher_->SetMutexArray(empty_mutex);
    if (this->rabitq_fused_searcher_ != nullptr) {
        this->rabitq_fused_searcher_->SetMutexArray(empty_mutex);
    }
    this->neighbors_mutex_ = empty_mutex;
    this->immutable_.store(true, std::memory_order_release);
}

void
HGraph::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    if (has_precise_reorder()) {
        high_precise_codes_->InitIO(reader_param);
    }
    basic_flatten_codes_->InitIO(reader_param);
    bottom_graph_->InitIO(reader_param);
}

void
HGraph::SetPreciseCodesIO(const std::shared_ptr<Reader>& reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    high_precise_codes_->InitIO(reader_param);
}

const static uint64_t QUERY_SAMPLE_SIZE = 10;
const static int64_t DEFAULT_TOPK = 100;

AnalyzerBasePtr
HGraph::CreateAnalyzer(const AnalyzerParam& param) const {
    return std::make_shared<HGraphAnalyzer>(this, param);
}

std::string
HGraph::GetStats() const {
    if (GetNumElements() == 0) {
        JsonType stats;
        stats["total_count"].SetInt64(0);
        stats["live_count"].SetInt64(0);
        stats["deleted_count"].SetInt64(GetNumberRemoved());
        stats["_analysis"]["skipped"]["structure"].SetString("index is empty");
        AddAnalysisMetadata(stats, GetName(), "stats", "partial");
        return stats.Dump(4);
    }
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = DEFAULT_TOPK;
    analyzer_param.base_sample_size = std::min(QUERY_SAMPLE_SIZE, this->total_count_.load());
    analyzer_param.search_params =
        fmt::format(R"({{"hgraph": {{"ef_search": {}}}}})", ef_construct_);
    auto analyzer = CreateAnalyzer(analyzer_param);
    JsonType stats = analyzer->GetStats();
    // Build-time cache hit-rate is a transient property of the
    // build_with_cache() path (taken only after ImportCache()), so it lives on
    // HGraph rather than in the post-hoc analyzer. A negative rate means this
    // index was not built from an imported cache.
    if (this->build_cache_hit_rate_ >= 0.0F) {
        stats["build_cache_hit_rate"].SetFloat(this->build_cache_hit_rate_);
        stats["build_cache_hit_nodes"].SetUint64(this->build_cache_hit_nodes_);
        stats["build_cache_missed_nodes"].SetUint64(this->build_cache_missed_nodes_);
    } else {
        stats["build_cache_hit_rate"]["skipped_reason"].SetString(
            "index was not built from an imported cache");
    }
    if (this->mci_cliques_ != nullptr) {
        const auto mci_stats = this->mci_cliques_->CollectStats(this->total_count_.load());
        stats["mci_has_index"].SetBool(mci_stats.has_index);
        stats["mci_total_nodes"].SetInt(static_cast<int64_t>(mci_stats.total_nodes));
        stats["mci_covered_nodes"].SetInt(static_cast<int64_t>(mci_stats.covered_nodes));
        stats["mci_covered_node_ratio"].SetFloat(static_cast<float>(mci_stats.covered_node_ratio));
        stats["mci_base_clique_count"].SetInt(static_cast<int64_t>(mci_stats.base_clique_count));
        stats["mci_delta_clique_count"].SetInt(static_cast<int64_t>(mci_stats.delta_clique_count));
        stats["mci_total_clique_count"].SetInt(static_cast<int64_t>(mci_stats.total_clique_count));
        stats["mci_retired_clique_count"].SetInt(
            static_cast<int64_t>(mci_stats.retired_clique_count));
        stats["mci_inactive_node_count"].SetInt(
            static_cast<int64_t>(mci_stats.inactive_node_count));
        stats["mci_base_membership_count"].SetInt(
            static_cast<int64_t>(mci_stats.base_membership_count));
        stats["mci_delta_extra_membership_count"].SetInt(
            static_cast<int64_t>(mci_stats.delta_extra_membership_count));
        stats["mci_delta_clique_membership_count"].SetInt(
            static_cast<int64_t>(mci_stats.delta_clique_membership_count));
        stats["mci_total_membership_count"].SetInt(
            static_cast<int64_t>(mci_stats.total_membership_count));
        stats["mci_avg_membership_per_node"].SetFloat(
            static_cast<float>(mci_stats.avg_membership_per_node));
        stats["mci_avg_clique_size"].SetFloat(static_cast<float>(mci_stats.avg_clique_size));
        stats["mci_max_membership_per_node"].SetInt(
            static_cast<int64_t>(mci_stats.max_membership_per_node));
        stats["mci_max_clique_size"].SetInt(static_cast<int64_t>(mci_stats.max_clique_size));
        stats["mci_memory_usage"].SetInt(
            static_cast<int64_t>(this->mci_cliques_->GetMemoryUsage()));
    }
    AddAnalysisMetadata(stats, GetName(), "stats", "complete");
    return stats.Dump(4);
}

void
HGraph::init_resize_bit_and_reorder() {
    if (use_reorder_) {
        auto reorder_codes = this->get_reorder_codes();
        auto fused_graph =
            reorder_codes.get() == basic_flatten_codes_.get() ? rabitq_fused_datacell_ : nullptr;
        reorder_ = std::make_shared<FlattenReorder>(reorder_codes, allocator_, fused_graph);
    }
}

void
HGraph::check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                                  const IndexCommonParam& common_param,
                                  bool is_create_new) {
    if (raw_vector_param == nullptr) {
        return;
    }

    if (is_create_new) {
        raw_vector_ = FlattenInterface::MakeInstance(raw_vector_param, common_param);
    }

    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ == nullptr) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }
    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    auto io_type_name = raw_vector_param->io_parameter->GetTypeName();
    if (io_type_name != IO_TYPE_VALUE_BLOCK_MEMORY_IO and io_type_name != IO_TYPE_VALUE_MEMORY_IO) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    if (basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = basic_flatten_codes_;
        has_raw_vector_ = true;
        return;
    }

    if (high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = high_precise_codes_;
        has_raw_vector_ = true;
        return;
    }
}

bool
HGraph::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    std::shared_lock<std::shared_mutex> force_remove_rlock;
    if (this->support_force_remove()) {
        force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
    }
    // check if id exists and get copied base data
    uint32_t inner_id = 0;
    {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        inner_id = this->label_table_->GetIdByLabel(id);
    }

    // the validation of the new vector
    void* new_base_vec = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, new_base, &new_base_vec, &data_size);
    if (this->rabitq_fused_datacell_ != nullptr) {
        CHECK_ARGUMENT(new_base->GetDim() == dim_,
                       "updated vector dimension must match the index dimension");
        CHECK_ARGUMENT(new_base_vec != nullptr, "updated vector must not be null");
        this->validate_fused_vector_data(static_cast<const float*>(new_base_vec), 1);
    }
    if (not force_update) {
        std::shared_lock label_lock(this->label_lookup_mutex_);

        float self_dist = 0.0F;
        if (this->rabitq_fused_datacell_ == nullptr) {
            // 1. check whether vectors are same
            Vector<int8_t> base_data(data_size, allocator_);
            GetVectorByInnerId(inner_id, reinterpret_cast<float*>(base_data.data()));
            const float old_self_dist =
                this->CalcDistanceById(reinterpret_cast<float*>(base_data.data()), id);
            self_dist = this->CalcDistanceById(static_cast<float*>(new_base_vec), id);
            if (std::abs(old_self_dist - self_dist) < 1e-3) {
                return true;
            }
        } else {
            self_dist = this->CalcDistanceById(static_cast<float*>(new_base_vec), id);
        }

        // 2. check whether the neighborhood relationship is same
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        for (auto neighbor_inner_id : neighbors) {
            // don't compare with itself
            if (neighbor_inner_id == inner_id) {
                continue;
            }

            float neighbor_dist = 0;
            try {
                neighbor_dist =
                    this->CalcDistanceById(static_cast<float*>(new_base_vec),
                                           this->label_table_->GetLabelById(neighbor_inner_id));
            } catch (const std::runtime_error& e) {
                // incase that neighbor has been deleted
                continue;
            }
            if (neighbor_dist < self_dist) {
                return false;
            }
        }
    }

    if (this->using_dedup_storage()) {
        auto duplicate_tracker = this->bottom_graph_->GetDuplicateTracker();
        CHECK_ARGUMENT(duplicate_tracker != nullptr,
                       "deduplicate_storage update requires duplicate tracker");
        if (duplicate_tracker->GetGroupSize(inner_id) > 1) {
            throw VsagException(
                ErrorType::UNSUPPORTED_INDEX_OPERATION,
                "updating a member of a deduplicated vector group is not supported");
        }
    }

    // note that only modify vector need to obtain unique lock
    // and the lock has been obtained inside datacell
    std::unique_lock<std::shared_mutex> fused_write_lock;
    std::shared_lock<std::shared_mutex> map_lock;
    if (this->rabitq_fused_datacell_ != nullptr) {
        fused_write_lock = std::unique_lock<std::shared_mutex>(this->global_mutex_);
    } else if (this->using_dedup_storage() && !this->immutable_.load(std::memory_order_acquire)) {
        map_lock = this->acquire_global_read_lock();
    }
    std::unique_lock<std::shared_mutex> codes_lock(this->persistent_codes_mutex_);
    bool update_status = basic_flatten_codes_->UpdateVector(new_base_vec, inner_id);
    if (update_status and rabitq_fused_datacell_ != nullptr) {
        this->sync_fused_node_codes(inner_id, new_base_vec);
    }
    if (has_precise_reorder()) {
        update_status = update_status && high_precise_codes_->UpdateVector(new_base_vec, inner_id);
    }
    return update_status;
}

bool
HGraph::UpdateExtraInfo(const DatasetPtr& new_base) {
    return InnerIndexInterface::UpdateExtraInfo(new_base);
}

std::string
HGraph::AnalyzeIndexBySearch(const SearchRequest& request) {
    CHECK_ARGUMENT(request.query_ != nullptr, "analysis query cannot be null");
    CHECK_ARGUMENT(request.topk_ > 0, "analysis topk must be greater than 0");
    if (GetNumElements() == 0) {
        JsonType stats;
        stats["_analysis"]["skipped"]["search"].SetString("index is empty");
        AddAnalysisMetadata(stats,
                            GetName(),
                            "search",
                            "not_applicable",
                            request.query_->GetNumElements(),
                            request.topk_);
        return stats.Dump(4);
    }
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    auto analyzer = CreateAnalyzer(analyzer_param);
    JsonType stats = analyzer->AnalyzeIndexBySearch(request);
    AddAnalysisMetadata(
        stats, GetName(), "search", "complete", request.query_->GetNumElements(), request.topk_);
    return stats.Dump(4);
}

void
HGraph::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
HGraph::cal_memory_usage() {
    auto memory = sizeof(HGraph);
    memory += this->neighbors_mutex_->GetMemoryUsage();
    memory += this->pool_->GetMemoryUsage();
    memory += this->label_table_->GetMemoryUsage();
    memory += this->basic_flatten_codes_->GetMemoryUsage();
    if (this->code_slot_map_ != nullptr) {
        memory += this->code_slot_map_->GetMemoryUsage();
    }
    memory += this->bottom_graph_->GetMemoryUsage();
    for (auto& graph : this->route_graphs_) {
        memory += graph->GetMemoryUsage();
    }
    if (has_precise_reorder()) {
        memory += this->high_precise_codes_->GetMemoryUsage();
    }

    if (this->extra_infos_ != nullptr and this->extra_info_size_ > 0) {
        memory += this->extra_infos_->GetMemoryUsage();
    }

    if (this->create_new_raw_vector_ and this->raw_vector_ != nullptr) {
        memory += raw_vector_->GetMemoryUsage();
    }
    if (this->mci_cliques_ != nullptr) {
        memory += this->mci_cliques_->GetMemoryUsage();
    }
    if (this->conjugate_graph_ != nullptr) {
        std::shared_lock graph_lock(this->conjugate_graph_mutex_);
        memory += this->conjugate_graph_->GetMemoryUsage();
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(memory);
}

}  // namespace vsag
