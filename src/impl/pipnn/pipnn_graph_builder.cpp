// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipnn_graph_builder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common.h"
#include "datacell/graph_interface.h"
#include "half_vector.h"
#include "hash_types.h"
#include "impl/blas/blas_function.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "pipnn_adaptive_pruning.h"
#include "pipnn_build_cache.h"
#include "simd/basic_func.h"
#include "simd/bf16_simd.h"
#include "simd/fp32_simd.h"
#include "utils/lock_strategy.h"

namespace vsag {
namespace {

constexpr uint64_t MAX_HASH_PLANES = 15;
constexpr uint64_t MAX_POINT_LOCK_COUNT = 1U << 16;
constexpr uint16_t COINCIDENT_HASH_FLAG = 1U << 15;
constexpr uint64_t MAX_PARTITION_ITERATIONS = 30;
constexpr uint64_t LEADER_CAP = 1000;
constexpr uint64_t PARTITION_STRIPE_SIZE = 256;
constexpr uint64_t PARTITION_ASSIGNMENT_CHUNK_STRIPES = 16;
constexpr uint64_t PARTITION_SEED = 1000;
constexpr uint64_t MIN_PARALLEL_PARTITION_POINTS = 4096;
constexpr uint64_t SMALL_TOP_K_LIMIT = 16;
constexpr uint64_t MIN_UNDERSIZED_LEAF_NEIGHBOR_COUNT = 4;
constexpr uint64_t CONCERNING_PARTITION_LEVEL = 2;
constexpr double MIN_PARTITION_SHRINK_RATIO = 0.8;

void
require_argument(bool condition, const std::string& message) {
    if (not condition) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, message);
    }
}

uint64_t
checked_product(uint64_t lhs, uint64_t rhs, const char* name) {
    if (lhs != 0 and rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                            std::string("PiPNN ") + name + " size overflows");
    }
    return lhs * rhs;
}

class UninitializedFloatBuffer {
public:
    UninitializedFloatBuffer(uint64_t value_count, Allocator* allocator, const char* name)
        : allocator_(allocator) {
        const uint64_t bytes = checked_product(value_count, sizeof(float), name);
        data_ = static_cast<float*>(allocator_->Allocate(bytes));
        if (data_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                std::string("PiPNN failed to allocate ") + name);
        }
    }

    ~UninitializedFloatBuffer() {
        allocator_->Deallocate(data_);
    }

    UninitializedFloatBuffer(const UninitializedFloatBuffer&) = delete;
    UninitializedFloatBuffer&
    operator=(const UninitializedFloatBuffer&) = delete;

    [[nodiscard]] float*
    Data() const {
        return data_;
    }

private:
    Allocator* allocator_;
    float* data_{nullptr};
};

uint64_t
mix_seed(uint64_t seed, uint64_t salt) {
    return seed * 6364136223846793005ULL + salt;
}

float
sanitize_distance(float distance) {
    if (not std::isfinite(distance)) {
        return std::numeric_limits<float>::infinity();
    }
    return distance == 0.0F ? 0.0F : distance;
}

uint16_t
distance_order_key(float distance) {
    const uint16_t bits = generic::FloatToBF16(sanitize_distance(distance));
    if ((bits & 0x8000U) != 0) {
        return static_cast<uint16_t>(~bits);
    }
    return static_cast<uint16_t>(bits ^ 0x8000U);
}

float
distance_from_order_key(uint16_t key) {
    const uint16_t bits =
        (key & 0x8000U) != 0 ? static_cast<uint16_t>(key ^ 0x8000U) : static_cast<uint16_t>(~key);
    return generic::BF16ToFloat(bits);
}

uint64_t
mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

uint64_t
fingerprint_fp32_row(const float* data, uint64_t dimensions) {
    const auto* bytes = reinterpret_cast<const char*>(data);
    uint64_t h0 = 0x9e3779b97f4a7c15ULL ^ (dimensions * 0xbf58476d1ce4e5b9ULL);
    uint64_t h1 = 0xbf58476d1ce4e5b9ULL;
    uint64_t h2 = 0x94d049bb133111ebULL;
    uint64_t h3 = 0xd6e8feb86659fd93ULL;
    uint64_t offset = 0;
    const uint64_t total_bytes = dimensions * sizeof(float);
    while (offset + 32 <= total_bytes) {
        uint64_t w0 = 0;
        uint64_t w1 = 0;
        uint64_t w2 = 0;
        uint64_t w3 = 0;
        std::memcpy(&w0, bytes + offset, sizeof(w0));
        std::memcpy(&w1, bytes + offset + 8, sizeof(w1));
        std::memcpy(&w2, bytes + offset + 16, sizeof(w2));
        std::memcpy(&w3, bytes + offset + 24, sizeof(w3));
        h0 = (h0 ^ w0) * 0x9e3779b97f4a7c15ULL;
        h1 = (h1 ^ w1) * 0xbf58476d1ce4e5b9ULL;
        h2 = (h2 ^ w2) * 0x94d049bb133111ebULL;
        h3 = (h3 ^ w3) * 0xd6e8feb86659fd93ULL;
        offset += 32;
    }
    while (offset + 8 <= total_bytes) {
        uint64_t word = 0;
        std::memcpy(&word, bytes + offset, sizeof(word));
        h0 = (h0 ^ word) * 0x9e3779b97f4a7c15ULL;
        offset += 8;
    }
    if (offset + 4 <= total_bytes) {
        uint32_t tail = 0;
        std::memcpy(&tail, bytes + offset, sizeof(tail));
        h1 = (h1 ^ static_cast<uint64_t>(tail)) * 0xbf58476d1ce4e5b9ULL;
    }
    return mix64(h0 ^ std::rotl(h1, 13) ^ std::rotl(h2, 29) ^ std::rotl(h3, 47));
}

uint64_t
hash_source_id(std::string_view text) {
    uint64_t hash =
        0x9e3779b97f4a7c15ULL ^ (static_cast<uint64_t>(text.size()) * 0xbf58476d1ce4e5b9ULL);
    uint64_t offset = 0;
    while (offset + 8 <= text.size()) {
        uint64_t word = 0;
        std::memcpy(&word, text.data() + offset, sizeof(word));
        hash = mix64(hash ^ word);
        offset += 8;
    }
    if (offset < text.size()) {
        uint64_t tail = 0;
        std::memcpy(&tail, text.data() + offset, text.size() - offset);
        hash = mix64(hash ^ tail);
    }
    return hash;
}

uint64_t
id_gap(InnerIdType first, InnerIdType second) {
    const uint64_t left = first;
    const uint64_t right = second;
    return left >= right ? left - right : right - left;
}

template <typename Element>
void
group_duplicate_rows(const Vector<InnerIdType>& ids,
                     const Vector<uint32_t>& order,
                     const Vector<const Element*>& rows,
                     uint64_t dimensions,
                     Allocator* allocator,
                     Vector<InnerIdType>& representative_ids,
                     Vector<const Element*>& representative_rows,
                     Vector<std::pair<InnerIdType, InnerIdType>>& duplicates) {
    const uint64_t row_bytes = checked_product(dimensions, sizeof(Element), "row");
    require_argument(row_bytes <= std::numeric_limits<std::size_t>::max(),
                     "PiPNN row size exceeds the platform limit");
    UnorderedMap<std::string_view, InnerIdType> representatives(allocator);
    representatives.reserve(rows.size());

    representative_ids.reserve(ids.size());
    representative_rows.reserve(rows.size());
    for (uint64_t local_id = 0; local_id < ids.size(); ++local_id) {
        const auto input = order.empty() ? local_id : order[local_id];
        const auto key = std::string_view(reinterpret_cast<const char*>(rows[input]),
                                          static_cast<std::size_t>(row_bytes));
        const auto [it, inserted] = representatives.emplace(key, ids[input]);
        if (inserted) {
            representative_ids.emplace_back(ids[input]);
            representative_rows.emplace_back(rows[input]);
        } else {
            duplicates.emplace_back(it.value(), ids[input]);
        }
    }
}

struct ReservoirEntry {
    // A slot is fully assigned before ReservoirState::size makes it readable.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    ReservoirEntry() noexcept {
    }

    ReservoirEntry(InnerIdType input_neighbor, uint16_t input_hash, uint16_t input_distance)
        : neighbor(input_neighbor), hash(input_hash), distance(input_distance) {
    }

    InnerIdType neighbor;
    uint16_t hash;
    uint16_t distance;
};

static_assert(sizeof(ReservoirEntry) == 8);

struct ReservoirState {
    uint16_t size{0};
    uint16_t farthest{0};
};

struct WorkItem {
    WorkItem(Vector<uint32_t>&& input_points, uint64_t input_level, uint64_t input_seed)
        : points(std::move(input_points)), level(input_level), seed(input_seed) {
    }

    WorkItem(WorkItem&&) = default;

    WorkItem&
    operator=(WorkItem&& other) noexcept(false) = default;

    Vector<uint32_t> points;
    uint64_t level{0};
    uint64_t seed{0};
};

using Leaf = Vector<uint32_t>;
using Leaves = Vector<Leaf>;

struct SplitResult {
    explicit SplitResult(Allocator* allocator) : pending(allocator), finished(allocator) {
    }

    Vector<WorkItem> pending;
    Leaves finished;
};

struct CertifiedEdge {
    uint32_t target{0};
    uint32_t old_target{0};
    uint16_t distance{0};
    uint64_t new_gap{0};
    InnerIdType new_id{0};
    uint64_t old_gap{0};
    InnerIdType old_id{0};
};

struct PruneScratch {
    explicit PruneScratch(Allocator* allocator)
        : ordered(allocator),
          selected(allocator),
          neighbors(allocator),
          certified_edges(allocator),
          selected_flags(allocator) {
    }

    Vector<std::pair<float, uint32_t>> ordered;
    Vector<uint32_t> selected;
    Vector<InnerIdType> neighbors;
    Vector<CertifiedEdge> certified_edges;
    Vector<uint8_t> selected_flags;
};

class PagedReservoirArena {
public:
    static constexpr uint64_t PAGE_ROWS = 2048;
    static constexpr uint32_t UNALLOCATED_SLOT = std::numeric_limits<uint32_t>::max();

    explicit PagedReservoirArena(Allocator* allocator)
        : allocator_(allocator), row_slots_(allocator) {
    }

    ~PagedReservoirArena() {
        Reset();
    }

    PagedReservoirArena(const PagedReservoirArena&) = delete;
    PagedReservoirArena&
    operator=(const PagedReservoirArena&) = delete;

    void
    Init(uint64_t total_rows, uint64_t reservoir_size) {
        Reset();
        total_rows_ = total_rows;
        reservoir_size_ = reservoir_size;
        if (total_rows_ == 0 or reservoir_size_ == 0) {
            return;
        }
        num_pages_ = (total_rows_ + PAGE_ROWS - 1) / PAGE_ROWS;
        const uint64_t table_bytes = checked_product(
            num_pages_, sizeof(std::atomic<ReservoirEntry*>), "reservoir page table");
        pages_ = static_cast<std::atomic<ReservoirEntry*>*>(allocator_->Allocate(table_bytes));
        if (pages_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "PiPNN failed to allocate reservoir page table");
        }
        for (uint64_t page = 0; page < num_pages_; ++page) {
            new (&pages_[page]) std::atomic<ReservoirEntry*>(nullptr);
        }
        row_slots_.assign(total_rows_, UNALLOCATED_SLOT);
        next_slot_.store(0, std::memory_order_relaxed);
    }

    void
    Reset() {
        if (pages_ != nullptr) {
            for (uint64_t page = 0; page < num_pages_; ++page) {
                ReservoirEntry* ptr = pages_[page].load(std::memory_order_relaxed);
                if (ptr != nullptr) {
                    allocator_->Deallocate(ptr);
                }
                std::destroy_at(&pages_[page]);
            }
            allocator_->Deallocate(pages_);
            pages_ = nullptr;
        }
        num_pages_ = 0;
        total_rows_ = 0;
        reservoir_size_ = 0;
        next_slot_.store(0, std::memory_order_relaxed);
        Vector<uint32_t>(allocator_).swap(row_slots_);
    }

    [[nodiscard]] bool
    Active() const {
        return pages_ != nullptr;
    }

    [[nodiscard]] const ReservoirEntry*
    RowOrNull(uint32_t source) const {
        if (pages_ == nullptr) {
            return nullptr;
        }
        const uint32_t slot = row_slots_[source];
        if (slot == UNALLOCATED_SLOT) {
            return nullptr;
        }
        const uint64_t page_index = static_cast<uint64_t>(slot) / PAGE_ROWS;
        const uint64_t row_offset = static_cast<uint64_t>(slot) % PAGE_ROWS;
        const ReservoirEntry* page = pages_[page_index].load(std::memory_order_acquire);
        return page == nullptr ? nullptr : page + row_offset * reservoir_size_;
    }

    ReservoirEntry*
    EnsureRow(uint32_t source) {
        uint32_t slot = row_slots_[source];
        if (slot == UNALLOCATED_SLOT) {
            slot = next_slot_.fetch_add(1, std::memory_order_relaxed);
            row_slots_[source] = slot;
        }
        const uint64_t page_index = static_cast<uint64_t>(slot) / PAGE_ROWS;
        const uint64_t row_offset = static_cast<uint64_t>(slot) % PAGE_ROWS;
        ReservoirEntry* page = pages_[page_index].load(std::memory_order_acquire);
        if (page == nullptr) {
            std::scoped_lock lock(page_mutex_);
            page = pages_[page_index].load(std::memory_order_relaxed);
            if (page == nullptr) {
                const uint64_t rows_in_page =
                    std::min<uint64_t>(PAGE_ROWS, total_rows_ - page_index * PAGE_ROWS);
                const uint64_t entry_count =
                    checked_product(rows_in_page, reservoir_size_, "reservoir page");
                const uint64_t bytes =
                    checked_product(entry_count, sizeof(ReservoirEntry), "reservoir page bytes");
                page = static_cast<ReservoirEntry*>(allocator_->Allocate(bytes));
                if (page == nullptr) {
                    throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                        "PiPNN failed to allocate reservoir page");
                }
                pages_[page_index].store(page, std::memory_order_release);
            }
        }
        return page + row_offset * reservoir_size_;
    }

private:
    Allocator* allocator_;
    std::atomic<ReservoirEntry*>* pages_{nullptr};
    uint64_t num_pages_{0};
    uint64_t total_rows_{0};
    uint64_t reservoir_size_{0};
    Vector<uint32_t> row_slots_;
    std::atomic<uint32_t> next_slot_{0};
    std::mutex page_mutex_;
};

class PointLockGuard {
public:
    PointLockGuard(PointsMutex* locks, uint32_t point) : locks_(locks), point_(point) {
        if (locks_ != nullptr) {
            locks_->Lock(point_);
        }
    }

    ~PointLockGuard() {
        if (locks_ != nullptr) {
            locks_->Unlock(point_);
        }
    }

private:
    PointsMutex* locks_;
    uint32_t point_;
};

using DuplicatePairs = Vector<std::pair<InnerIdType, InnerIdType>>;

class PiPNNPipeline {
public:
    PiPNNPipeline(const PiPNNGraphBuilderParameter& parameter,
                  uint64_t dimensions,
                  MetricType metric,
                  Allocator* allocator,
                  SafeThreadPool* thread_pool,
                  uint64_t thread_count,
                  GraphInterfacePtr graph,
                  const Vector<const float*>& rows)
        : parameter_(parameter),
          allocator_(allocator),
          graph_(std::move(graph)),
          thread_pool_(thread_pool),
          thread_count_(thread_count),
          dimensions_(dimensions),
          metric_(metric),
          rows_(&rows),
          fingerprints_(allocator),
          cached_rows_(allocator),
          ids_(allocator),
          hyperplanes_(allocator),
          reservoirs_(allocator),
          paged_reservoirs_(allocator),
          reservoir_states_(allocator) {
    }

    PiPNNPipeline(const PiPNNGraphBuilderParameter& parameter,
                  uint64_t dimensions,
                  MetricType metric,
                  Allocator* allocator,
                  SafeThreadPool* thread_pool,
                  uint64_t thread_count,
                  GraphInterfacePtr graph,
                  const Vector<const uint16_t*>& rows,
                  DataTypes data_type)
        : parameter_(parameter),
          allocator_(allocator),
          graph_(std::move(graph)),
          thread_pool_(thread_pool),
          thread_count_(thread_count),
          dimensions_(dimensions),
          metric_(metric),
          half_rows_(&rows),
          data_type_(data_type),
          fingerprints_(allocator),
          cached_rows_(allocator),
          ids_(allocator),
          hyperplanes_(allocator),
          reservoirs_(allocator),
          paged_reservoirs_(allocator),
          reservoir_states_(allocator) {
    }

    void
    ConfigureCache(const std::string* source_ids,
                   uint64_t source_id_count,
                   const PiPNNBuildCache* imported,
                   std::shared_ptr<PiPNNBuildCache>* exported) {
        source_ids_ = source_ids;
        source_id_count_ = source_id_count;
        imported_cache_ = imported;
        exported_cache_ = exported;
    }

    ~PiPNNPipeline() {
        release_sketch_buffers();
    }

    void
    Build(const Vector<InnerIdType>& ids_sequence, DuplicatePairs* duplicates);

private:
    void
    prepare(const Vector<InnerIdType>& ids_sequence);

    void
    prepare_sketches(bool compute_norms, bool compute_sketches, bool compute_fingerprints);

    void
    release_sketch_buffers();

    bool
    validate_and_index_source_ids(Vector<uint64_t>& source_hashes,
                                  Vector<uint32_t>& hash_table,
                                  uint64_t& table_mask,
                                  bool compute_fingerprints);

    void
    restore_cache();

    void
    capture_cache();

    void
    build_missing_leaf(const Leaf& leaf);

    [[nodiscard]] std::string
    cache_policy() const;

    [[nodiscard]] uint64_t
    fingerprint(uint32_t source) const;

    [[nodiscard]] Leaves
    partition() const;

    void
    split_work_item(const WorkItem& item,
                    Vector<WorkItem>& pending,
                    Leaves& finished,
                    bool allow_inner_parallelism) const;

    [[nodiscard]] Vector<uint32_t>
    sample_leaders(const WorkItem& item) const;

    [[nodiscard]] Leaves
    assign_to_leaders(const WorkItem& item,
                      const Vector<uint32_t>& leaders,
                      uint64_t fanout,
                      bool allow_parallelism) const;

    void
    select_leaders(uint32_t source,
                   const Vector<uint32_t>& leaders,
                   const Vector<InnerIdType>& leader_ids,
                   const Vector<float>& leader_norms,
                   const float* dots,
                   uint64_t fanout,
                   Vector<std::pair<float, uint32_t>>& candidates) const;

    [[nodiscard]] Leaves
    merge_undersized_leaves(Leaves leaves) const;

    void
    build_leaf(const Leaf& leaf);

    void
    group_near_duplicates(Leaves& leaves,
                          DuplicatePairs& duplicates,
                          Vector<uint32_t>& representatives) const;

    [[nodiscard]] float
    coordinate(uint32_t point, uint64_t dimension) const;

    [[nodiscard]] bool
    within_duplicate_threshold(uint32_t lhs, uint32_t rhs) const;

    template <MetricType metric, uint64_t fixed_k = 0>
    void
    build_small_leaf(const Leaf& leaf, UninitializedFloatBuffer& matrix, uint64_t requested_k);

    void
    insert_bidirectional(uint32_t first, uint32_t second, float distance);

    void
    insert_candidate(uint32_t source, uint32_t target, uint16_t hash, uint16_t distance_key);

    [[nodiscard]] std::pair<uint16_t, uint16_t>
    relative_hash(uint32_t source, uint32_t target) const;

    void
    update_farthest(uint32_t source);

    [[nodiscard]] float
    pair_distance(uint32_t lhs, uint32_t rhs) const;

    [[nodiscard]] float
    distance_from_dot(uint32_t lhs, uint32_t rhs, float dot) const;

    void
    robust_prune(uint32_t source,
                 const ReservoirEntry* row,
                 uint16_t size,
                 PruneScratch& scratch) const;

    void
    write_graph(const Vector<uint32_t>& representatives);

    void
    seed_cached_row(uint32_t source);

    [[nodiscard]] bool
    certify_cached_graph(uint32_t source, PruneScratch& scratch) const;

    void
    write_graph_row(uint32_t source, PruneScratch& scratch);

    void
    parallel_for(uint64_t total,
                 uint64_t block_size,
                 const std::function<void(uint64_t, uint64_t)>& task) const;

    ReservoirEntry*
    ensure_reservoir_row(uint32_t source) {
        if (paged_reservoirs_.Active()) {
            return paged_reservoirs_.EnsureRow(source);
        }
        return reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
    }

    [[nodiscard]] const ReservoirEntry*
    reservoir_row_or_null(uint32_t source) const {
        if (paged_reservoirs_.Active()) {
            return paged_reservoirs_.RowOrNull(source);
        }
        if (reservoirs_.empty()) {
            return nullptr;
        }
        return reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
    }

    [[nodiscard]] bool
    native_half() const {
        return data_type_ != DataTypes::DATA_TYPE_FLOAT;
    }

    [[nodiscard]] pipnn_half::Format
    half_format() const {
        return data_type_ == DataTypes::DATA_TYPE_FP16 ? pipnn_half::Format::FP16
                                                       : pipnn_half::Format::BF16;
    }

    [[nodiscard]] const uint16_t*
    half_row(uint32_t local_id) const {
        return (*half_rows_)[local_id];
    }

    void
    gather_row(uint32_t local_id, UninitializedFloatBuffer& buffer, uint64_t row) const {
        if (native_half()) {
            const auto* source = half_row(local_id);
            auto* destination = buffer.Data() + row * dimensions_;
            pipnn_half::DecodeRow(source, destination, dimensions_, half_format());
        } else {
            const auto* source = vector_by_local_id(local_id);
            std::copy(source,
                      source + static_cast<int64_t>(dimensions_),
                      buffer.Data() + static_cast<int64_t>(row * dimensions_));
        }
    }

    [[nodiscard]] const float*
    vector_by_local_id(uint32_t local_id) const {
        return (*rows_)[local_id];
    }

private:
    const PiPNNGraphBuilderParameter& parameter_;
    Allocator* allocator_;
    GraphInterfacePtr graph_;
    SafeThreadPool* thread_pool_;
    uint64_t thread_count_;

    uint64_t dimensions_;
    MetricType metric_;
    // Build() is synchronous, so the pipeline borrows rows only for its own lifetime.
    const Vector<const float*>* rows_{nullptr};
    const Vector<const uint16_t*>* half_rows_{nullptr};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    const std::string* source_ids_{nullptr};
    uint64_t source_id_count_{0};
    const PiPNNBuildCache* imported_cache_{nullptr};
    std::shared_ptr<PiPNNBuildCache>* exported_cache_{nullptr};
    Vector<uint64_t> fingerprints_;
    Vector<uint8_t> cached_rows_;
    bool warm_build_{false};
    bool reuse_graph_{false};
    bool ids_are_strictly_increasing_{false};
    bool ids_are_identity_{false};
    uint64_t reusable_count_{0};
    uint64_t clean_graph_count_{0};
    std::vector<uint32_t> old_to_new_;
    std::vector<uint32_t> new_to_old_;
    std::vector<uint16_t> surviving_counts_;
    std::vector<uint8_t> seeded_rows_;
    std::vector<uint8_t> clean_graph_rows_;
    std::vector<uint8_t> reused_graph_rows_;
    std::vector<uint8_t> certificate_rows_;
    uint64_t hyperplane_signature_{14695981039346656037ULL};
    uint64_t reservoir_size_{0};
    Vector<InnerIdType> ids_;
    Vector<float> hyperplanes_;
    float* norms_{nullptr};
    float* sketches_{nullptr};
    Vector<ReservoirEntry> reservoirs_;
    PagedReservoirArena paged_reservoirs_;
    Vector<ReservoirState> reservoir_states_;
    std::unique_ptr<PointsMutex> point_locks_;
    uint32_t point_lock_count_{0};
};

void
PiPNNPipeline::release_sketch_buffers() {
    if (norms_ != nullptr) {
        allocator_->Deallocate(norms_);
        norms_ = nullptr;
    }
    if (sketches_ != nullptr) {
        allocator_->Deallocate(sketches_);
        sketches_ = nullptr;
    }
    Vector<float>(allocator_).swap(hyperplanes_);
}

void
PiPNNPipeline::Build(const Vector<InnerIdType>& ids_sequence, DuplicatePairs* duplicates) {
    using Clock = std::chrono::steady_clock;
    std::array<double, 6> phase_seconds{};
    auto start = Clock::now();
    auto finish_phase = [&](uint64_t phase) {
        const auto now = Clock::now();
        phase_seconds[phase] = std::chrono::duration<double>(now - start).count();
        start = now;
    };
    prepare(ids_sequence);
    if (ids_.empty()) {
        return;
    }

    bool skip_leaf_pipeline = false;
    if (imported_cache_ != nullptr and not native_half()) {
        finish_phase(0);
        restore_cache();
        skip_leaf_pipeline =
            warm_build_ and reusable_count_ == ids_.size() and duplicates == nullptr;
        const bool all_clean_graph = skip_leaf_pipeline and reuse_graph_ and
                                     not parameter_.adaptive_pruning and
                                     clean_graph_count_ == ids_.size();
        const bool need_norms = metric_ != MetricType::METRIC_TYPE_IP and not all_clean_graph;
        const bool need_sketches = not skip_leaf_pipeline;
        if (need_norms or need_sketches) {
            prepare_sketches(need_norms, need_sketches, false);
        }
        if (not all_clean_graph) {
            reservoir_states_.resize(ids_.size());
            if (warm_build_ and reuse_graph_) {
                paged_reservoirs_.Init(ids_.size(), reservoir_size_);
            } else {
                const uint64_t reservoir_count =
                    checked_product(ids_.size(), reservoir_size_, "reservoir");
                require_argument(reservoir_count <= reservoirs_.max_size(),
                                 "PiPNN reservoir exceeds the allocator limit");
                reservoirs_.resize(reservoir_count);
                if (warm_build_) {
                    parallel_for(ids_.size(), 256, [&](uint64_t begin, uint64_t end) {
                        for (uint64_t source = begin; source < end; ++source) {
                            if (cached_rows_[source] != 0) {
                                seed_cached_row(static_cast<uint32_t>(source));
                            }
                        }
                    });
                }
            }
        }
        finish_phase(1);
    } else {
        if (exported_cache_ != nullptr) {
            exported_cache_->reset();
        }
        bool compute_fingerprints = false;
        if (source_ids_ != nullptr and not native_half() and exported_cache_ != nullptr) {
            Vector<uint64_t> source_hashes(allocator_);
            Vector<uint32_t> hash_table(allocator_);
            uint64_t table_mask = 0;
            compute_fingerprints =
                validate_and_index_source_ids(source_hashes, hash_table, table_mask, false);
            if (not compute_fingerprints) {
                source_ids_ = nullptr;
                source_id_count_ = 0;
            }
        }
        const uint64_t reservoir_count = checked_product(ids_.size(), reservoir_size_, "reservoir");
        require_argument(reservoir_count <= reservoirs_.max_size(),
                         "PiPNN reservoir exceeds the allocator limit");
        reservoirs_.resize(reservoir_count);
        reservoir_states_.resize(ids_.size());
        if (compute_fingerprints) {
            fingerprints_.resize(ids_.size());
        }
        prepare_sketches(metric_ != MetricType::METRIC_TYPE_IP, true, compute_fingerprints);
        finish_phase(0);
        finish_phase(1);
    }

    Vector<uint32_t> representatives(allocator_);
    if (not skip_leaf_pipeline) {
        auto leaves = partition();
        if (duplicates != nullptr) {
            group_near_duplicates(leaves, *duplicates, representatives);
        }
        finish_phase(2);
        parallel_for(leaves.size(), 1, [&](uint64_t begin, uint64_t end) {
            for (uint64_t leaf = begin; leaf < end; ++leaf) {
                if (warm_build_) {
                    build_missing_leaf(leaves[leaf]);
                } else {
                    build_leaf(leaves[leaf]);
                }
            }
        });
        finish_phase(3);
    } else {
        finish_phase(2);
        finish_phase(3);
    }
    write_graph(representatives);
    finish_phase(4);
    release_sketch_buffers();
    capture_cache();
    finish_phase(5);
    if (exported_cache_ != nullptr and *exported_cache_ != nullptr) {
        (*exported_cache_)->phase_seconds = phase_seconds;
    }
}

std::string
PiPNNPipeline::cache_policy() const {
    return "pipnn-reservoir-v4/" + std::to_string(dimensions_) + "/" +
           std::to_string(static_cast<int>(metric_)) + "/" +
           std::to_string(graph_->MaximumDegree()) + "/" +
           std::to_string(std::bit_cast<uint32_t>(parameter_.alpha)) + "/" +
           std::to_string(hyperplane_signature_) + "/" + parameter_.ToJson().Dump();
}

uint64_t
PiPNNPipeline::fingerprint(uint32_t source) const {
    return fingerprint_fp32_row((*rows_)[source], dimensions_);
}

bool
PiPNNPipeline::validate_and_index_source_ids(Vector<uint64_t>& source_hashes,
                                             Vector<uint32_t>& hash_table,
                                             uint64_t& table_mask,
                                             bool compute_fingerprints) {
    if (source_ids_ == nullptr or source_id_count_ != ids_.size()) {
        return false;
    }
    const uint64_t count = ids_.size();
    source_hashes.resize(count);
    if (compute_fingerprints) {
        fingerprints_.resize(count);
    }
    std::atomic<bool> has_empty{false};
    parallel_for(count, 256, [&](uint64_t begin, uint64_t end) {
        for (uint64_t source = begin; source < end; ++source) {
            const auto& id = source_ids_[source];
            if (id.empty()) {
                has_empty.store(true, std::memory_order_relaxed);
                continue;
            }
            source_hashes[source] = hash_source_id(id);
            if (compute_fingerprints) {
                fingerprints_[source] = fingerprint(static_cast<uint32_t>(source));
            }
        }
    });
    if (has_empty.load(std::memory_order_relaxed)) {
        return false;
    }
    constexpr uint32_t missing = std::numeric_limits<uint32_t>::max();
    const uint64_t cap = std::bit_ceil(std::max<uint64_t>(8, count * 2));
    table_mask = cap - 1;
    hash_table.assign(cap, missing);
    for (uint64_t source = 0; source < count; ++source) {
        const uint64_t h = source_hashes[source];
        const auto& id = source_ids_[source];
        uint64_t slot = h & table_mask;
        while (hash_table[slot] != missing) {
            const uint32_t existing = hash_table[slot];
            if (source_hashes[existing] == h and source_ids_[existing] == id) {
                return false;
            }
            slot = (slot + 1) & table_mask;
        }
        hash_table[slot] = static_cast<uint32_t>(source);
    }
    return true;
}

void
PiPNNPipeline::restore_cache() {
    if (exported_cache_ != nullptr) {
        exported_cache_->reset();
    }
    Vector<uint64_t> source_hashes(allocator_);
    Vector<uint32_t> hash_table(allocator_);
    uint64_t table_mask = 0;
    if (not validate_and_index_source_ids(source_hashes, hash_table, table_mask, true)) {
        source_ids_ = nullptr;
        source_id_count_ = 0;
        return;  // Partial/ambiguous source identities take the ordinary cold path.
    }
    if (imported_cache_ == nullptr or imported_cache_->policy != cache_policy() or
        imported_cache_->hashes.size() != imported_cache_->targets.size() or
        imported_cache_->graph_offsets.size() != imported_cache_->source_ids.size() + 1 or
        imported_cache_->inner_ids.size() != imported_cache_->source_ids.size()) {
        return;
    }
    const auto& cache = *imported_cache_;
    constexpr uint32_t missing = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> remap(cache.source_ids.size(), missing);
    parallel_for(cache.source_ids.size(), 512, [&](uint64_t begin, uint64_t end) {
        for (uint64_t old = begin; old < end; ++old) {
            const auto& old_id = cache.source_ids[old];
            if (old_id.empty()) {
                continue;
            }
            const uint64_t h = hash_source_id(old_id);
            uint64_t slot = h & table_mask;
            while (hash_table[slot] != missing) {
                const uint32_t candidate = hash_table[slot];
                if (source_hashes[candidate] == h and source_ids_[candidate] == old_id) {
                    if (fingerprints_[candidate] == cache.fingerprints[old]) {
                        remap[old] = candidate;
                    }
                    break;
                }
                slot = (slot + 1) & table_mask;
            }
        }
    });
    cached_rows_.assign(ids_.size(), 0);
    surviving_counts_.assign(ids_.size(), 0);
    const uint64_t max_degree = graph_->MaximumDegree();
    std::atomic<uint64_t> reusable_total{0};
    parallel_for(remap.size(), 512, [&](uint64_t begin, uint64_t end) {
        uint64_t local_reusable = 0;
        for (uint64_t old = begin; old < end; ++old) {
            const uint32_t source = remap[old];
            if (source == missing or cache.offsets[old] == cache.offsets[old + 1]) {
                continue;
            }
            const uint64_t cand_len = cache.offsets[old + 1] - cache.offsets[old];
            require_argument(cand_len <= reservoir_size_,
                             "PiPNN cached reservoir exceeds the configured capacity");
            require_argument(cache.graph_offsets[old + 1] - cache.graph_offsets[old] <= max_degree,
                             "PiPNN cached graph exceeds maximum degree");
            uint64_t live = 0;
            for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
                if (remap[cache.targets[edge]] != missing) {
                    ++live;
                }
            }
            surviving_counts_[source] = static_cast<uint16_t>(live);
            const uint64_t minimum = std::min<uint64_t>(max_degree, cand_len);
            if (live >= minimum) {
                cached_rows_[source] = 1;
                ++local_reusable;
            }
        }
        if (local_reusable > 0) {
            reusable_total.fetch_add(local_reusable, std::memory_order_relaxed);
        }
    });
    reusable_count_ = reusable_total.load(std::memory_order_relaxed);
    constexpr uint64_t minimum_cache_percent = 80;
    warm_build_ =
        reusable_count_ > 0 and reusable_count_ * 100 >= ids_.size() * minimum_cache_percent;
    if (not warm_build_) {
        std::fill(cached_rows_.begin(), cached_rows_.end(), 0);
        reusable_count_ = 0;
        return;
    }
    old_to_new_ = std::move(remap);
    new_to_old_.assign(ids_.size(), missing);
    seeded_rows_.assign(ids_.size(), 1);
    clean_graph_rows_.assign(ids_.size(), 0);
    reused_graph_rows_.assign(ids_.size(), 0);
    certificate_rows_.assign(ids_.size(), 0);
    reuse_graph_ = cache.reuse_final_graph;
    std::atomic<uint64_t> clean_total{0};
    parallel_for(old_to_new_.size(), 256, [&](uint64_t begin, uint64_t end) {
        uint64_t local_clean = 0;
        for (uint64_t old = begin; old < end; ++old) {
            const auto source = old_to_new_[old];
            if (source == missing) {
                continue;
            }
            new_to_old_[source] = static_cast<uint32_t>(old);
            if (cached_rows_[source] == 0) {
                continue;
            }
            seeded_rows_[source] = 0;
            const uint64_t g_begin = cache.graph_offsets[old];
            const uint64_t g_end = cache.graph_offsets[old + 1];
            bool complete = g_begin != g_end;
            uint16_t min_graph_dist = std::numeric_limits<uint16_t>::max();
            uint16_t max_graph_dist = 0;
            for (uint64_t edge = g_begin; edge < g_end; ++edge) {
                if (old_to_new_[cache.graph_targets[edge]] == missing) {
                    complete = false;
                    break;
                }
                const uint16_t d = cache.graph_distances[edge];
                min_graph_dist = std::min(min_graph_dist, d);
                max_graph_dist = std::max(max_graph_dist, d);
            }
            // Removing rejected candidates preserves pruning, but remapping graph IDs may
            // change BF16-distance tie order. Preserve the old proof only if every selected
            // neighbor retains its ordering relative to every surviving tied candidate.
            const auto old_source_id = cache.inner_ids[old];
            const auto new_source_id = ids_[source];
            for (uint64_t candidate = cache.offsets[old];
                 complete and candidate < cache.offsets[old + 1];
                 ++candidate) {
                const uint16_t cand_dist = cache.distances[candidate];
                if (cand_dist < min_graph_dist or cand_dist > max_graph_dist) {
                    continue;
                }
                const auto old_target = cache.targets[candidate];
                const auto target = old_to_new_[old_target];
                if (target == missing) {
                    continue;
                }
                const auto old_target_id = cache.inner_ids[old_target];
                const auto new_target_id = ids_[target];
                const auto old_target_key =
                    std::make_pair(id_gap(old_source_id, old_target_id), old_target_id);
                const auto new_target_key =
                    std::make_pair(id_gap(new_source_id, new_target_id), new_target_id);
                for (uint64_t edge = g_begin; edge < g_end; ++edge) {
                    if (cand_dist != cache.graph_distances[edge]) {
                        continue;
                    }
                    const auto old_selected = cache.graph_targets[edge];
                    const auto selected = old_to_new_[old_selected];
                    const auto old_selected_id = cache.inner_ids[old_selected];
                    const auto new_selected_id = ids_[selected];
                    const bool old_before = std::make_pair(id_gap(old_source_id, old_selected_id),
                                                           old_selected_id) < old_target_key;
                    const bool new_before = std::make_pair(id_gap(new_source_id, new_selected_id),
                                                           new_selected_id) < new_target_key;
                    if (old_before != new_before) {
                        complete = false;
                        break;
                    }
                }
            }
            if (complete) {
                clean_graph_rows_[source] = 1;
                ++local_clean;
            }
        }
        if (local_clean > 0) {
            clean_total.fetch_add(local_clean, std::memory_order_relaxed);
        }
    });
    clean_graph_count_ = clean_total.load(std::memory_order_relaxed);
}

void
PiPNNPipeline::seed_cached_row(uint32_t source) {
    if (seeded_rows_.empty() or seeded_rows_[source] != 0) {
        return;
    }
    seeded_rows_[source] = 1;
    const auto old = new_to_old_[source];
    const auto& cache = *imported_cache_;
    auto* row = ensure_reservoir_row(source);
    auto& state = reservoir_states_[source];
    for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
        const auto target = old_to_new_[cache.targets[edge]];
        if (target != std::numeric_limits<uint32_t>::max()) {
            row[state.size++] = ReservoirEntry(target, cache.hashes[edge], cache.distances[edge]);
        }
    }
    update_farthest(source);
}

void
PiPNNPipeline::capture_cache() {
    if (source_ids_ == nullptr or exported_cache_ == nullptr) {
        return;
    }
    const uint64_t count = ids_.size();
    auto snapshot = std::make_shared<PiPNNBuildCache>(allocator_);
    snapshot->policy = cache_policy();
    snapshot->fingerprints = std::move(fingerprints_);
    snapshot->source_ids.resize(count);
    snapshot->inner_ids.resize(count);
    snapshot->offsets.resize(count + 1, 0);
    snapshot->graph_offsets.resize(count + 1, 0);

    constexpr uint32_t missing = std::numeric_limits<uint32_t>::max();
    const uint64_t graph_cap = graph_->MaxCapacity();
    const bool use_identity = ids_are_identity_;
    const bool use_dense =
        not use_identity and graph_cap <= checked_product(std::max<uint64_t>(count, 1), 4, "id");
    Vector<uint32_t> dense_local_ids(allocator_);
    Vector<std::pair<InnerIdType, uint32_t>> sorted_local_ids(allocator_);
    if (use_dense) {
        dense_local_ids.assign(graph_cap, missing);
        parallel_for(count, 256, [&](uint64_t begin, uint64_t end) {
            for (uint64_t source = begin; source < end; ++source) {
                dense_local_ids[ids_[source]] = static_cast<uint32_t>(source);
            }
        });
    } else if (not use_identity and not ids_are_strictly_increasing_) {
        sorted_local_ids.resize(count);
        for (uint64_t source = 0; source < count; ++source) {
            sorted_local_ids[source] = {ids_[source], static_cast<uint32_t>(source)};
        }
        std::sort(sorted_local_ids.begin(),
                  sorted_local_ids.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    }
    auto lookup_local_id = [&](InnerIdType neighbor) -> uint32_t {
        if (use_identity) {
            return static_cast<uint32_t>(neighbor);
        }
        if (use_dense) {
            require_argument(neighbor < dense_local_ids.size(),
                             "PiPNN neighbor exceeds local ID map capacity");
            const uint32_t target = dense_local_ids[neighbor];
            require_argument(target != missing, "PiPNN neighbor is missing from local IDs");
            return target;
        }
        if (ids_are_strictly_increasing_) {
            const auto it = std::lower_bound(ids_.begin(), ids_.end(), neighbor);
            require_argument(it != ids_.end() and *it == neighbor,
                             "PiPNN neighbor is missing from sorted IDs");
            return static_cast<uint32_t>(it - ids_.begin());
        }
        const auto it = std::lower_bound(
            sorted_local_ids.begin(),
            sorted_local_ids.end(),
            neighbor,
            [](const auto& item, InnerIdType value) { return item.first < value; });
        require_argument(it != sorted_local_ids.end() and it->first == neighbor,
                         "PiPNN neighbor is missing from local IDs");
        return it->second;
    };

    parallel_for(count, 256, [&](uint64_t begin, uint64_t end) {
        for (uint64_t source = begin; source < end; ++source) {
            snapshot->source_ids[source] = source_ids_[source];
            snapshot->inner_ids[source] = ids_[source];
            uint64_t cand_count = 0;
            if (warm_build_ and reuse_graph_ and seeded_rows_[source] == 0) {
                cand_count = surviving_counts_[source];
            } else if (not reservoir_states_.empty()) {
                cand_count = reservoir_states_[source].size;
            }
            snapshot->offsets[source + 1] = cand_count;
            snapshot->graph_offsets[source + 1] = graph_->GetNeighborSize(ids_[source]);
        }
    });
    std::partial_sum(snapshot->offsets.begin(), snapshot->offsets.end(), snapshot->offsets.begin());
    std::partial_sum(snapshot->graph_offsets.begin(),
                     snapshot->graph_offsets.end(),
                     snapshot->graph_offsets.begin());

    snapshot->targets.resize(snapshot->offsets.back());
    snapshot->distances.resize(snapshot->offsets.back());
    snapshot->hashes.resize(snapshot->offsets.back());
    snapshot->graph_targets.resize(snapshot->graph_offsets.back());
    snapshot->graph_distances.resize(snapshot->graph_offsets.back());

    std::atomic<uint64_t> reused_rows{0};
    std::atomic<uint64_t> reused_graph_rows{0};
    std::atomic<uint64_t> certificate_rows{0};
    parallel_for(count, 256, [&](uint64_t begin, uint64_t end) {
        Vector<ReservoirEntry> entries(allocator_);
        Vector<InnerIdType> adjacency(allocator_);
        entries.reserve(reservoir_size_);
        uint64_t local_reused = 0;
        uint64_t local_reused_graph = 0;
        uint64_t local_cert = 0;
        for (uint64_t source = begin; source < end; ++source) {
            entries.clear();
            if (warm_build_ and reuse_graph_ and seeded_rows_[source] == 0) {
                const auto old = new_to_old_[source];
                const auto& cache = *imported_cache_;
                for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
                    const auto target = old_to_new_[cache.targets[edge]];
                    if (target != missing) {
                        entries.emplace_back(target, cache.hashes[edge], cache.distances[edge]);
                    }
                }
            } else if (not reservoir_states_.empty()) {
                const uint16_t size = reservoir_states_[source].size;
                if (size > 0) {
                    const auto* row = reservoir_row_or_null(static_cast<uint32_t>(source));
                    entries.assign(row, row + size);
                }
            }
            std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.neighbor < rhs.neighbor;
            });
            const uint64_t cand_base = snapshot->offsets[source];
            for (uint64_t i = 0; i < entries.size(); ++i) {
                snapshot->targets[cand_base + i] = entries[i].neighbor;
                snapshot->distances[cand_base + i] = entries[i].distance;
                snapshot->hashes[cand_base + i] = entries[i].hash;
            }
            graph_->GetNeighbors(ids_[source], adjacency);
            const uint64_t graph_base = snapshot->graph_offsets[source];
            require_argument(graph_base + adjacency.size() == snapshot->graph_offsets[source + 1],
                             "PiPNN graph neighbor count changed during snapshot capture");
            for (uint64_t i = 0; i < adjacency.size(); ++i) {
                const auto target = lookup_local_id(adjacency[i]);
                const auto it = std::lower_bound(
                    entries.begin(), entries.end(), target, [](const auto& entry, uint32_t id) {
                        return entry.neighbor < id;
                    });
                require_argument(it != entries.end() and it->neighbor == target,
                                 "PiPNN final neighbor is absent from the reservoir");
                snapshot->graph_targets[graph_base + i] = target;
                snapshot->graph_distances[graph_base + i] = it->distance;
            }
            if (warm_build_ and cached_rows_[source] != 0) {
                ++local_reused;
                local_reused_graph += reused_graph_rows_[source];
                local_cert += certificate_rows_[source];
            }
        }
        if (local_reused > 0) {
            reused_rows.fetch_add(local_reused, std::memory_order_relaxed);
            reused_graph_rows.fetch_add(local_reused_graph, std::memory_order_relaxed);
            certificate_rows.fetch_add(local_cert, std::memory_order_relaxed);
        }
    });
    snapshot->reused_rows = reused_rows.load(std::memory_order_relaxed);
    snapshot->reused_graph_rows = reused_graph_rows.load(std::memory_order_relaxed);
    snapshot->certificate_rows = certificate_rows.load(std::memory_order_relaxed);
    snapshot->reuse_mask = std::move(cached_rows_);
    *exported_cache_ = std::move(snapshot);
}

void
PiPNNPipeline::build_missing_leaf(const Leaf& leaf) {
    if (leaf.size() <= 1) {
        return;
    }
    const uint64_t requested =
        leaf.size() < parameter_.min_leaf_size
            ? std::max(parameter_.leaf_neighbor_count, MIN_UNDERSIZED_LEAF_NEIGHBOR_COUNT)
            : parameter_.leaf_neighbor_count;
    const uint64_t k = std::min<uint64_t>(requested, leaf.size() - 1);
    Vector<uint32_t> ordered(allocator_);
    ordered.reserve(leaf.size());
    for (const auto source : leaf) {
        if (cached_rows_[source] == 0) {
            ordered.push_back(source);
        }
    }
    const uint64_t missing_count = ordered.size();
    if (missing_count == 0) {
        return;
    }
    if (missing_count == leaf.size()) {
        build_leaf(leaf);
        return;
    }
    for (const auto source : leaf) {
        if (cached_rows_[source] != 0) {
            ordered.push_back(source);
        }
    }
    // Pack missing rows first and cached rows second when falling back to Sgemm;
    // for small FP32 missing counts, compute batched dot products zero-copy via SIMD.
    constexpr uint64_t small_missing_threshold = 16;
    constexpr uint64_t stack_dots_capacity = 2048;
    const uint64_t dot_count = checked_product(missing_count, leaf.size(), "cache dots");
    std::array<float, stack_dots_capacity> small_dots{};
    Vector<float> heap_dots(allocator_);
    float* dot_data = small_dots.data();
    if (dot_count > stack_dots_capacity) {
        heap_dots.resize(dot_count);
        dot_data = heap_dots.data();
    }
    if (not native_half() and missing_count <= small_missing_threshold) {
        std::array<const float*, small_missing_threshold> missing_ptrs{};
        for (uint64_t row = 0; row < missing_count; ++row) {
            missing_ptrs[row] = vector_by_local_id(ordered[row]);
        }
        uint64_t column = 0;
        for (; column + 4 <= leaf.size(); column += 4) {
            const float* t0 = vector_by_local_id(ordered[column + 0]);
            const float* t1 = vector_by_local_id(ordered[column + 1]);
            const float* t2 = vector_by_local_id(ordered[column + 2]);
            const float* t3 = vector_by_local_id(ordered[column + 3]);
            for (uint64_t row = 0; row < missing_count; ++row) {
                float d0 = 0.0F;
                float d1 = 0.0F;
                float d2 = 0.0F;
                float d3 = 0.0F;
                FP32ComputeIPBatch4(missing_ptrs[row], dimensions_, t0, t1, t2, t3, d0, d1, d2, d3);
                float* out = dot_data + row * leaf.size() + column;
                out[0] = d0;
                out[1] = d1;
                out[2] = d2;
                out[3] = d3;
            }
        }
        for (; column < leaf.size(); ++column) {
            const float* target = vector_by_local_id(ordered[column]);
            for (uint64_t row = 0; row < missing_count; ++row) {
                dot_data[row * leaf.size() + column] =
                    FP32ComputeIP(missing_ptrs[row], target, dimensions_);
            }
        }
    } else {
        UninitializedFloatBuffer points(
            checked_product(leaf.size(), dimensions_, "cache points"), allocator_, "cache points");
        for (uint64_t i = 0; i < leaf.size(); ++i) {
            gather_row(ordered[i], points, i);
        }
        BlasFunction::Sgemm(BlasFunction::RowMajor,
                            BlasFunction::NoTrans,
                            BlasFunction::Trans,
                            static_cast<int32_t>(missing_count),
                            static_cast<int32_t>(leaf.size()),
                            static_cast<int32_t>(dimensions_),
                            1.0F,
                            points.Data(),
                            static_cast<int32_t>(dimensions_),
                            points.Data(),
                            static_cast<int32_t>(dimensions_),
                            0.0F,
                            dot_data,
                            static_cast<int32_t>(leaf.size()));
    }

    Vector<std::pair<float, uint32_t>> nearest(allocator_);
    if (k > SMALL_TOP_K_LIMIT) {
        nearest.reserve(k);
    }
    std::array<std::pair<float, uint32_t>, SMALL_TOP_K_LIMIT> small_nearest{};
    for (uint64_t row = 0; row < missing_count; ++row) {
        const auto source = ordered[row];
        const auto source_id = ids_[source];
        const auto less = [&](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            const auto lhs_gap = id_gap(source_id, ids_[lhs.second]);
            const auto rhs_gap = id_gap(source_id, ids_[rhs.second]);
            if (lhs_gap != rhs_gap) {
                return lhs_gap < rhs_gap;
            }
            return ids_[lhs.second] < ids_[rhs.second];
        };
        const float* dot_row = dot_data + row * leaf.size();
        bool has_best_cached = false;
        std::pair<float, uint32_t> best_cached{0.0F, 0};
        if (k <= SMALL_TOP_K_LIMIT) {
            uint64_t retained = 0;
            float worst_distance = std::numeric_limits<float>::infinity();
            for (uint64_t column = 0; column < leaf.size(); ++column) {
                if (column == row) {
                    continue;
                }
                const auto target = ordered[column];
                const float distance = distance_from_dot(source, target, dot_row[column]);
                const auto candidate = std::make_pair(distance, target);
                if (column >= missing_count) {
                    if (not has_best_cached or less(candidate, best_cached)) {
                        best_cached = candidate;
                        has_best_cached = true;
                    }
                }
                if (retained < k) {
                    uint64_t pos = retained;
                    while (pos > 0 and less(candidate, small_nearest[pos - 1])) {
                        small_nearest[pos] = small_nearest[pos - 1];
                        --pos;
                    }
                    small_nearest[pos] = candidate;
                    ++retained;
                    if (retained == k) {
                        worst_distance = small_nearest[k - 1].first;
                    }
                } else if (distance <= worst_distance and less(candidate, small_nearest[k - 1])) {
                    uint64_t pos = k - 1;
                    while (pos > 0 and less(candidate, small_nearest[pos - 1])) {
                        small_nearest[pos] = small_nearest[pos - 1];
                        --pos;
                    }
                    small_nearest[pos] = candidate;
                    worst_distance = small_nearest[k - 1].first;
                }
            }
            bool bridged_to_cached = false;
            for (uint64_t i = 0; i < retained; ++i) {
                const auto& [distance, target] = small_nearest[i];
                if (cached_rows_[target] != 0) {
                    bridged_to_cached = true;
                }
                insert_bidirectional(source, target, distance);
            }
            if (has_best_cached and not bridged_to_cached) {
                insert_bidirectional(source, best_cached.second, best_cached.first);
            }
        } else {
            nearest.clear();
            for (uint64_t column = 0; column < leaf.size(); ++column) {
                if (column == row) {
                    continue;
                }
                const auto target = ordered[column];
                const float distance = distance_from_dot(source, target, dot_row[column]);
                const auto candidate = std::make_pair(distance, target);
                if (column >= missing_count) {
                    if (not has_best_cached or less(candidate, best_cached)) {
                        best_cached = candidate;
                        has_best_cached = true;
                    }
                }
                if (nearest.size() < k) {
                    nearest.push_back(candidate);
                    std::push_heap(nearest.begin(), nearest.end(), less);
                } else if (distance <= nearest.front().first and less(candidate, nearest.front())) {
                    std::pop_heap(nearest.begin(), nearest.end(), less);
                    nearest.back() = candidate;
                    std::push_heap(nearest.begin(), nearest.end(), less);
                }
            }
            std::sort(nearest.begin(), nearest.end(), less);
            bool bridged_to_cached = false;
            for (const auto& [distance, target] : nearest) {
                if (cached_rows_[target] != 0) {
                    bridged_to_cached = true;
                }
                insert_bidirectional(source, target, distance);
            }
            if (has_best_cached and not bridged_to_cached) {
                insert_bidirectional(source, best_cached.second, best_cached.first);
            }
        }
    }
}

void
PiPNNPipeline::prepare(const Vector<InnerIdType>& ids_sequence) {
    ids_.assign(ids_sequence.begin(), ids_sequence.end());
    require_argument(ids_.size() == (native_half() ? half_rows_->size() : rows_->size()),
                     "PiPNN IDs and rows must have equal sizes");
    require_argument(ids_.size() <= std::numeric_limits<uint32_t>::max(),
                     "PiPNN point count exceeds the local ID limit");
    if (ids_.empty()) {
        return;
    }

    const uint64_t graph_capacity = graph_->MaxCapacity();
    ids_are_strictly_increasing_ =
        std::adjacent_find(ids_.begin(), ids_.end(), [](InnerIdType lhs, InnerIdType rhs) {
            return lhs >= rhs;
        }) == ids_.end();
    ids_are_identity_ =
        ids_are_strictly_increasing_ and ids_.front() == 0 and ids_.back() == ids_.size() - 1;
    UnorderedSet<InnerIdType> seen_ids(allocator_);
    if (not ids_are_strictly_increasing_) {
        seen_ids.reserve(ids_.size());
        for (uint64_t local_id = 0; local_id < ids_.size(); ++local_id) {
            const auto id = ids_[local_id];
            require_argument(id < graph_capacity, "PiPNN input ID is outside the graph capacity");
            require_argument(seen_ids.emplace(id).second, "PiPNN input IDs must be unique");
            require_argument(native_half()
                                 ? half_row(static_cast<uint32_t>(local_id)) != nullptr
                                 : vector_by_local_id(static_cast<uint32_t>(local_id)) != nullptr,
                             "PiPNN input rows must not be null");
        }
    } else {
        parallel_for(ids_.size(), 256, [&](uint64_t begin, uint64_t end) {
            for (uint64_t local_id = begin; local_id < end; ++local_id) {
                require_argument(ids_[local_id] < graph_capacity,
                                 "PiPNN input ID is outside the graph capacity");
                require_argument(
                    native_half() ? half_row(static_cast<uint32_t>(local_id)) != nullptr
                                  : vector_by_local_id(static_cast<uint32_t>(local_id)) != nullptr,
                    "PiPNN input rows must not be null");
            }
        });
    }

    reservoir_size_ =
        std::max(parameter_.reservoir_size, static_cast<uint64_t>(graph_->MaximumDegree()));
    if (thread_pool_ != nullptr and thread_count_ > 1) {
        point_lock_count_ =
            static_cast<uint32_t>(std::min<uint64_t>(ids_.size(), MAX_POINT_LOCK_COUNT));
        point_locks_ = std::make_unique<PointsMutex>(point_lock_count_, allocator_);
    }

    const uint64_t plane_values =
        checked_product(parameter_.hash_plane_count, dimensions_, "hyperplane");
    hyperplanes_.resize(plane_values);
    std::mt19937_64 random(42);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& value : hyperplanes_) {
        value = normal(random);
        hyperplane_signature_ ^= std::bit_cast<uint32_t>(value);
        hyperplane_signature_ *= 1099511628211ULL;
    }
}

void
PiPNNPipeline::prepare_sketches(bool compute_norms,
                                bool compute_sketches,
                                bool compute_fingerprints) {
    if (compute_norms and norms_ == nullptr) {
        const uint64_t norm_bytes = checked_product(ids_.size(), sizeof(float), "norm bytes");
        norms_ = static_cast<float*>(allocator_->Allocate(norm_bytes));
        if (norms_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "PiPNN failed to allocate norms");
        }
    }
    if (compute_sketches and sketches_ == nullptr) {
        const uint64_t sketch_values =
            checked_product(ids_.size(), parameter_.hash_plane_count, "sketch");
        const uint64_t sketch_bytes = checked_product(sketch_values, sizeof(float), "sketch bytes");
        sketches_ = static_cast<float*>(allocator_->Allocate(sketch_bytes));
        if (sketches_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "PiPNN failed to allocate sketches");
        }
    }

    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        for (uint64_t local_id = begin; local_id < end; ++local_id) {
            if (native_half()) {
                const auto* vector = half_row(static_cast<uint32_t>(local_id));
                float norm_scale = 0.0F;
                if (compute_norms) {
                    const float norm =
                        pipnn_half::Preparation(vector, nullptr, dimensions_, half_format());
                    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                        const float length = std::sqrt(std::max(0.0F, norm));
                        norm_scale = length > 0.0F and std::isfinite(length) ? 1.0F / length : 0.0F;
                        norms_[local_id] = norm_scale;
                    } else {
                        norms_[local_id] = norm;
                    }
                } else if (metric_ == MetricType::METRIC_TYPE_COSINE and norms_ != nullptr) {
                    norm_scale = norms_[local_id];
                }
                if (compute_sketches) {
                    for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
                        const auto* hyperplane = hyperplanes_.data() + plane * dimensions_;
                        float dot =
                            pipnn_half::Preparation(vector, hyperplane, dimensions_, half_format());
                        if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                            dot *= norm_scale;
                        }
                        sketches_[local_id * parameter_.hash_plane_count + plane] = dot;
                    }
                }
                continue;
            }
            const auto* vector = vector_by_local_id(static_cast<uint32_t>(local_id));
            if (compute_fingerprints) {
                fingerprints_[local_id] = fingerprint_fp32_row(vector, dimensions_);
            }
            float norm_scale = 0.0F;
            if (compute_norms) {
                const float norm = FP32ComputeIP(vector, vector, dimensions_);
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    const float length = std::sqrt(std::max(0.0F, norm));
                    norm_scale = length > 0.0F and std::isfinite(length) ? 1.0F / length : 0.0F;
                    norms_[local_id] = norm_scale;
                } else {
                    norms_[local_id] = norm;
                }
            } else if (metric_ == MetricType::METRIC_TYPE_COSINE and norms_ != nullptr) {
                norm_scale = norms_[local_id];
            }
            if (compute_sketches) {
                uint64_t plane = 0;
                for (; plane + 4 <= parameter_.hash_plane_count; plane += 4) {
                    float d0 = 0.0F;
                    float d1 = 0.0F;
                    float d2 = 0.0F;
                    float d3 = 0.0F;
                    FP32ComputeIPBatch4(vector,
                                        dimensions_,
                                        hyperplanes_.data() + (plane + 0) * dimensions_,
                                        hyperplanes_.data() + (plane + 1) * dimensions_,
                                        hyperplanes_.data() + (plane + 2) * dimensions_,
                                        hyperplanes_.data() + (plane + 3) * dimensions_,
                                        d0,
                                        d1,
                                        d2,
                                        d3);
                    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                        d0 *= norm_scale;
                        d1 *= norm_scale;
                        d2 *= norm_scale;
                        d3 *= norm_scale;
                    }
                    const uint64_t base = local_id * parameter_.hash_plane_count + plane;
                    sketches_[base + 0] = d0;
                    sketches_[base + 1] = d1;
                    sketches_[base + 2] = d2;
                    sketches_[base + 3] = d3;
                }
                for (; plane < parameter_.hash_plane_count; ++plane) {
                    float dot = FP32ComputeIP(
                        vector, hyperplanes_.data() + plane * dimensions_, dimensions_);
                    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                        dot *= norm_scale;
                    }
                    sketches_[local_id * parameter_.hash_plane_count + plane] = dot;
                }
            }
        }
    });
}

Leaves
PiPNNPipeline::partition() const {
    Leaf initial(allocator_);
    initial.resize(ids_.size());
    for (uint64_t i = 0; i < ids_.size(); ++i) {
        initial[i] = static_cast<uint32_t>(i);
    }

    Leaves finished(allocator_);
    if (initial.size() <= parameter_.max_leaf_size) {
        finished.emplace_back(std::move(initial));
        return finished;
    }

    Vector<WorkItem> work(allocator_);
    work.emplace_back(std::move(initial), 0, PARTITION_SEED);
    for (uint64_t iteration = 0; iteration < MAX_PARTITION_ITERATIONS; ++iteration) {
        Vector<WorkItem> pending(allocator_);
        const bool all_items_are_small =
            std::all_of(work.begin(), work.end(), [](const auto& item) {
                return item.points.size() < MIN_PARALLEL_PARTITION_POINTS;
            });
        // Use one pool level at a time: outer work-item fan-out disables inner stripe fan-out.
        const bool parallelize_work_items = thread_pool_ != nullptr and thread_count_ > 1 and
                                            work.size() > 1 and
                                            (work.size() >= thread_count_ or all_items_are_small);
        if (parallelize_work_items) {
            Vector<std::unique_ptr<SplitResult>> results(allocator_);
            results.reserve(work.size());
            for (uint64_t item = 0; item < work.size(); ++item) {
                results.emplace_back(std::make_unique<SplitResult>(allocator_));
            }
            Vector<uint64_t> schedule(allocator_);
            schedule.resize(work.size());
            std::iota(schedule.begin(), schedule.end(), 0);
            // The assignment matrix has point_count * leader_count entries, which is a better
            // estimate than point_count alone when leader sampling has not reached its cap.
            auto estimated_assignment_work = [&](uint64_t item) {
                const uint64_t point_count = work[item].points.size();
                const auto sampled = static_cast<uint64_t>(
                    std::ceil(static_cast<double>(point_count) * parameter_.leader_sample_rate));
                const uint64_t leader_count =
                    std::min<uint64_t>(point_count, std::clamp<uint64_t>(sampled, 2, LEADER_CAP));
                return point_count * leader_count;
            };
            if (work.size() <= 4096) {
                std::sort(schedule.begin(), schedule.end(), [&](uint64_t lhs, uint64_t rhs) {
                    const uint64_t lhs_work = estimated_assignment_work(lhs);
                    const uint64_t rhs_work = estimated_assignment_work(rhs);
                    if (lhs_work != rhs_work) {
                        return lhs_work > rhs_work;
                    }
                    return lhs < rhs;
                });
            }
            uint64_t oversized_count = 0;
            uint64_t total_assignment_work = 0;
            uint64_t work_per_thread = 0;
            if (work.size() <= 4096) {
                for (uint64_t item = 0; item < work.size(); ++item) {
                    total_assignment_work += estimated_assignment_work(item);
                }
                work_per_thread = (total_assignment_work + thread_count_ - 1) / thread_count_;
                // An indivisible item larger than one worker's fair share creates a long tail.
                // Process those few items first with stripe-level parallelism across the pool.
                while (oversized_count < schedule.size() and
                       work[schedule[oversized_count]].points.size() >=
                           MIN_PARALLEL_PARTITION_POINTS and
                       estimated_assignment_work(schedule[oversized_count]) > work_per_thread) {
                    const uint64_t item = schedule[oversized_count];
                    split_work_item(
                        work[item], results[item]->pending, results[item]->finished, true);
                    ++oversized_count;
                }
            }
            parallel_for(work.size() - oversized_count, 1, [&](uint64_t begin, uint64_t end) {
                for (uint64_t slot = begin; slot < end; ++slot) {
                    const uint64_t item = schedule[oversized_count + slot];
                    split_work_item(
                        work[item], results[item]->pending, results[item]->finished, false);
                }
            });
            for (auto& result : results) {
                for (auto& item : result->pending) {
                    pending.emplace_back(std::move(item));
                }
                for (auto& leaf : result->finished) {
                    finished.emplace_back(std::move(leaf));
                }
            }
        } else {
            for (const auto& item : work) {
                split_work_item(item, pending, finished, true);
            }
        }
        if (pending.empty()) {
            return merge_undersized_leaves(std::move(finished));
        }
        work = std::move(pending);
    }

    throw VsagException(ErrorType::INTERNAL_ERROR, "PiPNN partition exceeded the iteration limit");
}

void
PiPNNPipeline::split_work_item(const WorkItem& item,
                               Vector<WorkItem>& pending,
                               Leaves& finished,
                               bool allow_inner_parallelism) const {
    const uint64_t requested_fanout =
        item.level < parameter_.fanout.size() ? parameter_.fanout[item.level] : 1;
    const auto leaders = sample_leaders(item);
    const uint64_t fanout = std::min<uint64_t>(requested_fanout, leaders.size());
    auto clusters = assign_to_leaders(item, leaders, fanout, allow_inner_parallelism);

    bool every_cluster_is_parent = not clusters.empty();
    for (const auto& cluster : clusters) {
        every_cluster_is_parent = every_cluster_is_parent and cluster.size() == item.points.size();
    }
    if (every_cluster_is_parent and fanout > 1) {
        Leaf unchanged(item.points.begin(), item.points.end(), allocator_);
        pending.emplace_back(
            std::move(unchanged), item.level + 1, mix_seed(item.seed, item.points.size()));
        return;
    }

    // Assignment visits the sorted, unique parent points in order, so every child keeps that
    // invariant without another sort and deduplication pass.
    for (uint64_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        auto& cluster = clusters[cluster_id];
        if (cluster.empty()) {
            continue;
        }
        if (cluster.size() <= parameter_.max_leaf_size) {
            finished.emplace_back(std::move(cluster));
            continue;
        }

        const bool did_not_shrink = cluster.size() == item.points.size() and fanout == 1;
        const bool shrank_too_slowly =
            item.level > CONCERNING_PARTITION_LEVEL &&
            static_cast<double>(cluster.size()) >
                static_cast<double>(item.points.size()) * MIN_PARTITION_SHRINK_RATIO;
        if (did_not_shrink or shrank_too_slowly) {
            if (shrank_too_slowly) {
                std::mt19937_64 random(mix_seed(item.seed, cluster.size()));
                std::shuffle(cluster.begin(), cluster.end(), random);
            }
            uint64_t begin = 0;
            while (begin < cluster.size()) {
                Leaf fallback(allocator_);
                if (begin > 0) {
                    fallback.emplace_back(cluster[begin - 1]);
                }
                const uint64_t capacity = parameter_.max_leaf_size - fallback.size();
                const uint64_t end = std::min<uint64_t>(begin + capacity, cluster.size());
                fallback.insert(fallback.end(),
                                cluster.begin() + static_cast<int64_t>(begin),
                                cluster.begin() + static_cast<int64_t>(end));
                finished.emplace_back(std::move(fallback));
                begin = end;
            }
            continue;
        }

        pending.emplace_back(std::move(cluster),
                             item.level + 1,
                             mix_seed(item.seed, cluster_id + item.points.size()));
    }
}

Vector<uint32_t>
PiPNNPipeline::sample_leaders(const WorkItem& item) const {
    const auto sampled = static_cast<uint64_t>(
        std::ceil(static_cast<double>(item.points.size()) * parameter_.leader_sample_rate));
    const uint64_t leader_count =
        std::min<uint64_t>(item.points.size(), std::clamp<uint64_t>(sampled, 2, LEADER_CAP));

    std::mt19937_64 random(mix_seed(item.seed, item.points.size()));
    if (leader_count * 4 < item.points.size()) {
        Vector<uint32_t> leaders(allocator_);
        leaders.reserve(leader_count);
        UnorderedSet<uint32_t> selected(allocator_);
        selected.reserve(leader_count);
        std::uniform_int_distribution<uint64_t> distribution(0, item.points.size() - 1);
        while (leaders.size() < leader_count) {
            const auto point = item.points[distribution(random)];
            if (selected.emplace(point).second) {
                leaders.emplace_back(point);
            }
        }
        return leaders;
    }

    Vector<uint32_t> leaders(item.points.begin(), item.points.end(), allocator_);
    std::shuffle(leaders.begin(), leaders.end(), random);
    leaders.resize(leader_count);
    return leaders;
}

void
PiPNNPipeline::select_leaders(uint32_t source,
                              const Vector<uint32_t>& leaders,
                              const Vector<InnerIdType>& leader_ids,
                              const Vector<float>& leader_norms,
                              const float* dots,
                              uint64_t fanout,
                              Vector<std::pair<float, uint32_t>>& candidates) const {
    auto less = [&](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return leader_ids[lhs.second] < leader_ids[rhs.second];
    };
    auto compute_distance = [&](uint64_t leader) -> float {
        const float dot = dots[leader];
        if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
            const float sum = norms_[source] + leader_norms[leader];
            return std::max(0.0F, sanitize_distance(sum - dot - dot));
        }
        if (metric_ == MetricType::METRIC_TYPE_IP) {
            return sanitize_distance(1.0F - dot);
        }
        const float similarity =
            std::clamp(dot * norms_[source] * leader_norms[leader], -1.0F, 1.0F);
        return sanitize_distance(1.0F - similarity);
    };
    candidates.clear();
    if (fanout <= SMALL_TOP_K_LIMIT) {
        uint64_t farthest = 0;
        float worst_distance = std::numeric_limits<float>::infinity();
        for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
            const float distance = compute_distance(leader);
            if (candidates.size() < fanout) {
                const auto candidate = std::make_pair(distance, static_cast<uint32_t>(leader));
                candidates.emplace_back(candidate);
                if (less(candidates[farthest], candidate)) {
                    farthest = candidates.size() - 1;
                }
                if (candidates.size() == fanout) {
                    worst_distance = candidates[farthest].first;
                }
            } else if (distance <= worst_distance) {
                const auto candidate = std::make_pair(distance, static_cast<uint32_t>(leader));
                if (less(candidate, candidates[farthest])) {
                    candidates[farthest] = candidate;
                    farthest = 0;
                    for (uint64_t selected = 1; selected < fanout; ++selected) {
                        if (less(candidates[farthest], candidates[selected])) {
                            farthest = selected;
                        }
                    }
                    worst_distance = candidates[farthest].first;
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), less);
        return;
    }
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        candidates.emplace_back(compute_distance(leader), static_cast<uint32_t>(leader));
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<int64_t>(fanout),
                      candidates.end(),
                      less);
}

Leaves
PiPNNPipeline::assign_to_leaders(const WorkItem& item,
                                 const Vector<uint32_t>& leaders,
                                 uint64_t fanout,
                                 bool allow_parallelism) const {
    Leaves clusters(allocator_);
    clusters.reserve(leaders.size());
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        clusters.emplace_back(allocator_);
    }

    const uint64_t leader_value_count =
        checked_product(leaders.size(), dimensions_, "leader matrix");
    UninitializedFloatBuffer leader_values(leader_value_count, allocator_, "leader matrix");
    Vector<InnerIdType> leader_ids(leaders.size(), allocator_);
    Vector<float> leader_norms(metric_ != MetricType::METRIC_TYPE_IP ? leaders.size() : 0,
                               allocator_);
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        const uint32_t leader_local = leaders[leader];
        gather_row(leader_local, leader_values, leader);
        leader_ids[leader] = ids_[leader_local];
        if (metric_ != MetricType::METRIC_TYPE_IP) {
            leader_norms[leader] = norms_[leader_local];
        }
    }

    if (allow_parallelism and thread_pool_ != nullptr and thread_count_ > 1 and
        item.points.size() >= MIN_PARALLEL_PARTITION_POINTS) {
        const uint64_t assignment_count =
            checked_product(item.points.size(), fanout, "partition assignments");
        const uint64_t assignment_bytes =
            checked_product(assignment_count, sizeof(uint32_t), "partition assignment bytes");
        auto* assignment_data = static_cast<uint32_t*>(allocator_->Allocate(assignment_bytes));
        if (assignment_data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "PiPNN failed to allocate partition assignments");
        }
        auto release_assignments = [this](uint32_t* data) { allocator_->Deallocate(data); };
        std::unique_ptr<uint32_t, decltype(release_assignments)> assignments(assignment_data,
                                                                             release_assignments);
        const uint64_t stripe_count =
            (item.points.size() + PARTITION_STRIPE_SIZE - 1) / PARTITION_STRIPE_SIZE;
        const uint64_t chunk_count = (stripe_count + PARTITION_ASSIGNMENT_CHUNK_STRIPES - 1) /
                                     PARTITION_ASSIGNMENT_CHUNK_STRIPES;
        const uint64_t offset_count =
            checked_product(chunk_count, leaders.size(), "partition cluster offsets");
        Vector<uint64_t> cluster_offsets(offset_count, 0, allocator_);
        auto stripe_range = [&](uint64_t chunk) {
            const uint64_t begin = chunk * PARTITION_ASSIGNMENT_CHUNK_STRIPES;
            const uint64_t end =
                std::min<uint64_t>(begin + PARTITION_ASSIGNMENT_CHUNK_STRIPES, stripe_count);
            return std::pair<uint64_t, uint64_t>{begin, end};
        };
        parallel_for(chunk_count, 1, [&](uint64_t chunk_begin, uint64_t chunk_end) {
            const uint64_t max_stripe_size =
                std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size());
            UninitializedFloatBuffer point_values(
                checked_product(max_stripe_size, dimensions_, "partition stripe"),
                allocator_,
                "partition stripe");
            UninitializedFloatBuffer dots(
                checked_product(max_stripe_size, leaders.size(), "partition distances"),
                allocator_,
                "partition distances");
            Vector<std::pair<float, uint32_t>> candidates(allocator_);
            candidates.reserve(fanout <= SMALL_TOP_K_LIMIT ? fanout : leaders.size());
            for (uint64_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
                auto* offsets = cluster_offsets.data() + chunk * leaders.size();
                const auto [stripe_begin, stripe_end] = stripe_range(chunk);
                for (uint64_t stripe = stripe_begin; stripe < stripe_end; ++stripe) {
                    const uint64_t begin = stripe * PARTITION_STRIPE_SIZE;
                    const uint64_t stripe_size =
                        std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size() - begin);
                    for (uint64_t point = 0; point < stripe_size; ++point) {
                        gather_row(item.points[begin + point], point_values, point);
                    }

                    BlasFunction::Sgemm(BlasFunction::RowMajor,
                                        BlasFunction::NoTrans,
                                        BlasFunction::Trans,
                                        static_cast<int32_t>(stripe_size),
                                        static_cast<int32_t>(leaders.size()),
                                        static_cast<int32_t>(dimensions_),
                                        1.0F,
                                        point_values.Data(),
                                        static_cast<int32_t>(dimensions_),
                                        leader_values.Data(),
                                        static_cast<int32_t>(dimensions_),
                                        0.0F,
                                        dots.Data(),
                                        static_cast<int32_t>(leaders.size()));

                    for (uint64_t point = 0; point < stripe_size; ++point) {
                        const uint64_t point_index = begin + point;
                        const uint32_t local_id = item.points[point_index];
                        select_leaders(local_id,
                                       leaders,
                                       leader_ids,
                                       leader_norms,
                                       dots.Data() + point * leaders.size(),
                                       fanout,
                                       candidates);
                        for (uint64_t selected = 0; selected < fanout; ++selected) {
                            const uint32_t leader = candidates[selected].second;
                            assignment_data[point_index * fanout + selected] = leader;
                            ++offsets[leader];
                        }
                    }
                }
            }
        });

        parallel_for(leaders.size(), 4, [&](uint64_t leader_begin, uint64_t leader_end) {
            for (uint64_t leader = leader_begin; leader < leader_end; ++leader) {
                uint64_t cluster_size = 0;
                for (uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
                    auto& offset = cluster_offsets[chunk * leaders.size() + leader];
                    const uint64_t count = offset;
                    offset = cluster_size;
                    cluster_size += count;
                }
                clusters[leader].resize(cluster_size);
            }
        });
        parallel_for(chunk_count, 1, [&](uint64_t chunk_begin, uint64_t chunk_end) {
            for (uint64_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
                auto* offsets = cluster_offsets.data() + chunk * leaders.size();
                const auto [stripe_begin, stripe_end] = stripe_range(chunk);
                const uint64_t begin = stripe_begin * PARTITION_STRIPE_SIZE;
                const uint64_t end =
                    std::min<uint64_t>(stripe_end * PARTITION_STRIPE_SIZE, item.points.size());
                for (uint64_t point = begin; point < end; ++point) {
                    for (uint64_t selected = 0; selected < fanout; ++selected) {
                        const uint32_t leader = assignment_data[point * fanout + selected];
                        clusters[leader][offsets[leader]++] = item.points[point];
                    }
                }
            }
        });
        return clusters;
    }

    const uint64_t max_stripe_size = std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size());
    UninitializedFloatBuffer point_values(
        checked_product(max_stripe_size, dimensions_, "partition stripe"),
        allocator_,
        "partition stripe");
    UninitializedFloatBuffer dots(
        checked_product(max_stripe_size, leaders.size(), "partition distances"),
        allocator_,
        "partition distances");
    Vector<std::pair<float, uint32_t>> candidates(allocator_);
    candidates.reserve(fanout <= SMALL_TOP_K_LIMIT ? fanout : leaders.size());
    for (uint64_t begin = 0; begin < item.points.size(); begin += PARTITION_STRIPE_SIZE) {
        const uint64_t stripe_size =
            std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size() - begin);
        for (uint64_t point = 0; point < stripe_size; ++point) {
            gather_row(item.points[begin + point], point_values, point);
        }

        BlasFunction::Sgemm(BlasFunction::RowMajor,
                            BlasFunction::NoTrans,
                            BlasFunction::Trans,
                            static_cast<int32_t>(stripe_size),
                            static_cast<int32_t>(leaders.size()),
                            static_cast<int32_t>(dimensions_),
                            1.0F,
                            point_values.Data(),
                            static_cast<int32_t>(dimensions_),
                            leader_values.Data(),
                            static_cast<int32_t>(dimensions_),
                            0.0F,
                            dots.Data(),
                            static_cast<int32_t>(leaders.size()));

        for (uint64_t point = 0; point < stripe_size; ++point) {
            const uint32_t local_id = item.points[begin + point];
            select_leaders(local_id,
                           leaders,
                           leader_ids,
                           leader_norms,
                           dots.Data() + point * leaders.size(),
                           fanout,
                           candidates);
            for (uint64_t selected = 0; selected < fanout; ++selected) {
                clusters[candidates[selected].second].emplace_back(local_id);
            }
        }
    }
    return clusters;
}

Leaves
PiPNNPipeline::merge_undersized_leaves(Leaves leaves) const {
    Leaves merged(allocator_);
    Leaves merged_small(allocator_);
    merged.reserve(leaves.size());

    auto merge_unique = [&](const Leaf& lhs, const Leaf& rhs) {
        Leaf result(allocator_);
        result.reserve(lhs.size() + rhs.size());
        std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
        return result;
    };

    Leaf accumulator(allocator_);
    for (auto& leaf : leaves) {
        if (leaf.size() >= parameter_.min_leaf_size) {
            merged.emplace_back(std::move(leaf));
            continue;
        }
        if (accumulator.empty()) {
            accumulator = std::move(leaf);
            continue;
        }
        auto candidate = merge_unique(accumulator, leaf);
        if (candidate.size() > parameter_.max_leaf_size) {
            merged_small.emplace_back(std::move(accumulator));
            accumulator = std::move(leaf);
        } else {
            accumulator = std::move(candidate);
        }
        if (accumulator.size() >= parameter_.min_leaf_size) {
            merged_small.emplace_back(std::move(accumulator));
            accumulator = Leaf(allocator_);
        }
    }

    if (not accumulator.empty()) {
        auto* preceding = merged_small.empty() ? &merged : &merged_small;
        if (not preceding->empty()) {
            auto candidate = merge_unique(preceding->back(), accumulator);
            if (candidate.size() <= parameter_.max_leaf_size) {
                preceding->back() = std::move(candidate);
                accumulator.clear();
            }
        }
        if (not accumulator.empty()) {
            merged_small.emplace_back(std::move(accumulator));
        }
    }
    for (auto& leaf : merged_small) {
        merged.emplace_back(std::move(leaf));
    }
    return merged;
}

float
PiPNNPipeline::coordinate(uint32_t point, uint64_t dimension) const {
    return native_half() ? pipnn_half::Decode(half_row(point)[dimension], half_format())
                         : vector_by_local_id(point)[dimension];
}

bool
PiPNNPipeline::within_duplicate_threshold(uint32_t lhs, uint32_t rhs) const {
    const double threshold = parameter_.duplicate_distance_threshold;
    double distance = 0.0;
    for (uint64_t dimension = 0; dimension < dimensions_; ++dimension) {
        const double diff =
            static_cast<double>(coordinate(lhs, dimension)) - coordinate(rhs, dimension);
        distance += diff * diff;
        if (distance > threshold) {
            return false;
        }
    }
    return std::isfinite(distance) and distance <= threshold;
}

void
PiPNNPipeline::group_near_duplicates(Leaves& leaves,
                                     DuplicatePairs& duplicates,
                                     Vector<uint32_t>& representatives) const {
    // A one-coordinate range check is a necessary condition for squared L2.
    // This avoids a second BLAS pass and cancellation in the norm-dot formula.
    Vector<float> first(ids_.size(), allocator_);
    Vector<uint64_t> offsets(ids_.size() + 1, 0, allocator_);
    for (uint64_t point = 0; point < ids_.size(); ++point) {
        first[point] = coordinate(static_cast<uint32_t>(point), 0);
    }
    // Overlapping partitions can contain more leaves than input points; check the
    // bound explicitly before storing leaf indices in 32 bits.
    require_argument(leaves.size() <= std::numeric_limits<uint32_t>::max(),
                     "PiPNN duplicate detection has too many leaves");
    for (auto& leaf : leaves) {
        for (const auto point : leaf) {
            if (std::isfinite(first[point])) {
                ++offsets[static_cast<uint64_t>(point) + 1];
            }
        }
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    Vector<uint32_t> memberships(offsets.back(), allocator_);
    {
        auto cursor = offsets;
        for (uint64_t leaf_id = 0; leaf_id < leaves.size(); ++leaf_id) {
            auto& leaf = leaves[leaf_id];
            std::sort(leaf.begin(), leaf.end(), [&](uint32_t lhs, uint32_t rhs) {
                const bool lhs_finite = std::isfinite(first[lhs]);
                const bool rhs_finite = std::isfinite(first[rhs]);
                if (lhs_finite != rhs_finite) {
                    return lhs_finite;
                }
                if (lhs_finite and first[lhs] != first[rhs]) {
                    return first[lhs] < first[rhs];
                }
                return lhs < rhs;
            });
            for (const auto point : leaf) {
                if (not std::isfinite(first[point])) {
                    break;
                }
                memberships[cursor[point]++] = static_cast<uint32_t>(leaf_id);
            }
        }
    }
    representatives.resize(ids_.size());
    Vector<uint32_t> checked(ids_.size(), std::numeric_limits<uint32_t>::max(), allocator_);
    const double threshold = parameter_.duplicate_distance_threshold;
    // Greedily process graph IDs in ascending order. Pick the lowest-ID existing
    // representative that shares a leaf and is within the threshold, or keep this point
    // as a new representative. Members never act as candidates: in the 1--2--4 test
    // chain, 2 joins 1 but cannot pull 4 into that group when 1 and 4 are too far apart.
    for (uint64_t point = 0; point < ids_.size(); ++point) {
        const auto source = static_cast<uint32_t>(point);
        auto representative = source;
        for (uint64_t pos = offsets[point]; pos < offsets[point + 1]; ++pos) {
            const auto& index = leaves[memberships[pos]];
            const auto begin = std::lower_bound(
                index.begin(), index.end(), first[source], [&](uint32_t target, float center) {
                    const double diff = static_cast<double>(first[target]) - center;
                    return std::isfinite(first[target]) and diff < 0.0 and diff * diff > threshold;
                });
            for (auto it = begin; it != index.end(); ++it) {
                const auto target = *it;
                const double diff = static_cast<double>(first[target]) - first[source];
                if (not std::isfinite(first[target]) or diff * diff > threshold) {
                    break;
                }
                if (target < representative and representatives[target] == target and
                    checked[target] != source) {
                    checked[target] = source;
                    if (within_duplicate_threshold(source, target)) {
                        representative = target;
                    }
                }
            }
        }
        representatives[source] = representative;
        if (representative != source) {
            duplicates.emplace_back(ids_[representative], ids_[source]);
        }
    }
    // Retain the union of leaf memberships without consuming Top-K slots on members.
    // Return the leaves to ID order before graph construction.
    for (auto& leaf : leaves) {
        if (not duplicates.empty()) {
            for (auto& point : leaf) {
                point = representatives[point];
            }
        }
        std::sort(leaf.begin(), leaf.end());
        if (not duplicates.empty()) {
            leaf.erase(std::unique(leaf.begin(), leaf.end()), leaf.end());
        }
    }
    if (duplicates.empty()) {
        representatives.clear();
    }
}

void
PiPNNPipeline::build_leaf(const Leaf& leaf) {
    if (leaf.size() <= 1) {
        return;
    }

    const uint64_t point_count = leaf.size();
    const uint64_t matrix_values = checked_product(point_count, dimensions_, "leaf matrix");
    UninitializedFloatBuffer matrix(matrix_values, allocator_, "leaf matrix");
    const uint64_t requested_neighbor_count =
        point_count < parameter_.min_leaf_size
            ? std::max(parameter_.leaf_neighbor_count, MIN_UNDERSIZED_LEAF_NEIGHBOR_COUNT)
            : parameter_.leaf_neighbor_count;
    const uint64_t neighbor_count = std::min<uint64_t>(requested_neighbor_count, point_count - 1);
    if (neighbor_count <= SMALL_TOP_K_LIMIT) {
        if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
            if (neighbor_count == 5) {
                build_small_leaf<MetricType::METRIC_TYPE_L2SQR, 5>(leaf, matrix, neighbor_count);
            } else {
                build_small_leaf<MetricType::METRIC_TYPE_L2SQR>(leaf, matrix, neighbor_count);
            }
            return;
        }
        if (metric_ == MetricType::METRIC_TYPE_IP) {
            if (neighbor_count == 5) {
                build_small_leaf<MetricType::METRIC_TYPE_IP, 5>(leaf, matrix, neighbor_count);
            } else {
                build_small_leaf<MetricType::METRIC_TYPE_IP>(leaf, matrix, neighbor_count);
            }
            return;
        }
    }
    for (uint64_t point = 0; point < point_count; ++point) {
        gather_row(leaf[point], matrix, point);
    }
    const uint64_t distance_values = checked_product(point_count, point_count, "leaf distances");
    UninitializedFloatBuffer distances(distance_values, allocator_, "leaf distances");
    BlasFunction::Ssyrk(BlasFunction::RowMajor,
                        BlasFunction::CblasLower,
                        BlasFunction::NoTrans,
                        static_cast<int32_t>(point_count),
                        static_cast<int32_t>(dimensions_),
                        1.0F,
                        matrix.Data(),
                        static_cast<int32_t>(dimensions_),
                        0.0F,
                        distances.Data(),
                        static_cast<int32_t>(point_count));

    if (neighbor_count <= SMALL_TOP_K_LIMIT) {
        std::array<std::pair<float, uint32_t>, SMALL_TOP_K_LIMIT> nearest{};
        for (uint64_t source = 0; source < point_count; ++source) {
            const auto source_id = ids_[leaf[source]];
            auto comparator = [&](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                const auto lhs_id = ids_[leaf[lhs.second]];
                const auto rhs_id = ids_[leaf[rhs.second]];
                const auto lhs_gap = id_gap(source_id, lhs_id);
                const auto rhs_gap = id_gap(source_id, rhs_id);
                if (lhs_gap != rhs_gap) {
                    return lhs_gap < rhs_gap;
                }
                return lhs_id < rhs_id;
            };

            uint64_t retained = 0;
            uint64_t farthest = 0;
            for (uint64_t target = 0; target < point_count; ++target) {
                if (source == target) {
                    continue;
                }
                const uint64_t row = std::max(source, target);
                const uint64_t column = std::min(source, target);
                const float distance = distance_from_dot(
                    leaf[source], leaf[target], distances.Data()[row * point_count + column]);
                const auto candidate = std::make_pair(distance, static_cast<uint32_t>(target));

                if (retained < neighbor_count) {
                    nearest[retained] = candidate;
                    if (retained == 0 or comparator(nearest[farthest], candidate)) {
                        farthest = retained;
                    }
                    ++retained;
                    continue;
                }
                if (not comparator(candidate, nearest[farthest])) {
                    continue;
                }
                nearest[farthest] = candidate;
                farthest = 0;
                for (uint64_t selected = 1; selected < retained; ++selected) {
                    if (comparator(nearest[farthest], nearest[selected])) {
                        farthest = selected;
                    }
                }
            }

            std::sort(nearest.begin(), nearest.begin() + retained, comparator);
            for (uint64_t candidate = 0; candidate < retained; ++candidate) {
                const uint32_t target = nearest[candidate].second;
                insert_bidirectional(leaf[source], leaf[target], nearest[candidate].first);
            }
        }
        return;
    }

    Vector<std::pair<float, uint32_t>> candidates(allocator_);
    candidates.reserve(point_count - 1);
    for (uint64_t source = 0; source < point_count; ++source) {
        candidates.clear();
        for (uint64_t target = 0; target < point_count; ++target) {
            if (source == target) {
                continue;
            }
            const uint64_t row = std::max(source, target);
            const uint64_t column = std::min(source, target);
            const float distance = distance_from_dot(
                leaf[source], leaf[target], distances.Data()[row * point_count + column]);
            candidates.emplace_back(distance, static_cast<uint32_t>(target));
        }
        const uint64_t retained = std::min<uint64_t>(neighbor_count, candidates.size());
        auto comparator = [&](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            const auto source_id = ids_[leaf[source]];
            const auto lhs_id = ids_[leaf[lhs.second]];
            const auto rhs_id = ids_[leaf[rhs.second]];
            const auto lhs_gap = id_gap(source_id, lhs_id);
            const auto rhs_gap = id_gap(source_id, rhs_id);
            if (lhs_gap != rhs_gap) {
                return lhs_gap < rhs_gap;
            }
            return lhs_id < rhs_id;
        };
        std::partial_sort(candidates.begin(),
                          candidates.begin() + static_cast<int64_t>(retained),
                          candidates.end(),
                          comparator);
        for (uint64_t candidate = 0; candidate < retained; ++candidate) {
            const uint32_t target = candidates[candidate].second;
            insert_bidirectional(leaf[source], leaf[target], candidates[candidate].first);
        }
    }
}

template <MetricType metric, uint64_t fixed_k>
void
PiPNNPipeline::build_small_leaf(const Leaf& leaf,
                                UninitializedFloatBuffer& matrix_buffer,
                                uint64_t requested_k) {
    auto* matrix = matrix_buffer.Data();
    const uint64_t neighbor_count = fixed_k == 0 ? requested_k : fixed_k;
    static_assert(metric == MetricType::METRIC_TYPE_L2SQR || metric == MetricType::METRIC_TYPE_IP);
    const uint64_t count = leaf.size();
    Vector<std::pair<float, uint32_t>> nearest(
        checked_product(count, neighbor_count, "leaf candidates"), allocator_);
    Vector<float> worst_distances(count, 0.0F, allocator_);
    Vector<InnerIdType> ids(count, allocator_);
    Vector<float> norms(count, 0.0F, allocator_);
    for (uint64_t point = 0; point < count; ++point) {
        ids[point] = ids_[leaf[point]];
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            norms[point] = norms_[leaf[point]];
        }
    }
    auto less = [&](uint64_t source, const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        const auto left_gap = id_gap(ids[source], ids[lhs.second]);
        const auto right_gap = id_gap(ids[source], ids[rhs.second]);
        return left_gap != right_gap ? left_gap < right_gap : ids[lhs.second] < ids[rhs.second];
    };
    auto from_dot = [&](uint64_t source, uint64_t target, float dot) {
        if (native_half()) {
            return distance_from_dot(leaf[source], leaf[target], dot);
        }
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            return std::max(0.0F, sanitize_distance(norms[source] + norms[target] - dot - dot));
        } else {
            return sanitize_distance(1.0F - dot);
        }
    };
    // Rows stay sorted by the full distance/ID comparator, so the last slot
    // is always the current worst candidate; no separate position array is needed.
    auto sorted_insert =
        [&](uint64_t source, auto* row, const std::pair<float, uint32_t>& candidate) {
            uint64_t pos = neighbor_count - 1;
            while (pos > 0 and less(source, candidate, row[pos - 1])) {
                row[pos] = row[pos - 1];
                --pos;
            }
            row[pos] = candidate;
        };
    auto offer = [&](uint64_t source, uint32_t target, float distance) {
        auto* row = nearest.data() + source * neighbor_count;
        const auto candidate = std::make_pair(distance, target);
        const uint64_t worst = neighbor_count - 1;
        if (not less(source, candidate, row[worst])) {
            return;
        }
        sorted_insert(source, row, candidate);
        worst_distances[source] = row[worst].first;
    };
    constexpr uint64_t strip_rows = 64;
    static_assert(strip_rows > SMALL_TOP_K_LIMIT);
    UninitializedFloatBuffer dot_buffer(
        checked_product(std::min(count, strip_rows), count, "leaf dot strip"),
        allocator_,
        "leaf dot strip");
    for (uint64_t row_begin = 0; row_begin < count; row_begin += strip_rows) {
        const auto row_end = std::min(count, row_begin + strip_rows);

        for (uint64_t point = row_begin; point < row_end; ++point) {
            if (native_half()) {
                gather_row(leaf[point], matrix_buffer, point);
            } else {
                const auto* source = vector_by_local_id(leaf[point]);
                std::copy(source, source + dimensions_, matrix + point * dimensions_);
            }
        }

        const auto* dots = dot_buffer.Data();
        if (row_begin == 0) {
            BlasFunction::Ssyrk(BlasFunction::RowMajor,
                                BlasFunction::CblasLower,
                                BlasFunction::NoTrans,
                                static_cast<int32_t>(row_end),
                                static_cast<int32_t>(dimensions_),
                                1.0F,
                                matrix,
                                static_cast<int32_t>(dimensions_),
                                0.0F,
                                dot_buffer.Data(),
                                static_cast<int32_t>(count));

            // Seed the first K+1 lists from their lower triangle. All later offers see
            // full lists, and every endpoint keeps ascending target order.
            for (uint64_t row = 1; row <= neighbor_count; ++row) {
                for (uint64_t column = 0; column < row; ++column) {
                    const auto distance = from_dot(row, column, dots[row * count + column]);
                    nearest[row * neighbor_count + column] = {distance,
                                                              static_cast<uint32_t>(column)};
                    nearest[column * neighbor_count + row - 1] = {distance,
                                                                  static_cast<uint32_t>(row)};
                }
            }
            for (uint64_t source = 0; source <= neighbor_count; ++source) {
                auto* row = nearest.data() + source * neighbor_count;
                std::sort(row, row + neighbor_count, [&](const auto& a, const auto& b) {
                    return less(source, a, b);
                });
                worst_distances[source] =
                    nearest[source * neighbor_count + neighbor_count - 1].first;
            }

        } else {
            BlasFunction::Sgemm(BlasFunction::RowMajor,
                                BlasFunction::NoTrans,
                                BlasFunction::Trans,
                                static_cast<int32_t>(row_end - row_begin),
                                static_cast<int32_t>(row_end),
                                static_cast<int32_t>(dimensions_),
                                1.0F,
                                matrix + row_begin * dimensions_,
                                static_cast<int32_t>(dimensions_),
                                matrix,
                                static_cast<int32_t>(dimensions_),
                                0.0F,
                                dot_buffer.Data(),
                                static_cast<int32_t>(count));
        }

        for (uint64_t row = std::max(row_begin, neighbor_count + 1); row < row_end; ++row) {
            const auto* dot_row = dots + (row - row_begin) * count;
            std::array<std::pair<float, uint32_t>, SMALL_TOP_K_LIMIT> current;
            for (uint64_t column = 0; column < neighbor_count; ++column) {
                const auto distance = from_dot(row, column, dot_row[column]);
                current[column] = {distance, static_cast<uint32_t>(column)};
                offer(column, static_cast<uint32_t>(row), distance);
            }
            std::sort(current.data(),
                      current.data() + neighbor_count,
                      [&](const auto& a, const auto& b) { return less(row, a, b); });
            const uint64_t worst = neighbor_count - 1;
            float current_threshold = current[worst].first;
            for (uint64_t block_begin = neighbor_count; block_begin < row; block_begin += 32) {
                const uint64_t length = std::min<uint64_t>(32, row - block_begin);
                std::array<float, 32> buffered_distances;
                uint32_t eligible = 0;
                // A stale threshold is only looser: prefiltering cannot discard
                // a valid candidate. Survivors are rechecked in original column order.
                const float stale_threshold = current_threshold;
                // Let the compiler choose whether to vectorize this loop (#3474).
                for (uint64_t offset = 0; offset < length; ++offset) {
                    const uint64_t column = block_begin + offset;
                    const float distance = from_dot(row, column, dot_row[column]);
                    buffered_distances[offset] = distance;
                    eligible |= static_cast<uint32_t>(
                                    distance <= std::max(stale_threshold, worst_distances[column]))
                                << offset;
                }
                while (eligible != 0) {
                    const uint64_t offset = std::countr_zero(eligible);
                    eligible &= eligible - 1;
                    const uint64_t column = block_begin + offset;
                    const auto distance = buffered_distances[offset];
                    // Equal distances must still reach the ID tie-breaker.
                    if (distance > std::max(current_threshold, worst_distances[column])) {
                        continue;
                    }
                    const auto candidate = std::make_pair(distance, static_cast<uint32_t>(column));
                    if (less(row, candidate, current[worst])) {
                        sorted_insert(row, current.data(), candidate);
                        current_threshold = current[worst].first;
                    }
                    offer(column, static_cast<uint32_t>(row), distance);
                }
            }
            std::copy_n(current.data(), neighbor_count, nearest.data() + row * neighbor_count);
            worst_distances[row] = current[worst].first;
        }
    }
    // Preserve source/target emission order and the existing bidirectional reservoir policy.
    for (uint64_t source = 0; source < count; ++source) {
        auto* row = nearest.data() + source * neighbor_count;
        for (uint64_t candidate = 0; candidate < neighbor_count; ++candidate) {
            const auto [distance, target] = row[candidate];
            insert_bidirectional(leaf[source], leaf[target], distance);
        }
    }
}

std::pair<uint16_t, uint16_t>
PiPNNPipeline::relative_hash(uint32_t source, uint32_t target) const {
    uint16_t forward = 0;
    uint16_t reverse = 0;
    bool coincident = true;
    const uint64_t source_offset = static_cast<uint64_t>(source) * parameter_.hash_plane_count;
    const uint64_t target_offset = static_cast<uint64_t>(target) * parameter_.hash_plane_count;
    for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
        const float source_value = sketches_[source_offset + plane];
        const float target_value = sketches_[target_offset + plane];
        if (target_value >= source_value) {
            forward |= static_cast<uint16_t>(1U << plane);
        }
        if (source_value >= target_value) {
            reverse |= static_cast<uint16_t>(1U << plane);
        }
        coincident = coincident and target_value == source_value;
    }
    if (coincident) {
        forward |= COINCIDENT_HASH_FLAG;
        reverse |= COINCIDENT_HASH_FLAG;
    }
    return {forward, reverse};
}

void
PiPNNPipeline::insert_bidirectional(uint32_t first, uint32_t second, float distance) {
    if (first == second) {
        return;
    }
    const auto [forward, reverse] = relative_hash(first, second);
    const uint16_t distance_key = distance_order_key(distance);
    // Keep the original forward-then-reverse publication order and per-source locks.
    insert_candidate(first, second, forward, distance_key);
    insert_candidate(second, first, reverse, distance_key);
}

void
PiPNNPipeline::insert_candidate(uint32_t source,
                                uint32_t target,
                                uint16_t hash,
                                uint16_t distance_key) {
    if (source == target) {
        return;
    }

    const auto incoming_key = std::make_tuple(distance_key, ids_[target], hash);
    const uint32_t lock_index = point_lock_count_ == 0 ? source : source % point_lock_count_;
    PointLockGuard lock(point_locks_.get(), lock_index);
    seed_cached_row(source);
    if (reuse_graph_) {
        clean_graph_rows_[source] = 0;
    }
    auto* row = ensure_reservoir_row(source);

    auto& state = reservoir_states_[source];

    if (state.size == reservoir_size_) {
        const auto& farthest = row[state.farthest];
        const auto farthest_key =
            std::make_tuple(farthest.distance, ids_[farthest.neighbor], farthest.hash);
        if (incoming_key >= farthest_key) {
            return;
        }
    }

    for (uint16_t index = 0; index < state.size; ++index) {
        auto& entry = row[index];
        if (entry.neighbor == target) {
            if (incoming_key < std::make_tuple(entry.distance, ids_[entry.neighbor], entry.hash)) {
                const bool was_farthest = index == state.farthest;
                entry = ReservoirEntry{target, hash, distance_key};
                if (was_farthest) {
                    update_farthest(source);
                }
            }
            return;
        }
        if (entry.hash != hash or (hash & COINCIDENT_HASH_FLAG) != 0) {
            continue;
        }
        if (incoming_key < std::make_tuple(entry.distance, ids_[entry.neighbor], entry.hash)) {
            const bool was_farthest = index == state.farthest;
            entry = ReservoirEntry{target, hash, distance_key};
            if (was_farthest) {
                update_farthest(source);
            }
        }
        return;
    }

    if (state.size < reservoir_size_) {
        const uint16_t inserted = state.size;
        row[inserted] = ReservoirEntry{target, hash, distance_key};
        ++state.size;
        if (inserted == 0) {
            state.farthest = 0;
        } else {
            const auto& farthest = row[state.farthest];
            const auto farthest_key =
                std::make_tuple(farthest.distance, ids_[farthest.neighbor], farthest.hash);
            if (incoming_key > farthest_key) {
                state.farthest = inserted;
            }
        }
        return;
    }

    row[state.farthest] = ReservoirEntry{target, hash, distance_key};
    update_farthest(source);
}

void
PiPNNPipeline::update_farthest(uint32_t source) {
    auto& state = reservoir_states_[source];
    if (state.size == 0) {
        state.farthest = 0;
        return;
    }
    const auto* row = reservoir_row_or_null(source);
    state.farthest = 0;
    for (uint16_t index = 1; index < state.size; ++index) {
        const auto& candidate = row[index];
        const auto& farthest = row[state.farthest];
        if (std::make_tuple(candidate.distance, ids_[candidate.neighbor], candidate.hash) >
            std::make_tuple(farthest.distance, ids_[farthest.neighbor], farthest.hash)) {
            state.farthest = index;
        }
    }
}

float
PiPNNPipeline::pair_distance(uint32_t lhs, uint32_t rhs) const {
    if (native_half()) {
        return distance_from_dot(
            lhs, rhs, pipnn_half::Dot(half_row(lhs), half_row(rhs), dimensions_, half_format()));
    }
    const auto* left = vector_by_local_id(lhs);
    const auto* right = vector_by_local_id(rhs);
    const float dot = FP32ComputeIP(left, right, dimensions_);
    return distance_from_dot(lhs, rhs, dot);
}

float
PiPNNPipeline::distance_from_dot(uint32_t lhs, uint32_t rhs, float dot) const {
    if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        const float sum = norms_[lhs] + norms_[rhs];
        return std::max(0.0F, sanitize_distance(sum - dot - dot));
    }
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        return sanitize_distance(1.0F - dot);
    }
    const float similarity = std::clamp(dot * norms_[lhs] * norms_[rhs], -1.0F, 1.0F);
    return sanitize_distance(1.0F - similarity);
}

void
PiPNNPipeline::robust_prune(uint32_t source,
                            const ReservoirEntry* row,
                            uint16_t size,
                            PruneScratch& scratch) const {
    auto& ordered = scratch.ordered;
    auto& selected = scratch.selected;
    auto& result = scratch.neighbors;
    ordered.clear();
    selected.clear();
    result.clear();
    ordered.reserve(size);
    for (uint16_t index = 0; index < size; ++index) {
        ordered.emplace_back(distance_from_order_key(row[index].distance), row[index].neighbor);
    }
    auto comparator = [&](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        const auto source_id = ids_[source];
        const auto lhs_gap = id_gap(source_id, ids_[lhs.second]);
        const auto rhs_gap = id_gap(source_id, ids_[rhs.second]);
        if (lhs_gap != rhs_gap) {
            return lhs_gap < rhs_gap;
        }
        return ids_[lhs.second] < ids_[rhs.second];
    };
    std::sort(ordered.begin(), ordered.end(), comparator);

    const uint64_t max_degree = graph_->MaximumDegree();
    // Use the original arithmetic path for a zero step as well. The adaptive
    // callback boundary can otherwise change floating-point rounding at a pruning tie.
    if (parameter_.adaptive_pruning and parameter_.adaptive_pruning_adjust_step != 0.0F) {
        const auto positions = select_pipnn_edges_adaptive(
            ordered,
            max_degree,
            parameter_.alpha,
            parameter_.adaptive_pruning_adjust_step,
            [this](uint32_t lhs, uint32_t rhs) { return pair_distance(lhs, rhs); },
            allocator_);
        result.reserve(positions.size());
        for (const auto position : positions) {
            result.emplace_back(ids_[ordered[position].second]);
        }
        return;
    }
    selected.reserve(std::min<uint64_t>(ordered.size(), max_degree));
    for (uint64_t candidate_index = 0; candidate_index < ordered.size(); ++candidate_index) {
        constexpr uint64_t prefetch_distance = 4;
        if (candidate_index + prefetch_distance < ordered.size()) {
            const auto local_id = ordered[candidate_index + prefetch_distance].second;
            if (native_half()) {
                const auto* future = reinterpret_cast<const char*>(half_row(local_id));
                const uint64_t bytes = dimensions_ * sizeof(uint16_t);
                for (uint64_t offset = 0; offset < bytes; offset += 64) {
                    Prefetch(future + offset);
                }
            } else {
                const auto* future = vector_by_local_id(local_id);
                for (uint64_t dim = 0; dim < dimensions_; dim += 16) {
                    Prefetch(future + dim);
                }
            }
        }
        const auto& [source_distance, candidate] = ordered[candidate_index];
        bool keep = true;
        for (const auto neighbor : selected) {
            if (parameter_.alpha * pair_distance(neighbor, candidate) < source_distance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.emplace_back(candidate);
            if (selected.size() == max_degree) {
                break;
            }
        }
    }

    result.reserve(selected.size());
    for (const auto neighbor : selected) {
        result.emplace_back(ids_[neighbor]);
    }
}

bool
PiPNNPipeline::certify_cached_graph(uint32_t source, PruneScratch& scratch) const {
    if (parameter_.adaptive_pruning or cached_rows_[source] == 0) {
        return false;
    }
    const auto& cache = *imported_cache_;
    const auto old = new_to_old_[source];
    const auto begin = cache.graph_offsets[old];
    const auto end = cache.graph_offsets[old + 1];
    if (begin == end) {
        return false;
    }
    const auto* row = reservoir_row_or_null(source);
    const auto size = reservoir_states_[source].size;
    if (row == nullptr or size == 0) {
        return false;
    }
    scratch.certified_edges.clear();
    scratch.selected_flags.assign(size, 0);
    const auto source_new_id = ids_[source];
    const auto source_old_id = cache.inner_ids[old];
    auto less = [](uint64_t lhs_gap,
                   InnerIdType lhs_id,
                   uint16_t lhs_key,
                   uint64_t rhs_gap,
                   InnerIdType rhs_id,
                   uint16_t rhs_key) {
        return std::tie(lhs_key, lhs_gap, lhs_id) < std::tie(rhs_key, rhs_gap, rhs_id);
    };
    constexpr uint32_t missing = std::numeric_limits<uint32_t>::max();
    for (uint64_t edge = begin; edge < end; ++edge) {
        const auto old_selected = cache.graph_targets[edge];
        const auto selected = old_to_new_[old_selected];
        if (selected == missing) {
            return false;
        }
        const auto distance = cache.graph_distances[edge];
        uint16_t found_pos = size;
        for (uint16_t pos = 0; pos < size; ++pos) {
            if (row[pos].neighbor == selected and row[pos].distance == distance) {
                found_pos = pos;
                break;
            }
        }
        if (found_pos == size) {
            return false;
        }
        scratch.selected_flags[found_pos] = 1;
        const auto new_id = ids_[selected];
        const auto new_gap = id_gap(source_new_id, new_id);
        const auto old_id = cache.inner_ids[old_selected];
        const auto old_gap = id_gap(source_old_id, old_id);
        if (not scratch.certified_edges.empty()) {
            const auto& prev = scratch.certified_edges.back();
            if (not less(prev.new_gap, prev.new_id, prev.distance, new_gap, new_id, distance)) {
                return false;
            }
        }
        scratch.certified_edges.push_back(
            CertifiedEdge{selected, old_selected, distance, new_gap, new_id, old_gap, old_id});
    }
    const auto& last_edge = scratch.certified_edges.back();
    const bool full_degree = scratch.certified_edges.size() == graph_->MaximumDegree();
    const auto old_begin = cache.targets.begin() + static_cast<int64_t>(cache.offsets[old]);
    const auto old_end = cache.targets.begin() + static_cast<int64_t>(cache.offsets[old + 1]);
    for (uint64_t i = 0; i < size; ++i) {
        if (scratch.selected_flags[i] != 0) {
            continue;
        }
        const auto candidate = row[i].neighbor;
        const auto distance = row[i].distance;
        const auto cand_new_id = ids_[candidate];
        const auto cand_new_gap = id_gap(source_new_id, cand_new_id);
        if (full_degree and less(last_edge.new_gap,
                                 last_edge.new_id,
                                 last_edge.distance,
                                 cand_new_gap,
                                 cand_new_id,
                                 distance)) {
            continue;
        }
        // Previously rejected candidates retain their proof if their relative order
        // against every selected neighbor survives inner-ID remapping.
        const auto old_target = new_to_old_[candidate];
        bool old_proof = false;
        if (old_target != missing) {
            const auto found = std::lower_bound(old_begin, old_end, old_target);
            old_proof = found != old_end and *found == old_target;
            if (old_proof) {
                const auto position = static_cast<uint64_t>(found - cache.targets.begin());
                old_proof = cache.distances[position] == distance;
            }
            if (old_proof) {
                const auto cand_old_id = cache.inner_ids[old_target];
                const auto cand_old_gap = id_gap(source_old_id, cand_old_id);
                for (const auto& edge : scratch.certified_edges) {
                    const bool earlier = less(edge.old_gap,
                                              edge.old_id,
                                              edge.distance,
                                              cand_old_gap,
                                              cand_old_id,
                                              distance);
                    const bool new_earlier = less(edge.new_gap,
                                                  edge.new_id,
                                                  edge.distance,
                                                  cand_new_gap,
                                                  cand_new_id,
                                                  distance);
                    if (earlier != new_earlier) {
                        old_proof = false;
                        break;
                    }
                }
            }
        }
        if (old_proof) {
            continue;
        }
        if (not native_half()) {
            const auto* future = vector_by_local_id(candidate);
            for (uint64_t dim = 0; dim < dimensions_; dim += 16) {
                Prefetch(future + dim);
            }
        }
        bool blocked = false;
        const float cand_dist = distance_from_order_key(distance);
        for (const auto& edge : scratch.certified_edges) {
            if (less(edge.new_gap,
                     edge.new_id,
                     edge.distance,
                     cand_new_gap,
                     cand_new_id,
                     distance) and
                parameter_.alpha * pair_distance(edge.target, candidate) < cand_dist) {
                blocked = true;
                break;
            }
        }
        if (not blocked) {
            return false;
        }
    }
    return true;
}

void
PiPNNPipeline::write_graph_row(uint32_t source, PruneScratch& scratch) {
    // Adaptive thresholds depend on rejected-candidate counts as well as selected edges.
    // Recompute them even when all previously selected neighbors survive.
    bool reuse =
        reuse_graph_ and not parameter_.adaptive_pruning and clean_graph_rows_[source] != 0;
    if (not reuse) {
        seed_cached_row(source);
        reuse = reuse_graph_ and certify_cached_graph(source, scratch);
        if (reuse) {
            certificate_rows_[source] = 1;
        }
    }
    if (reuse) {
        scratch.neighbors.clear();
        const auto& cache = *imported_cache_;
        const auto old = new_to_old_[source];
        for (uint64_t edge = cache.graph_offsets[old]; edge < cache.graph_offsets[old + 1];
             ++edge) {
            scratch.neighbors.push_back(ids_[old_to_new_[cache.graph_targets[edge]]]);
        }
        reused_graph_rows_[source] = 1;
    } else {
        const auto size = reservoir_states_.empty() ? uint16_t{0} : reservoir_states_[source].size;
        const auto* row = size == 0 ? nullptr : reservoir_row_or_null(source);
        robust_prune(source, row, size, scratch);
    }
    graph_->InsertNeighborsById(ids_[source], scratch.neighbors);
}

void
PiPNNPipeline::write_graph(const Vector<uint32_t>& representatives) {
    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        PruneScratch scratch(allocator_);
        for (uint64_t source = begin; source < end; ++source) {
            if (not representatives.empty() and representatives[source] != source) {
                continue;
            }
            write_graph_row(static_cast<uint32_t>(source), scratch);
        }
    });
}

void
PiPNNPipeline::parallel_for(uint64_t total,
                            uint64_t block_size,
                            const std::function<void(uint64_t, uint64_t)>& task) const {
    if (total == 0) {
        return;
    }
    if (thread_pool_ == nullptr or thread_count_ <= 1 or total <= block_size) {
        task(0, total);
        return;
    }

    std::atomic<uint64_t> next{0};
    const uint64_t block_count = (total + block_size - 1) / block_size;
    const uint64_t worker_count = std::min(thread_count_, block_count);
    Vector<std::future<void>> futures(allocator_);
    futures.reserve(worker_count);
    std::exception_ptr first_exception;
    try {
        for (uint64_t worker = 0; worker < worker_count; ++worker) {
            futures.emplace_back(thread_pool_->GeneralEnqueue([&]() {
                while (true) {
                    const uint64_t begin = next.fetch_add(block_size, std::memory_order_relaxed);
                    if (begin >= total) {
                        return;
                    }
                    task(begin, std::min(begin + block_size, total));
                }
            }));
        }
    } catch (...) {
        first_exception = std::current_exception();
    }

    // Submitted workers borrow stack and pipeline state, even when a later submission fails.
    // Drain them before unwinding either lifetime.
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (first_exception == nullptr) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

}  // namespace

void
PiPNNGraphBuilderParameter::FromJson(const JsonType& json) {
    adaptive_pruning = json.Contains(PIPNN_PARAMETER_ADAPTIVE_PRUNING)
                           ? json[PIPNN_PARAMETER_ADAPTIVE_PRUNING].GetBool()
                           : false;
    adaptive_pruning_adjust_step =
        json.Contains(PIPNN_PARAMETER_ADAPTIVE_PRUNING_ADJUST_STEP)
            ? json[PIPNN_PARAMETER_ADAPTIVE_PRUNING_ADJUST_STEP].GetFloat()
            : 0.06F;
    auto read_uint = [&](const char* key, uint64_t& target) {
        if (not json.Contains(key)) {
            return;
        }
        require_argument(json[key].IsNumberUnsigned(),
                         fmt::format("PiPNN {} must be a non-negative integer", key));
        target = json[key].GetUint64();
    };
    read_uint(PIPNN_PARAMETER_MAX_LEAF_SIZE, max_leaf_size);
    read_uint(PIPNN_PARAMETER_MIN_LEAF_SIZE, min_leaf_size);
    read_uint(PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT, leaf_neighbor_count);
    read_uint(PIPNN_PARAMETER_HASH_PLANE_COUNT, hash_plane_count);
    read_uint(PIPNN_PARAMETER_RESERVOIR_SIZE, reservoir_size);

    if (json.Contains(PIPNN_PARAMETER_LEADER_SAMPLE_RATE)) {
        require_argument(json[PIPNN_PARAMETER_LEADER_SAMPLE_RATE].IsNumber(),
                         "PiPNN pipnn_leader_sample_rate must be a number");
        leader_sample_rate = json[PIPNN_PARAMETER_LEADER_SAMPLE_RATE].GetFloat();
    }
    if (json.Contains(PIPNN_PARAMETER_FANOUT)) {
        const auto fanout_json = json[PIPNN_PARAMETER_FANOUT];
        require_argument(fanout_json.IsArray(), "PiPNN pipnn_fanout must be an array");
        fanout.clear();
        for (const auto& value : *fanout_json.GetInnerJson()) {
            const bool is_positive_signed = value.is_number_integer() and
                                            not value.is_number_unsigned() and
                                            value.get<int64_t>() > 0;
            const bool is_positive_unsigned =
                value.is_number_unsigned() and value.get<uint64_t>() > 0;
            require_argument(is_positive_signed or is_positive_unsigned,
                             "PiPNN pipnn_fanout values must be positive integers");
            fanout.emplace_back(value.get<uint64_t>());
        }
    }
}

JsonType
PiPNNGraphBuilderParameter::ToJson() const {
    JsonType json;
    json[PIPNN_PARAMETER_MAX_LEAF_SIZE].SetUint64(max_leaf_size);
    json[PIPNN_PARAMETER_MIN_LEAF_SIZE].SetUint64(min_leaf_size);
    json[PIPNN_PARAMETER_LEADER_SAMPLE_RATE].SetFloat(leader_sample_rate);
    *json[PIPNN_PARAMETER_FANOUT].GetInnerJson() = fanout;
    json[PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT].SetUint64(leaf_neighbor_count);
    json[PIPNN_PARAMETER_HASH_PLANE_COUNT].SetUint64(hash_plane_count);
    json[PIPNN_PARAMETER_RESERVOIR_SIZE].SetUint64(reservoir_size);
    json[PIPNN_PARAMETER_ADAPTIVE_PRUNING].SetBool(adaptive_pruning);
    json[PIPNN_PARAMETER_ADAPTIVE_PRUNING_ADJUST_STEP].SetFloat(adaptive_pruning_adjust_step);
    return json;
}

void
PiPNNGraphBuilderParameter::Validate(uint64_t max_degree) const {
    require_argument(max_leaf_size >= 2, "PiPNN max_leaf_size must be at least 2");
    require_argument(min_leaf_size > 0, "PiPNN min_leaf_size must be positive");
    require_argument(min_leaf_size <= max_leaf_size,
                     "PiPNN min_leaf_size must not exceed max_leaf_size");
    require_argument(max_leaf_size <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                     "PiPNN max_leaf_size exceeds the BLAS limit");
    require_argument(std::isfinite(leader_sample_rate) and leader_sample_rate > 0.0F and
                         leader_sample_rate <= 1.0F,
                     "PiPNN leader_sample_rate must be in (0, 1]");
    require_argument(not fanout.empty(), "PiPNN fanout must not be empty");
    require_argument(
        std::all_of(fanout.begin(), fanout.end(), [](uint64_t value) { return value > 0; }),
        "PiPNN fanout values must be positive");
    require_argument(leaf_neighbor_count > 0, "PiPNN leaf_neighbor_count must be positive");
    require_argument(hash_plane_count > 0 and hash_plane_count <= MAX_HASH_PLANES,
                     "PiPNN hash_plane_count must be in [1, 15]");
    require_argument(reservoir_size > 0, "PiPNN reservoir_size must be positive");
    require_argument(std::isfinite(alpha) and alpha >= 1.0F,
                     "PiPNN alpha must be finite and at least 1");
    if (adaptive_pruning) {
        validate_pipnn_adaptive_pruning(alpha, adaptive_pruning_adjust_step);
    }
    require_argument(max_degree > 0, "PiPNN graph degree must be positive");
    const uint64_t hash_capacity = 1ULL << hash_plane_count;
    require_argument(max_degree <= hash_capacity,
                     "PiPNN graph degree exceeds the relative-hash capacity");
    require_argument(std::max(reservoir_size, max_degree) <=
                         static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()),
                     "PiPNN reservoir exceeds the supported size");
}

PiPNNGraphBuilder::PiPNNGraphBuilder(PiPNNGraphBuilderParameter parameter,
                                     uint64_t dimensions,
                                     MetricType metric,
                                     Allocator* allocator,
                                     SafeThreadPool* thread_pool,
                                     uint64_t thread_count)
    : parameter_(std::move(parameter)),
      dimensions_(dimensions),
      metric_(metric),
      allocator_(allocator),
      thread_pool_(thread_pool),
      thread_count_(std::max<uint64_t>(1, thread_count)) {
    require_argument(dimensions_ > 0, "PiPNN dimensions must be positive");
    require_argument(dimensions_ <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                     "PiPNN dimensions exceed the BLAS limit");
    require_argument(metric_ == MetricType::METRIC_TYPE_L2SQR or
                         metric_ == MetricType::METRIC_TYPE_IP or
                         metric_ == MetricType::METRIC_TYPE_COSINE,
                     "PiPNN metric is not supported");
    require_argument(allocator_ != nullptr, "PiPNN allocator must not be null");
    require_argument(not parameter_.adaptive_pruning or metric_ == MetricType::METRIC_TYPE_L2SQR,
                     "PiPNN adaptive pruning only supports L2");
}

void
PiPNNGraphBuilder::Build(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids_sequence,
                         const Vector<const float*>& rows,
                         const std::vector<std::string>* source_ids,
                         const PiPNNBuildCache* imported_cache,
                         std::shared_ptr<PiPNNBuildCache>* exported_cache) const {
    Build(graph,
          ids_sequence,
          rows,
          source_ids == nullptr ? nullptr : source_ids->data(),
          source_ids == nullptr ? 0 : source_ids->size(),
          imported_cache,
          exported_cache);
}

void
PiPNNGraphBuilder::Build(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids_sequence,
                         const Vector<const float*>& rows,
                         const std::string* source_ids,
                         uint64_t source_id_count,
                         const PiPNNBuildCache* imported_cache,
                         std::shared_ptr<PiPNNBuildCache>* exported_cache) const {
    // Keep the imported snapshot alive in the caller when input and output share storage.
    build_typed(graph,
                ids_sequence,
                rows,
                DataTypes::DATA_TYPE_FLOAT,
                source_ids,
                source_id_count,
                imported_cache,
                exported_cache);
}

void
PiPNNGraphBuilder::Build(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids_sequence,
                         const Vector<const uint16_t*>& rows,
                         DataTypes data_type) const {
    require_argument(
        data_type == DataTypes::DATA_TYPE_FP16 or data_type == DataTypes::DATA_TYPE_BF16,
        "PiPNN native half rows require FP16 or BF16");
    build_typed(graph, ids_sequence, rows, data_type, nullptr, 0, nullptr, nullptr);
}

template <typename ElementType>
void
PiPNNGraphBuilder::build_typed(const GraphInterfacePtr& graph,
                               const Vector<InnerIdType>& ids_sequence,
                               const Vector<const ElementType*>& rows,
                               DataTypes data_type,
                               const std::string* source_ids,
                               uint64_t source_id_count,
                               const PiPNNBuildCache* imported_cache,
                               std::shared_ptr<PiPNNBuildCache>* exported_cache) const {
    require_argument(graph != nullptr, "PiPNN graph must not be null");
    parameter_.Validate(graph->MaximumDegree());
    const auto build_pipeline = [&](const Vector<InnerIdType>& ids,
                                    const Vector<const ElementType*>& input_rows,
                                    bool use_cache,
                                    DuplicatePairs* near_duplicates) {
        if constexpr (std::is_same_v<ElementType, float>) {
            PiPNNPipeline pipeline(parameter_,
                                   dimensions_,
                                   metric_,
                                   allocator_,
                                   thread_pool_,
                                   thread_count_,
                                   graph,
                                   input_rows);
            if (use_cache) {
                pipeline.ConfigureCache(
                    source_ids, source_id_count, imported_cache, exported_cache);
            }
            pipeline.Build(ids, near_duplicates);
        } else {
            PiPNNPipeline(parameter_,
                          dimensions_,
                          metric_,
                          allocator_,
                          thread_pool_,
                          thread_count_,
                          graph,
                          input_rows,
                          data_type)
                .Build(ids, near_duplicates);
        }
    };
    if (graph->GetDuplicateTracker() == nullptr) {
        build_pipeline(ids_sequence, rows, true, nullptr);
        return;
    }

    if (exported_cache != nullptr) {
        exported_cache->reset();  // Duplicate-group reservoirs require a separate cache format.
    }
    require_argument(ids_sequence.size() == rows.size(),
                     "PiPNN IDs and rows must have equal sizes");
    require_argument(ids_sequence.size() <= std::numeric_limits<uint32_t>::max(),
                     "PiPNN point count exceeds the local ID limit");
    UnorderedSet<InnerIdType> seen_ids(allocator_);
    seen_ids.reserve(ids_sequence.size());
    for (uint64_t local_id = 0; local_id < ids_sequence.size(); ++local_id) {
        require_argument(ids_sequence[local_id] < graph->MaxCapacity(),
                         "PiPNN input ID is outside the graph capacity");
        require_argument(seen_ids.emplace(ids_sequence[local_id]).second,
                         "PiPNN input IDs must be unique");
        require_argument(rows[local_id] != nullptr, "PiPNN input rows must not be null");
    }
    Vector<InnerIdType> representative_ids(allocator_);
    Vector<const ElementType*> representative_rows(allocator_);
    DuplicatePairs duplicates(allocator_);
    DuplicatePairs near_duplicates(allocator_);
    const bool use_threshold =
        metric_ == MetricType::METRIC_TYPE_L2SQR and parameter_.duplicate_distance_threshold > 0.0F;
    Vector<uint32_t> order(allocator_);
    if (use_threshold and not std::is_sorted(ids_sequence.begin(), ids_sequence.end())) {
        order.resize(ids_sequence.size());
        std::iota(order.begin(), order.end(), uint32_t{0});
        std::sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
            return ids_sequence[lhs] < ids_sequence[rhs];
        });
    }
    group_duplicate_rows(ids_sequence,
                         order,
                         rows,
                         dimensions_,
                         allocator_,
                         representative_ids,
                         representative_rows,
                         duplicates);
    build_pipeline(
        representative_ids, representative_rows, false, use_threshold ? &near_duplicates : nullptr);
    UnorderedMap<InnerIdType, InnerIdType> final_representatives(allocator_);
    if (not duplicates.empty() and not near_duplicates.empty()) {
        final_representatives.reserve(near_duplicates.size());
    }
    for (const auto& [representative, duplicate] : near_duplicates) {
        graph->SetDuplicateId(representative, duplicate);
        if (not duplicates.empty()) {
            final_representatives.emplace(duplicate, representative);
        }
    }
    for (const auto& [representative, duplicate] : duplicates) {
        const auto it = final_representatives.find(representative);
        graph->SetDuplicateId(it == final_representatives.end() ? representative : it.value(),
                              duplicate);
    }
}

}  // namespace vsag
