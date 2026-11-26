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
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "common.h"
#include "datacell/flatten_interface.h"
#include "datacell/flatten_pairwise_distance_interface.h"
#include "datacell/graph_interface.h"
#include "half_vector.h"
#include "hash_types.h"
#include "impl/blas/blas_function.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "pipnn_adaptive_pruning.h"
#include "pipnn_build_cache.h"
#include "pipnn_incremental.h"
#include "simd/basic_func.h"
#include "simd/bf16_simd.h"
#include "simd/fp32_simd.h"
#include "simd/pipnn_filter.h"
#include "simd/pipnn_ip_tile.h"
#include "utils/lock_strategy.h"

namespace vsag {
namespace {

constexpr uint64_t MAX_HASH_PLANES = 15;
constexpr uint64_t MAX_POINT_LOCK_COUNT = 1U << 16;
using pipnn_filter::COINCIDENT_HASH_FLAG;
constexpr uint64_t MAX_PARTITION_ITERATIONS = 30;
constexpr uint64_t LEADER_CAP = 1000;
constexpr uint64_t PARTITION_STRIPE_SIZE = 256;
constexpr uint64_t PARTITION_ASSIGNMENT_CHUNK_STRIPES = 16;
constexpr uint64_t PARTITION_SEED = 1000;
constexpr uint64_t MIN_PARALLEL_PARTITION_POINTS = 4096;
constexpr uint64_t SMALL_TOP_K_LIMIT = 16;
constexpr uint64_t LEAF_DOT_STRIP = 64;
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

struct AddCandidate {
    uint32_t source;
    ReservoirEntry entry;
};
static_assert(sizeof(AddCandidate) == 12);

struct WorkItem {
    WorkItem(Vector<uint32_t>&& input_points,
             uint64_t input_level,
             uint64_t input_seed,
             PiPNNRoute* trace = nullptr)
        : route(trace), points(std::move(input_points)), level(input_level), seed(input_seed) {
    }

    WorkItem(WorkItem&&) = default;

    WorkItem&
    operator=(WorkItem&& other) noexcept(false) = default;

    PiPNNRoute* route;
    Vector<uint32_t> points;
    uint64_t level{0};
    uint64_t seed{0};
};

struct Leaf : Vector<uint32_t> {
    using Vector<uint32_t>::Vector;
    std::vector<PiPNNRoute*> terminals;
};
using Leaves = Vector<Leaf>;
using LeafAdditions = std::map<uint32_t, Leaf>;

enum class AssignmentMode { Serial, BuildChunks, SplitStripes, RouteRoot };

struct SplitResult {
    explicit SplitResult(Allocator* allocator) : pending(allocator), finished(allocator) {
    }

    Vector<WorkItem> pending;
    Leaves finished;
};

struct LeafSplit {
    uint64_t parent;
    PiPNNRoute replacement;
    Leaves children;
    uint64_t end_child{0};
};

struct PruneScratch {
    explicit PruneScratch(Allocator* allocator)
        : ordered(allocator), selected(allocator), neighbors(allocator) {
    }

    Vector<std::pair<float, uint32_t>> ordered;
    Vector<uint32_t> selected;
    Vector<InnerIdType> neighbors;
};

struct EdgeUpdateScratch {
    explicit EdgeUpdateScratch(Allocator* allocator)
        : incoming(allocator), incumbents(allocator), neighbors(allocator) {
    }

    Vector<ReservoirEntry> incoming;
    Vector<ReservoirEntry> incumbents;
    Vector<InnerIdType> neighbors;
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

// Only ordinary leaves with unchanged top-k width need an old-row bitmap.
struct ReplayRows {
    uint64_t old_count{0};
    std::vector<uint64_t> changed;
};

// Split children and leaves with a new top-k width replay every row. Otherwise only
// changed old rows and the appended rows can contain a member of the new batch.
template <typename Submit>
void
visit_leaf_candidates(const PiPNNCachedLeaf& leaf, const ReplayRows* changes, Submit submit) {
    if (leaf.nearest.Width() == 0) {
        return;
    }
    auto visit_row = [&](uint64_t row) {
        for (uint32_t rank = 0; rank < leaf.nearest.Width(); ++rank) {
            submit(leaf.points[row], leaf.Neighbor(row, rank));
        }
    };
    uint64_t first = 0;
    if (changes != nullptr) {
        for (uint64_t word = 0; word < changes->changed.size(); ++word) {
            auto bits = changes->changed[word];
            while (bits != 0) {
                visit_row(word * 64 + std::countr_zero(bits));
                bits &= bits - 1;
            }
        }
        first = changes->old_count;
    }
    for (uint64_t row = first; row < leaf.points.size(); ++row) {
        visit_row(row);
    }
}

// The source's local ID determines the owner even when physical graph IDs are sparse.
// Preserve arrival order within each source in both grouping paths.
Vector<AddCandidate>
group_add_candidates(const std::vector<Vector<AddCandidate>>& buffers,
                     uint64_t shard,
                     uint64_t shards,
                     uint64_t source_count,
                     Allocator* allocator) {
    uint64_t count = 0;
    for (uint64_t i = shard; i < buffers.size(); i += shards) {
        count += buffers[i].size();
    }
    Vector<AddCandidate> grouped(allocator);
    const uint64_t range = (source_count + shards - 1 - shard) / shards;
    // Sparse shards sort just their records. Dense shards use a linear counting sort.
    if (count < (range + 7) / 8) {
        grouped.reserve(count);
        for (uint64_t i = shard; i < buffers.size(); i += shards) {
            grouped.insert(grouped.end(), buffers[i].begin(), buffers[i].end());
        }
        std::stable_sort(grouped.begin(), grouped.end(), [](const auto& a, const auto& b) {
            return a.source < b.source;
        });
    } else {
        Vector<uint64_t> offsets(range, 0, allocator);
        for (uint64_t i = shard; i < buffers.size(); i += shards) {
            for (const auto& event : buffers[i]) {
                ++offsets[event.source / shards];
            }
        }
        std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), uint64_t{0});
        grouped.resize(count);
        for (uint64_t i = shard; i < buffers.size(); i += shards) {
            for (const auto& event : buffers[i]) {
                grouped[offsets[event.source / shards]++] = event;
            }
        }
    }
    return grouped;
}

// Materialize a distance tile from stored codes. For leaf tiles, first_diagonal is
// the position of the first source in targets; each unordered pair is visited once.
void
code_distances(FlattenInterface& codes,
               std::span<const InnerIdType> sources,
               std::span<const InnerIdType> targets,
               float* output,
               std::optional<uint64_t> first_diagonal = std::nullopt) {
    if (auto* tiled = dynamic_cast<const FlattenPairwiseDistanceInterface*>(&codes);
        tiled && tiled->ComputePairwiseTile(sources, targets, output, first_diagonal)) {
        return;
    }
    for (uint64_t row = 0; row < sources.size(); ++row) {
        const auto end = first_diagonal ? *first_diagonal + row : targets.size();
        if (end)
            codes.QueryById(output + row * targets.size(), sources[row], targets.data(), end);
    }
}

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
                  std::span<const float* const> rows)
        : parameter_(parameter),
          allocator_(allocator),
          graph_(std::move(graph)),
          thread_pool_(thread_pool),
          thread_count_(thread_count),
          dimensions_(dimensions),
          metric_(metric),
          rows_(rows),
          fingerprints_(allocator),
          cached_rows_(allocator),
          reservoirs_(allocator),
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
          reservoirs_(allocator),
          reservoir_states_(allocator) {
    }

    void
    SetCodeDistances(FlattenInterface* codes, const PiPNNProjectionState* shared) {
        codes_ = codes;
        shared_sketches_ = shared;
    }
    void
    AppendProjections(PiPNNProjectionState& state) {
        prepare_sketches(&state, state.projections.Size(), rows_);
    }
    void
    ConfigureCache(const std::vector<std::string>* source_ids,
                   const PiPNNBuildCache* imported,
                   std::shared_ptr<PiPNNBuildCache>* exported) {
        source_ids_ = source_ids;
        imported_cache_ = imported;
        exported_cache_ = exported;
    }

    void
    Build(const Vector<InnerIdType>& ids_sequence, DuplicatePairs* duplicates);
    void
    Increment(PiPNNIncrementalState& state, size_t old_count);
    void
    BuildCached(PiPNNIncrementalState& state);
    void
    route_batch(const PiPNNRoute&, Leaf, LeafAdditions&, PiPNNIncrementalState&) const;
    void
    extend_leaf(PiPNNCachedLeaf&, const Leaf&, ReplayRows&) const;
    void
    cache_partition(Leaves&, PiPNNIncrementalState&);
    void
    rebuild_split_leaves(std::vector<LeafSplit>& splits,
                         const LeafAdditions& additions,
                         PiPNNIncrementalState& state);
    void
    replay_add(const PiPNNIncrementalState& state,
               uint64_t first_new,
               std::span<const size_t> hit_leaves,
               std::span<const ReplayRows> rows);
    void
    merge_candidate(ReservoirState& state, ReservoirEntry* row, const ReservoirEntry& candidate);
    void
    emit(const Leaf& leaf,
         uint32_t source,
         uint32_t rank,
         uint32_t target,
         float distance,
         PiPNNLeafTopK* cache);

private:
    void
    update_existing_edges(uint32_t source,
                          uint64_t first_new,
                          const PiPNNIncrementalState& state,
                          ReservoirState& candidates,
                          ReservoirEntry* row,
                          EdgeUpdateScratch& scratch);

    void
    prepare_build_reservoirs();
    void
    validate_build_input() const;

    void
    prepare_sketches(PiPNNProjectionState* state = nullptr,
                     size_t start = 0,
                     std::span<const float* const> batch = {});

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
    partition(PiPNNRoute* root = nullptr,
              const Leaf* subset = nullptr,
              uint64_t level = 0,
              bool allow_parallelism = true) const;

    void
    split_work_item(const WorkItem& item,
                    Vector<WorkItem>& pending,
                    Leaves& finished,
                    bool allow_inner_parallelism,
                    bool incremental_split = false) const;

    [[nodiscard]] Vector<uint32_t>
    sample_leaders(const WorkItem& item) const;

    [[nodiscard]] Leaves
    assign_to_leaders(const Vector<uint32_t>& points,
                      std::span<const uint32_t> leaders,
                      uint64_t fanout,
                      AssignmentMode mode,
                      std::vector<float>* root_vectors = nullptr) const;

    void
    select_leaders(uint32_t source,
                   std::span<const uint32_t> leaders,
                   const float* dots,
                   uint64_t fanout,
                   Vector<std::pair<float, uint32_t>>& candidates) const;

    [[nodiscard]] Leaves
    merge_undersized_leaves(Leaves leaves) const;

    uint64_t
    leaf_neighbor_count(uint64_t points) const;
    bool
    cache_split_rows(const Leaves& leaves, std::vector<PiPNNCachedLeaf>& caches);
    void
    select_leaf_neighbors(const Leaf& leaf,
                          PiPNNLeafTopK* cache,
                          const float* distances,
                          uint64_t neighbor_count,
                          uint64_t begin,
                          uint64_t end);
    void
    build_leaf(const Leaf& leaf, PiPNNLeafTopK* cache = nullptr);

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
    build_small_leaf(const Leaf& leaf,
                     UninitializedFloatBuffer& matrix,
                     uint64_t neighbor_count,
                     PiPNNLeafTopK* cache);

    void
    insert_pair(uint32_t source, uint32_t target, float distance);

    [[nodiscard]] pipnn_filter::RelativeHashes
    relative_hashes(uint32_t source, uint32_t target) const;

    void
    update_farthest(ReservoirState& state, const ReservoirEntry* row);

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
    write_graph(std::span<const uint32_t> representatives = {});

    void
    seed_cached_row(uint32_t source);

    [[nodiscard]] bool
    certify_cached_graph(uint32_t source) const;

    void
    write_graph_row(uint32_t source, PruneScratch& scratch);

    void
    parallel_for(uint64_t total,
                 uint64_t block_size,
                 const std::function<void(uint64_t, uint64_t)>& task) const;

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
    gather_row(uint32_t local_id, float* buffer, uint64_t row) const {
        if (native_half()) {
            const auto* source = half_row(local_id);
            auto* destination = buffer + row * dimensions_;
            pipnn_half::DecodeRow(source, destination, dimensions_, half_format());
        } else {
            const auto* source = vector_by_local_id(local_id);
            std::copy(source,
                      source + static_cast<int64_t>(dimensions_),
                      buffer + static_cast<int64_t>(row * dimensions_));
        }
    }

    [[nodiscard]] const float*
    vector_by_local_id(uint32_t local_id) const {
        return rows_[local_id];
    }

private:
    const pipnn_filter::Kernels& filters_{pipnn_filter::GetKernels()};
    const PiPNNGraphBuilderParameter& parameter_;
    Allocator* allocator_;
    GraphInterfacePtr graph_;
    SafeThreadPool* thread_pool_;
    uint64_t thread_count_;

    uint64_t dimensions_;
    MetricType metric_;
    // Build() is synchronous, so the pipeline borrows rows only for its own lifetime.
    std::span<const float* const> rows_;
    FlattenInterface* codes_{nullptr};
    const PiPNNProjectionState* shared_sketches_{nullptr};
    const Vector<const uint16_t*>* half_rows_{nullptr};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    const std::vector<std::string>* source_ids_{nullptr};
    const PiPNNBuildCache* imported_cache_{nullptr};
    std::shared_ptr<PiPNNBuildCache>* exported_cache_{nullptr};
    Vector<uint64_t> fingerprints_;
    Vector<uint8_t> cached_rows_;
    bool warm_build_{false};
    bool reuse_graph_{false};
    std::vector<uint32_t> old_to_new_;
    std::vector<uint32_t> new_to_old_;
    std::vector<uint8_t> seeded_rows_;
    std::vector<uint8_t> clean_graph_rows_;
    std::vector<uint8_t> reused_graph_rows_;
    std::vector<uint8_t> certificate_rows_;
    uint64_t hyperplane_signature_{14695981039346656037ULL};
    const uint64_t reservoir_size_{
        graph_ ? std::max<uint64_t>(parameter_.reservoir_size, graph_->MaximumDegree()) : 0};
    std::span<const InnerIdType> ids_;
    PiPNNFloatRows owned_norms_, owned_sketches_;
    PiPNNFloatRows* norms_{&owned_norms_};
    PiPNNFloatRows* sketches_{&owned_sketches_};
    Vector<ReservoirEntry> reservoirs_;
    Vector<ReservoirState> reservoir_states_;
    std::unique_ptr<PointsMutex> point_locks_;
    uint32_t point_lock_count_{0};
};

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
    ids_ = ids_sequence;
    validate_build_input();
    prepare_build_reservoirs();
    if (ids_.empty()) {
        return;
    }

    prepare_sketches();
    finish_phase(0);
    restore_cache();
    finish_phase(1);
    auto leaves = partition();
    Vector<uint32_t> representatives(allocator_);
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
    write_graph(representatives);
    finish_phase(4);
    capture_cache();
    finish_phase(5);
    if (exported_cache_ != nullptr and *exported_cache_ != nullptr) {
        (*exported_cache_)->phase_seconds = phase_seconds;
    }
}

std::string
PiPNNPipeline::cache_policy() const {
    return "pipnn-reservoir-v3/" + std::to_string(dimensions_) + "/" +
           std::to_string(static_cast<int>(metric_)) + "/" +
           std::to_string(graph_->MaximumDegree()) + "/" +
           std::to_string(std::bit_cast<uint32_t>(parameter_.alpha)) + "/" +
           std::to_string(hyperplane_signature_) + "/" + parameter_.ToJson().Dump();
}

uint64_t
PiPNNPipeline::fingerprint(uint32_t source) const {
    uint64_t hash = 14695981039346656037ULL;
    for (uint64_t dimension = 0; dimension < dimensions_; ++dimension) {
        const auto bits = std::bit_cast<uint32_t>(rows_[source][dimension]);
        for (uint64_t byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= (bits >> (byte * 8)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

void
PiPNNPipeline::restore_cache() {
    if (exported_cache_ != nullptr) {
        exported_cache_->reset();
    }
    if (source_ids_ == nullptr or source_ids_->size() != ids_.size()) {
        source_ids_ = nullptr;
        return;
    }
    UnorderedMap<std::string, uint32_t> current(allocator_);
    current.reserve(ids_.size());
    fingerprints_.resize(ids_.size());
    for (uint64_t source = 0; source < ids_.size(); ++source) {
        const auto& id = (*source_ids_)[source];
        if (id.empty() or not current.emplace(id, static_cast<uint32_t>(source)).second) {
            source_ids_ = nullptr;
            return;  // Partial/ambiguous source identities take the ordinary cold path.
        }
        fingerprints_[source] = fingerprint(static_cast<uint32_t>(source));
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
    for (uint64_t old = 0; old < cache.source_ids.size(); ++old) {
        const auto it = current.find(cache.source_ids[old]);
        if (it != current.end() and fingerprints_[it->second] == cache.fingerprints[old]) {
            remap[old] = it->second;
        }
    }
    cached_rows_.assign(ids_.size(), 0);
    for (uint64_t old = 0; old < remap.size(); ++old) {
        if (remap[old] == missing or cache.offsets[old] == cache.offsets[old + 1]) {
            continue;
        }
        require_argument(cache.offsets[old + 1] - cache.offsets[old] <= reservoir_size_,
                         "PiPNN cached reservoir exceeds the configured capacity");
        require_argument(
            cache.graph_offsets[old + 1] - cache.graph_offsets[old] <= graph_->MaximumDegree(),
            "PiPNN cached graph exceeds maximum degree");
        uint64_t live = 0;
        for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
            if (remap[cache.targets[edge]] != missing) {
                ++live;
            }
        }
        const uint64_t minimum = std::min<uint64_t>(graph_->MaximumDegree(),
                                                    cache.offsets[old + 1] - cache.offsets[old]);
        if (live >= minimum) {
            cached_rows_[remap[old]] = 1;
        }
    }
    const auto reusable =
        static_cast<uint64_t>(std::count(cached_rows_.begin(), cached_rows_.end(), uint8_t{1}));
    constexpr uint64_t minimum_cache_percent = 80;
    warm_build_ = reusable > 0 and reusable * 100 >= ids_.size() * minimum_cache_percent;
    if (not warm_build_) {
        std::fill(cached_rows_.begin(), cached_rows_.end(), 0);
        return;
    }
    old_to_new_ = std::move(remap);
    new_to_old_.assign(ids_.size(), missing);
    seeded_rows_.assign(ids_.size(), 1);
    clean_graph_rows_.assign(ids_.size(), 0);
    reused_graph_rows_.assign(ids_.size(), 0);
    certificate_rows_.assign(ids_.size(), 0);
    reuse_graph_ = cache.reuse_final_graph;
    for (uint64_t old = 0; old < old_to_new_.size(); ++old) {
        const auto source = old_to_new_[old];
        if (source == missing) {
            continue;
        }
        new_to_old_[source] = static_cast<uint32_t>(old);
        if (cached_rows_[source] == 0) {
            continue;
        }
        seeded_rows_[source] = 0;
        bool complete = cache.graph_offsets[old] != cache.graph_offsets[old + 1];
        for (uint64_t edge = cache.graph_offsets[old]; edge < cache.graph_offsets[old + 1];
             ++edge) {
            if (old_to_new_[cache.graph_targets[edge]] == missing) {
                complete = false;
                break;
            }
        }
        // Removing rejected candidates preserves pruning, but remapping graph IDs may
        // change BF16-distance tie order. Preserve the old proof only if every selected
        // neighbor retains its ordering relative to every surviving tied candidate.
        for (uint64_t candidate = cache.offsets[old];
             complete and candidate < cache.offsets[old + 1];
             ++candidate) {
            const auto old_target = cache.targets[candidate];
            const auto target = old_to_new_[old_target];
            if (target == missing) {
                continue;
            }
            for (uint64_t edge = cache.graph_offsets[old]; edge < cache.graph_offsets[old + 1];
                 ++edge) {
                if (cache.distances[candidate] != cache.graph_distances[edge]) {
                    continue;
                }
                const auto old_selected = cache.graph_targets[edge];
                const auto selected = old_to_new_[old_selected];
                const bool old_before =
                    std::make_pair(id_gap(cache.inner_ids[old], cache.inner_ids[old_selected]),
                                   cache.inner_ids[old_selected]) <
                    std::make_pair(id_gap(cache.inner_ids[old], cache.inner_ids[old_target]),
                                   cache.inner_ids[old_target]);
                const bool new_before =
                    std::make_pair(id_gap(ids_[source], ids_[selected]), ids_[selected]) <
                    std::make_pair(id_gap(ids_[source], ids_[target]), ids_[target]);
                if (old_before != new_before) {
                    complete = false;
                    break;
                }
            }
        }
        clean_graph_rows_[source] = complete ? 1 : 0;
        if (not reuse_graph_) {
            seed_cached_row(source);
        }
    }
}

void
PiPNNPipeline::seed_cached_row(uint32_t source) {
    if (seeded_rows_.empty() or seeded_rows_[source] != 0) {
        return;
    }
    seeded_rows_[source] = 1;
    const auto old = new_to_old_[source];
    const auto& cache = *imported_cache_;
    auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
    auto& state = reservoir_states_[source];
    for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
        const auto target = old_to_new_[cache.targets[edge]];
        if (target != std::numeric_limits<uint32_t>::max()) {
            row[state.size++] = ReservoirEntry(target, cache.hashes[edge], cache.distances[edge]);
        }
    }
    update_farthest(state, row);
}

void
PiPNNPipeline::capture_cache() {
    if (source_ids_ == nullptr or exported_cache_ == nullptr) {
        return;
    }
    auto snapshot = std::make_shared<PiPNNBuildCache>(allocator_);
    snapshot->policy = cache_policy();
    snapshot->source_ids.assign(source_ids_->begin(), source_ids_->end());
    snapshot->fingerprints = std::move(fingerprints_);
    snapshot->inner_ids.assign(ids_.begin(), ids_.end());
    snapshot->offsets.reserve(ids_.size() + 1);
    snapshot->graph_offsets.reserve(ids_.size() + 1);
    snapshot->offsets.push_back(0);
    snapshot->graph_offsets.push_back(0);
    UnorderedMap<InnerIdType, uint32_t> local_ids(allocator_);
    local_ids.reserve(ids_.size());
    for (uint64_t source = 0; source < ids_.size(); ++source) {
        local_ids.emplace(ids_[source], static_cast<uint32_t>(source));
    }
    Vector<ReservoirEntry> entries(allocator_);
    Vector<InnerIdType> adjacency(allocator_);
    entries.reserve(reservoir_size_);
    for (uint64_t source = 0; source < ids_.size(); ++source) {
        entries.clear();
        if (reuse_graph_ and seeded_rows_[source] == 0) {
            const auto old = new_to_old_[source];
            const auto& cache = *imported_cache_;
            for (uint64_t edge = cache.offsets[old]; edge < cache.offsets[old + 1]; ++edge) {
                const auto target = old_to_new_[cache.targets[edge]];
                if (target != std::numeric_limits<uint32_t>::max()) {
                    entries.emplace_back(target, cache.hashes[edge], cache.distances[edge]);
                }
            }
        } else {
            const auto* row = reservoirs_.data() + source * reservoir_size_;
            entries.assign(row, row + reservoir_states_[source].size);
        }
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.neighbor < rhs.neighbor;
        });
        for (const auto& entry : entries) {
            snapshot->targets.push_back(entry.neighbor);
            snapshot->distances.push_back(entry.distance);
            snapshot->hashes.push_back(entry.hash);
        }
        snapshot->offsets.push_back(snapshot->targets.size());
        graph_->GetNeighbors(ids_[source], adjacency);
        for (const auto neighbor : adjacency) {
            const auto target = local_ids.at(neighbor);
            const auto it = std::lower_bound(
                entries.begin(), entries.end(), target, [](const auto& entry, uint32_t id) {
                    return entry.neighbor < id;
                });
            require_argument(it != entries.end() and it->neighbor == target,
                             "PiPNN final neighbor is absent from the reservoir");
            snapshot->graph_targets.push_back(target);
            snapshot->graph_distances.push_back(it->distance);
        }
        snapshot->graph_offsets.push_back(snapshot->graph_targets.size());
        if (warm_build_ and cached_rows_[source] != 0) {
            ++snapshot->reused_rows;
            snapshot->reused_graph_rows += reused_graph_rows_[source];
            snapshot->certificate_rows += certificate_rows_[source];
        }
    }
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
    Vector<uint32_t> missing(allocator_);
    for (const auto source : leaf) {
        if (cached_rows_[source] == 0) {
            missing.push_back(source);
        }
    }
    if (missing.empty()) {
        return;
    }
    // Rectangular GEMM computes only missing rows, retaining BLAS throughput at high overlap.
    UninitializedFloatBuffer points(
        checked_product(leaf.size(), dimensions_, "cache points"), allocator_, "cache points");
    UninitializedFloatBuffer queries(
        checked_product(missing.size(), dimensions_, "cache queries"), allocator_, "cache queries");
    for (uint64_t i = 0; i < leaf.size(); ++i) {
        std::copy_n(rows_[leaf[i]], dimensions_, points.Data() + i * dimensions_);
    }
    for (uint64_t i = 0; i < missing.size(); ++i) {
        std::copy_n(rows_[missing[i]], dimensions_, queries.Data() + i * dimensions_);
    }
    UninitializedFloatBuffer dots(
        checked_product(missing.size(), leaf.size(), "cache dots"), allocator_, "cache dots");
    BlasFunction::Sgemm(BlasFunction::RowMajor,
                        BlasFunction::NoTrans,
                        BlasFunction::Trans,
                        static_cast<int32_t>(missing.size()),
                        static_cast<int32_t>(leaf.size()),
                        static_cast<int32_t>(dimensions_),
                        1.0F,
                        queries.Data(),
                        static_cast<int32_t>(dimensions_),
                        points.Data(),
                        static_cast<int32_t>(dimensions_),
                        0.0F,
                        dots.Data(),
                        static_cast<int32_t>(leaf.size()));
    Vector<std::pair<float, uint32_t>> nearest(allocator_);
    nearest.reserve(k);
    for (uint64_t row = 0; row < missing.size(); ++row) {
        const auto source = missing[row];
        nearest.clear();
        const auto less = [&](const auto& lhs, const auto& rhs) {
            return std::make_tuple(
                       lhs.first, id_gap(ids_[source], ids_[lhs.second]), ids_[lhs.second]) <
                   std::make_tuple(
                       rhs.first, id_gap(ids_[source], ids_[rhs.second]), ids_[rhs.second]);
        };
        for (uint64_t column = 0; column < leaf.size(); ++column) {
            const auto target = leaf[column];
            if (source == target) {
                continue;
            }
            const auto candidate = std::make_pair(
                distance_from_dot(source, target, dots.Data()[row * leaf.size() + column]), target);
            if (nearest.size() < k) {
                nearest.push_back(candidate);
                std::push_heap(nearest.begin(), nearest.end(), less);
            } else if (less(candidate, nearest.front())) {
                std::pop_heap(nearest.begin(), nearest.end(), less);
                nearest.back() = candidate;
                std::push_heap(nearest.begin(), nearest.end(), less);
            }
        }
        std::sort(nearest.begin(), nearest.end(), less);
        for (const auto& [distance, target] : nearest) {
            insert_pair(source, target, distance);
        }
    }
}

void
PiPNNPipeline::validate_build_input() const {
    require_argument(ids_.size() == (native_half() ? half_rows_->size() : rows_.size()),
                     "PiPNN IDs and rows must have equal sizes");
    require_argument(ids_.size() <= std::numeric_limits<uint32_t>::max(),
                     "PiPNN point count exceeds the local ID limit");
    if (ids_.empty()) {
        return;
    }

    const uint64_t graph_capacity = graph_->MaxCapacity();
    const bool ids_are_strictly_increasing =
        std::adjacent_find(ids_.begin(), ids_.end(), [](InnerIdType lhs, InnerIdType rhs) {
            return lhs >= rhs;
        }) == ids_.end();
    UnorderedSet<InnerIdType> seen_ids(allocator_);
    if (not ids_are_strictly_increasing) {
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
}

void
PiPNNPipeline::prepare_build_reservoirs() {
    if (ids_.empty()) {
        return;
    }
    const uint64_t reservoir_count = checked_product(ids_.size(), reservoir_size_, "reservoir");
    require_argument(reservoir_count <= reservoirs_.max_size(),
                     "PiPNN reservoir exceeds the allocator limit");

    reservoirs_.resize(reservoir_count);
    reservoir_states_.resize(ids_.size());
    if (thread_pool_ != nullptr and thread_count_ > 1) {
        point_lock_count_ =
            static_cast<uint32_t>(std::min<uint64_t>(ids_.size(), MAX_POINT_LOCK_COUNT));
        point_locks_ = std::make_unique<PointsMutex>(point_lock_count_, allocator_);
    }
}

void
PiPNNPipeline::prepare_sketches(PiPNNProjectionState* state,
                                size_t start,
                                std::span<const float* const> batch) {
    const auto total = batch.empty() ? ids_.size() : start + batch.size();
    const uint64_t plane_values =
        checked_product(parameter_.hash_plane_count, dimensions_, "hyperplane");
    std::vector<float> local_planes;
    auto& hyperplanes = state ? state->planes : local_planes;
    if (state) {
        sketches_ = &state->projections;
        norms_ = &state->norms;
    }
    if (start == 0) {
        if (shared_sketches_) {
            hyperplanes = shared_sketches_->planes;
        } else if (hyperplanes.empty()) {
            hyperplanes.resize(plane_values);
            std::mt19937_64 random(42);
            std::normal_distribution<float> normal(0.0F, 1.0F);
            for (auto& value : hyperplanes) value = normal(random);
        }
        require_argument(hyperplanes.size() == plane_values, "bad PiPNN projection plane shape");
        for (const auto value : hyperplanes) {
            hyperplane_signature_ ^= std::bit_cast<uint32_t>(value);
            hyperplane_signature_ *= 1099511628211ULL;
        }
    }
    sketches_->Resize(total, parameter_.hash_plane_count, allocator_);
    if (metric_ != MetricType::METRIC_TYPE_IP)
        norms_->Resize(total, 1, allocator_);
    parallel_for(total - start, 64, [&](uint64_t begin, uint64_t end) {
        for (uint64_t local_id = start + begin; local_id < start + end; ++local_id) {
            auto* sketch = sketches_->Row(local_id);
            if (shared_sketches_) {
                const auto id = ids_[local_id];
                std::copy_n(
                    shared_sketches_->projections.Row(id), parameter_.hash_plane_count, sketch);
                if (metric_ != MetricType::METRIC_TYPE_IP)
                    norms_->Scalar(local_id) = shared_sketches_->norms.Row(id)[0];
                continue;
            }
            if (native_half()) {
                const auto* vector = half_row(static_cast<uint32_t>(local_id));
                if (metric_ != MetricType::METRIC_TYPE_IP) {
                    const float norm =
                        pipnn_half::Preparation(vector, nullptr, dimensions_, half_format());
                    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                        const float length = std::sqrt(std::max(0.0F, norm));
                        norms_->Scalar(local_id) =
                            length > 0.0F and std::isfinite(length) ? 1.0F / length : 0.0F;
                    } else {
                        norms_->Scalar(local_id) = norm;
                    }
                }
                for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
                    const auto* hyperplane = hyperplanes.data() + plane * dimensions_;
                    float dot =
                        pipnn_half::Preparation(vector, hyperplane, dimensions_, half_format());
                    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                        dot *= norms_->Scalar(local_id);
                    }
                    sketch[plane] = dot;
                }
                continue;
            }
            const auto* vector = batch.empty() ? vector_by_local_id(static_cast<uint32_t>(local_id))
                                               : batch[local_id - start];
            if (metric_ != MetricType::METRIC_TYPE_IP) {
                float norm = 0.0F;
                for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                    norm += vector[dim] * vector[dim];
                }
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    const float length = std::sqrt(std::max(0.0F, norm));
                    norms_->Scalar(local_id) =
                        length > 0.0F and std::isfinite(length) ? 1.0F / length : 0.0F;
                } else {
                    norms_->Scalar(local_id) = norm;
                }
            }
            for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
                const auto* hyperplane = hyperplanes.data() + plane * dimensions_;
                float dot = 0.0F;
                for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                    dot += vector[dim] * hyperplane[dim];
                }
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    dot *= norms_->Scalar(local_id);
                }
                sketch[plane] = dot;
            }
        }
    });
}

Leaves
PiPNNPipeline::partition(PiPNNRoute* root,
                         const Leaf* subset,
                         uint64_t level,
                         bool allow_parallelism) const {
    Leaf initial(allocator_);
    if (subset)
        initial.assign(subset->begin(), subset->end());
    else {
        initial.resize(ids_.size());
        for (uint64_t i = 0; i < ids_.size(); ++i) {
            initial[i] = static_cast<uint32_t>(i);
        }
    }
    Leaves finished(allocator_);
    if (initial.size() <= parameter_.max_leaf_size) {
        if (root)
            initial.terminals.push_back(root);
        finished.emplace_back(std::move(initial));
        return finished;
    }

    Vector<WorkItem> work(allocator_);
    work.emplace_back(std::move(initial), level, PARTITION_SEED, root);
    for (uint64_t iteration = 0; iteration < MAX_PARTITION_ITERATIONS; ++iteration) {
        Vector<WorkItem> pending(allocator_);
        const bool all_items_are_small =
            std::all_of(work.begin(), work.end(), [](const auto& item) {
                return item.points.size() < MIN_PARALLEL_PARTITION_POINTS;
            });
        // Use one pool level at a time: outer work-item fan-out disables inner stripe fan-out.
        const bool parallelize_work_items = allow_parallelism and thread_pool_ != nullptr and
                                            thread_count_ > 1 and work.size() > 1 and
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
                split_work_item(item, pending, finished, allow_parallelism, subset != nullptr);
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
                               bool allow_inner_parallelism,
                               bool incremental_split) const {
    const uint64_t requested_fanout =
        item.level < parameter_.fanout.size() ? parameter_.fanout[item.level] : 1;
    const auto leaders = sample_leaders(item);
    const uint64_t fanout = std::min<uint64_t>(requested_fanout, leaders.size());
    const auto mode = !allow_inner_parallelism ? AssignmentMode::Serial
                      : incremental_split      ? AssignmentMode::SplitStripes
                                               : AssignmentMode::BuildChunks;
    auto clusters = assign_to_leaders(item.points, leaders, fanout, mode);

    bool every_cluster_is_parent = not clusters.empty();
    for (const auto& cluster : clusters) {
        every_cluster_is_parent = every_cluster_is_parent and cluster.size() == item.points.size();
    }
    if (every_cluster_is_parent and fanout > 1) {
        Leaf unchanged(item.points.begin(), item.points.end(), allocator_);
        pending.emplace_back(std::move(unchanged),
                             item.level + 1,
                             mix_seed(item.seed, item.points.size()),
                             item.route);
        return;
    }

    if (item.route) {
        item.route->leaders.assign(leaders.begin(), leaders.end());
        item.route->fanout = fanout;
        item.route->children.resize(leaders.size());
    }
    // Assignment visits the sorted, unique parent points in order, so every child keeps that
    // invariant without another sort and deduplication pass.
    for (uint64_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        auto& cluster = clusters[cluster_id];
        auto* child = item.route ? &item.route->children[cluster_id] : nullptr;
        if (child)
            cluster.terminals.push_back(child);
        if (cluster.empty()) {
            if (child)
                finished.emplace_back(std::move(cluster));
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
            if (child)
                child->children.resize((cluster.size() - 2) / (parameter_.max_leaf_size - 1) + 1);
            size_t part = 0;
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
                if (child) {
                    child->leaders.push_back(fallback.front());
                    fallback.terminals.push_back(&child->children.at(part++));
                }
                finished.emplace_back(std::move(fallback));
                begin = end;
            }
            continue;
        }

        pending.emplace_back(std::move(cluster),
                             item.level + 1,
                             mix_seed(item.seed, cluster_id + item.points.size()),
                             child);
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
                              std::span<const uint32_t> leaders,
                              const float* dots,
                              uint64_t fanout,
                              Vector<std::pair<float, uint32_t>>& candidates) const {
    auto less = [&](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return ids_[leaders[lhs.second]] < ids_[leaders[rhs.second]];
    };
    candidates.clear();
    if (fanout <= SMALL_TOP_K_LIMIT) {
        uint64_t farthest = 0;
        for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
            const auto candidate =
                std::make_pair(distance_from_dot(source, leaders[leader], dots[leader]),
                               static_cast<uint32_t>(leader));
            if (candidates.size() < fanout) {
                candidates.emplace_back(candidate);
                if (less(candidates[farthest], candidate)) {
                    farthest = candidates.size() - 1;
                }
            } else if (less(candidate, candidates[farthest])) {
                candidates[farthest] = candidate;
                farthest = 0;
                for (uint64_t selected = 1; selected < fanout; ++selected) {
                    if (less(candidates[farthest], candidates[selected])) {
                        farthest = selected;
                    }
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), less);
        return;
    }
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        candidates.emplace_back(distance_from_dot(source, leaders[leader], dots[leader]),
                                static_cast<uint32_t>(leader));
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<int64_t>(fanout),
                      candidates.end(),
                      less);
}

Leaves
PiPNNPipeline::assign_to_leaders(const Vector<uint32_t>& points,
                                 std::span<const uint32_t> leaders,
                                 uint64_t fanout,
                                 AssignmentMode mode,
                                 std::vector<float>* root_vectors) const {
    Leaves clusters(allocator_);
    clusters.reserve(leaders.size());
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        clusters.emplace_back(allocator_);
    }

    if (codes_) {
        Vector<InnerIdType> targets(allocator_);
        for (auto leader : leaders) targets.push_back(ids_[leader]);
        Vector<uint32_t> selected(checked_product(points.size(), fanout, "route assignments"),
                                  allocator_);
        constexpr uint64_t stripe = 128;
        auto assign = [&](uint64_t begin, uint64_t end) {
            Vector<float> distances(std::min(end - begin, stripe) * targets.size(), allocator_);
            Vector<InnerIdType> sources(allocator_);
            Vector<std::pair<float, uint32_t>> candidates(allocator_);
            for (uint64_t first = begin; first < end; first += stripe) {
                const auto last = std::min(first + stripe, end);
                sources.clear();
                for (auto i = first; i < last; ++i) sources.push_back(ids_[points[i]]);
                code_distances(*codes_, sources, targets, distances.data());
                for (auto i = first; i < last; ++i) {
                    select_leaders(points[i],
                                   leaders,
                                   distances.data() + (i - first) * targets.size(),
                                   fanout,
                                   candidates);
                    for (uint64_t j = 0; j < fanout; ++j)
                        selected[i * fanout + j] = candidates[j].second;
                }
            }
        };
        if (mode == AssignmentMode::Serial)
            assign(0, points.size());
        else
            parallel_for(
                points.size(),
                std::min<uint64_t>(stripe, (points.size() + thread_count_ - 1) / thread_count_),
                assign);
        for (uint64_t i = 0; i < points.size(); ++i)
            for (uint64_t j = 0; j < fanout; ++j)
                clusters[selected[i * fanout + j]].push_back(points[i]);
        return clusters;
    }

    std::optional<UninitializedFloatBuffer> local_leaders;
    const float* leader_matrix;
    if (root_vectors && !root_vectors->empty()) {
        leader_matrix = root_vectors->data();
    } else {
        const auto count = checked_product(leaders.size(), dimensions_, "leader matrix");
        float* values;
        if (root_vectors) {
            root_vectors->resize(count);
            values = root_vectors->data();
        } else {
            local_leaders.emplace(count, allocator_, "leader matrix");
            values = local_leaders->Data();
        }
        for (uint64_t i = 0; i < leaders.size(); ++i) gather_row(leaders[i], values, i);
        leader_matrix = values;
    }

    const uint64_t count = points.size();
    const bool root_chunks = mode == AssignmentMode::RouteRoot;
    const bool split_stripes = mode == AssignmentMode::SplitStripes;
    if (thread_pool_ != nullptr && thread_count_ > 1 &&
        ((root_chunks && count > 1) || (split_stripes && count > PARTITION_STRIPE_SIZE) ||
         (mode == AssignmentMode::BuildChunks && count >= MIN_PARALLEL_PARTITION_POINTS))) {
        const uint64_t assignment_count =
            checked_product(points.size(), fanout, "partition assignments");
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
        const uint64_t stripe_count = (count + PARTITION_STRIPE_SIZE - 1) / PARTITION_STRIPE_SIZE;
        const uint64_t stripes_per_chunk = split_stripes ? 1 : PARTITION_ASSIGNMENT_CHUNK_STRIPES;
        const uint64_t chunk_count =
            root_chunks ? std::min(thread_count_, count)
                        : (stripe_count + stripes_per_chunk - 1) / stripes_per_chunk;
        const uint64_t offset_count =
            checked_product(chunk_count, leaders.size(), "partition cluster offsets");
        Vector<uint64_t> cluster_offsets(offset_count, 0, allocator_);
        // Keep full BLAS stripes when enough exist. For a small root batch, split
        // rows directly so every nonempty task still owns a disjoint input range.
        auto chunk_start = [&](uint64_t chunk) -> uint64_t {
            if (!root_chunks)
                return std::min(count, chunk * stripes_per_chunk * PARTITION_STRIPE_SIZE);
            if (stripe_count < chunk_count)
                return chunk * count / chunk_count;
            return std::min(count, (chunk * stripe_count / chunk_count) * PARTITION_STRIPE_SIZE);
        };
        parallel_for(chunk_count, 1, [&](uint64_t chunk_begin, uint64_t chunk_end) {
            const uint64_t max_stripe_size =
                std::min<uint64_t>(PARTITION_STRIPE_SIZE, points.size());
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
                const auto end = chunk_start(chunk + 1);
                for (uint64_t begin = chunk_start(chunk); begin < end;
                     begin += PARTITION_STRIPE_SIZE) {
                    const uint64_t stripe_size = std::min(PARTITION_STRIPE_SIZE, end - begin);
                    for (uint64_t point = 0; point < stripe_size; ++point) {
                        gather_row(points[begin + point], point_values.Data(), point);
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
                                        leader_matrix,
                                        static_cast<int32_t>(dimensions_),
                                        0.0F,
                                        dots.Data(),
                                        static_cast<int32_t>(leaders.size()));

                    for (uint64_t point = 0; point < stripe_size; ++point) {
                        const uint64_t point_index = begin + point;
                        const uint32_t local_id = points[point_index];
                        select_leaders(local_id,
                                       leaders,
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
                const uint64_t begin = chunk_start(chunk);
                const uint64_t end = chunk_start(chunk + 1);
                for (uint64_t point = begin; point < end; ++point) {
                    for (uint64_t selected = 0; selected < fanout; ++selected) {
                        const uint32_t leader = assignment_data[point * fanout + selected];
                        clusters[leader][offsets[leader]++] = points[point];
                    }
                }
            }
        });
        return clusters;
    }

    const uint64_t max_stripe_size = std::min<uint64_t>(PARTITION_STRIPE_SIZE, points.size());
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
    for (uint64_t begin = 0; begin < points.size(); begin += PARTITION_STRIPE_SIZE) {
        const uint64_t stripe_size =
            std::min<uint64_t>(PARTITION_STRIPE_SIZE, points.size() - begin);
        for (uint64_t point = 0; point < stripe_size; ++point) {
            gather_row(points[begin + point], point_values.Data(), point);
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
                            leader_matrix,
                            static_cast<int32_t>(dimensions_),
                            0.0F,
                            dots.Data(),
                            static_cast<int32_t>(leaders.size()));

        for (uint64_t point = 0; point < stripe_size; ++point) {
            const uint32_t local_id = points[begin + point];
            select_leaders(
                local_id, leaders, dots.Data() + point * leaders.size(), fanout, candidates);
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
        result.terminals = lhs.terminals;
        result.terminals.insert(result.terminals.end(), rhs.terminals.begin(), rhs.terminals.end());
        return result;
    };

    Leaf accumulator(allocator_);
    for (auto& leaf : leaves) {
        if (leaf.empty() || leaf.size() >= parameter_.min_leaf_size) {
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

uint64_t
PiPNNPipeline::leaf_neighbor_count(uint64_t points) const {
    const auto requested =
        points < parameter_.min_leaf_size
            ? std::max(parameter_.leaf_neighbor_count, MIN_UNDERSIZED_LEAF_NEIGHBOR_COUNT)
            : parameter_.leaf_neighbor_count;
    return points < 2 ? 0 : std::min<uint64_t>(requested, points - 1);
}

// Keep the small-leaf BLAS strip shapes identical in serial and parallel rebuilds.
void
leaf_dot_strip(const float* matrix,
               uint64_t dims,
               uint64_t count,
               uint64_t begin,
               uint64_t end,
               float* output) {
    const auto k = static_cast<int32_t>(dims);
    const auto n = static_cast<int32_t>(end);
    const auto stride = static_cast<int32_t>(count);
    if (begin == 0) {
        BlasFunction::Ssyrk(BlasFunction::RowMajor,
                            BlasFunction::CblasLower,
                            BlasFunction::NoTrans,
                            n,
                            k,
                            1.0F,
                            matrix,
                            k,
                            0.0F,
                            output,
                            stride);
    } else {
        BlasFunction::Sgemm(BlasFunction::RowMajor,
                            BlasFunction::NoTrans,
                            BlasFunction::Trans,
                            static_cast<int32_t>(end - begin),
                            n,
                            k,
                            1.0F,
                            matrix + begin * dims,
                            k,
                            matrix,
                            k,
                            0.0F,
                            output,
                            stride);
    }
}

void
PiPNNPipeline::build_leaf(const Leaf& leaf, PiPNNLeafTopK* cache) {
    if (leaf.size() <= 1) {
        return;
    }

    const uint64_t point_count = leaf.size();
    const auto neighbor_count = leaf_neighbor_count(point_count);
    if (cache) {
        *cache = PiPNNLeafTopK(point_count, neighbor_count);
    }
    std::optional<UninitializedFloatBuffer> matrix_storage;
    if (!codes_) {
        matrix_storage.emplace(
            checked_product(point_count, dimensions_, "leaf matrix"), allocator_, "leaf matrix");
        auto& matrix = *matrix_storage;
        if (neighbor_count <= SMALL_TOP_K_LIMIT) {
            if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
                if (neighbor_count == 5) {
                    build_small_leaf<MetricType::METRIC_TYPE_L2SQR, 5>(
                        leaf, matrix, neighbor_count, cache);
                } else {
                    build_small_leaf<MetricType::METRIC_TYPE_L2SQR>(
                        leaf, matrix, neighbor_count, cache);
                }
                return;
            }
            if (metric_ == MetricType::METRIC_TYPE_IP) {
                if (neighbor_count == 5) {
                    build_small_leaf<MetricType::METRIC_TYPE_IP, 5>(
                        leaf, matrix, neighbor_count, cache);
                } else {
                    build_small_leaf<MetricType::METRIC_TYPE_IP>(
                        leaf, matrix, neighbor_count, cache);
                }
                return;
            }
        }
    }
    const uint64_t distance_values = checked_product(point_count, point_count, "leaf distances");
    UninitializedFloatBuffer distances(distance_values, allocator_, "leaf distances");
    if (codes_) {
        Vector<InnerIdType> targets(allocator_);
        for (auto id : leaf) targets.push_back(ids_[id]);
        code_distances(*codes_, targets, targets, distances.Data(), 0);
    } else {
        auto& matrix = *matrix_storage;
        for (uint64_t point = 0; point < point_count; ++point) {
            gather_row(leaf[point], matrix.Data(), point);
        }
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
    }
    select_leaf_neighbors(leaf, cache, distances.Data(), neighbor_count, 0, point_count);
}

void
PiPNNPipeline::select_leaf_neighbors(const Leaf& leaf,
                                     PiPNNLeafTopK* cache,
                                     const float* distances,
                                     uint64_t neighbor_count,
                                     uint64_t begin,
                                     uint64_t end) {
    const uint64_t point_count = leaf.size();
    if (neighbor_count <= SMALL_TOP_K_LIMIT) {
        std::array<std::pair<float, uint32_t>, SMALL_TOP_K_LIMIT> nearest{};
        for (uint64_t source = begin; source < end; ++source) {
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
                    leaf[source], leaf[target], distances[row * point_count + column]);
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
                emit(leaf, source, candidate, target, nearest[candidate].first, cache);
            }
        }
        return;
    }

    Vector<std::pair<float, uint32_t>> candidates(allocator_);
    candidates.reserve(point_count - 1);
    for (uint64_t source = begin; source < end; ++source) {
        candidates.clear();
        for (uint64_t target = 0; target < point_count; ++target) {
            if (source == target) {
                continue;
            }
            const uint64_t row = std::max(source, target);
            const uint64_t column = std::min(source, target);
            const float distance = distance_from_dot(
                leaf[source], leaf[target], distances[row * point_count + column]);
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
            emit(leaf, source, candidate, target, candidates[candidate].first, cache);
        }
    }
}

template <MetricType metric, uint64_t fixed_k>
void
PiPNNPipeline::build_small_leaf(const Leaf& leaf,
                                UninitializedFloatBuffer& matrix_buffer,
                                uint64_t requested_k,
                                PiPNNLeafTopK* cache) {
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
            norms[point] = norms_->Scalar(leaf[point]);
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
    constexpr uint64_t strip_rows = LEAF_DOT_STRIP;
    static_assert(strip_rows > SMALL_TOP_K_LIMIT);
    UninitializedFloatBuffer dot_buffer(
        checked_product(std::min(count, strip_rows), count, "leaf dot strip"),
        allocator_,
        "leaf dot strip");
    for (uint64_t row_begin = 0; row_begin < count; row_begin += strip_rows) {
        const auto row_end = std::min(count, row_begin + strip_rows);

        for (uint64_t point = row_begin; point < row_end; ++point) {
            if (native_half()) {
                gather_row(leaf[point], matrix_buffer.Data(), point);
            } else {
                const auto* source = vector_by_local_id(leaf[point]);
                std::copy(source, source + dimensions_, matrix + point * dimensions_);
            }
        }

        const auto* dots = dot_buffer.Data();
        leaf_dot_strip(matrix, dimensions_, count, row_begin, row_end, dot_buffer.Data());
        if (row_begin == 0) {
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
            emit(leaf, source, candidate, target, distance, cache);
        }
    }
}

pipnn_filter::RelativeHashes
PiPNNPipeline::relative_hashes(uint32_t source, uint32_t target) const {
    const auto planes = parameter_.hash_plane_count;
    return filters_.hashes(sketches_->Row(source), sketches_->Row(target), planes);
}

void
PiPNNPipeline::emit(const Leaf& leaf,
                    uint32_t source,
                    uint32_t rank,
                    uint32_t target,
                    float distance,
                    PiPNNLeafTopK* cache) {
    if (cache)
        cache->Row(source).Set(rank, {target, distance});
    else
        insert_pair(leaf[source], leaf[target], distance);
}

void
PiPNNPipeline::insert_pair(uint32_t source, uint32_t target, float distance) {
    if (source == target) {
        return;
    }
    const auto hashes = relative_hashes(source, target);
    const auto key = distance_order_key(distance);
    auto insert = [&](uint32_t from, uint32_t to, uint16_t hash) {
        const uint32_t lock_index = point_lock_count_ == 0 ? from : from % point_lock_count_;
        PointLockGuard lock(point_locks_.get(), lock_index);
        if (warm_build_) {
            seed_cached_row(from);
            if (reuse_graph_) {
                clean_graph_rows_[from] = 0;
            }
        }
        merge_candidate(
            reservoir_states_[from], reservoirs_.data() + from * reservoir_size_, {to, hash, key});
    };
    insert(source, target, hashes.forward);
    insert(target, source, hashes.reverse);
}

// Caller owns the source row or holds its lock.
void
PiPNNPipeline::merge_candidate(ReservoirState& state,
                               ReservoirEntry* row,
                               const ReservoirEntry& candidate) {
    const auto [target, hash, distance_key] = candidate;
    const auto incoming_key = std::make_tuple(distance_key, ids_[target], hash);

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
                    update_farthest(state, row);
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
                update_farthest(state, row);
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
    update_farthest(state, row);
}

void
PiPNNPipeline::update_farthest(ReservoirState& state, const ReservoirEntry* row) {
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
    if (codes_)
        return sanitize_distance(codes_->ComputePairVectors(ids_[lhs], ids_[rhs]));
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
    if (codes_)
        return sanitize_distance(dot);
    if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        const float sum = norms_->Scalar(lhs) + norms_->Scalar(rhs);
        return std::max(0.0F, sanitize_distance(sum - dot - dot));
    }
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        return sanitize_distance(1.0F - dot);
    }
    const float similarity =
        std::clamp(dot * norms_->Scalar(lhs) * norms_->Scalar(rhs), -1.0F, 1.0F);
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
        if (!codes_ && candidate_index + prefetch_distance < ordered.size()) {
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
PiPNNPipeline::certify_cached_graph(uint32_t source) const {
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
    const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
    const auto size = reservoir_states_[source].size;
    auto less = [&](uint32_t lhs, uint16_t lhs_key, uint32_t rhs, uint16_t rhs_key) {
        return std::make_tuple(lhs_key, id_gap(ids_[source], ids_[lhs]), ids_[lhs]) <
               std::make_tuple(rhs_key, id_gap(ids_[source], ids_[rhs]), ids_[rhs]);
    };
    for (uint64_t edge = begin; edge < end; ++edge) {
        const auto selected = old_to_new_[cache.graph_targets[edge]];
        if (selected == std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const auto distance = cache.graph_distances[edge];
        if (std::none_of(row, row + size, [&](const auto& entry) {
                return entry.neighbor == selected and entry.distance == distance;
            })) {
            return false;
        }
        if (edge > begin and not less(old_to_new_[cache.graph_targets[edge - 1]],
                                      cache.graph_distances[edge - 1],
                                      selected,
                                      distance)) {
            return false;
        }
    }
    const auto old_begin = cache.targets.begin() + static_cast<int64_t>(cache.offsets[old]);
    const auto old_end = cache.targets.begin() + static_cast<int64_t>(cache.offsets[old + 1]);
    for (uint64_t i = 0; i < size; ++i) {
        const auto candidate = row[i].neighbor;
        const auto distance = row[i].distance;
        if (end - begin == graph_->MaximumDegree() and
            less(old_to_new_[cache.graph_targets[end - 1]],
                 cache.graph_distances[end - 1],
                 candidate,
                 distance)) {
            continue;
        }
        bool selected_itself = false;
        for (uint64_t edge = begin; edge < end; ++edge) {
            if (old_to_new_[cache.graph_targets[edge]] == candidate) {
                selected_itself = true;
                break;
            }
        }
        if (selected_itself) {
            continue;
        }
        // Previously rejected candidates retain their proof if their relative order
        // against every selected neighbor survives inner-ID remapping.
        const auto old_target = new_to_old_[candidate];
        const auto found = std::lower_bound(old_begin, old_end, old_target);
        bool old_proof = found != old_end and *found == old_target;
        if (old_proof) {
            const auto position = static_cast<uint64_t>(found - cache.targets.begin());
            old_proof = cache.distances[position] == distance;
        }
        for (uint64_t edge = begin; old_proof and edge < end; ++edge) {
            const auto old_selected = cache.graph_targets[edge];
            const auto selected = old_to_new_[old_selected];
            const auto selected_distance = cache.graph_distances[edge];
            const bool earlier =
                std::make_tuple(selected_distance,
                                id_gap(cache.inner_ids[old], cache.inner_ids[old_selected]),
                                cache.inner_ids[old_selected]) <
                std::make_tuple(distance,
                                id_gap(cache.inner_ids[old], cache.inner_ids[old_target]),
                                cache.inner_ids[old_target]);
            old_proof = earlier == less(selected, selected_distance, candidate, distance);
        }
        if (old_proof) {
            continue;
        }
        bool blocked = false;
        for (uint64_t edge = begin; edge < end; ++edge) {
            const auto selected = old_to_new_[cache.graph_targets[edge]];
            if (less(selected, cache.graph_distances[edge], candidate, distance) and
                parameter_.alpha * pair_distance(selected, candidate) <
                    distance_from_order_key(distance)) {
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

// A streaming Add usually splits one parent into a handful of uneven children.
// Schedule disjoint strips from all children together; each phase drains the pool
// before readers consume the triangle or the new leaf caches become visible.
bool
PiPNNPipeline::cache_split_rows(const Leaves& leaves, std::vector<PiPNNCachedLeaf>& caches) {
    if (!thread_pool_ || leaves.size() >= thread_count_ || thread_count_ <= 1)
        return false;
    uint64_t values = 0, largest = 0;
    for (const auto& leaf : leaves) {
        const uint64_t n = leaf.size();
        // Bound additional FP32 scratch to 4 MiB. For large batches keep the
        // established leaf-level schedule and the streaming FP32 kernels.
        if (n > 1024 || (!codes_ && (metric_ == MetricType::METRIC_TYPE_COSINE ||
                                     leaf_neighbor_count(n) > SMALL_TOP_K_LIMIT)))
            return false;
        values += n * n;
        if (values > 1024 * 1024)
            return false;
        largest = std::max(largest, n);
    }
    if (largest < LEAF_DOT_STRIP)
        return false;

    struct Child {
        std::vector<InnerIdType> targets;
        std::optional<UninitializedFloatBuffer> vectors, distances;
    };
    struct Strip {
        uint64_t child, begin, end;
    };
    std::vector<Child> children(leaves.size());
    std::vector<Strip> strips;
    const uint64_t width = codes_ ? 8 : LEAF_DOT_STRIP;
    for (uint64_t i = 0; i < leaves.size(); ++i) {
        const auto& leaf = leaves[i];
        const uint64_t n = leaf.size();
        if (n < 2)
            continue;
        auto& child = children[i];
        caches[i].nearest = PiPNNLeafTopK(n, leaf_neighbor_count(n));
        child.distances.emplace(n * n, allocator_, "split distances");
        if (codes_) {
            child.targets.reserve(n);
            for (auto point : leaf) child.targets.push_back(ids_[point]);
        } else {
            child.vectors.emplace(
                checked_product(n, dimensions_, "split vectors"), allocator_, "split vectors");
        }
        // High rows touch more targets. Give them to the first available workers.
        for (uint64_t end = n; end != 0;) {
            const uint64_t begin = (end - 1) / width * width;
            strips.push_back({i, begin, end});
            end = begin;
        }
    }
    auto visit_strips = [&](auto task) {
        parallel_for(strips.size(), 1, [&](uint64_t begin, uint64_t end) {
            for (auto i = begin; i < end; ++i) task(strips[i]);
        });
    };
    if (!codes_) {
        visit_strips([&](const Strip& s) {
            auto* data = children[s.child].vectors->Data();
            for (auto i = s.begin; i < s.end; ++i) gather_row(leaves[s.child][i], data, i);
        });
    }
    visit_strips([&](const Strip& s) {
        auto& child = children[s.child];
        const uint64_t n = leaves[s.child].size();
        auto* out = child.distances->Data() + s.begin * n;
        if (codes_) {
            const auto ids = std::span<const InnerIdType>(child.targets);
            code_distances(*codes_, ids.subspan(s.begin, s.end - s.begin), ids, out, s.begin);
        } else {
            leaf_dot_strip(child.vectors->Data(), dimensions_, n, s.begin, s.end, out);
        }
    });
    visit_strips([&](const Strip& s) {
        auto& nearest = caches[s.child].nearest;
        select_leaf_neighbors(leaves[s.child],
                              &nearest,
                              children[s.child].distances->Data(),
                              nearest.Width(),
                              s.begin,
                              s.end);
    });
    return true;
}

void
PiPNNPipeline::cache_partition(Leaves& leaves, PiPNNIncrementalState& state) {
    const auto offset = state.leaves.size();
    std::vector<PiPNNCachedLeaf> caches(leaves.size());
    const bool rebuilt = offset != 0 && cache_split_rows(leaves, caches);
    parallel_for(leaves.size(), 1, [&](uint64_t begin, uint64_t end) {
        for (size_t i = begin; i < end; ++i) {
            auto& leaf = leaves[i];
            for (auto* terminal : leaf.terminals) terminal->leaf = offset + i;
            auto& cached = caches[i];
            cached.points.assign(leaf.begin(), leaf.end());
            if (!rebuilt)
                build_leaf(leaf, &cached.nearest);
        }
    });
    if (state.leaves.empty()) {
        state.leaves.swap(caches);
    } else {
        state.leaves.insert(state.leaves.end(),
                            std::make_move_iterator(caches.begin()),
                            std::make_move_iterator(caches.end()));
    }
}

void
PiPNNPipeline::update_existing_edges(uint32_t source,
                                     uint64_t first_new,
                                     const PiPNNIncrementalState& state,
                                     ReservoirState& candidates,
                                     ReservoirEntry* row,
                                     EdgeUpdateScratch& scratch) {
    auto& [incoming, incumbents, neighbors] = scratch;
    neighbors.clear();
    graph_->GetNeighbors(ids_[source], neighbors);
    incumbents.clear();
    // Sanitized distances encode at most +infinity (0xff80).
    constexpr uint16_t unknown_distance = UINT16_MAX;
    auto edge_distance = [&](ReservoirEntry& edge) {
        if (edge.distance == unknown_distance) {
            edge.distance = distance_order_key(pair_distance(source, edge.neighbor));
        }
        return edge.distance;
    };
    // The row initially contains just the batch's candidates. When capacity
    // cannot bind, an incumbent can exclude a newcomer only in the same bucket.
    const bool capacity_can_bind = candidates.size + neighbors.size() > reservoir_size_;
    for (const auto id : neighbors) {
        const auto local = state.LocalId(id);
        auto& edge = incumbents.emplace_back(
            local, relative_hashes(source, local).forward, unknown_distance);
        auto* end = row + candidates.size;
        if (capacity_can_bind ||
            ((edge.hash & COINCIDENT_HASH_FLAG) == 0 &&
             std::any_of(row, end, [&](const auto& e) { return e.hash == edge.hash; }))) {
            edge_distance(edge);
            merge_candidate(candidates, row, edge);
        }
    }
    incoming.clear();
    std::copy_if(
        row, row + candidates.size, std::back_inserter(incoming), [first_new](const auto& entry) {
            return entry.neighbor >= first_new;
        });
    // Reservoir insertion is parallel. Make batch competition independent of arrival order.
    std::sort(incoming.begin(), incoming.end(), [&](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.distance, ids_[lhs.neighbor]) <
               std::tie(rhs.distance, ids_[rhs.neighbor]);
    });
    bool changed = false;
    for (const auto& candidate : incoming) {
        if (neighbors.size() < graph_->MaximumDegree()) {
            incumbents.push_back(candidate);
            neighbors.push_back(ids_[candidate.neighbor]);
            changed = true;
            continue;
        }
        // Break nearest-hash ties by the farthest source distance, then largest ID.
        auto hamming = [&](const ReservoirEntry& edge) {
            return std::popcount(
                static_cast<unsigned>((edge.hash ^ candidate.hash) & ~COINCIDENT_HASH_FLAG));
        };
        auto competitor = incumbents.begin();
        auto closest = hamming(*competitor);
        for (auto it = competitor + 1; it != incumbents.end(); ++it) {
            const auto bits = hamming(*it);
            if (bits < closest ||
                (bits == closest &&
                 std::make_pair(edge_distance(*it), ids_[it->neighbor]) >
                     std::make_pair(edge_distance(*competitor), ids_[competitor->neighbor]))) {
                competitor = it;
                closest = bits;
            }
        }
        if (candidate.distance < edge_distance(*competitor)) {
            neighbors[competitor - incumbents.begin()] = ids_[candidate.neighbor];
            *competitor = candidate;
            changed = true;
        }
    }
    if (changed) {
        graph_->InsertNeighborsById(ids_[source], neighbors);
    }
}

void
PiPNNPipeline::replay_add(const PiPNNIncrementalState& state,
                          uint64_t first_new,
                          std::span<const size_t> hit_leaves,
                          std::span<const ReplayRows> rows) {
    const uint64_t batch = ids_.size() - first_new;
    // Keep small Adds cheap and give the pool enough independent shards for larger batches.
    const uint64_t shards =
        thread_pool_ && thread_count_ > 1 ? std::min<uint64_t>(ids_.size(), batch < 8 ? 8 : 64) : 1;
    const uint64_t chunks = std::clamp<uint64_t>((hit_leaves.size() + 15) / 16, 1, 128);
    std::vector<Vector<AddCandidate>> buffers;
    buffers.reserve(chunks * shards);
    for (uint64_t i = 0; i < chunks * shards; ++i) {
        buffers.emplace_back(allocator_);
    }
    // Each leaf task owns its buffers. Both directions share the leaf's distance key.
    parallel_for(chunks, 1, [&](uint64_t begin, uint64_t end) {
        for (uint64_t chunk = begin; chunk < end; ++chunk) {
            auto* output = buffers.data() + chunk * shards;
            auto submit = [&](uint32_t u, const PiPNNLocalNeighbor& neighbor) {
                const auto v = neighbor.target;
                if (u == v || (u < first_new && v < first_new)) {
                    return;
                }
                const auto hashes = relative_hashes(u, v);
                const auto key = distance_order_key(neighbor.distance);
                output[u % shards].push_back({u, {v, hashes.forward, key}});
                output[v % shards].push_back({v, {u, hashes.reverse, key}});
            };
            for (uint64_t pos = hit_leaves.size() * chunk / chunks;
                 pos < hit_leaves.size() * (chunk + 1) / chunks;
                 ++pos) {
                visit_leaf_candidates(state.leaves[hit_leaves[pos]],
                                      pos < rows.size() ? &rows[pos] : nullptr,
                                      submit);
            }
        }
    });
    // All events for a source now have a single owner. Consume one bounded reservoir at a time.
    parallel_for(shards, 1, [&](uint64_t begin, uint64_t end) {
        Vector<ReservoirEntry> row(reservoir_size_, allocator_);
        PruneScratch prune(allocator_);
        EdgeUpdateScratch update(allocator_);
        for (uint64_t shard = begin; shard < end; ++shard) {
            const auto grouped =
                group_add_candidates(buffers, shard, shards, ids_.size(), allocator_);
            auto it = grouped.begin();
            auto collect = [&](uint32_t source) {
                ReservoirState candidates;
                while (it != grouped.end() && it->source == source) {
                    merge_candidate(candidates, row.data(), it->entry);
                    ++it;
                }
                return candidates;
            };
            while (it != grouped.end() && it->source < first_new) {
                const auto source = it->source;
                auto candidates = collect(source);
                update_existing_edges(source, first_new, state, candidates, row.data(), update);
            }
            // Every new ID owns a graph row, including singleton leaves with no candidates.
            const auto first = first_new + (shard + shards - first_new % shards) % shards;
            for (uint64_t source = first; source < ids_.size(); source += shards) {
                const auto candidates = collect(source);
                robust_prune(source, row.data(), candidates.size, prune);
                graph_->InsertNeighborsById(ids_[source], prune.neighbors);
            }
        }
    });
}

void
PiPNNPipeline::BuildCached(PiPNNIncrementalState& state) {
    ids_ = state.ids;
    prepare_build_reservoirs();
    prepare_sketches(&state);
    auto leaves = partition(&state.root);
    cache_partition(leaves, state);
    state.UpdateLeafDepths(state.root, 0);
    parallel_for(state.leaves.size(), 4, [&](uint64_t begin, uint64_t end) {
        for (uint64_t i = begin; i < end; ++i) {
            visit_leaf_candidates(state.leaves[i], nullptr, [&](uint32_t u, const auto& neighbor) {
                insert_pair(u, neighbor.target, neighbor.distance);
            });
        }
    });
    write_graph();
}

void
PiPNNPipeline::route_batch(const PiPNNRoute& route,
                           Leaf points,
                           LeafAdditions& additions,
                           PiPNNIncrementalState& state) const {
    struct NodeBatch {
        const PiPNNRoute* node;
        Leaf points;
    };
    std::vector<NodeBatch> pending;
    // Merged terminals can share a leaf or replacement tree. Only the caller
    // publishes to additions, in route order, after a wave of workers has joined.
    auto dispatch = [&](const PiPNNRoute& start, Leaf& group) {
        if (group.empty())
            return;
        auto* node = &start;
        while (node->children.empty()) {
            const auto& replacement = state.leaves.at(node->leaf).replacement;
            if (replacement.children.empty() && replacement.leaf == UINT32_MAX) {
                auto& target = additions.try_emplace(node->leaf, allocator_).first->second;
                target.insert(target.end(), group.begin(), group.end());
                return;
            }
            node = &replacement;
        }
        pending.push_back({node, std::move(group)});
    };
    dispatch(route, points);
    bool root = true;
    while (!pending.empty()) {
        std::vector<NodeBatch> layer;
        layer.swap(pending);
        std::vector<Leaves> children(layer.size(), Leaves(allocator_));
        auto assign = [&](uint64_t i, AssignmentMode mode) {
            const auto& [node, group] = layer[i];
            // The first branching node stays the effective root when terminal leaves split.
            auto* cache = mode == AssignmentMode::RouteRoot ? &state.root_leader_vectors : nullptr;
            children[i] = assign_to_leaders(group, node->leaders, node->fanout, mode, cache);
        };
        if (root) {
            assign(0, AssignmentMode::RouteRoot);
            root = false;
        } else {
            // Each worker routes a whole node without submitting nested pool jobs.
            parallel_for(layer.size(), 1, [&](uint64_t begin, uint64_t end) {
                for (uint64_t i = begin; i < end; ++i) assign(i, AssignmentMode::Serial);
            });
        }
        for (uint64_t i = 0; i < layer.size(); ++i) {
            for (uint64_t j = 0; j < children[i].size(); ++j) {
                dispatch(layer[i].node->children[j], children[i][j]);
            }
        }
    }
}

void
PiPNNPipeline::extend_leaf(PiPNNCachedLeaf& leaf, const Leaf& batch, ReplayRows& changes) const {
    const auto m = leaf.points.size(), b = batch.size(), n = m + b;
    const auto old_k = leaf.nearest.Width();
    auto requested = n < parameter_.min_leaf_size ? std::max(parameter_.leaf_neighbor_count,
                                                             MIN_UNDERSIZED_LEAF_NEIGHBOR_COUNT)
                                                  : parameter_.leaf_neighbor_count;
    const auto k = std::min<uint64_t>(requested, n - 1);
    // A single append can touch thousands of leaves. Bound spare member IDs in each
    // leaf while retaining geometric growth for repeated inserts.
    if (n > leaf.points.capacity()) {
        const auto next =
            std::min<uint64_t>(parameter_.max_leaf_size, m + std::max<size_t>(8, m / 8));
        leaf.points.reserve(std::max<uint64_t>(n, next));
    }
    leaf.points.insert(leaf.points.end(), batch.begin(), batch.end());
    if (!k)
        return;
    // Read coordinates in place. One inner product supplies both endpoints' top-k updates.
    UninitializedFloatBuffer dots(
        checked_product(b, n, "incremental products"), allocator_, "incremental products");
    if (codes_) {
        Vector<InnerIdType> targets(allocator_);
        for (auto id : leaf.points) targets.push_back(ids_[id]);
        code_distances(
            *codes_, std::span<const InnerIdType>(targets).subspan(m), targets, dots.Data(), m);
    } else {
        const auto processed = ComputePiPNNOldRowTiles(
            rows_.data(), leaf.points.data(), m, b, dimensions_, dots.Data());
        for (size_t i = processed; i + 1 < n; ++i) {
            const auto* point = vector_by_local_id(leaf.points[i]);
            size_t j = std::max(m, i + 1);
            for (; j + 4 <= n; j += 4) {
                auto* out = dots.Data() + (j - m) * n + i;
                // The batch kernel adds its products to the four output slots.
                out[0] = out[n] = out[2 * n] = out[3 * n] = 0.0F;
                FP32ComputeIPBatch4(point,
                                    dimensions_,
                                    vector_by_local_id(leaf.points[j]),
                                    vector_by_local_id(leaf.points[j + 1]),
                                    vector_by_local_id(leaf.points[j + 2]),
                                    vector_by_local_id(leaf.points[j + 3]),
                                    out[0],
                                    out[n],
                                    out[2 * n],
                                    out[3 * n]);
            }
            for (; j < n; ++j) {
                dots.Data()[(j - m) * n + i] =
                    FP32ComputeIP(point, vector_by_local_id(leaf.points[j]), dimensions_);
            }
        }
    }
    // A width change also changes the row stride. Keep the old layout for that merge.
    PiPNNLeafTopK previous;
    if (old_k == k) {
        leaf.nearest.AppendRows(b);
    } else {
        PiPNNLeafTopK resized(n, k);
        previous = std::exchange(leaf.nearest, std::move(resized));
    }
    std::vector<PiPNNLocalNeighbor> row;
    row.reserve(std::max(n, b + old_k));
    const auto select_rows = filters_.kth;
    if (old_k == k) {
        changes.old_count = m;
        changes.changed.resize((m + 63) / 64);
    }
    const bool batch_filter = select_rows != nullptr && old_k == k && m >= 16;
    std::vector<float> thresholds(batch_filter ? m : 0);
    std::vector<uint32_t> selected(batch_filter ? m : 0);
    for (uint64_t i = 0; i < thresholds.size(); ++i) {
        thresholds[i] = leaf.nearest.Row(i).Distance(k - 1);
    }
    auto order_for = [&](uint32_t u) {
        return [&, u](const PiPNNLocalNeighbor& lhs, const PiPNNLocalNeighbor& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            const auto left = ids_[leaf.points[lhs.target]];
            const auto right = ids_[leaf.points[rhs.target]];
            return std::make_pair(id_gap(ids_[u], left), left) <
                   std::make_pair(id_gap(ids_[u], right), right);
        };
    };
    auto offer = [&](uint64_t i, PiPNNLocalNeighbor candidate) {
        const auto ranks = leaf.nearest.Row(i);
        const auto less = order_for(leaf.points[i]);
        if (less(candidate, ranks.Get(k - 1))) {
            uint32_t first = 0, last = k - 1;
            while (first < last) {
                const auto mid = first + (last - first) / 2;
                if (less(ranks.Get(mid), candidate))
                    first = mid + 1;
                else
                    last = mid;
            }
            ranks.ReplaceLast(first, k - 1, candidate);
            changes.changed[i / 64] |= uint64_t{1} << (i % 64);
        }
        return ranks.Distance(k - 1);
    };
    for (uint64_t i = batch_filter ? m : 0; i < n; ++i) {
        const auto u = leaf.points[i];
        if (i < m && old_k == k) {
            for (uint64_t j = m; j < n; ++j) {
                const auto v = leaf.points[j];
                const auto dot = dots.Data()[(j - m) * n + i];
                offer(i, {static_cast<uint32_t>(j), distance_from_dot(u, v, dot)});
            }
            continue;
        }

        row.clear();
        if (i < m && old_k != 0) {
            const auto old_row = previous.Row(i);
            for (uint32_t rank = 0; rank < old_k; ++rank) row.push_back(old_row.Get(rank));
        }
        for (uint64_t j = (i < m ? m : 0); j < n; ++j) {
            if (i == j)
                continue;
            const auto v = leaf.points[j];
            auto& dot = dots.Data()[(std::max(i, j) - m) * n + std::min(i, j)];
            const float distance = distance_from_dot(u, v, dot);
            row.push_back({static_cast<uint32_t>(j), distance});
            if (batch_filter && j < m) {
                // Neither endpoint needs this dot again. Cosine rounds in source-first order.
                dot = metric_ == MetricType::METRIC_TYPE_COSINE ? distance_from_dot(v, u, dot)
                                                                : distance;
            }
        }
        if (batch_filter) {
            const auto* distances = dots.Data() + (i - m) * n;
            const auto count = select_rows(distances, thresholds.data(), m, selected.data());
            for (uint64_t t = 0; t < count; ++t) {
                const auto old = selected[t];
                thresholds[old] = offer(old, {static_cast<uint32_t>(i), distances[old]});
            }
        }
        std::partial_sort(row.begin(), row.begin() + k, row.end(), order_for(u));
        const auto ranks = leaf.nearest.Row(i);
        for (uint32_t rank = 0; rank < k; ++rank) ranks.Set(rank, row[rank]);
    }
}

void
PiPNNPipeline::rebuild_split_leaves(std::vector<LeafSplit>& splits,
                                    const LeafAdditions& additions,
                                    PiPNNIncrementalState& state) {
    if (splits.empty()) {
        return;
    }
    // A lone parent can use the pool internally. With multiple parents the outer loop owns it.
    parallel_for(splits.size(), 1, [&](uint64_t begin, uint64_t end) {
        for (uint64_t i = begin; i < end; ++i) {
            auto& split = splits[i];
            const auto& points = state.leaves[split.parent].points;
            const auto& batch = additions.at(split.parent);
            Leaf all(points.begin(), points.end(), allocator_);
            all.insert(all.end(), batch.begin(), batch.end());
            std::sort(all.begin(), all.end());
            split.children = partition(
                &split.replacement, &all, state.leaves[split.parent].depth, splits.size() == 1);
        }
    });

    // Flatten in parent order so child IDs do not depend on worker completion order.
    Leaves children(allocator_);
    for (auto& split : splits) {
        children.insert(children.end(),
                        std::make_move_iterator(split.children.begin()),
                        std::make_move_iterator(split.children.end()));
        split.end_child = state.leaves.size() + children.size();
    }
    cache_partition(children, state);
}

void
PiPNNPipeline::Increment(PiPNNIncrementalState& state, size_t old_count) {
    ids_ = state.ids;
    prepare_sketches(
        &state, old_count, codes_ && !shared_sketches_ ? rows_ : std::span<const float* const>{});
    LeafAdditions additions;
    Leaf incoming(allocator_);
    for (size_t i = old_count; i < ids_.size(); ++i) incoming.push_back(i);
    route_batch(state.root, std::move(incoming), additions, state);
    std::vector<size_t> hit_leaves;
    std::vector<LeafSplit> splits;
    for (auto& [i, batch] : additions) {
        std::sort(batch.begin(), batch.end());
        batch.erase(std::unique(batch.begin(), batch.end()), batch.end());
        if (state.leaves[i].points.size() + batch.size() <= parameter_.max_leaf_size) {
            hit_leaves.push_back(i);
        } else {
            splits.push_back({i, {}, Leaves(allocator_)});
        }
    }
    uint64_t child = state.leaves.size();
    rebuild_split_leaves(splits, additions, state);

    const auto ordinary_count = hit_leaves.size();
    for (auto& split : splits) {
        for (; child < split.end_child; ++child) {
            const auto& points = state.leaves[child].points;
            if (std::any_of(points.begin(), points.end(), [old_count](uint32_t id) {
                    return id >= old_count;
                })) {
                hit_leaves.push_back(child);
            }
        }
        auto& parent = state.leaves[split.parent];
        const auto depth = parent.depth;
        parent = {};
        parent.depth = depth;
        parent.replacement = std::move(split.replacement);
        state.UpdateLeafDepths(parent.replacement, depth);
    }
    // Ordinary hits come first. A hit split child already has a freshly rebuilt top-k.
    std::vector<ReplayRows> changed_rows(ordinary_count);
    parallel_for(ordinary_count, 1, [&](uint64_t begin, uint64_t end) {
        for (size_t j = begin; j < end; ++j) {
            const auto leaf = hit_leaves[j];
            extend_leaf(state.leaves[leaf], additions.at(leaf), changed_rows[j]);
        }
    });
    replay_add(state, old_count, hit_leaves, changed_rows);
}

void
PiPNNPipeline::write_graph_row(uint32_t source, PruneScratch& scratch) {
    // Adaptive thresholds depend on rejected-candidate counts as well as selected edges.
    // Recompute them even when all previously selected neighbors survive.
    bool reuse =
        reuse_graph_ and not parameter_.adaptive_pruning and clean_graph_rows_[source] != 0;
    if (not reuse) {
        seed_cached_row(source);
        reuse = reuse_graph_ and certify_cached_graph(source);
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
        const auto& state = reservoir_states_[source];
        const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
        robust_prune(source, row, state.size, scratch);
    }
    graph_->InsertNeighborsById(ids_[source], scratch.neighbors);
}

void
PiPNNPipeline::write_graph(std::span<const uint32_t> representatives) {
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
    add_use_codes = json.Contains("pipnn_add_use_codes") && json["pipnn_add_use_codes"].GetBool();
    incremental_add =
        json.Contains("pipnn_incremental_add") && json["pipnn_incremental_add"].GetBool();
    require_argument(!add_use_codes || incremental_add,
                     "pipnn_add_use_codes requires pipnn_incremental_add");
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
    if (add_use_codes)
        json["pipnn_add_use_codes"].SetBool(true);
    if (incremental_add)
        json["pipnn_incremental_add"].SetBool(true);
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

bool
PiPNNGraphBuilderParameter::CanLoadIncremental(const PiPNNGraphBuilderParameter& stored) const {
    if (!incremental_add || !stored.incremental_add)
        return true;
    // The encoded-distance reader can consume an exact snapshot's stored projections.
    // An exact reader cannot reconstruct the discarded historical vectors from codes.
    auto required = *this;
    if (add_use_codes)
        required.add_use_codes = stored.add_use_codes;
    return required.ToJson().Dump() == stored.ToJson().Dump();
}

void
PiPNNGraphBuilderParameter::ValidateAddDistanceCodes(FlattenInterface& codes) const {
    if (add_use_codes)
        require_argument(
            codes.SupportSplitCodeStorage() ||
                GetBaseQuantizerName(codes.GetQuantizerName()) != QUANTIZATION_TYPE_VALUE_RABITQ,
            "PiPNN code distances require RaBitQ split or a quantizer with symmetric distances");
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
PiPNNGraphBuilder::Update(const GraphInterfacePtr& graph,
                          const Vector<InnerIdType>& ids,
                          const Vector<const float*>& input,
                          PiPNNIncrementalState& state) const {
    update(graph, ids, input, state, {});
}

void
PiPNNGraphBuilder::UpdateFromSharedRows(const GraphInterfacePtr& graph,
                                        const Vector<InnerIdType>& ids,
                                        std::span<const float* const> rows_by_id,
                                        PiPNNIncrementalState& state) const {
    require_argument(!graph->GetDuplicateTracker(), "shared PiPNN rows require unique graph IDs");
    Vector<const float*> input(allocator_);
    input.reserve(ids.size());
    for (auto id : ids) {
        require_argument(id < rows_by_id.size(), "missing global PiPNN row");
        input.push_back(rows_by_id[id]);
    }
    update(graph, ids, input, state, rows_by_id);
}

void
PiPNNGraphBuilder::AppendProjections(PiPNNProjectionState& sketches,
                                     std::span<const float* const> rows) const {
    if (rows.empty())
        return;
    PiPNNPipeline pipeline(
        parameter_, dimensions_, metric_, allocator_, thread_pool_, thread_count_, nullptr, rows);
    pipeline.AppendProjections(sketches);
}

void
PiPNNGraphBuilder::UpdateWithCodes(const GraphInterfacePtr& graph,
                                   const Vector<InnerIdType>& ids,
                                   const Vector<const float*>& input,
                                   PiPNNIncrementalState& state,
                                   FlattenInterface& codes,
                                   const PiPNNProjectionState* shared) const {
    update(graph, ids, input, state, {}, &codes, shared);
}

void
PiPNNGraphBuilder::update(const GraphInterfacePtr& graph,
                          const Vector<InnerIdType>& ids,
                          const Vector<const float*>& input,
                          PiPNNIncrementalState& state,
                          std::span<const float* const> rows_by_id,
                          FlattenInterface* codes,
                          const PiPNNProjectionState* shared) const {
    require_argument(shared ? input.empty() : ids.size() == input.size(),
                     "incremental row count mismatch");
    if (ids.empty())
        return;
    parameter_.Validate(graph->MaximumDegree());
    const auto old = state.ids.size();
    require_argument(old + ids.size() <= UINT32_MAX, "too many PiPNN points");
    const auto capacity = graph->MaxCapacity();
    require_argument(state.RequiredCapacity() <= capacity,
                     "PiPNN historical ID exceeds graph capacity");
    const bool increasing =
        std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<>()) == ids.end();
    UnorderedSet<InnerIdType> seen(allocator_);
    if (!increasing)
        seen.reserve(ids.size());
    for (uint64_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i];
        require_argument(
            (shared ? id < shared->projections.Size() : input[i] != nullptr) && id < capacity,
            "invalid incremental PiPNN row or ID");
        require_argument(!state.ContainsId(id) && graph->GetGroupId(id) == id &&
                             (increasing || seen.emplace(id).second),
                         "duplicate incremental PiPNN ID");
    }
    if (old) {
        require_argument(
            state.planes.size() == dimensions_ * parameter_.hash_plane_count &&
                state.projections.Width() == parameter_.hash_plane_count &&
                state.projections.Size() == old &&
                state.norms.Size() == (metric_ == MetricType::METRIC_TYPE_IP ? 0 : old),
            "incompatible PiPNN sketch state");
    }
    Vector<InnerIdType> unique_ids(allocator_);
    Vector<const float*> unique_rows(allocator_);
    std::vector<std::pair<InnerIdType, InnerIdType>> aliases;
    if (graph->GetDuplicateTracker()) {
        require_argument(!parameter_.add_use_codes, "coded PiPNN requires unique graph IDs");
        std::unordered_map<std::string_view, InnerIdType> owners;
        const auto key = [&](const float* v) {
            return std::string_view(reinterpret_cast<const char*>(v), dimensions_ * sizeof(float));
        };
        owners.reserve(old + ids.size());
        for (uint64_t i = 0; i < old; ++i) owners.emplace(key(state.GetVector(i)), state.ids[i]);
        for (uint64_t i = 0; i < ids.size(); ++i) {
            auto [it, added] = owners.emplace(key(input[i]), ids[i]);
            if (added) {
                unique_ids.push_back(ids[i]);
                unique_rows.push_back(input[i]);
            } else
                aliases.emplace_back(it->second, ids[i]);
        }
    }
    const auto& accepted = graph->GetDuplicateTracker() ? unique_ids : ids;
    const auto& accepted_rows = graph->GetDuplicateTracker() ? unique_rows : input;
    if (!accepted.empty()) {
        Vector<const float*> indirect(allocator_);
        std::span<const float* const> local_rows;
        if (parameter_.add_use_codes && rows_by_id.empty()) {
            require_argument(!old || codes, "missing PiPNN distance codes");
            state.AppendIds(accepted);
            local_rows = accepted_rows;
        } else if (rows_by_id.empty()) {
            state.ReserveVectors(old + accepted.size(), dimensions_, allocator_);
            state.AppendIds(accepted);
            for (uint64_t i = 0; i < accepted.size(); ++i)
                state.SetVector(old + i, accepted_rows[i]);
            local_rows = state.Rows();
        } else {
            require_argument(state.RequiredCapacity() <= rows_by_id.size(),
                             "missing historical PiPNN row");
            state.AppendIds(accepted);
            if (state.HasDenseIds()) {
                local_rows = rows_by_id.first(state.ids.size());
            } else {
                indirect.reserve(state.ids.size());
                for (auto id : state.ids) indirect.push_back(rows_by_id[id]);
                local_rows = indirect;
            }
        }
        PiPNNPipeline pipeline(parameter_,
                               dimensions_,
                               metric_,
                               allocator_,
                               thread_pool_,
                               thread_count_,
                               graph,
                               local_rows);
        pipeline.SetCodeDistances(codes, shared);
        if (old)
            pipeline.Increment(state, old);
        else
            pipeline.BuildCached(state);
    }
    for (auto [owner, alias] : aliases) graph->SetDuplicateId(owner, alias);
}

void
PiPNNGraphBuilder::Build(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids_sequence,
                         const Vector<const float*>& rows,
                         const std::vector<std::string>* source_ids,
                         const PiPNNBuildCache* imported_cache,
                         std::shared_ptr<PiPNNBuildCache>* exported_cache) const {
    // Keep the imported snapshot alive in the caller when input and output share storage.
    build_typed(graph,
                ids_sequence,
                rows,
                DataTypes::DATA_TYPE_FLOAT,
                source_ids,
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
    build_typed(graph, ids_sequence, rows, data_type, nullptr, nullptr, nullptr);
}

template <typename ElementType>
void
PiPNNGraphBuilder::build_typed(const GraphInterfacePtr& graph,
                               const Vector<InnerIdType>& ids_sequence,
                               const Vector<const ElementType*>& rows,
                               DataTypes data_type,
                               const std::vector<std::string>* source_ids,
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
                pipeline.ConfigureCache(source_ids, imported_cache, exported_cache);
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
