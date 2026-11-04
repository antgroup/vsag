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
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "common.h"
#include "datacell/graph_interface.h"
#include "hash_types.h"
#include "impl/blas/blas_function.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "pipnn_adaptive_pruning.h"
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
id_gap(InnerIdType first, InnerIdType second) {
    const uint64_t left = first;
    const uint64_t right = second;
    return left >= right ? left - right : right - left;
}

void
group_duplicate_rows(const Vector<InnerIdType>& ids,
                     const Vector<const float*>& rows,
                     uint64_t dimensions,
                     Allocator* allocator,
                     Vector<InnerIdType>& representative_ids,
                     Vector<const float*>& representative_rows,
                     Vector<std::pair<InnerIdType, InnerIdType>>& duplicates) {
    const uint64_t row_bytes = checked_product(dimensions, sizeof(float), "row");
    require_argument(row_bytes <= std::numeric_limits<std::size_t>::max(),
                     "PiPNN row size exceeds the platform limit");
    UnorderedMap<std::string_view, InnerIdType> representatives(allocator);
    representatives.reserve(rows.size());

    representative_ids.reserve(ids.size());
    representative_rows.reserve(rows.size());
    for (uint64_t local_id = 0; local_id < ids.size(); ++local_id) {
        const auto key = std::string_view(reinterpret_cast<const char*>(rows[local_id]),
                                          static_cast<std::size_t>(row_bytes));
        const auto [it, inserted] = representatives.emplace(key, ids[local_id]);
        if (inserted) {
            representative_ids.emplace_back(ids[local_id]);
            representative_rows.emplace_back(rows[local_id]);
        } else {
            duplicates.emplace_back(it.value(), ids[local_id]);
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

struct PruneScratch {
    explicit PruneScratch(Allocator* allocator)
        : ordered(allocator), selected(allocator), neighbors(allocator) {
    }

    Vector<std::pair<float, uint32_t>> ordered;
    Vector<uint32_t> selected;
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
          rows_(rows),
          ids_(allocator),
          reservoirs_(allocator),
          reservoir_states_(allocator) {
    }

    ~PiPNNPipeline() {
        if (norms_ != nullptr) {
            allocator_->Deallocate(norms_);
        }
        if (sketches_ != nullptr) {
            allocator_->Deallocate(sketches_);
        }
    }

    void
    Build(const Vector<InnerIdType>& ids_sequence);

private:
    void
    prepare(const Vector<InnerIdType>& ids_sequence);

    void
    prepare_sketches();

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
                   const float* dots,
                   uint64_t fanout,
                   Vector<std::pair<float, uint32_t>>& candidates) const;

    [[nodiscard]] Leaves
    merge_undersized_leaves(Leaves leaves) const;

    void
    build_leaf(const Leaf& leaf);

    template <MetricType metric>
    void
    build_small_leaf(const Leaf& leaf, float* matrix, uint64_t neighbor_count);

    void
    insert_candidate(uint32_t source, uint32_t target, float distance);

    [[nodiscard]] uint16_t
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
    write_graph() const;

    void
    parallel_for(uint64_t total,
                 uint64_t block_size,
                 const std::function<void(uint64_t, uint64_t)>& task) const;

    [[nodiscard]] const float*
    vector_by_local_id(uint32_t local_id) const {
        return rows_[local_id];
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
    const Vector<const float*>& rows_;
    uint64_t reservoir_size_{0};
    Vector<InnerIdType> ids_;
    float* norms_{nullptr};
    float* sketches_{nullptr};
    Vector<ReservoirEntry> reservoirs_;
    Vector<ReservoirState> reservoir_states_;
    std::unique_ptr<PointsMutex> point_locks_;
    uint32_t point_lock_count_{0};
};

void
PiPNNPipeline::Build(const Vector<InnerIdType>& ids_sequence) {
    prepare(ids_sequence);
    if (ids_.empty()) {
        return;
    }

    prepare_sketches();
    const auto leaves = partition();
    parallel_for(leaves.size(), 1, [&](uint64_t begin, uint64_t end) {
        for (uint64_t leaf = begin; leaf < end; ++leaf) {
            build_leaf(leaves[leaf]);
        }
    });
    write_graph();
}

void
PiPNNPipeline::prepare(const Vector<InnerIdType>& ids_sequence) {
    ids_.assign(ids_sequence.begin(), ids_sequence.end());
    require_argument(ids_.size() == rows_.size(), "PiPNN IDs and rows must have equal sizes");
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
            require_argument(rows_[local_id] != nullptr, "PiPNN input rows must not be null");
        }
    } else {
        parallel_for(ids_.size(), 256, [&](uint64_t begin, uint64_t end) {
            for (uint64_t local_id = begin; local_id < end; ++local_id) {
                require_argument(ids_[local_id] < graph_capacity,
                                 "PiPNN input ID is outside the graph capacity");
                require_argument(rows_[local_id] != nullptr, "PiPNN input rows must not be null");
            }
        });
    }

    reservoir_size_ =
        std::max(parameter_.reservoir_size, static_cast<uint64_t>(graph_->MaximumDegree()));
    const uint64_t reservoir_count = checked_product(ids_.size(), reservoir_size_, "reservoir");
    require_argument(reservoir_count <= reservoirs_.max_size(),
                     "PiPNN reservoir exceeds the allocator limit");

    if (metric_ != MetricType::METRIC_TYPE_IP) {
        const uint64_t norm_bytes = checked_product(ids_.size(), sizeof(float), "norm bytes");
        norms_ = static_cast<float*>(allocator_->Allocate(norm_bytes));
        if (norms_ == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "PiPNN failed to allocate norms");
        }
    }
    reservoirs_.resize(reservoir_count);
    reservoir_states_.resize(ids_.size());
    if (thread_pool_ != nullptr and thread_count_ > 1) {
        point_lock_count_ =
            static_cast<uint32_t>(std::min<uint64_t>(ids_.size(), MAX_POINT_LOCK_COUNT));
        point_locks_ = std::make_unique<PointsMutex>(point_lock_count_, allocator_);
    }
}

void
PiPNNPipeline::prepare_sketches() {
    const uint64_t plane_values =
        checked_product(parameter_.hash_plane_count, dimensions_, "hyperplane");
    const uint64_t sketch_values =
        checked_product(ids_.size(), parameter_.hash_plane_count, "sketch");
    Vector<float> hyperplanes(plane_values, allocator_);
    const uint64_t sketch_bytes = checked_product(sketch_values, sizeof(float), "sketch bytes");
    sketches_ = static_cast<float*>(allocator_->Allocate(sketch_bytes));
    if (sketches_ == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "PiPNN failed to allocate sketches");
    }

    std::mt19937_64 random(42);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& value : hyperplanes) {
        value = normal(random);
    }

    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        for (uint64_t local_id = begin; local_id < end; ++local_id) {
            const auto* vector = vector_by_local_id(static_cast<uint32_t>(local_id));
            if (metric_ != MetricType::METRIC_TYPE_IP) {
                float norm = 0.0F;
                for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                    norm += vector[dim] * vector[dim];
                }
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    const float length = std::sqrt(std::max(0.0F, norm));
                    norms_[local_id] =
                        length > 0.0F and std::isfinite(length) ? 1.0F / length : 0.0F;
                } else {
                    norms_[local_id] = norm;
                }
            }
            for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
                const auto* hyperplane = hyperplanes.data() + plane * dimensions_;
                float dot = 0.0F;
                for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                    dot += vector[dim] * hyperplane[dim];
                }
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    dot *= norms_[local_id];
                }
                sketches_[local_id * parameter_.hash_plane_count + plane] = dot;
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
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        const auto* source = vector_by_local_id(leaders[leader]);
        std::copy(source,
                  source + static_cast<int64_t>(dimensions_),
                  leader_values.Data() + static_cast<int64_t>(leader * dimensions_));
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
                        const auto* source = vector_by_local_id(item.points[begin + point]);
                        std::copy(source,
                                  source + static_cast<int64_t>(dimensions_),
                                  point_values.Data() + static_cast<int64_t>(point * dimensions_));
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
            const auto* source = vector_by_local_id(item.points[begin + point]);
            std::copy(source,
                      source + static_cast<int64_t>(dimensions_),
                      point_values.Data() + static_cast<int64_t>(point * dimensions_));
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
            build_small_leaf<MetricType::METRIC_TYPE_L2SQR>(leaf, matrix.Data(), neighbor_count);
            return;
        }
        if (metric_ == MetricType::METRIC_TYPE_IP) {
            build_small_leaf<MetricType::METRIC_TYPE_IP>(leaf, matrix.Data(), neighbor_count);
            return;
        }
    }
    for (uint64_t point = 0; point < point_count; ++point) {
        const auto* source = vector_by_local_id(leaf[point]);
        std::copy(source,
                  source + static_cast<int64_t>(dimensions_),
                  matrix.Data() + static_cast<int64_t>(point * dimensions_));
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
                insert_candidate(leaf[source], leaf[target], nearest[candidate].first);
                insert_candidate(leaf[target], leaf[source], nearest[candidate].first);
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
            insert_candidate(leaf[source], leaf[target], candidates[candidate].first);
            insert_candidate(leaf[target], leaf[source], candidates[candidate].first);
        }
    }
}

template <MetricType metric>
void
PiPNNPipeline::build_small_leaf(const Leaf& leaf, float* matrix, uint64_t neighbor_count) {
    static_assert(metric == MetricType::METRIC_TYPE_L2SQR || metric == MetricType::METRIC_TYPE_IP);
    const uint64_t count = leaf.size();
    Vector<std::pair<float, uint32_t>> nearest(
        checked_product(count, neighbor_count, "leaf candidates"), allocator_);
    Vector<uint64_t> farthest(count, 0, allocator_);
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
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            return std::max(0.0F, sanitize_distance(norms[source] + norms[target] - dot - dot));
        } else {
            return sanitize_distance(1.0F - dot);
        }
    };
    auto find_farthest = [&](uint64_t source, const auto* row) {
        uint64_t worst = 0;
        for (uint64_t point = 1; point < neighbor_count; ++point) {
            if (less(source, row[worst], row[point])) {
                worst = point;
            }
        }
        return worst;
    };
    auto offer = [&](uint64_t source, uint32_t target, float distance) {
        auto* row = nearest.data() + source * neighbor_count;
        const auto candidate = std::make_pair(distance, target);
        auto& worst = farthest[source];
        if (not less(source, candidate, row[worst])) {
            return;
        }
        row[worst] = candidate;
        worst = find_farthest(source, row);
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
            const auto* source = vector_by_local_id(leaf[point]);
            std::copy(source, source + dimensions_, matrix + point * dimensions_);
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
                farthest[source] = find_farthest(source, nearest.data() + source * neighbor_count);
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
            auto worst = find_farthest(row, current.data());
            for (uint64_t column = neighbor_count; column < row; ++column) {
                const auto distance = from_dot(row, column, dot_row[column]);
                const auto column_worst = farthest[column];
                // Equal distances must still reach the ID tie-breaker.
                if (distance > std::max(current[worst].first,
                                        nearest[column * neighbor_count + column_worst].first)) {
                    continue;
                }
                const auto candidate = std::make_pair(distance, static_cast<uint32_t>(column));
                if (less(row, candidate, current[worst])) {
                    current[worst] = candidate;
                    worst = find_farthest(row, current.data());
                }
                offer(column, static_cast<uint32_t>(row), distance);
            }
            std::copy_n(current.data(), neighbor_count, nearest.data() + row * neighbor_count);
            farthest[row] = worst;
        }
    }
    // Preserve source/target emission order and the existing bidirectional reservoir policy.
    for (uint64_t source = 0; source < count; ++source) {
        auto* row = nearest.data() + source * neighbor_count;
        std::sort(row, row + neighbor_count, [&](const auto& lhs, const auto& rhs) {
            return less(source, lhs, rhs);
        });
        for (uint64_t candidate = 0; candidate < neighbor_count; ++candidate) {
            const auto [distance, target] = row[candidate];
            insert_candidate(leaf[source], leaf[target], distance);
            insert_candidate(leaf[target], leaf[source], distance);
        }
    }
}

uint16_t
PiPNNPipeline::relative_hash(uint32_t source, uint32_t target) const {
    uint16_t hash = 0;
    bool coincident = true;
    const uint64_t source_offset = static_cast<uint64_t>(source) * parameter_.hash_plane_count;
    const uint64_t target_offset = static_cast<uint64_t>(target) * parameter_.hash_plane_count;
    for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
        const float source_value = sketches_[source_offset + plane];
        const float target_value = sketches_[target_offset + plane];
        if (target_value >= source_value) {
            hash |= static_cast<uint16_t>(1U << plane);
        }
        coincident = coincident and target_value == source_value;
    }
    if (coincident) {
        hash |= COINCIDENT_HASH_FLAG;
    }
    return hash;
}

void
PiPNNPipeline::insert_candidate(uint32_t source, uint32_t target, float distance) {
    if (source == target) {
        return;
    }

    const uint16_t hash = relative_hash(source, target);
    const uint16_t distance_key = distance_order_key(distance);
    const auto incoming_key = std::make_tuple(distance_key, ids_[target], hash);
    const uint32_t lock_index = point_lock_count_ == 0 ? source : source % point_lock_count_;
    PointLockGuard lock(point_locks_.get(), lock_index);
    auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;

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
    auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
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
    if (parameter_.adaptive_pruning) {
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
            const auto* future =
                vector_by_local_id(ordered[candidate_index + prefetch_distance].second);
            for (uint64_t dim = 0; dim < dimensions_; dim += 16) {
                Prefetch(future + dim);
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

void
PiPNNPipeline::write_graph() const {
    if (thread_pool_ == nullptr or thread_count_ <= 1 or ids_.size() <= 1) {
        PruneScratch scratch(allocator_);
        for (uint64_t source = 0; source < ids_.size(); ++source) {
            const auto& state = reservoir_states_[source];
            const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
            robust_prune(static_cast<uint32_t>(source), row, state.size, scratch);
            graph_->InsertNeighborsById(ids_[source], scratch.neighbors);
        }
        return;
    }

    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        PruneScratch scratch(allocator_);
        for (uint64_t source = begin; source < end; ++source) {
            const auto& state = reservoir_states_[source];
            const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
            robust_prune(static_cast<uint32_t>(source), row, state.size, scratch);
            graph_->InsertNeighborsById(ids_[source], scratch.neighbors);
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
                         const Vector<const float*>& rows) const {
    require_argument(graph != nullptr, "PiPNN graph must not be null");
    parameter_.Validate(graph->MaximumDegree());
    if (graph->GetDuplicateTracker() == nullptr) {
        PiPNNPipeline(
            parameter_, dimensions_, metric_, allocator_, thread_pool_, thread_count_, graph, rows)
            .Build(ids_sequence);
        return;
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
    Vector<const float*> representative_rows(allocator_);
    Vector<std::pair<InnerIdType, InnerIdType>> duplicates(allocator_);
    group_duplicate_rows(ids_sequence,
                         rows,
                         dimensions_,
                         allocator_,
                         representative_ids,
                         representative_rows,
                         duplicates);
    PiPNNPipeline(parameter_,
                  dimensions_,
                  metric_,
                  allocator_,
                  thread_pool_,
                  thread_count_,
                  graph,
                  representative_rows)
        .Build(representative_ids);
    for (const auto& [representative, duplicate] : duplicates) {
        graph->SetDuplicateId(representative, duplicate);
    }
}

}  // namespace vsag
