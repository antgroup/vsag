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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
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
#include "simd/bf16_simd.h"
#include "utils/lock_strategy.h"

namespace vsag {
namespace {

constexpr uint64_t MAX_HASH_PLANES = 15;
constexpr uint16_t COINCIDENT_HASH_FLAG = 1U << 15;
constexpr uint64_t MAX_PARTITION_ITERATIONS = 30;
constexpr uint64_t LEADER_CAP = 1000;
constexpr uint64_t PARTITION_STRIPE_SIZE = 256;
constexpr uint64_t PARTITION_SEED = 1000;
constexpr uint64_t MIN_PARALLEL_PARTITION_POINTS = 4096;

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

uint64_t
mix_seed(uint64_t seed, uint64_t salt) {
    return seed * 6364136223846793005ULL + salt;
}

float
normalized_distance(float distance) {
    if (not std::isfinite(distance)) {
        return std::numeric_limits<float>::infinity();
    }
    return std::max(0.0F, distance);
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
          rows_(rows),
          ids_(allocator),
          norms_(allocator),
          sketches_(allocator),
          reservoirs_(allocator),
          reservoir_states_(allocator) {
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

    [[nodiscard]] Leaves
    merge_undersized_leaves(Leaves leaves) const;

    void
    build_leaf(const Leaf& leaf);

    void
    insert_candidate(uint32_t source, uint32_t target, float distance);

    [[nodiscard]] uint16_t
    relative_hash(uint32_t source, uint32_t target) const;

    void
    update_farthest(uint32_t source);

    [[nodiscard]] float
    pair_distance(uint32_t lhs, uint32_t rhs) const;

    Vector<InnerIdType>
    robust_prune(uint32_t source, const ReservoirEntry* row, uint16_t size) const;

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
    // Build() is synchronous, so the pipeline borrows rows only for its own lifetime.
    const Vector<const float*>& rows_;
    uint64_t reservoir_size_{0};
    Vector<InnerIdType> ids_;
    Vector<float> norms_;
    Vector<float> sketches_;
    Vector<ReservoirEntry> reservoirs_;
    Vector<ReservoirState> reservoir_states_;
    std::unique_ptr<PointsMutex> point_locks_;
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
    UnorderedSet<InnerIdType> seen_ids(allocator_);
    seen_ids.reserve(ids_.size());
    for (uint64_t local_id = 0; local_id < ids_.size(); ++local_id) {
        const auto id = ids_[local_id];
        require_argument(id < graph_capacity, "PiPNN input ID is outside the graph capacity");
        require_argument(seen_ids.emplace(id).second, "PiPNN input IDs must be unique");
        require_argument(rows_[local_id] != nullptr, "PiPNN input rows must not be null");
    }

    reservoir_size_ =
        std::max(parameter_.reservoir_size, static_cast<uint64_t>(graph_->MaximumDegree()));
    const uint64_t reservoir_count = checked_product(ids_.size(), reservoir_size_, "reservoir");
    require_argument(reservoir_count <= reservoirs_.max_size(),
                     "PiPNN reservoir exceeds the allocator limit");

    norms_.resize(ids_.size(), 0.0F);
    parallel_for(ids_.size(), 256, [&](uint64_t begin, uint64_t end) {
        for (uint64_t local_id = begin; local_id < end; ++local_id) {
            const auto* vector = vector_by_local_id(static_cast<uint32_t>(local_id));
            float norm = 0.0F;
            for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                norm += vector[dim] * vector[dim];
            }
            norms_[local_id] = norm;
        }
    });
    reservoirs_.resize(reservoir_count);
    reservoir_states_.resize(ids_.size());
    if (thread_pool_ != nullptr and thread_count_ > 1) {
        point_locks_ =
            std::make_unique<PointsMutex>(static_cast<uint32_t>(ids_.size()), allocator_);
    }
}

void
PiPNNPipeline::prepare_sketches() {
    const uint64_t plane_values =
        checked_product(parameter_.hash_plane_count, dimensions_, "hyperplane");
    const uint64_t sketch_values =
        checked_product(ids_.size(), parameter_.hash_plane_count, "sketch");
    Vector<float> hyperplanes(plane_values, allocator_);
    sketches_.resize(sketch_values);

    std::mt19937_64 random(42);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& value : hyperplanes) {
        value = normal(random);
    }

    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        for (uint64_t local_id = begin; local_id < end; ++local_id) {
            const auto* vector = vector_by_local_id(static_cast<uint32_t>(local_id));
            for (uint64_t plane = 0; plane < parameter_.hash_plane_count; ++plane) {
                const auto* hyperplane = hyperplanes.data() + plane * dimensions_;
                float dot = 0.0F;
                for (uint64_t dim = 0; dim < dimensions_; ++dim) {
                    dot += vector[dim] * hyperplane[dim];
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
            parallel_for(work.size(), 1, [&](uint64_t begin, uint64_t end) {
                for (uint64_t item = begin; item < end; ++item) {
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

    for (uint64_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        auto& cluster = clusters[cluster_id];
        if (cluster.empty()) {
            continue;
        }
        std::sort(cluster.begin(), cluster.end());
        cluster.erase(std::unique(cluster.begin(), cluster.end()), cluster.end());
        if (cluster.size() <= parameter_.max_leaf_size) {
            finished.emplace_back(std::move(cluster));
            continue;
        }

        if (cluster.size() == item.points.size() and fanout == 1) {
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

    Vector<uint32_t> shuffled(item.points.begin(), item.points.end(), allocator_);
    std::mt19937_64 random(mix_seed(item.seed, item.points.size()));
    std::shuffle(shuffled.begin(), shuffled.end(), random);
    shuffled.resize(leader_count);
    return shuffled;
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
    Vector<float> leader_values(leader_value_count, allocator_);
    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
        const auto* source = vector_by_local_id(leaders[leader]);
        std::copy(source,
                  source + static_cast<int64_t>(dimensions_),
                  leader_values.begin() + static_cast<int64_t>(leader * dimensions_));
    }

    if (allow_parallelism and thread_pool_ != nullptr and thread_count_ > 1 and
        item.points.size() >= MIN_PARALLEL_PARTITION_POINTS) {
        const uint64_t assignment_count =
            checked_product(item.points.size(), fanout, "partition assignments");
        Vector<uint32_t> assignments(assignment_count, allocator_);
        const uint64_t stripe_count =
            (item.points.size() + PARTITION_STRIPE_SIZE - 1) / PARTITION_STRIPE_SIZE;
        parallel_for(stripe_count, 4, [&](uint64_t stripe_begin, uint64_t stripe_end) {
            Vector<float> point_values(allocator_);
            Vector<float> dots(allocator_);
            Vector<std::pair<float, uint32_t>> candidates(allocator_);
            candidates.reserve(leaders.size());
            for (uint64_t stripe = stripe_begin; stripe < stripe_end; ++stripe) {
                const uint64_t begin = stripe * PARTITION_STRIPE_SIZE;
                const uint64_t stripe_size =
                    std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size() - begin);
                const uint64_t point_value_count =
                    checked_product(stripe_size, dimensions_, "partition stripe");
                const uint64_t dot_count =
                    checked_product(stripe_size, leaders.size(), "partition distances");
                point_values.resize(point_value_count);
                dots.resize(dot_count);

                for (uint64_t point = 0; point < stripe_size; ++point) {
                    const auto* source = vector_by_local_id(item.points[begin + point]);
                    std::copy(source,
                              source + static_cast<int64_t>(dimensions_),
                              point_values.begin() + static_cast<int64_t>(point * dimensions_));
                }

                BlasFunction::Sgemm(BlasFunction::RowMajor,
                                    BlasFunction::NoTrans,
                                    BlasFunction::Trans,
                                    static_cast<int32_t>(stripe_size),
                                    static_cast<int32_t>(leaders.size()),
                                    static_cast<int32_t>(dimensions_),
                                    1.0F,
                                    point_values.data(),
                                    static_cast<int32_t>(dimensions_),
                                    leader_values.data(),
                                    static_cast<int32_t>(dimensions_),
                                    0.0F,
                                    dots.data(),
                                    static_cast<int32_t>(leaders.size()));

                for (uint64_t point = 0; point < stripe_size; ++point) {
                    const uint64_t point_index = begin + point;
                    const uint32_t local_id = item.points[point_index];
                    candidates.clear();
                    for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
                        const float distance =
                            normalized_distance(norms_[local_id] + norms_[leaders[leader]] -
                                                2.0F * dots[point * leaders.size() + leader]);
                        candidates.emplace_back(distance, static_cast<uint32_t>(leader));
                    }
                    auto comparator = [&](const auto& lhs, const auto& rhs) {
                        if (lhs.first != rhs.first) {
                            return lhs.first < rhs.first;
                        }
                        return ids_[leaders[lhs.second]] < ids_[leaders[rhs.second]];
                    };
                    std::partial_sort(candidates.begin(),
                                      candidates.begin() + static_cast<int64_t>(fanout),
                                      candidates.end(),
                                      comparator);
                    for (uint64_t selected = 0; selected < fanout; ++selected) {
                        assignments[point_index * fanout + selected] = candidates[selected].second;
                    }
                }
            }
        });

        for (uint64_t point = 0; point < item.points.size(); ++point) {
            for (uint64_t selected = 0; selected < fanout; ++selected) {
                clusters[assignments[point * fanout + selected]].emplace_back(item.points[point]);
            }
        }
        return clusters;
    }

    Vector<float> point_values(allocator_);
    Vector<float> dots(allocator_);
    Vector<std::pair<float, uint32_t>> candidates(allocator_);
    candidates.reserve(leaders.size());
    for (uint64_t begin = 0; begin < item.points.size(); begin += PARTITION_STRIPE_SIZE) {
        const uint64_t stripe_size =
            std::min<uint64_t>(PARTITION_STRIPE_SIZE, item.points.size() - begin);
        const uint64_t point_value_count =
            checked_product(stripe_size, dimensions_, "partition stripe");
        const uint64_t dot_count =
            checked_product(stripe_size, leaders.size(), "partition distances");
        point_values.resize(point_value_count);
        dots.resize(dot_count);

        for (uint64_t point = 0; point < stripe_size; ++point) {
            const auto* source = vector_by_local_id(item.points[begin + point]);
            std::copy(source,
                      source + static_cast<int64_t>(dimensions_),
                      point_values.begin() + static_cast<int64_t>(point * dimensions_));
        }

        BlasFunction::Sgemm(BlasFunction::RowMajor,
                            BlasFunction::NoTrans,
                            BlasFunction::Trans,
                            static_cast<int32_t>(stripe_size),
                            static_cast<int32_t>(leaders.size()),
                            static_cast<int32_t>(dimensions_),
                            1.0F,
                            point_values.data(),
                            static_cast<int32_t>(dimensions_),
                            leader_values.data(),
                            static_cast<int32_t>(dimensions_),
                            0.0F,
                            dots.data(),
                            static_cast<int32_t>(leaders.size()));

        for (uint64_t point = 0; point < stripe_size; ++point) {
            const uint32_t local_id = item.points[begin + point];
            candidates.clear();
            for (uint64_t leader = 0; leader < leaders.size(); ++leader) {
                const float distance =
                    normalized_distance(norms_[local_id] + norms_[leaders[leader]] -
                                        2.0F * dots[point * leaders.size() + leader]);
                candidates.emplace_back(distance, static_cast<uint32_t>(leader));
            }
            auto comparator = [&](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return ids_[leaders[lhs.second]] < ids_[leaders[rhs.second]];
            };
            std::partial_sort(candidates.begin(),
                              candidates.begin() + static_cast<int64_t>(fanout),
                              candidates.end(),
                              comparator);
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
    Leaves small(allocator_);
    for (auto& leaf : leaves) {
        if (leaf.size() >= parameter_.min_leaf_size) {
            merged.emplace_back(std::move(leaf));
        } else {
            small.emplace_back(std::move(leaf));
        }
    }

    auto merge_unique = [&](const Leaf& lhs, const Leaf& rhs) {
        Leaf result(lhs.begin(), lhs.end(), allocator_);
        result.insert(result.end(), rhs.begin(), rhs.end());
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    };

    Leaf accumulator(allocator_);
    for (const auto& leaf : small) {
        auto candidate = merge_unique(accumulator, leaf);
        if (candidate.size() > parameter_.max_leaf_size and not accumulator.empty()) {
            merged.emplace_back(std::move(accumulator));
            accumulator = Leaf(leaf.begin(), leaf.end(), allocator_);
        } else {
            accumulator = std::move(candidate);
        }
        if (accumulator.size() >= parameter_.min_leaf_size) {
            merged.emplace_back(std::move(accumulator));
            accumulator = Leaf(allocator_);
        }
    }

    if (not accumulator.empty()) {
        if (not merged.empty()) {
            auto candidate = merge_unique(merged.back(), accumulator);
            if (candidate.size() <= parameter_.max_leaf_size) {
                merged.back() = std::move(candidate);
                return merged;
            }
        }
        merged.emplace_back(std::move(accumulator));
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
    const uint64_t distance_values = checked_product(point_count, point_count, "leaf distances");
    Vector<float> matrix(matrix_values, allocator_);
    Vector<float> distances(distance_values, allocator_);
    for (uint64_t point = 0; point < point_count; ++point) {
        const auto* source = vector_by_local_id(leaf[point]);
        std::copy(source,
                  source + static_cast<int64_t>(dimensions_),
                  matrix.begin() + static_cast<int64_t>(point * dimensions_));
    }

    BlasFunction::Sgemm(BlasFunction::RowMajor,
                        BlasFunction::NoTrans,
                        BlasFunction::Trans,
                        static_cast<int32_t>(point_count),
                        static_cast<int32_t>(point_count),
                        static_cast<int32_t>(dimensions_),
                        -2.0F,
                        matrix.data(),
                        static_cast<int32_t>(dimensions_),
                        matrix.data(),
                        static_cast<int32_t>(dimensions_),
                        0.0F,
                        distances.data(),
                        static_cast<int32_t>(point_count));

    const uint64_t neighbor_count =
        std::min<uint64_t>(parameter_.leaf_neighbor_count, point_count - 1);
    Vector<std::pair<float, uint32_t>> candidates(allocator_);
    candidates.reserve(point_count - 1);
    for (uint64_t source = 0; source < point_count; ++source) {
        candidates.clear();
        for (uint64_t target = 0; target < point_count; ++target) {
            if (source == target) {
                continue;
            }
            const float distance = normalized_distance(distances[source * point_count + target] +
                                                       norms_[leaf[source]] + norms_[leaf[target]]);
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
    const uint16_t distance_key = generic::FloatToBF16(normalized_distance(distance));
    const auto incoming_key = std::make_tuple(distance_key, ids_[target], hash);
    PointLockGuard lock(point_locks_.get(), source);
    auto& state = reservoir_states_[source];
    auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;

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
        row[state.size] = ReservoirEntry{target, hash, distance_key};
        ++state.size;
        update_farthest(source);
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
    float distance = 0.0F;
    for (uint64_t dim = 0; dim < dimensions_; ++dim) {
        const float delta = left[dim] - right[dim];
        distance += delta * delta;
    }
    return normalized_distance(distance);
}

Vector<InnerIdType>
PiPNNPipeline::robust_prune(uint32_t source, const ReservoirEntry* row, uint16_t size) const {
    Vector<uint32_t> candidate_ids(allocator_);
    candidate_ids.reserve(size);
    for (uint16_t index = 0; index < size; ++index) {
        candidate_ids.emplace_back(row[index].neighbor);
    }
    std::sort(candidate_ids.begin(), candidate_ids.end(), [&](uint32_t lhs, uint32_t rhs) {
        return ids_[lhs] < ids_[rhs];
    });
    candidate_ids.erase(std::unique(candidate_ids.begin(), candidate_ids.end()),
                        candidate_ids.end());

    Vector<std::pair<float, uint32_t>> ordered(allocator_);
    ordered.reserve(candidate_ids.size());
    for (const auto candidate : candidate_ids) {
        const float distance = pair_distance(source, candidate);
        ordered.emplace_back(distance, candidate);
    }
    std::sort(ordered.begin(), ordered.end(), [&](const auto& lhs, const auto& rhs) {
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
    });

    const uint64_t max_degree = graph_->MaximumDegree();
    Vector<uint32_t> selected(allocator_);
    selected.reserve(std::min<uint64_t>(ordered.size(), max_degree));
    for (const auto& [source_distance, candidate] : ordered) {
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

    Vector<InnerIdType> result(allocator_);
    result.reserve(selected.size());
    for (const auto neighbor : selected) {
        result.emplace_back(ids_[neighbor]);
    }
    return result;
}

void
PiPNNPipeline::write_graph() const {
    if (thread_pool_ == nullptr or thread_count_ <= 1 or ids_.size() <= 1) {
        for (uint64_t source = 0; source < ids_.size(); ++source) {
            const auto& state = reservoir_states_[source];
            const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
            auto neighbors = robust_prune(static_cast<uint32_t>(source), row, state.size);
            graph_->InsertNeighborsById(ids_[source], neighbors);
        }
        return;
    }

    Vector<Vector<InnerIdType>> graph_rows(allocator_);
    graph_rows.reserve(ids_.size());
    for (uint64_t source = 0; source < ids_.size(); ++source) {
        graph_rows.emplace_back(allocator_);
    }
    parallel_for(ids_.size(), 64, [&](uint64_t begin, uint64_t end) {
        for (uint64_t source = begin; source < end; ++source) {
            const auto& state = reservoir_states_[source];
            const auto* row = reservoirs_.data() + static_cast<uint64_t>(source) * reservoir_size_;
            graph_rows[source] = robust_prune(static_cast<uint32_t>(source), row, state.size);
        }
    });
    for (uint64_t source = 0; source < ids_.size(); ++source) {
        graph_->InsertNeighborsById(ids_[source], graph_rows[source]);
    }
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

    std::exception_ptr first_exception;
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
                                     Allocator* allocator,
                                     SafeThreadPool* thread_pool,
                                     uint64_t thread_count)
    : parameter_(std::move(parameter)),
      dimensions_(dimensions),
      allocator_(allocator),
      thread_pool_(thread_pool),
      thread_count_(std::max<uint64_t>(1, thread_count)) {
    require_argument(dimensions_ > 0, "PiPNN dimensions must be positive");
    require_argument(dimensions_ <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                     "PiPNN dimensions exceed the BLAS limit");
    require_argument(allocator_ != nullptr, "PiPNN allocator must not be null");
}

void
PiPNNGraphBuilder::Build(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids_sequence,
                         const Vector<const float*>& rows) const {
    require_argument(graph != nullptr, "PiPNN graph must not be null");
    parameter_.Validate(graph->MaximumDegree());
    if (graph->GetDuplicateTracker() == nullptr) {
        PiPNNPipeline(parameter_, dimensions_, allocator_, thread_pool_, thread_count_, graph, rows)
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
