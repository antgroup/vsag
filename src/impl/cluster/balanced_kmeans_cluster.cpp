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

#include "balanced_kmeans_cluster.h"

#include <fmt/format.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "impl/blas/blas_function.h"
#include "impl/logger/logger.h"
#include "simd/fp32_simd.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

/// Vectors one scoring task handles. With CENTROID_TILE it bounds the distance tile
/// every task holds, which is what keeps the working set free of a count x k matrix.
constexpr uint64_t ROWS_PER_TASK = 64;
/// Centroids per GEMM call while scoring.
constexpr uint64_t CENTROID_TILE = 1024;
/// Centroids one task scores when a small batch is split by centroids.
constexpr uint64_t COLUMNS_PER_TASK = 512;
/// Upper bound on those tasks, which bounds the partial lists they keep.
constexpr uint64_t MAX_COLUMN_TASKS = 64;
/// Batches up to this many vectors may be split by centroids.
constexpr uint64_t MAX_COLUMN_SPLIT_ROWS = 512;
/// Clusters whose means one task recomputes.
constexpr uint64_t CLUSTERS_PER_TASK = 64;
/// Vectors per reduction task. Fixed, so sums do not depend on the thread count.
constexpr uint64_t ROWS_PER_REDUCTION = 4096;
/// Factor by which the auction tolerance shrinks from one phase to the next.
constexpr double EPSILON_SCALING = 5.0;
/// First tolerance of an auction without useful prices, relative to the cost gap.
constexpr double COLD_EPSILON_FRACTION = 0.25;
/// Phases an auction warm-started with the previous iteration's prices runs above
/// its final tolerance.
constexpr int32_t WARM_PHASES = 2;
/// Smallest tolerance relative to the magnitude of costs and prices, so that every
/// bid raises a price by an amount a double can represent.
constexpr double EPSILON_FLOOR_RATIO = 1e-12;
/// Work, per vector, that one assignment may spend at the default work_budget:
/// bids, and vectors scored again against every centroid (each re-scored vector
/// costs as much as scoring it in a full pass). The first assignment starts from
/// zero prices and gets more. Measured on SIFT, GloVe and uniform data, the first
/// assignment stays below 16 bids and 14 re-scorings per vector, and every later
/// one below 7 and 4; dense blobs that need many clusters each pass these within a
/// few iterations.
constexpr double FIRST_BIDS_PER_VECTOR = 64.0;
constexpr double FIRST_RESCORES_PER_VECTOR = 24.0;
constexpr double LATER_BIDS_PER_VECTOR = 32.0;
constexpr double LATER_RESCORES_PER_VECTOR = 8.0;

constexpr double INF = std::numeric_limits<double>::infinity();
/// Largest squared norm a vector or centroid may have. ||x - c||^2 is at most
/// 2 ||x||^2 + 2 ||c||^2 and |<x, c>| at most ||x|| ||c||, so below this every cost,
/// including the intermediate sums of the GEMM that computes it, is finite in float.
constexpr double MAX_SQUARED_NORM = static_cast<double>(std::numeric_limits<float>::max()) / 4.0;
/// The squared-L2 expansion ||x||^2 + ||c||^2 - 2 <x, c> runs in float, whose rounding
/// is about 2^-24 of ||x||^2 + ||c||^2. When the mean of the centroids lies more than
/// this many times their mean squared distance to it away from the origin, that
/// rounding reaches the auction's epsilon (1e-4 of the costs), so the expansion is
/// taken around the mean instead.
constexpr double CENTERING_RATIO = 1024.0;

/// Keeps BLAS on one thread inside a task and gives the thread its OpenMP setting
/// back afterwards: a task may run on the caller's own thread.
class SingleThreadedBlas {
public:
    SingleThreadedBlas() : saved_(omp_get_max_threads()) {
        omp_set_num_threads(1);
    }

    ~SingleThreadedBlas() {
        omp_set_num_threads(saved_);
    }

    SingleThreadedBlas(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas&
    operator=(const SingleThreadedBlas&) = delete;
    SingleThreadedBlas(SingleThreadedBlas&&) = delete;
    SingleThreadedBlas&
    operator=(SingleThreadedBlas&&) = delete;

private:
    const int saved_;
};

/// Runs `func(task)` for every task on the pool and waits for all of them before
/// rethrowing the first failure, so no task outlives the state it references. A
/// null pool runs the tasks on the calling thread.
template <typename Func>
void
parallel_for(const SafeThreadPoolPtr& pool, uint64_t task_count, const Func& func) {
    if (task_count == 0) {
        return;
    }
    if (task_count == 1 or pool == nullptr) {
        for (uint64_t task = 0; task < task_count; ++task) {
            func(task);
        }
        return;
    }
    std::vector<std::future<void>> futures;
    futures.reserve(task_count);
    try {
        for (uint64_t task = 0; task < task_count; ++task) {
            futures.emplace_back(pool->GeneralEnqueue(func, task));
        }
    } catch (...) {
        // The tasks already queued reference the caller's state: let them finish first.
        for (auto& future : futures) {
            future.wait();
        }
        throw;
    }
    for (auto& future : futures) {
        future.wait();
    }
    for (auto& future : futures) {
        future.get();
    }
}

void
check_vectors(const float* datas,
              uint64_t count,
              int32_t dim,
              const char* name,
              const SafeThreadPoolPtr& pool) {
    if (datas == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, fmt::format("{} cannot be null", name));
    }
    const auto width = static_cast<uint64_t>(dim);
    const uint64_t rows_per_task = std::max<uint64_t>(1, ROWS_PER_REDUCTION * 64 / width);
    const uint64_t task_count = (count + rows_per_task - 1) / rows_per_task;
    std::atomic<bool> finite{true};
    std::atomic<bool> bounded{true};
    parallel_for(pool, task_count, [&](uint64_t task) {
        const uint64_t end = std::min(count, (task + 1) * rows_per_task);
        for (uint64_t row = task * rows_per_task; row < end; ++row) {
            const float* x = datas + row * width;
            double squared = 0.0;
            for (uint64_t d = 0; d < width; ++d) {
                if (not std::isfinite(x[d])) {
                    finite.store(false, std::memory_order_relaxed);
                    return;
                }
                squared += static_cast<double>(x[d]) * static_cast<double>(x[d]);
            }
            if (squared > MAX_SQUARED_NORM) {
                bounded.store(false, std::memory_order_relaxed);
                return;
            }
        }
    });
    if (not finite.load()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("{} must contain only finite values", name));
    }
    if (not bounded.load()) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("{} must have squared norms of at most {}, or distances overflow float",
                        name,
                        MAX_SQUARED_NORM));
    }
}

void
check_problem(uint32_t k, uint64_t count, int32_t dim, uint32_t candidate_count) {
    if (dim <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "dim must be positive");
    }
    if (k == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "k must be positive");
    }
    if (count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "count must be positive");
    }
    // Vector ids are kept as 32-bit values next to every candidate.
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("count must not exceed {}", std::numeric_limits<uint32_t>::max()));
    }
    if (candidate_count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "candidate_count must be positive");
    }
}

/// cost(x, c) = row bias + centroid bias - row scale * <x, c>, which is ||x - c||^2
/// for L2SQR and the IVF routing key otherwise.
struct CostModel {
    BalancedAssignMetric metric{BalancedAssignMetric::L2SQR};
    const float* datas{nullptr};
    uint64_t dim{0};
    const float* centroids{nullptr};
    uint64_t k{0};
    /// ||c - center||^2 for L2SQR, zeros otherwise; `k` entries.
    const float* centroid_bias{nullptr};
    /// What the expansion scores against: for L2SQR the centroids minus `center`,
    /// otherwise the centroids themselves; `k x dim`.
    const float* scored_centroids{nullptr};
    /// L2SQR only, and only when an offset shared by the centroids would cancel away the
    /// precision of small distances (see CENTERING_RATIO): their mean, taken off vectors
    /// and centroids before the expansion; `dim` entries. Null otherwise.
    const float* center{nullptr};

    void
    RowTerms(const float* x, double& bias, double& scale) const {
        if (metric == BalancedAssignMetric::L2SQR) {
            bias = static_cast<double>(FP32ComputeIP(x, x, dim));
            scale = 2.0;
            return;
        }
        bias = 1.0;
        scale = 1.0;
        if (metric == BalancedAssignMetric::COSINE) {
            const double norm = std::sqrt(static_cast<double>(FP32ComputeIP(x, x, dim)));
            scale = norm > 0.0 ? 1.0 / norm : 0.0;
        }
    }

    /// The same cost, computed directly rather than through the expansion.
    [[nodiscard]] double
    Cost(uint64_t row, uint64_t cluster) const {
        const float* x = datas + row * dim;
        const float* c = centroids + cluster * dim;
        if (metric == BalancedAssignMetric::L2SQR) {
            return static_cast<double>(FP32ComputeL2Sqr(x, c, dim));
        }
        double bias = 0.0;
        double scale = 0.0;
        RowTerms(x, bias, scale);
        return bias - scale * static_cast<double>(FP32ComputeIP(x, c, dim));
    }
};

/// The clusters a vector considers between two scorings: `width` of them by
/// increasing cost + price at scoring time, and a lower bound on cost + price for
/// every other cluster. Prices only rise inside an auction, so the bound stays a
/// lower bound until the vector is scored again.
struct CandidateTable {
    CandidateTable(uint64_t count, uint64_t candidate_width, Allocator* allocator)
        : width(candidate_width),
          ids(count * candidate_width, 0, allocator),
          costs(count * candidate_width, 0.0F, allocator),
          bounds(count, INF, allocator) {
    }

    [[nodiscard]] uint64_t
    MemoryUsage() const {
        return ids.size() * sizeof(uint32_t) + costs.size() * sizeof(float) +
               bounds.size() * sizeof(double);
    }

    uint64_t width{0};
    Vector<uint32_t> ids;
    Vector<float> costs;
    Vector<double> bounds;
};

/// The `keep` cheapest clusters of a few rows, each list sorted by cost + price.
/// Strict comparisons keep the lower cluster id first among equal values, so the
/// lists do not depend on how the work was split.
struct RowLists {
    RowLists(uint64_t rows, uint64_t keep_count, Allocator* allocator)
        : keep(keep_count),
          nets(rows * keep_count, INF, allocator),
          costs(rows * keep_count, 0.0F, allocator),
          ids(rows * keep_count, 0, allocator),
          filled(rows, 0, allocator) {
    }

    void
    Offer(uint64_t row, double net, float cost, uint64_t cluster) {
        double* row_nets = nets.data() + row * keep;
        uint64_t& size = filled[row];
        if (size == keep and not(net < row_nets[keep - 1])) {
            return;
        }
        float* row_costs = costs.data() + row * keep;
        uint32_t* row_ids = ids.data() + row * keep;
        uint64_t pos = size == keep ? keep - 1 : size;
        while (pos > 0 and net < row_nets[pos - 1]) {
            row_nets[pos] = row_nets[pos - 1];
            row_costs[pos] = row_costs[pos - 1];
            row_ids[pos] = row_ids[pos - 1];
            --pos;
        }
        row_nets[pos] = net;
        row_costs[pos] = cost;
        row_ids[pos] = static_cast<uint32_t>(cluster);
        if (size < keep) {
            ++size;
        }
    }

    uint64_t keep{0};
    Vector<double> nets;
    Vector<float> costs;
    Vector<uint32_t> ids;
    Vector<uint64_t> filled;
};

/// Scores `local` contiguous vectors against centroids [first, last), one GEMM tile
/// at a time, and offers every cost to `lists`.
void
score_columns(const CostModel& model,
              const double* prices,
              const float* block,
              uint64_t local,
              const Vector<double>& bias,
              const Vector<double>& scale,
              uint64_t first,
              uint64_t last,
              RowLists& lists,
              Vector<float>& tile) {
    const uint64_t dim = model.dim;
    for (uint64_t begin = first; begin < last; begin += CENTROID_TILE) {
        const uint64_t cols = std::min(CENTROID_TILE, last - begin);
        // tile[r * cols + j] = <x_r, c_(begin + j)>
        BlasFunction::Sgemm(BlasFunction::ColMajor,
                            BlasFunction::Trans,
                            BlasFunction::NoTrans,
                            static_cast<int32_t>(cols),
                            static_cast<int32_t>(local),
                            static_cast<int32_t>(dim),
                            1.0F,
                            model.scored_centroids + begin * dim,
                            static_cast<int32_t>(dim),
                            block,
                            static_cast<int32_t>(dim),
                            0.0F,
                            tile.data(),
                            static_cast<int32_t>(cols));
        for (uint64_t r = 0; r < local; ++r) {
            const float* dots = tile.data() + r * cols;
            for (uint64_t j = 0; j < cols; ++j) {
                const uint64_t cluster = begin + j;
                // Rounded to the float the table keeps before it is ranked, so the
                // order, the bound on the others and every later comparison see one
                // value; otherwise a rounded-up cost can sit above the bound of an
                // equal one, and the vector would be scored again without end.
                const auto cost =
                    static_cast<float>(bias[r] + static_cast<double>(model.centroid_bias[cluster]) -
                                       scale[r] * static_cast<double>(dots[j]));
                lists.Offer(r, static_cast<double>(cost) + prices[cluster], cost, cluster);
            }
        }
    }
}

/// Copies the vectors to score into one block (unless they already are one) and
/// computes their cost terms.
const float*
gather_rows(const CostModel& model,
            const uint32_t* rows,
            uint64_t begin,
            uint64_t local,
            Vector<float>& gathered,
            Vector<double>& bias,
            Vector<double>& scale) {
    const uint64_t dim = model.dim;
    const float* block = model.datas + begin * dim;
    if (rows != nullptr or model.center != nullptr) {
        gathered.resize(local * dim);
        for (uint64_t r = 0; r < local; ++r) {
            const uint64_t row =
                rows == nullptr ? begin + r : static_cast<uint64_t>(rows[begin + r]);
            const float* x = model.datas + row * dim;
            float* out = gathered.data() + r * dim;
            if (model.center != nullptr) {
                for (uint64_t d = 0; d < dim; ++d) {
                    out[d] = x[d] - model.center[d];
                }
            } else {
                std::memcpy(out, x, dim * sizeof(float));
            }
        }
        block = gathered.data();
    }
    bias.assign(local, 0.0);
    scale.assign(local, 0.0);
    for (uint64_t r = 0; r < local; ++r) {
        model.RowTerms(block + r * dim, bias[r], scale[r]);
    }
    return block;
}

void
write_lists(const RowLists& lists,
            const uint32_t* rows,
            uint64_t begin,
            uint64_t local,
            CandidateTable& table) {
    const uint64_t width = table.width;
    const uint64_t keep = lists.keep;
    for (uint64_t r = 0; r < local; ++r) {
        const uint64_t row = rows == nullptr ? begin + r : rows[begin + r];
        std::copy_n(lists.ids.data() + r * keep, width, table.ids.data() + row * width);
        std::copy_n(lists.costs.data() + r * keep, width, table.costs.data() + row * width);
        table.bounds[row] = keep > width ? lists.nets[r * keep + width] : INF;
    }
}

/// Scores vectors against every centroid at `prices` and rewrites their candidates.
/// `rows` lists the vectors to score; null scores all `row_count` of them in order.
///
/// Many vectors are split by rows, so every task scores ROWS_PER_TASK of them
/// against all centroids. A few vectors against many centroids, which is what the
/// auction asks for when it scores a small batch again, are split by centroids
/// instead, so that such a batch still uses every thread; the partial lists are
/// then merged. Either way a task holds one tile of at most CENTROID_TILE columns.
void
score_rows(const CostModel& model,
           const double* prices,
           const uint32_t* rows,
           uint64_t row_count,
           CandidateTable& table,
           Allocator* allocator,
           const SafeThreadPoolPtr& pool) {
    const uint64_t keep = model.k > table.width ? table.width + 1 : table.width;
    const uint64_t row_tasks = (row_count + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    const uint64_t column_tasks = std::min(MAX_COLUMN_TASKS, model.k / COLUMNS_PER_TASK);
    if (pool == nullptr or row_tasks >= column_tasks or row_count > MAX_COLUMN_SPLIT_ROWS) {
        parallel_for(pool, row_tasks, [&](uint64_t task) {
            const SingleThreadedBlas single_threaded;
            const uint64_t begin = task * ROWS_PER_TASK;
            const uint64_t local = std::min(ROWS_PER_TASK, row_count - begin);
            Vector<float> gathered(allocator);
            Vector<double> bias(allocator);
            Vector<double> scale(allocator);
            const float* block = gather_rows(model, rows, begin, local, gathered, bias, scale);
            RowLists lists(local, keep, allocator);
            Vector<float> tile(local * std::min(model.k, CENTROID_TILE), 0.0F, allocator);
            score_columns(model, prices, block, local, bias, scale, 0, model.k, lists, tile);
            write_lists(lists, rows, begin, local, table);
        });
        return;
    }

    Vector<float> gathered(allocator);
    Vector<double> bias(allocator);
    Vector<double> scale(allocator);
    const float* block = gather_rows(model, rows, 0, row_count, gathered, bias, scale);
    const uint64_t span = (model.k + column_tasks - 1) / column_tasks;
    std::vector<RowLists> partial;
    partial.reserve(column_tasks);
    for (uint64_t task = 0; task < column_tasks; ++task) {
        partial.emplace_back(row_count, keep, allocator);
    }
    parallel_for(pool, column_tasks, [&](uint64_t task) {
        const SingleThreadedBlas single_threaded;
        const uint64_t first = std::min(model.k, task * span);
        const uint64_t last = std::min(model.k, first + span);
        Vector<float> tile(row_count * std::min(model.k, CENTROID_TILE), 0.0F, allocator);
        score_columns(
            model, prices, block, row_count, bias, scale, first, last, partial[task], tile);
    });
    // Merging in centroid order keeps the lower id first among equal costs, as above.
    RowLists merged(row_count, keep, allocator);
    for (const auto& lists : partial) {
        for (uint64_t r = 0; r < row_count; ++r) {
            for (uint64_t t = 0; t < lists.filled[r]; ++t) {
                const uint64_t at = r * keep + t;
                merged.Offer(r, lists.nets[at], lists.costs[at], lists.ids[at]);
            }
        }
    }
    write_lists(merged, rows, 0, row_count, table);
}

/// Exact capacities: `count % k` clusters hold one vector more than the others. The
/// extra seats go to the clusters that most vectors prefer at the current prices.
/// With fewer vectors than clusters every cluster offers one seat and the auction
/// decides which of them stay empty.
Vector<uint32_t>
choose_capacities(const CandidateTable& table, uint64_t count, uint64_t k, Allocator* allocator) {
    const bool spare = count < k;
    const auto base = spare ? 1U : static_cast<uint32_t>(count / k);
    const uint64_t extra = spare ? 0 : count % k;
    Vector<uint32_t> capacity(k, base, allocator);
    if (extra == 0) {
        return capacity;
    }
    Vector<uint64_t> demand(k, 0, allocator);
    for (uint64_t row = 0; row < count; ++row) {
        ++demand[table.ids[row * table.width]];
    }
    Vector<uint32_t> order(k, 0, allocator);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(),
                      order.begin() + static_cast<int64_t>(extra),
                      order.end(),
                      [&](uint32_t a, uint32_t b) {
                          return demand[a] != demand[b] ? demand[a] > demand[b] : a < b;
                      });
    for (uint64_t i = 0; i < extra; ++i) {
        ++capacity[order[i]];
    }
    return capacity;
}

/// Typical difference between the cheapest and the dearest remembered cluster of a
/// vector: the scale on which prices have to move, and so the scale of epsilon.
double
cost_spread(const CandidateTable& table, uint64_t count) {
    double spread = 0.0;
    double magnitude = 0.0;
    for (uint64_t row = 0; row < count; ++row) {
        const float* costs = table.costs.data() + row * table.width;
        const auto [low, high] = std::minmax_element(costs, costs + table.width);
        spread += static_cast<double>(*high) - static_cast<double>(*low);
        magnitude += std::abs(static_cast<double>(*low));
    }
    spread /= static_cast<double>(count);
    if (spread > 0.0) {
        return spread;
    }
    magnitude /= static_cast<double>(count);
    return magnitude > 0.0 ? magnitude : 1.0;
}

/// Smallest epsilon that still moves a price by a representable amount.
double
epsilon_floor(const CandidateTable& table, const Vector<double>& prices) {
    double magnitude = 0.0;
    for (const auto cost : table.costs) {
        magnitude = std::max(magnitude, std::abs(static_cast<double>(cost)));
    }
    for (const auto price : prices) {
        magnitude = std::max(magnitude, std::abs(price));
    }
    return std::max(magnitude, 1.0) * EPSILON_FLOOR_RATIO;
}

/// The most bids and re-scored vectors one assignment may spend.
struct WorkBudget {
    uint64_t bids{std::numeric_limits<uint64_t>::max()};
    uint64_t rescored{std::numeric_limits<uint64_t>::max()};

    static WorkBudget
    For(uint64_t count, double factor, bool first) {
        WorkBudget budget;
        const auto limit = [&](double per_vector) {
            const double value = per_vector * factor * static_cast<double>(count);
            return value >= static_cast<double>(std::numeric_limits<uint64_t>::max())
                       ? std::numeric_limits<uint64_t>::max()
                       : static_cast<uint64_t>(value);
        };
        budget.bids = limit(first ? FIRST_BIDS_PER_VECTOR : LATER_BIDS_PER_VECTOR);
        budget.rescored = limit(first ? FIRST_RESCORES_PER_VECTOR : LATER_RESCORES_PER_VECTOR);
        return budget;
    }
};

/// The auction that solves one capacity-constrained assignment.
///
/// Every cluster keeps its holders in a min-heap of their bids, inside one slice of
/// a shared array. While a cluster has a free seat its price is its floor price;
/// once full, its price is its lowest bid, the amount a newcomer has to beat.
///
/// A vector bids for the cluster j with the lowest cost + price `a`, offering the
/// current price plus `b - a + epsilon`, where `b` is the second lowest cost + price
/// over the other clusters. At that moment it would pay at most `b + epsilon` in
/// cost + price, prices only rise afterwards, and so every holder stays within
/// epsilon of its best option (epsilon complementary slackness). When all seats are
/// taken, the assignment is optimal within `count * epsilon`.
class CapacityAuction {
public:
    using Rescore = std::function<void(const Vector<uint32_t>&, const Vector<double>&)>;
    using ExactCost = std::function<double(uint64_t, uint64_t)>;

    CapacityAuction(const Vector<uint32_t>& capacity,
                    const Vector<double>& floor_prices,
                    CandidateTable& table,
                    WorkBudget budget,
                    Allocator* allocator)
        : count_(table.bounds.size()),
          k_(capacity.size()),
          width_(table.width),
          table_(table),
          capacity_(capacity),
          offset_(capacity.size() + 1, 0, allocator),
          size_(capacity.size(), 0, allocator),
          floor_(floor_prices),
          bid_(Seats(capacity), 0.0, allocator),
          holder_(Seats(capacity), 0, allocator),
          owner_(table.bounds.size(), -1, allocator),
          queue_(table.bounds.size() + 1, 0, allocator),
          deferred_(allocator),
          budget_(budget),
          allocator_(allocator) {
        for (uint64_t j = 0; j < k_; ++j) {
            offset_[j + 1] = offset_[j] + capacity_[j];
        }
    }

    /// Seats over all clusters: the vector count, or more when some may stay empty.
    static uint64_t
    Seats(const Vector<uint32_t>& capacity) {
        return std::accumulate(capacity.begin(), capacity.end(), uint64_t{0});
    }

    /// Bytes this auction holds per vector and per cluster.
    [[nodiscard]] uint64_t
    MemoryUsage() const {
        return bid_.size() * sizeof(double) + holder_.size() * sizeof(uint32_t) +
               owner_.size() * sizeof(int32_t) + queue_.size() * sizeof(uint32_t) +
               k_ * (2 * sizeof(uint32_t) + sizeof(uint64_t) + sizeof(double));
    }

    /// Assigns every vector, scaling epsilon from `epsilon_start` down to
    /// `epsilon_final`. With `warm`, a complete assignment for the same capacities,
    /// every vector that is still within epsilon of its best option keeps its
    /// cluster and only the others bid. Returns false, leaving `labels` and `prices`
    /// untouched, once the work budget is spent.
    bool
    Solve(double epsilon_start,
          double epsilon_final,
          const Vector<int32_t>* warm,
          const Rescore& rescore,
          Vector<int32_t>& labels,
          Vector<double>& prices,
          BalancedKMeansStats& stats) {
        double epsilon = std::max(epsilon_start, epsilon_final);
        if (warm == nullptr) {
            for (uint64_t row = 0; row < count_; ++row) {
                enqueue(static_cast<uint32_t>(row));
            }
        } else {
            reseat(*warm, epsilon);
        }
        bool first_phase = true;
        while (true) {
            if (not first_phase) {
                tighten(epsilon);
            }
            if (not run_phase(epsilon, rescore, stats)) {
                return false;
            }
            if (epsilon <= epsilon_final) {
                break;
            }
            epsilon = std::max(epsilon_final, epsilon / EPSILON_SCALING);
            first_phase = false;
        }
        std::copy(owner_.begin(), owner_.end(), labels.begin());
        for (uint64_t j = 0; j < k_; ++j) {
            prices[j] = price_of(j);
        }
        return true;
    }

    /// After a Solve that ran out of budget: seats every vector still waiting at its
    /// cheapest cluster with a free seat, by cost + price. The capacities stay exact;
    /// only the optimality bound is given up for these vectors.
    void
    Complete(const ExactCost& exact_cost, Vector<int32_t>& labels, BalancedKMeansStats& stats) {
        Vector<uint32_t> waiting(deferred_);
        while (head_ != tail_) {
            waiting.push_back(dequeue());
        }
        deferred_.clear();
        // Clusters with a free seat, and where each one sits in that list.
        Vector<uint64_t> open(allocator_);
        Vector<int64_t> position(k_, -1, allocator_);
        for (uint64_t j = 0; j < k_; ++j) {
            if (size_[j] < capacity_[j]) {
                position[j] = static_cast<int64_t>(open.size());
                open.push_back(j);
            }
        }
        for (const auto row : waiting) {
            // The cheapest open cluster among the remembered ones is the cheapest open
            // cluster overall when it is not dearer than the bound on all the others.
            const uint64_t base = static_cast<uint64_t>(row) * width_;
            double best = INF;
            uint64_t cluster = k_;
            for (uint64_t t = 0; t < width_; ++t) {
                const uint64_t candidate = table_.ids[base + t];
                const double net =
                    static_cast<double>(table_.costs[base + t]) + price_of(candidate);
                if (position[candidate] >= 0 and net < best) {
                    best = net;
                    cluster = candidate;
                }
            }
            if (cluster == k_ or best > table_.bounds[row]) {
                best = INF;
                for (const auto candidate : open) {
                    const double net = exact_cost(row, candidate) + price_of(candidate);
                    // The first open cluster is taken whatever its cost, so a seat is
                    // always found while vectors are waiting.
                    if (cluster == k_ or net < best) {
                        best = net;
                        cluster = candidate;
                    }
                }
            }
            seat(cluster, row, price_of(cluster));
            if (size_[cluster] == capacity_[cluster]) {
                const auto pos = static_cast<uint64_t>(position[cluster]);
                open[pos] = open.back();
                position[open[pos]] = static_cast<int64_t>(pos);
                open.pop_back();
                position[cluster] = -1;
            }
        }
        stats.fallback_assignments += waiting.size();
        std::copy(owner_.begin(), owner_.end(), labels.begin());
    }

private:
    [[nodiscard]] double
    price_of(uint64_t cluster) const {
        return size_[cluster] < capacity_[cluster] ? floor_[cluster] : bid_[offset_[cluster]];
    }

    void
    enqueue(uint32_t row) {
        queue_[tail_] = row;
        tail_ = tail_ + 1 == queue_.size() ? 0 : tail_ + 1;
    }

    uint32_t
    dequeue() {
        const uint32_t row = queue_[head_];
        head_ = head_ + 1 == queue_.size() ? 0 : head_ + 1;
        return row;
    }

    void
    sift_up(uint64_t base, uint64_t pos) {
        const double bid = bid_[base + pos];
        const uint32_t holder = holder_[base + pos];
        while (pos > 0) {
            const uint64_t parent = (pos - 1) / 2;
            if (not(bid < bid_[base + parent])) {
                break;
            }
            bid_[base + pos] = bid_[base + parent];
            holder_[base + pos] = holder_[base + parent];
            pos = parent;
        }
        bid_[base + pos] = bid;
        holder_[base + pos] = holder;
    }

    void
    sift_down(uint64_t base, uint64_t size, uint64_t pos) {
        const double bid = bid_[base + pos];
        const uint32_t holder = holder_[base + pos];
        while (true) {
            uint64_t child = 2 * pos + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size and bid_[base + child + 1] < bid_[base + child]) {
                ++child;
            }
            if (not(bid_[base + child] < bid)) {
                break;
            }
            bid_[base + pos] = bid_[base + child];
            holder_[base + pos] = holder_[base + child];
            pos = child;
        }
        bid_[base + pos] = bid;
        holder_[base + pos] = holder;
    }

    /// Takes a free seat of `cluster` with the given bid.
    void
    seat(uint64_t cluster, uint32_t row, double bid) {
        owner_[row] = static_cast<int32_t>(cluster);
        const uint64_t base = offset_[cluster];
        const uint64_t pos = size_[cluster]++;
        bid_[base + pos] = bid;
        holder_[base + pos] = row;
        sift_up(base, pos);
    }

    /// Seats every vector of a previous assignment that satisfies epsilon
    /// complementary slackness at the floor prices, with the largest bid that keeps
    /// it satisfied; queues the others. No cluster overflows, because the previous
    /// assignment met the same capacities. A seated vector's cluster can only become
    /// dearer up to its own bid, and every other cluster only dearer, so it stays
    /// within epsilon of its best option, which is the invariant the auction needs.
    void
    reseat(const Vector<int32_t>& previous, double epsilon) {
        for (uint64_t row = 0; row < count_; ++row) {
            const auto cluster = static_cast<uint64_t>(previous[row]);
            const uint64_t base = row * width_;
            double own = INF;
            double alternative = table_.bounds[row];
            for (uint64_t t = 0; t < width_; ++t) {
                const uint64_t candidate = table_.ids[base + t];
                const auto cost = static_cast<double>(table_.costs[base + t]);
                if (candidate == cluster) {
                    own = cost;
                } else {
                    alternative = std::min(alternative, cost + floor_[candidate]);
                }
            }
            if (own + floor_[cluster] <= alternative + epsilon) {
                seat(cluster, static_cast<uint32_t>(row), alternative - own + epsilon);
            } else {
                enqueue(static_cast<uint32_t>(row));
            }
        }
    }

    /// The bid of one vector at the current prices: its cheapest cluster and the
    /// price plus `second - best + epsilon`. False when a cluster it does not
    /// remember may now be cheaper, which means it has to be scored again first.
    bool
    evaluate(uint32_t row, double epsilon, uint64_t& target, double& offer) const {
        const uint64_t base = static_cast<uint64_t>(row) * width_;
        double best = INF;
        double second = INF;
        uint64_t best_cluster = 0;
        for (uint64_t t = 0; t < width_; ++t) {
            const uint64_t cluster = table_.ids[base + t];
            const double net = static_cast<double>(table_.costs[base + t]) + price_of(cluster);
            if (net < best) {
                second = best;
                best = net;
                best_cluster = cluster;
            } else if (net < second) {
                second = net;
            }
        }
        const double bound = table_.bounds[row];
        if (best > bound) {
            return false;
        }
        second = std::min(second, bound);
        const double gap = std::isfinite(second) ? second - best : 0.0;
        target = best_cluster;
        // Epsilon is sized from the costs and prices an auction starts with, but a price
        // can grow far beyond them, up to the bound of a remote cluster, where adding
        // epsilon is lost to rounding: the bid would leave the price where it was, and
        // equal bids could evict each other forever. Every bid raises the price by a
        // step the price can represent.
        const double price = price_of(best_cluster);
        offer = price + gap + std::max(epsilon, std::abs(price) * EPSILON_FLOOR_RATIO);
        return true;
    }

    /// The most a holder of `cluster` would now bid for its seat while staying
    /// within epsilon of its best other option; minus infinity when it does not
    /// remember the cluster. Uses a lower bound on the other options, so it never
    /// claims more than the holder would actually offer.
    [[nodiscard]] double
    willingness(uint32_t holder, uint64_t cluster, double epsilon) const {
        const uint64_t base = static_cast<uint64_t>(holder) * width_;
        double own = INF;
        double alternative = table_.bounds[holder];
        for (uint64_t t = 0; t < width_; ++t) {
            const uint64_t candidate = table_.ids[base + t];
            const auto cost = static_cast<double>(table_.costs[base + t]);
            if (candidate == cluster) {
                own = cost;
            } else {
                alternative = std::min(alternative, cost + price_of(candidate));
            }
        }
        return std::isfinite(own) and std::isfinite(alternative) ? alternative - own + epsilon
                                                                 : -INF;
    }

    /// A bid for a full cluster. Its lowest holder bid when other clusters were
    /// cheaper; if those have risen since, it would now bid more to stay, which is
    /// exactly what it would do after an eviction, so it raises its bid in place
    /// instead of leaving and coming back. The newcomer takes the seat only from a
    /// holder that would really rather go, and waits for another turn otherwise.
    void
    challenge(uint64_t cluster, uint32_t row, double offer, double epsilon) {
        const uint64_t base = offset_[cluster];
        while (true) {
            const uint32_t holder = holder_[base];
            const double defense = willingness(holder, cluster, epsilon);
            if (defense > bid_[base]) {
                bid_[base] = defense;
                sift_down(base, size_[cluster], 0);
                if (not(offer > bid_[base])) {
                    enqueue(row);
                    return;
                }
                continue;
            }
            owner_[holder] = -1;
            enqueue(holder);
            bid_[base] = offer;
            holder_[base] = row;
            sift_down(base, size_[cluster], 0);
            owner_[row] = static_cast<int32_t>(cluster);
            return;
        }
    }

    /// Bids until every vector holds a seat; false once the work budget is spent.
    bool
    run_phase(double epsilon, const Rescore& rescore, BalancedKMeansStats& stats) {
        while (true) {
            while (head_ != tail_) {
                if (bids_ >= budget_.bids) {
                    return false;
                }
                const uint32_t row = dequeue();
                uint64_t target = 0;
                double offer = 0.0;
                if (not evaluate(row, epsilon, target, offer)) {
                    deferred_.push_back(row);
                    continue;
                }
                ++bids_;
                ++stats.bids;
                if (size_[target] < capacity_[target]) {
                    seat(target, row, offer);
                } else {
                    challenge(target, row, offer, epsilon);
                }
            }
            if (deferred_.empty()) {
                return true;
            }
            if (rescored_ + deferred_.size() > budget_.rescored) {
                return false;
            }
            Vector<double> prices(k_, 0.0, allocator_);
            for (uint64_t j = 0; j < k_; ++j) {
                prices[j] = price_of(j);
            }
            rescore(deferred_, prices);
            rescored_ += deferred_.size();
            stats.rescored_vectors += deferred_.size();
            for (const auto row : deferred_) {
                enqueue(row);
            }
            deferred_.clear();
        }
    }

    /// Moves to a smaller epsilon. A holder that is no longer within epsilon of its
    /// best option gives up its seat; the others lower their recorded bid to the
    /// most they could have offered under the new epsilon, which is never below the
    /// current price. Prices therefore do not change here, and a cluster that loses
    /// holders keeps its price as floor, so prices still only rise.
    void
    tighten(double epsilon) {
        for (uint64_t cluster = 0; cluster < k_; ++cluster) {
            const double price = price_of(cluster);
            const uint64_t base = offset_[cluster];
            uint64_t kept = 0;
            for (uint64_t slot = 0; slot < size_[cluster]; ++slot) {
                const uint32_t row = holder_[base + slot];
                const uint64_t row_base = static_cast<uint64_t>(row) * width_;
                double own = INF;
                double alternative = table_.bounds[row];
                for (uint64_t t = 0; t < width_; ++t) {
                    const uint64_t candidate = table_.ids[row_base + t];
                    const auto cost = static_cast<double>(table_.costs[row_base + t]);
                    if (candidate == cluster) {
                        own = cost;
                    } else {
                        alternative = std::min(alternative, cost + price_of(candidate));
                    }
                }
                if (not(own + price <= alternative + epsilon)) {
                    owner_[row] = -1;
                    enqueue(row);
                    continue;
                }
                bid_[base + kept] = std::min(bid_[base + slot], alternative - own + epsilon);
                holder_[base + kept] = row;
                ++kept;
            }
            if (kept < size_[cluster]) {
                floor_[cluster] = std::max(floor_[cluster], price);
            }
            size_[cluster] = kept;
            for (uint64_t pos = kept / 2; pos > 0; --pos) {
                sift_down(base, kept, pos - 1);
            }
        }
    }

private:
    const uint64_t count_{0};
    const uint64_t k_{0};
    const uint64_t width_{0};
    CandidateTable& table_;

    const Vector<uint32_t>& capacity_;
    Vector<uint64_t> offset_;
    Vector<uint32_t> size_;
    Vector<double> floor_;
    Vector<double> bid_;
    Vector<uint32_t> holder_;
    Vector<int32_t> owner_;
    Vector<uint32_t> queue_;
    uint64_t head_{0};
    uint64_t tail_{0};
    Vector<uint32_t> deferred_;
    const WorkBudget budget_;
    uint64_t bids_{0};
    uint64_t rescored_{0};
    Allocator* const allocator_{nullptr};
};

/// Runs one assignment on a scored table: an auction warm-started from `labels`
/// when `warm` is set. Returns false, leaving `labels` and `prices` as they were,
/// when the budget ran out, unless `complete` asks for the remaining vectors to be
/// seated by Complete() instead. `memory` receives the auction's working set.
bool
solve_assignment(const CostModel& model,
                 const Vector<uint32_t>& capacity,
                 Vector<double>& prices,
                 CandidateTable& table,
                 bool warm,
                 double epsilon_ratio,
                 WorkBudget budget,
                 bool complete,
                 Vector<int32_t>& labels,
                 BalancedKMeansStats& stats,
                 uint64_t& memory,
                 Allocator* allocator,
                 const SafeThreadPoolPtr& pool) {
    const double spread = cost_spread(table, labels.size());
    const double epsilon_final = std::max(epsilon_ratio * spread, epsilon_floor(table, prices));
    // With spare seats a single phase keeps the clusters nobody bid for at the lowest
    // price, which the optimality bound of an assignment with empty clusters needs; a
    // later phase could leave a cluster empty at a raised price.
    const bool spare = CapacityAuction::Seats(capacity) > labels.size();
    double epsilon_start = warm ? epsilon_final * std::pow(EPSILON_SCALING, WARM_PHASES)
                                : COLD_EPSILON_FRACTION * spread;
    if (spare) {
        epsilon_start = epsilon_final;
    }
    CapacityAuction auction(capacity, prices, table, budget, allocator);
    memory = std::max(memory, auction.MemoryUsage());
    auto rescore = [&](const Vector<uint32_t>& rows, const Vector<double>& now) {
        score_rows(model, now.data(), rows.data(), rows.size(), table, allocator, pool);
    };
    if (auction.Solve(epsilon_start,
                      epsilon_final,
                      warm ? &labels : nullptr,
                      rescore,
                      labels,
                      prices,
                      stats)) {
        // Only price differences matter; keeping the smallest at 0 stops them drifting.
        const double lowest = *std::min_element(prices.begin(), prices.end());
        for (auto& price : prices) {
            price -= lowest;
        }
        return true;
    }
    if (complete) {
        auction.Complete([&](uint64_t row, uint64_t cluster) { return model.Cost(row, cluster); },
                         labels,
                         stats);
        return true;
    }
    return false;
}

/// Whether squared-L2 costs against these centroids have to be expanded around their
/// mean (see CENTERING_RATIO); leaves the mean in `center`. Decided once per run, so
/// the center stays fixed while the centroids move.
bool
choose_center(const float* centroids, uint64_t k, uint64_t dim, Vector<float>& center) {
    // Summed in double in a fixed order, so the decision does not depend on threads.
    Vector<double> sum(dim, 0.0, center.get_allocator());
    for (uint64_t j = 0; j < k; ++j) {
        for (uint64_t d = 0; d < dim; ++d) {
            sum[d] += static_cast<double>(centroids[j * dim + d]);
        }
    }
    double offset = 0.0;
    for (uint64_t d = 0; d < dim; ++d) {
        center[d] = static_cast<float>(sum[d] / static_cast<double>(k));
        offset += static_cast<double>(center[d]) * static_cast<double>(center[d]);
    }
    double spread = 0.0;
    for (uint64_t j = 0; j < k; ++j) {
        for (uint64_t d = 0; d < dim; ++d) {
            const double diff = static_cast<double>(centroids[j * dim + d] - center[d]);
            spread += diff * diff;
        }
    }
    spread /= static_cast<double>(k);
    return offset > CENTERING_RATIO * spread;
}

/// Fills the centroid terms of `model` for its current centroids: for L2SQR their
/// squared norms, and when the model is centered the centered centroids first; for
/// the other metrics a zero bias, as their centroids are scored as they are.
void
prepare_centroids(const CostModel& model,
                  Vector<float>& scored,
                  Vector<float>& bias,
                  const SafeThreadPoolPtr& pool) {
    const uint64_t dim = model.dim;
    if (model.metric != BalancedAssignMetric::L2SQR) {
        std::fill(bias.begin(), bias.end(), 0.0F);
        return;
    }
    const uint64_t task_count = (model.k + CLUSTERS_PER_TASK - 1) / CLUSTERS_PER_TASK;
    parallel_for(pool, task_count, [&](uint64_t task) {
        const uint64_t end = std::min(model.k, (task + 1) * CLUSTERS_PER_TASK);
        for (uint64_t j = task * CLUSTERS_PER_TASK; j < end; ++j) {
            const float* c = model.centroids + j * dim;
            if (model.center == nullptr) {
                bias[j] = FP32ComputeIP(c, c, dim);
                continue;
            }
            float* out = scored.data() + j * dim;
            for (uint64_t d = 0; d < dim; ++d) {
                out[d] = c[d] - model.center[d];
            }
            bias[j] = FP32ComputeIP(out, out, dim);
        }
    });
}

}  // namespace

BalancedKMeansCluster::BalancedKMeansCluster(int32_t dim,
                                             Allocator* allocator,
                                             SafeThreadPoolPtr thread_pool)
    : allocator_(allocator), thread_pool_(std::move(thread_pool)), dim_(dim) {
    if (allocator_ == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "allocator cannot be null");
    }
    if (thread_pool_ == nullptr) {
        thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    }
}

BalancedKMeansCluster::BalancedKMeansCluster(int32_t dim,
                                             Allocator* allocator,
                                             InlineExecution /*unused*/)
    : allocator_(allocator), dim_(dim) {
}

BalancedKMeansCluster::~BalancedKMeansCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

void
BalancedKMeansCluster::allocate_centroids(uint32_t k) {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    const uint64_t bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_) * sizeof(float);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(bytes));
}

Vector<int32_t>
BalancedKMeansCluster::Run(uint32_t k,
                           const float* datas,
                           uint64_t count,
                           const BalancedKMeansOptions& options,
                           double* err) {
    check_problem(k, count, dim_, options.candidate_count);
    // Every cluster needs a member for its mean.
    if (k > count) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "k cannot be larger than count");
    }
    if (options.iterations <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "iterations must be positive");
    }
    if (not std::isfinite(options.epsilon_ratio) or options.epsilon_ratio <= 0.0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "epsilon_ratio must be positive");
    }
    if (not std::isfinite(options.tolerance) or options.tolerance < 0.0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "tolerance must not be negative");
    }
    if (std::isnan(options.work_budget) or options.work_budget < 0.0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "work_budget must not be negative");
    }
    if (options.init_method != KMeansInitMethod::RANDOM and
        options.init_method != KMeansInitMethod::KMEANS_PLUS_PLUS) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "unknown init_method");
    }
    check_vectors(datas, count, dim_, "datas", thread_pool_);

    stats_ = BalancedKMeansStats{};
    allocate_centroids(k);
    Vector<int32_t> labels(count, 0, allocator_);
    double loss = 0.0;
    if (k == 1) {
        this->update_centroids(datas, count, k, labels);
        loss = this->mean_squared_distance(datas, count, labels);
        stats_.iterations = 1;
        stats_.loss_history.push_back(loss);
    } else {
        const auto outcome = this->run_flat(k, datas, count, options, nullptr, labels);
        if (outcome == FlatOutcome::FIRST_OVER_BUDGET) {
            this->run_two_level(k, datas, count, options, labels);
            loss = this->mean_squared_distance(datas, count, labels);
            stats_.loss_history.assign(1, loss);
            stats_.iterations = options.iterations;
        } else if (outcome == FlatOutcome::LATER_OVER_BUDGET) {
            // The flat iterations got this far before the budget ran out; keep whichever
            // of their result and the two-level one has the lower loss.
            stats_.stopped_on_budget = true;
            const double flat_loss = stats_.loss_history.back();
            const auto dim = static_cast<uint64_t>(dim_);
            Vector<int32_t> flat_labels(labels);
            Vector<float> flat_centroids(
                k_centroids_, k_centroids_ + static_cast<uint64_t>(k) * dim, allocator_);
            this->run_two_level(k, datas, count, options, labels);
            loss = this->mean_squared_distance(datas, count, labels);
            if (flat_loss <= loss) {
                labels = flat_labels;
                std::copy(flat_centroids.begin(), flat_centroids.end(), k_centroids_);
                loss = flat_loss;
                stats_.two_level = false;
            } else {
                stats_.loss_history.assign(1, loss);
                stats_.iterations = options.iterations;
            }
        } else {
            loss = stats_.loss_history.back();
        }
    }
    stats_.working_set_bytes += labels.size() * sizeof(int32_t);
    if (err != nullptr) {
        *err = loss;
    }
    return labels;
}

BalancedKMeansCluster::FlatOutcome
BalancedKMeansCluster::run_flat(uint32_t k,
                                const float* datas,
                                uint64_t count,
                                const BalancedKMeansOptions& options,
                                const Vector<uint32_t>* capacity,
                                Vector<int32_t>& labels) {
    std::random_device rd;
    std::mt19937 gen(options.random_seed.has_value() ? *options.random_seed : rd());
    if (options.init_method == KMeansInitMethod::KMEANS_PLUS_PLUS) {
        this->seed_kmeans_plus_plus(datas, count, k, gen);
    } else {
        this->seed_random(datas, count, k, gen);
    }

    const auto dim = static_cast<uint64_t>(dim_);
    const uint64_t width = std::min<uint64_t>(options.candidate_count, k);
    CandidateTable table(count, width, allocator_);
    Vector<float> centroid_bias(k, 0.0F, allocator_);
    Vector<float> center(dim, 0.0F, allocator_);
    const bool centered = choose_center(k_centroids_, k, dim, center);
    Vector<float> scored_centroids(centered ? static_cast<uint64_t>(k) * dim : 0, 0.0F, allocator_);
    Vector<float> previous_centroids(static_cast<uint64_t>(k) * dim, 0.0F, allocator_);
    Vector<double> prices(k, 0.0, allocator_);
    Vector<uint32_t> sizes(allocator_);
    Vector<int32_t> previous(count, -1, allocator_);
    const CostModel model{BalancedAssignMetric::L2SQR,
                          datas,
                          dim,
                          k_centroids_,
                          k,
                          centroid_bias.data(),
                          centered ? scored_centroids.data() : k_centroids_,
                          centered ? center.data() : nullptr};

    uint64_t auction_bytes = 0;
    auto record_working_set = [&]() {
        stats_.working_set_bytes =
            std::max(stats_.working_set_bytes,
                     table.MemoryUsage() + auction_bytes + previous.size() * sizeof(int32_t) +
                         static_cast<uint64_t>(k) * dim * sizeof(float) +
                         scored_centroids.size() * sizeof(float));
    };
    double loss = 0.0;
    auto outcome = FlatOutcome::COMPLETED;
    for (int32_t iteration = 0; iteration < options.iterations; ++iteration) {
        prepare_centroids(model, scored_centroids, centroid_bias, thread_pool_);
        score_rows(model, prices.data(), nullptr, count, table, allocator_, thread_pool_);
        if (iteration == 0) {
            sizes =
                capacity != nullptr ? *capacity : choose_capacities(table, count, k, allocator_);
        }
        const bool warm = iteration > 0;
        const bool solved = solve_assignment(
            model,
            sizes,
            prices,
            table,
            warm,
            options.epsilon_ratio,
            WorkBudget::For(count,
                            options.work_budget * (warm ? 1.0 : first_assignment_budget_factor_),
                            not warm),
            false,
            labels,
            stats_,
            auction_bytes,
            allocator_,
            thread_pool_);
        if (not solved) {
            if (not warm) {
                record_working_set();
                return FlatOutcome::FIRST_OVER_BUDGET;
            }
            // `labels` still holds the previous assignment, and the centroids are its
            // means: the result of the previous iteration.
            outcome = FlatOutcome::LATER_OVER_BUDGET;
            break;
        }
        // Dual warm start: moving centroid j by d lowers the mean cost of its members by
        // ||d||^2, so raising its price by as much keeps the previous prices close to
        // what clears the next assignment.
        std::copy_n(k_centroids_, static_cast<uint64_t>(k) * dim, previous_centroids.begin());
        this->update_centroids(datas, count, k, labels);
        for (uint64_t j = 0; j < k; ++j) {
            prices[j] += static_cast<double>(
                FP32ComputeL2Sqr(previous_centroids.data() + j * dim, k_centroids_ + j * dim, dim));
        }
        const double previous_loss = loss;
        loss = this->mean_squared_distance(datas, count, labels);
        stats_.loss_history.push_back(loss);
        stats_.iterations = iteration + 1;
        logger::trace("balanced k-means iteration {}/{}: loss {}, bids {}, rescored {}",
                      iteration + 1,
                      options.iterations,
                      loss,
                      stats_.bids,
                      stats_.rescored_vectors);
        if (labels == previous) {
            break;
        }
        if (iteration > 0 and options.tolerance > 0.0 and
            previous_loss - loss <= options.tolerance * previous_loss) {
            break;
        }
        previous = labels;
    }
    record_working_set();
    return outcome;
}

void
BalancedKMeansCluster::run_two_level(uint32_t k,
                                     const float* datas,
                                     uint64_t count,
                                     const BalancedKMeansOptions& options,
                                     Vector<int32_t>& labels) {
    const auto dim = static_cast<uint64_t>(dim_);
    const auto groups = static_cast<uint32_t>(
        std::clamp<double>(std::round(std::sqrt(static_cast<double>(k))), 1.0, k));
    if (groups <= 1) {
        // Too few clusters to split: the flat iterations, without a budget.
        auto unbounded = options;
        unbounded.work_budget = INF;
        this->run_flat(k, datas, count, unbounded, nullptr, labels);
        return;
    }
    stats_.two_level = true;
    // Leaves per super-cluster, as even as possible, and which leaves take one vector
    // more: the `count % k` larger leaves are spread over the super-clusters in order.
    const uint64_t base = count / k;
    uint64_t larger = count % k;
    Vector<uint32_t> leaves(groups, k / groups, allocator_);
    for (uint32_t s = 0; s < k % groups; ++s) {
        ++leaves[s];
    }
    Vector<uint32_t> larger_leaves(groups, 0, allocator_);
    for (uint32_t s = 0; larger > 0; s = (s + 1) % groups) {
        if (larger_leaves[s] < leaves[s]) {
            ++larger_leaves[s];
            --larger;
        }
    }
    Vector<uint32_t> capacity(groups, 0, allocator_);
    Vector<uint32_t> first_leaf(groups, 0, allocator_);
    for (uint32_t s = 0; s < groups; ++s) {
        capacity[s] = static_cast<uint32_t>(leaves[s] * base + larger_leaves[s]);
        first_leaf[s] = s == 0 ? 0 : first_leaf[s - 1] + leaves[s - 1];
    }

    // Both levels must finish, so they run without a budget.
    auto level_options = options;
    level_options.work_budget = INF;
    BalancedKMeansCluster top(dim_, allocator_, thread_pool_);
    Vector<int32_t> top_labels(count, 0, allocator_);
    top.allocate_centroids(groups);
    top.run_flat(groups, datas, count, level_options, &capacity, top_labels);
    stats_.bids += top.stats_.bids;
    stats_.rescored_vectors += top.stats_.rescored_vectors;
    stats_.working_set_bytes = std::max(stats_.working_set_bytes, top.stats_.working_set_bytes);

    Vector<uint64_t> start(static_cast<uint64_t>(groups) + 1, 0, allocator_);
    for (const auto label : top_labels) {
        ++start[static_cast<uint64_t>(label) + 1];
    }
    std::partial_sum(start.begin(), start.end(), start.begin());
    Vector<uint64_t> cursor(start.begin(), start.end() - 1, allocator_);
    Vector<uint32_t> members(count, 0, allocator_);
    for (uint64_t row = 0; row < count; ++row) {
        members[cursor[static_cast<uint64_t>(top_labels[row])]++] = static_cast<uint32_t>(row);
    }

    std::mutex stats_mutex;
    parallel_for(thread_pool_, groups, [&](uint64_t s) {
        const uint64_t size = start[s + 1] - start[s];
        Vector<float> part(size * dim, 0.0F, allocator_);
        for (uint64_t i = 0; i < size; ++i) {
            std::memcpy(part.data() + i * dim,
                        datas + static_cast<uint64_t>(members[start[s] + i]) * dim,
                        dim * sizeof(float));
        }
        auto part_options = level_options;
        if (options.random_seed.has_value()) {
            part_options.random_seed = *options.random_seed + static_cast<uint32_t>(s) + 1;
        }
        // Each part runs on its own task, so it uses no pool of its own.
        BalancedKMeansCluster leaf(dim_, allocator_, InlineExecution{});
        Vector<int32_t> leaf_labels(size, 0, allocator_);
        if (leaves[s] == 1) {
            leaf.allocate_centroids(1);
            leaf.update_centroids(part.data(), size, 1, leaf_labels);
        } else {
            leaf.allocate_centroids(leaves[s]);
            leaf.run_flat(leaves[s], part.data(), size, part_options, nullptr, leaf_labels);
        }
        for (uint64_t i = 0; i < size; ++i) {
            labels[members[start[s] + i]] = static_cast<int32_t>(first_leaf[s] + leaf_labels[i]);
        }
        std::copy_n(leaf.k_centroids_,
                    static_cast<uint64_t>(leaves[s]) * dim,
                    k_centroids_ + static_cast<uint64_t>(first_leaf[s]) * dim);
        std::lock_guard lock(stats_mutex);
        stats_.bids += leaf.stats_.bids;
        stats_.rescored_vectors += leaf.stats_.rescored_vectors;
    });
}

Vector<int32_t>
BalancedKMeansCluster::Assign(const float* centroids,
                              uint32_t k,
                              const float* datas,
                              uint64_t count,
                              BalancedAssignMetric metric,
                              uint32_t candidate_count,
                              double* cost,
                              double work_budget) {
    check_problem(k, count, dim_, candidate_count);
    if (std::isnan(work_budget) or work_budget < 0.0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "work_budget must not be negative");
    }
    if (metric != BalancedAssignMetric::L2SQR and metric != BalancedAssignMetric::INNER_PRODUCT and
        metric != BalancedAssignMetric::COSINE) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "unknown balanced assignment metric");
    }
    check_vectors(centroids, k, dim_, "centroids", thread_pool_);
    check_vectors(datas, count, dim_, "datas", thread_pool_);

    stats_ = BalancedKMeansStats{};
    const auto dim = static_cast<uint64_t>(dim_);
    Vector<float> centroid_bias(k, 0.0F, allocator_);
    Vector<float> center(dim, 0.0F, allocator_);
    const bool centered =
        metric == BalancedAssignMetric::L2SQR and choose_center(centroids, k, dim, center);
    Vector<float> scored_centroids(centered ? static_cast<uint64_t>(k) * dim : 0, 0.0F, allocator_);
    const CostModel model{metric,
                          datas,
                          dim,
                          centroids,
                          k,
                          centroid_bias.data(),
                          centered ? scored_centroids.data() : centroids,
                          centered ? center.data() : nullptr};
    prepare_centroids(model, scored_centroids, centroid_bias, thread_pool_);

    Vector<int32_t> labels(count, 0, allocator_);
    uint64_t auction_bytes = 0;
    if (k > 1) {
        const uint64_t width = std::min<uint64_t>(candidate_count, k);
        CandidateTable table(count, width, allocator_);
        Vector<double> prices(k, 0.0, allocator_);
        score_rows(model, prices.data(), nullptr, count, table, allocator_, thread_pool_);
        const auto capacity = choose_capacities(table, count, k, allocator_);
        const BalancedKMeansOptions defaults;
        solve_assignment(model,
                         capacity,
                         prices,
                         table,
                         false,
                         defaults.epsilon_ratio,
                         WorkBudget::For(count, work_budget, true),
                         true,
                         labels,
                         stats_,
                         auction_bytes,
                         allocator_,
                         thread_pool_);
        stats_.working_set_bytes =
            table.MemoryUsage() + auction_bytes + scored_centroids.size() * sizeof(float);
    }

    const uint64_t task_count = (count + ROWS_PER_REDUCTION - 1) / ROWS_PER_REDUCTION;
    Vector<double> partial(task_count, 0.0, allocator_);
    parallel_for(thread_pool_, task_count, [&](uint64_t task) {
        const uint64_t end = std::min(count, (task + 1) * ROWS_PER_REDUCTION);
        double sum = 0.0;
        for (uint64_t row = task * ROWS_PER_REDUCTION; row < end; ++row) {
            sum += model.Cost(row, static_cast<uint64_t>(labels[row]));
        }
        partial[task] = sum;
    });
    const double mean =
        std::accumulate(partial.begin(), partial.end(), 0.0) / static_cast<double>(count);
    stats_.loss_history.push_back(mean);
    stats_.working_set_bytes += labels.size() * sizeof(int32_t);
    if (cost != nullptr) {
        *cost = mean;
    }
    return labels;
}

void
// NOLINTNEXTLINE(readability-make-member-function-const)
BalancedKMeansCluster::seed_random(const float* datas,
                                   uint64_t count,
                                   uint32_t k,
                                   std::mt19937& gen) {
    // Robert Floyd's sampling of k distinct rows; sorted, so the centroid order does
    // not depend on how the set hashes.
    std::unordered_set<uint64_t> chosen;
    chosen.reserve(k);
    for (uint64_t upper = count - k; upper < count; ++upper) {
        std::uniform_int_distribution<uint64_t> dist(0, upper);
        if (not chosen.insert(dist(gen)).second) {
            chosen.insert(upper);
        }
    }
    std::vector<uint64_t> rows(chosen.begin(), chosen.end());
    std::sort(rows.begin(), rows.end());
    const auto dim = static_cast<uint64_t>(dim_);
    for (uint64_t c = 0; c < k; ++c) {
        std::memcpy(k_centroids_ + c * dim, datas + rows[c] * dim, dim * sizeof(float));
    }
}

void
BalancedKMeansCluster::seed_kmeans_plus_plus(const float* datas,
                                             uint64_t count,
                                             uint32_t k,
                                             std::mt19937& gen) {
    const auto dim = static_cast<uint64_t>(dim_);
    std::uniform_int_distribution<uint64_t> any_row(0, count - 1);
    std::memcpy(k_centroids_, datas + any_row(gen) * dim, dim * sizeof(float));

    const uint64_t task_count = (count + ROWS_PER_REDUCTION - 1) / ROWS_PER_REDUCTION;
    Vector<float> nearest(count, std::numeric_limits<float>::max(), allocator_);
    Vector<double> partial(task_count, 0.0, allocator_);
    for (uint64_t c = 1; c < k; ++c) {
        const float* last = k_centroids_ + (c - 1) * dim;
        parallel_for(thread_pool_, task_count, [&](uint64_t task) {
            const uint64_t end = std::min(count, (task + 1) * ROWS_PER_REDUCTION);
            double sum = 0.0;
            for (uint64_t row = task * ROWS_PER_REDUCTION; row < end; ++row) {
                nearest[row] =
                    std::min(nearest[row], FP32ComputeL2Sqr(datas + row * dim, last, dim));
                sum += static_cast<double>(nearest[row]);
            }
            partial[task] = sum;
        });
        const double total = std::accumulate(partial.begin(), partial.end(), 0.0);
        uint64_t pick = count - 1;
        if (total > 0.0) {
            std::uniform_real_distribution<double> unit(0.0, total);
            const double threshold = unit(gen);
            double cumulative = 0.0;
            uint64_t task = 0;
            while (task + 1 < task_count and cumulative + partial[task] < threshold) {
                cumulative += partial[task];
                ++task;
            }
            const uint64_t end = std::min(count, (task + 1) * ROWS_PER_REDUCTION);
            for (uint64_t row = task * ROWS_PER_REDUCTION; row < end; ++row) {
                cumulative += static_cast<double>(nearest[row]);
                if (cumulative >= threshold) {
                    pick = row;
                    break;
                }
            }
        } else {
            pick = any_row(gen);
        }
        std::memcpy(k_centroids_ + c * dim, datas + pick * dim, dim * sizeof(float));
    }
}

void
BalancedKMeansCluster::update_centroids(const float* datas,
                                        uint64_t count,
                                        uint32_t k,
                                        const Vector<int32_t>& labels) {
    const auto dim = static_cast<uint64_t>(dim_);
    // Counting sort of the vectors by cluster; each task then sums its clusters'
    // members in input order, so the means do not depend on scheduling.
    Vector<uint64_t> offset(static_cast<uint64_t>(k) + 1, 0, allocator_);
    for (const auto label : labels) {
        ++offset[static_cast<uint64_t>(label) + 1];
    }
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    Vector<uint64_t> cursor(offset.begin(), offset.end() - 1, allocator_);
    Vector<uint32_t> members(count, 0, allocator_);
    for (uint64_t row = 0; row < count; ++row) {
        members[cursor[static_cast<uint64_t>(labels[row])]++] = static_cast<uint32_t>(row);
    }
    const uint64_t task_count = (k + CLUSTERS_PER_TASK - 1) / CLUSTERS_PER_TASK;
    parallel_for(thread_pool_, task_count, [&](uint64_t task) {
        const SingleThreadedBlas single_threaded;
        const uint64_t end = std::min<uint64_t>(k, (task + 1) * CLUSTERS_PER_TASK);
        for (uint64_t j = task * CLUSTERS_PER_TASK; j < end; ++j) {
            float* centroid = k_centroids_ + j * dim;
            // Never 0: every cluster holds at least floor(count / k) >= 1 vectors.
            const uint64_t size = offset[j + 1] - offset[j];
            std::fill_n(centroid, dim, 0.0F);
            for (uint64_t m = offset[j]; m < offset[j + 1]; ++m) {
                BlasFunction::Saxpy(static_cast<int32_t>(dim),
                                    1.0F,
                                    datas + static_cast<uint64_t>(members[m]) * dim,
                                    1,
                                    centroid,
                                    1);
            }
            BlasFunction::Sscal(
                static_cast<int32_t>(dim), 1.0F / static_cast<float>(size), centroid, 1);
        }
    });
}

double
BalancedKMeansCluster::mean_squared_distance(const float* datas,
                                             uint64_t count,
                                             const Vector<int32_t>& labels) const {
    const auto dim = static_cast<uint64_t>(dim_);
    const uint64_t task_count = (count + ROWS_PER_REDUCTION - 1) / ROWS_PER_REDUCTION;
    Vector<double> partial(task_count, 0.0, allocator_);
    parallel_for(thread_pool_, task_count, [&](uint64_t task) {
        const uint64_t end = std::min(count, (task + 1) * ROWS_PER_REDUCTION);
        double sum = 0.0;
        for (uint64_t row = task * ROWS_PER_REDUCTION; row < end; ++row) {
            sum += static_cast<double>(FP32ComputeL2Sqr(
                datas + row * dim, k_centroids_ + static_cast<uint64_t>(labels[row]) * dim, dim));
        }
        partial[task] = sum;
    });
    return std::accumulate(partial.begin(), partial.end(), 0.0) / static_cast<double>(count);
}

}  // namespace vsag
