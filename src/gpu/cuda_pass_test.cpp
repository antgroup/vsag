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

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cuda_backend.h"
#include "gpu_plan.h"
#include "unittest.h"

// What the compute passes produce, checked against a host reference.
//
// Sizing is covered in gpu_plan_test.cpp, which needs no device. The cases here
// do: with the backend compiled out the file collapses to the refusal cases,
// and with it compiled in but no card present each of the rest skips.

using namespace vsag::gpu;

namespace {

/// The plans refuse anything small, so every case here passes a min_work of 1
/// and works at sizes a test can check by hand.
constexpr uint64_t kForceOffload = 1;

std::vector<float>
RandomRows(uint64_t rows, int32_t dim, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(rows * static_cast<uint64_t>(dim));
    for (auto& v : out) {
        v = dist(gen);
    }
    return out;
}

}  // namespace

// Holds in every build. With the backend compiled out these are the only cases
// in the file that run, and they pin the promise the stub makes.
TEST_CASE("A pass that refuses leaves the caller's buffers alone", "[ut][gpu_pass]") {
    const int32_t dim = 8;
    const uint64_t rows = 64;
    const uint32_t k = 4;
    const auto data = RandomRows(rows, dim, 7);
    const auto centroids = RandomRows(k, dim, 8);

    // The default threshold is far above a problem this size, so every pass
    // refuses whether or not a device is present.
    std::vector<int32_t> labels(rows, -7);
    double error = -1.0;
    REQUIRE_FALSE(CudaAssignNearest(
        data.data(), rows, centroids.data(), k, dim, labels.data(), &error, 1ULL << 30, 0));
    REQUIRE(std::all_of(labels.begin(), labels.end(), [](int32_t v) { return v == -7; }));
    REQUIRE(error == -1.0);

    std::vector<float> seeds(static_cast<uint64_t>(k) * dim, -7.0F);
    const std::vector<float> uniforms(2 * static_cast<uint64_t>(k), 0.5F);
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(
        data.data(), rows, dim, k, uniforms.data(), seeds.data(), 1ULL << 30, 0));
    REQUIRE(std::all_of(seeds.begin(), seeds.end(), [](float v) { return v == -7.0F; }));

    std::vector<float> sums(static_cast<uint64_t>(k) * dim, -7.0F);
    std::vector<int32_t> counts(k, -7);
    REQUIRE_FALSE(CudaAccumulateCentroids(
        data.data(), rows, dim, labels.data(), k, sums.data(), counts.data(), 1ULL << 30, 0));
    REQUIRE(std::all_of(sums.begin(), sums.end(), [](float v) { return v == -7.0F; }));
    REQUIRE(std::all_of(counts.begin(), counts.end(), [](int32_t v) { return v == -7; }));
}

TEST_CASE("A null or degenerate argument is refused", "[ut][gpu_pass]") {
    const int32_t dim = 4;
    const uint64_t rows = 32;
    const uint32_t k = 2;
    const auto data = RandomRows(rows, dim, 9);
    std::vector<int32_t> labels(rows, 0);
    std::vector<float> out(static_cast<uint64_t>(k) * dim, 0.0F);
    std::vector<int32_t> counts(k, 0);
    const std::vector<float> uniforms(2 * static_cast<uint64_t>(k), 0.25F);

    // Every pointer of every pass, because a null is the one degenerate argument
    // the plans cannot see: they are handed sizes, not addresses, so nothing but
    // these guards stands between a null and the first transfer.
    REQUIRE_FALSE(CudaAssignNearest(
        nullptr, rows, data.data(), k, dim, labels.data(), nullptr, 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaAssignNearest(
        data.data(), rows, nullptr, k, dim, labels.data(), nullptr, 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaAssignNearest(
        data.data(), rows, data.data(), k, dim, nullptr, nullptr, 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaAssignNearest(
        data.data(), 0, data.data(), k, dim, labels.data(), nullptr, 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(
        nullptr, rows, dim, k, uniforms.data(), out.data(), 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(
        data.data(), rows, dim, k, nullptr, out.data(), 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(
        data.data(), rows, dim, k, uniforms.data(), nullptr, 1ULL << 30, kForceOffload));
    // More centroids than there are points has no k-means++ answer.
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(data.data(),
                                         rows,
                                         dim,
                                         static_cast<uint32_t>(rows) + 1,
                                         uniforms.data(),
                                         out.data(),
                                         1ULL << 30,
                                         kForceOffload));
    REQUIRE_FALSE(CudaAccumulateCentroids(nullptr,
                                          rows,
                                          dim,
                                          labels.data(),
                                          k,
                                          out.data(),
                                          counts.data(),
                                          1ULL << 30,
                                          kForceOffload));
    REQUIRE_FALSE(CudaAccumulateCentroids(
        data.data(), rows, dim, nullptr, k, out.data(), counts.data(), 1ULL << 30, kForceOffload));
    REQUIRE_FALSE(CudaAccumulateCentroids(data.data(),
                                          rows,
                                          dim,
                                          labels.data(),
                                          k,
                                          nullptr,
                                          counts.data(),
                                          1ULL << 30,
                                          kForceOffload));
    REQUIRE_FALSE(CudaAccumulateCentroids(
        data.data(), rows, dim, labels.data(), k, out.data(), nullptr, 1ULL << 30, kForceOffload));
}

#ifdef VSAG_ENABLE_CUDA

namespace {

bool
NoDevice() {
    return not CudaAvailable();
}

double
SquaredDistance(const float* a, const float* b, int32_t dim) {
    double acc = 0.0;
    for (int32_t j = 0; j < dim; ++j) {
        const double d = static_cast<double>(a[j]) - static_cast<double>(b[j]);
        acc += d * d;
    }
    return acc;
}

/// The distance to the nearest centroid, computed the obvious way.
double
NearestDistanceOnHost(const float* row, const float* centroids, uint64_t k, int32_t dim) {
    double best = SquaredDistance(row, centroids, dim);
    for (uint64_t c = 1; c < k; ++c) {
        best =
            std::min(best, SquaredDistance(row, centroids + c * static_cast<uint64_t>(dim), dim));
    }
    return best;
}

/// Random points sit at near ties that the reassociated product can resolve
/// either way, so what is asked of a label is that the centroid it names is
/// nearest to within float rounding, not that its index matches the host's.
void
RequireNearest(const std::vector<float>& query,
               const std::vector<float>& centroids,
               const std::vector<int32_t>& labels,
               uint64_t rows,
               uint64_t k,
               int32_t dim) {
    for (uint64_t i = 0; i < rows; ++i) {
        REQUIRE(labels[i] >= 0);
        REQUIRE(static_cast<uint64_t>(labels[i]) < k);
        const double best = NearestDistanceOnHost(query.data() + i * dim, centroids.data(), k, dim);
        const double chosen = SquaredDistance(
            query.data() + i * dim, centroids.data() + static_cast<uint64_t>(labels[i]) * dim, dim);
        REQUIRE(chosen <= best + 1e-3 * (1.0 + best));
    }
}

}  // namespace

TEST_CASE("Assignment picks the centroid the host would", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 16;
    const uint64_t rows = 3000;
    const uint64_t k = 48;

    // Well separated centroids, each query a small perturbation of one of them.
    // The nearest is then unambiguous and the device has to agree exactly.
    std::vector<float> centroids(k * static_cast<uint64_t>(dim), 0.0F);
    for (uint64_t c = 0; c < k; ++c) {
        for (int32_t j = 0; j < dim; ++j) {
            centroids[c * dim + j] = static_cast<float>(c) * 100.0F + static_cast<float>(j);
        }
    }
    std::mt19937 gen(11);
    std::uniform_real_distribution<float> jitter(-1.0F, 1.0F);
    std::vector<float> query(rows * static_cast<uint64_t>(dim));
    std::vector<int32_t> expected(rows);
    for (uint64_t i = 0; i < rows; ++i) {
        const uint64_t owner = i % k;
        expected[i] = static_cast<int32_t>(owner);
        for (int32_t j = 0; j < dim; ++j) {
            query[i * dim + j] = centroids[owner * dim + j] + jitter(gen);
        }
    }

    std::vector<int32_t> labels(rows, -1);
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              labels.data(),
                              nullptr,
                              CudaSuggestedBudget(kChunkedBudgetCap),
                              kForceOffload));
    REQUIRE(labels == expected);
}

TEST_CASE("Every pass runs at the smallest legal shape", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // The cases above start at four dimensions and sixteen centroids. One of each
    // is where the launch geometry is least like the data: the argmin block is 256
    // threads wide whatever `kc` is, so with a single centroid 255 of them hold
    // FLT_MAX and -1 through the reduction, and with a single dimension only lane
    // zero enters the strided loops the row norms and the distances are built from.
    for (const int32_t dim : {1, 3}) {
        for (const uint32_t k : {1U, 3U}) {
            CAPTURE(dim, k);
            const uint64_t rows = 5;
            const auto query = RandomRows(rows, dim, 71);
            const auto centroids = RandomRows(k, dim, 72);
            const uint64_t budget = CudaSuggestedBudget(kChunkedBudgetCap);

            std::vector<int32_t> labels(rows, -7);
            double error = -1.0;
            REQUIRE(CudaAssignNearest(query.data(),
                                      rows,
                                      centroids.data(),
                                      k,
                                      dim,
                                      labels.data(),
                                      &error,
                                      budget,
                                      kForceOffload));
            RequireNearest(query, centroids, labels, rows, k, dim);
            // The mean squared distance, against the labels just returned.
            double host_error = 0.0;
            for (uint64_t i = 0; i < rows; ++i) {
                host_error +=
                    SquaredDistance(query.data() + i * dim,
                                    centroids.data() + static_cast<uint64_t>(labels[i]) * dim,
                                    dim);
            }
            host_error /= static_cast<double>(rows);
            REQUIRE(std::abs(error - host_error) <= 1e-4 * (1.0 + host_error));

            // Accumulation over the labels the assignment just produced.
            std::vector<float> sums(static_cast<uint64_t>(k) * dim, 7.0F);
            std::vector<int32_t> counts(k, 5);
            REQUIRE(CudaAccumulateCentroids(query.data(),
                                            rows,
                                            dim,
                                            labels.data(),
                                            k,
                                            sums.data(),
                                            counts.data(),
                                            budget,
                                            kForceOffload));
            std::vector<double> want_sums(static_cast<uint64_t>(k) * dim, 0.0);
            std::vector<int32_t> want_counts(k, 0);
            for (uint64_t i = 0; i < rows; ++i) {
                want_counts[labels[i]]++;
                for (int32_t j = 0; j < dim; ++j) {
                    want_sums[static_cast<uint64_t>(labels[i]) * dim + j] += query[i * dim + j];
                }
            }
            REQUIRE(counts == want_counts);
            for (uint64_t i = 0; i < sums.size(); ++i) {
                REQUIRE(std::abs(static_cast<double>(sums[i]) - want_sums[i]) <=
                        1e-4 * (1.0 + std::abs(want_sums[i])));
            }

            // Seeding needs k <= rows, which holds for both k here.
            const std::vector<float> uniforms(2 * static_cast<uint64_t>(k), 0.5F);
            std::vector<float> seeds(static_cast<uint64_t>(k) * dim, -7.0F);
            REQUIRE(CudaKMeansPlusPlusInit(query.data(),
                                           rows,
                                           dim,
                                           k,
                                           uniforms.data(),
                                           seeds.data(),
                                           CudaSuggestedBudget(0),
                                           kForceOffload));
            // Every seed is a row of the input, copied whole.
            for (uint32_t c = 0; c < k; ++c) {
                bool found = false;
                for (uint64_t i = 0; i < rows and not found; ++i) {
                    found = std::equal(query.begin() + static_cast<int64_t>(i * dim),
                                       query.begin() + static_cast<int64_t>((i + 1) * dim),
                                       seeds.begin() + static_cast<int64_t>(c * dim));
                }
                REQUIRE(found);
            }
        }
    }
}

TEST_CASE("Assignment splits into chunks without changing its answer", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 32;
    const uint64_t rows = 4000;
    const uint64_t k = 64;
    const uint64_t tight_budget = 1100000;
    const auto query = RandomRows(rows, dim, 21);
    const auto centroids = RandomRows(k, dim, 22);

    // A budget that holds the centroids and a little over the minimum chunk, so
    // the pass has to make several passes over the query rows.
    const AssignPlan tight = PlanAssign(rows, k, dim, tight_budget, kForceOffload);
    REQUIRE(tight.offload);
    REQUIRE(tight.n_chunk < rows);

    std::vector<int32_t> chunked(rows, -1);
    double chunked_error = -1.0;
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              chunked.data(),
                              &chunked_error,
                              tight_budget,
                              kForceOffload));

    std::vector<int32_t> whole(rows, -1);
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              whole.data(),
                              nullptr,
                              CudaSuggestedBudget(kChunkedBudgetCap),
                              kForceOffload));
    // Not the same bucket ids. Chunking changes the SGEMM's n, and cuBLAS is
    // free to pick a different kernel for a different shape, so the two runs can
    // break a tie between equidistant centroids their own way. What must not
    // change is how far away the centroid the label names is.
    RequireNearest(query, centroids, chunked, rows, k, dim);
    RequireNearest(query, centroids, whole, rows, k, dim);
    for (uint64_t i = 0; i < rows; ++i) {
        const double from_chunked =
            SquaredDistance(query.data() + i * dim,
                            centroids.data() + static_cast<uint64_t>(chunked[i]) * dim,
                            dim);
        const double from_whole = SquaredDistance(
            query.data() + i * dim, centroids.data() + static_cast<uint64_t>(whole[i]) * dim, dim);
        REQUIRE(std::abs(from_chunked - from_whole) <= 1e-4 * (1.0 + from_whole));
    }

    // The reported error is the mean squared distance. That is what the host
    // assignment returns and what the k-means loop compares across iterations,
    // so a pass that reported a sum would stop the loop at the wrong point.
    double host_error = 0.0;
    for (uint64_t i = 0; i < rows; ++i) {
        host_error += SquaredDistance(query.data() + i * dim,
                                      centroids.data() + static_cast<uint64_t>(chunked[i]) * dim,
                                      dim);
    }
    host_error /= static_cast<double>(rows);
    REQUIRE(std::abs(chunked_error - host_error) <= 1e-4 * (1.0 + host_error));
}

TEST_CASE("Assignment agrees with the host at a build-sized shape", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // The correctness cases above run at four to thirty-two dimensions and a few
    // dozen centroids. This is the shape the feature exists for, and it combines
    // two things none of them does: a dimension a real corpus has, and a centroid
    // slice wider than the argmin block, so each thread walks the distance column
    // in four strides rather than one. The 100000-centroid case exercises that
    // stride at four dimensions only.
    const int32_t dim = 128;
    const uint64_t k = 1024;
    const uint64_t rows = 2000;
    const auto query = RandomRows(rows, dim, 91);
    const auto centroids = RandomRows(k, dim, 92);

    const AssignPlan plan =
        PlanAssign(rows, k, dim, CudaSuggestedBudget(kChunkedBudgetCap), kForceOffload);
    REQUIRE(plan.offload);
    REQUIRE(plan.centroids_resident);
    // The slice is wider than the block, which is what makes the stride run.
    REQUIRE(plan.k_chunk > 256);

    std::vector<int32_t> labels(rows, -1);
    double error = -1.0;
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              labels.data(),
                              &error,
                              CudaSuggestedBudget(kChunkedBudgetCap),
                              kForceOffload));
    RequireNearest(query, centroids, labels, rows, k, dim);

    double host_error = 0.0;
    for (uint64_t i = 0; i < rows; ++i) {
        host_error += SquaredDistance(
            query.data() + i * dim, centroids.data() + static_cast<uint64_t>(labels[i]) * dim, dim);
    }
    host_error /= static_cast<double>(rows);
    REQUIRE(std::abs(error - host_error) <= 1e-4 * (1.0 + host_error));
}

TEST_CASE("Assignment streams centroids that cannot stay resident", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // More centroids than one slice holds, under a budget too small to keep
    // them resident. That is the path where the pass walks the centroid set in
    // slices and folds each slice into the running best, and where an offset
    // mistake would name a centroid from the wrong slice.
    const int32_t dim = 4;
    const uint64_t rows = 1;
    const uint64_t k = 100000;
    const uint64_t budget = 1000000;
    const auto query = RandomRows(rows, dim, 31);
    const auto centroids = RandomRows(k, dim, 32);

    const AssignPlan plan = PlanAssign(rows, k, dim, budget, kForceOffload);
    REQUIRE(plan.offload);
    REQUIRE_FALSE(plan.centroids_resident);
    REQUIRE(plan.k_chunk < k);

    std::vector<int32_t> labels(rows, -1);
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              labels.data(),
                              nullptr,
                              budget,
                              kForceOffload));
    RequireNearest(query, centroids, labels, rows, k, dim);
}

TEST_CASE("The reported error survives centroid slicing", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // The error was only ever checked with the centroids resident; the streaming
    // case above asks for none. Here both loops run, and three ways of getting the
    // error wrong under them were each confirmed to fail this: dropping the launch
    // that folds in the query's own squared norm, summing only the first row chunk,
    // and losing the slice offset so a label names a centroid from the wrong slice.
    //
    // Moving that norm launch inside the slice loop, which is what it looks like it
    // should catch, is in fact harmless: the argmin overwrites best_val whenever a
    // later slice wins, so an earlier slice's addition is wiped rather than
    // accumulated, and only the last one survives.
    //
    // The nearest centroid is not re-derived here. Over this many centroids a host
    // reference would cost minutes, and it is covered at rows = 1 by the streaming
    // case. What is checked is that the error agrees with the labels returned.
    const int32_t dim = 32;
    const uint64_t k = 300000;
    const uint64_t rows = 1500;
    const uint64_t budget = 70ULL << 20;
    const auto query = RandomRows(rows, dim, 111);
    const auto centroids = RandomRows(k, dim, 112);

    const AssignPlan plan = PlanAssign(rows, k, dim, budget, kForceOffload);
    REQUIRE(plan.offload);
    // Both loops have to run, or the case proves nothing.
    REQUIRE_FALSE(plan.centroids_resident);
    REQUIRE(plan.k_chunk < k);
    REQUIRE(plan.n_chunk < rows);

    std::vector<int32_t> labels(rows, -1);
    double error = -1.0;
    REQUIRE(CudaAssignNearest(query.data(),
                              rows,
                              centroids.data(),
                              k,
                              dim,
                              labels.data(),
                              &error,
                              budget,
                              kForceOffload));

    double host_error = 0.0;
    for (uint64_t i = 0; i < rows; ++i) {
        REQUIRE(labels[i] >= 0);
        REQUIRE(static_cast<uint64_t>(labels[i]) < k);
        host_error += SquaredDistance(
            query.data() + i * dim, centroids.data() + static_cast<uint64_t>(labels[i]) * dim, dim);
    }
    host_error /= static_cast<double>(rows);
    REQUIRE(std::abs(error - host_error) <= 1e-3 * (1.0 + host_error));
}

TEST_CASE("Accumulation sums each cluster and counts its points", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 8;
    const uint64_t rows = 5000;
    const uint32_t k = 16;
    const uint64_t budget = 40000;
    const auto data = RandomRows(rows, dim, 41);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> pick(-2, static_cast<int32_t>(k) + 1);
    std::vector<int32_t> labels(rows);
    for (auto& l : labels) {
        l = pick(gen);
    }

    // Labels outside [0, k) are skipped, which is how the host loop treats the
    // ones the assignment pass could not place.
    std::vector<double> expect_sums(static_cast<uint64_t>(k) * dim, 0.0);
    std::vector<int32_t> expect_counts(k, 0);
    for (uint64_t i = 0; i < rows; ++i) {
        const int32_t l = labels[i];
        if (l < 0 or static_cast<uint32_t>(l) >= k) {
            continue;
        }
        expect_counts[l]++;
        for (int32_t j = 0; j < dim; ++j) {
            expect_sums[static_cast<uint64_t>(l) * dim + j] +=
                static_cast<double>(data[i * dim + j]);
        }
    }

    // A budget just over the minimum chunk, so the points are streamed in
    // several passes and the accumulators have to survive across them.
    const AccumulatePlan plan = PlanAccumulate(rows, dim, k, budget, kForceOffload);
    REQUIRE(plan.offload);
    REQUIRE(plan.n_chunk < rows);

    // Filled with a sentinel, not zeroed: the pass promises to overwrite these
    // rather than add into them, and zeroed buffers cannot tell the two apart.
    // The caller's host fallback does add, which is why the difference matters.
    std::vector<float> sums(static_cast<uint64_t>(k) * dim, 7.0F);
    std::vector<int32_t> counts(k, 5);
    REQUIRE(CudaAccumulateCentroids(data.data(),
                                    rows,
                                    dim,
                                    labels.data(),
                                    k,
                                    sums.data(),
                                    counts.data(),
                                    budget,
                                    kForceOffload));
    REQUIRE(counts == expect_counts);
    // The device adds with atomics, so the order differs from the host loop and
    // the sums agree only to float rounding over a few hundred terms.
    for (uint64_t i = 0; i < sums.size(); ++i) {
        REQUIRE(std::abs(static_cast<double>(sums[i]) - expect_sums[i]) <=
                1e-3 * (1.0 + std::abs(expect_sums[i])));
    }
}

TEST_CASE("Seeding draws distinct rows of the dataset", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 8;
    const uint64_t rows = 4096;
    const uint32_t k = 24;
    const auto data = RandomRows(rows, dim, 51);

    std::mt19937 gen(52);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::vector<float> uniforms(2 * static_cast<uint64_t>(k));
    for (auto& u : uniforms) {
        u = unit(gen);
    }

    std::vector<float> centroids(static_cast<uint64_t>(k) * dim, 0.0F);
    REQUIRE(CudaKMeansPlusPlusInit(data.data(),
                                   rows,
                                   dim,
                                   k,
                                   uniforms.data(),
                                   centroids.data(),
                                   CudaSuggestedBudget(0),
                                   kForceOffload));

    // Every centroid is a row of the dataset, copied whole. Finding which row
    // also shows the draws are distinct, which is what k-means++ gives as long
    // as no point sits on a centroid already chosen.
    std::vector<uint64_t> chosen;
    for (uint32_t c = 0; c < k; ++c) {
        uint64_t found = rows;
        for (uint64_t i = 0; i < rows and found == rows; ++i) {
            if (std::equal(
                    data.begin() + static_cast<int64_t>(i * dim),
                    data.begin() + static_cast<int64_t>((i + 1) * dim),
                    centroids.begin() + static_cast<int64_t>(static_cast<uint64_t>(c) * dim))) {
                found = i;
            }
        }
        REQUIRE(found < rows);
        chosen.push_back(found);
    }
    // The first draw is the caller's first uniform, as it is on the host.
    REQUIRE(chosen[0] ==
            static_cast<uint64_t>(static_cast<double>(uniforms[0]) * static_cast<double>(rows)));
    std::sort(chosen.begin(), chosen.end());
    REQUIRE(std::unique(chosen.begin(), chosen.end()) == chosen.end());
}

TEST_CASE("Accumulation and seeding cross the stride they are launched with", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // Both of those kernels are launched one warp wide and walk the dimension in
    // strides of 32. The cases above use eight dimensions, so the stride never
    // iterates and only eight of the thirty-two lanes ever hold data. At 128 it
    // runs four times with every lane loaded, and for seeding the shuffle has to
    // combine four partial sums per lane rather than one value.
    //
    // What a cluster is lives in dimensions 64 and up, past the first two strides,
    // so a stride that stops early cannot tell the clusters apart.
    const int32_t dim = 128;
    const uint32_t k = 4;
    const uint64_t per_cluster = 150;
    const uint64_t rows = k * per_cluster;

    std::vector<float> data(rows * static_cast<uint64_t>(dim), 0.0F);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> jitter(-0.5F, 0.5F);
    for (uint64_t i = 0; i < rows; ++i) {
        const uint64_t cluster = i / per_cluster;
        for (int32_t j = 0; j < dim; ++j) {
            const bool identifies = j >= 64 and static_cast<uint64_t>(j - 64) / 16 == cluster;
            data[i * dim + j] = (identifies ? 1000.0F : 0.0F) + jitter(gen);
        }
    }
    const uint64_t budget = CudaSuggestedBudget(kChunkedBudgetCap);

    // Accumulation: a dimension left out of the stride shows up directly in the sums.
    std::vector<int32_t> labels(rows);
    for (uint64_t i = 0; i < rows; ++i) {
        labels[i] = static_cast<int32_t>(i / per_cluster);
    }
    std::vector<float> sums(static_cast<uint64_t>(k) * dim, 7.0F);
    std::vector<int32_t> counts(k, 5);
    REQUIRE(CudaAccumulateCentroids(data.data(),
                                    rows,
                                    dim,
                                    labels.data(),
                                    k,
                                    sums.data(),
                                    counts.data(),
                                    budget,
                                    kForceOffload));
    std::vector<double> want(static_cast<uint64_t>(k) * dim, 0.0);
    for (uint64_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < dim; ++j) {
            want[static_cast<uint64_t>(labels[i]) * dim + j] += data[i * dim + j];
        }
    }
    REQUIRE(counts == std::vector<int32_t>(k, static_cast<int32_t>(per_cluster)));
    for (uint64_t i = 0; i < want.size(); ++i) {
        REQUIRE(std::abs(static_cast<double>(sums[i]) - want[i]) <=
                1e-3 * (1.0 + std::abs(want[i])));
    }

    // Seeding: with the clusters this far apart every draw after the first lands in
    // one not yet covered, so the four seeds come from four different clusters. A
    // truncated stride sees no separation and cannot produce that.
    std::vector<float> uniforms(2 * static_cast<uint64_t>(k));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    for (auto& u : uniforms) {
        u = unit(gen);
    }
    std::vector<float> seeds(static_cast<uint64_t>(k) * dim, -7.0F);
    REQUIRE(CudaKMeansPlusPlusInit(data.data(),
                                   rows,
                                   dim,
                                   k,
                                   uniforms.data(),
                                   seeds.data(),
                                   CudaSuggestedBudget(0),
                                   kForceOffload));
    std::vector<uint64_t> seed_clusters;
    for (uint32_t c = 0; c < k; ++c) {
        uint64_t found = rows;
        for (uint64_t i = 0; i < rows and found == rows; ++i) {
            if (std::equal(data.begin() + static_cast<int64_t>(i * dim),
                           data.begin() + static_cast<int64_t>((i + 1) * dim),
                           seeds.begin() + static_cast<int64_t>(static_cast<uint64_t>(c) * dim))) {
                found = i;
            }
        }
        REQUIRE(found < rows);
        seed_clusters.push_back(found / per_cluster);
    }
    std::sort(seed_clusters.begin(), seed_clusters.end());
    REQUIRE(std::unique(seed_clusters.begin(), seed_clusters.end()) == seed_clusters.end());
}

TEST_CASE("Seeding refuses a dataset that cannot stay resident", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 8;
    const uint64_t rows = 4096;
    const uint32_t k = 24;
    const auto data = RandomRows(rows, dim, 61);
    const std::vector<float> uniforms(2 * static_cast<uint64_t>(k), 0.5F);

    // Seeding reads the whole dataset once per centroid, so it keeps it on the
    // device for the pass and refuses rather than streaming it back k times.
    std::vector<float> centroids(static_cast<uint64_t>(k) * dim, -7.0F);
    REQUIRE_FALSE(CudaKMeansPlusPlusInit(
        data.data(), rows, dim, k, uniforms.data(), centroids.data(), 1024, kForceOffload));
    REQUIRE(std::all_of(centroids.begin(), centroids.end(), [](float v) { return v == -7.0F; }));
}

TEST_CASE("A budget too small for a chunk keeps the pass on the CPU", "[ut][gpu_pass]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 32;
    const uint64_t rows = 4000;
    const uint64_t k = 64;
    const auto query = RandomRows(rows, dim, 71);
    const auto centroids = RandomRows(k, dim, 72);

    std::vector<int32_t> labels(rows, -7);
    REQUIRE_FALSE(CudaAssignNearest(
        query.data(), rows, centroids.data(), k, dim, labels.data(), nullptr, 4096, kForceOffload));
    REQUIRE(std::all_of(labels.begin(), labels.end(), [](int32_t v) { return v == -7; }));

    std::vector<float> sums(k * static_cast<uint64_t>(dim), -7.0F);
    std::vector<int32_t> counts(k, -7);
    REQUIRE_FALSE(CudaAccumulateCentroids(query.data(),
                                          rows,
                                          dim,
                                          labels.data(),
                                          static_cast<uint32_t>(k),
                                          sums.data(),
                                          counts.data(),
                                          256,
                                          kForceOffload));
    REQUIRE(std::all_of(sums.begin(), sums.end(), [](float v) { return v == -7.0F; }));
    REQUIRE(std::all_of(counts.begin(), counts.end(), [](int32_t v) { return v == -7; }));
}

#endif  // VSAG_ENABLE_CUDA
