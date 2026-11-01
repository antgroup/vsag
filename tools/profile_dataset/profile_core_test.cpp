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

// Unit tests for the numeric building blocks of profile_dataset.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <vector>

#include "deterministic_random.h"
#include "hdf5_source.h"
#include "kmeans.h"
#include "linear_algebra.h"
#include "neighbor_search.h"
#include "quantization_simulation.h"
#include "sha256.h"
#include "statistics.h"
#include "vector_math.h"

using namespace vsag::dataset_profile;
using Catch::Approx;

namespace {

std::vector<float>
random_matrix(uint64_t rows, uint64_t dim, uint64_t seed) {
    DeterministicRng rng(seed);
    std::vector<float> data(rows * dim);
    for (auto& value : data) {
        value = static_cast<float>(rng.UniformDouble() * 2.0 - 1.0);
    }
    return data;
}

}  // namespace

TEST_CASE("DeterministicRng matches an independent xoshiro256** reference", "[profile][random]") {
    // Golden values come from a separate Python implementation of SplitMix64 + xoshiro256**.
    DeterministicRng rng(42);
    CHECK(rng.Next() == 0x15780b2e0c2ec716ULL);
    CHECK(rng.Next() == 0x6104d9866d113a7eULL);
    CHECK(rng.Next() == 0xae17533239e499a1ULL);
    DeterministicRng bounded(42);
    const std::vector<uint64_t> expected = {742, 102, 9, 193, 476};
    for (const uint64_t value : expected) {
        CHECK(bounded.Uniform(1000) == value);
    }
    CHECK(derive_seed(42, "sample/train") == 0xee8f5666e1fae8c5ULL);
    CHECK(derive_seed(42, "sample/train") != derive_seed(42, "sample/test"));
    CHECK(derive_seed(42, "pairs/train") != derive_seed(43, "pairs/train"));
    REQUIRE_THROWS_AS(rng.Uniform(0), std::invalid_argument);
}

TEST_CASE("Uniform draws are bounded and roughly uniform", "[profile][random]") {
    DeterministicRng rng(7);
    std::vector<uint64_t> counts(10, 0);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t value = rng.Uniform(10);
        REQUIRE(value < 10);
        ++counts[value];
    }
    for (const auto count : counts) {
        CHECK(count > 9500);
        CHECK(count < 10500);
    }
    for (int i = 0; i < 1000; ++i) {
        const double value = rng.UniformDouble();
        REQUIRE(value >= 0.0);
        REQUIRE(value < 1.0);
    }
}

TEST_CASE("Floyd sampling is sorted, unique, bounded and reproducible", "[profile][random]") {
    DeterministicRng a(11);
    DeterministicRng b(11);
    DeterministicRng c(12);
    const auto first = sample_without_replacement(1000000, 5000, a);
    const auto second = sample_without_replacement(1000000, 5000, b);
    const auto other = sample_without_replacement(1000000, 5000, c);
    REQUIRE(first.size() == 5000);
    CHECK(first == second);
    CHECK(first != other);
    CHECK(std::is_sorted(first.begin(), first.end()));
    CHECK(std::set<uint64_t>(first.begin(), first.end()).size() == first.size());
    CHECK(first.back() < 1000000);
    const auto all = sample_without_replacement(10, 50, a);
    CHECK(all == std::vector<uint64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    std::vector<uint64_t> values(100);
    std::iota(values.begin(), values.end(), 0);
    shuffle_in_place(values, a);
    std::vector<uint64_t> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    CHECK(sorted[0] == 0);
    CHECK(sorted[99] == 99);
    CHECK(values != sorted);
}

TEST_CASE("Moments match direct formulas and merge exactly across partitions",
          "[profile][statistics]") {
    const std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, 13.0, -1.5};
    double mean = 0.0;
    for (const double x : data) {
        mean += x;
    }
    mean /= static_cast<double>(data.size());
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
    for (const double x : data) {
        m2 += std::pow(x - mean, 2);
        m3 += std::pow(x - mean, 3);
        m4 += std::pow(x - mean, 4);
    }
    const auto n = static_cast<double>(data.size());
    Moments sequential;
    for (const double x : data) {
        sequential.Add(x);
    }
    CHECK(sequential.mean == Approx(mean).epsilon(1e-12));
    CHECK(sequential.Variance() == Approx(m2 / n).epsilon(1e-12));
    CHECK(sequential.Skewness() == Approx(std::sqrt(n) * m3 / std::pow(m2, 1.5)).epsilon(1e-10));
    CHECK(sequential.ExcessKurtosis() == Approx(n * m4 / (m2 * m2) - 3.0).epsilon(1e-10));
    for (uint64_t split = 0; split <= data.size(); ++split) {
        Moments left;
        Moments right;
        for (uint64_t i = 0; i < data.size(); ++i) {
            (i < split ? left : right).Add(data[i]);
        }
        left.Merge(right);
        CHECK(left.count == data.size());
        CHECK(left.mean == Approx(mean).epsilon(1e-12));
        CHECK(left.m2 == Approx(m2).epsilon(1e-12));
        CHECK(left.m3 == Approx(m3).epsilon(1e-10));
        CHECK(left.m4 == Approx(m4).epsilon(1e-10));
    }
    Moments constant;
    for (int i = 0; i < 100; ++i) {
        constant.Add(0.1);
    }
    CHECK(constant.Skewness() == 0.0);
    CHECK(constant.ExcessKurtosis() == 0.0);
}

TEST_CASE("Histogram bins, out-of-range counts and degenerate ranges", "[profile][statistics]") {
    Histogram histogram(0.0, 10.0, 5);
    for (const double value : {-1.0, 0.0, 1.9, 2.0, 9.99, 10.0, 11.0}) {
        histogram.Add(value);
    }
    const JsonType json = histogram.ToJson();
    CHECK(json["counts"] == JsonType::array({2, 1, 0, 0, 2}));
    CHECK(json["below"] == 1);
    CHECK(json["above"] == 1);
    CHECK(histogram.Total() == 7);
    Histogram constant(3.0, 3.0, 50);
    constant.Add(3.0);
    CHECK(constant.Counts().size() == 1);
    CHECK(constant.Counts()[0] == 1);
    Histogram other(0.0, 10.0, 4);
    REQUIRE_THROWS(histogram.Merge(other));
}

TEST_CASE("Quantiles, summaries and distribution distances", "[profile][statistics]") {
    const std::vector<double> sorted = {1.0, 2.0, 3.0, 4.0};
    CHECK(quantile_sorted(sorted, 0.25) == Approx(1.75));
    CHECK(quantile_sorted(sorted, 0.5) == Approx(2.5));
    CHECK(quantile_sorted(sorted, 1.0) == Approx(4.0));
    std::vector<double> values = {3.0,
                                  std::numeric_limits<double>::quiet_NaN(),
                                  1.0,
                                  std::numeric_limits<double>::infinity(),
                                  2.0};
    const JsonType summary = summarize_values(values);
    CHECK(summary["count"] == 3);
    CHECK(summary["min"] == 1.0);
    CHECK(summary["max"] == 3.0);
    CHECK(summary["p50"] == 2.0);
    std::vector<double> empty;
    CHECK(summarize_values(empty)["count"] == 0);
    CHECK(ks_statistic({1, 2, 3}, {1, 2, 3}) == 0.0);
    CHECK(ks_statistic({1, 2, 3}, {10, 11}) == 1.0);
    CHECK(gini_sorted({1, 1, 1, 1}) == Approx(0.0));
    CHECK(gini_sorted({0, 0, 0, 4}) == Approx(0.75));
    CHECK(round_significant(123456.789, 3) == Approx(123000.0));
    CHECK(round_significant(0.000123456, 2) == Approx(0.00012));
    CHECK(json_number(std::numeric_limits<double>::quiet_NaN()).is_null());
    const JsonType integers = integer_histogram({0, 1, 1, 2, 5}, 50);
    CHECK(integers["counts"].size() == 6);
    CHECK(integers["counts"][1] == 2);
}

TEST_CASE("Robust histogram ranges clip only heavy tails", "[profile][statistics]") {
    std::vector<double> bulk;
    bulk.reserve(1001);
    for (int i = 0; i < 1000; ++i) {
        bulk.push_back(i / 1000.0);
    }
    const JsonType plain = histogram_of_sorted(bulk, 10);
    CHECK(plain["clipped"] == false);
    CHECK(plain["below"] == 0);
    bulk.push_back(1e6);
    const JsonType clipped = histogram_of_sorted(bulk, 10);
    CHECK(clipped["clipped"] == true);
    CHECK(clipped["above"].get<uint64_t>() >= 1);
}

TEST_CASE("SHA-256 matches FIPS 180-4 test vectors", "[profile][sha256]") {
    CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sha256_hex(std::string(1000, 'a')) ==
          "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST_CASE("Distance kernels follow VSAG conventions", "[profile][math]") {
    const auto a = random_matrix(1, 37, 1);
    const auto b = random_matrix(1, 37, 2);
    double l2 = 0.0;
    double ip = 0.0;
    for (uint64_t d = 0; d < 37; ++d) {
        l2 += (static_cast<double>(a[d]) - b[d]) * (static_cast<double>(a[d]) - b[d]);
        ip += static_cast<double>(a[d]) * b[d];
    }
    CHECK(l2_sqr(a.data(), b.data(), 37) == Approx(l2).epsilon(1e-12));
    CHECK(inner_product(a.data(), b.data(), 37) == Approx(ip).epsilon(1e-12));
    double fused_ip = 0.0;
    double fused_l2 = 0.0;
    inner_product_and_l2_sqr(a.data(), b.data(), 37, fused_ip, fused_l2);
    CHECK(fused_ip == Approx(ip).epsilon(1e-12));
    CHECK(fused_l2 == Approx(l2).epsilon(1e-12));
    const double na = std::sqrt(squared_norm(a.data(), 37));
    const double nb = std::sqrt(squared_norm(b.data(), 37));
    CHECK(metric_distance(Metric::L2, a.data(), b.data(), 37, na, nb) == Approx(l2));
    CHECK(metric_distance(Metric::IP, a.data(), b.data(), 37, na, nb) == Approx(1.0 - ip));
    const double cosine = ip / (na * nb);
    CHECK(metric_distance(Metric::COSINE, a.data(), b.data(), 37, na, nb) == Approx(1.0 - cosine));
    CHECK(geometric_distance(Metric::COSINE, a.data(), b.data(), 37, na, nb) ==
          Approx(std::sqrt(2.0 * (1.0 - cosine))));
    double geometric = 0.0;
    CHECK(geometric_from_metric(Metric::L2, 9.0, geometric));
    CHECK(geometric == 3.0);
    CHECK_FALSE(geometric_from_metric(Metric::IP, 0.5, geometric));
}

TEST_CASE("IEEE classification and half-precision rounding", "[profile][math]") {
    CHECK(is_finite_value(1.0F));
    CHECK_FALSE(is_finite_value(std::numeric_limits<float>::infinity()));
    CHECK_FALSE(is_finite_value(std::numeric_limits<float>::quiet_NaN()));
    CHECK(is_nan_value(std::numeric_limits<float>::quiet_NaN()));
    CHECK_FALSE(is_nan_value(-std::numeric_limits<float>::infinity()));
    CHECK(canonical_bits(-0.0F) == canonical_bits(0.0F));
    CHECK(canonical_bits(std::numeric_limits<float>::quiet_NaN()) ==
          canonical_bits(-std::numeric_limits<float>::quiet_NaN()));
    CHECK(round_through_fp16(1.0F) == 1.0F);
    CHECK(round_through_fp16(65504.0F) == 65504.0F);
    CHECK(std::isinf(round_through_fp16(65520.0F)));
    CHECK(round_through_fp16(0.1F) == 0.0999755859375F);
    CHECK(round_through_fp16(1e-8F) == 0.0F);
    CHECK(round_through_fp16(-2.0F) == -2.0F);
    CHECK(round_through_bf16(1.0F) == 1.0F);
    CHECK(round_through_bf16(3.14159F) == 3.140625F);
}

TEST_CASE("TopK keeps the smallest neighbors independent of insertion order", "[profile][knn]") {
    std::vector<std::pair<double, uint64_t>> items;
    items.reserve(200);
    for (uint64_t i = 0; i < 200; ++i) {
        items.emplace_back(static_cast<double>((i * 37) % 50), i);
    }
    TopK forward(10);
    for (const auto& [distance, index] : items) {
        forward.Push(distance, index);
    }
    TopK backward(10);
    for (auto& item : std::ranges::reverse_view(items)) {
        backward.Push(item.first, item.second);
    }
    const auto a = forward.Sorted();
    const auto b = backward.Sorted();
    REQUIRE(a.size() == 10);
    for (uint64_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].distance == b[i].distance);
        CHECK(a[i].index == b[i].index);
    }
    CHECK(a.front().distance == 0.0);
    CHECK(a.front().index == 0);
    TopK merged(10);
    TopK left(10);
    TopK right(10);
    for (uint64_t i = 0; i < items.size(); ++i) {
        (i % 2 == 0 ? left : right).Push(items[i].first, items[i].second);
    }
    merged.Merge(left);
    merged.Merge(right);
    const auto c = merged.Sorted();
    for (uint64_t i = 0; i < a.size(); ++i) {
        CHECK(c[i].index == a[i].index);
    }
}

TEST_CASE("Dense k-NN search is exact for every metric and any block split", "[profile][knn]") {
    const uint64_t rows = 600;
    const uint64_t dim = 13;
    const auto base = random_matrix(rows, dim, 21);
    std::vector<double> norms(rows);
    for (uint64_t r = 0; r < rows; ++r) {
        norms[r] = std::sqrt(squared_norm(base.data() + r * dim, dim));
    }
    std::vector<uint8_t> valid(rows, 1);
    valid[5] = 0;
    for (const Metric metric : {Metric::L2, Metric::IP, Metric::COSINE}) {
        DenseQuerySet queries;
        queries.dim = dim;
        for (const uint64_t row : {0ULL, 17ULL, 300ULL}) {
            queries.vectors.insert(queries.vectors.end(),
                                   base.begin() + static_cast<int64_t>(row * dim),
                                   base.begin() + static_cast<int64_t>((row + 1) * dim));
            queries.norms.push_back(norms[row]);
            queries.self_rows.push_back(row);
        }
        DenseKnnSearch whole(queries, metric, 5);
        whole.Scan(base.data(), norms.data(), valid.data(), 0, rows);
        DenseKnnSearch blocks(queries, metric, 5);
        for (uint64_t begin = 0; begin < rows; begin += 97) {
            const uint64_t count = std::min<uint64_t>(97, rows - begin);
            blocks.Scan(base.data() + begin * dim,
                        norms.data() + begin,
                        valid.data() + begin,
                        begin,
                        count);
        }
        const auto a = whole.MetricNeighbors();
        const auto b = blocks.MetricNeighbors();
        for (uint64_t q = 0; q < queries.Size(); ++q) {
            std::vector<std::pair<double, uint64_t>> brute;
            for (uint64_t r = 0; r < rows; ++r) {
                if (r == queries.self_rows[q] || valid[r] == 0) {
                    continue;
                }
                brute.emplace_back(metric_distance(metric,
                                                   queries.vectors.data() + q * dim,
                                                   base.data() + r * dim,
                                                   dim,
                                                   queries.norms[q],
                                                   norms[r]),
                                   r);
            }
            std::sort(brute.begin(), brute.end());
            REQUIRE(a[q].size() == 5);
            for (uint64_t j = 0; j < 5; ++j) {
                CHECK(a[q][j].index == brute[j].second);
                CHECK(a[q][j].distance == Approx(brute[j].first).epsilon(1e-12));
                CHECK(b[q][j].index == a[q][j].index);
                CHECK(b[q][j].distance == a[q][j].distance);
            }
        }
        const auto geometric = whole.GeometricDistances();
        CHECK(std::is_sorted(geometric[0].begin(), geometric[0].end()));
    }
}

TEST_CASE("Sparse k-NN search matches brute force and pads with zero-overlap distance",
          "[profile][knn]") {
    SparseQuerySet base;
    DeterministicRng rng(5);
    const uint64_t rows = 300;
    for (uint64_t r = 0; r < rows; ++r) {
        std::set<uint32_t> ids;
        while (ids.size() < 4) {
            ids.insert(static_cast<uint32_t>(rng.Uniform(60)));
        }
        double norm = 0.0;
        for (const uint32_t id : ids) {
            base.ids.push_back(id);
            const auto value = static_cast<float>(rng.UniformDouble());
            base.values.push_back(value);
            norm += static_cast<double>(value) * value;
        }
        base.offsets.push_back(base.ids.size());
        base.norms.push_back(std::sqrt(norm));
        base.self_rows.push_back(r);
    }
    SparseQuerySet queries;
    queries.ids = {base.ids.begin(), base.ids.begin() + static_cast<int64_t>(base.offsets[2])};
    queries.values = {base.values.begin(),
                      base.values.begin() + static_cast<int64_t>(base.offsets[2])};
    queries.offsets = {0, base.offsets[1], base.offsets[2]};
    queries.norms = {base.norms[0], base.norms[1]};
    queries.self_rows = {0, 1};
    queries.ids.push_back(999);  // a third query whose only term occurs nowhere
    queries.values.push_back(1.0F);
    queries.offsets.push_back(queries.ids.size());
    queries.norms.push_back(1.0);
    queries.self_rows.push_back(NO_ROW);
    SparseBatch batch;
    batch.Clear();
    batch.ids = base.ids;
    batch.values = base.values;
    batch.offsets = base.offsets;
    for (const Metric metric : {Metric::IP, Metric::COSINE}) {
        SparseKnnSearch search(queries, metric, 6);
        search.Scan(batch, base.norms);
        const auto neighbors = search.MetricNeighbors();
        for (uint64_t q = 0; q < 2; ++q) {
            std::vector<std::pair<double, uint64_t>> brute;
            for (uint64_t r = 0; r < rows; ++r) {
                if (r == q) {
                    continue;
                }
                const double dot = sparse_dot(queries.ids.data() + queries.offsets[q],
                                              queries.values.data() + queries.offsets[q],
                                              queries.offsets[q + 1] - queries.offsets[q],
                                              base.ids.data() + base.offsets[r],
                                              base.values.data() + base.offsets[r],
                                              base.offsets[r + 1] - base.offsets[r]);
                if (dot != 0.0) {
                    brute.emplace_back(metric == Metric::IP
                                           ? 1.0 - dot
                                           : 1.0 - dot / (queries.norms[q] * base.norms[r]),
                                       r);
                }
            }
            std::sort(brute.begin(), brute.end());
            for (uint64_t j = 0; j < 6 && j < brute.size(); ++j) {
                CHECK(neighbors[q][j].index == brute[j].second);
                CHECK(neighbors[q][j].distance == Approx(brute[j].first).epsilon(1e-9));
            }
        }
        REQUIRE(neighbors[2].size() == 6);
        CHECK(neighbors[2][0].distance == 1.0);
        CHECK(neighbors[2][0].index == NO_ROW);
        CHECK(search.QueriesWithFewOverlaps() >= 1);
    }
}

TEST_CASE("Intrinsic dimensionality estimators recover the dimension of a subspace",
          "[profile][knn]") {
    // 4000 points uniform in a 3-dimensional cube embedded in 20 dimensions.
    const uint64_t rows = 4000;
    const uint64_t dim = 20;
    DeterministicRng rng(9);
    std::vector<float> data(rows * dim, 0.0F);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t d = 0; d < 3; ++d) {
            data[r * dim + d * 5] = static_cast<float>(rng.UniformDouble());
        }
    }
    DenseQuerySet queries;
    queries.dim = dim;
    for (uint64_t q = 0; q < 300; ++q) {
        queries.vectors.insert(queries.vectors.end(),
                               data.begin() + static_cast<int64_t>(q * dim),
                               data.begin() + static_cast<int64_t>((q + 1) * dim));
        queries.norms.push_back(std::sqrt(squared_norm(data.data() + q * dim, dim)));
        queries.self_rows.push_back(q);
    }
    std::vector<double> norms(rows);
    for (uint64_t r = 0; r < rows; ++r) {
        norms[r] = std::sqrt(squared_norm(data.data() + r * dim, dim));
    }
    std::vector<uint8_t> valid(rows, 1);
    DenseKnnSearch search(queries, Metric::L2, 20);
    search.Scan(data.data(), norms.data(), valid.data(), 0, rows);
    std::vector<double> lids;
    std::vector<double> ratios;
    for (const auto& distances : search.GeometricDistances()) {
        lids.push_back(lid_mle(distances));
        ratios.push_back(distances[1] / distances[0]);
    }
    std::sort(lids.begin(), lids.end());
    CHECK(quantile_sorted(lids, 0.5) > 2.2);
    CHECK(quantile_sorted(lids, 0.5) < 4.0);
    const double two_nn = two_nn_dimension(ratios);
    CHECK(two_nn > 2.2);
    CHECK(two_nn < 4.0);
    CHECK(std::isnan(lid_mle({0.0, 0.0, 1.0})));
    CHECK(std::isnan(lid_mle({2.0, 2.0, 2.0})));
}

TEST_CASE("Symmetric eigen-decomposition is accurate and orthonormal", "[profile][linalg]") {
    std::vector<double> values;
    std::vector<double> vectors;
    symmetric_eigen({2.0, 1.0, 1.0, 2.0}, 2, values, vectors);
    CHECK(values[0] == Approx(1.0));
    CHECK(values[1] == Approx(3.0));
    const uint64_t n = 24;
    const auto random = random_matrix(n, n, 31);
    std::vector<double> matrix(n * n);
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            matrix[i * n + j] = static_cast<double>(random[i * n + j]) + random[j * n + i];
        }
    }
    symmetric_eigen(matrix, n, values, vectors);
    double trace = 0.0;
    double eigen_sum = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        trace += matrix[i * n + i];
        eigen_sum += values[i];
    }
    CHECK(eigen_sum == Approx(trace).margin(1e-9));
    CHECK(std::is_sorted(values.begin(), values.end()));
    for (uint64_t j = 0; j < n; ++j) {
        for (uint64_t i = 0; i < n; ++i) {
            double av = 0.0;
            for (uint64_t k = 0; k < n; ++k) {
                av += matrix[i * n + k] * vectors[k * n + j];
            }
            CHECK(av == Approx(values[j] * vectors[i * n + j]).margin(1e-9));
        }
        for (uint64_t other = 0; other < n; ++other) {
            double dot = 0.0;
            for (uint64_t k = 0; k < n; ++k) {
                dot += vectors[k * n + j] * vectors[k * n + other];
            }
            CHECK(dot == Approx(j == other ? 1.0 : 0.0).margin(1e-9));
        }
    }
}

TEST_CASE("PCA finds a planar subspace with a reproducible orientation", "[profile][linalg]") {
    const uint64_t rows = 500;
    const uint64_t dim = 10;
    DeterministicRng rng(3);
    std::vector<float> data(rows * dim);
    for (uint64_t r = 0; r < rows; ++r) {
        const double s = rng.UniformDouble() * 10.0;
        const double t = rng.UniformDouble();
        for (uint64_t d = 0; d < dim; ++d) {
            data[r * dim + d] =
                static_cast<float>(s * (d % 2 == 0 ? 1.0 : 0.0) + t * (d == 3 ? 1.0 : 0.0) + 1.0);
        }
    }
    const PcaResult pca = compute_pca(data, rows, dim);
    REQUIRE(pca.eigenvalues.size() == dim);
    CHECK(pca.eigenvalues[0] > 10.0);
    CHECK(pca.eigenvalues[1] > 0.01);
    CHECK(pca.eigenvalues[2] < 1e-6);
    double x = 0.0;
    double y = 0.0;
    project_2d(pca, data.data(), x, y);
    const PcaResult again = compute_pca(data, rows, dim);
    CHECK(again.axis1 == pca.axis1);
    CHECK(again.axis2 == pca.axis2);
}

TEST_CASE("k-means separates clusters reproducibly", "[profile][kmeans]") {
    const uint64_t rows = 900;
    const uint64_t dim = 4;
    DeterministicRng rng(13);
    std::vector<float> data(rows * dim);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t d = 0; d < dim; ++d) {
            data[r * dim + d] =
                static_cast<float>(static_cast<double>(r % 3) * 20.0 + rng.UniformDouble());
        }
    }
    const KMeansResult a = run_kmeans(data, rows, dim, 3, 50, 99);
    const KMeansResult b = run_kmeans(data, rows, dim, 3, 50, 99);
    CHECK(a.assignments == b.assignments);
    CHECK(a.converged);
    std::vector<uint64_t> sizes = a.sizes;
    std::sort(sizes.begin(), sizes.end());
    CHECK(sizes == std::vector<uint64_t>({300, 300, 300}));
    CHECK(silhouette_score(data, rows, dim, a.assignments, a.k) > 0.9);
    const KMeansResult capped = run_kmeans(data, 2, dim, 5, 10, 1);
    CHECK(capped.k == 2);
}

TEST_CASE("Quantization simulation mirrors VSAG range rules", "[profile][quantization]") {
    std::vector<float> column(1000);
    for (uint64_t i = 0; i < column.size(); ++i) {
        column[i] = static_cast<float>(i);
    }
    std::vector<float> lower;
    std::vector<float> upper;
    // ignore_count = max(1, floor(999 * 0.05)) = 49: the 49th smallest and largest values.
    truncated_bounds(column, 1000, 1, 0.05, lower, upper);
    CHECK(lower[0] == 48.0F);
    CHECK(upper[0] == 951.0F);
    truncated_bounds(column, 1000, 1, 0.001, lower, upper);
    CHECK(lower[0] == 0.0F);
    CHECK(upper[0] == 999.0F);
    const auto base = random_matrix(800, 16, 41);
    const auto queries = random_matrix(40, 16, 42);
    const JsonType result = simulate_quantization(base, 800, queries, 40, 16, Metric::L2, 10);
    CHECK(result["schemes"]["fp16"]["recall_at_k"].get<double>() > 0.97);
    CHECK(result["schemes"]["sq8"]["recall_at_k"].get<double>() > 0.85);
    CHECK(result["schemes"]["sq4"]["mean_relative_reconstruction_error"].get<double>() >
          result["schemes"]["sq8"]["mean_relative_reconstruction_error"].get<double>());
    CHECK(result["schemes"]["sq8_uniform"]["encodes_query"] == true);
    CHECK(result["schemes"]["sq8"]["code_bytes_per_vector"] == 16.0);
}
