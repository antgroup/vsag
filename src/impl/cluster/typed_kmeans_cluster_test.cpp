// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <atomic>
#include <catch2/matchers/catch_matchers.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "kmeans_cluster.h"
#include "typed_kmeans_input.h"
#include "unittest.h"

namespace {
// Independent arithmetic reference, deliberately not the production decoder.
float
ReferenceHalf(uint16_t bits, vsag::DataTypes type) {
    if (type == vsag::DataTypes::DATA_TYPE_BF16) {
        const uint32_t expanded = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &expanded, sizeof(value));
        return value;
    }
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    float value = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -24)
                                : std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
    return bits & 0x8000U ? -value : value;
}
}  // namespace

TEST_CASE("Typed kmeans matches independently rounded FP32", "[ut][TypedKMeans]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    // A single worker also pins the original assignment error summation order.
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(1), true);
    for (const auto type : {vsag::DataTypes::DATA_TYPE_FP16, vsag::DataTypes::DATA_TYPE_BF16}) {
        for (const uint64_t count : {uint64_t{41}, uint64_t{65539}}) {
            constexpr int32_t dim = 3;
            constexpr uint32_t k = 4;
            constexpr uint64_t stride = 11;  // Deliberately unaligned, padded rows.
            std::vector<uint8_t> source(count * stride + 1, 0xCC);
            std::vector<float> reference(count * dim);
            for (uint64_t i = 0; i < count; ++i) {
                for (int32_t j = 0; j < dim; ++j) {
                    // FP16/BF16 raw patterns with nonzero rounded mantissas, both signs.
                    uint16_t bits = static_cast<uint16_t>(
                        (type == vsag::DataTypes::DATA_TYPE_FP16 ? 0x3000 : 0x3E00) +
                        ((i * 131 + j * 53) % 1700));
                    if ((i + j) % 3 == 0) {
                        bits |= 0x8000U;
                    }
                    std::memcpy(source.data() + 1 + i * stride + j * 2, &bits, 2);
                    reference[i * dim + j] = ReferenceHalf(bits, type);
                }
            }
            for (const auto init :
                 {vsag::KMeansInitMethod::RANDOM, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS}) {
                for (const bool convergence : {false, true}) {
                    vsag::KMeansCluster typed(dim, allocator.get(), pool);
                    vsag::KMeansCluster fp32(dim, allocator.get(), pool);
                    double actual_error = 0;
                    double expected_error = 0;
                    auto expected = fp32.Run(k,
                                             reference.data(),
                                             count,
                                             3,
                                             &expected_error,
                                             convergence,
                                             1e-6F,
                                             init,
                                             777U,
                                             true);
                    auto actual = typed.Run(k,
                                            source.data() + 1,
                                            count,
                                            type,
                                            stride,
                                            source.size() - 1,
                                            3,
                                            &actual_error,
                                            convergence,
                                            1e-6F,
                                            init,
                                            777U,
                                            true);
                    REQUIRE(actual == expected);
                    REQUIRE(actual_error == expected_error);
                    REQUIRE(std::memcmp(typed.k_centroids_,
                                        fp32.k_centroids_,
                                        k * dim * sizeof(float)) == 0);
                }
            }
        }
    }
}

TEST_CASE("Typed kmeans validates view before changing centroids", "[ut][TypedKMeans]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::KMeansCluster cluster(2, allocator.get());
    uint16_t data[4] = {0, 1, 0x3C00, 0xBC00};
    const auto type = vsag::DataTypes::DATA_TYPE_FP16;
    REQUIRE_THROWS(cluster.Run(1, nullptr, 2, type, 4, 8));
    REQUIRE_THROWS(cluster.Run(0, data, 2, type, 4, 8));
    REQUIRE_THROWS(cluster.Run(3, data, 2, type, 4, 8));
    REQUIRE_THROWS(cluster.Run(1, data, 0, type, 4, 8));
    REQUIRE_THROWS(cluster.Run(1, data, 2, type, 3, 8));
    REQUIRE_THROWS(cluster.Run(1, data, 2, type, 4, 7));
    REQUIRE_THROWS(cluster.Run(1, data, 2, vsag::DataTypes::DATA_TYPE_INT8, 4, 8));
    REQUIRE_THROWS(cluster.Run(1, data, 3, type, std::numeric_limits<uint64_t>::max(), 8));
    vsag::KMeansCluster invalid_dim(0, allocator.get());
    REQUIRE_THROWS(invalid_dim.Run(1, data, 2, type, 4, 8));
}

TEST_CASE("Typed kmeans preserves zero-weight initialization and empty clusters",
          "[ut][TypedKMeans]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(1), true);
    std::vector<uint16_t> source(57, 0x3C00);
    std::vector<float> reference(57, 1.0F);
    for (const int iterations : {0, 1, 4}) {
        vsag::KMeansCluster typed(3, allocator.get(), pool);
        vsag::KMeansCluster fp32(3, allocator.get(), pool);
        double actual_error = 0;
        double expected_error = 0;
        auto expected = fp32.Run(7,
                                 reference.data(),
                                 19,
                                 iterations,
                                 &expected_error,
                                 true,
                                 1e-6F,
                                 vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                 123U,
                                 true);
        auto actual = typed.Run(7,
                                source.data(),
                                19,
                                vsag::DataTypes::DATA_TYPE_FP16,
                                6,
                                source.size() * sizeof(uint16_t),
                                iterations,
                                &actual_error,
                                true,
                                1e-6F,
                                vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                123U,
                                true);
        REQUIRE(actual == expected);
        REQUIRE(actual_error == expected_error);
        REQUIRE(std::memcmp(typed.k_centroids_, fp32.k_centroids_, 21 * sizeof(float)) == 0);
    }
}

TEST_CASE("Typed kmeans IEEE decoder edge patterns", "[ut][TypedKMeans]") {
    const uint16_t bits[] = {0, 0x8000, 1, 0x03FF, 0x0400, 0x7C00, 0xFC00, 0x7E01};
    vsag::TypedKMeansInput input(
        bits, 1, 8, vsag::DataTypes::DATA_TYPE_FP16, sizeof(bits), sizeof(bits));
    float decoded[8];
    input.Read(0, decoded);
    uint32_t actual[8];
    std::memcpy(actual, decoded, sizeof(actual));
    // Bit assertions remain meaningful under Release -ffast-math signed-zero assumptions.
    REQUIRE(actual[0] == 0x00000000U);
    REQUIRE(actual[1] == 0x80000000U);
    REQUIRE(actual[2] == 0x33800000U);
    REQUIRE(actual[3] == 0x387FC000U);
    REQUIRE(actual[4] == 0x38800000U);
    REQUIRE(actual[5] == 0x7F800000U);
    REQUIRE(actual[6] == 0xFF800000U);
    REQUIRE((actual[7] & 0x7F800000U) == 0x7F800000U);
    REQUIRE((actual[7] & 0x007FFFFFU) != 0);
}

TEST_CASE("Typed kmeans row provider identity normalize residual", "[ut][TypedKMeans]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(1), true);
    constexpr uint64_t count = 41;
    constexpr int32_t dim = 3;
    for (const auto type : {vsag::DataTypes::DATA_TYPE_FP16, vsag::DataTypes::DATA_TYPE_BF16}) {
        std::vector<uint16_t> bits(count * dim);
        std::vector<float> rounded(count * dim);
        for (uint64_t i = 0; i < bits.size(); ++i) {
            bits[i] = static_cast<uint16_t>(
                (type == vsag::DataTypes::DATA_TYPE_FP16 ? 0x3000 : 0x3E00) + i * 7);
            rounded[i] = ReferenceHalf(bits[i], type);
        }
        vsag::TypedKMeansInput raw(bits.data(), count, dim, type, dim * 2, bits.size() * 2);
        for (const int mode : {0, 1, 2}) {
            struct Context {
                const vsag::TypedKMeansInput* raw;
                int mode;
            } context{&raw, mode};
            auto reader = +[](const void* opaque, uint64_t row, float* out) {
                const auto& ctx = *static_cast<const Context*>(opaque);
                ctx.raw->Read(row, out);
                if (ctx.mode == 1) {
                    float norm = 0;
                    for (int j = 0; j < dim; ++j) {
                        norm += out[j] * out[j];
                    }
                    const float scale = 1.0F / std::sqrt(norm);
                    for (int j = 0; j < dim; ++j) {
                        out[j] *= scale;
                    }
                } else if (ctx.mode == 2) {
                    for (int j = 0; j < dim; ++j) {
                        out[j] -= 0.125F * (j + 1);
                    }
                }
            };
            vsag::TypedKMeansInput provider(count, dim, reader, &context);
            std::vector<float> expected_data = rounded;
            for (uint64_t i = 0; i < count; ++i) {
                float* row = expected_data.data() + i * dim;
                if (mode == 1) {
                    const float norm = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
                    const float scale = 1.0F / std::sqrt(norm);
                    for (int j = 0; j < dim; ++j) {
                        row[j] *= scale;
                    }
                } else if (mode == 2) {
                    for (int j = 0; j < dim; ++j) {
                        row[j] -= 0.125F * (j + 1);
                    }
                }
            }
            for (const auto init :
                 {vsag::KMeansInitMethod::RANDOM, vsag::KMeansInitMethod::KMEANS_PLUS_PLUS}) {
                vsag::KMeansCluster actual(dim, allocator.get(), pool);
                vsag::KMeansCluster reference(dim, allocator.get(), pool);
                double actual_error = 0;
                double expected_error = 0;
                auto labels =
                    actual.Run(4, provider, 3, &actual_error, false, 1e-6F, init, 777U, true);
                auto expected = reference.Run(4,
                                              expected_data.data(),
                                              count,
                                              3,
                                              &expected_error,
                                              false,
                                              1e-6F,
                                              init,
                                              777U,
                                              true);
                REQUIRE(labels == expected);
                REQUIRE(actual_error == expected_error);
                REQUIRE(std::memcmp(actual.k_centroids_,
                                    reference.k_centroids_,
                                    4 * dim * sizeof(float)) == 0);
            }
        }
    }
}

TEST_CASE("Typed kmeans helper bounds and CPU contract", "[ut][TypedKMeans]") {
    const uint16_t data[4] = {};
    float out[4];
    vsag::TypedKMeansInput input(data, 2, 2, vsag::DataTypes::DATA_TYPE_FP16, 4, 8);
    REQUIRE(input.Count() == 2);
    REQUIRE(input.Dim() == 2);
    REQUIRE_THROWS(input.Read(2, out));
    REQUIRE_THROWS(input.Read(0, nullptr));
    REQUIRE_THROWS(input.ReadRows(1, 2, out));
    REQUIRE_THROWS(input.ReadRows(std::numeric_limits<uint64_t>::max(), 2, out));
    REQUIRE_NOTHROW(input.ReadRows(2, 0, nullptr));
    REQUIRE_THROWS(vsag::TypedKMeansInput(2, 2, nullptr, nullptr));
    REQUIRE_THROWS(vsag::TypedKMeansInput(
        reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max() - 2),
        2,
        2,
        vsag::DataTypes::DATA_TYPE_FP16,
        4,
        8));
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(1), true);
    int32_t labels[3] = {};
    REQUIRE_THROWS(
        vsag::TypedNearestCentroidAssign(out, 1, input, 3, 2, pool, allocator.get(), labels));
    REQUIRE_THROWS(
        vsag::TypedNearestCentroidAssign(out, 1, input, 2, 1, pool, allocator.get(), labels));
    vsag::KMeansGpuConfig gpu;
    gpu.enabled = true;
    vsag::KMeansCluster cluster(2, allocator.get(), pool, gpu);
    REQUIRE_THROWS(cluster.Run(1, input));
    REQUIRE_THROWS(cluster.Run(1, data, 2, vsag::DataTypes::DATA_TYPE_FP16, 4, 8));
}

TEST_CASE("Typed provider task exceptions propagate", "[ut][TypedKMeans]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(2), true);
    struct Context {
        mutable std::atomic<uint64_t> calls{0};
    } context;
    auto reader = +[](const void* opaque, uint64_t, float* out) {
        const auto& ctx = *static_cast<const Context*>(opaque);
        // Random init reads once, assignment reads 41; fail inside update task.
        if (++ctx.calls > 42) {
            throw std::runtime_error("provider update failure");
        }
        out[0] = 1.0F;
        out[1] = 2.0F;
    };
    vsag::TypedKMeansInput provider(41, 2, reader, &context);
    vsag::KMeansCluster cluster(2, allocator.get(), pool);
    REQUIRE_THROWS_WITH(
        cluster.Run(
            1, provider, 1, nullptr, false, 1e-6F, vsag::KMeansInitMethod::RANDOM, 777U, true),
        "provider update failure");
}

TEST_CASE("Typed submission unwind joins borrowed tasks and preserves original error",
          "[ut][TypedKMeans][TypedKMeansUnwind]") {
    for (const bool inline_failure : {false, true}) {
        std::vector<std::future<void>> futures;
        std::promise<void> release;
        const auto ready = release.get_future().share();
        std::atomic<uint64_t> completed{0};
        // The tasks borrow this stack context; checking after the catch must see
        // both writes, even though the first task also fails during cleanup.
        uint64_t borrowed[2] = {};
        auto submit = [&](uint64_t i) -> std::future<void> {
            if (i == 2) {
                release.set_value();
                throw std::runtime_error("injected submission failure");
            }
            return std::async(std::launch::async, [&, i]() {
                ready.wait();
                borrowed[i] = i + 1;
                ++completed;
                if (i == 0) {
                    throw std::runtime_error("secondary task failure");
                }
            });
        };
        auto inline_task = [&]() {
            release.set_value();
            throw std::runtime_error("inline callable failure");
        };
        REQUIRE_THROWS_WITH(
            vsag::RunTypedKMeansTasks(futures,
                                      3,
                                      [&]() {
                                          for (uint64_t i = 0; i < 3; ++i) {
                                              if (inline_failure && i == 2) {
                                                  inline_task();
                                              } else {
                                                  futures.emplace_back(submit(i));
                                              }
                                          }
                                      }),
            inline_failure ? "inline callable failure" : "injected submission failure");
        REQUIRE(completed.load() == 2);
        REQUIRE(borrowed[0] == 1);
        REQUIRE(borrowed[1] == 2);
        REQUIRE(futures.empty());
    }
}

TEST_CASE("Typed task drain continues after future errors",
          "[ut][TypedKMeans][TypedKMeansUnwind]") {
    std::vector<std::future<void>> futures;
    uint64_t completed = 0;
    // An invalid future must report no_state without invoking get() or skipping later tasks.
    futures.emplace_back();
    futures.emplace_back(std::async(std::launch::deferred, [&]() {
        ++completed;
        throw std::runtime_error("secondary callback failure");
    }));
    futures.emplace_back(std::async(std::launch::deferred, [&]() { ++completed; }));
    bool caught_no_state = false;
    try {
        vsag::WaitTypedKMeansTasks(futures);
    } catch (const std::future_error& e) {
        caught_no_state = (e.code() == std::make_error_code(std::future_errc::no_state));
    }
    REQUIRE(caught_no_state);
    REQUIRE(completed == 2);
    REQUIRE(futures.empty());

    // A later invalid future must not overwrite an earlier failure or skip remaining tasks.
    futures.emplace_back(std::async(std::launch::deferred, [&]() {
        ++completed;
        throw std::runtime_error("primary callback failure");
    }));
    futures.emplace_back();
    futures.emplace_back(std::async(std::launch::deferred, [&]() { ++completed; }));
    REQUIRE_THROWS_WITH(vsag::WaitTypedKMeansTasks(futures), "primary callback failure");
    REQUIRE(completed == 4);
    REQUIRE(futures.empty());

    // Also exercise the successful-submission / failing-task path of the wrapper.
    REQUIRE_THROWS_WITH(
        vsag::RunTypedKMeansTasks(futures,
                                  1,
                                  [&]() {
                                      futures.emplace_back(std::async(std::launch::deferred, [&]() {
                                          ++completed;
                                          throw std::runtime_error("callback failure");
                                      }));
                                  }),
        "callback failure");
    REQUIRE(completed == 5);
    REQUIRE(futures.empty());
}

TEST_CASE("Typed future storage is reserved before submission",
          "[ut][TypedKMeans][TypedKMeansUnwind]") {
    std::vector<std::future<void>> futures;
    bool submitted = false;
    REQUIRE_THROWS_AS(
        vsag::RunTypedKMeansTasks(futures, futures.max_size() + 1, [&]() { submitted = true; }),
        std::length_error);
    REQUIRE_FALSE(submitted);
    REQUIRE(futures.empty());
    vsag::RunTypedKMeansTasks(futures, 3, [&]() {
        const auto capacity = futures.capacity();
        REQUIRE(capacity >= 3);
        for (uint64_t i = 0; i < 3; ++i) {
            futures.emplace_back(std::async(std::launch::deferred, []() {}));
            REQUIRE(futures.capacity() == capacity);
        }
    });
    REQUIRE(futures.empty());
}

TEST_CASE("Typed provider inline update failure propagates",
          "[ut][TypedKMeans][TypedKMeansUnwind]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    auto pool = std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(1), true);
    auto run = pool->GeneralEnqueue([&]() {
        if (!pool->IsCurrentThread()) {
            throw std::runtime_error("expected inline worker context");
        }
        struct Context {
            mutable uint64_t calls{0};
        } context;
        auto reader = +[](const void* opaque, uint64_t, float* out) {
            const auto& ctx = *static_cast<const Context*>(opaque);
            if (++ctx.calls > 42) {
                throw std::runtime_error("inline provider update failure");
            }
            out[0] = 1.0F;
            out[1] = 2.0F;
        };
        vsag::TypedKMeansInput provider(41, 2, reader, &context);
        vsag::KMeansCluster cluster(2, allocator.get(), pool);
        cluster.Run(
            1, provider, 1, nullptr, false, 1e-6F, vsag::KMeansInitMethod::RANDOM, 777U, true);
    });
    REQUIRE_THROWS_WITH(run.get(), "inline provider update failure");
}
