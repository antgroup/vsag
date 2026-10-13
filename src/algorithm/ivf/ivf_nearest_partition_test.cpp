
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

#include "ivf_nearest_partition.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/inner_search_param.h"
#include "impl/logger/capturing_logger_test.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"
using namespace vsag;

TEST_CASE("IVF Nearest Partition Basic Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    std::vector<SafeThreadPoolPtr> pools{thread_pool, nullptr};
    int64_t dim = 128;
    int64_t bucket_count = 20;
    for (auto& tp : pools) {
        IndexCommonParam param;
        param.dim_ = 128;
        param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        param.allocator_ = allocator;
        param.thread_pool_ = tp;

        IVFPartitionStrategyParametersPtr strategy_param =
            std::make_shared<IVFPartitionStrategyParameters>();
        auto partition = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);

        auto dataset = Dataset::Make();
        int64_t data_count = 1000L;
        auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
        dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

        partition->Train(dataset);
        auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);
        REQUIRE(class_result.size() == data_count);

        auto index = partition->route_index_ptr_;
        // Match ClassifyDatas so this checks its routing rather than HGraph search breadth.
        std::string route_search_param = R"(
        {
            "hgraph": {
                "ef_search": 10
            }
        }
        )";
        FilterPtr filter = nullptr;
        for (int64_t i = 0; i < data_count; ++i) {
            auto query = Dataset::Make();
            query->Dim(dim)->Float32Vectors(vec.data() + i * dim)->NumElements(1)->Owner(false);
            auto result = index->KnnSearch(query, 1, route_search_param, filter);
            auto id = result->GetIds()[0];
            REQUIRE(id == class_result[i]);
        }
    }
}

TEST_CASE("IVF Nearest Partition Serialize Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    int64_t dim = 128;
    int64_t bucket_count = 20;
    IndexCommonParam param;
    param.dim_ = 128;
    param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    param.allocator_ = allocator;
    IVFPartitionStrategyParametersPtr strategy_param =
        std::make_shared<IVFPartitionStrategyParameters>();
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);

    auto dataset = Dataset::Make();
    int64_t data_count = 1000L;
    auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
    dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    partition->Train(dataset);
    auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);
    REQUIRE(class_result.size() == data_count);

    auto partition2 = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);
    test_serializion(*partition, *partition2);

    auto restored_class_result = partition2->ClassifyDatas(vec.data(), data_count, 1, nullptr);
    REQUIRE(restored_class_result == class_result);
}

TEST_CASE("IVF Nearest Partition Routing Statistics Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto thread_pool = SafeThreadPool::FactoryDefaultThreadPool();
    std::vector<SafeThreadPoolPtr> pools{thread_pool, nullptr};
    int64_t dim = 128;
    int64_t bucket_count = 20;
    for (auto& tp : pools) {
        IndexCommonParam param;
        param.dim_ = dim;
        param.metric_ = MetricType::METRIC_TYPE_L2SQR;
        param.allocator_ = allocator;
        param.thread_pool_ = tp;

        IVFPartitionStrategyParametersPtr strategy_param =
            std::make_shared<IVFPartitionStrategyParameters>();
        auto partition = std::make_unique<IVFNearestPartition>(bucket_count, param, strategy_param);

        auto dataset = Dataset::Make();
        int64_t data_count = 1000L;
        auto vec = fixtures::generate_vectors(data_count, dim, true, 95);
        dataset->Float32Vectors(vec.data())->Dim(dim)->NumElements(data_count)->Owner(false);

        partition->Train(dataset);
        auto class_result = partition->ClassifyDatas(vec.data(), data_count, 1, nullptr);

        // The search path accumulates routing statistics through a non-null QueryContext. The
        // statistics parsing must produce the same bucket assignment and non-zero routing stats
        // (the parse now runs outside the reduce lock).
        SearchStatistics stats;
        QueryContext ctx;
        ctx.stats = &stats;
        auto stats_result = partition->ClassifyDatas(vec.data(), data_count, 1, &ctx);
        REQUIRE(stats_result == class_result);
        REQUIRE(stats.dist_cmp.load() > 0);
        auto dumped = JsonType::Parse(stats.Dump());
        REQUIRE(dumped["distance_evaluations_by_phase"]["routing"].GetUint64() > 0);
    }
}

TEST_CASE("IVF Nearest Partition Centroid Scan Test", "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t data_count = 256;
    constexpr BucketIdType buckets_per_data = 4;
    const std::vector<MetricType> metrics{
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};

    for (const auto metric : metrics) {
        IndexCommonParam common_param;
        common_param.dim_ = dim;
        common_param.metric_ = metric;
        common_param.allocator_ = allocator;

        auto strategy_param = std::make_shared<IVFPartitionStrategyParameters>();
        strategy_param->use_route_graph = false;
        auto partition =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, strategy_param);
        REQUIRE(partition->route_index_ptr_ == nullptr);

        const auto untrained_result =
            partition->ClassifyDatas(nullptr, 1, buckets_per_data, nullptr);
        REQUIRE(untrained_result == Vector<BucketIdType>(buckets_per_data,
                                                         static_cast<BucketIdType>(-1),
                                                         allocator.get()));
        auto untrained_restored =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, strategy_param);
        test_serializion(*partition, *untrained_restored);
        REQUIRE(untrained_restored->ClassifyDatas(nullptr, 1, buckets_per_data, nullptr) ==
                untrained_result);

        // KMeans sees exactly bucket_count distinct points, so initialization only permutes
        // the centroids. Powers of two on separate axes keep L2 arithmetic exact and cosine
        // centroids unit length. Random centroids can produce near-ties that BLAS and the
        // independent SIMD distance oracle round differently.
        std::vector<float> vectors(data_count * dim, 0.0F);
        for (int64_t i = 0; i < data_count; ++i) {
            const auto axis = i % bucket_count;
            vectors[i * dim + axis] = static_cast<float>(1 << (axis % 4));
        }
        auto dataset = Dataset::Make();
        dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);
        partition->Train(dataset);
        REQUIRE(partition->route_index_ptr_ == nullptr);

        // Exercise different norms, signs and query directions, including exact ties. These
        // queries are separate from the training points and retain exact bucket-ID checks.
        std::vector<float> queries(data_count * dim, 0.0F);
        for (int64_t i = 1; i < data_count; ++i) {
            const auto scale = static_cast<float>(1 + i / bucket_count);
            for (int64_t j = 0; j < buckets_per_data; ++j) {
                queries[i * dim + (i + j) % bucket_count] =
                    scale * static_cast<float>(j + 1) * (i % 2 == 0 ? 0.25F : -0.25F);
            }
        }
        auto actual =
            partition->ClassifyDatas(queries.data(), data_count, buckets_per_data, nullptr);
        REQUIRE(actual.size() == static_cast<uint64_t>(data_count * buckets_per_data));

        Vector<float> centroid(dim, allocator.get());
        Vector<float> normalized_query(dim, allocator.get());
        Vector<std::pair<float, BucketIdType>> expected(bucket_count, allocator.get());
        for (int64_t i = 0; i < data_count; ++i) {
            const auto* query = queries.data() + i * dim;
            if (metric == MetricType::METRIC_TYPE_COSINE) {
                Normalize(query, normalized_query.data(), dim);
                query = normalized_query.data();
            }
            for (BucketIdType b = 0; b < bucket_count; ++b) {
                partition->GetCentroid(b, centroid);
                float distance = 0.0F;
                if (metric == MetricType::METRIC_TYPE_L2SQR) {
                    distance = FP32ComputeL2Sqr(query, centroid.data(), dim);
                } else {
                    distance = 1.0F - FP32ComputeIP(query, centroid.data(), dim);
                }
                expected[b] = {distance, b};
            }
            std::sort(expected.begin(), expected.end());
            for (BucketIdType j = 0; j < buckets_per_data; ++j) {
                CAPTURE(static_cast<int>(metric), i, j);
                REQUIRE(actual[i * buckets_per_data + j] == expected[j].second);
            }
        }

        auto graph_config_param = std::make_shared<IVFPartitionStrategyParameters>();
        graph_config_param->use_route_graph = true;
        auto restored =
            std::make_unique<IVFNearestPartition>(bucket_count, common_param, graph_config_param);
        test_serializion(*partition, *restored);
        REQUIRE(restored->route_index_ptr_ == nullptr);
        REQUIRE(restored->ClassifyDatas(queries.data(), data_count, buckets_per_data, nullptr) ==
                actual);
    }
}

TEST_CASE("IVF Nearest Partition Route Graph Layout Cross Config Test",
          "[ut][IVFNearestPartition]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t data_count = 256;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto graph_param = std::make_shared<IVFPartitionStrategyParameters>();
    graph_param->use_route_graph = true;
    auto graph_partition =
        std::make_unique<IVFNearestPartition>(bucket_count, common_param, graph_param);

    auto vectors = fixtures::generate_vectors(data_count, dim, true, 95);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);
    graph_partition->Train(dataset);
    auto expected = graph_partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    auto scan_config_param = std::make_shared<IVFPartitionStrategyParameters>();
    scan_config_param->use_route_graph = false;
    auto restored =
        std::make_unique<IVFNearestPartition>(bucket_count, common_param, scan_config_param);
    REQUIRE(restored->route_index_ptr_ == nullptr);
    test_serializion(*graph_partition, *restored);
    REQUIRE(restored->route_index_ptr_ != nullptr);
    REQUIRE(restored->ClassifyDatas(vectors.data(), data_count, 1, nullptr) == expected);
}

namespace {

std::vector<std::vector<float>>
GatherCentroids(IVFNearestPartition& partition,
                BucketIdType bucket_count,
                int64_t dim,
                Allocator* allocator) {
    std::vector<std::vector<float>> centroids;
    Vector<float> centroid(dim, allocator);
    for (BucketIdType b = 0; b < bucket_count; ++b) {
        partition.GetCentroid(b, centroid);
        centroids.emplace_back(centroid.begin(), centroid.end());
    }
    return centroids;
}

/// Checks a build-time assignment against the centroids it was made from.
///
/// Not by comparing bucket ids. Two trained centroids can coincide, because
/// k-means may seed two into one cluster and leave another empty, and then
/// which of the pair a point is assigned to is a genuine tie that the device
/// and the host are each free to break their own way. What holds whatever ran is
/// that the offered assignment is no further from its centroid than the host's.
///
/// `host_is_exact` says whether the host path this was compared against compares
/// against every centroid, which the scan layout does and the route graph does
/// not: that one searches with ef_search 10 and is approximate once buckets_count
/// is past a few dozen. Only where the host is exact may either label be required
/// to name the nearest centroid outright, and requiring it of `on_device`
/// regardless would fail in a build with no device, where `on_device` is the
/// host's answer.
///
/// Squared L2 to the normalised query is the right reference for cosine too:
/// the host routing key there is `1 - <x, c>` over normalised centroids, and
/// the two differ by a positive constant factor.
void
RequireSameRouting(const std::vector<std::vector<float>>& centroids,
                   const std::vector<float>& queries,
                   int64_t count,
                   int64_t dim,
                   const Vector<BucketIdType>& on_device,
                   const Vector<BucketIdType>& on_host,
                   bool cosine,
                   bool host_is_exact) {
    std::vector<float> unit(dim);
    for (int64_t i = 0; i < count; ++i) {
        const float* query = queries.data() + i * dim;
        if (cosine) {
            Normalize(query, unit.data(), dim);
            query = unit.data();
        }
        float best = std::numeric_limits<float>::max();
        for (const auto& centroid : centroids) {
            best = std::min(best, FP32ComputeL2Sqr(query, centroid.data(), dim));
        }
        const float tolerance = 1e-3F * (1.0F + best);
        // Both are checked before either indexes the centroid set: the host's
        // label is the reference the next assertion reads, and an untrained
        // partition returns INVALID_BUCKET_ID, which would index it at -1.
        REQUIRE(on_device[i] >= 0);
        REQUIRE(on_device[i] < static_cast<BucketIdType>(centroids.size()));
        REQUIRE(on_host[i] >= 0);
        REQUIRE(on_host[i] < static_cast<BucketIdType>(centroids.size()));
        const float device_distance = FP32ComputeL2Sqr(query, centroids[on_device[i]].data(), dim);
        const float host_distance = FP32ComputeL2Sqr(query, centroids[on_host[i]].data(), dim);
        REQUIRE(device_distance <= host_distance + tolerance);
        if (host_is_exact) {
            REQUIRE(host_distance <= best + tolerance);
            REQUIRE(device_distance <= best + tolerance);
        }
    }
}

}  // namespace

TEST_CASE("IVF Nearest Partition GPU Build Fallback Test", "[ut][IVFNearestPartition]") {
    // Asking for the CUDA backend must not change what the partition returns.
    // Where there is no device, or the backend is compiled out, the build-time
    // assignment falls back to the host path and has to produce the same
    // buckets; where there is one, this compares the two paths directly.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t per_cluster = 16;
    constexpr int64_t data_count = bucket_count * per_cluster;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    // Well separated clusters, so which centroid is nearest does not turn on
    // how the distance was reassociated and the two paths must agree exactly.
    std::vector<float> vectors(data_count * dim, 0.0F);
    for (int64_t i = 0; i < data_count; ++i) {
        const int64_t cluster = i / per_cluster;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = (j == cluster % dim ? 100.0F : 0.0F) +
                                   static_cast<float>((i * 31 + j) % 7) * 0.001F;
        }
    }
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = false;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    // The same trained partition, asked to use the device. The threshold has to
    // come down as well: the calibrated default keeps a problem this small on
    // the CPU whatever the flag says.
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    const auto on_device = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    const auto centroids = GatherCentroids(*partition, bucket_count, dim, allocator.get());
    RequireSameRouting(centroids, vectors, data_count, dim, on_device, on_host, false, true);

    // Search never reaches the device, whatever the flag says: it routes one
    // query at a time, which is the shape the graph and scan paths are for.
    //
    // Comparing buckets would not show that, because the device would name the
    // same ones. What only the host path does is count its distance
    // evaluations, so a run that came back with routing statistics is a run that
    // stayed on the host.
    //
    // One bucket per query, which is the shape the device pass accepts; asking
    // for several is refused for its own reasons and would prove nothing here.
    InnerSearchParam search_param;
    search_param.scan_bucket_size = 1;
    SearchStatistics stats;
    QueryContext search_ctx;
    search_ctx.stats = &stats;
    const auto searched =
        partition->ClassifyDatasForSearch(vectors.data(), 1, search_param, &search_ctx);
    REQUIRE(searched.size() == search_param.scan_bucket_size);
    REQUIRE(searched[0] == on_host[0]);
    REQUIRE(stats.dist_cmp.load() > 0);
}

TEST_CASE("IVF Nearest Partition GPU Build Leaves Inner Product On The Host",
          "[ut][IVFNearestPartition]") {
    // The device pass minimises squared L2. Inner product orders buckets by the
    // dot alone, which disagrees wherever centroid norms differ, so that metric
    // has to stay on the host even with the backend switched on.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 8;
    constexpr int64_t data_count = 200;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;
    common_param.allocator_ = allocator;

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = false;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);

    auto vectors = fixtures::generate_vectors(data_count, dim, false, 71);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);
    partition->Train(dataset);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    REQUIRE(partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr) == on_host);
}

TEST_CASE("IVF Nearest Partition GPU Build Route Graph Layout Test", "[ut][IVFNearestPartition]") {
    // With a routing graph the centroids live inside it rather than in
    // centroids_, so the device path has to gather them out before it can run.
    // The scan layout above never exercises that gather.
    //
    // At 16 centroids the graph search reaches all of them, so the host and the
    // device agree on every vector here. What that difference looks like at a
    // size where the search is not nearly exhaustive is the case below.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t per_cluster = 16;
    constexpr int64_t data_count = bucket_count * per_cluster;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    std::vector<float> vectors(data_count * dim, 0.0F);
    for (int64_t i = 0; i < data_count; ++i) {
        const int64_t cluster = i / per_cluster;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = (j == cluster % dim ? 100.0F : 0.0F) +
                                   static_cast<float>((i * 31 + j) % 7) * 0.001F;
        }
    }
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = true;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);
    REQUIRE(partition->route_index_ptr_ != nullptr);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    const auto on_device = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    // GetCentroid reads the graph as well, so this also shows the gather read
    // the centroids the graph actually holds.
    const auto centroids = GatherCentroids(*partition, bucket_count, dim, allocator.get());
    RequireSameRouting(
        centroids, vectors, data_count, dim, on_device, on_host, false, /*host_is_exact=*/false);
}

TEST_CASE("IVF Nearest Partition GPU Build Assigns No Worse Than The Route Graph",
          "[ut][IVFNearestPartition]") {
    // The routing graph is searched with ef_search 10. Once buckets_count is
    // past a few dozen that search stops being nearly exhaustive, so the host's
    // build-time assignment is approximate while the device compares against
    // every centroid. Measured on 20000 random vectors of 64 dimensions, the two
    // placed roughly 10% of vectors differently at 256 centroids and 12% at 1000,
    // the device naming the nearer one every time. The shares move a little
    // between runs, because Train seeds the clustering from random_device.
    //
    // So what is asserted is a direction, not an equality: no vector further
    // from its centroid than the host put it, and no worse on average. Without a
    // device both arms are the host path and the two are equal, so this passes
    // without discriminating, as every case here that needs one does.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 256;
    constexpr int64_t data_count = 8000;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto vectors = fixtures::generate_vectors(data_count, dim, true, 131);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = true;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    const auto on_device = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    const auto centroids = GatherCentroids(*partition, bucket_count, dim, allocator.get());
    RequireSameRouting(
        centroids, vectors, data_count, dim, on_device, on_host, false, /*host_is_exact=*/false);

    double host_total = 0.0;
    double device_total = 0.0;
    for (int64_t i = 0; i < data_count; ++i) {
        const float* query = vectors.data() + i * dim;
        host_total += FP32ComputeL2Sqr(query, centroids[on_host[i]].data(), dim);
        device_total += FP32ComputeL2Sqr(query, centroids[on_device[i]].data(), dim);
    }
    REQUIRE(device_total <= host_total * (1.0 + 1e-6));
}

TEST_CASE("IVF Nearest Partition GPU Build Cosine Test", "[ut][IVFNearestPartition]") {
    // Cosine is the second metric the device path accepts. It agrees with
    // squared L2 only because the centroids are normalised during training, so
    // the norm term is the same for all of them; this pins that.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t per_cluster = 16;
    constexpr int64_t data_count = bucket_count * per_cluster;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_COSINE;
    common_param.allocator_ = allocator;

    // Directions far apart, and lengths differing by half within a direction.
    //
    // The spread has to stay well under the gap between directions. Training
    // clusters the raw vectors by squared L2 and only normalises the centroids
    // afterwards, so a wider spread splits one direction into two clusters
    // whose centroids coincide once normalised, and which of the two a point is
    // assigned to is then a genuine tie rather than anything to assert on.
    std::vector<float> vectors(data_count * dim, 0.0F);
    for (int64_t i = 0; i < data_count; ++i) {
        const int64_t cluster = i / per_cluster;
        const float scale =
            1.0F + 0.5F * static_cast<float>(i % per_cluster) / static_cast<float>(per_cluster - 1);
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = (j == cluster % dim ? 1.0F : 0.001F) * scale;
        }
    }
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = false;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    const auto on_device = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    const auto centroids = GatherCentroids(*partition, bucket_count, dim, allocator.get());
    RequireSameRouting(centroids, vectors, data_count, dim, on_device, on_host, true, true);
}

TEST_CASE("IVF Nearest Partition GPU Build Rejects An Absent Device", "[ut][IVFNearestPartition]") {
    // Naming a device the machine does not have keeps the build on the host
    // rather than quietly using a different card. This holds on any machine:
    // where there is no CUDA at all the same fallback runs one step earlier.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 16;
    constexpr BucketIdType bucket_count = 8;
    constexpr int64_t data_count = 128;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto vectors = fixtures::generate_vectors(data_count, dim, true, 83);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = false;
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);

    const auto on_host = partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;
    param->gpu_device_id = 4096;
    REQUIRE(partition->ClassifyDatas(vectors.data(), data_count, 1, nullptr) == on_host);
}

TEST_CASE("IVF Nearest Partition GPU Build Declines What It Does Not Compute",
          "[ut][IVFNearestPartition]") {
    // Two cases the device pass does not answer, both of which have to reach
    // the host path unchanged rather than be refused outright. Neither needs a
    // device to observe, so they hold in both build configurations.
    //
    // What they return is the host's answer either way, so the trace is the only
    // place the reason is observable, and a caller who turned the backend on and
    // saw no change has nothing else to read.
    vsag::testing::CapturingLogger capture;
    const vsag::testing::ScopedLogger installed(&capture);
    const auto traced = [&capture](const std::string& reason) {
        return std::any_of(
            capture.lines.begin(), capture.lines.end(), [&reason](const std::string& line) {
                return line.find("IVFNearestPartition::ClassifyDatas") != std::string::npos and
                       line.find(reason) != std::string::npos;
            });
    };
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 16;
    constexpr BucketIdType bucket_count = 8;
    constexpr int64_t data_count = 128;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    auto vectors = fixtures::generate_vectors(data_count, dim, true, 91);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto param = std::make_shared<IVFPartitionStrategyParameters>();
    param->use_route_graph = false;
    param->enable_gpu_build = true;
    param->gpu_min_work_threshold = 1;

    // Untrained: there are no centroids to read, so every slot stays invalid
    // rather than the pass being handed uninitialised storage.
    auto untrained = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    const auto before_training = untrained->ClassifyDatas(vectors.data(), data_count, 1, nullptr);
    REQUIRE(before_training.size() == data_count);
    REQUIRE(std::all_of(before_training.begin(), before_training.end(), [](BucketIdType b) {
        return b == static_cast<BucketIdType>(-1);
    }));
    REQUIRE(traced("the partition is not trained"));

    // More than one bucket per vector is a nearest-k question, which the host
    // routing answers and the device pass does not.
    auto partition = std::make_unique<IVFNearestPartition>(bucket_count, common_param, param);
    partition->Train(dataset);
    constexpr BucketIdType buckets_per_data = 3;
    capture.lines.clear();
    const auto many =
        partition->ClassifyDatas(vectors.data(), data_count, buckets_per_data, nullptr);
    REQUIRE(many.size() == data_count * buckets_per_data);
    REQUIRE(traced("asks for more than one bucket per vector"));

    // And with the backend off there is nothing to explain, so nothing is said.
    param->enable_gpu_build = false;
    capture.lines.clear();
    REQUIRE(partition->ClassifyDatas(vectors.data(), data_count, buckets_per_data, nullptr) ==
            many);
    REQUIRE_FALSE(traced("bucket per vector"));
}

TEST_CASE("IVF Nearest Partition GPU Build Serializes Unchanged", "[ut][IVFNearestPartition]") {
    // The four GPU settings decide how the centroids are computed, not what the
    // index looks like afterwards. None of them is written, so a partition
    // trained with the backend on has to load into a reader that knows nothing
    // about it and route the same way.
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    constexpr int64_t dim = 32;
    constexpr BucketIdType bucket_count = 16;
    constexpr int64_t per_cluster = 16;
    constexpr int64_t data_count = bucket_count * per_cluster;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = allocator;

    std::vector<float> vectors(data_count * dim, 0.0F);
    for (int64_t i = 0; i < data_count; ++i) {
        const int64_t cluster = i / per_cluster;
        for (int64_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = (j == cluster % dim ? 100.0F : 0.0F) +
                                   static_cast<float>((i * 31 + j) % 7) * 0.001F;
        }
    }
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(vectors.data())->Dim(dim)->NumElements(data_count)->Owner(false);

    auto gpu_param = std::make_shared<IVFPartitionStrategyParameters>();
    gpu_param->use_route_graph = false;
    gpu_param->enable_gpu_build = true;
    gpu_param->gpu_min_work_threshold = 1;
    auto trained = std::make_unique<IVFNearestPartition>(bucket_count, common_param, gpu_param);
    trained->Train(dataset);
    const auto before = trained->ClassifyDatas(vectors.data(), data_count, 1, nullptr);

    // The reader is configured without any GPU setting at all.
    auto host_param = std::make_shared<IVFPartitionStrategyParameters>();
    host_param->use_route_graph = false;
    auto restored = std::make_unique<IVFNearestPartition>(bucket_count, common_param, host_param);
    test_serializion(*trained, *restored);
    REQUIRE(restored->ClassifyDatas(vectors.data(), data_count, 1, nullptr) == before);

    // The bytes a partition writes do not depend on the settings either: the
    // same trained state, serialized twice with the settings on and off, is
    // identical.
    auto to_string = [](IVFNearestPartition& partition) {
        std::stringstream ss;
        vsag::IOStreamWriter writer(ss);
        partition.Serialize(writer);
        return ss.str();
    };
    const auto with_settings = to_string(*trained);
    gpu_param->enable_gpu_build = false;
    gpu_param->gpu_min_work_threshold = 0;
    REQUIRE(to_string(*trained) == with_settings);
}
