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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "algorithm/build_cache.h"
#include "allocator/memory_record_allocator.h"
#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/sparse_graph_datacell.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "dataset_impl.h"
#include "hgraph.h"
#include "hgraph_fast_build.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/standard_heap.h"
#include "impl/label_table/label_table.h"
#include "impl/searcher/basic_searcher.h"
#include "impl/searcher/hybrid_mci_searcher.h"
#include "impl/searcher/mci_searcher.h"
#include "impl/searcher/parallel_searcher.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "io/memory_block_io/memory_block_io.h"
#include "layout/fixed_layout.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "quantization/scalar_quantization/sq8_uniform_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/stream_reader.h"
#include "unittest.h"
#include "utils/filter_search_skip_strategy.h"
#include "utils/linear_congruential_generator.h"
#include "utils/visited_list.h"
#include "vsag/deserialize_reader.h"
#include "vsag/engine.h"
#include "vsag/options.h"

// Expected model availability is classified from the supported test profiles, independently
// of the production capability result. Unmodeled profiles must exercise the typed fallback.
#if (defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION == 15007 || _LIBCPP_VERSION == 210106) && \
     defined(_LIBCPP_ABI_VERSION) && _LIBCPP_ABI_VERSION == 1 &&                            \
     !defined(_LIBCPP_ABI_OPTIMIZED_FUNCTION)) ||                                           \
    defined(__GLIBCXX__)
#define HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE 1
#else
#define HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE 0
#endif

namespace vsag {

// Narrow test access: no estimator breakdown or production byte total is read.
struct HGraphMemoryEstimateTestAccess {
    using Worker = HGraph::InsertAddBatchWorker;
    using WorkContext = HGraph::InsertAddBatchWorkContext;
    using WorkerBind = decltype(std::bind(std::declval<Worker>()));

    static const HGraphParameter&
    Parameters(const HGraph& graph) {
        return static_cast<const HGraphParameter&>(*graph.create_param_ptr_);
    }

    static uint64_t
    RetainedFanoutCapacity(const HGraph& graph) {
        return graph.pipnn_param_.fanout.capacity();
    }

    static std::default_random_engine
    Generator(const HGraph& graph) {
        return graph.level_generator_;
    }

    static uint64_t
    InitialCapacity(const HGraph& graph) {
        return graph.build_estimate_initial_capacity_;
    }

    static uint64_t
    RoundedCapacity(const HGraph& graph, uint64_t count) {
        return HGraph::build_estimate_checked_capacity(count, graph.resize_increase_count_bit_);
    }

    static bool
    LifecycleClean(const HGraph& graph) {
        return graph.build_estimate_lifecycle_clean_.load();
    }

    static std::array<uint64_t, 4>
    CacheStorage(const HGraph& graph) {
        return {graph.cache_->source_ids_.size(),
                graph.cache_->source_ids_.capacity(),
                graph.cache_->neighbors_.size(),
                graph.cache_->neighbors_.bucket_count()};
    }

    static void
    SetUnsupportedResizeBit(HGraph& graph) {
        graph.resize_increase_count_bit_ = 31;
    }

    static void
    RunWorkers(SafeThreadPool& pool, uint64_t threads, WorkContext& context) {
        HGraph::run_insert_add_batch_workers(pool, threads, context);
    }
};

}  // namespace vsag

namespace {

constexpr bool EXPECT_MODELED_BUILD_MEMORY = HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE != 0;

vsag::DatasetPtr
MakeFloatDataset(std::vector<float>& vectors,
                 std::vector<int64_t>& ids,
                 int64_t dim,
                 int64_t count) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

vsag::DatasetPtr
MakeFloatDatasetWithSourceIds(std::vector<float>& vectors,
                              std::vector<int64_t>& ids,
                              std::vector<std::string>& source_ids,
                              int64_t dim,
                              int64_t count) {
    auto dataset = MakeFloatDataset(vectors, ids, dim, count);
    dataset->SourceID(source_ids.data());
    return dataset;
}

vsag::DatasetPtr
MakeFloatQuery(std::vector<float>& vector, int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(1)->Dim(dim)->Float32Vectors(vector.data())->Owner(false);
    return dataset;
}

class BlockSizeLimitGuard {
public:
    explicit BlockSizeLimitGuard(uint64_t block_size_limit)
        : old_block_size_limit_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(block_size_limit);
    }

    ~BlockSizeLimitGuard() {
        vsag::Options::Instance().set_block_size_limit(old_block_size_limit_);
    }

private:
    uint64_t old_block_size_limit_;
};

vsag::IndexCommonParam
MakeCommonParam(int64_t dim, uint64_t thread_count = 0) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    if (thread_count > 0) {
        common_param.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
        common_param.thread_pool_->SetPoolSize(thread_count);
    }
    return common_param;
}

vsag::JsonType
MakeFp32HGraphJson(bool deduplicate_storage = true, int64_t build_thread_count = 1) {
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true
    })");
    hgraph_json["build_thread_count"].SetInt(build_thread_count);
    hgraph_json["deduplicate_storage"].SetBool(deduplicate_storage);
    return hgraph_json;
}

vsag::JsonType
MakeFp32ReorderHGraphJson() {
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["use_reorder"].SetBool(true);
    hgraph_json["precise_quantization_type"].SetString("fp32");
    return hgraph_json;
}

vsag::JsonType
MakeRabitQRawVectorHGraphJson() {
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["base_quantization_type"].SetString("rabitq");
    hgraph_json["rabitq_bits_per_dim_base"].SetInt(1);
    hgraph_json["rabitq_bits_per_dim_query"].SetInt(32);
    hgraph_json["store_raw_vector"].SetBool(true);
    return hgraph_json;
}

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
MakeHGraphIndex(const vsag::JsonType& hgraph_json, const vsag::IndexCommonParam& common_param) {
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);
}

bool
ResultContainsId(const vsag::DatasetPtr& result, int64_t id) {
    const auto* ids = result->GetIds();
    return std::find(ids, ids + result->GetDim(), id) != ids + result->GetDim();
}

void
RequireRangeContains(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index,
                     const vsag::DatasetPtr& query,
                     std::initializer_list<int64_t> expected_ids,
                     int64_t expected_count = -1,
                     float radius = 0.01F,
                     const std::string& parameters = R"({"hgraph": {"ef_search": 32}})") {
    auto search_result = index->RangeSearch(query, radius, parameters);
    REQUIRE(search_result.has_value());
    if (expected_count >= 0) {
        REQUIRE(search_result.value()->GetDim() == expected_count);
    }
    for (auto expected_id : expected_ids) {
        REQUIRE(ResultContainsId(search_result.value(), expected_id));
    }
}

const std::string kBruteForceSearchParams =
    R"({"hgraph": {"ef_search": 32, "brute_force_threshold": 1.0}})";

// This decorator owns its phase accounting; the shared fixture is unchanged.
class PhaseCountingAllocator : public vsag::Allocator {
public:
    struct Snapshot {
        uint64_t current;
        uint64_t constructor_peak;
        uint64_t phase_peak;
        uint64_t requests;
    };

    std::string
    Name() override {
        return "HGraphPhaseCountingAllocator";
    }

    void*
    Allocate(uint64_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++requests_;
        auto* result = backing_.Allocate(bytes);
        UpdatePeak();
        return result;
    }

    void*
    Reallocate(void* pointer, uint64_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++requests_;
        auto* result = backing_.Reallocate(pointer, bytes);
        UpdatePeak();
        return result;
    }

    void
    Deallocate(void* pointer) override {
        std::lock_guard<std::mutex> lock(mutex_);
        backing_.Deallocate(pointer);
        UpdatePeak();
    }

    void
    BeginBuildPhase() {
        std::lock_guard<std::mutex> lock(mutex_);
        constructor_peak_ = backing_.GetMemoryPeak();
        phase_peak_ = backing_.GetCurrentMemory();
        armed_ = true;
    }

    Snapshot
    Read() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {backing_.GetCurrentMemory(), constructor_peak_, phase_peak_, requests_};
    }

private:
    void
    UpdatePeak() {
        if (armed_) {
            phase_peak_ = std::max(phase_peak_, backing_.GetCurrentMemory());
        }
    }

    mutable std::mutex mutex_;
    fixtures::MemoryRecordAllocator backing_;
    uint64_t requests_{0};
    uint64_t constructor_peak_{0};
    uint64_t phase_peak_{0};
    bool armed_{false};
};

vsag::JsonType
MakeBuildMemoryJson(const std::string& quantizer = "fp32",
                    uint64_t degree = 8,
                    uint64_t threads = 1,
                    uint64_t sample_count = 1537) {
    auto json = MakeFp32HGraphJson(false, static_cast<int64_t>(threads));
    json["base_quantization_type"].SetString(quantizer);
    json["max_degree"].SetUint64(degree);
    json["ef_construction"].SetUint64(64);
    json["support_duplicate"].SetBool(false);
    json["deduplicate_storage"].SetBool(false);
    json["train_sample_count"].SetUint64(std::max(uint64_t{512}, sample_count));
    return json;
}

struct BuildMemoryData {
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    vsag::DatasetPtr dataset;
    uint64_t checksum{1469598103934665603ULL};

    BuildMemoryData(uint64_t count, uint64_t dim) : vectors(count * dim), ids(count) {
        std::iota(ids.begin(), ids.end(), int64_t{0});
        for (uint64_t row = 0; row < count; ++row) {
            for (uint64_t col = 0; col < dim; ++col) {
                const int64_t value =
                    static_cast<int64_t>((row * row + (row + 1) * (col + 17)) % 1021) - 510;
                auto& element = vectors[row * dim + col];
                element = static_cast<float>(value) / 512.0F;
            }
        }
        dataset =
            MakeFloatDataset(vectors, ids, static_cast<int64_t>(dim), static_cast<int64_t>(count));
        RefreshChecksum();
    }

    void
    RefreshChecksum() {
        checksum = 1469598103934665603ULL;
        for (const auto element : vectors) {
            uint32_t bits = 0;
            std::memcpy(&bits, &element, sizeof(bits));
            checksum = (checksum ^ bits) * 1099511628211ULL;
        }
        for (const auto id : ids) {
            checksum = (checksum ^ static_cast<uint64_t>(id)) * 1099511628211ULL;
        }
    }
};

std::shared_ptr<vsag::HGraph>
InnerHGraph(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index) {
    auto graph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(graph != nullptr);
    return graph;
}

template <typename F>
void
RequireEstimateError(F&& operation, vsag::ErrorType expected) {
    bool caught = false;
    try {
        operation();
    } catch (const vsag::VsagException& exception) {
        REQUIRE(exception.error_.type == expected);
        caught = true;
    }
    REQUIRE(caught);
}

void
RequireUnsupportedBuildEstimate(const vsag::IndexPtr& index) {
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ESTIMATE_BUILD_MEMORY));
    for (uint64_t count : {uint64_t{0}, uint64_t{7}}) {
        RequireEstimateError([&] { (void)index->EstimateBuildMemory(count); },
                             vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    }
}

// A false runtime capability cannot skip the positive expectation on a modeled profile.
void
RequireExpectedBuildMemoryProfile(const vsag::IndexPtr& index) {
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ESTIMATE_BUILD_MEMORY) ==
            EXPECT_MODELED_BUILD_MEMORY);
    if constexpr (EXPECT_MODELED_BUILD_MEMORY) {
        REQUIRE(index->EstimateBuildMemory(0) > 0);
        REQUIRE(index->EstimateBuildMemory(7) >= index->EstimateMemory(7));
    } else {
        RequireUnsupportedBuildEstimate(index);
    }
}

// This observing wrapper is used only for dispatch, never the byte matrix.
class GatedBuildPool : public vsag::ThreadPool {
public:
    explicit GatedBuildPool(uint64_t fail_submission = std::numeric_limits<uint64_t>::max())
        : backend_(std::make_shared<vsag::DefaultThreadPool>(8)),
          fail_submission_(fail_submission) {
    }

    ~GatedBuildPool() override {
        Release();
        backend_->WaitUntilEmpty();
    }

    std::future<void>
    Enqueue(std::function<void()> task) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto submission = submissions_++;
            if (submission == fail_submission_) {
                submission_failed_ = true;
                changed_.notify_all();
                throw std::runtime_error("controlled Build submission failure");
            }
        }
        return backend_->Enqueue([this, task = std::move(task)] {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ++started_;
                ++active_;
                max_active_ = std::max(max_active_, active_);
                changed_.notify_all();
                changed_.wait(lock, [this] { return released_; });
            }
            try {
                task();
            } catch (...) {
                Finish();
                throw;
            }
            Finish();
        });
    }

    bool
    AwaitStarted(uint64_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(10), [&] { return started_ >= count; });
    }

    bool
    AwaitFinished(uint64_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(
            lock, std::chrono::seconds(10), [&] { return finished_ >= count; });
    }

    bool
    AwaitSubmissionFailure() {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(
            lock, std::chrono::seconds(10), [this] { return submission_failed_; });
    }

    void
    Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }

    std::array<uint64_t, 3>
    Counts() {
        std::lock_guard<std::mutex> lock(mutex_);
        return {submissions_, finished_, max_active_};
    }

    void
    WaitUntilEmpty() override {
        backend_->WaitUntilEmpty();
    }

    void
    SetPoolSize(uint64_t limit) override {
        backend_->SetPoolSize(limit);
    }

    void
    SetQueueSizeLimit(uint64_t limit) override {
        backend_->SetQueueSizeLimit(limit);
    }

private:
    void
    Finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        --active_;
        ++finished_;
        changed_.notify_all();
    }

    std::shared_ptr<vsag::DefaultThreadPool> backend_;
    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t fail_submission_;
    uint64_t submissions_{0};
    uint64_t started_{0};
    uint64_t finished_{0};
    uint64_t active_{0};
    uint64_t max_active_{0};
    bool submission_failed_{false};
    bool released_{false};
};

class BuildPoolReleaseGuard {
public:
    explicit BuildPoolReleaseGuard(std::shared_ptr<GatedBuildPool> pool) : pool_(std::move(pool)) {
    }
    ~BuildPoolReleaseGuard() {
        pool_->Release();
    }

private:
    std::shared_ptr<GatedBuildPool> pool_;
};

#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

template <typename T>
constexpr uint64_t
TestSharedRequest() {
#if defined(_LIBCPP_VERSION)
    return sizeof(std::__shared_ptr_emplace<T, std::allocator<T>>);
#else
    return sizeof(std::_Sp_counted_ptr_inplace<T, std::allocator<T>, __gnu_cxx::_S_atomic>);
#endif
}

constexpr uint64_t
TestRoundUp(uint64_t bytes, uint64_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

// Storage bounds, not surrogate implementations of inaccessible production classes.
// Worker3 checks RemoveListFilter's actual compiler record layout against this bound.
constexpr uint64_t kFilterAlignment = std::max(alignof(vsag::Filter), alignof(void*));
constexpr uint64_t kFilterUpperSize = TestRoundUp(
    TestRoundUp(sizeof(vsag::Filter), alignof(void*)) + 2 * sizeof(void*), kFilterAlignment);
using RemoveListFilterStorageUpper =
    typename std::aligned_storage<kFilterUpperSize, kFilterAlignment>::type;
static_assert(kFilterUpperSize == 24 && kFilterAlignment == 8,
              "Recheck the actual implementation-local filter record on this target");
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106
static_assert(TestSharedRequest<RemoveListFilterStorageUpper>() == 48,
              "Apple21 actual RemoveListFilter shared allocation is 48 total bytes");
#elif defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE == 11
static_assert(TestSharedRequest<RemoveListFilterStorageUpper>() == 40,
              "GCC11 actual RemoveListFilter shared allocation is 40 total bytes");
#endif

uint64_t
TestFunctionTargetUpper(uint64_t capture_bytes, bool nothrow_copy) {
    static_assert(sizeof(void*) == 8, "The validated layout model is LP64");
#if defined(_LIBCPP_VERSION)
    // __function::__base has a vptr; retain all capture padding and final alignment.
    const auto target = TestRoundUp(sizeof(void*) + capture_bytes, alignof(void*));
    return target <= 3 * sizeof(void*) && nothrow_copy ? 0 : target;
#else
    // These inaccessible captures own shared_ptr/function and are not location-invariant.
    (void)nothrow_copy;
    return capture_bytes;
#endif
}

template <typename Bound>
uint64_t
TestPackagedTaskRequest() {
    using Task = std::packaged_task<void()>;
#if defined(_LIBCPP_VERSION)
    using Callable = std::__packaged_task_func<Bound, std::allocator<Bound>, void()>;
    using Buffer = typename std::aligned_storage<3 * sizeof(void*)>::type;
    // This SDK's promise<void> constructor made one 144-byte align-8 request.
    // get_future made none. The future object is charged separately.
#if _LIBCPP_VERSION == 210106
    static_assert(sizeof(std::__assoc_sub_state) == 144, "Apple21 promise state request");
    static_assert(TestSharedRequest<Task>() == 64, "Apple21 packaged_task shared control");
#endif
    return TestSharedRequest<Task>() + sizeof(std::__assoc_sub_state) +
           (sizeof(Callable) <= sizeof(Buffer) ? 0 : sizeof(Callable));
#else
    // allocate_shared<_Task_state> plus new _Result<void>. The separate
    // promise<void> _State_baseV2 control is not this packaged task.
    using State = std::__future_base::_Task_state<Bound, std::allocator<int>, void()>;
    using Control = std::_Sp_counted_ptr_inplace<State, std::allocator<int>, __gnu_cxx::_S_atomic>;
    return TestSharedRequest<Task>() + sizeof(Control) + sizeof(std::__future_base::_Result<void>);
#endif
}

uint64_t
TestSchedulerTaskRequest() {
    using Access = vsag::HGraphMemoryEstimateTestAccess;
    using InnerBind = decltype(std::bind(std::declval<std::function<void()>>()));
    static_assert(sizeof(Access::Worker) == sizeof(void*), "Audit the real worker capture");
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION == 210106
    using WorkerCall =
        std::__packaged_task_func<Access::WorkerBind, std::allocator<Access::WorkerBind>, void()>;
    using InnerCall = std::__packaged_task_func<InnerBind, std::allocator<InnerBind>, void()>;
    static_assert(sizeof(WorkerCall) == 24, "Apple21 worker packaged callable fits SBO");
    static_assert(sizeof(InnerCall) == 48, "Apple21 function bind misses packaged SBO");
    static_assert(sizeof(std::__function::__func<Access::WorkerBind, void()>) == 24,
                  "Apple21 __func takes two parameters");
    static_assert(sizeof(std::__function::__func<InnerBind, void()>) == 48,
                  "Apple21 function-bind target misses function SBO");
#elif defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE == 11
    using WorkerState =
        std::__future_base::_Task_state<Access::WorkerBind, std::allocator<int>, void()>;
    using InnerState = std::__future_base::_Task_state<InnerBind, std::allocator<int>, void()>;
    using WorkerControl =
        std::_Sp_counted_ptr_inplace<WorkerState, std::allocator<int>, __gnu_cxx::_S_atomic>;
    using InnerControl =
        std::_Sp_counted_ptr_inplace<InnerState, std::allocator<int>, __gnu_cxx::_S_atomic>;
    static_assert(sizeof(WorkerControl) == 72, "GCC11 worker task-state control");
    static_assert(sizeof(InnerControl) == 96, "GCC11 function-bind task-state control");
    static_assert(sizeof(std::__future_base::_Result<void>) == 16, "GCC11 void result request");
#endif
    // Invocation captures one shared_ptr. The safe wrapper captures one function.
    // Unnamed enqueue lambdas stay a size bound; their closure types were not dumped.
    const auto invocation =
        TestFunctionTargetUpper(sizeof(std::shared_ptr<std::packaged_task<void()>>), true);
    const auto wrapper = TestFunctionTargetUpper(sizeof(std::function<void()>), false) + invocation;
    // Both packaged-task layers, both invocation targets, and both wrapper copies (D=1).
    return TestPackagedTaskRequest<Access::WorkerBind>() + TestPackagedTaskRequest<InnerBind>() +
           2 * invocation + 2 * wrapper;
}

uint64_t
TestBuildQueueRequest(uint64_t workers) {
    // Occupancy zero is the constructor's empty queue, not a missing deque.
    // Pinned deque push_back/pop_front source rules, including old/new pointer maps.
#if defined(_LIBCPP_VERSION)
    const auto elements_per_block = 4096 / sizeof(std::function<void()>);
    const auto blocks = (workers + elements_per_block - 1) / elements_per_block + 2;
    return 4096 * blocks + 2 * sizeof(void*) * std::max(uint64_t{1}, 2 * blocks);
#else
    const auto elements_per_block = 512 / sizeof(std::function<void()>);
    const auto blocks = workers / elements_per_block + 2;
    return 512 * blocks + 2 * sizeof(void*) * std::max(uint64_t{8}, 4 * blocks + 2);
#endif
}

// Doubling std::vector: the realloc that first holds `count` elements keeps the
// previous buffer. This is that peak, not a 2K reserve past the final capacity.
uint64_t
TestDoublingBufferPeak(uint64_t count, uint64_t element) {
    if (count == 0) {
        return 0;
    }
    uint64_t capacity = 1;
    while (capacity < count) {
        REQUIRE(capacity <= std::numeric_limits<uint64_t>::max() / 2);
        capacity *= 2;
    }
    const uint64_t previous = capacity == 1 ? 0 : capacity / 2;
    return (capacity + previous) * element;
}

// File-local skip strategies are a Filter base plus either two doubles or one
// generator and one double. make_unique allocates the larger derived object.
constexpr uint64_t
TestSkipStrategyUpper() {
    constexpr uint64_t alignment =
        std::max(alignof(vsag::FilterSearchSkipStrategy),
                 std::max(alignof(vsag::LinearCongruentialGenerator), alignof(double)));
    constexpr uint64_t base = TestRoundUp(sizeof(vsag::FilterSearchSkipStrategy), alignment);
    constexpr uint64_t generator =
        TestRoundUp(sizeof(vsag::LinearCongruentialGenerator), alignment);
    constexpr uint64_t fields = std::max<uint64_t>(2 * sizeof(double), generator + sizeof(double));
    return TestRoundUp(base + fields, alignment);
}

// Retained nodes and bucket pointers for the four keys sample_train_data inserts.
uint64_t
TestSampledDatasetMapBytes() {
    using Value = std::variant<int64_t,
                               const float*,
                               const char*,
                               const int8_t*,
                               const uint16_t*,
                               const int64_t*,
                               const std::string*,
                               const vsag::SparseVector*,
                               const vsag::AttributeSet*,
                               const uint32_t*,
                               const vsag::MultiVector*,
                               std::vector<std::vector<std::string>>>;
    using Map = std::unordered_map<std::string, Value>;
#if defined(_LIBCPP_VERSION)
    using Node = std::__hash_node<std::__hash_value_type<std::string, Value>, void*>;
#else
    using Node =
        std::__detail::_Hash_node<std::pair<const std::string, Value>,
                                  std::__cache_default<std::string, std::hash<std::string>>::value>;
#endif
    const char* keys[] = {vsag::NUM_ELEMENTS, vsag::DIM, vsag::FLOAT32_VECTORS, vsag::IDS};
    Map map;
    for (const char* key : keys) {
        REQUIRE(std::string(key).size() <= std::string{}.capacity());
        map.emplace(key, Value{int64_t{0}});
    }
    map[vsag::FLOAT32_VECTORS] = static_cast<const float*>(nullptr);
    map[vsag::IDS] = static_cast<const int64_t*>(nullptr);
    REQUIRE(map.size() == 4);
    return 4 * sizeof(Node) + map.bucket_count() * sizeof(void*);
}

struct StandardContribution {
    std::string name;
    uint64_t count;
    uint64_t requested_size;
    std::string source;
};

class StandardContributionTable {
public:
    void
    Add(std::string name, uint64_t count, uint64_t bytes, std::string source) {
        if (count > 0) {
            REQUIRE(bytes <= std::numeric_limits<uint64_t>::max() / count);
        }
        const auto product = count * bytes;
        REQUIRE(total_ <= std::numeric_limits<uint64_t>::max() - product);
        total_ += product;
        rows_.push_back({std::move(name), count, bytes, std::move(source)});
    }

    uint64_t
    Total() const {
        return total_;
    }

    vsag::JsonType
    Report() const {
        vsag::JsonType result;
        for (const auto& row : rows_) {
            result[row.name]["count"].SetUint64(row.count);
            result[row.name]["requested_size"].SetUint64(row.requested_size);
            result[row.name]["bytes"].SetUint64(row.count * row.requested_size);
            result[row.name]["source"].SetString(row.source);
        }
        return result;
    }

private:
    uint64_t total_{0};
    std::vector<StandardContribution> rows_;
};

void
AddOwnedString(StandardContributionTable& table,
               const std::string& name,
               const std::string& value) {
    // Fresh named string objects on these target libraries use SSO up to empty.capacity().
    const auto inline_capacity = std::string{}.capacity();
    const auto heap_bytes = value.capacity() > inline_capacity ? value.capacity() + 1 : 0;
    table.Add(name, 1, heap_bytes, "retained std::string capacity; pinned basic_string ABI");
}

template <typename Quantizer, typename QuantizerParameter>
void
AddConcreteScalarObjects(StandardContributionTable& table) {
    using Layout = vsag::FixedLayout<vsag::MemoryBlockIO>;
    using Cell = vsag::FlattenDataCell<Quantizer, Layout>;
    table.Add("flatten_cell", 1, TestSharedRequest<Cell>(), "flatten_datacell_factory_impl.h");
    table.Add("quantizer", 1, TestSharedRequest<Quantizer>(), "flatten_datacell.h:301");
    table.Add("fixed_layout", 1, TestSharedRequest<Layout>(), "flatten_datacell.h:303");
    table.Add("quantizer_parameter",
              1,
              TestSharedRequest<QuantizerParameter>(),
              "QuantizerParameter::GetQuantizerParameterByJson");
}

template <vsag::MetricType metric>
uint64_t
AddMetricScalarObjects(StandardContributionTable& table, const std::string& quantizer) {
    if (quantizer == "fp32") {
        AddConcreteScalarObjects<vsag::FP32Quantizer<metric>, vsag::FP32QuantizerParameter>(table);
        return TestSharedRequest<vsag::Computer<vsag::FP32Quantizer<metric>>>();
    }
    if (quantizer == "sq8") {
        AddConcreteScalarObjects<vsag::SQ8Quantizer<metric>, vsag::ScalarQuantizerParameter<8>>(
            table);
        return TestSharedRequest<vsag::Computer<vsag::SQ8Quantizer<metric>>>();
    }
    REQUIRE(quantizer == "sq8_uniform");
    AddConcreteScalarObjects<vsag::SQ8UniformQuantizer<metric>, vsag::SQ8UniformQuantizerParameter>(
        table);
    return TestSharedRequest<vsag::Computer<vsag::SQ8UniformQuantizer<metric>>>();
}

StandardContributionTable
IndependentStandardEnvelope(const vsag::HGraph& graph,
                            const std::string& quantizer,
                            uint64_t dim,
                            uint64_t degree,
                            uint64_t count,
                            uint64_t sample_cap,
                            uint64_t threads,
                            bool has_pool,
                            vsag::MetricType metric) {
    using namespace vsag;
    using Access = HGraphMemoryEstimateTestAccess;
    const auto& param = Access::Parameters(graph);
    REQUIRE(param.resize_increase_count_bit == 10);
    REQUIRE(param.train_sample_count == static_cast<int64_t>(std::max(uint64_t{512}, sample_cap)));
    const auto initial_capacity = Access::InitialCapacity(graph);
    const auto capacity = std::max(initial_capacity, TestRoundUp(count, uint64_t{1} << 10));
    // A supplied pool is caller-owned. The index creates a pool only when build
    // threads exceed one and the caller passed none.
    const bool owns_pool = !has_pool && threads > 1;
    const bool scheduled = has_pool || owns_pool;
    uint64_t workers = 1;
    if (count == 0) {
        workers = 0;
    } else if (scheduled) {
        workers = std::min(count, std::max(uint64_t{1}, threads));
    }
    const auto sample = std::min(count, std::max(uint64_t{512}, sample_cap));
    const auto trainer_sample = std::min(sample, uint64_t{100000});
    StandardContributionTable table;
    table.Add(
        "public_index", 1, TestSharedRequest<IndexImpl<HGraph>>(), "index_creators/IndexImpl");
    table.Add("inner_hgraph", 1, TestSharedRequest<HGraph>(), "index_impl.h:58");
    table.Add("build_cache", 1, sizeof(BuildCache), "hgraph.cpp constructor make_unique");
    table.Add("feature_list_object", 1, sizeof(IndexFeatureList), "inner_index_interface.cpp:76");
    table.Add("feature_list_data", 1, (INDEX_FEATURE_COUNT + 7) / 8, "index_feature_list.cpp:37");
    table.Add("label_table", 1, TestSharedRequest<LabelTable>(), "inner_index_interface.cpp:73");
    table.Add("reverse_map_object",
              1,
              sizeof(PGUnorderedMap<LabelType, InnerIdType>),
              "label_remap.cpp:35; bucket arrays measured through Allocator");
    table.Add("deletion_filter_upper",
              1,
              TestSharedRequest<RemoveListFilterStorageUpper>(),
              "label_table.cpp RemoveListFilter; actual target layout check required");
    table.Add("point_mutex_owner", 1, TestSharedRequest<PointsMutex>(), "hgraph.cpp:108");
    table.Add("basic_searcher", 1, TestSharedRequest<BasicSearcher>(), "hgraph.cpp:115");
    table.Add("mci_searcher", 1, TestSharedRequest<MCISearcher>(), "hgraph.cpp:116");
    table.Add("hybrid_searcher", 1, TestSharedRequest<HybridMCISearcher>(), "hgraph.cpp:117");
    table.Add("parallel_searcher", 1, TestSharedRequest<ParallelSearcher>(), "hgraph.cpp:156");
    table.Add("bottom_graph",
              1,
              TestSharedRequest<GraphDataCell<MemoryBlockIO>>(),
              "GraphInterface::MakeInstance");
    table.Add("block_io_objects",
              2,
              TestSharedRequest<MemoryBlockIO>(),
              "graph_datacell.h and fixed_layout.h; code and graph IO");
    uint64_t computer_bytes = 0;
    switch (metric) {
        case MetricType::METRIC_TYPE_L2SQR:
            computer_bytes =
                AddMetricScalarObjects<MetricType::METRIC_TYPE_L2SQR>(table, quantizer);
            break;
        case MetricType::METRIC_TYPE_IP:
            computer_bytes = AddMetricScalarObjects<MetricType::METRIC_TYPE_IP>(table, quantizer);
            break;
        case MetricType::METRIC_TYPE_COSINE:
            computer_bytes =
                AddMetricScalarObjects<MetricType::METRIC_TYPE_COSINE>(table, quantizer);
            break;
        default:
            FAIL("An unlisted metric needs an independent concrete-type table");
    }
    table.Add(
        "hgraph_parameters", 1, TestSharedRequest<HGraphParameter>(), "HGraphParameter factory");
    table.Add("parameter_pipnn_fanout",
              param.pipnn_param.fanout.capacity(),
              sizeof(uint64_t),
              "pipnn_graph_builder.h fanout initializer; retained parameter buffer");
    table.Add("hgraph_pipnn_fanout",
              Access::RetainedFanoutCapacity(graph),
              sizeof(uint64_t),
              "hgraph.cpp constructor copies pipnn_param_; retained index buffer");
    table.Add("flatten_parameters",
              1,
              TestSharedRequest<FlattenDataCellParameter>(),
              "hgraph_parameter.cpp base_codes_param");
    table.Add("graph_parameters",
              1,
              TestSharedRequest<GraphDataCellParameter>(),
              "GraphInterfaceParameter::GetGraphParameterByJson");
    table.Add("route_parameters",
              1,
              TestSharedRequest<SparseGraphDatacellParameter>(),
              "hgraph_parameter.cpp hierarchical_graph_param");
    table.Add("io_parameters",
              2,
              TestSharedRequest<MemoryBlockIOParameter>(),
              "base and graph MemoryBlockIOParameter factories");
    REQUIRE(param.extra_info_param != nullptr);
    table.Add("extra_info_parameters",
              1,
              TestSharedRequest<ExtraInfoDataCellParameter>(),
              "inner_index_parameter.cpp:265, even when extra-info size is zero");
    table.Add("extra_info_io_parameters",
              1,
              TestSharedRequest<MemoryBlockIOParameter>(),
              "extra_info_datacell_parameter.cpp default IO");
    AddOwnedString(table, "param_name", param.name);
    AddOwnedString(table, "param_graph_type", param.graph_type);
    AddOwnedString(table, "param_reorder_source", param.reorder_source);
    AddOwnedString(table, "param_mci_source", param.mci_parameters.knng_source);
    AddOwnedString(table, "param_mci_path", param.mci_parameters.knng_path);
    AddOwnedString(table, "flatten_name", param.base_codes_param->name);
    AddOwnedString(table, "base_io_name", param.base_codes_param->io_parameter->GetTypeName());
    auto graph_parameter =
        std::dynamic_pointer_cast<GraphDataCellParameter>(param.bottom_graph_param);
    REQUIRE(graph_parameter != nullptr);
    AddOwnedString(table, "graph_io_name", graph_parameter->io_parameter_->GetTypeName());
    AddOwnedString(table, "extra_io_name", param.extra_info_param->io_parameter->GetTypeName());
    // Copied graph/MCI names and the scalar quantizer type names are SSO in this configuration.
    REQUIRE(param.graph_type.size() <= std::string{}.capacity());
    REQUIRE(param.mci_parameters.knng_source.size() <= std::string{}.capacity());
    REQUIRE(param.base_codes_param->quantizer_parameter->GetTypeName().size() <=
            std::string{}.capacity());

    const uint64_t pool_copies = capacity > initial_capacity ? 2 : 1;
    const uint64_t lists = (count == 0 ? 1 : 16 * std::max(uint64_t{1}, workers)) + pool_copies - 1;
    table.Add("visited_pool_objects",
              pool_copies,
              TestSharedRequest<VisitedListPool>(),
              "hgraph_build.cpp resize; old/new pool coexist only on replacement");
    table.Add("visited_deque_objects",
              pool_copies * 16,
              sizeof(Deque<std::shared_ptr<VisitedList>>),
              "resource_object_pool.h make_unique; elements/maps measured through Allocator");
    const auto pool_capture = sizeof(VisitedListPool*) + sizeof(uint64_t) + sizeof(Allocator*);
#if defined(_LIBCPP_VERSION)
    const auto pool_function = TestRoundUp(sizeof(void*) + pool_capture, alignof(void*));
#else
    const auto pool_function = pool_capture;
#endif
    table.Add("visited_constructor_function",
              pool_copies,
              pool_function,
              "resource_object_pool.h constructor: pool pointer, size, allocator captures");
    table.Add("visited_list_objects",
              lists,
              TestSharedRequest<VisitedList>(),
              "resource_object_pool.h make_shared; bit arrays measured through Allocator");

    // Independent replay at the actual degree. Production deliberately replays at degree four.
    std::array<uint64_t, 64> route_counts{};
    auto engine = Access::Generator(graph);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const auto multiplier = 1.0 / std::log(static_cast<double>(degree));
    for (uint64_t row = 0; row < count; ++row) {
        const auto probability = uniform(engine);
        REQUIRE(probability > 0.0);
        const auto draw = -std::log(probability) * multiplier;
        REQUIRE(std::isfinite(draw));
        REQUIRE(draw >= 0.0);
        REQUIRE(draw < static_cast<double>(route_counts.size()));
        for (uint64_t level = 0; level < static_cast<uint64_t>(draw); ++level) {
            ++route_counts[level];
        }
    }
    for (uint64_t level = 0; level < route_counts.size(); ++level) {
        if (route_counts[level] == 0) {
            continue;
        }
        const auto suffix = std::to_string(level);
        table.Add("route_graph_" + suffix,
                  1,
                  TestSharedRequest<SparseGraphDataCell>(),
                  "hgraph.cpp generate_one_route_graph");
        table.Add("route_vector_headers_" + suffix,
                  route_counts[level],
                  sizeof(Vector<InnerIdType>),
                  "sparse_graph_datacell.cpp make_unique; bucket/neighbor arrays measured");
    }
    table.Add("retained_sq_arrays",
              quantizer == "sq8" ? 2 : 0,
              dim * sizeof(float),
              "scalar_quantizer.cpp:35-36, actual quantizer behavior");
    table.Add("trainer_sample",
              count > 0 && quantizer != "fp32" ? 1 : 0,
              trainer_sample * dim * sizeof(float),
              "scalar_quantization_trainer.cpp sample_datas");
    table.Add("uniform_training_bounds",
              count > 0 && quantizer == "sq8_uniform" ? 2 : 0,
              dim * sizeof(float),
              "scalar_quantization_trainer.cpp TrainUniform upper/lower");
    uint64_t truncation_bytes = 0;
    if (count > 0 && quantizer == "sq8") {
        const auto ignore = std::max<uint64_t>(
            1, static_cast<uint64_t>(static_cast<double>(trainer_sample - 1) * 0.001));
        truncation_bytes = 2 * TestDoublingBufferPeak(ignore + 1, sizeof(float));
    }
    table.Add("bit8_truncation_heaps",
              truncation_bytes > 0 ? 1 : 0,
              truncation_bytes,
              "scalar_quantization_trainer.cpp:93 bit8 ignore 0.001; two heap buffers");
    table.Add("sampled_dataset",
              sample < count ? 1 : 0,
              TestSharedRequest<DatasetImpl>(),
              "util_functions.cpp sampled_dataset; input dataset excluded");
    table.Add("sampled_dataset_map",
              sample < count ? 1 : 0,
              sample < count ? TestSampledDatasetMapBytes() : 0,
              "util_functions.cpp four keys: num_elements, dim, f32_vectors, ids");
    // At nested route pruning, five shared objects survive together: the bottom probe heap,
    // route result, route filtered result, neighbor candidates and nested queue_closest.
    // Emptying either retained result does not destroy its shared object.
    table.Add(
        "worker_heap_objects",
        5 * workers,
        TestSharedRequest<StandardHeap<true, false>>(),
        "hgraph_build.cpp probe/route result/filtered; pruning_strategy.cpp candidates/closest");
    table.Add("worker_computer_objects",
              workers,
              computer_bytes,
              "Quantizer::FactoryComputer; no transforms or reorder cells");
    table.Add("skip_strategy_upper",
              workers,
              TestSkipStrategyUpper(),
              "basic_searcher.cpp skip strategy; file-local derived object upper bound");
    table.Add("failed_ids",
              0,
              sizeof(int64_t),
              "AddBatch.failed_ids remains empty for successful unique labels");
    table.Add("scheduler_chains",
              scheduled ? workers : 0,
              TestSchedulerTaskRequest(),
              "actual InsertAddBatchWorker bind and two packaged-task layers, D=1");
    table.Add("scheduler_futures",
              scheduled ? workers : 0,
              sizeof(std::future<void>),
              "run_insert_add_batch_workers reserve W; future object only");
    table.Add("scheduler_queue",
              scheduled && workers > 0 ? 1 : 0,
              TestBuildQueueRequest(workers),
              "fresh built-in deque/map source envelope; pre-existing workers excluded");
    uint64_t owned_pool_bytes = 0;
    if (owns_pool) {
        const uint64_t started = std::max(threads, Options::Instance().num_threads_building());
        owned_pool_bytes = TestSharedRequest<SafeThreadPool>() + sizeof(DefaultThreadPool) +
                           sizeof(progschj::ThreadPool) +
                           TestDoublingBufferPeak(started, sizeof(std::thread));
        if (count == 0) {
            owned_pool_bytes += TestBuildQueueRequest(0);
        }
    }
    table.Add("owned_constructor_pool",
              owns_pool ? 1 : 0,
              owned_pool_bytes,
              "inner_index_interface.cpp:86; only when threads exceed one and no pool is supplied");
    return table;
}

void
RunBuildMemoryCell(const std::string& quantizer,
                   uint64_t dim,
                   uint64_t degree,
                   bool parallel,
                   uint64_t count,
                   uint64_t sample_cap,
                   bool mixed_labels = false,
                   vsag::MetricType metric = vsag::MetricType::METRIC_TYPE_L2SQR,
                   bool primary_cell = false) {
    BuildMemoryData data(count, dim);
    if (mixed_labels) {
        for (uint64_t row = 0; row < count; ++row) {
            data.ids[row] =
                row % 2 == 0 ? -static_cast<int64_t>(row / 2) - 1 : static_cast<int64_t>(row / 2);
        }
        std::reverse(data.ids.begin(), data.ids.end());
        data.RefreshChecksum();
    }
    auto recorder = std::make_shared<PhaseCountingAllocator>();
    auto common = MakeCommonParam(static_cast<int64_t>(dim));
    common.allocator_ = recorder;
    common.metric_ = metric;
    if (parallel) {
        common.thread_pool_ =
            std::make_shared<vsag::SafeThreadPool>(std::make_shared<vsag::DefaultThreadPool>(8));
    }
    const uint64_t threads = parallel ? 4 : 1;
    auto index =
        MakeHGraphIndex(MakeBuildMemoryJson(quantizer, degree, threads, sample_cap), common);
    const auto standard = IndependentStandardEnvelope(
        *InnerHGraph(index), quantizer, dim, degree, count, sample_cap, threads, parallel, metric);
    recorder->BeginBuildPhase();
    const auto initial = recorder->Read();
    REQUIRE(initial.current > 0);
    REQUIRE(initial.phase_peak == initial.current);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ESTIMATE_BUILD_MEMORY));
    const auto estimate = index->EstimateBuildMemory(count);
    const auto steady = index->EstimateMemory(count);
    REQUIRE(index->EstimateBuildMemory(count) == estimate);
    REQUIRE(recorder->Read().requests == initial.requests);
    REQUIRE(recorder->Read().current == initial.current);
    REQUIRE(recorder->Read().phase_peak == initial.current);
    const auto built = index->Build(data.dataset);
    const auto measured = recorder->Read();
    vsag::JsonType report;
    report["primary_matrix_cell"].SetBool(primary_cell);
    report["metric"].SetInt(static_cast<int>(metric));
    report["compiler"].SetString(__VERSION__);
    report["block_limit"].SetUint64(vsag::Options::Instance().block_size_limit());
    report["ef_construction"].SetUint64(64);
    report["support_duplicate"].SetBool(false);
    report["deduplicate_storage"].SetBool(false);
    report["quantizer"].SetString(quantizer);
    report["dim"].SetUint64(dim);
    report["degree"].SetUint64(degree);
    report["requested_threads"].SetUint64(threads);
    report["pool_width"].SetUint64(parallel ? 8 : 0);
    report["count"].SetUint64(count);
    report["sample_cap"].SetUint64(std::max(uint64_t{512}, sample_cap));
    report["checksum"].SetUint64(data.checksum);
    report["mixed_labels"].SetBool(mixed_labels);
    report["A0"].SetUint64(initial.current);
    report["constructor_peak"].SetUint64(initial.constructor_peak);
    report["Aphasepeak"].SetUint64(measured.phase_peak);
    report["Afinal"].SetUint64(measured.current);
    report["Utest"].SetUint64(standard.Total());
    report["contributions"].SetJson(standard.Report());
    report["steady"].SetUint64(steady);
    report["peak_estimate"].SetUint64(estimate);
    report["build_success"].SetBool(built.has_value());
    report["inserted"].SetInt(index->GetNumElements());
#if defined(_LIBCPP_VERSION)
    report["stdlib"].SetString("libc++");
    report["stdlib_version"].SetUint64(_LIBCPP_VERSION);
#else
    report["stdlib"].SetString("libstdc++ headers");
    report["stdlib_version"].SetUint64(__GLIBCXX__);
#endif
    std::cout << "HGRAPH_BUILD_MEMORY_CELL " << report.Dump() << std::endl;
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    REQUIRE(index->GetNumElements() == static_cast<int64_t>(count));
    REQUIRE(estimate >= steady);
    REQUIRE(measured.phase_peak <= std::numeric_limits<uint64_t>::max() - standard.Total());
    REQUIRE(estimate >= measured.phase_peak + standard.Total());
    index.reset();
    REQUIRE(recorder->Read().current == 0);
}

#endif  // HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

void
AppendCacheWord(std::string& bytes, uint64_t value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

class ComponentFaultReader : public vsag::StreamReader, public vsag::DeserializeReader {
public:
    explicit ComponentFaultReader(std::string bytes) : bytes_(std::move(bytes)) {
        vsag::ReadFuncStreamReader footer_reader(
            [this](uint64_t offset, uint64_t size, void* dest) { Copy(offset, size, dest); },
            0,
            bytes_.size());
        auto footer = vsag::Footer::Parse(footer_reader);
        REQUIRE(footer != nullptr);
        body_end_ = bytes_.size() - footer->Length();
        REQUIRE(body_end_ > 0);
    }

    uint64_t
    Size() const override {
        return bytes_.size();
    }

    uint64_t
    Length() override {
        return bytes_.size();
    }

    uint64_t
    GetCursor() const override {
        return cursor_;
    }

    void
    Seek(uint64_t cursor) override {
        if (cursor > bytes_.size()) {
            throw vsag::VsagException(vsag::ErrorType::INVALID_BINARY, "controlled seek bounds");
        }
        cursor_ = cursor;
    }

    void
    Read(char* dest, uint64_t size) override {
        Read(cursor_, size, dest);
        cursor_ += size;
    }

    void
    Read(uint64_t offset, uint64_t size, void* dest) override {
        if (size > 0 && offset < body_end_) {
            component_read_.store(true);
            throw vsag::VsagException(vsag::ErrorType::INVALID_BINARY,
                                      "controlled component-body read failure");
        }
        Copy(offset, size, dest);
    }

    bool
    ReachedComponent() const {
        return component_read_.load();
    }

private:
    void
    Copy(uint64_t offset, uint64_t size, void* dest) {
        if (size > bytes_.size() || offset > bytes_.size() - size) {
            throw vsag::VsagException(vsag::ErrorType::INVALID_BINARY, "controlled read bounds");
        }
        if (size > 0) {
            std::memcpy(dest, bytes_.data() + offset, size);
        }
    }

    std::string bytes_;
    uint64_t cursor_{0};
    uint64_t body_end_{0};
    std::atomic<bool> component_read_{false};
};

struct ControlledRowFailure : public std::exception {
    explicit ControlledRowFailure(uint64_t row) : row(row) {
    }
    const char*
    what() const noexcept override {
        return "controlled row failure";
    }
    uint64_t row;
};

struct ControlledRows {
    ControlledRows() {
        for (auto& counter : attempts) {
            counter.store(0);
        }
    }

    std::array<std::atomic<uint64_t>, 7> attempts{};
    std::promise<void> release_low;
    std::shared_future<void> low_gate{release_low.get_future().share()};

    static void
    Invoke(void* opaque, uint64_t row) {
        auto& state = *static_cast<ControlledRows*>(opaque);
        ++state.attempts[row];
        if (row == 0) {
            state.low_gate.wait();
            throw ControlledRowFailure(0);
        }
        if (row == 1) {
            throw ControlledRowFailure(1);
        }
    }
};

}  // namespace

TEST_CASE("HGraph build memory validates the profile and reports modeled ownership cells",
          "[ut][hgraph][estimate_build_memory][estimate_build_memory_matrix]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto profile_index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    RequireExpectedBuildMemoryProfile(profile_index);
#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE
    uint64_t reported = 0;
    for (const auto* quantizer : {"fp32", "sq8", "sq8_uniform"}) {
        for (uint64_t dim : {uint64_t{16}, uint64_t{128}}) {
            for (uint64_t degree : {uint64_t{8}, uint64_t{32}}) {
                for (bool parallel : {false, true}) {
                    {
                        INFO(quantizer << "/d" << dim << "/M" << degree << "/parallel" << parallel);
                        RunBuildMemoryCell(quantizer,
                                           dim,
                                           degree,
                                           parallel,
                                           1537,
                                           1537,
                                           false,
                                           vsag::MetricType::METRIC_TYPE_L2SQR,
                                           true);
                        ++reported;
                    }
                }
            }
        }
    }
    REQUIRE(reported == 24);
#endif
}

#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

TEST_CASE("HGraph build memory retains zero baseline and does not allocate while estimating",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto recorder = std::make_shared<PhaseCountingAllocator>();
    auto common = MakeCommonParam(16);
    common.allocator_ = recorder;
    auto index = MakeHGraphIndex(MakeBuildMemoryJson(), common);
    recorder->BeginBuildPhase();
    const auto before = recorder->Read();
    REQUIRE(before.current > 0);
    REQUIRE(before.constructor_peak >= before.current);
    REQUIRE(index->EstimateMemory(0) == 0);
    REQUIRE(index->EstimateBuildMemory(0) >= before.current);
    for (uint64_t count : {uint64_t{0}, uint64_t{1}, uint64_t{1537}, uint64_t{65537}}) {
        const auto estimate = index->EstimateBuildMemory(count);
        REQUIRE(estimate >= index->EstimateMemory(count));
        REQUIRE(index->EstimateBuildMemory(count) == estimate);
    }
    const auto after = recorder->Read();
    REQUIRE(after.requests == before.requests);
    REQUIRE(after.current == before.current);
    REQUIRE(after.phase_peak == before.phase_peak);
    auto empty = vsag::Dataset::Make();
    REQUIRE(index->Build(empty).has_value());
    REQUIRE(index->GetNumElements() == 0);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ESTIMATE_BUILD_MEMORY));
    REQUIRE(recorder->Read().current == before.current);
    const auto standard = IndependentStandardEnvelope(
        *InnerHGraph(index), "fp32", 16, 8, 0, 1537, 1, false, vsag::MetricType::METRIC_TYPE_L2SQR);
    REQUIRE(index->EstimateBuildMemory(0) >= before.current + standard.Total());
    index.reset();
    REQUIRE(recorder->Read().current == 0);
}

TEST_CASE("HGraph build memory covers sampling and storage boundaries",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    for (uint64_t count : {uint64_t{1},
                           uint64_t{511},
                           uint64_t{512},
                           uint64_t{513},
                           uint64_t{1023},
                           uint64_t{1024},
                           uint64_t{1025},
                           uint64_t{2048}}) {
        {
            INFO("sample-cap-512/N" << count);
            RunBuildMemoryCell("sq8", 16, 8, false, count, 512);
        }
    }
    // FP32 d=128 crosses a 256-KiB code block at row 513.
    for (uint64_t count : {uint64_t{511}, uint64_t{512}, uint64_t{513}}) {
        {
            INFO("FP32 block crossing/N" << count);
            RunBuildMemoryCell("fp32", 128, 8, false, count, count);
        }
    }
    for (uint64_t count : {uint64_t{1024}, uint64_t{1025}}) {
        {
            INFO("graph block crossing/N" << count);
            RunBuildMemoryCell("fp32", 16, 32, false, count, count);
        }
    }
    RunBuildMemoryCell("sq8_uniform", 16, 8, false, 513, 512, true);
}

TEST_CASE("HGraph build memory uses stored blocks and monotone count dimension degree threads",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto common = MakeCommonParam(16);
    auto index = MakeHGraphIndex(MakeBuildMemoryJson(), common);
    const auto fixed_geometry = index->EstimateBuildMemory(1537);
    {
        BlockSizeLimitGuard changed_global(uint64_t{1} << 19);
        REQUIRE(index->EstimateBuildMemory(1537) >= index->EstimateMemory(1537));
        // The steady floor reads Options. Restoring it must restore the estimate.
    }
    REQUIRE(index->EstimateBuildMemory(1537) == fixed_geometry);
    auto larger_blocks = [&] {
        BlockSizeLimitGuard different_geometry(uint64_t{1} << 19);
        return MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    }();
    REQUIRE(larger_blocks->EstimateBuildMemory(1) > index->EstimateBuildMemory(1));

    uint64_t previous = 0;
    for (uint64_t count : {uint64_t{0},
                           uint64_t{1},
                           uint64_t{511},
                           uint64_t{512},
                           uint64_t{513},
                           uint64_t{1023},
                           uint64_t{1024},
                           uint64_t{1025},
                           uint64_t{1537},
                           uint64_t{2048},
                           uint64_t{65537}}) {
        const auto current = index->EstimateBuildMemory(count);
        CAPTURE(count, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    previous = 0;
    for (uint64_t dim : {uint64_t{8},
                         uint64_t{15},
                         uint64_t{16},
                         uint64_t{17},
                         uint64_t{31},
                         uint64_t{32},
                         uint64_t{33},
                         uint64_t{128}}) {
        auto candidate = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(dim));
        const auto current = candidate->EstimateBuildMemory(1537);
        CAPTURE(dim, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    previous = 0;
    for (uint64_t degree :
         {uint64_t{4}, uint64_t{5}, uint64_t{8}, uint64_t{16}, uint64_t{32}, uint64_t{64}}) {
        auto candidate = MakeHGraphIndex(MakeBuildMemoryJson("sq8", degree), common);
        const auto current = candidate->EstimateBuildMemory(1537);
        CAPTURE(degree, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    auto pool_common = MakeCommonParam(16);
    pool_common.thread_pool_ =
        std::make_shared<vsag::SafeThreadPool>(std::make_shared<vsag::DefaultThreadPool>(8));
    previous = 0;
    for (uint64_t threads : {uint64_t{1}, uint64_t{2}, uint64_t{4}, uint64_t{8}}) {
        auto candidate = MakeHGraphIndex(MakeBuildMemoryJson("sq8", 8, threads), pool_common);
        const auto current = candidate->EstimateBuildMemory(1537);
        CAPTURE(threads, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    // Repeat independent axes with a real supplied pool and fixed requested T=4.
    auto parallel_index = MakeHGraphIndex(MakeBuildMemoryJson("fp32", 8, 4), pool_common);
    previous = 0;
    for (uint64_t count : {uint64_t{0},
                           uint64_t{1},
                           uint64_t{511},
                           uint64_t{512},
                           uint64_t{513},
                           uint64_t{1023},
                           uint64_t{1024},
                           uint64_t{1025},
                           uint64_t{1537},
                           uint64_t{2048},
                           uint64_t{65537}}) {
        const auto current = parallel_index->EstimateBuildMemory(count);
        CAPTURE(count, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    previous = 0;
    for (uint64_t dim : {uint64_t{8},
                         uint64_t{15},
                         uint64_t{16},
                         uint64_t{17},
                         uint64_t{31},
                         uint64_t{32},
                         uint64_t{33},
                         uint64_t{128}}) {
        auto dimensional = pool_common;
        dimensional.dim_ = static_cast<int64_t>(dim);
        auto candidate = MakeHGraphIndex(MakeBuildMemoryJson("fp32", 8, 4), dimensional);
        const auto current = candidate->EstimateBuildMemory(1537);
        CAPTURE(dim, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    previous = 0;
    for (uint64_t degree :
         {uint64_t{4}, uint64_t{5}, uint64_t{8}, uint64_t{16}, uint64_t{32}, uint64_t{64}}) {
        auto candidate = MakeHGraphIndex(MakeBuildMemoryJson("sq8", degree, 4), pool_common);
        const auto current = candidate->EstimateBuildMemory(1537);
        CAPTURE(degree, previous, current);
        REQUIRE(current >= previous);
        previous = current;
    }
    for (bool parallel : {false, true}) {
        const auto& selected_common = parallel ? pool_common : common;
        auto fp32 = MakeHGraphIndex(MakeBuildMemoryJson("fp32"), selected_common);
        for (const auto* quantizer : {"sq8", "sq8_uniform"}) {
            auto scalar = MakeHGraphIndex(MakeBuildMemoryJson(quantizer), selected_common);
            REQUIRE(fp32->EstimateBuildMemory(1537) >= scalar->EstimateBuildMemory(1537));
        }
    }
}

#endif  // HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

TEST_CASE("HGraph build memory rejects invalid rounded capacity before replay",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    auto graph = InnerHGraph(index);
    RequireExpectedBuildMemoryProfile(index);
    constexpr auto expected_error = EXPECT_MODELED_BUILD_MEMORY
                                        ? vsag::ErrorType::INVALID_ARGUMENT
                                        : vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION;
    constexpr uint64_t last_valid = (uint64_t{1} << 31) - 1024;
    REQUIRE(vsag::HGraphMemoryEstimateTestAccess::RoundedCapacity(*graph, last_valid) ==
            last_valid);
    for (uint64_t count :
         {last_valid + 1, uint64_t{1} << 31, std::numeric_limits<uint64_t>::max()}) {
        RequireEstimateError([&] { (void)index->EstimateBuildMemory(count); }, expected_error);
    }
    // The capacity guard dominates later N-only arithmetic overflows in this API.
    // V2's independent steady-estimator arithmetic case remains unchanged below.
}

TEST_CASE("HGraph build memory rejects unsupported configurations including zero",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    for (const auto* option : {"support_duplicate",
                               "use_reorder",
                               "use_elp_optimizer",
                               "store_raw_vector",
                               "support_force_remove",
                               "use_attribute_filter",
                               "use_conjugate_graph",
                               "use_mci"}) {
        DYNAMIC_SECTION(option) {
            auto json = MakeBuildMemoryJson();
            json[option].SetBool(true);
            if (std::string(option) == "use_reorder") {
                json["precise_quantization_type"].SetString("fp32");
            }
            RequireUnsupportedBuildEstimate(MakeHGraphIndex(json, MakeCommonParam(128)));
        }
    }
    // persist_source_id is stored, not rejected. The modeled input is vectors plus
    // numeric labels and carries no source-ID payload.
    {
        auto json = MakeBuildMemoryJson();
        json["persist_source_id"].SetBool(true);
        auto indexed = MakeHGraphIndex(json, MakeCommonParam(128));
        RequireExpectedBuildMemoryProfile(indexed);
        BuildMemoryData labeled(1, 128);
        REQUIRE(labeled.dataset->GetSourceID() == nullptr);
    }
    for (const auto* quantizer : {"sq4", "rabitq"}) {
        DYNAMIC_SECTION(quantizer) {
            RequireUnsupportedBuildEstimate(
                MakeHGraphIndex(MakeBuildMemoryJson(quantizer), MakeCommonParam(128)));
        }
    }
    for (const auto* graph_type : {"odescent", "pipnn"}) {
        DYNAMIC_SECTION(graph_type) {
            auto json = MakeBuildMemoryJson();
            json["graph_type"].SetString(graph_type);
            RequireUnsupportedBuildEstimate(MakeHGraphIndex(json, MakeCommonParam(128)));
        }
    }
    auto deduplicated = MakeBuildMemoryJson();
    deduplicated["support_duplicate"].SetBool(true);
    deduplicated["deduplicate_storage"].SetBool(true);
    RequireUnsupportedBuildEstimate(MakeHGraphIndex(deduplicated, MakeCommonParam(128)));
    auto resize_limit = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    vsag::HGraphMemoryEstimateTestAccess::SetUnsupportedResizeBit(*InnerHGraph(resize_limit));
    RequireUnsupportedBuildEstimate(resize_limit);
    auto extra_common = MakeCommonParam(128);
    extra_common.extra_info_size_ = 8;
    RequireUnsupportedBuildEstimate(MakeHGraphIndex(MakeBuildMemoryJson(), extra_common));
    for (const auto* io_slot : {"base_io_type", "graph_io_type"}) {
        auto memory_io = MakeBuildMemoryJson();
        memory_io[io_slot].SetString("memory_io");
        RequireUnsupportedBuildEstimate(MakeHGraphIndex(memory_io, MakeCommonParam(128)));
    }
    auto compressed = MakeBuildMemoryJson();
    compressed["graph_storage_type"].SetString("compressed");
    RequireUnsupportedBuildEstimate(MakeHGraphIndex(compressed, MakeCommonParam(128)));
}

TEST_CASE("HGraph build memory rejects nonempty immutable and Tune states",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    SECTION("successful Build changes active capability") {
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
        BuildMemoryData data(3, 16);
        RequireExpectedBuildMemoryProfile(index);
        REQUIRE(index->Build(data.dataset).has_value());
        REQUIRE(index->GetNumElements() == 3);
        RequireUnsupportedBuildEstimate(index);
    }
    SECTION("immutable empty index") {
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
        REQUIRE(index->SetImmutable().has_value());
        RequireUnsupportedBuildEstimate(index);
    }
    SECTION("Tune entry at count zero invalidates the lifecycle") {
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
        REQUIRE(index->GetNumElements() == 0);
        (void)index->Tune("{}");
        REQUIRE_FALSE(vsag::HGraphMemoryEstimateTestAccess::LifecycleClean(*InnerHGraph(index)));
        RequireUnsupportedBuildEstimate(index);
    }
}

TEST_CASE("HGraph bounded Build dispatch gates requested workers on a wider pool",
          "[ut][hgraph][build_dispatch]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    const std::array<std::pair<uint64_t, uint64_t>, 3> cells{{{1, 17}, {4, 17}, {4, 2}}};
    for (const auto& cell : cells) {
        const auto threads = cell.first;
        const auto count = cell.second;
        DYNAMIC_SECTION("T" << threads << "/N" << count) {
            auto pool = std::make_shared<GatedBuildPool>();
            BuildPoolReleaseGuard release(pool);
            auto common = MakeCommonParam(16);
            common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
            auto index = MakeHGraphIndex(MakeBuildMemoryJson("fp32", 8, threads), common);
            BuildMemoryData data(count, 16);
            auto build = std::async(std::launch::async, [&] { return index->Build(data.dataset); });
            const auto workers = std::min(threads, count);
            const bool gated = pool->AwaitStarted(workers);
            pool->Release();
            auto result = build.get();
            pool->WaitUntilEmpty();
            REQUIRE(pool->AwaitFinished(workers));
            REQUIRE(gated);
            REQUIRE(result.has_value());
            REQUIRE(result.value().empty());
            REQUIRE(index->GetNumElements() == static_cast<int64_t>(count));
            const auto counters = pool->Counts();
            REQUIRE(counters[0] == workers);
            REQUIRE(counters[1] == workers);
            REQUIRE(counters[2] == workers);
            vsag::JsonType report;
            report["requested_threads"].SetUint64(threads);
            report["pool_width"].SetUint64(8);
            report["rows"].SetUint64(count);
            report["submitted"].SetUint64(counters[0]);
            report["finished"].SetUint64(counters[1]);
            report["gated_overlap"].SetUint64(counters[2]);
            std::cout << "HGRAPH_BUILD_DISPATCH_CELL " << report.Dump() << std::endl;
        }
    }
}

TEST_CASE("HGraph bounded dispatch drains launched work after enqueue failure",
          "[ut][hgraph][build_dispatch]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto pool = std::make_shared<GatedBuildPool>(1);
    BuildPoolReleaseGuard release(pool);
    auto common = MakeCommonParam(16);
    common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    auto index = MakeHGraphIndex(MakeBuildMemoryJson("fp32", 8, 4), common);
    BuildMemoryData data(17, 16);
    auto build = std::async(std::launch::async, [&] { return index->Build(data.dataset); });
    const bool gated = pool->AwaitStarted(1);
    const bool failed_submission = pool->AwaitSubmissionFailure();
    const bool still_waiting =
        build.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    pool->Release();
    auto result = build.get();
    pool->WaitUntilEmpty();
    REQUIRE(pool->AwaitFinished(1));
    REQUIRE(gated);
    REQUIRE(failed_submission);
    REQUIRE(still_waiting);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(pool->Counts()[0] == 2);
    REQUIRE(pool->Counts()[1] == 1);
}

TEST_CASE("HGraph bounded dispatch also applies to unsupported-estimator Add",
          "[ut][hgraph][build_dispatch]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto pool = std::make_shared<GatedBuildPool>();
    BuildPoolReleaseGuard release(pool);
    auto common = MakeCommonParam(16);
    common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    auto json = MakeBuildMemoryJson("fp32", 8, 4);
    json["support_duplicate"].SetBool(true);
    auto index = MakeHGraphIndex(json, common);
    RequireUnsupportedBuildEstimate(index);
    BuildMemoryData data(17, 16);
    auto add = std::async(std::launch::async, [&] { return index->Add(data.dataset); });
    const bool gated = pool->AwaitStarted(4);
    pool->Release();
    auto result = add.get();
    pool->WaitUntilEmpty();
    REQUIRE(pool->AwaitFinished(4));
    REQUIRE(gated);
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    REQUIRE(index->GetNumElements() == 17);
    REQUIRE(pool->Counts()[0] == 4);
}

TEST_CASE("HGraph build memory notices real cache entries and retained empty cache storage",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    const auto fresh = vsag::HGraphMemoryEstimateTestAccess::CacheStorage(*InnerHGraph(index));
    REQUIRE(fresh[0] == 0);
    REQUIRE(fresh[1] == 0);
    std::string empty_bytes;
    AppendCacheWord(empty_bytes, 0);
    std::stringstream empty_first(empty_bytes);
    REQUIRE(index->ImportCache(empty_first).has_value());
    RequireExpectedBuildMemoryProfile(index);

    std::string live_bytes;
    AppendCacheWord(live_bytes, 1);
    AppendCacheWord(live_bytes, 1);
    live_bytes += "a";
    AppendCacheWord(live_bytes, 1);
    const vsag::InnerIdType id = 0;
    live_bytes.append(reinterpret_cast<const char*>(&id), sizeof(id));
    std::stringstream live_cache(live_bytes);
    REQUIRE(index->ImportCache(live_cache).has_value());
    const auto loaded = vsag::HGraphMemoryEstimateTestAccess::CacheStorage(*InnerHGraph(index));
    REQUIRE(loaded[0] == 1);
    REQUIRE(loaded[2] == 1);
    REQUIRE(index->GetNumElements() == 0);
    RequireUnsupportedBuildEstimate(index);
    std::stringstream empty_again(empty_bytes);
    REQUIRE(index->ImportCache(empty_again).has_value());
    const auto retained = vsag::HGraphMemoryEstimateTestAccess::CacheStorage(*InnerHGraph(index));
    REQUIRE(retained[0] == 0);
    REQUIRE(retained[2] == 0);
    REQUIRE(retained[1] > 0);
    RequireUnsupportedBuildEstimate(index);

    auto failed = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    std::string invalid;
    AppendCacheWord(invalid, 2);
    AppendCacheWord(invalid, 1000);
    invalid.resize(40, '\0');  // Pass count validation, reserve two entries, then fail the string.
    std::stringstream bad_cache(invalid);
    REQUIRE_FALSE(failed->ImportCache(bad_cache).has_value());
    const auto partial = vsag::HGraphMemoryEstimateTestAccess::CacheStorage(*InnerHGraph(failed));
    REQUIRE(partial[0] == 0);
    REQUIRE(partial[1] >= 2);
    RequireUnsupportedBuildEstimate(failed);
}

TEST_CASE("HGraph build memory lifecycle markers precede Merge and both component load failures",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    auto common = MakeCommonParam(16);
    common.thread_pool_ =
        std::make_shared<vsag::SafeThreadPool>(std::make_shared<vsag::DefaultThreadPool>(2));
    auto source = MakeHGraphIndex(MakeBuildMemoryJson(), common);
    BuildMemoryData data(3, 16);
    REQUIRE(source->Build(data.dataset).has_value());
    std::stringstream serialized;
    REQUIRE(source->Serialize(serialized).has_value());
    const auto bytes = serialized.str();

    SECTION("sequential component loading reaches the marker before a body-read error") {
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), common);
        auto graph = InnerHGraph(index);
        ComponentFaultReader reader(bytes);
        RequireEstimateError([&] { graph->Deserialize(static_cast<vsag::StreamReader&>(reader)); },
                             vsag::ErrorType::INVALID_BINARY);
        REQUIRE(reader.ReachedComponent());
        REQUIRE_FALSE(vsag::HGraphMemoryEstimateTestAccess::LifecycleClean(*graph));
        RequireUnsupportedBuildEstimate(index);
    }
    SECTION("parallel component loading reaches the marker before a body-read error") {
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), common);
        ComponentFaultReader reader(bytes);
        REQUIRE_FALSE(index->ParallelDeserialize(reader).has_value());
        REQUIRE(reader.ReachedComponent());
        REQUIRE_FALSE(vsag::HGraphMemoryEstimateTestAccess::LifecycleClean(*InnerHGraph(index)));
        RequireUnsupportedBuildEstimate(index);
    }
    SECTION("Merge rejects a different duplicate setting after entering the marker") {
        auto other_json = MakeBuildMemoryJson();
        other_json["support_duplicate"].SetBool(true);
        auto other = MakeHGraphIndex(other_json, common);
        auto index = MakeHGraphIndex(MakeBuildMemoryJson(), common);
        std::vector<vsag::MergeUnit> units(1);
        units[0].index = other;
        REQUIRE_FALSE(index->Merge(units).has_value());
        REQUIRE(index->GetNumElements() == 0);
        REQUIRE_FALSE(vsag::HGraphMemoryEstimateTestAccess::LifecycleClean(*InnerHGraph(index)));
        RequireUnsupportedBuildEstimate(index);
    }
}

TEST_CASE("HGraph bounded workers select the lowest row error and attempt every row",
          "[ut][hgraph][build_dispatch]") {
    using Access = vsag::HGraphMemoryEstimateTestAccess;
    ControlledRows rows;
    Access::WorkContext context;
    context.row_function = &ControlledRows::Invoke;
    context.row_state = &rows;
    context.row_count = rows.attempts.size();
    auto pool =
        std::make_shared<vsag::SafeThreadPool>(std::make_shared<vsag::DefaultThreadPool>(8));
    auto dispatched =
        std::async(std::launch::async, [&] { Access::RunWorkers(*pool, 4, context); });
    // Observe the actual protected error record while row zero is blocked. The timeout is only
    // a failure exit; ordering is established by the row-one record, not by elapsed time.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool higher_recorded_first = false;
    do {
        {
            std::lock_guard<std::mutex> lock(context.error_mutex);
            higher_recorded_first = context.error_row == 1;
        }
        if (!higher_recorded_first) {
            std::this_thread::yield();
        }
    } while (!higher_recorded_first && std::chrono::steady_clock::now() < deadline);
    rows.release_low.set_value();
    bool lowest_rethrown = false;
    try {
        dispatched.get();
    } catch (const ControlledRowFailure& error) {
        REQUIRE(error.row == 0);
        lowest_rethrown = true;
    }
    REQUIRE(higher_recorded_first);
    REQUIRE(lowest_rethrown);
    for (const auto& attempts : rows.attempts) {
        REQUIRE(attempts.load() == 1);
    }
}

TEST_CASE("HGraph build memory default Engine forwards supported scalar estimates",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    vsag::Engine engine(nullptr);
    for (const auto* quantizer : {"fp32", "sq8"}) {
        for (uint64_t threads : {uint64_t{1}, uint64_t{4}}) {
            vsag::JsonType parameters;
            parameters["dtype"].SetString("float32");
            parameters["metric_type"].SetString("l2");
            parameters["dim"].SetUint64(16);
            parameters["hgraph"].SetJson(MakeBuildMemoryJson(quantizer, 8, threads, 17));
            auto created = engine.CreateIndex("hgraph", parameters.Dump());
            REQUIRE(created.has_value());
            vsag::IndexPtr index = created.value();
            RequireExpectedBuildMemoryProfile(index);
            const auto steady = index->EstimateMemory(17);
#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE
            REQUIRE(index->EstimateBuildMemory(0) > 0);
            const auto peak = index->EstimateBuildMemory(17);
            REQUIRE(peak >= steady);
#endif
            BuildMemoryData data(17, 16);
            auto built = index->Build(data.dataset);
            REQUIRE(built.has_value());
            REQUIRE(built.value().empty());
            REQUIRE(index->GetNumElements() == 17);
            vsag::JsonType report;
            report["quantizer"].SetString(quantizer);
            report["requested_threads"].SetUint64(threads);
            report["count"].SetUint64(17);
            report["steady"].SetUint64(steady);
            report["expected_modeled_profile"].SetBool(EXPECT_MODELED_BUILD_MEMORY);
#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE
            report["peak_estimate"].SetUint64(peak);
#endif
            report["build_success"].SetBool(true);
            report["accounting"].SetString("public default Engine integration; no byte recorder");
            std::cout << "HGRAPH_BUILD_MEMORY_ENGINE " << report.Dump() << std::endl;
        }
    }
}

#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

TEST_CASE("HGraph build memory covers every advertised dense metric",
          "[ut][hgraph][estimate_build_memory]") {
    BlockSizeLimitGuard blocks(uint64_t{1} << 18);
    for (auto metric : {vsag::MetricType::METRIC_TYPE_IP, vsag::MetricType::METRIC_TYPE_COSINE}) {
        for (const auto* quantizer : {"fp32", "sq8", "sq8_uniform"}) {
            RunBuildMemoryCell(quantizer, 16, 8, false, 17, 17, false, metric);
        }
    }
}

#endif  // HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE

TEST_CASE("HGraph one bounded worker keeps parallel error semantics including T zero",
          "[ut][hgraph][build_dispatch]") {
    using Access = vsag::HGraphMemoryEstimateTestAccess;
    for (uint64_t threads : {uint64_t{0}, uint64_t{1}}) {
        std::array<uint64_t, 7> attempts{};
        Access::WorkContext context;
        context.row_count = attempts.size();
        context.row_state = &attempts;
        context.row_function = [](void* opaque, uint64_t row) {
            auto& visited = *static_cast<std::array<uint64_t, 7>*>(opaque);
            ++visited[row];
            if (row == 0) {
                throw ControlledRowFailure(0);
            }
        };
        auto pool =
            std::make_shared<vsag::SafeThreadPool>(std::make_shared<vsag::DefaultThreadPool>(8));
        bool caught = false;
        try {
            Access::RunWorkers(*pool, threads, context);
        } catch (const ControlledRowFailure& error) {
            REQUIRE(error.row == 0);
            caught = true;
        }
        REQUIRE(caught);
        for (auto visits : attempts) {
            REQUIRE(visits == 1);
        }
    }
    auto observing = std::make_shared<GatedBuildPool>();
    vsag::SafeThreadPool pool(observing);
    Access::WorkContext empty;
    REQUIRE_NOTHROW(Access::RunWorkers(pool, 4, empty));
    REQUIRE(observing->Counts()[0] == 0);
}

TEST_CASE("HGraph new worker allocation types match the target source rules",
          "[ut][hgraph][estimate_build_memory][build_dispatch]") {
    using Access = vsag::HGraphMemoryEstimateTestAccess;
    using BackendBind = decltype(std::bind(std::declval<std::function<void()>>()));
    static_assert(sizeof(Access::Worker) == sizeof(void*));
    static_assert(alignof(Access::Worker) == alignof(void*));
    static_assert(std::is_nothrow_copy_constructible<Access::Worker>::value);
#if HGRAPH_BUILD_MEMORY_TEST_MODELED_PROFILE
    const auto user_task = TestPackagedTaskRequest<Access::WorkerBind>();
    const auto backend_task = TestPackagedTaskRequest<BackendBind>();
    const auto total = TestSchedulerTaskRequest();
    REQUIRE(user_task >= TestSharedRequest<std::packaged_task<void()>>());
    REQUIRE(backend_task >= TestSharedRequest<std::packaged_task<void()>>());
    REQUIRE(total >= user_task + backend_task);
    vsag::JsonType report;
    report["worker_object"].SetUint64(sizeof(Access::Worker));
    report["worker_bind"].SetUint64(sizeof(Access::WorkerBind));
    report["backend_bind"].SetUint64(sizeof(BackendBind));
    report["user_task_request"].SetUint64(user_task);
    report["backend_task_request"].SetUint64(backend_task);
    report["complete_chain_request"].SetUint64(total);
    report["filter_object_upper"].SetUint64(kFilterUpperSize);
    report["filter_total_allocation_upper"].SetUint64(
        TestSharedRequest<RemoveListFilterStorageUpper>());
    report["compiler"].SetString(__VERSION__);
#if defined(_LIBCPP_VERSION)
    report["libcpp_version"].SetUint64(_LIBCPP_VERSION);
#else
    report["glibcxx_header_version"].SetUint64(__GLIBCXX__);
#endif
    std::cout << "HGRAPH_BUILD_MEMORY_LAYOUT " << report.Dump() << std::endl;
#else
    auto index = MakeHGraphIndex(MakeBuildMemoryJson(), MakeCommonParam(16));
    RequireExpectedBuildMemoryProfile(index);
#endif
}

TEST_CASE("HGraph memory estimation checks integer arithmetic",
          "[ut][hgraph][estimate_memory_arithmetic]") {
    constexpr uint64_t normal_block_size = uint64_t{1} << 18;
    BlockSizeLimitGuard normal_block_size_guard(normal_block_size);
    auto common_param = MakeCommonParam(8);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);

    REQUIRE(index->GetNumElements() == 0);
    REQUIRE(index->EstimateMemory(0) == 0);

    const auto small_estimate = index->EstimateMemory(8192);
    REQUIRE(small_estimate > 0);
    {
        BlockSizeLimitGuard block_size_guard(uint64_t{1} << 19);
        const auto larger_block_estimate = index->EstimateMemory(8192);
        REQUIRE(larger_block_estimate > small_estimate);
        REQUIRE(larger_block_estimate - small_estimate == normal_block_size);
    }
    REQUIRE(index->EstimateMemory(uint64_t{1} << 32) > small_estimate);

    uint64_t precision_estimate = 0;
    {
        BlockSizeLimitGuard block_size_guard(uint64_t{1} << 53);
        precision_estimate = index->EstimateMemory(1);
    }
    {
        BlockSizeLimitGuard block_size_guard((uint64_t{1} << 53) + 1);
        const auto larger_block_estimate = index->EstimateMemory(1);
        REQUIRE(larger_block_estimate > precision_estimate);
        REQUIRE(larger_block_estimate - precision_estimate == 2);
    }

    auto require_invalid_argument = [&index](uint64_t count) {
        bool caught_invalid_argument = false;
        try {
            (void)index->EstimateMemory(count);
        } catch (const vsag::VsagException& e) {
            REQUIRE(e.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
            caught_invalid_argument = true;
        } catch (...) {
            FAIL("Expected VsagException for unrepresentable memory estimation");
        }
        REQUIRE(caught_invalid_argument);
    };

    // FP32 code bytes overflow before block rounding.
    require_invalid_argument(uint64_t{1} << 59);
    {
        // The FP32 byte product fits, but rounding it to two blocks does not.
        BlockSizeLimitGuard block_size_guard((uint64_t{1} << 63) + 1);
        require_invalid_argument((uint64_t{1} << 58) + 1024);
    }
    {
        // Each store fits one block, but their accumulated sum does not.
        BlockSizeLimitGuard block_size_guard(uint64_t{1} << 63);
        require_invalid_argument(1);
    }
    REQUIRE(index->GetNumElements() == 0);
}

TEST_CASE("HGraph UpdateId preserves equal-label no-op", "[ut][hgraph][update]") {
    auto common_param = MakeCommonParam(8);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);

    auto result = index->UpdateId(404, 404);
    REQUIRE(result.has_value());
    REQUIRE(result.value());
}

TEST_CASE("HGraph non-fused SetImmutable does not require a fused searcher",
          "[ut][hgraph][immutable]") {
    constexpr int64_t dim = 8;
    auto common_param = MakeCommonParam(dim);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);
    std::vector<float> vectors(static_cast<uint64_t>(dim) * 2, 0.0F);
    vectors[dim] = 1.0F;
    std::vector<int64_t> ids{10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);

    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->SetImmutable().has_value());

    auto query = MakeFloatQuery(vectors, dim);
    auto result = index->KnnSearch(query, 1, kBruteForceSearchParams);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == ids[0]);
}

TEST_CASE("HGraph exact duplicate fallback supports every dense data type",
          "[ut][hgraph][duplicate][data_type]") {
    constexpr int64_t dim = 8;
    auto hgraph_json = MakeFp32HGraphJson();
    std::vector<int64_t> base_ids = {10, 20};
    std::vector<int64_t> duplicate_id = {30};

    SECTION("int8") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_INT8;
        hgraph_json["base_quantization_type"].SetString("int8");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<int8_t> base_vectors = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Int8Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Int8Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }

    SECTION("fp16") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FP16;
        hgraph_json["base_quantization_type"].SetString("fp16");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<uint16_t> base_vectors(static_cast<uint64_t>(dim * 2));
        for (int64_t i = 0; i < dim; ++i) {
            base_vectors[dim + i] = vsag::generic::FloatToFP16(static_cast<float>(i + 1));
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Float16Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Float16Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }

    SECTION("bf16") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_BF16;
        hgraph_json["base_quantization_type"].SetString("bf16");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<uint16_t> base_vectors(static_cast<uint64_t>(dim * 2));
        for (int64_t i = 0; i < dim; ++i) {
            base_vectors[dim + i] = vsag::generic::FloatToBF16(static_cast<float>(i + 1));
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Float16Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Float16Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }
}

TEST_CASE("HGraph deduplicate_storage rejects non-dense vector representations",
          "[ut][hgraph][duplicate][data_type][config]") {
    constexpr int64_t dim = 16;
    auto hgraph_json = MakeFp32HGraphJson();

    SECTION("sparse") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_SPARSE;
        common_param.metric_ = vsag::MetricType::METRIC_TYPE_IP;
        hgraph_json["base_quantization_type"].SetString("sparse");
        REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
    }

    SECTION("multi-vector") {
        auto common_param = MakeCommonParam(dim);
        common_param.repr_ = vsag::RecordRepr::MULTI_VECTOR;
        REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
    }
}

TEST_CASE("HGraph Add exact duplicate keeps duplicate label searchable",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = allocator;

    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::vector<float> duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    auto add_result = index->Add(duplicate);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->CheckIdExist(duplicate_id[0]));

    auto search_result = index->RangeSearch(duplicate, 0.01F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 2);

    const auto* result_ids = search_result.value()->GetIds();
    std::vector<int64_t> returned_ids(result_ids, result_ids + search_result.value()->GetDim());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), base_ids[1]) != returned_ids.end());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), duplicate_id[0]) !=
            returned_ids.end());

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);
    auto deserialize_result = restored->Deserialize(binary.value());
    REQUIRE(deserialize_result.has_value());
    REQUIRE(restored->CheckIdExist(duplicate_id[0]));

    auto restored_search_result =
        restored->RangeSearch(duplicate, 0.01F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(restored_search_result.has_value());
    REQUIRE(restored_search_result.value()->GetDim() == 2);

    const auto* restored_result_ids = restored_search_result.value()->GetIds();
    std::vector<int64_t> restored_returned_ids(
        restored_result_ids, restored_result_ids + restored_search_result.value()->GetDim());
    REQUIRE(std::find(restored_returned_ids.begin(), restored_returned_ids.end(), base_ids[1]) !=
            restored_returned_ids.end());
    REQUIRE(std::find(restored_returned_ids.begin(),
                      restored_returned_ids.end(),
                      duplicate_id[0]) != restored_returned_ids.end());
}

TEST_CASE("HGraph RabitQ Build exact duplicate with temporary read codes",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 960;
    constexpr int64_t count = 3;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = allocator;

    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true,
        "rabitq_bits_per_dim_base": 1,
        "rabitq_bits_per_dim_query": 32
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> vectors(static_cast<uint64_t>(dim * count));
    for (int64_t d = 0; d < dim; ++d) {
        vectors[d] = static_cast<float>(d % 17) * 0.01F;
        vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
        vectors[2 * dim + d] = vectors[dim + d];
    }
    std::vector<int64_t> ids = {10, 20, 30};
    auto base = MakeFloatDataset(vectors, ids, dim, count);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->CheckIdExist(ids[2]));

    std::vector<float> query(vectors.begin() + dim, vectors.begin() + 2 * dim);
    std::vector<int64_t> query_id = {100};
    auto query_dataset = MakeFloatDataset(query, query_id, dim, 1);
    auto search_result =
        index->RangeSearch(query_dataset, 1000000.0F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(search_result.has_value());

    const auto* result_ids = search_result.value()->GetIds();
    std::vector<int64_t> returned_ids(result_ids, result_ids + search_result.value()->GetDim());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), ids[1]) != returned_ids.end());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), ids[2]) != returned_ids.end());
}

TEST_CASE("HGraph fp32 Build exact duplicate with deduplicate_storage",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.0F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30, 40};
    auto base = MakeFloatDataset(vectors, ids, dim, 4);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> query_vector = {1.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage uses representative vector for approximate duplicates",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["duplicate_distance_threshold"].SetFloat(0.0001F);
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.005F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30, 40};
    auto base = MakeFloatDataset(vectors, ids, dim, 4);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> query_vector = {1.005F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);

    auto representative_distance = index->CalcDistanceById(query_vector.data(), ids[1]);
    auto approximate_duplicate_distance = index->CalcDistanceById(query_vector.data(), ids[2]);
    REQUIRE(representative_distance.has_value());
    REQUIRE(approximate_duplicate_distance.has_value());
    REQUIRE(approximate_duplicate_distance.value() == representative_distance.value());
}

TEST_CASE("HGraph deduplicate_storage keeps physical flatten capacity slot-based",
          "[ut][hgraph][duplicate][serialize][memory]") {
    constexpr int64_t dim = 16;
    constexpr int64_t count = 5000;
    BlockSizeLimitGuard block_size_limit_guard(256UL * 1024);
    auto common_param = MakeCommonParam(dim);

    std::vector<float> vectors(static_cast<uint64_t>(dim * count), 1.0F);
    std::vector<int64_t> ids(static_cast<uint64_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        ids[static_cast<uint64_t>(i)] = i;
    }
    auto base = MakeFloatDataset(vectors, ids, dim, count);

    auto dedup_index = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto dedup_build = dedup_index->Build(base);
    REQUIRE(dedup_build.has_value());
    REQUIRE(dedup_build.value().empty());
    auto dedup_hgraph = std::dynamic_pointer_cast<vsag::HGraph>(dedup_index->GetInnerIndex());
    REQUIRE(dedup_hgraph != nullptr);
    REQUIRE(dedup_hgraph->GetCodeStorageCounts() ==
            std::pair<vsag::InnerIdType, vsag::CodeSlotIdType>{count, 1});

    auto old_storage_index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);
    auto old_storage_build = old_storage_index->Build(base);
    REQUIRE(old_storage_build.has_value());
    REQUIRE(old_storage_build.value().empty());

    auto dedup_memory = dedup_index->GetMemoryUsageDetail();
    auto old_storage_memory = old_storage_index->GetMemoryUsageDetail();
    REQUIRE(dedup_memory.at("basic_flatten_codes") < old_storage_memory.at("basic_flatten_codes"));
    REQUIRE(dedup_memory.count("code_slot_map") == 1);
    uint64_t dedup_total = 0;
    for (const auto& item : dedup_memory) {
        dedup_total += item.second;
    }
    uint64_t old_storage_total = 0;
    for (const auto& item : old_storage_memory) {
        old_storage_total += item.second;
    }
    REQUIRE(dedup_total < old_storage_total);
}

TEST_CASE("HGraph support_duplicate without deduplicate_storage keeps old storage semantics",
          "[ut][hgraph][duplicate][regression][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson(false);
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());
    REQUIRE(index->GetNumElements() == 3);
    RequireRangeContains(index, duplicate, {20, 30}, 2);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->GetNumElements() == 3);
    RequireRangeContains(restored, duplicate, {20, 30}, 2);

    auto dedup_index = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto wrong_deserialize = dedup_index->Deserialize(binary.value());
    REQUIRE_FALSE(wrong_deserialize.has_value());
    REQUIRE(wrong_deserialize.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("HGraph deduplicate_storage Add unique then duplicate of added point",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 1);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> unique_vector = {2.0F, 0.0F};
    std::vector<int64_t> unique_id = {20};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(index->Add(unique).has_value());

    std::vector<float> duplicate_vector = {2.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());
    REQUIRE(index->GetNumElements() == 3);

    auto query = MakeFloatQuery(unique_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage handles duplicate chains", "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 1);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector_1 = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id_1 = {20};
    auto duplicate_1 = MakeFloatDataset(duplicate_vector_1, duplicate_id_1, dim, 1);
    REQUIRE(index->Add(duplicate_1).has_value());

    std::vector<float> duplicate_vector_2 = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id_2 = {30};
    auto duplicate_2 = MakeFloatDataset(duplicate_vector_2, duplicate_id_2, dim, 1);
    REQUIRE(index->Add(duplicate_2).has_value());
    REQUIRE(index->GetNumElements() == 3);

    auto query = MakeFloatQuery(base_vectors, dim);
    RequireRangeContains(index, query, {10, 20, 30}, 3, 0.01F, kBruteForceSearchParams);
    for (auto id : {10, 20, 30}) {
        auto distance = index->CalcDistanceById(base_vectors.data(), id);
        REQUIRE(distance.has_value());
        REQUIRE(distance.value() == 0.0F);
    }
}

TEST_CASE("HGraph deduplicate_storage skips duplicate external labels without binding storage",
          "[ut][hgraph][duplicate][label][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> add_vectors = {
        3.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> add_ids = {20, 30};
    auto add = MakeFloatDataset(add_vectors, add_ids, dim, 2);
    auto add_result = index->Add(add);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value() == std::vector<int64_t>{20});
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE_FALSE(index->CheckIdExist(40));

    std::vector<float> query_vector = {1.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage Build skips duplicate external labels",
          "[ut][hgraph][duplicate][label][build]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        2.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 3);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value() == std::vector<int64_t>{10});
    REQUIRE(index->GetNumElements() == 2);
    REQUIRE(index->CheckIdExist(10));
    REQUIRE(index->CheckIdExist(20));
}

TEST_CASE("HGraph deduplicate_storage deserialize keeps map extensible",
          "[ut][hgraph][duplicate][serialize][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> first_duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> first_duplicate_id = {30};
    auto first_duplicate = MakeFloatDataset(first_duplicate_vector, first_duplicate_id, dim, 1);
    REQUIRE(index->Add(first_duplicate).has_value());

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());

    auto wrong_config = MakeFp32HGraphJson(false);
    auto wrong_index = MakeHGraphIndex(wrong_config, common_param);
    auto wrong_deserialize = wrong_index->Deserialize(binary.value());
    REQUIRE_FALSE(wrong_deserialize.has_value());
    REQUIRE(wrong_deserialize.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    auto deserialize_result = restored->Deserialize(binary.value());
    REQUIRE(deserialize_result.has_value());
    REQUIRE(restored->GetNumElements() == 3);

    std::vector<float> unique_vector = {2.0F, 0.0F};
    std::vector<int64_t> unique_id = {40};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(restored->Add(unique).has_value());

    std::vector<float> second_duplicate_vector = {2.0F, 0.0F};
    std::vector<int64_t> second_duplicate_id = {50};
    auto second_duplicate = MakeFloatDataset(second_duplicate_vector, second_duplicate_id, dim, 1);
    REQUIRE(restored->Add(second_duplicate).has_value());
    REQUIRE(restored->GetNumElements() == 5);

    std::vector<float> base_duplicate_query_vector = {1.0F, 0.0F};
    auto base_duplicate_query = MakeFloatQuery(base_duplicate_query_vector, dim);
    RequireRangeContains(restored, base_duplicate_query, {20, 30}, 2);

    std::vector<float> added_duplicate_query_vector = {2.0F, 0.0F};
    auto added_duplicate_query = MakeFloatQuery(added_duplicate_query_vector, dim);
    RequireRangeContains(restored, added_duplicate_query, {40, 50}, 2);
}

TEST_CASE("HGraph deduplicate_storage rejects v0.14 serialization",
          "[ut][hgraph][duplicate][serialize]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    common_param.use_old_serial_format_ = true;
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    auto binary = index->Serialize();
    REQUIRE_FALSE(binary.has_value());
    REQUIRE(binary.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(binary.error().message.find("v0.14") != std::string::npos);
}

TEST_CASE("HGraph fused RaBitQ rejects v0.14 serialization",
          "[ut][hgraph][rabitq_split][fused][serialize]") {
    constexpr int64_t dim = 64;
    auto common_param = MakeCommonParam(dim);
    common_param.use_old_serial_format_ = true;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 2,
        "rabitq_bits_per_dim_precise": 6,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "nsw",
        "max_degree": 8,
        "ef_construction": 32,
        "use_reorder": true,
        "reorder_source": "base",
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    auto binary = index->Serialize();
    REQUIRE_FALSE(binary.has_value());
    REQUIRE(binary.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(binary.error().message.find("v0.14") != std::string::npos);
}

TEST_CASE("HGraph fused RaBitQ GetStats decodes vectors from node records",
          "[ut][hgraph][rabitq_split][fused][stats]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "rabitq_use_fht": true,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "nsw",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "use_reorder": true,
        "reorder_source": "base",
        "store_raw_vector": false,
        "use_mci": false,
        "support_duplicate": true,
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors(static_cast<uint64_t>(count) * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[static_cast<uint64_t>(i) * dim + d] =
                0.1F + static_cast<float>((i * 17 + d * 13) % 79) / 100.0F;
        }
    }
    std::copy_n(vectors.data(), dim, vectors.data() + static_cast<uint64_t>(count - 1) * dim);
    auto base = MakeFloatDataset(vectors, ids, dim, count);
    REQUIRE(index->Build(base).has_value());

    std::string stats;
    REQUIRE_NOTHROW(stats = index->GetStats());
    REQUIRE_FALSE(stats.empty());
    const auto parsed_stats = vsag::JsonType::Parse(stats);
    REQUIRE(parsed_stats["duplicate_ratio"].GetFloat() > 0.0F);

    const std::vector<int64_t> fetch_ids = {ids.front(), ids.back()};
    auto fetched = index->GetInnerIndex()->GetVectorByIds(
        fetch_ids.data(), static_cast<int64_t>(fetch_ids.size()), nullptr);
    REQUIRE(fetched != nullptr);
    REQUIRE(fetched->GetNumElements() == static_cast<int64_t>(fetch_ids.size()));
    REQUIRE(fetched->GetDim() == dim);
    const auto* fetched_vectors = fetched->GetFloat32Vectors();
    REQUIRE(fetched_vectors != nullptr);
    for (uint64_t i = 0; i < static_cast<uint64_t>(fetch_ids.size()) * dim; ++i) {
        REQUIRE(std::isfinite(fetched_vectors[i]));
    }
}

TEST_CASE("HGraph fused RaBitQ remains mutable after fast build and deserialize",
          "[ut][hgraph][rabitq_split][fused][lifecycle]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    constexpr int64_t extra_info_size = 4;
    auto common_param = MakeCommonParam(dim);
    common_param.extra_info_size_ = extra_info_size;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 2,
        "rabitq_bits_per_dim_precise": 6,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "odescent",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "use_reorder": true,
        "reorder_source": "base",
        "use_attribute_filter": true,
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ADD_AFTER_BUILD));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_MERGE_INDEX));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_CLONE));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_EXPORT_MODEL));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_TUNE));
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT));
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT));

    std::vector<float> vectors(static_cast<uint64_t>(count) * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i + 100;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[static_cast<uint64_t>(i) * dim + d] =
                static_cast<float>((i * 17 + d * 13) % 101) / 101.0F;
        }
    }
    auto base = MakeFloatDataset(vectors, ids, dim, count);
    std::vector<char> base_extra_infos(count * extra_info_size, 'a');
    base->ExtraInfos(base_extra_infos.data())->ExtraInfoSize(extra_info_size);
    std::vector<vsag::AttributeValue<int64_t>> base_attributes(count);
    std::vector<vsag::AttributeSet> base_attribute_sets(count);
    for (int64_t i = 0; i < count; ++i) {
        base_attributes[static_cast<uint64_t>(i)].name_ = "tenant";
        base_attributes[static_cast<uint64_t>(i)].GetValue().push_back(1);
        base_attribute_sets[static_cast<uint64_t>(i)].attrs_.push_back(
            &base_attributes[static_cast<uint64_t>(i)]);
    }
    base->AttributeSets(base_attribute_sets.data());
    REQUIRE(index->Build(base).has_value());

    std::vector<float> add_vector(dim, 2.0F);
    std::vector<int64_t> add_id{1000};
    auto add = MakeFloatDataset(add_vector, add_id, dim, 1);
    std::vector<char> add_extra_info(extra_info_size, 'b');
    add->ExtraInfos(add_extra_info.data())->ExtraInfoSize(extra_info_size);
    vsag::AttributeValue<int64_t> add_attribute;
    add_attribute.name_ = "tenant";
    add_attribute.GetValue().push_back(1);
    vsag::AttributeSet add_attribute_set;
    add_attribute_set.attrs_.push_back(&add_attribute);
    add->AttributeSets(&add_attribute_set);
    auto add_result = index->Add(add);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->CheckIdExist(add_id.front()));

    std::vector<float> updated_vector(dim, 3.0F);
    std::vector<int64_t> updated_id{ids.front()};
    auto update = MakeFloatDataset(updated_vector, updated_id, dim, 1);
    auto update_result = index->UpdateVector(updated_id.front(), update, true);
    REQUIRE(update_result.has_value());
    REQUIRE(update_result.value());

    constexpr int64_t renamed_id = 2000;
    auto update_id_result = index->UpdateId(updated_id.front(), renamed_id);
    REQUIRE(update_id_result.has_value());
    REQUIRE(update_id_result.value());
    REQUIRE_FALSE(index->CheckIdExist(updated_id.front()));
    REQUIRE(index->CheckIdExist(renamed_id));

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->CheckIdExist(renamed_id));

    std::vector<float> restored_add_vector(dim, 4.0F);
    std::vector<int64_t> restored_add_id{3000};
    auto restored_add = MakeFloatDataset(restored_add_vector, restored_add_id, dim, 1);
    std::vector<char> restored_add_extra_info(extra_info_size, 'c');
    restored_add->ExtraInfos(restored_add_extra_info.data())->ExtraInfoSize(extra_info_size);
    vsag::AttributeValue<int64_t> restored_add_attribute;
    restored_add_attribute.name_ = "tenant";
    restored_add_attribute.GetValue().push_back(1);
    vsag::AttributeSet restored_add_attribute_set;
    restored_add_attribute_set.attrs_.push_back(&restored_add_attribute);
    restored_add->AttributeSets(&restored_add_attribute_set);
    REQUIRE(restored->Add(restored_add).has_value());
    REQUIRE(restored->CheckIdExist(restored_add_id.front()));

    vsag::AttributeValue<int64_t> updated_attribute;
    updated_attribute.name_ = "tenant";
    updated_attribute.GetValue().push_back(2);
    vsag::AttributeSet updated_attribute_set;
    updated_attribute_set.attrs_.push_back(&updated_attribute);
    REQUIRE(restored->UpdateAttribute(renamed_id, updated_attribute_set).has_value());

    std::vector<char> updated_extra_info(extra_info_size, 'd');
    auto extra_info_update = vsag::Dataset::Make();
    extra_info_update->NumElements(1)
        ->Ids(&renamed_id)
        ->ExtraInfos(updated_extra_info.data())
        ->ExtraInfoSize(extra_info_size)
        ->Owner(false);
    auto extra_info_result = restored->UpdateExtraInfo(extra_info_update);
    REQUIRE(extra_info_result.has_value());
    REQUIRE(extra_info_result.value());
    std::vector<char> fetched_extra_info(extra_info_size);
    REQUIRE(restored->GetExtraInfoByIds(&renamed_id, 1, fetched_extra_info.data()).has_value());
    REQUIRE(fetched_extra_info == updated_extra_info);

    const std::vector<int64_t> remove_ids{add_id.front()};
    auto remove_result = restored->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);
    REQUIRE_FALSE(restored->CheckIdExist(add_id.front()));

    auto require_unsupported = [](const auto& result) {
        REQUIRE_FALSE(result.has_value());
        CAPTURE(result.error().message);
        REQUIRE(result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    };
    auto train_result = index->Train(base);
    require_unsupported(train_result);
    REQUIRE(train_result.error().message == "fused RaBitQ HGraph does not support Train");
    require_unsupported(index->Remove(remove_ids, vsag::RemoveMode::FORCE_REMOVE));
    require_unsupported(index->Tune("{}"));
    require_unsupported(index->Merge({}));
    require_unsupported(index->Clone());
    require_unsupported(index->ExportModel());

    std::stringstream cache;
    require_unsupported(index->ExportCache(cache));
    require_unsupported(index->ImportCache(cache));

    REQUIRE(restored->SetImmutable().has_value());
    require_unsupported(restored->Add(restored_add));
    require_unsupported(restored->UpdateVector(renamed_id, update, true));
}

TEST_CASE("HGraph deduplicate_storage supports precise reorder code path",
          "[ut][hgraph][duplicate][reorder][add]") {
    constexpr int64_t dim = 4;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32ReorderHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        0.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector = {1.0F, 0.0F, 0.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());

    auto query = MakeFloatQuery(duplicate_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2, 0.01F, kBruteForceSearchParams);

    auto knn = index->KnnSearch(
        query, 2, R"({"hgraph": {"ef_search": 16, "brute_force_threshold": 1.0}})");
    REQUIRE(knn.has_value());
    REQUIRE(knn.value()->GetDim() == 2);
    REQUIRE(ResultContainsId(knn.value(), 20));
    REQUIRE(ResultContainsId(knn.value(), 30));
}

TEST_CASE("HGraph deduplicate_storage supports RabitQ raw-vector graph read path",
          "[ut][hgraph][duplicate][raw_vector][add]") {
    constexpr int64_t dim = 960;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeRabitQRawVectorHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors(dim * 3);
    for (int64_t d = 0; d < dim; ++d) {
        base_vectors[d] = static_cast<float>(d % 17) * 0.01F;
        base_vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
        base_vectors[2 * dim + d] = 2.0F + static_cast<float>(d % 11) * 0.01F;
    }
    std::vector<int64_t> base_ids = {10, 20, 30};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 3);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector(base_vectors.begin() + dim, base_vectors.begin() + dim * 2);
    std::vector<int64_t> duplicate_id = {40};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    auto duplicate_add_result = index->Add(duplicate);
    REQUIRE(duplicate_add_result.has_value());
    REQUIRE(duplicate_add_result.value().empty());

    std::vector<float> unique_vector(dim);
    for (int64_t d = 0; d < dim; ++d) {
        unique_vector[d] = 10.0F + static_cast<float>(d % 19) * 0.02F;
    }
    std::vector<int64_t> unique_id = {50};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    auto unique_add_result = index->Add(unique);
    REQUIRE(unique_add_result.has_value());
    REQUIRE(unique_add_result.value().empty());
    REQUIRE(index->CheckIdExist(unique_id[0]));
    auto unique_distance = index->CalcDistanceById(unique_vector.data(), unique_id[0]);
    REQUIRE(unique_distance.has_value());
    REQUIRE(unique_distance.value() == 0.0F);

    auto duplicate_query = MakeFloatQuery(duplicate_vector, dim);
    RequireRangeContains(index, duplicate_query, {20, 40}, 2, 0.01F, kBruteForceSearchParams);

    auto unique_query = MakeFloatQuery(unique_vector, dim);
    RequireRangeContains(index, unique_query, {50}, 1, 0.01F, kBruteForceSearchParams);
}

TEST_CASE("HGraph deduplicate_storage Tune keeps physical slots isolated",
          "[ut][hgraph][duplicate][tune]") {
    constexpr int64_t dim = 960;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeRabitQRawVectorHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors(dim * 2);
    for (int64_t d = 0; d < dim; ++d) {
        base_vectors[d] = static_cast<float>(d % 17) * 0.01F;
        base_vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
    }
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector(base_vectors.begin() + dim, base_vectors.end());
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());

    auto tune_result = index->Tune(R"({
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "support_duplicate": true,
            "deduplicate_storage": true,
            "store_raw_vector": true
        }
    })");
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value());

    std::vector<float> unique_vector(dim);
    for (int64_t d = 0; d < dim; ++d) {
        unique_vector[d] = 10.0F + static_cast<float>(d % 19) * 0.02F;
    }
    std::vector<int64_t> unique_id = {40};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(index->Add(unique).has_value());

    auto duplicate_base_distance =
        index->CalcDistanceById(duplicate_vector.data(), duplicate_id[0], false);
    REQUIRE(duplicate_base_distance.has_value());
    REQUIRE(duplicate_base_distance.value() == 0.0F);
}

TEST_CASE("HGraph deduplicate_storage rejects Merge", "[ut][hgraph][duplicate][merge]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();

    auto target = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE_FALSE(target->CheckFeature(vsag::IndexFeature::SUPPORT_MERGE_INDEX));
    std::vector<float> target_vectors = {0.0F, 0.0F};
    std::vector<int64_t> target_ids = {10};
    auto target_data = MakeFloatDataset(target_vectors, target_ids, dim, 1);
    REQUIRE(target->Build(target_data).has_value());

    auto source = MakeHGraphIndex(hgraph_json, common_param);
    std::vector<float> source_vectors = {4.0F, 0.0F};
    std::vector<int64_t> source_ids = {20};
    auto source_data = MakeFloatDataset(source_vectors, source_ids, dim, 1);
    REQUIRE(source->Build(source_data).has_value());

    vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
        return std::make_tuple(true, id);
    };
    auto merge_result = target->Merge({vsag::MergeUnit{source, id_map}});
    REQUIRE_FALSE(merge_result.has_value());
    REQUIRE(merge_result.error().message.find("deduplicate_storage") != std::string::npos);
    REQUIRE(target->GetNumElements() == 1);
}

TEST_CASE("HGraph deduplicate_storage ExportModel keeps an empty reusable model",
          "[ut][hgraph][duplicate][export_model]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    auto model_result = index->ExportModel();
    REQUIRE(model_result.has_value());
    auto model = model_result.value();
    REQUIRE(model->GetNumElements() == 0);

    std::vector<float> model_vectors = {
        3.0F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> model_ids = {30, 40};
    auto model_data = MakeFloatDataset(model_vectors, model_ids, dim, 2);
    REQUIRE(model->Build(model_data).has_value());

    std::vector<float> query_vector = {3.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    auto search_result = model->RangeSearch(query, 0.01F, kBruteForceSearchParams);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 2);
    REQUIRE(ResultContainsId(search_result.value(), 30));
    REQUIRE(ResultContainsId(search_result.value(), 40));
}

TEST_CASE("HGraph deduplicate_storage rejects UpdateVector for a shared duplicate group",
          "[ut][hgraph][duplicate][update]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30};
    auto base = MakeFloatDataset(vectors, ids, dim, 3);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> updated_vector = {3.0F, 0.0F};
    std::vector<int64_t> update_id = {30};
    auto update_data = MakeFloatDataset(updated_vector, update_id, dim, 1);
    auto update_result = index->UpdateVector(update_id[0], update_data, true);
    REQUIRE_FALSE(update_result.has_value());
    REQUIRE(update_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    std::vector<float> original_duplicate_vector = {1.0F, 0.0F};
    auto representative_distance =
        index->CalcDistanceById(original_duplicate_vector.data(), 20, false);
    REQUIRE(representative_distance.has_value());
    REQUIRE(representative_distance.value() == 0.0F);

    auto duplicate_distance = index->CalcDistanceById(original_duplicate_vector.data(), 30, false);
    REQUIRE(duplicate_distance.has_value());
    REQUIRE(duplicate_distance.value() == 0.0F);

    std::vector<float> unique_update_vector = {4.0F, 0.0F};
    std::vector<int64_t> unique_update_id = {10};
    auto unique_update_data = MakeFloatDataset(unique_update_vector, unique_update_id, dim, 1);
    auto unique_update_result = index->UpdateVector(unique_update_id[0], unique_update_data, true);
    REQUIRE(unique_update_result.has_value());
    REQUIRE(unique_update_result.value());
    auto unique_distance =
        index->CalcDistanceById(unique_update_vector.data(), unique_update_id[0], false);
    REQUIRE(unique_distance.has_value());
    REQUIRE(unique_distance.value() == 0.0F);
}

TEST_CASE("HGraph deduplicate_storage batch Add supports internal parallel add",
          "[ut][hgraph][duplicate][parallel][add]") {
    constexpr int64_t dim = 4;
    constexpr uint64_t build_threads = 4;
    auto common_param = MakeCommonParam(dim, build_threads);
    auto hgraph_json = MakeFp32HGraphJson(true, static_cast<int64_t>(build_threads));
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> seed_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        100.0F,
        0.0F,
        0.0F,
        0.0F,
        200.0F,
        0.0F,
        0.0F,
        0.0F,
        300.0F,
        0.0F,
        0.0F,
        0.0F,
    };
    std::vector<int64_t> seed_ids = {10, 20, 30, 40};
    auto seed = MakeFloatDataset(seed_vectors, seed_ids, dim, 4);
    REQUIRE(index->Build(seed).has_value());

    constexpr int64_t add_count = 8;
    std::vector<float> add_vectors(dim * add_count);
    std::vector<int64_t> add_ids(add_count);
    for (int64_t i = 0; i < add_count; ++i) {
        add_ids[i] = 100 + i;
        if (i < 4) {
            std::copy(seed_vectors.begin() + dim,
                      seed_vectors.begin() + dim * 2,
                      add_vectors.begin() + i * dim);
            continue;
        }
        for (int64_t d = 0; d < dim; ++d) {
            add_vectors[i * dim + d] = static_cast<float>(1000 + i * 100 + d);
        }
    }

    auto batch = MakeFloatDataset(add_vectors, add_ids, dim, add_count);
    auto add_result = index->Add(batch);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->GetNumElements() == 12);
    for (auto id : add_ids) {
        REQUIRE(index->CheckIdExist(id));
    }

    std::vector<float> duplicate_query_vector(seed_vectors.begin() + dim,
                                              seed_vectors.begin() + dim * 2);
    auto duplicate_query = MakeFloatQuery(duplicate_query_vector, dim);
    RequireRangeContains(index, duplicate_query, {20, 100, 101, 102, 103}, 5);

    std::vector<float> unique_query_vector(add_vectors.begin() + dim * 4,
                                           add_vectors.begin() + dim * 5);
    auto unique_query = MakeFloatQuery(unique_query_vector, dim);
    RequireRangeContains(index, unique_query, {104}, 1);
}

TEST_CASE("HGraph deduplicate_storage physical growth is single-flight",
          "[ut][hgraph][duplicate][parallel][resize]") {
    constexpr int64_t dim = 128;
    constexpr int64_t base_count = 500;
    constexpr uint64_t search_threads = 8;
    constexpr uint64_t resize_threads = 32;
    constexpr vsag::InnerIdType required_capacity = 513;
    BlockSizeLimitGuard block_size_limit_guard(256UL * 1024);

    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors(dim * base_count, 0.0F);
    std::vector<int64_t> ids(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        ids[i] = i;
    }
    auto base = MakeFloatDataset(vectors, ids, dim, base_count);
    REQUIRE(index->Build(base).has_value());

    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);

    std::vector<float> query_vector(dim, 0.0F);
    auto query = MakeFloatQuery(query_vector, dim);
    std::atomic<bool> keep_searching{true};
    std::atomic<uint64_t> searchers_started{0};
    std::vector<std::future<bool>> searchers;
    searchers.reserve(search_threads);
    for (uint64_t i = 0; i < search_threads; ++i) {
        searchers.emplace_back(std::async(std::launch::async, [&]() {
            searchers_started.fetch_add(1, std::memory_order_release);
            while (keep_searching.load(std::memory_order_acquire)) {
                auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 128}})");
                if (not result.has_value()) {
                    return false;
                }
            }
            return true;
        }));
    }
    while (searchers_started.load(std::memory_order_acquire) < search_threads) {
        std::this_thread::yield();
    }

    std::atomic<uint64_t> resizers_ready{0};
    std::atomic<bool> start_resizing{false};
    std::vector<std::future<void>> resizers;
    resizers.reserve(resize_threads);
    for (uint64_t i = 0; i < resize_threads; ++i) {
        resizers.emplace_back(std::async(std::launch::async, [&]() {
            resizers_ready.fetch_add(1, std::memory_order_release);
            while (not start_resizing.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            hgraph->ensure_physical_code_capacity(required_capacity);
        }));
    }
    while (resizers_ready.load(std::memory_order_acquire) < resize_threads) {
        std::this_thread::yield();
    }
    start_resizing.store(true, std::memory_order_release);

    bool all_resizers_finished = true;
    for (auto& resizer : resizers) {
        if (resizer.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            all_resizers_finished = false;
            break;
        }
    }
    keep_searching.store(false, std::memory_order_release);
    for (auto& searcher : searchers) {
        REQUIRE(searcher.get());
    }
    for (auto& resizer : resizers) {
        resizer.get();
    }
    REQUIRE(all_resizers_finished);
}

TEST_CASE("HGraph deduplicate_storage concurrent Add keeps visible duplicates searchable",
          "[ut][hgraph][duplicate][concurrent][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> seed_vectors = {
        0.0F,
        0.0F,
        100.0F,
        0.0F,
    };
    std::vector<int64_t> seed_ids = {10, 20};
    auto seed = MakeFloatDataset(seed_vectors, seed_ids, dim, 2);
    REQUIRE(index->Build(seed).has_value());

    constexpr int64_t add_count = 10;
    std::vector<std::future<bool>> futures;
    futures.reserve(add_count);
    for (int64_t i = 0; i < add_count; ++i) {
        futures.emplace_back(std::async(std::launch::async, [index, i, dim]() {
            std::vector<float> vector(dim);
            if (i % 2 == 0) {
                vector[0] = 0.0F;
                vector[1] = 0.0F;
            } else {
                vector[0] = static_cast<float>(1000 + i * 100);
                vector[1] = 0.0F;
            }
            std::vector<int64_t> id = {1000 + i};
            auto dataset = MakeFloatDataset(vector, id, dim, 1);
            return index->Add(dataset).has_value();
        }));
    }
    for (auto& future : futures) {
        REQUIRE(future.get());
    }

    REQUIRE(index->GetNumElements() == 12);
    for (int64_t i = 0; i < add_count; ++i) {
        REQUIRE(index->CheckIdExist(1000 + i));
    }

    std::vector<float> duplicate_query_vector = {0.0F, 0.0F};
    auto duplicate_query = MakeFloatQuery(duplicate_query_vector, dim);
    RequireRangeContains(index,
                         duplicate_query,
                         {10, 1000, 1002, 1004, 1006, 1008},
                         6,
                         0.01F,
                         kBruteForceSearchParams);

    std::vector<float> unique_query_vector = {1100.0F, 0.0F};
    auto unique_query = MakeFloatQuery(unique_query_vector, dim);
    RequireRangeContains(index, unique_query, {1001}, 1, 0.01F, kBruteForceSearchParams);
}

TEST_CASE("HGraph deduplicate_storage rejects cache accelerated Build",
          "[ut][hgraph][duplicate][cache]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 4;
    auto common_param = MakeCommonParam(dim);

    std::vector<float> vectors(dim * count);
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>(i * 10 + d);
        }
    }
    std::vector<int64_t> ids = {10, 20, 30, 40};
    std::vector<std::string> source_ids = {"sid_10", "sid_20", "sid_30", "sid_40"};
    auto base = MakeFloatDatasetWithSourceIds(vectors, ids, source_ids, dim, count);

    auto cache_source_json = MakeFp32HGraphJson(false);
    auto cache_source = MakeHGraphIndex(cache_source_json, common_param);
    REQUIRE(cache_source->Build(base).has_value());

    std::stringstream cache_stream;
    auto export_result = cache_source->ExportCache(cache_stream);
    REQUIRE(export_result.has_value());
    cache_stream.seekg(0);

    auto warmed = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto import_result = warmed->ImportCache(cache_stream);
    REQUIRE(import_result.has_value());

    auto build_result = warmed->Build(base);
    REQUIRE_FALSE(build_result.has_value());
    REQUIRE(build_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(build_result.error().message.find("deduplicate_storage") != std::string::npos);
}

TEST_CASE("HGraph deduplicate_storage rejects unsupported graph type",
          "[ut][hgraph][duplicate][config]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["graph_type"].SetString("odescent");

    REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
}

TEST_CASE("HGraph optimized build accepts MRLE RaBitQ split with raw-vector storage",
          "[ut][hgraph][optimized_build][MRLE]") {
    constexpr int64_t dim = 64;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "tq",
        "tq_chain": "mrle, rabitq",
        "mrle_dim": 32,
        "precise_quantization_type": "rabitq",
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "use_reorder": true,
        "build_by_base": false,
        "max_degree": 8,
        "ef_construction": 32
    })");
    auto index = MakeHGraphIndex(hgraph_json, MakeCommonParam(dim, 2));
    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);

    vsag::HGraphOptimizedBuildSession session(*hgraph);
    REQUIRE(session.Active());
}

TEST_CASE("HGraph fused Add preencoding preserves row mapping and rejects failed batches",
          "[ut][hgraph][fused][preencode]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    const auto threads = GENERATE(0U, 4U);
    const auto bits = GENERATE(1, 3);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    auto common = MakeCommonParam(dim, threads);
    common.metric_ = metric;
    auto param = vsag::JsonType::Parse(R"({
        "base_quantization_type":"rabitq", "precise_quantization_type":"rabitq",
        "base_io_type":"memory_io", "base_supplement_io_type":"memory_io",
        "graph_io_type":"memory_io", "graph_storage_type":"flat", "graph_type":"nsw",
        "rabitq_use_fht":true, "rabitq_fused_datacell":true,
        "use_reorder":true, "reorder_source":"base", "store_raw_vector":false,
        "use_mci":false, "max_degree":8, "ef_construction":32
    })");
    param["build_thread_count"].SetInt(threads == 0 ? 1 : threads);
    param["rabitq_bits_per_dim_base"].SetInt(bits);
    param["rabitq_bits_per_dim_precise"].SetInt(8 - bits);
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = 0.01F * static_cast<float>((i * 17 + d * 13) % 97 - 48);
        }
    }
    auto dataset = MakeFloatDataset(vectors, ids, dim, count);
    auto index = MakeHGraphIndex(param, common);
    auto seed = MakeFloatDataset(vectors, ids, dim, 16);
    REQUIRE(index->Add(seed).has_value());
    auto original = index->CalcDistanceById(vectors.data(), ids.front());
    REQUIRE(original.has_value());

    // A finite input can still overflow encoding. Other worker ranges contain valid rows.
    auto invalid = vectors;
    std::fill_n(invalid.data() + 53 * dim, dim, std::numeric_limits<float>::max());
    auto failed = index->Add(MakeFloatDataset(invalid, ids, dim, count));
    REQUIRE_FALSE(failed.has_value());
    REQUIRE(failed.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(index->GetNumElements() == 16);
    for (int64_t i = 16; i < count; ++i) {
        REQUIRE_FALSE(index->CheckIdExist(ids[i]));
    }
    REQUIRE(index->CalcDistanceById(vectors.data(), ids.front()).value() == original.value());

    // Retrying the same labels must work; skipped input row 0 must not shift cached codes.
    auto added = index->Add(dataset);
    REQUIRE(added.has_value());
    REQUIRE(added.value() == std::vector<int64_t>(ids.begin(), ids.begin() + 16));
    REQUIRE(index->GetNumElements() == count);
    auto serial_common = MakeCommonParam(dim);
    serial_common.metric_ = metric;
    param["build_thread_count"].SetInt(1);
    auto serial = MakeHGraphIndex(param, serial_common);
    REQUIRE(serial->Add(seed).has_value());
    for (int64_t i = 16; i < count; ++i) {
        auto one = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Ids(ids.data() + i)
                       ->Float32Vectors(vectors.data() + i * dim)
                       ->Owner(false);
        REQUIRE(serial->Add(one).has_value());
    }
    auto decoded = index->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    auto expected = serial->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    REQUIRE(decoded->GetNumElements() == count);
    for (int64_t i = 0; i < count * dim; ++i) {
        REQUIRE(decoded->GetFloat32Vectors()[i] == expected->GetFloat32Vectors()[i]);
    }
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = MakeHGraphIndex(param, serial_common);
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    REQUIRE(restored->GetNumElements() == count);
    auto restored_vectors = restored->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    for (int64_t i = 0; i < count * dim; ++i) {
        REQUIRE(restored_vectors->GetFloat32Vectors()[i] == expected->GetFloat32Vectors()[i]);
    }
}
