
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

#pragma once

#include <algorithm>
#include <limits>
#include <shared_mutex>
#include <type_traits>

#include "bucket_interface.h"
#include "impl/inner_search_param.h"
#include "io/container/io_array.h"
#include "io/read_cache/lru_page_cache.h"
#include "io/read_cache/page.h"
#include "io/reader_io/reader_io_parameter.h"
#include "quantization/product_quantization/pq_fastscan_quantizer.h"
#include "simd/fp32_simd.h"
#include "utils/byte_buffer.h"
#include "utils/timer.h"

namespace vsag {

inline IOParamPtr
AdjustBucketReadCacheParam(const IOParamPtr& io_param, BucketIdType bucket_count) {
    if (io_param == nullptr or not io_param->enable_read_cache_ or bucket_count == 0) {
        return io_param;
    }
    JsonType json = io_param->ToJson();
    uint64_t total_pages = io_param->read_cache_total_size_ / Page::DEFAULT_PAGE_SIZE;
    uint64_t pages_per_bucket = total_pages / bucket_count;
    json[READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetUint64(pages_per_bucket * Page::DEFAULT_PAGE_SIZE);
    auto adjusted_io_param = IOParameter::GetIOParameterByJson(json);
    auto source_reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(io_param);
    auto adjusted_reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(adjusted_io_param);
    if (source_reader_param != nullptr and adjusted_reader_param != nullptr) {
        adjusted_reader_param->reader = source_reader_param->reader;
    }
    return adjusted_io_param;
}

template <typename QuantTmpl, typename IOTmpl>
class BucketDataCell : public BucketInterface {
public:
    explicit BucketDataCell(const QuantizerParamPtr& quantization_param,
                            const IOParamPtr& io_param,
                            const IndexCommonParam& common_param,
                            BucketIdType bucket_count,
                            bool use_residual = false);

    void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id,
                   QueryContext* = nullptr,
                   InnerIdType* scanned_inner_ids = nullptr,
                   InnerIdType max_scan_size = std::numeric_limits<InnerIdType>::max(),
                   InnerIdType* scanned_size = nullptr) override {
        auto comp = static_cast<Computer<QuantTmpl>*>(computer.get());
        return this->scan_bucket_by_id(
            result_dists, comp, bucket_id, scanned_inner_ids, max_scan_size, scanned_size);
    }

    float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        return this->query_one_by_id(comp, bucket_id, offset_id);
    }

    float
    ComputePairVectors(BucketIdType bucket_id, InnerIdType id1, InnerIdType id2) override;

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const BucketIdType* bucket_ids,
          const InnerIdType* offset_ids,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        if (id_count > 0 and GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "PQFastScan doesn't support ComputeDist, only support "
                                "ComputeBatchDist");
        }
        auto comp = static_cast<Computer<QuantTmpl>*>(computer.get());
        this->query(result_dists, comp, bucket_ids, offset_ids, id_count, ctx);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override;

    void
    Train(const void* data, uint64_t count) override;

    InnerIdType
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) override;

    void
    BatchInsertVector(const void* vectors,
                      const BucketIdType* bucket_ids,
                      const InnerIdType* inner_ids,
                      InnerIdType count,
                      InnerIdType* out_offsets) override;

    void
    InsertVectorWithOffset(const void* vector,
                           BucketIdType bucket_id,
                           InnerIdType inner_id,
                           InnerIdType offset_id) override;

    InnerIdType*
    GetInnerIds(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        return this->inner_ids_[bucket_id].data();
    }

    void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) override {
        this->check_valid_bucket_id(bucket_id);
        this->datas_[bucket_id].Prefetch(offset_id * code_size_, code_size_);
    }

    void
    Package() override {
        if (GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
            this->package_fastscan();
        }
    }

    void
    Unpack() override {
        if (GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
            this->unpack_fastscan();
        }
    }

    void
    ExportModel(const BucketInterfacePtr& other) const override;

    void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(LvalueOrRvalue<StreamReader> reader) override;

    void
    InitIO(const IOParamPtr& io_param) override {
        if constexpr (IOTmpl::SkipDeserialize) {
            this->init_shared_read_cache_io(io_param);
            return;
        }
        auto adjusted_io_param = AdjustBucketReadCacheParam(io_param, this->bucket_count_);
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            this->datas_[bucket_id].InitIO(adjusted_io_param);
        }
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return this->quantizer_->Name();
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return metric_;
    }

    [[nodiscard]] InnerIdType
    GetBucketSize(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        return this->bucket_sizes_[bucket_id];
    }

    [[nodiscard]] InnerIdType
    GetBucketScanCapacity(BucketIdType bucket_id) override {
        check_valid_bucket_id(bucket_id);
        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        return this->bucket_sizes_[bucket_id];
    }

    void
    GetCodesById(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* data) const override;

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override {
        uint64_t memory = sizeof(BucketDataCell);
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; bucket_id++) {
            memory += this->datas_[bucket_id].GetMemoryUsage();
            memory += this->inner_ids_[bucket_id].size() * sizeof(InnerIdType);
            memory += this->residual_bias_[bucket_id].size() * sizeof(float);
            memory += sizeof(std::shared_mutex) + sizeof(InnerIdType);
        }
        return memory;
    }

private:
    void
    init_shared_read_cache_io(const IOParamPtr& io_param) {
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            this->datas_[bucket_id].SetReadCache(nullptr);
        }
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            this->datas_[bucket_id].InitIO(io_param);
        }

        if (io_param == nullptr or not io_param->enable_read_cache_) {
            return;
        }
        uint64_t cache_page_count = io_param->read_cache_total_size_ / Page::DEFAULT_PAGE_SIZE;
        if (cache_page_count == 0) {
            return;
        }

        uint64_t page_id_base = 0;
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            uint64_t bucket_page_count = get_page_count(this->datas_[bucket_id].size_);
            if (bucket_page_count > UINT64_MAX - page_id_base) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    "bucket read cache page id overflow");
            }
            page_id_base += bucket_page_count;
        }

        auto shared_cache = std::make_shared<LRUPageCache>(cache_page_count);
        page_id_base = 0;
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            uint64_t bucket_page_count = get_page_count(this->datas_[bucket_id].size_);
            this->datas_[bucket_id].SetReadCache(shared_cache, page_id_base);
            page_id_base += bucket_page_count;
        }
    }

    static uint64_t
    get_page_count(uint64_t data_size) {
        return data_size / Page::DEFAULT_PAGE_SIZE +
               static_cast<uint64_t>(data_size % Page::DEFAULT_PAGE_SIZE != 0);
    }

    inline void
    check_valid_bucket_id(BucketIdType bucket_id) {
        if (bucket_id >= this->bucket_count_ or bucket_id < 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "visited invalid bucket id");
        }
    }

    inline void
    scan_bucket_by_id(float* result_dists,
                      Computer<QuantTmpl>* computer,
                      const BucketIdType& bucket_id,
                      InnerIdType* scanned_inner_ids,
                      InnerIdType max_scan_size,
                      InnerIdType* scanned_size);

    inline float
    query_one_by_id(const std::shared_ptr<Computer<QuantTmpl>>& computer,
                    const BucketIdType& bucket_id,
                    const InnerIdType& offset_id);

    inline void
    query(float* result_dists,
          Computer<QuantTmpl>* computer,
          const BucketIdType* bucket_ids,
          const InnerIdType* offset_ids,
          InnerIdType id_count,
          QueryContext* ctx);

    inline void
    encode_vector(const void* vector, BucketIdType bucket_id, uint8_t* codes, float& res_score);

    inline void
    check_valid_bucket_capacity(uint64_t next_size, bool need_resize);

    inline void
    package_fastscan();

    inline void
    unpack_fastscan();

private:
    std::shared_ptr<QuantTmpl> quantizer_{nullptr};

    IOArray<IOTmpl> datas_;

    Vector<InnerIdType> bucket_sizes_;

    Vector<std::shared_mutex> bucket_mutexes_;

    Vector<Vector<InnerIdType>> inner_ids_;

    Allocator* const allocator_{nullptr};

    Vector<Vector<float>> residual_bias_;

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};

    uint64_t input_dim_{0};

    static constexpr InnerIdType EMPTY_INNER_ID = std::numeric_limits<InnerIdType>::max();

    // Bound sparse fixed-offset metadata growth to reject accidental huge hole allocation.
    static constexpr uint64_t MAX_BUCKET_ENTRIES =
        std::min(static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()),
                 static_cast<uint64_t>(1) << 30);

    // Zero-fill holes in chunks to keep the temporary buffer bounded while avoiding tiny writes.
    static constexpr uint64_t ZERO_FILL_BLOCK_ENTRIES = 1024;
};

template <typename QuantTmpl, typename IOTmpl>
BucketDataCell<QuantTmpl, IOTmpl>::BucketDataCell(const QuantizerParamPtr& quantization_param,
                                                  const IOParamPtr& io_param,
                                                  const IndexCommonParam& common_param,
                                                  BucketIdType bucket_count,
                                                  bool use_residual)
    : BucketInterface(),
      datas_(common_param.allocator_.get(),
             AdjustBucketReadCacheParam(io_param, bucket_count),
             common_param),
      bucket_sizes_(bucket_count, 0, common_param.allocator_.get()),
      inner_ids_(bucket_count,
                 Vector<InnerIdType>(common_param.allocator_.get()),
                 common_param.allocator_.get()),
      bucket_mutexes_(bucket_count, common_param.allocator_.get()),
      allocator_(common_param.allocator_.get()),
      residual_bias_(bucket_count, Vector<float>(allocator_), allocator_),
      metric_(common_param.metric_),
      input_dim_(common_param.dim_) {
    this->bucket_count_ = bucket_count;
    this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
    this->code_size_ = quantizer_->GetCodeSize();
    this->use_residual_ = use_residual;
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*quantizer_));

    datas_.Resize(bucket_count);
}

template <typename QuantTmpl, typename IOTmpl>
float
BucketDataCell<QuantTmpl, IOTmpl>::query_one_by_id(
    const std::shared_ptr<Computer<QuantTmpl>>& computer,
    const BucketIdType& bucket_id,
    const InnerIdType& offset_id) {
    if (bucket_id >= this->bucket_count_ or bucket_id < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "visited invalid bucket id");
    }
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_bucket_id(bucket_id);
    if (offset_id >= this->bucket_sizes_[bucket_id]) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
    }
    if (this->inner_ids_[bucket_id][offset_id] == EMPTY_INNER_ID) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("visited empty offset in bucket: bucket_id={}, offset_id={}",
                        bucket_id,
                        offset_id));
    }
    float ret;
    bool need_release = false;
    const auto* codes =
        this->datas_[bucket_id].Read(code_size_, offset_id * code_size_, need_release);
    computer->ComputeDist(codes, &ret);
    if (need_release) {
        this->datas_[bucket_id].Release(codes);
    }

    if (use_residual_) {
        Vector<float> centroid(this->quantizer_->GetDim(), allocator_);
        strategy_->GetCentroid(bucket_id, centroid);
        auto ip_distance =
            FP32ComputeIP(computer->raw_query_.data(), centroid.data(), this->quantizer_->GetDim());
        if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
            ip_distance *= 2;
            ret = ret - residual_bias_[bucket_id][offset_id];
        }
        ret -= ip_distance;
    }
    return ret;
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::query(float* result_dists,
                                         Computer<QuantTmpl>* computer,
                                         const BucketIdType* bucket_ids,
                                         const InnerIdType* offset_ids,
                                         InnerIdType id_count,
                                         QueryContext* ctx) {
    if (id_count == 0) {
        return;
    }

    struct QueryRequest {
        BucketIdType bucket_id;
        InnerIdType offset_id;
        InnerIdType result_index;
    };

    Allocator* search_alloc = select_query_allocator(ctx, allocator_);
    Vector<QueryRequest> requests(search_alloc);
    requests.reserve(id_count);
    for (InnerIdType i = 0; i < id_count; ++i) {
        check_valid_bucket_id(bucket_ids[i]);
        requests.emplace_back(QueryRequest{bucket_ids[i], offset_ids[i], i});
    }
    std::sort(requests.begin(), requests.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.bucket_id != rhs.bucket_id) {
            return lhs.bucket_id < rhs.bucket_id;
        }
        if (lhs.offset_id != rhs.offset_id) {
            return lhs.offset_id < rhs.offset_id;
        }
        return lhs.result_index < rhs.result_index;
    });

    uint64_t io_count = 0;
    double io_cost_ms = 0.0F;
    Vector<InnerIdType> unique_offsets(search_alloc);
    Vector<uint64_t> read_sizes(search_alloc);
    Vector<uint64_t> read_offsets(search_alloc);
    unique_offsets.reserve(id_count);
    read_sizes.reserve(id_count);
    read_offsets.reserve(id_count);
    ByteBuffer codes(static_cast<uint64_t>(id_count) * code_size_, search_alloc);
    Vector<float> unique_dists(id_count, 0.0F, search_alloc);
    uint64_t group_begin = 0;
    while (group_begin < requests.size()) {
        const auto bucket_id = requests[group_begin].bucket_id;
        uint64_t group_end = group_begin + 1;
        while (group_end < requests.size() and requests[group_end].bucket_id == bucket_id) {
            ++group_end;
        }

        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        for (uint64_t i = group_begin; i < group_end; ++i) {
            const auto offset_id = requests[i].offset_id;
            if (offset_id >= this->bucket_sizes_[bucket_id]) {
                throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
            }
            if (this->inner_ids_[bucket_id][offset_id] == EMPTY_INNER_ID) {
                throw VsagException(
                    ErrorType::INVALID_ARGUMENT,
                    fmt::format("visited empty offset in bucket: bucket_id={}, offset_id={}",
                                bucket_id,
                                offset_id));
            }
        }

        unique_offsets.clear();
        for (uint64_t i = group_begin; i < group_end; ++i) {
            const auto offset_id = requests[i].offset_id;
            if (unique_offsets.empty() or unique_offsets.back() != offset_id) {
                unique_offsets.emplace_back(offset_id);
            }
        }

        read_sizes.clear();
        read_offsets.clear();
        for (const auto offset_id : unique_offsets) {
            if (not read_offsets.empty() and
                read_offsets.back() + read_sizes.back() ==
                    static_cast<uint64_t>(offset_id) * code_size_ and
                code_size_ <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) and
                read_sizes.back() <=
                    static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) - code_size_) {
                read_sizes.back() += code_size_;
            } else {
                read_sizes.emplace_back(code_size_);
                read_offsets.emplace_back(static_cast<uint64_t>(offset_id) * code_size_);
            }
        }

        bool read_success = false;
        double group_io_cost_ms = 0.0F;
        {
            Timer timer(group_io_cost_ms);
            read_success = this->datas_[bucket_id].MultiRead(
                codes.data, read_sizes.data(), read_offsets.data(), read_sizes.size());
        }
        if (not read_success) {
            throw VsagException(ErrorType::READ_ERROR, "failed to batch read bucket data");
        }
        io_count += read_sizes.size();
        io_cost_ms += group_io_cost_ms;

        computer->ScanBatchDists(unique_offsets.size(), codes.data, unique_dists.data());
        if (use_residual_) {
            Vector<float> centroid(this->quantizer_->GetDim(), search_alloc);
            strategy_->GetCentroid(bucket_id, centroid);
            auto ip_distance = FP32ComputeIP(
                computer->raw_query_.data(), centroid.data(), this->quantizer_->GetDim());
            if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
                ip_distance *= 2;
                for (uint64_t i = 0; i < unique_offsets.size(); ++i) {
                    unique_dists[i] -= residual_bias_[bucket_id][unique_offsets[i]];
                }
            }
            for (uint64_t i = 0; i < unique_offsets.size(); ++i) {
                unique_dists[i] -= ip_distance;
            }
        }

        uint64_t unique_index = 0;
        for (uint64_t i = group_begin; i < group_end; ++i) {
            if (i > group_begin and requests[i].offset_id != requests[i - 1].offset_id) {
                ++unique_index;
            }
            result_dists[requests[i].result_index] = unique_dists[unique_index];
        }
        group_begin = group_end;
    }

    if constexpr (not IOTmpl::InMemory) {
        if (ctx != nullptr and ctx->stats != nullptr) {
            ctx->stats->io_cnt.fetch_add(static_cast<uint32_t>(io_count),
                                         std::memory_order_relaxed);
            ctx->stats->io_time_ms.fetch_add(static_cast<uint32_t>(io_cost_ms),
                                             std::memory_order_relaxed);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::scan_bucket_by_id(float* result_dists,
                                                     Computer<QuantTmpl>* computer,
                                                     const BucketIdType& bucket_id,
                                                     InnerIdType* scanned_inner_ids,
                                                     InnerIdType max_scan_size,
                                                     InnerIdType* scanned_size) {
    if (bucket_id >= this->bucket_count_ or bucket_id < 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "visited invalid bucket id");
    }
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    constexpr InnerIdType scan_block_size = 32;
    InnerIdType offset = 0;
    this->check_valid_bucket_id(bucket_id);
    const auto bucket_size = std::min(this->bucket_sizes_[bucket_id], max_scan_size);
    if (scanned_size != nullptr) {
        *scanned_size = bucket_size;
    }
    if (scanned_inner_ids != nullptr) {
        std::copy_n(this->inner_ids_[bucket_id].begin(), bucket_size, scanned_inner_ids);
    }
    auto data_count = bucket_size;
    while (data_count > 0) {
        auto compute_count = std::min(data_count, scan_block_size);
        bool need_release = false;
        const auto* codes = this->datas_[bucket_id].Read(
            code_size_ * compute_count, offset * code_size_, need_release);
        computer->ScanBatchDists(compute_count, codes, result_dists + offset);
        if (need_release) {
            this->datas_[bucket_id].Release(codes);
        }
        data_count -= compute_count;
        offset += compute_count;
    }

    auto ip_distance = 0.0F;
    Vector<float> centroid(this->quantizer_->GetDim(), allocator_);
    if (use_residual_) {
        strategy_->GetCentroid(bucket_id, centroid);
        ip_distance =
            FP32ComputeIP(computer->raw_query_.data(), centroid.data(), this->quantizer_->GetDim());
        if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
            ip_distance *= 2;
            FP32Sub(result_dists, residual_bias_[bucket_id].data(), result_dists, offset);
        }
        // TODO(inabao): optimize this loop with simd
        for (InnerIdType i = 0; i < offset; ++i) {
            result_dists[i] -= ip_distance;
        }
    }
    for (InnerIdType i = 0; i < offset; ++i) {
        if (this->inner_ids_[bucket_id][i] == EMPTY_INNER_ID) {
            result_dists[i] = std::numeric_limits<float>::max();
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
ComputerInterfacePtr
BucketDataCell<QuantTmpl, IOTmpl>::FactoryComputer(const void* query) {
    const auto* float_query = reinterpret_cast<const float*>(query);
    auto computer = this->quantizer_->FactoryComputer();
    auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
    if (use_residual_) {
        comp->raw_query_.resize(this->quantizer_->GetDim());
        if (metric_ == MetricType::METRIC_TYPE_COSINE) {
            Normalize(float_query, comp->raw_query_.data(), this->quantizer_->GetDim());
        } else {
            memcpy(
                comp->raw_query_.data(), float_query, sizeof(float) * this->quantizer_->GetDim());
        }
        float_query = comp->raw_query_.data();
    }
    comp->SetQuery(float_query);
    return computer;
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Train(const void* data, uint64_t count) {
    Vector<float> train_data_buffer(allocator_);
    if (use_residual_) {
        auto data_ptr = static_cast<const float*>(data);
        auto dim = this->quantizer_->GetDim();
        train_data_buffer.resize(count * dim);
        if (metric_ == MetricType::METRIC_TYPE_COSINE) {
            for (int i = 0; i < count; ++i) {
                Normalize(data_ptr + i * dim, train_data_buffer.data() + i * dim, dim);
            }
            data_ptr = train_data_buffer.data();
        }
        Vector<float> centroid(this->quantizer_->GetDim(), allocator_);
        auto buckets = strategy_->ClassifyDatas(data_ptr, count, 1, nullptr);
        for (int i = 0; i < count; ++i) {
            strategy_->GetCentroid(buckets[i], centroid);
            for (int j = 0; j < dim; ++j) {
                train_data_buffer[i * dim + j] = data_ptr[i * dim + j] - centroid[j];
            }
        }
        data = train_data_buffer.data();
    }
    this->quantizer_->Train(reinterpret_cast<const float*>(data), count);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::encode_vector(const void* vector,
                                                 BucketIdType bucket_id,
                                                 uint8_t* codes,
                                                 float& res_score) {
    auto vector_ptr = static_cast<const float*>(vector);
    res_score = 0.0F;
    if (not use_residual_) {
        this->quantizer_->EncodeOne(vector_ptr, codes);
        return;
    }

    Vector<float> centroid(this->quantizer_->GetDim(), this->allocator_);
    Vector<float> sub_data(this->quantizer_->GetDim(), this->allocator_);
    Vector<float> normalize_data(this->quantizer_->GetDim(), this->allocator_);
    strategy_->GetCentroid(bucket_id, centroid);
    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(vector_ptr, normalize_data.data(), quantizer_->GetDim());
        vector_ptr = normalize_data.data();
    }
    FP32Sub(vector_ptr, centroid.data(), sub_data.data(), quantizer_->GetDim());
    vector_ptr = sub_data.data();
    this->quantizer_->EncodeOne(vector_ptr, codes);
    if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->quantizer_->DecodeOne(codes, normalize_data.data());
        res_score =
            -2 * FP32ComputeIP(centroid.data(), normalize_data.data(), this->quantizer_->GetDim()) -
            FP32ComputeIP(centroid.data(), centroid.data(), this->quantizer_->GetDim());
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::check_valid_bucket_capacity(uint64_t next_size,
                                                               bool need_resize) {
    if (need_resize && (next_size > MAX_BUCKET_ENTRIES ||
                        next_size > std::numeric_limits<size_t>::max() / sizeof(InnerIdType) ||
                        (use_residual_ && metric_ == MetricType::METRIC_TYPE_L2SQR &&
                         next_size > std::numeric_limits<size_t>::max() / sizeof(float)) ||
                        ZERO_FILL_BLOCK_ENTRIES > std::numeric_limits<uint64_t>::max() /
                                                      static_cast<uint64_t>(code_size_))) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "invalid offset id for bucket: exceeds maximum bucket capacity");
    }
}

template <typename QuantTmpl, typename IOTmpl>
InnerIdType
BucketDataCell<QuantTmpl, IOTmpl>::InsertVector(const void* vector,
                                                BucketIdType bucket_id,
                                                InnerIdType inner_id) {
    check_valid_bucket_id(bucket_id);
    if (inner_id == EMPTY_INNER_ID) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid inner id for bucket");
    }

    InnerIdType offset_id;
    float res_score = 0.0F;
    ByteBuffer codes(static_cast<uint64_t>(code_size_), this->allocator_);
    encode_vector(vector, bucket_id, codes.data, res_score);
    {
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        offset_id = this->bucket_sizes_[bucket_id];
        if (offset_id >= std::numeric_limits<InnerIdType>::max() ||
            static_cast<uint64_t>(offset_id) >
                std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(code_size_)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
        }
        auto next_size = static_cast<uint64_t>(offset_id) + 1;
        check_valid_bucket_capacity(next_size, this->inner_ids_[bucket_id].size() < next_size);
        this->datas_[bucket_id].Write(
            codes.data,
            code_size_,
            static_cast<uint64_t>(offset_id) * static_cast<uint64_t>(code_size_));
        if (this->inner_ids_[bucket_id].size() < next_size) {
            this->inner_ids_[bucket_id].resize(next_size, EMPTY_INNER_ID);
            if (use_residual_ && metric_ == MetricType::METRIC_TYPE_L2SQR) {
                this->residual_bias_[bucket_id].resize(next_size, 0.0F);
            }
        }
        inner_ids_[bucket_id][offset_id] = inner_id;
        if (use_residual_ && metric_ == MetricType::METRIC_TYPE_L2SQR) {
            residual_bias_[bucket_id][offset_id] = res_score;
        }
        this->bucket_sizes_[bucket_id] = static_cast<InnerIdType>(next_size);
    }
    return offset_id;
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::BatchInsertVector(const void* vectors,
                                                     const BucketIdType* bucket_ids,
                                                     const InnerIdType* inner_ids,
                                                     InnerIdType count,
                                                     InnerIdType* out_offsets) {
    if (count == 0) {
        return;
    }
    if (vectors == nullptr or bucket_ids == nullptr or inner_ids == nullptr or
        out_offsets == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "batch bucket insert requires non-null inputs and output");
    }
    if (this->bucket_count_ <= 0 or input_dim_ == 0 or code_size_ == 0 or
        (use_residual_ and strategy_ == nullptr)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid batch bucket insert size");
    }
    const auto bucket_count = static_cast<uint64_t>(this->bucket_count_);
    const auto batch_count = static_cast<uint64_t>(count);
    const auto max_allocation_size = std::numeric_limits<size_t>::max();
    if (bucket_count == std::numeric_limits<uint64_t>::max() or
        bucket_count + 1 > max_allocation_size / sizeof(uint64_t) or
        batch_count > max_allocation_size / sizeof(uint64_t) or
        batch_count > max_allocation_size / sizeof(InnerIdType) or
        batch_count > max_allocation_size / sizeof(float) / input_dim_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid batch bucket insert size");
    }

    Vector<uint64_t> bucket_counts(bucket_count, 0, this->allocator_);
    for (uint64_t i = 0; i < batch_count; ++i) {
        check_valid_bucket_id(bucket_ids[i]);
        if (inner_ids[i] == EMPTY_INNER_ID) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid inner id for bucket");
        }
        ++bucket_counts[bucket_ids[i]];
    }

    Vector<uint64_t> bucket_begins(bucket_count + 1, 0, this->allocator_);
    for (uint64_t bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        bucket_begins[bucket_id + 1] = bucket_begins[bucket_id] + bucket_counts[bucket_id];
    }
    Vector<uint64_t> next_positions(
        bucket_begins.begin(), bucket_begins.end() - 1, this->allocator_);
    Vector<InnerIdType> order(batch_count, 0, this->allocator_);
    for (uint64_t i = 0; i < batch_count; ++i) {
        const auto position = next_positions[bucket_ids[i]]++;
        order[position] = static_cast<InnerIdType>(i);
    }

    const auto* float_vectors = static_cast<const float*>(vectors);
    for (uint64_t bucket_id = 0; bucket_id < bucket_count; ++bucket_id) {
        const auto bucket_insert_count = bucket_counts[bucket_id];
        if (bucket_insert_count == 0) {
            continue;
        }
        if (bucket_insert_count > max_allocation_size / static_cast<uint64_t>(code_size_)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid batch bucket insert size");
        }

        const auto group_begin = bucket_begins[bucket_id];
        ByteBuffer codes(bucket_insert_count * static_cast<uint64_t>(code_size_), this->allocator_);
        Vector<float> residual_scores(this->allocator_);
        if (use_residual_ and metric_ == MetricType::METRIC_TYPE_L2SQR) {
            residual_scores.resize(bucket_insert_count);
        }
        for (uint64_t local_index = 0; local_index < bucket_insert_count; ++local_index) {
            const auto position = group_begin + local_index;
            const auto input_index = order[position];
            float residual_score = 0.0F;
            encode_vector(float_vectors + input_index * input_dim_,
                          static_cast<BucketIdType>(bucket_id),
                          codes.data + local_index * static_cast<uint64_t>(code_size_),
                          residual_score);
            if (not residual_scores.empty()) {
                residual_scores[local_index] = residual_score;
            }
        }

        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        const auto append_begin = static_cast<uint64_t>(this->bucket_sizes_[bucket_id]);
        if (bucket_insert_count >
            static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()) - append_begin) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
        }
        const auto next_size = append_begin + bucket_insert_count;
        if (next_size > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(code_size_)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
        }
        check_valid_bucket_capacity(next_size, this->inner_ids_[bucket_id].size() < next_size);
        this->datas_[bucket_id].Write(codes.data,
                                      bucket_insert_count * static_cast<uint64_t>(code_size_),
                                      append_begin * static_cast<uint64_t>(code_size_));
        if (this->inner_ids_[bucket_id].size() < next_size) {
            this->inner_ids_[bucket_id].resize(next_size, EMPTY_INNER_ID);
        }
        if (not residual_scores.empty() and this->residual_bias_[bucket_id].size() < next_size) {
            this->residual_bias_[bucket_id].resize(next_size, 0.0F);
        }
        for (uint64_t local_index = 0; local_index < bucket_insert_count; ++local_index) {
            const auto position = group_begin + local_index;
            const auto input_index = order[position];
            const auto offset = append_begin + local_index;
            this->inner_ids_[bucket_id][offset] = inner_ids[input_index];
            if (not residual_scores.empty()) {
                this->residual_bias_[bucket_id][offset] = residual_scores[local_index];
            }
            out_offsets[input_index] = static_cast<InnerIdType>(offset);
        }
        this->bucket_sizes_[bucket_id] =
            static_cast<InnerIdType>(append_begin + bucket_insert_count);
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::InsertVectorWithOffset(const void* vector,
                                                          BucketIdType bucket_id,
                                                          InnerIdType inner_id,
                                                          InnerIdType offset_id) {
    check_valid_bucket_id(bucket_id);
    if (inner_id == EMPTY_INNER_ID) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid inner id for bucket");
    }
    if constexpr (std::is_signed_v<InnerIdType>) {
        if (offset_id < 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
        }
    }
    if (offset_id >= std::numeric_limits<InnerIdType>::max() ||
        static_cast<uint64_t>(offset_id) >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(code_size_)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
    }

    float res_score = 0.0F;
    ByteBuffer codes(static_cast<uint64_t>(code_size_), this->allocator_);
    encode_vector(vector, bucket_id, codes.data, res_score);
    {
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        auto next_size = static_cast<uint64_t>(offset_id) + 1;
        auto need_resize = this->inner_ids_[bucket_id].size() < next_size;
        check_valid_bucket_capacity(next_size, need_resize);
        if (need_resize) {
            this->inner_ids_[bucket_id].resize(next_size, EMPTY_INNER_ID);
            if (use_residual_ && metric_ == MetricType::METRIC_TYPE_L2SQR) {
                this->residual_bias_[bucket_id].resize(next_size, 0.0F);
            }
        }
        if (offset_id > this->bucket_sizes_[bucket_id]) {
            auto zero_fill_entries = std::min<uint64_t>(
                static_cast<uint64_t>(offset_id - this->bucket_sizes_[bucket_id]),
                ZERO_FILL_BLOCK_ENTRIES);
            auto zero_fill_bytes = zero_fill_entries * static_cast<uint64_t>(code_size_);
            ByteBuffer zero_codes(zero_fill_bytes, this->allocator_);
            std::fill_n(zero_codes.data, zero_fill_bytes, 0);
            for (auto hole_offset = this->bucket_sizes_[bucket_id]; hole_offset < offset_id;) {
                auto fill_entries = std::min<uint64_t>(
                    static_cast<uint64_t>(offset_id - hole_offset), ZERO_FILL_BLOCK_ENTRIES);
                this->datas_[bucket_id].Write(
                    zero_codes.data,
                    fill_entries * static_cast<uint64_t>(code_size_),
                    static_cast<uint64_t>(hole_offset) * static_cast<uint64_t>(code_size_));
                hole_offset += static_cast<InnerIdType>(fill_entries);
            }
        }
        this->datas_[bucket_id].Write(
            codes.data,
            code_size_,
            static_cast<uint64_t>(offset_id) * static_cast<uint64_t>(code_size_));
        this->inner_ids_[bucket_id][offset_id] = inner_id;
        if (use_residual_ && metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id][offset_id] = res_score;
        }
        this->bucket_sizes_[bucket_id] =
            std::max(this->bucket_sizes_[bucket_id], static_cast<InnerIdType>(next_size));
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Serialize(StreamWriter& writer) {
    BucketInterface::Serialize(writer);
    quantizer_->Serialize(writer);
    for (BucketIdType i = 0; i < this->bucket_count_; ++i) {
        datas_[i].Serialize(writer);
        StreamWriter::WriteVector(writer, inner_ids_[i]);
        if (use_residual_) {
            StreamWriter::WriteVector(writer, residual_bias_[i]);
        }
    }
    StreamWriter::WriteVector(writer, this->bucket_sizes_);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::Deserialize(LvalueOrRvalue<StreamReader> reader) {
    BucketInterface::Deserialize(reader);
    quantizer_->Deserialize(reader);
    this->backend_ =
        QuantizerDistanceBackend<QuantTmpl>::Get(static_cast<const QuantTmpl&>(*quantizer_));
    for (BucketIdType i = 0; i < this->bucket_count_; ++i) {
        datas_[i].Deserialize(reader);
        StreamReader::ReadVector(reader, inner_ids_[i]);
        if (use_residual_) {
            StreamReader::ReadVector(reader, residual_bias_[i]);
        }
    }
    StreamReader::ReadVector(reader, this->bucket_sizes_);
    if (this->bucket_sizes_.size() != static_cast<uint64_t>(this->bucket_count_)) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "serialized bucket size vector does not match bucket count");
    }
    for (BucketIdType i = 0; i < this->bucket_count_; ++i) {
        const auto bucket_size = static_cast<uint64_t>(this->bucket_sizes_[i]);
        if (this->inner_ids_[i].size() < bucket_size) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("serialized bucket {} inner id count is smaller than bucket size", i));
        }
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR and
            this->residual_bias_[i].size() < bucket_size) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("serialized bucket {} residual bias count is smaller than bucket size",
                            i));
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::package_fastscan() {
    ByteBuffer buffer(code_size_ * 32, this->allocator_);
    for (int64_t i = 0; i < this->bucket_count_; ++i) {
        auto bucket_size = this->bucket_sizes_[i];
        if (bucket_size == 0) {
            continue;
        }
        bool need_release = false;
        const auto* codes = this->datas_[i].Read(code_size_ * bucket_size, 0, need_release);
        InnerIdType begin = 0;
        while (begin < bucket_size) {
            auto valid_size = bucket_size - begin;
            if (valid_size > 32) {
                valid_size = 32;
            }
            quantizer_->Package32(codes + begin * code_size_, buffer.data, valid_size);
            this->datas_[i].Write(buffer.data, code_size_ * 32, begin * code_size_);
            begin += 32;
        }
        if (need_release) {
            this->datas_[i].Release(codes);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::unpack_fastscan() {
    ByteBuffer buffer(code_size_ * 32, this->allocator_);
    for (int64_t i = 0; i < this->bucket_count_; ++i) {
        auto bucket_size = (this->bucket_sizes_[i] + 31) / 32 * 32;
        if (bucket_size == 0) {
            continue;
        }
        bool need_release = false;
        const auto* codes = this->datas_[i].Read(code_size_ * bucket_size, 0, need_release);
        InnerIdType begin = 0;
        while (begin < bucket_size) {
            const uint8_t* src_block = codes + begin * code_size_;
            quantizer_->Unpack32(src_block, buffer.data);
            this->datas_[i].Write(buffer.data, code_size_ * 32, begin * code_size_);
            begin += 32;
        }
        if (need_release) {
            this->datas_[i].Release(codes);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::ExportModel(const BucketInterfacePtr& other) const {
    std::stringstream ss;
    IOStreamWriter writer(ss);
    this->quantizer_->Serialize(writer);
    ss.seekg(0, std::ios::beg);
    IOStreamReader reader(ss);
    auto ptr = std::dynamic_pointer_cast<BucketDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Export model's bucket datacell failed");
    }
    ptr->quantizer_->Deserialize(reader);
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::MergeOther(const BucketInterfacePtr& other, InnerIdType bias) {
    auto ptr = std::dynamic_pointer_cast<BucketDataCell<QuantTmpl, IOTmpl>>(other);
    if (ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Merge other's bucket datacell failed");
    }
    for (int i = 0; i < ptr->bucket_count_; ++i) {
        bool need_release = false;
        if (ptr->bucket_sizes_[i] == 0) {
            continue;
        }
        auto* other_data =
            ptr->datas_[i].Read(ptr->bucket_sizes_[i] * this->code_size_, 0, need_release);
        this->datas_[i].Write(other_data,
                              ptr->bucket_sizes_[i] * this->code_size_,
                              this->bucket_sizes_[i] * this->code_size_);
        if (need_release) {
            ptr->datas_[i].Release(other_data);
        }
        this->bucket_sizes_[i] += ptr->bucket_sizes_[i];
        this->inner_ids_[i].reserve(this->bucket_sizes_[i]);
        for (auto id : ptr->inner_ids_[i]) {
            this->inner_ids_[i].emplace_back(id == EMPTY_INNER_ID ? EMPTY_INNER_ID : id + bias);
        }
    }
}

template <typename QuantTmpl, typename IOTmpl>
void
BucketDataCell<QuantTmpl, IOTmpl>::GetCodesById(BucketIdType bucket_id,
                                                InnerIdType offset_id,
                                                uint8_t* data) const {
    if (bucket_id >= this->bucket_count_) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("Get code by inner id failed: bucket id ({}) is invalid", bucket_id));
    }
    if (offset_id >= this->bucket_sizes_[bucket_id]) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("Get code by inner id failed: offset id ({}) is invalid", offset_id));
    }
    this->datas_[bucket_id].Read(this->code_size_, offset_id * this->code_size_, data);
}

template <typename QuantTmpl, typename IOTmpl>
float
BucketDataCell<QuantTmpl, IOTmpl>::ComputePairVectors(BucketIdType bucket_id,
                                                      InnerIdType id1,
                                                      InnerIdType id2) {
    if (bucket_id >= this->bucket_count_) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("ComputePairVectors failed: bucket id ({}) is invalid", bucket_id));
    }
    if (id1 >= this->bucket_sizes_[bucket_id] || id2 >= this->bucket_sizes_[bucket_id]) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("ComputePairVectors failed: offset id is invalid"));
    }
    if (this->inner_ids_[bucket_id][id1] == EMPTY_INNER_ID ||
        this->inner_ids_[bucket_id][id2] == EMPTY_INNER_ID) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            "ComputePairVectors failed: cannot compute distance for deleted vector");
    }
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    constexpr size_t kStackThreshold = 256;
    uint8_t stack_buf1[kStackThreshold];
    uint8_t stack_buf2[kStackThreshold];

    uint8_t *code1_ptr, *code2_ptr;
    std::vector<uint8_t> heap_buf1, heap_buf2;

    if (this->code_size_ <= kStackThreshold) {
        code1_ptr = stack_buf1;
        code2_ptr = stack_buf2;
    } else {
        heap_buf1.resize(this->code_size_);
        heap_buf2.resize(this->code_size_);
        code1_ptr = heap_buf1.data();
        code2_ptr = heap_buf2.data();
    }

    this->datas_[bucket_id].Read(this->code_size_, id1 * this->code_size_, code1_ptr);
    this->datas_[bucket_id].Read(this->code_size_, id2 * this->code_size_, code2_ptr);
    return this->quantizer_->Compute(code1_ptr, code2_ptr);
}
}  // namespace vsag
