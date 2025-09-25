
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

#include "diskann.h"

#include <local_file_reader.h>

#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

#include "datacell/flatten_datacell.h"
#include "dataset_impl.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "storage/empty_index_binary_set.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/timer.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

const static float MACRO_TO_MILLI = 1000;
const static int64_t DATA_LIMIT = 2;
const static size_t MAXIMAL_BEAM_SEARCH = 64;
const static size_t MINIMAL_BEAM_SEARCH = 1;
const static int MINIMAL_R = 8;
const static int MAXIMAL_R = 64;
const static int VECTOR_PER_BLOCK = 1;
const static float GRAPH_SLACK = 1.3 * 1.05;
const static size_t MINIMAL_SECTOR_LEN = 4096;
const static std::string BUILD_STATUS = "status";
const static std::string BUILD_CURRENT_ROUND = "round";
const static std::string BUILD_NODES = "builded_nodes";
const static std::string BUILD_FAILED_LOC = "failed_loc";

template <typename T>
Binary
to_binary(T& value) {
    Binary binary;
    binary.size = sizeof(T);
    binary.data = std::shared_ptr<int8_t[]>(new int8_t[binary.size]);
    std::memcpy(binary.data.get(), &value, binary.size);
    return binary;
}

template <typename T>
T
from_binary(const Binary& binary) {
    T value;
    std::memcpy(&value, binary.data.get(), binary.size);
    return value;
}

class LocalMemoryReader : public Reader {
public:
    LocalMemoryReader(std::stringstream& file) {
        file_ << file.rdbuf();
        file_.seekg(0, std::ios::end);
        size_ = file_.tellg();
    }

    ~LocalMemoryReader() override = default;

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(static_cast<int64_t>(offset), std::ios::beg);
        file_.read((char*)dest, static_cast<int64_t>(len));
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        if (not pool_) {
            pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        }
        pool_->GeneralEnqueue([this,  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
                               offset,
                               len,
                               dest,
                               callback]() {
            this->Read(offset, len, dest);
            callback(IOErrorCode::IO_SUCCESS, "success");
        });
    }

    uint64_t
    Size() const override {
        return size_;
    }

private:
    std::stringstream file_;
    uint64_t size_;
    std::mutex mutex_;
    std::shared_ptr<SafeThreadPool> pool_;
};

class IStreamReader : public Reader {
public:
    IStreamReader(std::istream& in_stream, int64_t base_offset, int64_t size)
        : in_stream_(in_stream), base_offset_(base_offset), size_(size) {
    }

    ~IStreamReader() override = default;

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        in_stream_.seekg(static_cast<std::streamsize>(base_offset_ + offset), std::ios::beg);
        in_stream_.read((char*)dest, static_cast<std::streamsize>(len));
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        Read(offset, len, dest);
        callback(IOErrorCode::IO_SUCCESS, "success");
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return size_;
    }

private:
    std::istream& in_stream_;
    int64_t base_offset_{0};
    uint64_t size_{0};
    std::mutex mutex_;
};

uint64_t
get_stringstream_size(const std::stringstream& stream) {
    std::streambuf* buf = stream.rdbuf();
    std::streamsize size = buf->pubseekoff(
        0, std::stringstream::end, std::stringstream::in);  // get the stream buffer size
    buf->pubseekpos(0, std::stringstream::in);              // reset pointer pos
    return size;
}

Binary
convert_stream_to_binary(const std::stringstream& stream) {
    std::streambuf* buf = stream.rdbuf();
    auto size = static_cast<std::streamsize>(get_stringstream_size(stream));
    std::shared_ptr<int8_t[]> binary_data(new int8_t[size]);
    buf->sgetn((char*)binary_data.get(), size);
    Binary binary{
        .data = binary_data,
        .size = (size_t)size,
    };
    return std::move(binary);
}

void
convert_binary_to_stream(const Binary& binary, std::stringstream& stream) {
    stream.str("");
    if (binary.data && binary.size > 0) {
        stream.write((const char*)binary.data.get(), static_cast<int64_t>(binary.size));
    }
}

void
copy_istream_to_stringstream(std::stringstream& to, std::istream& from, uint64_t size) {
    uint64_t done_size = 0;

    const int32_t buffer_size = 1024;
    char buffer[buffer_size] = {};

    while (done_size < size) {
        auto remain = size - done_size;
        auto copy_size =
            static_cast<std::streamsize>(remain / buffer_size > 0 ? buffer_size : remain);
        from.read(buffer, copy_size);
        to.write(buffer, copy_size);

        done_size += copy_size;
    }
}

void
copy_stringstream_to_ostream(std::ostream& to, std::stringstream& from, uint64_t size) {
    uint64_t done_size = 0;

    const int32_t buffer_size = 1024;
    char buffer[buffer_size] = {};

    while (done_size < size) {
        auto remain = size - done_size;
        auto copy_size =
            static_cast<std::streamsize>(remain / buffer_size > 0 ? buffer_size : remain);
        from.read(buffer, copy_size);
        to.write(buffer, copy_size);

        done_size += copy_size;
    }
}

DiskANN::DiskANN(DiskannParameters& diskann_params, const IndexCommonParam& index_common_param)
    : metric_(diskann_params.metric),
      L_(static_cast<int32_t>(diskann_params.ef_construction)),
      R_(static_cast<int32_t>(diskann_params.max_degree)),
      p_val_(diskann_params.pq_sample_rate),
      disk_pq_dims_(diskann_params.pq_dims),
      dim_(index_common_param.dim_),
      preload_(diskann_params.use_preload),
      use_reference_(diskann_params.use_reference),
      use_opq_(diskann_params.use_opq),
      use_bsa_(diskann_params.use_bsa),
      diskann_params_(diskann_params),
      common_param_(index_common_param) {
    status_ = IndexStatus::EMPTY;
    batch_read_ = [&](const std::vector<read_request>& requests,
                      bool async,
                      const CallBack& callBack) -> void {
        if (async) {
            for (const auto& req : requests) {
                auto [offset, len, dest] = req;
                disk_layout_reader_->AsyncRead(offset, len, dest, callBack);
            }
        } else {
            std::atomic<bool> succeed(true);
            std::string error_message;
            std::atomic<int> counter(static_cast<int>(requests.size()));
            std::promise<void> total_promise;
            auto total_future = total_promise.get_future();
            for (const auto& req : requests) {
                auto [offset, len, dest] = req;
                auto callback = [&counter, &total_promise, &succeed, &error_message](
                                    IOErrorCode code, const std::string& message) {
                    if (code != vsag::IOErrorCode::IO_SUCCESS) {
                        bool expected = true;
                        if (succeed.compare_exchange_strong(expected, false)) {
                            error_message = message;
                        }
                    }
                    if (--counter == 0) {
                        total_promise.set_value();
                    }
                };
                disk_layout_reader_->AsyncRead(offset, len, dest, callback);
            }
            total_future.wait();
            if (not succeed) {
                throw VsagException(ErrorType::READ_ERROR, "failed to read diskann index");
            }
        }
    };

    R_ = std::min(MAXIMAL_R, std::max(MINIMAL_R, R_));

    // When the length of the vector is too long, set sector_len_ to the size of storing a vector along with its linkage list.
    sector_len_ = std::max(
        MINIMAL_SECTOR_LEN,                                                           // NOLINT
        (size_t)(dim_ * sizeof(float) + (R_ * GRAPH_SLACK + 1) * sizeof(uint32_t)) *  // NOLINT
            VECTOR_PER_BLOCK);                                                        // NOLINT

    this->feature_list_ = std::make_shared<IndexFeatureList>();
    this->init_feature_list();
}

tl::expected<std::vector<int64_t>, Error>
DiskANN::build(const DatasetPtr& base) {
    try {
        if (base->GetNumElements() <= 0) {
            empty_index_ = true;
            return std::vector<int64_t>();
        }

        auto data_dim = base->GetDim();
        CHECK_ARGUMENT(data_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", data_dim, dim_));

        std::unique_lock lock(rw_mutex_);

        if (this->index_) {
            LOG_ERROR_AND_RETURNS(ErrorType::BUILD_TWICE, "failed to build index: build twice");
        }

        const auto* vectors = base->GetFloat32Vectors();
        const auto* ids = base->GetIds();
        auto data_num = base->GetNumElements();

        std::vector<size_t> failed_locs;
        if (diskann_params_.graph_type == GRAPH_TYPE_ODESCENT) {
            SlowTaskTimer t("odescent build full (graph)");
            FlattenDataCellParamPtr flatten_param =
                std::make_shared<vsag::FlattenDataCellParameter>();
            flatten_param->quantizer_parameter = std::make_shared<FP32QuantizerParameter>();
            flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
            vsag::FlattenInterfacePtr flatten_interface_ptr =
                vsag::FlattenInterface::MakeInstance(flatten_param, this->common_param_);
            flatten_interface_ptr->Train(vectors, data_num);
            flatten_interface_ptr->BatchInsertVector(vectors, data_num);
            auto param = std::make_shared<ODescentParameter>();
            param->max_degree = 2LL * R_;
            param->alpha = diskann_params_.alpha;
            param->turn = diskann_params_.turn;
            param->sample_rate = diskann_params_.sample_rate;
            vsag::ODescent graph(param,
                                 flatten_interface_ptr,
                                 common_param_.allocator_.get(),
                                 common_param_.thread_pool_.get());
            graph.Build();
            graph.SaveGraph(graph_stream_);
            auto data_num_int32 = static_cast<int32_t>(data_num);
            auto data_dim_int32 = static_cast<int32_t>(data_dim);
            tag_stream_.write((char*)&data_num_int32, sizeof(data_num_int32));
            tag_stream_.write((char*)&data_dim_int32, sizeof(data_dim_int32));
            tag_stream_.write((char*)ids, static_cast<std::streamsize>(data_num * sizeof(ids)));
        } else if (diskann_params_.graph_type == DISKANN_GRAPH_TYPE_VAMANA) {
            SlowTaskTimer t("diskann build full (graph)");
            // build graph
            build_index_ = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
                metric_, data_dim, data_num, false, true, false, false, 0, false);
            std::vector<int64_t> tags(ids, ids + data_num);
            auto index_build_params =
                diskann::IndexWriteParametersBuilder(L_, R_)
                    .with_num_threads(Options::Instance().num_threads_building())
                    .build();
            failed_locs =
                build_index_->build(vectors, data_num, index_build_params, tags, use_reference_);
            build_index_->save(graph_stream_, tag_stream_);
            build_index_.reset();
        }
        {
            SlowTaskTimer t("diskann build full (pq)");
            diskann::generate_disk_quantized_data<float>(vectors,
                                                         data_num,
                                                         data_dim,
                                                         failed_locs,
                                                         pq_pivots_stream_,
                                                         disk_pq_compressed_vectors_,
                                                         metric_,
                                                         p_val_,
                                                         disk_pq_dims_,
                                                         use_opq_,
                                                         use_bsa_);
        }
        {
            SlowTaskTimer t("diskann build full (disk layout)");
            diskann::create_disk_layout<float>(vectors,
                                               data_num,
                                               data_dim,
                                               failed_locs,
                                               graph_stream_,
                                               disk_layout_stream_,
                                               sector_len_,
                                               metric_);
        }

        std::vector<int64_t> failed_ids;
        std::transform(failed_locs.begin(),
                       failed_locs.end(),
                       std::back_inserter(failed_ids),
                       [&ids](const auto& index) { return ids[index]; });

        disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);
        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(
            reader_, metric_, sector_len_, dim_, use_bsa_));
        index_->load_from_separate_paths(
            pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);
        if (preload_) {
            index_->load_graph(graph_stream_);
        } else {
            graph_stream_.clear();
        }
        status_ = IndexStatus::MEMORY;
        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to build(invalid argument): ", e.what());
    }
}

tl::expected<DatasetPtr, Error>
DiskANN::knn_search(const DatasetPtr& query,
                    int64_t k,
                    const std::string& parameters,
                    const BitsetPtr& invalid) const {
    // check filter
    std::function<bool(int64_t)> filter = nullptr;
    if (invalid != nullptr) {
        filter = [invalid](int64_t offset) -> bool {
            int64_t bit_index = offset & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
    }
    return this->knn_search(query, k, parameters, filter);
};

tl::expected<DatasetPtr, Error>
DiskANN::knn_search(const DatasetPtr& query,
                    int64_t k,
                    const std::string& parameters,
                    const std::function<bool(int64_t)>& filter) const {
#ifndef ENABLE_TESTS
    SlowTaskTimer t("diskann knnsearch", 200);
#endif

    // cannot perform search on empty index
    if (empty_index_) {
        return DatasetImpl::MakeEmptyDataset();
    }

    try {
        if (!index_) {
            LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY,
                                  "failed to search: diskann index is empty");
        }

        // check query vector
        auto query_num = query->GetNumElements();
        auto query_dim = query->GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check k
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k))
        k = std::min(k, GetNumElements());

        // check search parameters
        auto params = DiskannSearchParameters::FromJson(parameters);
        int64_t ef_search = params.ef_search;
        size_t beam_search = params.beam_search;
        int64_t io_limit = params.io_limit;
        bool reorder = params.use_reorder;

        // ensure that in the topK scenario, ef_search > io_limit and io_limit > k.
        if (reorder && preload_) {
            ef_search = std::max(2 * k, ef_search);
            io_limit = std::max(2 * k, io_limit);
        } else {
            ef_search = std::max(ef_search, k);
            io_limit = std::min(ef_search, std::max(io_limit, k));
        }
        io_limit = std::min(io_limit, GetNumElements());
        ef_search = std::min(ef_search, GetNumElements());

        beam_search = std::min(beam_search, MAXIMAL_BEAM_SEARCH);
        beam_search = std::max(beam_search, MINIMAL_BEAM_SEARCH);

        uint64_t labels[query_num * k];
        auto* distances = new float[query_num * k];
        auto* ids = new int64_t[query_num * k];
        diskann::QueryStats query_stats[query_num];
        for (int i = 0; i < query_num; i++) {
            try {
                double time_cost = 0;
                {
                    std::shared_lock lock(rw_mutex_);
                    Timer timer(time_cost);
                    if (preload_) {
                        if (params.use_async_io) {
                            k = index_->cached_beam_search_async(
                                query->GetFloat32Vectors() + i * dim_,
                                k,
                                ef_search,
                                labels + i * k,
                                distances + i * k,
                                beam_search,
                                filter,
                                io_limit,
                                reorder,
                                query_stats + i);
                        } else {
                            k = index_->cached_beam_search_memory(
                                query->GetFloat32Vectors() + i * dim_,
                                k,
                                ef_search,
                                labels + i * k,
                                distances + i * k,
                                beam_search,
                                filter,
                                io_limit,
                                reorder,
                                query_stats + i);
                        }
                    } else {
                        k = index_->cached_beam_search(query->GetFloat32Vectors() + i * dim_,
                                                       k,
                                                       ef_search,
                                                       labels + i * k,
                                                       distances + i * k,
                                                       beam_search,
                                                       filter,
                                                       io_limit,
                                                       false,
                                                       query_stats + i);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    result_queues_[STATSTIC_KNN_IO].Push(static_cast<float>(query_stats[i].n_ios));
                    result_queues_[STATSTIC_KNN_TIME].Push(static_cast<float>(time_cost));
                    result_queues_[STATSTIC_KNN_IO_TIME].Push(
                        (query_stats[i].io_us / static_cast<float>(query_stats[i].n_ios)) /
                        MACRO_TO_MILLI);
                }

            } catch (const std::runtime_error& e) {
                delete[] distances;
                delete[] ids;
                LOG_ERROR_AND_RETURNS(ErrorType::INTERNAL_ERROR,
                                      "failed to perform knn search on diskann: ",
                                      e.what());
            }
        }

        auto result = Dataset::Make();
        result->NumElements(query->GetNumElements())->Dim(0);

        if (k == 0) {
            delete[] distances;
            delete[] ids;
            return std::move(result);
        }
        for (int i = 0; i < query_num * k; ++i) {
            ids[i] = static_cast<int64_t>(labels[i]);
        }

        result->NumElements(query_num)->Dim(k)->Distances(distances)->Ids(ids);
        return std::move(result);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform knn_search(invalid argument): ",
                              e.what());
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to perform knn_search(not enough memory): ",
                              e.what());
    }
}

tl::expected<DatasetPtr, Error>
DiskANN::range_search(const DatasetPtr& query,
                      float radius,
                      const std::string& parameters,
                      const BitsetPtr& invalid,
                      int64_t limited_size) const {
    // check filter
    std::function<bool(int64_t)> filter = nullptr;
    if (invalid != nullptr) {
        filter = [invalid](int64_t offset) -> bool {
            int64_t bit_index = offset & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
    }
    return this->range_search(query, radius, parameters, filter, limited_size);
};

tl::expected<DatasetPtr, Error>
DiskANN::range_search(const DatasetPtr& query,
                      float radius,
                      const std::string& parameters,
                      const std::function<bool(int64_t)>& filter,
                      int64_t limited_size) const {
#ifndef ENABLE_TESTS
    SlowTaskTimer t("diskann rangesearch", 200);
#endif

    // cannot perform search on empty index
    if (empty_index_) {
        auto ret = Dataset::Make();
        ret->Dim(0)->NumElements(1);
        return ret;
    }

    try {
        if (!index_) {
            LOG_ERROR_AND_RETURNS(
                ErrorType::INDEX_EMPTY,
                fmt::format("failed to search: {} index is empty", INDEX_DISKANN));
        }

        // check query vector
        int64_t query_num = query->GetNumElements();
        int64_t query_dim = query->GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check radius
        CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))
        CHECK_ARGUMENT(query_num == 1, fmt::format("query.num({}) must be equal to 1", query_num));

        // check limited_size
        CHECK_ARGUMENT(limited_size != 0,
                       fmt::format("limited_size({}) must not be equal to 0", limited_size))

        // check search parameters
        auto params = DiskannSearchParameters::FromJson(parameters);
        size_t beam_search = params.beam_search;
        int64_t ef_search = params.ef_search;
        CHECK_ARGUMENT(ef_search > 0,
                       fmt::format("ef_search({}) must be greater than 0", ef_search));

        bool reorder = params.use_reorder;
        int64_t io_limit = params.io_limit;

        beam_search = std::min(beam_search, MAXIMAL_BEAM_SEARCH);
        beam_search = std::max(beam_search, MINIMAL_BEAM_SEARCH);

        std::vector<uint64_t> labels;
        std::vector<float> range_distances;
        diskann::QueryStats query_stats;
        try {
            double time_cost = 0;
            {
                std::shared_lock lock(rw_mutex_);
                Timer timer(time_cost);
                index_->range_search(query->GetFloat32Vectors(),
                                     radius,
                                     ef_search,
                                     ef_search * 2,
                                     labels,
                                     range_distances,
                                     beam_search,
                                     io_limit,
                                     reorder,
                                     filter,
                                     preload_,
                                     params.use_async_io,
                                     &query_stats);
            }
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);

                result_queues_[STATSTIC_RANGE_IO].Push(static_cast<float>(query_stats.n_ios));
                result_queues_[STATSTIC_RANGE_HOP].Push(static_cast<float>(query_stats.n_hops));
                result_queues_[STATSTIC_RANGE_TIME].Push(static_cast<float>(time_cost));
                result_queues_[STATSTIC_RANGE_CACHE_HIT].Push(
                    static_cast<float>(query_stats.n_cache_hits));
                result_queues_[STATSTIC_RANGE_IO_TIME].Push(
                    (query_stats.io_us / static_cast<float>(query_stats.n_ios)) / MACRO_TO_MILLI);
            }
        } catch (const std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(
                ErrorType::INTERNAL_ERROR, "failed to perform range search on diskann: ", e.what());
        }

        auto k = static_cast<int64_t>(labels.size());
        size_t target_size = k;

        auto result = Dataset::Make();
        if (k == 0) {
            return std::move(result);
        }
        if (limited_size >= 1) {
            target_size = std::min((size_t)limited_size, target_size);
        }

        auto* dis = new float[target_size];
        auto* ids = new int64_t[target_size];
        for (int i = 0; i < target_size; ++i) {
            ids[i] = static_cast<int64_t>(labels[i]);
            dis[i] = range_distances[i];
        }

        result->NumElements(query_num)
            ->Dim(static_cast<int64_t>(target_size))
            ->Distances(dis)
            ->Ids(ids);
        return std::move(result);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform range_search(invalid argument): ",
                              e.what());
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to perform range_search(not enough memory): ",
                              e.what());
    }
}

tl::expected<BinarySet, Error>
DiskANN::serialize() const {
    SlowTaskTimer t("diskann serialize");

    auto metadata = std::make_shared<Metadata>();
    metadata->SetVersion("v0.15");

    if (status_ == IndexStatus::EMPTY) {
        // TODO(wxyu): remove this if condition
        // if (not Options::Instance().new_version()) {
        //     // return a special binaryset means empty
        //     return EmptyIndexBinarySet::Make("EMPTY_DISKANN");
        // }

        metadata->SetEmptyIndex(true);
        BinarySet bs;
        bs.Set(SERIAL_META_KEY, metadata->ToBinary());
        return bs;
    }

    try {
        std::shared_lock lock(rw_mutex_);
        BinarySet bs;

        bs.Set(DISKANN_PQ, convert_stream_to_binary(pq_pivots_stream_));
        bs.Set(DISKANN_COMPRESSED_VECTOR, convert_stream_to_binary(disk_pq_compressed_vectors_));
        bs.Set(DISKANN_LAYOUT_FILE, convert_stream_to_binary(disk_layout_stream_));
        bs.Set(DISKANN_TAG_FILE, convert_stream_to_binary(tag_stream_));
        if (preload_) {
            bs.Set(DISKANN_GRAPH, convert_stream_to_binary(graph_stream_));
            metadata->Set("support_preload", true);
        }

        bs.Set(SERIAL_META_KEY, metadata->ToBinary());
        return bs;
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, ""));
    }
}

#define DISKANN_CHECK_SELF_EMPTY                                           \
    if (this->index_) {                                                    \
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,                  \
                              "failed to deserialize: index is not empty") \
    }

tl::expected<void, Error>
DiskANN::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");

    std::unique_lock lock(rw_mutex_);
    DISKANN_CHECK_SELF_EMPTY;

    // new version serialization will contains the META_KEY
    if (binary_set.Contains(SERIAL_META_KEY)) {
        logger::debug("parse with new version format");

        auto metadata = std::make_shared<Metadata>(binary_set.Get(SERIAL_META_KEY));
        logger::debug("version: ", metadata->Version());

        // if some feature only works in specify version, use metadata->Version() likes below:
        /*
         * if (metadata->Version() == "v0.15") {
         *     ... // load data with specify format
         * }
         */

        if (metadata->EmptyIndex()) {
            empty_index_ = true;
            return {};
        }

        convert_binary_to_stream(binary_set.Get(DISKANN_LAYOUT_FILE), disk_layout_stream_);
        auto graph = binary_set.Get(DISKANN_GRAPH);
        if (/* trying to use graph-preload mode if parameter sets */ preload_) {
            if (not metadata->Get("support_preload").GetBool()) {
                LOG_ERROR_AND_RETURNS(
                    ErrorType::MISSING_FILE,
                    fmt::format("missing file: {} when deserialize diskann index", DISKANN_GRAPH));
            }
            convert_binary_to_stream(graph, graph_stream_);
        } else if (/* not use graph-preload mode, but contains */ graph.data) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }

        load_disk_index(binary_set);
        status_ = IndexStatus::MEMORY;

        return {};
    }

    // the original deserial logic, edit ONLY NECESSARY
    logger::debug("parse with v0.11 version format (matadata no found)");

    // check if binaryset is a empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        empty_index_ = true;
        return {};
    }

    convert_binary_to_stream(binary_set.Get(DISKANN_LAYOUT_FILE), disk_layout_stream_);
    auto graph = binary_set.Get(DISKANN_GRAPH);
    if (preload_) {
        if (graph.data) {
            convert_binary_to_stream(graph, graph_stream_);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("missing file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph.data) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    load_disk_index(binary_set);
    status_ = IndexStatus::MEMORY;

    return {};
}

tl::expected<void, Error>
DiskANN::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    std::unique_lock lock(rw_mutex_);
    DISKANN_CHECK_SELF_EMPTY;

    // new version serialization will contains the META_KEY
    if (reader_set.Contains(SERIAL_META_KEY)) {
        logger::debug("parse with new version format");

        auto reader = reader_set.Get(SERIAL_META_KEY);
        std::string str_metadata(reader->Size() + 1, '\0');
        reader->Read(0, reader->Size(), str_metadata.data());
        auto metadata = std::make_shared<Metadata>(str_metadata);
        logger::debug("version: ", metadata->Version());

        if (metadata->EmptyIndex()) {
            empty_index_ = true;
            return {};
        }

        std::stringstream pq_pivots_stream;
        std::stringstream disk_pq_compressed_vectors;
        std::stringstream graph;
        std::stringstream tag_stream;

        {
            auto pq_reader = reader_set.Get(DISKANN_PQ);
            auto pq_pivots_data = std::make_unique<char[]>(pq_reader->Size());
            pq_reader->Read(0, pq_reader->Size(), pq_pivots_data.get());
            pq_pivots_stream.write(pq_pivots_data.get(), static_cast<int64_t>(pq_reader->Size()));
            pq_pivots_stream.seekg(0);
        }

        {
            auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
            auto compressed_vector_data =
                std::make_unique<char[]>(compressed_vector_reader->Size());
            compressed_vector_reader->Read(
                0, compressed_vector_reader->Size(), compressed_vector_data.get());
            disk_pq_compressed_vectors.write(
                compressed_vector_data.get(),
                static_cast<int64_t>(compressed_vector_reader->Size()));
            disk_pq_compressed_vectors.seekg(0);
        }

        {
            auto tag_reader = reader_set.Get(DISKANN_TAG_FILE);
            auto tag_data = std::make_unique<char[]>(tag_reader->Size());
            tag_reader->Read(0, tag_reader->Size(), tag_data.get());
            tag_stream.write(tag_data.get(), static_cast<int64_t>(tag_reader->Size()));
            tag_stream.seekg(0);
        }

        disk_layout_reader_ = reader_set.Get(DISKANN_LAYOUT_FILE);
        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(
            reader_, metric_, sector_len_, dim_, use_bsa_));
        index_->load_from_separate_paths(pq_pivots_stream, disk_pq_compressed_vectors, tag_stream);

        auto graph_reader = reader_set.Get(DISKANN_GRAPH);
        if (/* trying to use graph-preload mode, if parameter sets */ preload_) {
            if (not metadata->Get("support_preload").GetBool()) {
                LOG_ERROR_AND_RETURNS(
                    ErrorType::MISSING_FILE,
                    fmt::format("miss file: {} when deserialize diskann index", DISKANN_GRAPH));
            }
            auto graph_data = std::make_unique<char[]>(graph_reader->Size());
            graph_reader->Read(0, graph_reader->Size(), graph_data.get());
            graph.write(graph_data.get(), static_cast<int64_t>(graph_reader->Size()));
            graph.seekg(0);
            index_->load_graph(graph);
        } else if (/* not use graph-preload mode, but contains */ graph_reader) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
        status_ = IndexStatus::HYBRID;

        return {};
    }

    // the original deserial logic, edit ONLY NECESSARY
    logger::debug("parse with v0.11 version format (matadata no found)");

    // check if readerset is a empty index
    if (reader_set.Contains(BLANK_INDEX)) {
        empty_index_ = true;
        return {};
    }

    std::stringstream pq_pivots_stream;
    std::stringstream disk_pq_compressed_vectors;
    std::stringstream graph;
    std::stringstream tag_stream;

    {
        auto pq_reader = reader_set.Get(DISKANN_PQ);
        auto pq_pivots_data = std::make_unique<char[]>(pq_reader->Size());
        pq_reader->Read(0, pq_reader->Size(), pq_pivots_data.get());
        pq_pivots_stream.write(pq_pivots_data.get(), static_cast<int64_t>(pq_reader->Size()));
        pq_pivots_stream.seekg(0);
    }

    {
        auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
        auto compressed_vector_data = std::make_unique<char[]>(compressed_vector_reader->Size());
        compressed_vector_reader->Read(
            0, compressed_vector_reader->Size(), compressed_vector_data.get());
        disk_pq_compressed_vectors.write(compressed_vector_data.get(),
                                         static_cast<int64_t>(compressed_vector_reader->Size()));
        disk_pq_compressed_vectors.seekg(0);
    }

    {
        auto tag_reader = reader_set.Get(DISKANN_TAG_FILE);
        auto tag_data = std::make_unique<char[]>(tag_reader->Size());
        tag_reader->Read(0, tag_reader->Size(), tag_data.get());
        tag_stream.write(tag_data.get(), static_cast<int64_t>(tag_reader->Size()));
        tag_stream.seekg(0);
    }

    disk_layout_reader_ = reader_set.Get(DISKANN_LAYOUT_FILE);
    reader_.reset(new LocalFileReader(batch_read_));
    index_.reset(
        new diskann::PQFlashIndex<float, int64_t>(reader_, metric_, sector_len_, dim_, use_bsa_));
    index_->load_from_separate_paths(pq_pivots_stream, disk_pq_compressed_vectors, tag_stream);

    auto graph_reader = reader_set.Get(DISKANN_GRAPH);
    if (preload_) {
        if (graph_reader) {
            auto graph_data = std::make_unique<char[]>(graph_reader->Size());
            graph_reader->Read(0, graph_reader->Size(), graph_data.get());
            graph.write(graph_data.get(), static_cast<int64_t>(graph_reader->Size()));
            graph.seekg(0);
            index_->load_graph(graph);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("miss file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph_reader) {
            logger::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    status_ = IndexStatus::HYBRID;

    return {};
}

#define WRITE_FOOTER_AND_RETURN                           \
    {                                                     \
        auto footer = std::make_shared<Footer>(metadata); \
        IOStreamWriter writer(out_stream);                \
        footer->Write(writer);                            \
        return {};                                        \
    }

#define WRITE_DATACELL_WITH_NAME(out_stream, name, datacell_stream)                    \
    datacell_offsets[(name)].SetInt(offset);                                           \
    auto datacell_stream##_size = get_stringstream_size((datacell_stream));            \
    datacell_sizes[(name)].SetInt(datacell_stream##_size);                             \
    copy_stringstream_to_ostream(out_stream, datacell_stream, datacell_stream##_size); \
    offset += datacell_stream##_size;

tl::expected<void, Error>
DiskANN::serialize(std::ostream& out_stream) {
    SlowTaskTimer t("diskann serialize");

    auto metadata = std::make_shared<Metadata>();
    metadata->SetVersion("v0.15");

    if (status_ == IndexStatus::EMPTY) {
        metadata->SetEmptyIndex(true);

        WRITE_FOOTER_AND_RETURN;
    }

    JsonType datacell_offsets;
    JsonType datacell_sizes;
    uint64_t offset = 0;

    WRITE_DATACELL_WITH_NAME(out_stream, DISKANN_PQ, pq_pivots_stream_);
    WRITE_DATACELL_WITH_NAME(out_stream, DISKANN_COMPRESSED_VECTOR, disk_pq_compressed_vectors_);
    WRITE_DATACELL_WITH_NAME(out_stream, DISKANN_LAYOUT_FILE, disk_layout_stream_);
    WRITE_DATACELL_WITH_NAME(out_stream, DISKANN_TAG_FILE, tag_stream_);

    if (preload_) {
        WRITE_DATACELL_WITH_NAME(out_stream, DISKANN_GRAPH, graph_stream_);
        metadata->Set("support_preload", true);
    }

    metadata->Set("datacell_offsets", datacell_offsets);
    metadata->Set("datacell_sizes", datacell_sizes);

    WRITE_FOOTER_AND_RETURN;
}

#define READ_DATACELL_WITH_NAME(in_stream, name, datacell_stream) \
    in_stream.seekg(datacell_offsets[(name)].GetInt());           \
    copy_istream_to_stringstream((datacell_stream), in_stream, datacell_sizes[(name)].GetInt());

tl::expected<void, Error>
DiskANN::deserialize(std::istream& in_stream) {
    SlowTaskTimer t("diskann deserialize");

    try {
        IOStreamReader reader(in_stream);
        auto footer = Footer::Parse(reader);
        if (footer == nullptr) {
            throw std::runtime_error("unknown diskann serialization");
        }

        auto metadata = footer->GetMetadata();
        if (metadata->EmptyIndex()) {
            return {};
        }

        JsonType datacell_offsets = metadata->Get("datacell_offsets");
        logger::debug("datacell_offsets: {}", datacell_offsets.Dump());
        JsonType datacell_sizes = metadata->Get("datacell_sizes");
        logger::debug("datacell_sizes: {}", datacell_sizes.Dump());

        std::stringstream pq_pivots_stream;
        std::stringstream disk_pq_compressed_vectors;
        std::stringstream graph;
        std::stringstream tag_stream;

        READ_DATACELL_WITH_NAME(in_stream, DISKANN_PQ, pq_pivots_stream);
        READ_DATACELL_WITH_NAME(in_stream, DISKANN_COMPRESSED_VECTOR, disk_pq_compressed_vectors);
        READ_DATACELL_WITH_NAME(in_stream, DISKANN_TAG_FILE, tag_stream);

        disk_layout_reader_ =
            std::make_shared<IStreamReader>(in_stream,
                                            datacell_offsets[DISKANN_LAYOUT_FILE].GetInt(),
                                            datacell_sizes[DISKANN_LAYOUT_FILE].GetInt());

        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(
            reader_, metric_, sector_len_, dim_, use_bsa_));
        index_->load_from_separate_paths(pq_pivots_stream, disk_pq_compressed_vectors, tag_stream);

        if (preload_) {
            if (not metadata->Get("support_preload").GetBool()) {
                LOG_ERROR_AND_RETURNS(
                    ErrorType::MISSING_FILE,
                    fmt::format("miss file: {} when deserialize diskann index", DISKANN_GRAPH));
            }

            READ_DATACELL_WITH_NAME(in_stream, DISKANN_GRAPH, graph);
            index_->load_graph(graph);
        }
        status_ = IndexStatus::HYBRID;

    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const VsagException& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

std::string
DiskANN::GetStats() const {
    JsonType j;
    j[STATSTIC_DATA_NUM].SetInt(GetNumElements());
    j[STATSTIC_INDEX_NAME].SetString(INDEX_DISKANN);
    j[STATSTIC_MEMORY].SetInt(GetMemoryUsage());

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (auto& item : result_queues_) {
            j[item.first].SetFloat(item.second.GetAvgResult());
        }
    }

    return j.Dump(4);
}

int64_t
DiskANN::GetEstimateBuildMemory(const int64_t num_elements) const {
    int64_t estimate_memory_usage = 0;
    // Memory usage of graph (1.365 is the relaxation factor used by DiskANN during graph construction.)
    estimate_memory_usage += (num_elements * R_ * sizeof(uint32_t)  // NOLINT
                              + num_elements * (R_ + 1) * sizeof(uint32_t)) *
                             GRAPH_SLACK;
    // Memory usage of disk layout
    if (sector_len_ > MINIMAL_SECTOR_LEN) {
        estimate_memory_usage +=
            static_cast<int64_t>((num_elements + 1) * sector_len_ * sizeof(uint8_t));
    } else {
        size_t single_node =
            (size_t)(dim_ * sizeof(float) + (R_ * GRAPH_SLACK + 1) * sizeof(uint32_t)) *  // NOLINT
            VECTOR_PER_BLOCK;
        size_t node_per_sector = MINIMAL_SECTOR_LEN / single_node;
        size_t sector_size = num_elements / node_per_sector + 1;
        estimate_memory_usage +=
            static_cast<int64_t>((sector_size + 1) * sector_len_ * sizeof(uint8_t));
    }
    // Memory usage of the ID mapping.
    estimate_memory_usage += static_cast<int64_t>(num_elements * sizeof(int64_t) * 2);
    // Memory usage of the compressed PQ vectors.
    estimate_memory_usage +=
        static_cast<int64_t>(disk_pq_dims_ * num_elements * sizeof(uint8_t) * 2);
    // Memory usage of the PQ centers and chunk offsets.
    estimate_memory_usage +=
        static_cast<int64_t>(256 * dim_ * sizeof(float) * (3 + 1) + dim_ * sizeof(uint32_t));
    return estimate_memory_usage;
}

template <typename Container>
Binary
serialize_to_binary(const Container& container) {
    using ValueType = typename Container::value_type;
    size_t total_size = container.size() * sizeof(ValueType);
    std::shared_ptr<int8_t[]> raw_data(new int8_t[total_size], std::default_delete<int8_t[]>());

    int8_t* data_ptr = raw_data.get();
    for (const ValueType& value : container) {
        std::memcpy(data_ptr, &value, sizeof(ValueType));
        data_ptr += sizeof(ValueType);
    }

    Binary binary_data{raw_data, total_size};
    return binary_data;
}

template <typename Container>
Container
deserialize_from_binary(const Binary& binary_data) {
    using ValueType = typename Container::value_type;

    Container deserialized_container;
    const int8_t* data_ptr = binary_data.data.get();
    size_t num_elements = binary_data.size / sizeof(ValueType);

    for (size_t i = 0; i < num_elements; ++i) {
        ValueType value;
        std::memcpy(&value, data_ptr, sizeof(ValueType));
        deserialized_container.insert(deserialized_container.end(), value);
        data_ptr += sizeof(ValueType);
    }

    return std::move(deserialized_container);
}

template <typename T>
Binary
serialize_vector_to_binary(std::vector<T> data) {
    if (data.empty()) {
        return {};
    }
    size_t total_size = data.size() * sizeof(T);
    std::shared_ptr<int8_t[]> raw_data(new int8_t[total_size], std::default_delete<int8_t[]>());
    int8_t* data_ptr = raw_data.get();
    std::memcpy(data_ptr, data.data(), total_size);
    Binary binary_data{raw_data, total_size};
    return binary_data;
}

template <typename T>
std::vector<T>
deserialize_vector_from_binary(const Binary& binary_data) {
    std::vector<T> deserialized_container;
    if (binary_data.size == 0) {
        return std::move(deserialized_container);
    }
    const int8_t* data_ptr = binary_data.data.get();
    size_t num_elements = binary_data.size / sizeof(T);
    deserialized_container.resize(num_elements);
    std::memcpy(deserialized_container.data(), data_ptr, num_elements * sizeof(T));
    return std::move(deserialized_container);
}

tl::expected<Index::Checkpoint, Error>
DiskANN::continue_build(const DatasetPtr& base, const BinarySet& binary_set) {
    std::unique_lock lock(rw_mutex_);
    try {
        BuildStatus build_status = BuildStatus::BEGIN;
        if (not binary_set.GetKeys().empty()) {
            Binary status_binary = binary_set.Get(BUILD_STATUS);
            CHECK_ARGUMENT(status_binary.data != nullptr, "missing status while partial building");
            build_status = from_binary<BuildStatus>(status_binary);
        }
        CHECK_ARGUMENT(
            base->GetDim() == dim_,
            fmt::format("base.dim({}) must be equal to index.dim({})", base->GetDim(), dim_));
        CHECK_ARGUMENT(
            base->GetNumElements() >= DATA_LIMIT,
            "number of elements must be greater equal than " + std::to_string(DATA_LIMIT));
        if (this->index_ && status_ != IndexStatus::BUILDING) {
            LOG_ERROR_AND_RETURNS(ErrorType::BUILD_TWICE, "failed to build index: build twice");
        }
        status_ = IndexStatus::BUILDING;
        BinarySet after_binary_set;
        switch (build_status) {
            case BEGIN: {
                int round = 1;
                SlowTaskTimer t(fmt::format("diskann build graph {}/{}", round, build_batch_num_));
                build_status = BuildStatus::GRAPH;
                build_partial_graph(base, binary_set, after_binary_set, round);
                round++;
                after_binary_set.Set(BUILD_CURRENT_ROUND, to_binary<int>(round));
                break;
            }
            case GRAPH: {
                int round = from_binary<int>(binary_set.Get(BUILD_CURRENT_ROUND));
                SlowTaskTimer t(
                    fmt::format("diskann build (graph {}/{})", round, build_batch_num_));
                build_partial_graph(base, binary_set, after_binary_set, round);
                if (round < build_batch_num_) {
                    round++;
                } else {
                    build_status = BuildStatus::EDGE_PRUNE;
                }
                after_binary_set.Set(BUILD_CURRENT_ROUND, to_binary<int>(round));
                break;
            }
            case EDGE_PRUNE: {
                SlowTaskTimer t(fmt::format("diskann build (edge prune)"));
                int round = from_binary<int>(binary_set.Get(BUILD_CURRENT_ROUND));
                build_partial_graph(base, binary_set, after_binary_set, round);
                build_status = BuildStatus::PQ;
                break;
            }
            case PQ: {
                SlowTaskTimer t(fmt::format("diskann build (pq)"));
                auto failed_locs =
                    deserialize_vector_from_binary<size_t>(after_binary_set.Get(BUILD_FAILED_LOC));
                diskann::generate_disk_quantized_data<float>(base->GetFloat32Vectors(),
                                                             base->GetNumElements(),
                                                             dim_,
                                                             failed_locs,
                                                             pq_pivots_stream_,
                                                             disk_pq_compressed_vectors_,
                                                             metric_,
                                                             p_val_,
                                                             disk_pq_dims_,
                                                             use_opq_);
                after_binary_set = binary_set;
                after_binary_set.Set(DISKANN_PQ, convert_stream_to_binary(pq_pivots_stream_));
                after_binary_set.Set(DISKANN_COMPRESSED_VECTOR,
                                     convert_stream_to_binary(disk_pq_compressed_vectors_));
                build_status = BuildStatus::DISK_LAYOUT;
                break;
            }
            case DISK_LAYOUT: {
                SlowTaskTimer t(fmt::format("diskann build (disk layout)"));
                auto failed_locs =
                    deserialize_vector_from_binary<size_t>(after_binary_set.Get(BUILD_FAILED_LOC));
                convert_binary_to_stream(binary_set.Get(DISKANN_GRAPH), graph_stream_);
                diskann::create_disk_layout<float>(base->GetFloat32Vectors(),
                                                   base->GetNumElements(),
                                                   dim_,
                                                   failed_locs,
                                                   graph_stream_,
                                                   disk_layout_stream_,
                                                   sector_len_,
                                                   metric_);
                load_disk_index(binary_set);
                build_status = BuildStatus::FINISH;
                status_ = IndexStatus::MEMORY;
                break;
            }
            case FINISH:
                logger::warn("build process is finished");
        }
        after_binary_set.Set(BUILD_STATUS, to_binary<BuildStatus>(build_status));
        Checkpoint checkpoint{.data = after_binary_set,
                              .finish = build_status == BuildStatus::FINISH};
        return checkpoint;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to build(invalid argument): ", e.what());
    }
}

tl::expected<void, Error>
DiskANN::build_partial_graph(const DatasetPtr& base,
                             const BinarySet& binary_set,
                             BinarySet& after_binary_set,
                             int round) {
    const auto* vectors = base->GetFloat32Vectors();
    const auto* ids = base->GetIds();
    auto data_num = base->GetNumElements();
    std::vector<int64_t> tags(ids, ids + data_num);
    {
        // build graph
        build_index_ = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
            metric_, dim_, data_num, false, true, false, false, 0, false);

        std::unordered_set<uint32_t> builded_nodes;
        if (round > 1) {
            std::stringstream graph_stream;
            std::stringstream tag_stream;
            convert_binary_to_stream(binary_set.Get(DISKANN_GRAPH), graph_stream);
            convert_binary_to_stream(binary_set.Get(DISKANN_TAG_FILE), tag_stream);

            build_index_->load(graph_stream, tag_stream, Options::Instance().num_threads_io(), L_);
            builded_nodes =
                deserialize_from_binary<std::unordered_set<uint32_t>>(binary_set.Get(BUILD_NODES));
        }

        auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_)
                                      .with_num_threads(Options::Instance().num_threads_building())
                                      .build();
        std::vector<size_t> failed_locs =
            build_index_->build(vectors,
                                dim_,
                                index_build_params,
                                tags,
                                use_reference_,
                                round,
                                static_cast<int32_t>(build_batch_num_),
                                &builded_nodes);
        build_index_->save(graph_stream_, tag_stream_);
        after_binary_set.Set(BUILD_NODES,
                             serialize_to_binary<std::unordered_set<uint32_t>>(builded_nodes));
        after_binary_set.Set(BUILD_FAILED_LOC, serialize_vector_to_binary<size_t>(failed_locs));
        build_index_.reset();
    }
    after_binary_set.Set(DISKANN_GRAPH, convert_stream_to_binary(graph_stream_));
    after_binary_set.Set(DISKANN_TAG_FILE, convert_stream_to_binary(tag_stream_));
    return {};
}

tl::expected<void, Error>
DiskANN::load_disk_index(const BinarySet& binary_set) {
    disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);
    reader_.reset(new LocalFileReader(batch_read_));
    index_.reset(
        new diskann::PQFlashIndex<float, int64_t>(reader_, metric_, sector_len_, dim_, use_bsa_));

    convert_binary_to_stream(binary_set.Get(DISKANN_COMPRESSED_VECTOR),
                             disk_pq_compressed_vectors_);
    convert_binary_to_stream(binary_set.Get(DISKANN_PQ), pq_pivots_stream_);
    convert_binary_to_stream(binary_set.Get(DISKANN_TAG_FILE), tag_stream_);
    index_->load_from_separate_paths(pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);
    if (preload_) {
        index_->load_graph(graph_stream_);
    } else {
        graph_stream_.str("");
    }
    return {};
}

bool
DiskANN::CheckFeature(IndexFeature feature) const {
    return this->feature_list_->CheckFeature(feature);
}
void
DiskANN::init_feature_list() {
    // build
    this->feature_list_->SetFeatures({
        SUPPORT_BUILD,
    });

    //  search
    this->feature_list_->SetFeatures({
        SUPPORT_KNN_SEARCH,
        SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        SUPPORT_RANGE_SEARCH,
        SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
    });

    // concurrency
    this->feature_list_->SetFeatures({
        SUPPORT_SEARCH_CONCURRENT,
    });

    // serialize
    this->feature_list_->SetFeatures({IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
                                      IndexFeature::SUPPORT_DESERIALIZE_FILE,
                                      IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
                                      IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
                                      IndexFeature::SUPPORT_SERIALIZE_FILE});
}

}  // namespace vsag
