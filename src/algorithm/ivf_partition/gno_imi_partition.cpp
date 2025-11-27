
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

#include "gno_imi_partition.h"

#include <cblas.h>
#include <fmt/format.h>

#include <atomic>
#include <fstream>
#include <numeric>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/cluster/kmeans_cluster.h"
#include "inner_string_params.h"
#include "utils/util_functions.h"

namespace vsag {

static constexpr const char* SEARCH_PARAM_TEMPLATE_STR = R"(
{{
    "hnsw": {{
        "ef_search": {}
    }}
}}
)";

// C = A * B^T
void
matmul(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    cblas_sgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<blasint>(N),
                static_cast<blasint>(M),
                static_cast<blasint>(K),
                1.0F,
                B,
                static_cast<blasint>(K),
                A,
                static_cast<blasint>(K),
                0.0F,
                C,
                static_cast<blasint>(N));
}

GNOIMIPartition::GNOIMIPartition(const IndexCommonParam& common_param,
                                 const IVFPartitionStrategyParametersPtr& param)
    : IVFPartitionStrategy(common_param,
                           param->gnoimi_param->first_order_buckets_count *
                               param->gnoimi_param->second_order_buckets_count),
      bucket_count_s_(param->gnoimi_param->first_order_buckets_count),
      bucket_count_t_(param->gnoimi_param->second_order_buckets_count),
      data_centroids_s_(allocator_),
      data_centroids_t_(allocator_),
      norms_s_(allocator_),
      norms_t_(allocator_),
      precomputed_terms_st_(allocator_),
      common_param_(common_param) {
    data_centroids_s_.resize(bucket_count_s_ * dim_);
    data_centroids_t_.resize(bucket_count_t_ * dim_);
    norms_s_.resize(bucket_count_s_);
    norms_t_.resize(bucket_count_t_);
    precomputed_terms_st_.resize(static_cast<long>(bucket_count_s_) * bucket_count_t_);

    param_ptr_ = std::make_shared<BruteForceParameter>();
    param_ptr_->base_codes_param = std::make_shared<FlattenDataCellParameter>();
    JsonType memory_json;
    memory_json["type"].SetString(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    param_ptr_->base_codes_param->io_parameter = IOParameter::GetIOParameterByJson(memory_json);
    JsonType quantizer_json;
    quantizer_json["type"].SetString(QUANTIZATION_TYPE_VALUE_FP32);
    param_ptr_->base_codes_param->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(quantizer_json);
}

void
GNOIMIPartition::Train(const DatasetPtr dataset) {
    auto dim = this->dim_;
    auto centroids_s = Dataset::Make();
    auto centroids_t = Dataset::Make();
    const auto* vectors = dataset->GetFloat32Vectors();
    auto num_element = dataset->GetNumElements();
    Vector<float> norm_vectors(allocator_);
    if (metric_type_ == MetricType::METRIC_TYPE_COSINE) {
        norm_vectors.resize(num_element * dim);
        for (int64_t i = 0; i < num_element; ++i) {
            Normalize(vectors + i * dim_, norm_vectors.data() + i * dim_, dim_);
        }
        vectors = norm_vectors.data();
    }

    Vector<LabelType> ids_centroids_s(this->bucket_count_s_, allocator_);
    Vector<LabelType> ids_centroids_t(this->bucket_count_t_, allocator_);
    Vector<float> data_centroids_s_tmp(this->bucket_count_s_ * dim_, allocator_);
    Vector<float> data_centroids_t_tmp(this->bucket_count_t_ * dim_, allocator_);

    std::iota(ids_centroids_s.begin(), ids_centroids_s.end(), 0);
    std::iota(ids_centroids_t.begin(), ids_centroids_t.end(), 0);
    centroids_s->Ids(ids_centroids_s.data())
        ->Dim(dim)
        ->Float32Vectors(data_centroids_s_tmp.data())
        ->NumElements(this->bucket_count_s_)
        ->Owner(false);
    centroids_t->Ids(ids_centroids_t.data())
        ->Dim(dim)
        ->Float32Vectors(data_centroids_t_tmp.data())
        ->NumElements(this->bucket_count_t_)
        ->Owner(false);

    KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_);
    Vector<float> residuals(vectors, vectors + num_element * dim, allocator_);

    auto train_and_get_residual = [&, this](const DatasetPtr& centroids,
                                            float* data_centroids,
                                            double* err) {
        cls.Run(centroids->GetNumElements(), residuals.data(), num_element, 30, err);
        memcpy(data_centroids, cls.k_centroids_, dim * centroids->GetNumElements() * sizeof(float));
        BruteForce route_index(param_ptr_, common_param_);
        auto build_result = route_index.Build(centroids);
        auto assign = this->inner_classify_datas(route_index, residuals.data(), num_element);
        this->GetResidual(num_element, vectors, residuals.data(), data_centroids, assign.data());
    };

    // train loop
    double min_err = std::numeric_limits<double>::max();
    for (size_t i = 0; i < 2; ++i) {
        double err_to_s = 0.0;
        double err_to_t = 0.0;
        train_and_get_residual(centroids_s, data_centroids_s_tmp.data(), &err_to_s);
        logger::info("gnoimi train iter: {}, err of centroids_s: {}", i, err_to_s);

        train_and_get_residual(centroids_t, data_centroids_t_tmp.data(), &err_to_t);
        logger::info("gnoimi train iter: {}, err of centroids_t: {}", i, err_to_t);

        if (err_to_t < min_err) {
            min_err = err_to_t;
            std::copy(data_centroids_s_tmp.begin(),
                      data_centroids_s_tmp.end(),
                      data_centroids_s_.begin());
            std::copy(data_centroids_t_tmp.begin(),
                      data_centroids_t_tmp.end(),
                      data_centroids_t_.begin());
        }
    }

    for (BucketIdType i = 0; i < bucket_count_s_; ++i) {
        auto norm_sqr = FP32ComputeIP(
            data_centroids_s_.data() + i * dim_, data_centroids_s_.data() + i * dim_, dim_);
        norms_s_[i] = norm_sqr / 2;
    }

    Vector<std::pair<float, BucketIdType>> norms_t(bucket_count_t_, this->allocator_);
    for (BucketIdType i = 0; i < bucket_count_t_; ++i) {
        auto norm_sqr = FP32ComputeIP(
            data_centroids_t_.data() + i * dim_, data_centroids_t_.data() + i * dim_, dim_);
        norms_t[i].first = norm_sqr / 2;
        norms_t[i].second = i;
    }

    // Rearrange data_centroids_t_ based on ascending order of their norms.
    std::sort(norms_t.begin(), norms_t.end(), std::greater<>());
    std::vector<float> temp_data(bucket_count_t_ * dim_, 0.0);
    for (BucketIdType i = 0; i < bucket_count_t_; ++i) {
        BucketIdType src_idx = norms_t[i].second;
        size_t src_offset = src_idx * dim_;
        size_t dst_offset = i * dim_;
        std::copy(data_centroids_t_.data() + src_offset,
                  data_centroids_t_.data() + src_offset + dim_,
                  temp_data.data() + dst_offset);
        norms_t_[i] = norms_t[i].first;
    }
    std::copy(temp_data.begin(), temp_data.end(), data_centroids_t_.data());

    Vector<float> ip_st(static_cast<long>(bucket_count_s_) * bucket_count_t_, allocator_);
    matmul(data_centroids_s_.data(),
           data_centroids_t_.data(),
           ip_st.data(),
           bucket_count_s_,
           bucket_count_t_,
           dim_);
    for (BucketIdType i = 0; i < bucket_count_s_ * bucket_count_t_; ++i) {
        BucketIdType cur_bucket_id_t = i % bucket_count_t_;
        precomputed_terms_st_[i] = norms_t_[cur_bucket_id_t] + ip_st[i];
    }

    this->is_trained_ = true;
}

Vector<BucketIdType>
GNOIMIPartition::ClassifyDatas(const void* datas,
                               int64_t count,
                               BucketIdType buckets_per_data,
                               Statistics& stats) const {
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    inner_joint_classify_datas(
        reinterpret_cast<const float*>(datas), count, buckets_per_data, result.data(), stats);
    return result;
}

Vector<BucketIdType>
GNOIMIPartition::ClassifyDatasForSearch(const void* datas,
                                        int64_t count,
                                        const InnerSearchParam& param,
                                        Statistics& stats) {
    Vector<float> norm_vectors(allocator_);
    if (metric_type_ == MetricType::METRIC_TYPE_COSINE) {
        norm_vectors.resize(count * dim_);
        for (int64_t i = 0; i < count; ++i) {
            Normalize(
                static_cast<const float*>(datas) + i * dim_, norm_vectors.data() + i * dim_, dim_);
        }
        datas = norm_vectors.data();
    }
    auto buckets_per_data = param.scan_bucket_size;
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    auto candidate_count_s = bucket_count_s_;
    Vector<BucketIdType> candidate_s_id(candidate_count_s, this->allocator_);
    Vector<float> candidate_s_dist(candidate_count_s, this->allocator_);
    Vector<float> dist_to_s(bucket_count_s_ * count, this->allocator_);
    Vector<float> dist_to_t(bucket_count_t_ * count, this->allocator_);
    auto* dist_to_s_data = dist_to_s.data();
    auto* dist_to_t_data = dist_to_t.data();
    auto* candidate_s_id_data = candidate_s_id.data();
    auto* candidate_s_dist_data = candidate_s_dist.data();

    matmul(reinterpret_cast<const float*>(datas),
           data_centroids_s_.data(),
           dist_to_s_data,
           count,
           bucket_count_s_,
           dim_);
    matmul(reinterpret_cast<const float*>(datas),
           data_centroids_t_.data(),
           dist_to_t_data,
           count,
           bucket_count_t_,
           dim_);

    for (size_t i = 0; i < count; i++) {
        auto qnorm = FP32ComputeIP(reinterpret_cast<const float*>(datas) + i * dim_,
                                   reinterpret_cast<const float*>(datas) + i * dim_,
                                   dim_) /
                     2;
        MaxHeap heap(this->allocator_);
        for (size_t j = 0; j < bucket_count_s_; ++j) {
            auto dist_term_s = norms_s_[j] - dist_to_s_data[i * bucket_count_s_ + j];
            if (heap.size() < candidate_count_s || dist_term_s < heap.top().first) {
                heap.emplace(dist_term_s, j);
            }
            if (heap.size() > candidate_count_s) {
                heap.pop();
            }
        }
        for (auto j = static_cast<int64_t>(candidate_count_s - 1); j >= 0; --j) {
            candidate_s_id_data[j] = static_cast<BucketIdType>(heap.top().second);
            candidate_s_dist_data[j] = heap.top().first;
            heap.pop();
        }
        CHECK_ARGUMENT(heap.empty(), fmt::format("Unexpected non-empty heap after pop candidates"));

        auto scan_bucket_count_s = static_cast<BucketIdType>(
            std::floor(static_cast<float>(bucket_count_s_) * param.first_order_scan_ratio));
        scan_bucket_count_s = std::max(scan_bucket_count_s, 1);
        for (size_t j = 0; j < scan_bucket_count_s; ++j) {
            for (size_t k = 0; k < bucket_count_t_; ++k) {
                auto cur_bucket_id_s = candidate_s_id_data[j];
                auto cur_bucket_id_t = k;
                float dist_term_st = candidate_s_dist_data[j] +
                                     precomputed_terms_st_[static_cast<unsigned long>(
                                                               cur_bucket_id_s * bucket_count_t_) +
                                                           cur_bucket_id_t] -
                                     dist_to_t_data[i * bucket_count_t_ + cur_bucket_id_t];

                auto cur_bucket_id_global =
                    static_cast<long>(cur_bucket_id_s) * bucket_count_t_ + cur_bucket_id_t;
                if (heap.size() < buckets_per_data || dist_term_st < heap.top().first) {
                    heap.emplace(dist_term_st, cur_bucket_id_global);
                }
                if (heap.size() > buckets_per_data) {
                    heap.pop();
                }
            }
        }
        BucketIdType size = std::min((BucketIdType)heap.size(), buckets_per_data);
        for (auto j = static_cast<int64_t>(size - 1); j >= 0 && !heap.empty(); --j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(heap.top().second);
            heap.pop();
        }
    }
    return result;
}

void
GNOIMIPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    StreamWriter::WriteObj(writer, this->bucket_count_s_);
    StreamWriter::WriteObj(writer, this->bucket_count_t_);
    StreamWriter::WriteVector(writer, this->data_centroids_s_);
    StreamWriter::WriteVector(writer, this->data_centroids_t_);
    StreamWriter::WriteVector(writer, this->norms_s_);
    StreamWriter::WriteVector(writer, this->norms_t_);
    StreamWriter::WriteVector(writer, this->precomputed_terms_st_);
}
void
GNOIMIPartition::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    IVFPartitionStrategy::Deserialize(reader);
    StreamReader::ReadObj<BucketIdType>(reader, this->bucket_count_s_);
    StreamReader::ReadObj<BucketIdType>(reader, this->bucket_count_t_);
    StreamReader::ReadVector(reader, this->data_centroids_s_);
    StreamReader::ReadVector(reader, this->data_centroids_t_);
    StreamReader::ReadVector(reader, this->norms_s_);
    StreamReader::ReadVector(reader, this->norms_t_);
    StreamReader::ReadVector(reader, this->precomputed_terms_st_);
}

Vector<BucketIdType>
GNOIMIPartition::inner_classify_datas(BruteForce& route_index, const float* datas, int64_t count) {
    BucketIdType buckets_per_data = 1;
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    for (int64_t i = 0; i < count; ++i) {
        auto query = Dataset::Make();
        query->Dim(this->dim_)
            ->Float32Vectors(datas + i * this->dim_)
            ->NumElements(1)
            ->Owner(false);
        auto search_param = fmt::format(
            SEARCH_PARAM_TEMPLATE_STR, std::max(10L, static_cast<int64_t>(buckets_per_data * 1.2)));
        FilterPtr filter = nullptr;
        auto search_result = route_index.KnnSearch(query, buckets_per_data, search_param, filter);
        const auto* result_ids = search_result->GetIds();

        for (int64_t j = 0; j < buckets_per_data; ++j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(result_ids[j]);
        }
    }
    return result;
}

void
GNOIMIPartition::inner_joint_classify_datas(const float* datas,
                                            int64_t count,
                                            BucketIdType buckets_per_data,
                                            BucketIdType* result,
                                            Statistics& stats) const {
    Vector<float> dist_to_s(bucket_count_s_ * count, this->allocator_);
    Vector<float> dist_to_t(bucket_count_t_ * count, this->allocator_);
    Vector<std::pair<float, BucketIdType>> precomputed_terms_s(bucket_count_s_, this->allocator_);

    matmul(datas, data_centroids_s_.data(), dist_to_s.data(), count, bucket_count_s_, dim_);
    matmul(datas, data_centroids_t_.data(), dist_to_t.data(), count, bucket_count_t_, dim_);
    // |x - s - t|^2 = |x|^2 + |s|^2 + |t|^2 - 2xs - 2xt + 2st
    // precomputed_terms_s: |x - s|^2 = |s|^2 - 2xs + |x|^2
    // precomputed_terms_st: |t|^2 + 2st
    float total_err = 0.0;
    uint32_t dist_cmp = 0;
    for (size_t i = 0; i < count; ++i) {
        auto data_norm = FP32ComputeIP(datas + i * dim_, datas + i * dim_, dim_);
        for (BucketIdType j = 0; j < bucket_count_s_; ++j) {
            precomputed_terms_s[j].first =
                norms_s_[j] - dist_to_s[i * bucket_count_s_ + j] + data_norm / 2;
            precomputed_terms_s[j].second = j;
        }
        std::sort(precomputed_terms_s.begin(), precomputed_terms_s.end());

        MaxHeap heap(this->allocator_);
        for (size_t j = 0; j < bucket_count_s_; ++j) {
            float cur_precomputed_term_s = precomputed_terms_s[j].first;
            BucketIdType cur_bucket_id_s = precomputed_terms_s[j].second;

            for (BucketIdType k = 0; k < bucket_count_t_; ++k) {
                BucketIdType cur_bucket_id_t = k;
                if (heap.size() >= buckets_per_data &&
                    std::sqrt(cur_precomputed_term_s) - std::sqrt(norms_t_[cur_bucket_id_t]) >
                        std::sqrt(heap.top().first)) {
                    break;
                }

                int cur_bucket_id_global = cur_bucket_id_s * bucket_count_t_ + cur_bucket_id_t;
                float dist = cur_precomputed_term_s - dist_to_t[i * bucket_count_t_ + k] +
                             precomputed_terms_st_[cur_bucket_id_global];
                ++dist_cmp;

                if (heap.size() < buckets_per_data || dist < heap.top().first) {
                    heap.emplace(dist, cur_bucket_id_global);
                }
                if (heap.size() > buckets_per_data) {
                    heap.pop();
                }
            }
        }

        for (auto j = static_cast<int64_t>(buckets_per_data - 1); j >= 0; --j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(heap.top().second);
            if (j == 0) {
                total_err += heap.top().first;
            }
            heap.pop();
        }
    }

    stats.dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
}

void
GNOIMIPartition::GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) {
    if (!is_trained_ || bucket_id >= bucket_count_) {
        throw std::runtime_error("Invalid bucket_id or partition not trained");
    }
    auto bucket_id_s = bucket_id / bucket_count_t_;
    auto bucket_id_t = bucket_id % bucket_count_t_;
    FP32Add(data_centroids_s_.data() + bucket_id_s * dim_,
            data_centroids_t_.data() + bucket_id_t * dim_,
            centroid.data(),
            dim_);
}

}  // namespace vsag
