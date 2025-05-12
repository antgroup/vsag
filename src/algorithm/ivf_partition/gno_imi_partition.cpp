
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

#include <fmt/format-inl.h>

#include <fstream>

#include "impl/kmeans_cluster.h"
#include "inner_string_params.h"
#include "safe_allocator.h"
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
matmul(const float* A, const float* B, float* C, int32_t M, int32_t N, int32_t K) {
    cblas_sgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<blasint>(N),
                static_cast<blasint>(M),
                K,
                1.0F,
                B,
                K,
                A,
                K,
                0.0F,
                C,
                static_cast<blasint>(N));
}

GNOIMIPartition::GNOIMIPartition(const IndexCommonParam& common_param,
                                 const IVFPartitionStrategyParametersPtr& param)
    : IVFPartitionStrategy(common_param,
                           param->gnoimi_param->first_order_buckets_count *
                               param->gnoimi_param->second_order_buckets_count),
      bucket_count_S_(param->gnoimi_param->first_order_buckets_count),
      bucket_count_T_(param->gnoimi_param->second_order_buckets_count),
      data_centroids_S_(allocator_),
      data_centroids_T_(allocator_),
      norms_S_(allocator_),
      norms_T_(allocator_),
      precomputed_terms_ST_(allocator_),
      common_param_(common_param) {
    data_centroids_S_.resize(bucket_count_S_ * dim_);
    data_centroids_T_.resize(bucket_count_T_ * dim_);
    norms_S_.resize(bucket_count_S_);
    norms_T_.resize(bucket_count_T_);
    precomputed_terms_ST_.resize(bucket_count_S_ * bucket_count_T_);

    param_ptr_ = std::make_shared<BruteForceParameter>();
    param_ptr_->flatten_param = std::make_shared<FlattenDataCellParameter>();
    JsonType memory_json = {
        {"type", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    };
    param_ptr_->flatten_param->io_parameter = IOParameter::GetIOParameterByJson(memory_json);
    JsonType quantizer_json = {
        {"type", QUANTIZATION_TYPE_VALUE_FP32},
    };
    param_ptr_->flatten_param->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(quantizer_json);
}

void
GNOIMIPartition::Train(const DatasetPtr dataset) {
    auto dim = this->dim_;
    auto centroidsS = Dataset::Make();
    auto centroidsT = Dataset::Make();
    const auto* vectors = dataset->GetFloat32Vectors();
    auto num_element = dataset->GetNumElements();
    Vector<LabelType> ids_centroidsS(this->bucket_count_S_, allocator_);
    Vector<LabelType> ids_centroidsT(this->bucket_count_T_, allocator_);
    Vector<float> data_centroids_S_tmp(this->bucket_count_S_ * dim_, allocator_);
    Vector<float> data_centroids_T_tmp(this->bucket_count_T_ * dim_, allocator_);

    std::iota(ids_centroidsS.begin(), ids_centroidsS.end(), 0);
    std::iota(ids_centroidsT.begin(), ids_centroidsT.end(), 0);
    centroidsS->Ids(ids_centroidsS.data())
        ->Dim(dim)
        ->Float32Vectors(data_centroids_S_tmp.data())
        ->NumElements(this->bucket_count_S_)
        ->Owner(false);
    centroidsT->Ids(ids_centroidsT.data())
        ->Dim(dim)
        ->Float32Vectors(data_centroids_T_tmp.data())
        ->NumElements(this->bucket_count_T_)
        ->Owner(false);

    KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_);
    Vector<float> residuals(vectors, vectors + num_element * dim, allocator_);

    auto train_and_get_residual = [&, this](
                                      DatasetPtr centroids, float* data_centroids, float* err) {
        cls.Run(centroids->GetNumElements(), residuals.data(), num_element, 30, err);
        memcpy(data_centroids, cls.k_centroids_, dim * centroids->GetNumElements() * sizeof(float));
        BruteForce route_index(param_ptr_, common_param_);
        auto build_result = route_index.Build(centroids);
        auto assign = this->inner_classify_datas(route_index, residuals.data(), num_element);
        this->GetResidual(num_element, vectors, residuals.data(), data_centroids, assign.data());
    };

    // train loop
    float min_err = std::numeric_limits<float>::max();
    for (size_t i = 0; i < 2; ++i) {
        float err_to_S = 0.0f, err_to_T = 0.0f;
        train_and_get_residual(centroidsS, data_centroids_S_tmp.data(), &err_to_S);
        logger::info("gnoimi train iter: {}, err of centroidsS: {}", i, err_to_S);

        train_and_get_residual(centroidsT, data_centroids_T_tmp.data(), &err_to_T);
        logger::info("gnoimi train iter: {}, err of centroidsT: {}", i, err_to_T);
        
        if (err_to_T < min_err) {
            min_err = err_to_T;
            std::copy(data_centroids_S_tmp.begin(),
                      data_centroids_S_tmp.end(),
                      data_centroids_S_.begin());
            std::copy(data_centroids_T_tmp.begin(),
                      data_centroids_T_tmp.end(),
                      data_centroids_T_.begin());
        }
    }

    for (size_t i = 0; i < bucket_count_S_; ++i) {
        auto norm_sqr = FP32ComputeIP(
            data_centroids_S_.data() + i * dim_, data_centroids_S_.data() + i * dim_, dim_);
        norms_S_[i] = norm_sqr / 2;
    }

    Vector<std::pair<float, BucketIdType>> norms_T(bucket_count_T_, this->allocator_);
    for (size_t i = 0; i < bucket_count_T_; ++i) {
        auto norm_sqr = FP32ComputeIP(
            data_centroids_T_.data() + i * dim_, data_centroids_T_.data() + i * dim_, dim_);
        norms_T[i].first = norm_sqr / 2;
        norms_T[i].second = i;
    }

    // Rearrange data_centroids_T_ based on ascending order of their norms.
    std::sort(norms_T.begin(), norms_T.end(), std::greater<std::pair<float, BucketIdType>>());
    std::vector<float> temp_data(bucket_count_T_ * dim_, 0.0f);
    for (size_t i = 0; i < bucket_count_T_; ++i) {
        BucketIdType src_idx = norms_T[i].second;
        size_t src_offset = src_idx * dim_;
        size_t dst_offset = i * dim_;
        std::copy(data_centroids_T_.data() + src_offset,
                  data_centroids_T_.data() + src_offset + dim_,
                  temp_data.data() + dst_offset);
        norms_T_[i] = norms_T[i].first;
    }
    std::copy(temp_data.begin(), temp_data.end(), data_centroids_T_.data());

    Vector<float> ip_ST(bucket_count_S_ * bucket_count_T_, allocator_);
    matmul(data_centroids_S_.data(),
           data_centroids_T_.data(),
           ip_ST.data(),
           bucket_count_S_,
           bucket_count_T_,
           dim_);
    for (uint32_t i = 0; i < bucket_count_S_ * bucket_count_T_; ++i) {
        BucketIdType cur_bucket_id_T = i % bucket_count_T_;
        precomputed_terms_ST_[i] = norms_T_[cur_bucket_id_T] + ip_ST[i];
    }

    this->is_trained_ = true;
}

Vector<BucketIdType>
GNOIMIPartition::ClassifyDatas(const void* datas, int64_t count, BucketIdType buckets_per_data) {
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    inner_joint_classify_datas(
        reinterpret_cast<const float*>(datas), count, buckets_per_data, result.data());
    return result;
}

Vector<BucketIdType>
GNOIMIPartition::ClassifyDatasForSearch(const void* datas,
                                        int64_t count,
                                        const InnerSearchParam& param) {
    auto buckets_per_data = param.scan_bucket_size;
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    auto candidate_count_S = bucket_count_S_;
    Vector<BucketIdType> candidate_S_id(candidate_count_S, this->allocator_);
    Vector<float> candidate_S_dist(candidate_count_S, this->allocator_);
    Vector<float> dist_to_S(bucket_count_S_ * count, this->allocator_);
    Vector<float> dist_to_T(bucket_count_T_ * count, this->allocator_);
    auto dist_to_S_data = dist_to_S.data();
    auto dist_to_T_data = dist_to_T.data();
    auto candidate_S_id_data = candidate_S_id.data();
    auto candidate_S_dist_data = candidate_S_dist.data();

    matmul(reinterpret_cast<const float*>(datas),
           data_centroids_S_.data(),
           dist_to_S_data,
           count,
           bucket_count_S_,
           dim_);
    matmul(reinterpret_cast<const float*>(datas),
           data_centroids_T_.data(),
           dist_to_T_data,
           count,
           bucket_count_T_,
           dim_);

    for (size_t i = 0; i < count; i++) {
        auto qnorm = FP32ComputeIP(reinterpret_cast<const float*>(datas) + i * dim_,
                                   reinterpret_cast<const float*>(datas) + i * dim_,
                                   dim_) /
                     2;
        MaxHeap heap(this->allocator_);
        for (size_t j = 0; j < bucket_count_S_; ++j) {
            auto dist_term_S = norms_S_[j] - dist_to_S_data[i * bucket_count_S_ + j];
            if (heap.size() < candidate_count_S || dist_term_S < heap.top().first) {
                heap.emplace(dist_term_S, j);
            }
            if (heap.size() > candidate_count_S) {
                heap.pop();
            }
        }
        for (auto j = static_cast<int64_t>(candidate_count_S - 1); j >= 0; --j) {
            candidate_S_id_data[j] = heap.top().second;
            candidate_S_dist_data[j] = heap.top().first;
            heap.pop();
        }
        CHECK_ARGUMENT(heap.empty(), fmt::format("Unexpected non-empty heap after pop candidates"));

        BucketIdType scan_bucket_count_S = static_cast<BucketIdType>(
            std::floor(bucket_count_S_ * param.first_order_scan_ratio));
        scan_bucket_count_S = std::max(scan_bucket_count_S, 1);
        for (size_t j = 0; j < scan_bucket_count_S; ++j) {
            for (size_t k = 0; k < bucket_count_T_; ++k) {
                auto cur_bucket_id_S = candidate_S_id_data[j];
                auto cur_bucket_id_T = k;
                float dist_term_ST =
                    candidate_S_dist_data[j] +
                    precomputed_terms_ST_[cur_bucket_id_S * bucket_count_T_ + cur_bucket_id_T] -
                    dist_to_T_data[i * bucket_count_T_ + cur_bucket_id_T];

                int cur_bucket_id_global = cur_bucket_id_S * bucket_count_T_ + cur_bucket_id_T;
                if (heap.size() < buckets_per_data || dist_term_ST < heap.top().first) {
                    heap.emplace(dist_term_ST, cur_bucket_id_global);
                }
                if (heap.size() > buckets_per_data) {
                    heap.pop();
                }
            }
        }
        BucketIdType size = std::min((BucketIdType)heap.size(), buckets_per_data);
        for (auto j = static_cast<int64_t>(size - 1); j >= 0 && !heap.empty(); --j) {
            result[i * buckets_per_data + j] = heap.top().second;
            heap.pop();
        }
    }
    return result;
}

void
GNOIMIPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    StreamWriter::WriteObj(writer, this->bucket_count_S_);
    StreamWriter::WriteObj(writer, this->bucket_count_T_);
    StreamWriter::WriteVector(writer, this->data_centroids_S_);
    StreamWriter::WriteVector(writer, this->data_centroids_T_);
    StreamWriter::WriteVector(writer, this->norms_S_);
    StreamWriter::WriteVector(writer, this->norms_T_);
    StreamWriter::WriteVector(writer, this->precomputed_terms_ST_);
}
void
GNOIMIPartition::Deserialize(StreamReader& reader) {
    IVFPartitionStrategy::Deserialize(reader);
    StreamReader::ReadObj<BucketIdType>(reader, this->bucket_count_S_);
    StreamReader::ReadObj<BucketIdType>(reader, this->bucket_count_T_);
    StreamReader::ReadVector(reader, this->data_centroids_S_);
    StreamReader::ReadVector(reader, this->data_centroids_T_);
    StreamReader::ReadVector(reader, this->norms_S_);
    StreamReader::ReadVector(reader, this->norms_T_);
    StreamReader::ReadVector(reader, this->precomputed_terms_ST_);
}
void
GNOIMIPartition::factory_router_index(const IndexCommonParam& common_param) {
    auto param_ptr = std::make_shared<BruteForceParameter>();
    param_ptr->flatten_param = std::make_shared<FlattenDataCellParameter>();
    JsonType memory_json = {
        {"type", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    };
    param_ptr->flatten_param->io_parameter = IOParameter::GetIOParameterByJson(memory_json);
    JsonType quantizer_json = {
        {"type", QUANTIZATION_TYPE_VALUE_FP32},
    };
    param_ptr->flatten_param->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(quantizer_json);
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
                                            BucketIdType* result) {
    Vector<float> dist_to_S(bucket_count_S_ * count, this->allocator_);
    Vector<float> dist_to_T(bucket_count_T_ * count, this->allocator_);
    Vector<std::pair<float, BucketIdType>> precomputed_terms_S(bucket_count_S_, this->allocator_);

    matmul(datas, data_centroids_S_.data(), dist_to_S.data(), count, bucket_count_S_, dim_);
    matmul(datas, data_centroids_T_.data(), dist_to_T.data(), count, bucket_count_T_, dim_);
    // |x - s - t|^2 = |x|^2 + |s|^2 + |t|^2 - 2xs - 2xt + 2st
    // precomputed_terms_S: |x - s|^2 = |s|^2 - 2xs + |x|^2
    // precomputed_terms_ST: |t|^2 + 2st
    float total_err = 0.0;
    for (size_t i = 0; i < count; ++i) {
        auto data_norm = FP32ComputeIP(datas + i * dim_, datas + i * dim_, dim_);
        for (size_t j = 0; j < bucket_count_S_; ++j) {
            precomputed_terms_S[j].first =
                norms_S_[j] - dist_to_S[i * bucket_count_S_ + j] + data_norm / 2;
            precomputed_terms_S[j].second = j;
        }
        std::sort(precomputed_terms_S.begin(), precomputed_terms_S.end());

        MaxHeap heap(this->allocator_);
        for (size_t j = 0; j < bucket_count_S_; ++j) {
            float cur_precomputed_term_S = precomputed_terms_S[j].first;
            BucketIdType cur_bucket_id_S = precomputed_terms_S[j].second;

            for (size_t k = 0; k < bucket_count_T_; ++k) {
                BucketIdType cur_bucket_id_T = k;
                if (heap.size() >= buckets_per_data &&
                    std::sqrt(cur_precomputed_term_S) - std::sqrt(norms_T_[cur_bucket_id_T]) >
                        std::sqrt(heap.top().first)) {
                    break;
                }

                int cur_bucket_id_global = cur_bucket_id_S * bucket_count_T_ + cur_bucket_id_T;
                float dist = cur_precomputed_term_S - dist_to_T[i * bucket_count_T_ + k] +
                             precomputed_terms_ST_[cur_bucket_id_global];

                if (heap.size() < buckets_per_data || dist < heap.top().first) {
                    heap.emplace(dist, cur_bucket_id_global);
                }
                if (heap.size() > buckets_per_data) {
                    heap.pop();
                }
            }
        }

        for (auto j = static_cast<int64_t>(buckets_per_data - 1); j >= 0; --j) {
            result[i * buckets_per_data + j] = heap.top().second;
            if (j == 0) {
                total_err += heap.top().first;
            }
            heap.pop();
        }
    }
}

}  // namespace vsag
