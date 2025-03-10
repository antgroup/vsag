
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

#include <fmt/format-inl.h>

#include "impl/kmeans_cluster.h"
#include "vsag/factory.h"

namespace vsag {

static constexpr const char* SEARCH_PARAM_TEMPLATE_STR = R"(
{{
    "hnsw": {{
        "ef_search": {}
    }}
}}
)";

static constexpr const char* BRUTE_FORCE_FACTORY_STR = R"(
{{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": {}
}}
)";

IVFNearestPartition::IVFNearestPartition(Allocator* allocator,
                                         BucketIdType bucket_count,
                                         int64_t dim,
                                         IVFNearestPartitionTrainerType trainer_type)
    : IVFPartitionStrategy(allocator, bucket_count, dim), trainer_type_(trainer_type) {
    this->factory_router_index();
}

void
IVFNearestPartition::Train(const DatasetPtr dataset) {
    auto dim = this->dim_;
    auto centroids = Dataset::Make();
    Vector<float> data(bucket_count_ * dim, allocator_);
    Vector<LabelType> ids(this->bucket_count_, allocator_);
    std::iota(ids.begin(), ids.end(), 0);
    centroids->Ids(ids.data())
        ->Dim(dim)
        ->Float32Vectors(data.data())
        ->NumElements(this->bucket_count_)
        ->Owner(false);

    if (trainer_type_ == IVFNearestPartitionTrainerType::KMeansTrainer) {
        KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_);
        cls.Run(this->bucket_count_, dataset->GetFloat32Vectors(), dataset->GetNumElements());
        memcpy(data.data(), cls.k_centroids_, dim * this->bucket_count_ * sizeof(float));
    } else if (trainer_type_ == IVFNearestPartitionTrainerType::RandomTrainer) {
        // TODO(LHT) implement
    }

    auto build_result = this->route_index_ptr_->Build(centroids);
    if (not build_result.has_value()) {
        throw std::runtime_error("ivf train failed");
    }
    this->is_trained_ = true;
}

Vector<BucketIdType>
IVFNearestPartition::ClassifyDatas(const void* datas,
                                   int64_t count,
                                   BucketIdType buckets_per_data) {
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    for (int64_t i = 0; i < count; ++i) {
        auto query = Dataset::Make();
        query->Dim(this->dim_)
            ->Float32Vectors(reinterpret_cast<const float*>(datas) + i * this->dim_)
            ->NumElements(1)
            ->Owner(false);
        auto search_param = fmt::format(
            SEARCH_PARAM_TEMPLATE_STR, std::max(10L, static_cast<int64_t>(buckets_per_data * 1.2)));
        auto search_result =
            this->route_index_ptr_->KnnSearch(query, buckets_per_data, search_param).value();
        const auto* result_ids = search_result->GetIds();

        for (int64_t j = 0; j < buckets_per_data; ++j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(result_ids[j]);
        }
    }
    return result;
}
void
IVFNearestPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    auto binary = this->route_index_ptr_->Serialize().value();
    auto keys = binary.GetKeys();
    uint64_t key_count = keys.size();
    StreamWriter::WriteObj(writer, key_count);
    for (const auto& str : keys) {
        StreamWriter::WriteString(writer, str);
        auto data = binary.Get(str);
        StreamWriter::WriteObj(writer, data.size);
        writer.Write(reinterpret_cast<const char*>(data.data.get()), data.size);
    }
}
void
IVFNearestPartition::Deserialize(StreamReader& reader) {
    IVFPartitionStrategy::Deserialize(reader);
    BinarySet binary;
    uint64_t key_count;
    StreamReader::ReadObj(reader, key_count);
    for (int i = 0; i < key_count; ++i) {
        auto key = StreamReader::ReadString(reader);
        Binary b;
        StreamReader::ReadObj(reader, b.size);
        std::shared_ptr<int8_t[]> bin(new int8_t[b.size]);
        b.data = bin;
        reader.Read(reinterpret_cast<char*>(bin.get()), b.size);
        binary.Set(key, b);
    }
    this->route_index_ptr_->Deserialize(binary);
}
void
IVFNearestPartition::factory_router_index() {
    this->route_index_ptr_ =
        Factory::CreateIndex(
            INDEX_BRUTE_FORCE, fmt::format(BRUTE_FORCE_FACTORY_STR, this->dim_), this->allocator_)
            .value();
}
}  // namespace vsag
