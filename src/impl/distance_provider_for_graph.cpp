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

#include "impl/distance_provider_for_graph.h"

#include <limits>

#include "datacell/bucket_interface.h"

namespace vsag {

float
FlattenIdDistanceProvider::QueryDistance(InnerIdType id, QueryContext* ctx) const {
    float distance = 0.0F;
    flatten_->QueryById(&distance, query_id_, &id, 1, ctx);
    return distance;
}

void
FlattenIdDistanceProvider::BatchQueryDistance(float* distances,
                                              const InnerIdType* ids,
                                              InnerIdType count,
                                              QueryContext* ctx) const {
    flatten_->QueryById(distances, query_id_, ids, count, ctx);
}

float
FlattenIdDistanceProvider::PairwiseDistance(InnerIdType id1,
                                            InnerIdType id2,
                                            const ComputerInterfacePtr& /*computer*/) const {
    return flatten_->ComputePairVectors(id1, id2);
}

ComputerInterfacePtr
FlattenIdDistanceProvider::FactoryComputerById(InnerIdType /*id*/) const {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "FlattenIdDistanceProvider cannot create a computer by vector ID");
}

void
FlattenIdDistanceProvider::Prefetch(InnerIdType id) const {
    flatten_->Prefetch(id);
}

BucketDistanceProvider::BucketDistanceProvider(std::shared_ptr<BucketInterface> bucket,
                                               BucketIdType bucket_id,
                                               ComputerInterfacePtr computer,
                                               ComputerFactory computer_factory,
                                               const InnerIdType* inner_ids,
                                               BucketIdType buckets_per_data)
    : bucket_(std::move(bucket)),
      bucket_id_(bucket_id),
      computer_(std::move(computer)),
      computer_factory_(std::move(computer_factory)),
      inner_ids_(inner_ids),
      buckets_per_data_(buckets_per_data) {
    CHECK_ARGUMENT(buckets_per_data_ > 0, "buckets_per_data must be positive");
}

BucketDistanceProvider::BucketDistanceProvider(std::shared_ptr<BucketInterface> bucket,
                                               BucketIdType bucket_id,
                                               ComputerInterfacePtr computer,
                                               const InnerIdType* inner_ids,
                                               BucketIdType buckets_per_data)
    : BucketDistanceProvider(
          std::move(bucket), bucket_id, std::move(computer), {}, inner_ids, buckets_per_data) {
}

float
BucketDistanceProvider::QueryDistance(InnerIdType id, QueryContext* /*ctx*/) const {
    if (not IsValid(id)) {
        return std::numeric_limits<float>::infinity();
    }
    return bucket_->QueryOneById(computer_, bucket_id_, id);
}

float
BucketDistanceProvider::PairwiseDistance(InnerIdType id1,
                                         InnerIdType id2,
                                         const ComputerInterfacePtr& computer) const {
    CHECK_ARGUMENT(IsValid(id1) && IsValid(id2), "cannot compute distance for a bucket hole");
    return bucket_->QueryOneById(
        computer != nullptr ? computer : FactoryComputerById(id1), bucket_id_, id2);
}

ComputerInterfacePtr
BucketDistanceProvider::FactoryComputerById(InnerIdType id) const {
    CHECK_ARGUMENT(IsValid(id), "cannot create a computer for a bucket hole");
    if (not computer_factory_) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "BucketDistanceProvider has no source-computer factory");
    }
    return computer_factory_(id);
}

void
BucketDistanceProvider::Prefetch(InnerIdType id) const {
    if (IsValid(id)) {
        bucket_->Prefetch(bucket_id_, id);
    }
}

InnerIdType
BucketDistanceProvider::OriginalId(InnerIdType id) const {
    return inner_ids_ == nullptr ? id
                                 : (IsValid(id) ? inner_ids_[id] / buckets_per_data_
                                                : std::numeric_limits<InnerIdType>::max());
}

bool
BucketDistanceProvider::IsValid(InnerIdType id) const {
    return inner_ids_ == nullptr || (id < bucket_->GetBucketSize(bucket_id_) &&
                                     inner_ids_[id] != std::numeric_limits<InnerIdType>::max());
}

}  // namespace vsag
