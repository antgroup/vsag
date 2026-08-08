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

#include <functional>
#include <memory>
#include <utility>

#include "basic_types.h"
#include "datacell/flatten_interface.h"
#include "quantization/computer.h"
#include "query_context.h"
#include "vsag_exception.h"

namespace vsag {

class BucketInterface;

class DistanceProviderForGraph {
public:
    virtual ~DistanceProviderForGraph() = default;

    [[nodiscard]] virtual float
    QueryDistance(InnerIdType id, QueryContext* ctx = nullptr) const = 0;

    virtual void
    BatchQueryDistance(float* distances,
                       const InnerIdType* ids,
                       InnerIdType count,
                       QueryContext* ctx = nullptr) const {
        for (InnerIdType i = 0; i < count; ++i) {
            distances[i] = this->QueryDistance(ids[i], ctx);
        }
    }

    [[nodiscard]] virtual float
    PairwiseDistance(InnerIdType id1,
                     InnerIdType id2,
                     const ComputerInterfacePtr& computer = nullptr) const = 0;

    [[nodiscard]] virtual ComputerInterfacePtr
    FactoryComputerById(InnerIdType id) const = 0;

    [[nodiscard]] virtual bool
    SupportsComputerById() const {
        return false;
    }

    virtual void
    Prefetch(InnerIdType id) const {
    }

    [[nodiscard]] virtual InnerIdType
    OriginalId(InnerIdType id) const {
        return id;
    }

    [[nodiscard]] virtual bool
    IsValid(InnerIdType) const {
        return true;
    }
};

class FlattenDistanceProvider final : public DistanceProviderForGraph {
public:
    FlattenDistanceProvider(FlattenInterfacePtr flatten, ComputerInterfacePtr computer)
        : flatten_(std::move(flatten)), computer_(std::move(computer)) {
    }

    [[nodiscard]] float
    QueryDistance(InnerIdType id, QueryContext* ctx = nullptr) const override {
        float distance = 0.0F;
        flatten_->Query(&distance, computer_, &id, 1, ctx);
        return distance;
    }

    void
    BatchQueryDistance(float* distances,
                       const InnerIdType* ids,
                       InnerIdType count,
                       QueryContext* ctx = nullptr) const override {
        flatten_->Query(distances, computer_, ids, count, ctx);
    }

    [[nodiscard]] float
    PairwiseDistance(InnerIdType id1, InnerIdType id2, const ComputerInterfacePtr&) const override {
        return flatten_->ComputePairVectors(id1, id2);
    }

    [[nodiscard]] ComputerInterfacePtr
    FactoryComputerById(InnerIdType) const override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "FlattenDistanceProvider cannot create a computer by vector ID");
    }

    void
    Prefetch(InnerIdType id) const override {
        flatten_->Prefetch(id);
    }

private:
    FlattenInterfacePtr flatten_;
    ComputerInterfacePtr computer_;
};

class FlattenIdDistanceProvider final : public DistanceProviderForGraph {
public:
    FlattenIdDistanceProvider(FlattenInterfacePtr flatten, InnerIdType query_id)
        : flatten_(std::move(flatten)), query_id_(query_id) {
    }

    [[nodiscard]] float
    QueryDistance(InnerIdType id, QueryContext* ctx = nullptr) const override;

    void
    BatchQueryDistance(float* distances,
                       const InnerIdType* ids,
                       InnerIdType count,
                       QueryContext* ctx = nullptr) const override;

    [[nodiscard]] float
    PairwiseDistance(InnerIdType id1,
                     InnerIdType id2,
                     const ComputerInterfacePtr& computer = nullptr) const override;

    [[nodiscard]] ComputerInterfacePtr
    FactoryComputerById(InnerIdType id) const override;

    void
    Prefetch(InnerIdType id) const override;

private:
    FlattenInterfacePtr flatten_;
    InnerIdType query_id_;
};

class BucketDistanceProvider final : public DistanceProviderForGraph {
public:
    using ComputerFactory = std::function<ComputerInterfacePtr(InnerIdType)>;

    BucketDistanceProvider(std::shared_ptr<BucketInterface> bucket,
                           BucketIdType bucket_id,
                           ComputerInterfacePtr computer,
                           ComputerFactory computer_factory,
                           const InnerIdType* inner_ids = nullptr,
                           BucketIdType buckets_per_data = 1);

    // Query-only provider for searches. Pairwise operations require the factory overload above.
    BucketDistanceProvider(std::shared_ptr<BucketInterface> bucket,
                           BucketIdType bucket_id,
                           ComputerInterfacePtr computer,
                           const InnerIdType* inner_ids = nullptr,
                           BucketIdType buckets_per_data = 1);

    [[nodiscard]] float
    QueryDistance(InnerIdType id, QueryContext* ctx = nullptr) const override;

    [[nodiscard]] float
    PairwiseDistance(InnerIdType id1,
                     InnerIdType id2,
                     const ComputerInterfacePtr& computer = nullptr) const override;

    [[nodiscard]] ComputerInterfacePtr
    FactoryComputerById(InnerIdType id) const override;

    [[nodiscard]] bool
    SupportsComputerById() const override {
        return static_cast<bool>(computer_factory_);
    }

    void
    Prefetch(InnerIdType id) const override;

    [[nodiscard]] InnerIdType
    OriginalId(InnerIdType id) const override;

    [[nodiscard]] bool
    IsValid(InnerIdType id) const override;

private:
    std::shared_ptr<BucketInterface> bucket_;
    BucketIdType bucket_id_;
    ComputerInterfacePtr computer_;
    ComputerFactory computer_factory_;
    const InnerIdType* inner_ids_;
    BucketIdType buckets_per_data_;
};

}  // namespace vsag
