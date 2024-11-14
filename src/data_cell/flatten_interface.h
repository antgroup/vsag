
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

#include <nlohmann/json.hpp>
#include <string>

#include "index/index_common_param.h"
#include "quantization/computer.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"

namespace vsag {
class FlattenInterface;
using FlattenInterfacePtr = std::shared_ptr<FlattenInterface>;

class FlattenInterface {
public:
    FlattenInterface() = default;

    static FlattenInterfacePtr
    MakeInstance(const JsonType& flatten_interface_param, const IndexCommonParam& common_param);

public:
    virtual void
    Query(float* result_dists,
          ComputerInterfacePtr computer,
          const InnerIdType* idx,
          InnerIdType id_count) = 0;

    virtual ComputerInterfacePtr
    FactoryComputer(const float* query) = 0;

    virtual void
    Train(const float* data, uint64_t count) = 0;

    virtual void
    InsertVector(const float* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) = 0;

    virtual void
    BatchInsertVector(const float* vectors, InnerIdType count, InnerIdType* idx = nullptr) = 0;

    virtual float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) = 0;

    virtual void
    Prefetch(InnerIdType id) = 0;

public:
    virtual void
    SetMaxCapacity(InnerIdType capacity) {
        this->max_capacity_ = capacity;
    };

    [[nodiscard]] virtual const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const {
        return nullptr;
    }

    virtual bool
    GetCodesById(InnerIdType id, uint8_t* codes) const {
        return false;
    }

    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        return this->total_count_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

public:
    InnerIdType total_count_{0};
    InnerIdType max_capacity_{1000000};
    uint32_t code_size_{0};
};

}  // namespace vsag
