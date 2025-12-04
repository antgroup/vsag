
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

#include <memory>
#include <string>
#include <vector>
namespace vsag {

enum class IndexDetailDataType {
    TYPE_2DArray_INT64,
    TYPE_1DArray_INT64,
    TYPE_SCALAR_INT64,
    TYPE_SCALAR_DOUBLE,
    TYPE_SCALAR_STRING,
    TYPE_SCALAR_BOOL,
};

class IndexDetailInfo {
public:
    std::string name;
    std::string description;
    IndexDetailDataType type;

    IndexDetailInfo() = default;

    explicit IndexDetailInfo(const std::string& name,
                             const std::string& description,
                             IndexDetailDataType type)
        : name(name), description(description), type(type) {
    }
};

extern const char* INDEX_DETAIL_NAME_NUM_ELEMENTS;
extern const char* INDEX_DETAIL_NAME_LABEL_TABLE;
class DetailData {
public:
    virtual ~DetailData() = default;

    virtual std::vector<int64_t>
    GetData1DArrayInt64() = 0;

    virtual const std::vector<int64_t>&
    GetData1DArrayInt64() const = 0;

    virtual std::vector<std::vector<int64_t>>
    GetData2DArrayInt64() = 0;

    virtual const std::vector<std::vector<int64_t>>&
    GetData2DArrayInt64() const = 0;

    virtual std::string
    GetDataScalarString() = 0;

    virtual const std::string&
    GetDataScalarString() const = 0;

    virtual bool
    GetDataScalarBool() = 0;

    virtual int64_t
    GetDataScalarInt64() = 0;

    virtual double
    GetDataScalarDouble() = 0;
};

using DetailDataPtr = std::shared_ptr<DetailData>;

}  // namespace vsag
