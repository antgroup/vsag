
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

#include <vsag/index_detail_info.h>

#include <variant>
#include <vector>

namespace vsag {

class DetailDataImpl : public DetailData {
public:
    using Data1DArrayInt64 = std::vector<int64_t>;
    using Data2DArrayInt64 = std::vector<std::vector<int64_t>>;
    using DataScalarInt64 = int64_t;
    using DataScalarDouble = double;
    using DataScalarString = std::string;
    using DataScalarBool = bool;

    explicit DetailDataImpl() : DetailData(){};

    virtual ~DetailDataImpl() = default;

    std::vector<int64_t>
    GetData1DArrayInt64() override {
        return std::get<Data1DArrayInt64>(data_);
    }

    const std::vector<int64_t>&
    GetData1DArrayInt64() const override {
        return std::get<Data1DArrayInt64>(data_);
    }

    std::vector<std::vector<int64_t>>
    GetData2DArrayInt64() override {
        return std::get<Data2DArrayInt64>(data_);
    }

    const std::vector<std::vector<int64_t>>&
    GetData2DArrayInt64() const override {
        return std::get<Data2DArrayInt64>(data_);
    }

    std::string
    GetDataScalarString() override {
        return std::get<DataScalarString>(data_);
    }

    const std::string&
    GetDataScalarString() const override {
        return std::get<DataScalarString>(data_);
    }

    bool
    GetDataScalarBool() override {
        return std::get<DataScalarBool>(data_);
    }

    int64_t
    GetDataScalarInt64() override {
        return std::get<DataScalarInt64>(data_);
    }

    double
    GetDataScalarDouble() override {
        return std::get<DataScalarDouble>(data_);
    }

public:
    void
    SetData1DArrayInt64(const Data1DArrayInt64& data) {
        data_ = data;
    }

    void
    SetData2DArrayInt64(const Data2DArrayInt64& data) {
        data_ = data;
    }

    void
    SetDataScalarInt64(const DataScalarInt64& data) {
        data_ = data;
    }

    void
    SetDataScalarDouble(const DataScalarDouble& data) {
        data_ = data;
    }

    void
    SetDataScalarString(const DataScalarString& data) {
        data_ = data;
    }

    void
    SetDataScalarBool(const DataScalarBool& data) {
        data_ = data;
    }

private:
    std::variant<Data1DArrayInt64,
                 Data2DArrayInt64,
                 DataScalarInt64,
                 DataScalarDouble,
                 DataScalarString,
                 DataScalarBool>
        data_;
};
}  // namespace vsag
