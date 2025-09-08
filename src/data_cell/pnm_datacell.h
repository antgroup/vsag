
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

#include <limits>
#include <memory>

#include "common.h"
#include "flatten_interface.h"
#include "quantization/quantizer.h"
#include <pnm_engine_def.h>
#include <pnmesdk_client_c.h>
namespace vsag {
/*
* thread unsafe
*/
template <typename QuantTmpl>
class PnmDatacell : public FlattenInterface {
public:
    explicit PnmDatacell(const QuantizerParamPtr& quantization_param, const IndexCommonParam& common_param):
    allocator_(common_param.allocator_.get()){
        context_.data_type = data_type::FP32;
        context_.vecdim = common_param.dim_;
        this->quantizer_ = std::make_shared<QuantTmpl>(quantization_param, common_param);
        auto database_id = pnmesdk_db_open(&context_, 0, database_name_);
        context_.database_id = database_id;
    }

    ~PnmDatacell() {
        pnmesdk_db_del(&context_);
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator = nullptr) override {
        auto comp = std::static_pointer_cast<Computer<QuantTmpl>>(computer);
        calculate_config config;
        Vector<uint64_t> ids(id_count, allocator_);
        for (auto i = 0; i < id_count; ++i) {
            ids[i] = idx[i];
        }
        config.ids_list = ids.data();
        config.ids_size = id_count;
        config.result_list = result_dists;
        config.target_vector = comp->buf_;
        database_context_cal(&context_, &config);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return this->factory_computer((const float*)query);
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        return 0.0F;
    }

    void
    Train(const void* data, uint64_t count) override {
    }

    void
    InsertVector(const void* vector, InnerIdType idx) override {
        if (total_count_ + 1 != idx) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "invalid idx");
        }
        pnmesdk_db_storage(&context_, const_cast<char*>(static_cast<const char *>(vector)), 1);
    }

    bool
    UpdateVector(const void* vector,
                 InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override {
        return false;
    }

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override {
        pnmesdk_db_storage(&context_, const_cast<char*>(static_cast<const char *>(vectors)), count);
    }

    bool
    Decode(const uint8_t* codes, DataType* data) override {
        return false;
    }

    void
    Resize(InnerIdType new_capacity) override {
    }

    void
    Prefetch(InnerIdType id) override {
    };

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {

    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return "pnm_datacell";
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return MetricType::METRIC_TYPE_L2SQR;
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        return nullptr;
    }

    [[nodiscard]] bool
    InMemory() const override {
        return false;
    }

    bool
    HoldMolds() const override {
        return false;
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        return false;
    }

    void
    Serialize(StreamWriter& writer) override {

    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {

    }

    void
    InitIO(const IOParamPtr& io_param) override {
    }

public:
    std::shared_ptr<Quantizer<QuantTmpl>> quantizer_{nullptr};

    Allocator* const allocator_{nullptr};

private:
    ComputerInterfacePtr
    factory_computer(const float* query) {
        auto computer = this->quantizer_->FactoryComputer();
        computer->SetQuery(query);
        return computer;
    }
private:
    database_context context_;
    char database_name_[5] = "test";
};
}  // namespace vsag
