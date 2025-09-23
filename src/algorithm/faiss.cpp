
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

#include "faiss.h"

#include <optional>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "faiss/faiss/index_factory.h"
#include "faiss/faiss/index_io.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
#include "vsag/constants.h"

namespace vsag {

Faiss::Faiss(const FaissParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {
    this->faiss_string_ = param->faiss_string;
    auto metric = faiss::MetricType::METRIC_L2;
    if (common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        metric = faiss::MetricType::METRIC_L2;
    } else if (common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    }
    this->index_path_ = param->index_path;
    this->index_ = faiss::index_factory(dim_, faiss_string_.c_str(), metric);
}

std::vector<int64_t>
Faiss::Build(const vsag::DatasetPtr& data) {
    this->Train(data);
    return this->Add(data);
}

void
Faiss::Train(const DatasetPtr& data) {
    this->index_->train(data->GetNumElements(), data->GetFloat32Vectors());
}

std::vector<int64_t>
Faiss::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    this->index_->add(data->GetNumElements(), data->GetFloat32Vectors());
    return failed_ids;
}

DatasetPtr
Faiss::KnnSearch(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 const FilterPtr& filter) const {
    std::shared_lock read_lock(this->global_mutex_);
    auto faiss_ids = std::vector<int64_t>(k);
    auto faiss_dists = std::vector<float>(k);
    this->index_->search(query->GetNumElements(),
                         query->GetFloat32Vectors(),
                         k,
                         faiss_dists.data(),
                         faiss_ids.data());
    auto [dataset_results, dists, ids] = create_fast_dataset(static_cast<int64_t>(k), allocator_);
    for (auto j = 0; j < k; ++j) {
        dists[j] = faiss_dists[j];
        ids[j] = faiss_ids[j];
    }
    return std::move(dataset_results);
}

DatasetPtr
Faiss::RangeSearch(const vsag::DatasetPtr& query,
                   float radius,
                   const std::string& parameters,
                   const vsag::FilterPtr& filter,
                   int64_t limited_size) const {
    auto dataset = Dataset::Make();
    return dataset;
}

class FaissIOWrapper : public faiss::IOWriter {
public:
    FaissIOWrapper(StreamWriter& writer) : writer_(writer) {
    }
    size_t
    operator()(const void* ptr, size_t size, size_t nitems) override {
        writer_.Write(reinterpret_cast<const char*>(ptr), size * nitems);
        return size * nitems;
    }
    int
    filedescriptor() override {
        return -1;
    }

private:
    StreamWriter& writer_;
};

void
Faiss::Serialize(StreamWriter& writer) const {
    FaissIOWrapper faiss_iowrapper(writer);
    faiss::write_index(this->index_, &faiss_iowrapper);
}

void
Faiss::Serialize(std::ostream& ostream) const {
    faiss::write_index(this->index_, this->index_path_.c_str());
}

class FaissIOReader : public faiss::IOReader {
public:
    FaissIOReader(StreamReader& reader) : reader_(reader) {
    }
    size_t
    operator()(void* ptr, size_t size, size_t nitems) override {
        reader_.Read(reinterpret_cast<char*>(ptr), size * nitems);
        return nitems;
    }
    int
    filedescriptor() override {
        return -1;
    }

private:
    StreamReader& reader_;
};

void
Faiss::Deserialize(StreamReader& reader) {
    FaissIOReader faiss_ioreader(reader);
    this->index_ = faiss::read_index(&faiss_ioreader);
}

void
Faiss::Deserialize(std::istream& in_stream) {
    this->index_ = faiss::read_index(this->index_path_.c_str());
}

void
Faiss::InitFeatures() {
    // About Train
}

static const std::string FAISS_PARAMS_TEMPLATE =
    R"(
    {
        "type": "faiss",
        "{FAISS_STRING_KEY}": "FlatL2",
        "{FAISS_INDEX_PATH_KEY}": "/tmp/faiss.index"
    })";

ParamPtr
Faiss::CheckAndMappingExternalParam(const JsonType& external_param,
                                    const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {
            FAISS_STRING,
            {
                FAISS_STRING_KEY,
            },
        },
        {
            FAISS_INDEX_PATH,
            {
                FAISS_INDEX_PATH_KEY,
            },
        },
    };

    std::string str = format_map(FAISS_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto faiss_parameter = std::make_shared<FaissParameter>();
    faiss_parameter->FromJson(inner_json);

    return faiss_parameter;
}

}  // namespace vsag
