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

#include "index_utils.h"

#include <fmt/format.h>

#include <memory>

#include "common.h"
#include "dataset_impl.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag::index_utils {

DatasetPtr
MakeEmptyResult(const std::string& stats_json) {
    auto dataset_result = DatasetImpl::MakeEmptyDataset();
    if (!stats_json.empty()) {
        dataset_result->Statistics(stats_json);
    }
    return dataset_result;
}

void
WriteIndexFooter(StreamWriter& writer, const JsonType& basic_info) {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("basic_info", basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

bool
ReadIndexFooter(StreamReader& reader, JsonType& basic_info) {
    auto footer = Footer::Parse(reader);
    if (footer == nullptr) {
        return false;
    }
    auto metadata = footer->GetMetadata();
    if (metadata == nullptr || metadata->EmptyIndex()) {
        throw VsagException(ErrorType::INDEX_EMPTY, "index is empty");
    }
    basic_info = metadata->Get("basic_info");
    return true;
}

void
ValidateSearchQuery(DataTypes data_type, int64_t dim, const DatasetPtr& query) {
    if (data_type != DataTypes::DATA_TYPE_SPARSE) {
        int64_t query_dim = query->GetDim();
        CHECK_ARGUMENT(query_dim == dim,
                       fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim));
    }
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");
}

void
ValidateKnnArgs(DataTypes data_type, int64_t dim, const DatasetPtr& query, int64_t k) {
    ValidateSearchQuery(data_type, dim, query);
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
}

void
ValidateRangeArgs(
    DataTypes data_type, int64_t dim, const DatasetPtr& query, float radius, int64_t limited_size) {
    ValidateSearchQuery(data_type, dim, query);
    CHECK_ARGUMENT(radius >= 0.0F,
                   fmt::format("radius({}) must be greater than or equal to 0", radius));
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));
}

}  // namespace vsag::index_utils
