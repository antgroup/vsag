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

#include "build_cache_tool_common.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace vsag::tools {

namespace {

std::string
trim_left(std::string text) {
    auto iter = std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return not std::isspace(ch);
    });
    text.erase(text.begin(), iter);
    return text;
}

std::string
load_text_file(const std::string& path) {
    std::ifstream input(path);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open text file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

void
validate_feature_ids(const std::vector<std::string>& feature_ids) {
    if (feature_ids.empty()) {
        throw std::runtime_error("id_map is empty");
    }

    std::unordered_set<std::string> unique_feature_ids;
    unique_feature_ids.reserve(feature_ids.size());
    for (const auto& feature_id : feature_ids) {
        if (feature_id.empty()) {
            throw std::runtime_error("id_map contains empty feature id");
        }
        auto inserted = unique_feature_ids.emplace(feature_id);
        if (not inserted.second) {
            throw std::runtime_error("duplicate feature_id found: " + feature_id);
        }
    }
}

void
align_row_counts(std::vector<int64_t>* labels,
                 std::vector<std::string>* feature_ids,
                 std::vector<float>* vectors,
                 int64_t dim,
                 AlignmentPolicy alignment_policy,
                 DatasetAlignmentStats* stats) {
    if (labels == nullptr || feature_ids == nullptr || stats == nullptr) {
        throw std::runtime_error("alignment input is nullptr");
    }

    const auto label_count = labels->size();
    const auto feature_id_count = feature_ids->size();
    uint64_t vector_count = 0;
    if (vectors != nullptr) {
        if (dim <= 0) {
            throw std::runtime_error("dim must be positive when vectors are provided");
        }
        vector_count = static_cast<uint64_t>(vectors->size() / static_cast<size_t>(dim));
    }

    stats->raw_label_count = label_count;
    stats->raw_feature_id_count = feature_id_count;
    stats->raw_vector_count = vector_count;

    uint64_t aligned_count = std::min<uint64_t>(label_count, feature_id_count);
    if (vectors != nullptr) {
        aligned_count = std::min<uint64_t>(aligned_count, vector_count);
    }

    const bool mismatch = labels->size() != feature_ids->size() ||
                          (vectors != nullptr &&
                           (static_cast<uint64_t>(labels->size()) != vector_count ||
                            static_cast<uint64_t>(feature_ids->size()) != vector_count));

    if (mismatch && alignment_policy == AlignmentPolicy::STRICT) {
        throw std::runtime_error("dataset row count mismatch: labels=" +
                                 std::to_string(labels->size()) +
                                 ", feature_ids=" +
                                 std::to_string(feature_ids->size()) +
                                 ", vectors=" +
                                 std::to_string(vector_count));
    }

    if (aligned_count == 0) {
        throw std::runtime_error("aligned dataset is empty");
    }

    stats->aligned_count = aligned_count;
    stats->was_truncated = mismatch;

    labels->resize(aligned_count);
    feature_ids->resize(aligned_count);
    if (vectors != nullptr) {
        vectors->resize(static_cast<size_t>(aligned_count) * static_cast<size_t>(dim));
    }
}

DatasetPtr
make_feature_id_dataset(const std::vector<int64_t>& labels,
                        const std::vector<std::string>& feature_ids) {
    auto* labels_data = new int64_t[labels.size()];
    auto* feature_ids_data = new std::string[feature_ids.size()];
    std::copy(labels.begin(), labels.end(), labels_data);
    std::copy(feature_ids.begin(), feature_ids.end(), feature_ids_data);

    return Dataset::Make()
        ->Owner(true)
        ->NumElements(static_cast<int64_t>(labels.size()))
        ->Ids(labels_data)
        ->FeatureIds(feature_ids_data);
}

DatasetPtr
make_float32_dataset(const std::vector<int64_t>& labels,
                     const std::vector<std::string>& feature_ids,
                     const std::vector<float>& vectors,
                     int64_t dim) {
    auto* labels_data = new int64_t[labels.size()];
    auto* feature_ids_data = new std::string[feature_ids.size()];
    auto* vectors_data = new float[vectors.size()];

    std::copy(labels.begin(), labels.end(), labels_data);
    std::copy(feature_ids.begin(), feature_ids.end(), feature_ids_data);
    std::copy(vectors.begin(), vectors.end(), vectors_data);

    return Dataset::Make()
        ->Owner(true)
        ->NumElements(static_cast<int64_t>(labels.size()))
        ->Dim(dim)
        ->Ids(labels_data)
        ->Float32Vectors(vectors_data)
        ->FeatureIds(feature_ids_data);
}

}  // namespace

std::string
StripCr(std::string line) {
    if (not line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

AlignmentPolicy
ParseAlignmentPolicy(const std::string& text) {
    if (text == "strict") {
        return AlignmentPolicy::STRICT;
    }
    if (text == "truncate_to_min") {
        return AlignmentPolicy::TRUNCATE_TO_MIN;
    }
    throw std::runtime_error(
        "invalid --alignment_policy, expected one of: strict, truncate_to_min");
}

const char*
AlignmentPolicyName(AlignmentPolicy policy) {
    switch (policy) {
        case AlignmentPolicy::STRICT:
            return "strict";
        case AlignmentPolicy::TRUNCATE_TO_MIN:
            return "truncate_to_min";
    }
    return "unknown";
}

BuildParameterConfig
ParseBuildParameterSource(const std::string& source) {
    if (source.empty()) {
        throw std::runtime_error("build parameter source is empty");
    }

    auto normalized = trim_left(source);
    if (normalized.empty()) {
        throw std::runtime_error("build parameter source is empty");
    }

    std::string raw_json = normalized.front() == '{' ? source : load_text_file(source);
    auto parsed = JsonType::Parse(raw_json);

    BuildParameterConfig config;
    config.raw_json = raw_json;

    if (parsed.Contains("type")) {
        config.index_name = parsed["type"].GetString();
    } else if (parsed.Contains("hgraph")) {
        config.index_name = "hgraph";
    }

    if (config.index_name != "hgraph") {
        throw std::runtime_error("build parameter must describe HGraph index");
    }
    if (not parsed.Contains("dim")) {
        throw std::runtime_error("build parameter missing dim");
    }
    if (not parsed.Contains("dtype")) {
        throw std::runtime_error("build parameter missing dtype");
    }
    if (not parsed.Contains("metric_type")) {
        throw std::runtime_error("build parameter missing metric_type");
    }

    config.dim = parsed["dim"].GetInt();
    config.dtype = parsed["dtype"].GetString();
    config.metric_type = parsed["metric_type"].GetString();
    return config;
}

std::vector<int64_t>
ReadLabels(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open label file: " + path);
    }

    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || (size % static_cast<std::streamoff>(sizeof(int64_t))) != 0) {
        throw std::runtime_error("label.bin size is not aligned to int64: " + path);
    }

    std::vector<int64_t> labels(static_cast<size_t>(size / sizeof(int64_t)));
    if (not labels.empty()) {
        input.read(reinterpret_cast<char*>(labels.data()), size);
    }
    if (not input.good() && not input.eof()) {
        throw std::runtime_error("failed to read label file: " + path);
    }
    return labels;
}

std::vector<std::string>
ReadFeatureIds(const std::string& path) {
    std::ifstream input(path);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open feature id file: " + path);
    }

    std::vector<std::string> feature_ids;
    std::string line;
    while (std::getline(input, line)) {
        feature_ids.emplace_back(StripCr(std::move(line)));
    }

    if (not feature_ids.empty() && feature_ids.back().empty()) {
        feature_ids.pop_back();
    }
    return feature_ids;
}

std::vector<float>
ReadFloat32Vectors(const std::string& path, int64_t dim) {
    if (dim <= 0) {
        throw std::runtime_error("dim must be positive when reading vector_data.bin");
    }

    std::ifstream input(path, std::ios::binary);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open vector file: " + path);
    }

    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || (size % static_cast<std::streamoff>(sizeof(float))) != 0) {
        throw std::runtime_error("vector_data.bin size is not aligned to float32: " + path);
    }

    const auto float_count = static_cast<uint64_t>(size / sizeof(float));
    if (float_count % static_cast<uint64_t>(dim) != 0) {
        throw std::runtime_error("vector_data.bin float count is not aligned to dim: " + path);
    }

    if (float_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("vector_data.bin is too large to load into memory: " + path);
    }

    std::vector<float> vectors(static_cast<size_t>(float_count));
    if (not vectors.empty()) {
        input.read(reinterpret_cast<char*>(vectors.data()), size);
    }
    if (not input.good() && not input.eof()) {
        throw std::runtime_error("failed to read vector file: " + path);
    }
    return vectors;
}

ImportedDataset
LoadFeatureIdDatasetFromFiles(const std::string& label_path,
                              const std::string& feature_id_path,
                              AlignmentPolicy alignment_policy) {
    auto labels = ReadLabels(label_path);
    auto feature_ids = ReadFeatureIds(feature_id_path);

    ImportedDataset imported;
    align_row_counts(&labels, &feature_ids, nullptr, 0, alignment_policy, &imported.stats);
    validate_feature_ids(feature_ids);
    imported.dataset = make_feature_id_dataset(labels, feature_ids);
    return imported;
}

ImportedDataset
LoadFloat32DatasetFromFiles(const std::string& label_path,
                            const std::string& feature_id_path,
                            const std::string& vector_path,
                            int64_t dim,
                            AlignmentPolicy alignment_policy) {
    auto labels = ReadLabels(label_path);
    auto feature_ids = ReadFeatureIds(feature_id_path);
    auto vectors = ReadFloat32Vectors(vector_path, dim);

    ImportedDataset imported;
    align_row_counts(&labels, &feature_ids, &vectors, dim, alignment_policy, &imported.stats);
    validate_feature_ids(feature_ids);
    imported.dataset = make_float32_dataset(labels, feature_ids, vectors, dim);
    return imported;
}

}  // namespace vsag::tools