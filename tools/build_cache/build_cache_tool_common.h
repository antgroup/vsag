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

#include <cstdint>
#include <string>
#include <vector>

#include <vsag/vsag.h>

#include "typing.h"

namespace vsag::tools {

enum class AlignmentPolicy {
    STRICT,
    TRUNCATE_TO_MIN,
};

struct DatasetAlignmentStats {
    uint64_t raw_label_count = 0;
    uint64_t raw_feature_id_count = 0;
    uint64_t raw_vector_count = 0;
    uint64_t aligned_count = 0;
    bool was_truncated = false;
};

struct BuildParameterConfig {
    std::string raw_json;
    std::string index_name;
    std::string dtype;
    std::string metric_type;
    int64_t dim = 0;
};

struct ImportedDataset {
    DatasetPtr dataset;
    DatasetAlignmentStats stats;
};

std::string
StripCr(std::string line);

AlignmentPolicy
ParseAlignmentPolicy(const std::string& text);

const char*
AlignmentPolicyName(AlignmentPolicy policy);

BuildParameterConfig
ParseBuildParameterSource(const std::string& source);

std::vector<int64_t>
ReadLabels(const std::string& path);

std::vector<std::string>
ReadFeatureIds(const std::string& path);

std::vector<float>
ReadFloat32Vectors(const std::string& path, int64_t dim);

ImportedDataset
LoadFeatureIdDatasetFromFiles(const std::string& label_path,
                              const std::string& feature_id_path,
                              AlignmentPolicy alignment_policy);

ImportedDataset
LoadFloat32DatasetFromFiles(const std::string& label_path,
                            const std::string& feature_id_path,
                            const std::string& vector_path,
                            int64_t dim,
                            AlignmentPolicy alignment_policy);

}  // namespace vsag::tools