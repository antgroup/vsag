
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

#include <vsag/factory.h>
#include <vsag/index.h>
#include <vsag/vsag_c_api.h>

#include <fstream>

class VsagIndex {
public:
    VsagIndex(std::shared_ptr<vsag::Index> index) : index_(std::move(index)) {
    }

    std::shared_ptr<vsag::Index> index_;
};

extern "C" {
vsag_index_t
vsag_index_factory(const char* index_name, const char* index_param) {
    try {
        auto index = vsag::Factory::CreateIndex(index_name, index_param);
        if (index.has_value()) {
            return new VsagIndex(index.value());
        }
        return nullptr;

    } catch (const std::exception& e) {
        return nullptr;
    }
}

bool
vsag_index_destroy(vsag_index_t index) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        delete vsag_index;
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool
vsag_index_build(
    vsag_index_t index, const float* data, const int64_t* ids, uint64_t dim, uint64_t count) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            auto dataset = vsag::Dataset::Make();
            dataset->Owner(false)
                ->Dim(static_cast<int64_t>(dim))
                ->NumElements(static_cast<int64_t>(count))
                ->Ids(ids)
                ->Float32Vectors(data);
            vsag_index->index_->Build(dataset);
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool
vsag_index_knn_search(vsag_index_t index,
                      const float* query,
                      uint64_t dim,
                      int64_t k,
                      const char* parameters,
                      float* results,
                      int64_t* ids) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            auto query_dataset = vsag::Dataset::Make();
            query_dataset->Owner(false)
                ->Dim(static_cast<int64_t>(dim))
                ->NumElements(static_cast<int64_t>(1))
                ->Float32Vectors(query);
            auto result = vsag_index->index_->KnnSearch(query_dataset, k, parameters);
            if (result.has_value()) {
                const auto* ids_view = result.value()->GetIds();
                const auto* dists_view = result.value()->GetDistances();
                auto real_k = result.value()->GetDim();
                for (int i = 0; i < real_k; ++i) {
                    ids[i] = ids_view[i];
                    results[i] = dists_view[i];
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool
vsag_serialize_file(vsag_index_t index, const char* file_path) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            std::ofstream file(file_path, std::ios::binary);
            vsag_index->index_->Serialize(file);
            file.close();
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool
vsag_deserialize_file(vsag_index_t index, const char* file_path) {
    try {
        auto* vsag_index = static_cast<VsagIndex*>(index);
        if (vsag_index != nullptr) {
            std::ifstream file(file_path, std::ios::binary);
            vsag_index->index_->Deserialize(file);
            file.close();
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}
}
