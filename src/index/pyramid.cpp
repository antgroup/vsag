
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

#include "pyramid.h"

namespace vsag {

tl::expected<std::vector<int64_t>, Error>
Pyramid::Build(const DatasetPtr& base) {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const std::function<bool(int64_t)>& filter) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     int64_t limited_size) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     BitsetPtr invalid,
                     int64_t limited_size) const {
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const std::function<bool(int64_t)>& filter,
                     int64_t limited_size) const {
    return {};
}

tl::expected<BinarySet, Error>
Pyramid::Serialize() const {
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const BinarySet& binary_set) {
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const ReaderSet& reader_set) {
    return {};
}

int64_t
Pyramid::GetNumElements() const {
    return 0;
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

}  // namespace vsag
