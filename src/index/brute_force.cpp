
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

#include "brute_force.h"
namespace vsag {

BruteForce::BruteForce(const BruteForceParameter& param, const IndexCommonParam& common_param)
    : Index(), dim_(common_param.dim_), allocator_(common_param.allocator_) {
}
BruteForce::~BruteForce() {
}

int64_t
BruteForce::GetMemoryUsage() const {
    return 0;
}

uint64_t
BruteForce::EstimateMemory(uint64_t num_elements) const {
    return Index::EstimateMemory(num_elements);
}

bool
BruteForce::CheckFeature(IndexFeature feature) const {
    return Index::CheckFeature(feature);
}

std::vector<int64_t>
BruteForce::build(const DatasetPtr& data) {
    return std::vector<int64_t>();
}

std::vector<int64_t>
BruteForce::add(const DatasetPtr& data) {
    return std::vector<int64_t>();
}

DatasetPtr
BruteForce::knn_search(const DatasetPtr& query,
                       int64_t k,
                       const std::string& parameters,
                       const std::function<bool(int64_t)>& filter) const {
    return vsag::DatasetPtr();
}

DatasetPtr
BruteForce::range_search(const DatasetPtr& query,
                         float radius,
                         const std::string& parameters,
                         BaseFilterFunctor* filter_ptr,
                         int64_t limited_size) const {
    return vsag::DatasetPtr();
}

float
BruteForce::calculate_distance_by_id(const float* vector, int64_t id) const {
    return 0;
}

BinarySet
BruteForce::serialize() const {
    return BinarySet();
}

void
BruteForce::serialize(std::ostream& out_stream) {
}

void
BruteForce::deserialize(std::istream& in_stream) {
}

void
BruteForce::deserialize(const BinarySet& binary_set) {
}

void
BruteForce::deserialize(const ReaderSet& reader_set) {
}

}  // namespace vsag
