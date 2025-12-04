
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

#include "dataset_impl.h"

#include <cstring>

#include "typing.h"
#include "vsag_exception.h"

namespace vsag {

DatasetPtr
Dataset::Make() {
    return std::make_shared<DatasetImpl>();
}

DatasetPtr
DatasetImpl::MakeEmptyDataset() {
    auto result = std::make_shared<DatasetImpl>();
    result->Dim(0)->NumElements(1);
    return result;
}

template <typename T>
inline T*
new_element(T*& old_dest, size_t old_count, size_t new_total) {
    T* dest = new T[new_total];
    if (old_dest != nullptr) {
        memcpy(dest, old_dest, old_count * sizeof(T));
    }
    delete[] old_dest;  // Free the old memory if it was allocated with new[]
    old_dest = nullptr;
    return dest;
}

template <typename T>
inline T*
allocator_element(Allocator* allocator, T* old_dest, size_t new_size_in_bytes) {
    if (old_dest != nullptr) {
        return static_cast<T*>(allocator->Reallocate(old_dest, new_size_in_bytes));
    }
    return static_cast<T*>(allocator->Allocate(new_size_in_bytes));
}

template <typename T>
T*
allocate_and_copy(
    const T* src, size_t count, Allocator* allocator, T* old_dest = nullptr, size_t old_count = 0) {
    if (src == nullptr || count == 0) {
        return nullptr;
    }
    if (old_dest == nullptr && old_count > 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Old destination cannot be null if old count is greater than zero");
    }
    if (old_dest && old_count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Old count must be greater than zero if old destination is provided");
    }

    T* dest;
    if (allocator != nullptr) {
        dest = allocator_element<T>(allocator, old_dest, (old_count + count) * sizeof(T));
    } else {
        dest = new_element<T>(old_dest, old_count, old_count + count);
    }
    memcpy(dest + old_count, src, count * sizeof(T));
    return dest;
}

void
copy_sparse_vector(const SparseVector& src, SparseVector* dest, Allocator* allocator) {
    size_t len = src.len_;
    if (allocator != nullptr) {
        dest->ids_ = static_cast<uint32_t*>(allocator->Allocate(len * sizeof(uint32_t)));
        dest->vals_ = static_cast<float*>(allocator->Allocate(len * sizeof(float)));
    } else {
        dest->ids_ = new uint32_t[len];
        dest->vals_ = new float[len];
    }
    dest->len_ = len;
    std::memcpy(dest->ids_, src.ids_, len * sizeof(uint32_t));
    std::memcpy(dest->vals_, src.vals_, len * sizeof(float));
}

SparseVector*
allocate_and_copy_sparse_vectors(const SparseVector* src,
                                 size_t count,
                                 Allocator* allocator,
                                 SparseVector* old_dest = nullptr,
                                 size_t old_count = 0) {
    if (src == nullptr || count == 0) {
        return old_dest;
    }

    size_t new_total = old_count + count;
    SparseVector* dest = nullptr;

    if (allocator != nullptr) {
        dest =
            allocator_element<SparseVector>(allocator, old_dest, new_total * sizeof(SparseVector));
    } else {
        dest = new_element<SparseVector>(old_dest, old_count, new_total);
    }

    for (size_t i = old_count; i < new_total; ++i) {
        const SparseVector& src_vec = src[i - old_count];
        copy_sparse_vector(src_vec, &dest[i], allocator);
    }
    return dest;
}

DatasetPtr
DatasetImpl::DeepCopy(Allocator* allocator) const {
    auto* allocator_ref = allocator != nullptr ? allocator : this->allocator_;
    auto copy_dataset = std::make_shared<DatasetImpl>();
    copy_dataset->Owner(true, allocator_ref);

    auto num_elements = this->GetNumElements();
    auto dim = this->GetDim();

    copy_dataset->NumElements(num_elements);
    copy_dataset->Dim(dim);

    copy_dataset->Ids(allocate_and_copy(this->GetIds(), num_elements, allocator_ref));
    copy_dataset->Distances(
        allocate_and_copy(this->GetDistances(), num_elements * dim, allocator_ref));
    copy_dataset->Int8Vectors(
        allocate_and_copy(this->GetInt8Vectors(), num_elements * dim, allocator_ref));
    copy_dataset->Float32Vectors(
        allocate_and_copy(this->GetFloat32Vectors(), num_elements * dim, allocator_ref));

    if (this->GetExtraInfoSize() != 0) {
        copy_dataset->ExtraInfoSize(this->GetExtraInfoSize());
        copy_dataset->ExtraInfos(allocate_and_copy(
            this->GetExtraInfos(), num_elements * this->GetExtraInfoSize(), allocator_ref));
    }
    copy_dataset->SparseVectors(
        allocate_and_copy_sparse_vectors(this->GetSparseVectors(), num_elements, allocator_ref));

    auto* paths = new std::string[num_elements];
    copy_dataset->Paths(paths);
    for (int i = 0; i < num_elements; ++i) {
        paths[i] += this->GetPaths()[i];
    }

    if (this->GetAttributeSets() != nullptr) {
        const auto* attrsets = this->GetAttributeSets();
        auto* attrsets_copy = new AttributeSet[num_elements];
        copy_dataset->AttributeSets(attrsets_copy);

        for (int i = 0; i < num_elements; ++i) {
            attrsets_copy[i].attrs_.reserve(attrsets[i].attrs_.size());
            for (const auto& attr : attrsets[i].attrs_) {
                attrsets_copy[i].attrs_.emplace_back(attr->DeepCopy());
            }
        }
    }

    return copy_dataset;
}

#define APPEND_DATA(KEY, TYPE, SETTER_FUNC, MULTIPLIER)                                          \
    if (auto iter = this->data_.find(KEY); iter != this->data_.end()) {                          \
        if (other->Get##SETTER_FUNC() == nullptr) {                                              \
            throw VsagException(ErrorType::INVALID_ARGUMENT,                                     \
                                "Cannot append dataset without " #KEY " to dataset with " #KEY); \
        }                                                                                        \
        auto ptr = const_cast<TYPE>(std::get<const TYPE>(iter->second));                         \
        this->SETTER_FUNC(allocate_and_copy(other->Get##SETTER_FUNC(),                           \
                                            new_num_elements*(MULTIPLIER),                       \
                                            this->allocator_,                                    \
                                            ptr,                                                 \
                                            old_num_elements*(MULTIPLIER)));                     \
    }

DatasetPtr
DatasetImpl::Append(const DatasetPtr& other) {
    if (!owner_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Cannot append to a non-owner dataset");
    }
    if (this->GetDim() != other->GetDim()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Cannot append datasets with different dimensions");
    }
    if (other->GetExtraInfoSize() != this->GetExtraInfoSize()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Cannot append datasets with different extra info sizes");
    }

    auto old_num_elements = this->GetNumElements();
    auto new_num_elements = other->GetNumElements();
    auto dim = this->GetDim();

    this->NumElements(old_num_elements + new_num_elements);

    APPEND_DATA(IDS, int64_t*, Ids, 1);
    APPEND_DATA(DISTS, float*, Distances, dim);
    APPEND_DATA(INT8_VECTORS, int8_t*, Int8Vectors, dim);
    APPEND_DATA(FLOAT32_VECTORS, float*, Float32Vectors, dim);
    if (this->GetExtraInfoSize() != 0) {
        APPEND_DATA(EXTRA_INFOS, char*, ExtraInfos, this->GetExtraInfoSize());
    }

    // append paths
    if (auto iter = this->data_.find(DATASET_PATHS); iter != this->data_.end()) {
        if (other->GetPaths() == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Cannot append dataset without paths to dataset with paths");
        }
        auto* ptr = const_cast<std::string*>(std::get<const std::string*>(iter->second));
        auto* paths_copy = new std::string[old_num_elements + new_num_elements];
        for (int i = 0; i < old_num_elements; ++i) {
            paths_copy[i] += ptr[i];
        }
        delete[] ptr;  // Free the old memory if it was allocated with new[]
        ptr = nullptr;
        for (int i = 0; i < new_num_elements; ++i) {
            paths_copy[old_num_elements + i] += other->GetPaths()[i];
        }
        this->Paths(paths_copy);
    }

    // append sparse vectors
    if (auto iter = this->data_.find(SPARSE_VECTORS); iter != this->data_.end()) {
        if (other->GetSparseVectors() == nullptr) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "Cannot append dataset without sparse vectors to dataset with sparse vectors");
        }
        auto* ptr = const_cast<SparseVector*>(std::get<const SparseVector*>(iter->second));
        this->SparseVectors(allocate_and_copy_sparse_vectors(
            other->GetSparseVectors(), new_num_elements, this->allocator_, ptr, old_num_elements));
    }

    // append attribute sets
    if (auto iter = this->data_.find(ATTRIBUTE_SETS); iter != this->data_.end()) {
        if (other->GetAttributeSets() == nullptr) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "Cannot append dataset without attribute sets to dataset with attribute sets");
        }
        auto* ptr = const_cast<AttributeSet*>(std::get<const AttributeSet*>(iter->second));
        auto* attrsets_copy = new AttributeSet[new_num_elements + old_num_elements];
        this->AttributeSets(attrsets_copy);
        for (int i = 0; i < old_num_elements; ++i) {
            attrsets_copy[i].attrs_.swap(ptr[i].attrs_);
        }
        delete[] ptr;
        ptr = nullptr;
        const auto* other_attribute_sets = other->GetAttributeSets();
        for (int i = 0; i < new_num_elements; ++i) {
            attrsets_copy[old_num_elements + i].attrs_.reserve(
                other_attribute_sets[i].attrs_.size());
            for (const auto& attr : other_attribute_sets[i].attrs_) {
                attrsets_copy[old_num_elements + i].attrs_.emplace_back(attr->DeepCopy());
            }
        }
    }
    return shared_from_this();
}

std::vector<std::string>
DatasetImpl::GetStatistics(const std::vector<std::string>& stat_keys) const {
    auto json = JsonType::Parse(this->Statistics_);
    std::vector<std::string> result;
    for (const auto& key : stat_keys) {
        if (json.Contains(key)) {
            result.emplace_back(json[key].Dump());
        } else {
            result.emplace_back("");
        }
    }
    return result;
}

};  // namespace vsag
