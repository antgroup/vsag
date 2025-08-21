
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
T*
AllocateAndCopy(
    const T* src, size_t count, Allocator* allocator, T* old_dest = nullptr, size_t old_count = 0) {
    if (!src || count == 0) {
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
    if (allocator) {
        if (old_dest) {
            // If old_dest is provided, we need to reallocate new memory
            dest =
                static_cast<T*>(allocator->Reallocate(old_dest, (old_count + count) * sizeof(T)));
        } else {
            // Allocate new memory for the new count
            dest = static_cast<T*>(allocator->Allocate(count * sizeof(T)));
        }
    } else {
        if (old_dest) {
            // If old_dest is provided, we need to allocate new memory
            dest = new T[old_count + count];
            memcpy(dest, old_dest, old_count * sizeof(T));
            delete[] old_dest;  // Free the old memory if it was allocated with new[]
        } else {
            dest = new T[count];
        }
    }
    memcpy(dest + old_count, src, count * sizeof(T));
    return dest;
}

void
copy_sparse_vector(const SparseVector& src, SparseVector* dest, Allocator* allocator) {
    size_t len = src.len_;
    if (allocator) {
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
AllocateAndCopySparseVectors(const SparseVector* src,
                             size_t count,
                             Allocator* allocator,
                             SparseVector* old_dest = nullptr,
                             size_t old_count = 0) {
    if (!src || count == 0)
        return old_dest;

    size_t new_total = old_count + count;
    SparseVector* dest = nullptr;

    if (allocator) {
        if (old_dest) {
            dest = static_cast<SparseVector*>(
                allocator->Reallocate(old_dest, new_total * sizeof(SparseVector)));
        } else {
            dest =
                static_cast<SparseVector*>(allocator->Allocate(new_total * sizeof(SparseVector)));
        }
    } else {
        dest = new SparseVector[new_total];
        for (size_t i = 0; i < old_count; ++i) {
            dest[i] = old_dest[i];
        }
        delete[] old_dest;
    }

    for (size_t i = old_count; i < new_total; ++i) {
        const SparseVector& src_vec = src[i - old_count];
        copy_sparse_vector(src_vec, &dest[i], allocator);
    }
    return dest;
}

DatasetPtr
DatasetImpl::DeepCopy(Allocator* allocator) const {
    auto allocator_ref = allocator != nullptr ? allocator : this->allocator_;
    auto copy_dataset = std::make_shared<DatasetImpl>();
    copy_dataset->Owner(true, allocator_ref);

    auto num_elements = this->GetNumElements();
    auto dim = this->GetDim();

    copy_dataset->NumElements(num_elements);
    copy_dataset->Dim(dim);

    copy_dataset->Ids(AllocateAndCopy(this->GetIds(), num_elements, allocator_ref));
    copy_dataset->Distances(
        AllocateAndCopy(this->GetDistances(), num_elements * dim, allocator_ref));
    copy_dataset->Int8Vectors(
        AllocateAndCopy(this->GetInt8Vectors(), num_elements * dim, allocator_ref));
    copy_dataset->Float32Vectors(
        AllocateAndCopy(this->GetFloat32Vectors(), num_elements * dim, allocator_ref));
    copy_dataset->SparseVectors(
        AllocateAndCopySparseVectors(this->GetSparseVectors(), num_elements, allocator_ref));

    auto paths = new std::string[num_elements];
    copy_dataset->Paths(paths);
    for (int i = 0; i < num_elements; ++i) {
        paths[i] = this->GetPaths()[i];
    }

    if (this->GetAttributeSets() != nullptr) {
        const auto* attrsets = this->GetAttributeSets();
        AttributeSet* attrsets_copy = new AttributeSet[num_elements];
        copy_dataset->AttributeSets(attrsets_copy);

        for (int i = 0; i < num_elements; ++i) {
            attrsets_copy[i].attrs_.reserve(attrsets[i].attrs_.size());
            for (const auto& attr : attrsets[i].attrs_) {
                attrsets_copy[i].attrs_.emplace_back(attr->DeepCopy());
            }
        }
    }

    if (this->GetExtraInfoSize() != 0) {
        if (this->GetExtraInfos() == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Extra info size is set but extra infos is null");
        }
        auto extra_info_size = this->GetExtraInfoSize();
        copy_dataset->ExtraInfoSize(extra_info_size);
        char* extra_info = nullptr;
        if (allocator_ref) {
            extra_info =
                static_cast<char*>(allocator_ref->Allocate(extra_info_size * num_elements));
        } else {
            extra_info = new char[extra_info_size * num_elements];
        }
        copy_dataset->ExtraInfos(extra_info);
        std::memcpy(extra_info, this->GetExtraInfos(), extra_info_size * num_elements);
    }

    return copy_dataset;
}

#define APPEND_DATA(KEY, TYPE, SETTER_FUNC, MULTIPLIER)                                          \
    if (auto iter = this->data_.find(KEY); iter != this->data_.end()) {                          \
        if (other->Get##SETTER_FUNC() == nullptr) {                                              \
            throw VsagException(ErrorType::INVALID_ARGUMENT,                                     \
                                "Cannot append dataset without " #KEY " to dataset with " #KEY); \
        }                                                                                        \
        auto ptr = const_cast<TYPE*>(std::get<const TYPE*>(iter->second));                       \
        this->SETTER_FUNC(AllocateAndCopy(other->Get##SETTER_FUNC(),                             \
                                          new_num_elements*(MULTIPLIER),                         \
                                          this->allocator_,                                      \
                                          ptr,                                                   \
                                          old_num_elements*(MULTIPLIER)));                       \
    }

DatasetPtr
DatasetImpl::Append(const DatasetPtr& other) {
    if (owner_ == false) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Cannot append to a non-owner dataset");
    }
    if (this->GetDim() != other->GetDim()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Cannot append datasets with different dimensions");
    }

    auto old_num_elements = this->GetNumElements();
    auto new_num_elements = other->GetNumElements();
    auto dim = this->GetDim();

    this->NumElements(old_num_elements + new_num_elements);

    // append ids
    APPEND_DATA(IDS, int64_t, Ids, 1);

    // append distances
    APPEND_DATA(DISTS, float, Distances, dim);

    // append int8 vectors
    APPEND_DATA(INT8_VECTORS, int8_t, Int8Vectors, dim);

    // append float32 vectors
    APPEND_DATA(FLOAT32_VECTORS, float, Float32Vectors, dim);

    // append paths
    if (auto iter = this->data_.find(DATASET_PATHS); iter != this->data_.end()) {
        if (other->GetPaths() == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Cannot append dataset without paths to dataset with paths");
        }
        auto ptr = const_cast<std::string*>(std::get<const std::string*>(iter->second));
        std::string* paths_copy = new std::string[old_num_elements + new_num_elements];
        for (int i = 0; i < old_num_elements; ++i) {
            paths_copy[i] = ptr[i];
        }
        delete[] ptr;  // Free the old memory if it was allocated with new[]
        for (int i = 0; i < new_num_elements; ++i) {
            paths_copy[old_num_elements + i] = other->GetPaths()[i];
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
        auto ptr = const_cast<SparseVector*>(std::get<const SparseVector*>(iter->second));
        this->SparseVectors(AllocateAndCopySparseVectors(
            other->GetSparseVectors(), new_num_elements, this->allocator_, ptr, old_num_elements));
    }

    // append attribute sets
    if (auto iter = this->data_.find(ATTRIBUTE_SETS); iter != this->data_.end()) {
        if (other->GetAttributeSets() == nullptr) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "Cannot append dataset without attribute sets to dataset with attribute sets");
        }
        auto ptr = const_cast<AttributeSet*>(std::get<const AttributeSet*>(iter->second));
        AttributeSet* attrsets_copy = new AttributeSet[new_num_elements + old_num_elements];
        this->AttributeSets(attrsets_copy);
        for (int i = 0; i < old_num_elements; ++i) {
            attrsets_copy[i].attrs_.swap(ptr[i].attrs_);
        }
        delete[] ptr;
        auto other_attribute_sets = other->GetAttributeSets();
        for (int i = 0; i < new_num_elements; ++i) {
            attrsets_copy[old_num_elements + i].attrs_.reserve(
                other_attribute_sets[i].attrs_.size());
            for (const auto& attr : other_attribute_sets[i].attrs_) {
                attrsets_copy[old_num_elements + i].attrs_.emplace_back(attr->DeepCopy());
            }
        }
    }

    // append extra infos
    if (auto iter = this->data_.find(EXTRA_INFOS); iter != this->data_.end()) {
        if (other->GetExtraInfoSize() != this->GetExtraInfoSize()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Cannot append datasets with different extra info sizes");
        }
        if (other->GetExtraInfos() == nullptr) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "Cannot append dataset without extra infos to dataset with extra infos");
        }
        auto ptr = const_cast<char*>(
            reinterpret_cast<const char*>(std::get<const int64_t*>(iter->second)));
        size_t extra_info_size = this->GetExtraInfoSize();
        if (allocator_) {
            ptr = static_cast<char*>(allocator_->Reallocate(
                ptr, extra_info_size * (old_num_elements + new_num_elements)));
            std::memcpy(ptr + extra_info_size * old_num_elements,
                        other->GetExtraInfos(),
                        extra_info_size * new_num_elements);
            this->ExtraInfos(ptr);
        } else {
            char* extra_info = new char[extra_info_size * (old_num_elements + new_num_elements)];
            std::memcpy(extra_info, ptr, extra_info_size * old_num_elements);
            delete[] ptr;  // Free the old memory if it was allocated with new[]
            std::memcpy(extra_info + extra_info_size * old_num_elements,
                        other->GetExtraInfos(),
                        extra_info_size * new_num_elements);
            this->ExtraInfos(extra_info);
        }
    }

    return shared_from_this();
}

};  // namespace vsag
