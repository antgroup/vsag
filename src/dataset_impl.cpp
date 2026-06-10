
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
#include <unordered_set>
#include <vector>

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
static inline T*
new_element(T*& old_dest, uint64_t old_count, uint64_t new_total) {
    T* dest = new T[new_total];
    if (old_dest != nullptr) {
        memcpy(dest, old_dest, old_count * sizeof(T));
    }
    delete[] old_dest;  // Free the old memory if it was allocated with new[]
    old_dest = nullptr;
    return dest;
}

template <typename T>
static inline T*
allocator_element(Allocator* allocator, T* old_dest, uint64_t new_size_in_bytes) {
    if (old_dest != nullptr) {
        return static_cast<T*>(allocator->Reallocate(old_dest, new_size_in_bytes));
    }
    return static_cast<T*>(allocator->Allocate(new_size_in_bytes));
}

template <typename T>
static inline T*
allocate_and_copy(const T* src,
                  uint64_t count,
                  Allocator* allocator,
                  T* old_dest = nullptr,
                  uint64_t old_count = 0) {
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

static void
copy_sparse_vector(const SparseVector& src, SparseVector* dest, Allocator* allocator) {
    uint64_t len = src.len_;
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

    // Copy optional original token sequence (preserves order & duplicates).
    dest->token_seq_len_ = src.token_seq_len_;
    dest->token_sequence_ = nullptr;
    if (src.token_seq_len_ > 0 && src.token_sequence_ != nullptr) {
        uint64_t token_len = src.token_seq_len_;
        if (allocator != nullptr) {
            dest->token_sequence_ =
                static_cast<uint32_t*>(allocator->Allocate(token_len * sizeof(uint32_t)));
        } else {
            dest->token_sequence_ = new uint32_t[token_len];
        }
        std::memcpy(dest->token_sequence_, src.token_sequence_, token_len * sizeof(uint32_t));
    }
}

static void
copy_multi_vector(const MultiVector& src,
                  MultiVector* dest,
                  int64_t multi_vector_dim,
                  Allocator* allocator) {
    dest->len_ = src.len_;
    if (src.len_ == 0) {
        dest->vectors_ = nullptr;
        return;
    }
    uint64_t num_floats = static_cast<uint64_t>(src.len_) * multi_vector_dim;
    if (allocator != nullptr) {
        dest->vectors_ = static_cast<float*>(allocator->Allocate(num_floats * sizeof(float)));
    } else {
        dest->vectors_ = new float[num_floats];
    }
    std::memcpy(dest->vectors_, src.vectors_, num_floats * sizeof(float));
}

static MultiVector*
allocate_and_copy_multi_vectors(const MultiVector* src,
                                uint64_t count,
                                int64_t multi_vector_dim,
                                Allocator* allocator,
                                MultiVector* old_dest = nullptr,
                                uint64_t old_count = 0) {
    if (src == nullptr || count == 0) {
        return old_dest;
    }

    uint64_t new_total = old_count + count;
    MultiVector* dest = nullptr;

    if (allocator != nullptr) {
        dest = allocator_element<MultiVector>(allocator, old_dest, new_total * sizeof(MultiVector));
    } else {
        dest = new_element<MultiVector>(old_dest, old_count, new_total);
    }

    for (uint64_t i = old_count; i < new_total; ++i) {
        const MultiVector& src_vec = src[i - old_count];
        copy_multi_vector(src_vec, &dest[i], multi_vector_dim, allocator);
    }
    return dest;
}

static std::string*
allocate_and_copy_paths(const std::string* src, uint64_t count) {
    if (src == nullptr) {
        return nullptr;
    }

    auto* dest = new std::string[count];
    for (uint64_t i = 0; i < count; ++i) {
        dest[i] = src[i];
    }
    return dest;
}

static SparseVector*
allocate_and_copy_sparse_vectors(const SparseVector* src,
                                 uint64_t count,
                                 Allocator* allocator,
                                 SparseVector* old_dest = nullptr,
                                 uint64_t old_count = 0) {
    if (src == nullptr || count == 0) {
        return old_dest;
    }

    uint64_t new_total = old_count + count;
    SparseVector* dest = nullptr;

    if (allocator != nullptr) {
        dest =
            allocator_element<SparseVector>(allocator, old_dest, new_total * sizeof(SparseVector));
    } else {
        dest = new_element<SparseVector>(old_dest, old_count, new_total);
    }

    for (uint64_t i = old_count; i < new_total; ++i) {
        const SparseVector& src_vec = src[i - old_count];
        copy_sparse_vector(src_vec, &dest[i], allocator);
    }
    return dest;
}

template <typename T>
static inline void*
void_ptr(const T* ptr) {
    return static_cast<void*>(const_cast<T*>(ptr));
}

template <typename T>
static inline void*
void_ptr(T* ptr) {
    return static_cast<void*>(ptr);
}

DatasetImpl::~DatasetImpl() {  // NOLINT
    if (not this->owner_) {
        return;
    }

    if (allocator_ != nullptr) {
        allocator_->Deallocate(void_ptr(DatasetImpl::GetIds()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetDistances()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetInt8Vectors()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetFloat16Vectors()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetFloat32Vectors()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetExtraInfos()));
        allocator_->Deallocate(void_ptr(DatasetImpl::GetVectorCounts()));
        const auto* sparse_vectors = DatasetImpl::GetSparseVectors();
        if (sparse_vectors != nullptr) {
            for (int i = 0; i < DatasetImpl::GetNumElements(); i++) {
                if (sparse_vectors[i].ids_ != nullptr) {
                    allocator_->Deallocate(void_ptr(sparse_vectors[i].ids_));
                }
                if (sparse_vectors[i].vals_ != nullptr) {
                    allocator_->Deallocate(void_ptr(sparse_vectors[i].vals_));
                }
                if (sparse_vectors[i].token_sequence_ != nullptr) {
                    allocator_->Deallocate(void_ptr(sparse_vectors[i].token_sequence_));
                }
            }
            allocator_->Deallocate(void_ptr(DatasetImpl::GetSparseVectors()));
        }
        const MultiVector* multi_vectors = DatasetImpl::GetMultiVectors();
        if (multi_vectors != nullptr) {
            for (int64_t i = 0; i < DatasetImpl::GetNumElements(); i++) {
                if (multi_vectors[i].vectors_ != nullptr) {
                    allocator_->Deallocate(void_ptr(multi_vectors[i].vectors_));
                }
            }
            allocator_->Deallocate(void_ptr(DatasetImpl::GetMultiVectors()));
        }

    } else {
        delete[] DatasetImpl::GetIds();
        delete[] DatasetImpl::GetDistances();
        delete[] DatasetImpl::GetInt8Vectors();
        delete[] DatasetImpl::GetFloat16Vectors();
        delete[] DatasetImpl::GetFloat32Vectors();
        delete[] DatasetImpl::GetExtraInfos();
        delete[] DatasetImpl::GetVectorCounts();

        if (DatasetImpl::GetSparseVectors() != nullptr) {
            for (int i = 0; i < DatasetImpl::GetNumElements(); i++) {
                delete[] DatasetImpl::GetSparseVectors()[i].ids_;
                delete[] DatasetImpl::GetSparseVectors()[i].vals_;
                delete[] DatasetImpl::GetSparseVectors()[i].token_sequence_;
            }
            delete[] DatasetImpl::GetSparseVectors();
        }

        if (DatasetImpl::GetMultiVectors() != nullptr) {
            for (int64_t i = 0; i < DatasetImpl::GetNumElements(); i++) {
                delete[] DatasetImpl::GetMultiVectors()[i].vectors_;
            }
            delete[] DatasetImpl::GetMultiVectors();
        }
    }
    std::unordered_set<const std::string*> released_paths;
    auto release_paths = [&released_paths](const std::string* paths) {
        if (paths != nullptr && released_paths.insert(paths).second) {
            delete[] paths;
        }
    };
    release_paths(DatasetImpl::GetPaths());
    for (const auto& [key, value] : this->data_) {
        if (IsHierarchyPathsKey(key)) {
            release_paths(std::get<const std::string*>(value));
        }
    }
    if (DatasetImpl::GetAttributeSets() != nullptr) {
        const auto* attrsets = DatasetImpl::GetAttributeSets();
        for (int i = 0; i < DatasetImpl::GetNumElements(); ++i) {
            for (auto* attr : attrsets[i].attrs_) {
                delete attr;
            }
        }
        delete[] attrsets;
    }
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
    if (this->GetIds() != nullptr) {
        copy_dataset->Ids(allocate_and_copy(this->GetIds(), num_elements, allocator_ref));
    }
    if (this->GetDistances() != nullptr) {
        copy_dataset->Distances(
            allocate_and_copy(this->GetDistances(), num_elements * dim, allocator_ref));
    }
    if (this->GetInt8Vectors() != nullptr) {
        copy_dataset->Int8Vectors(
            allocate_and_copy(this->GetInt8Vectors(), num_elements * dim, allocator_ref));
    }
    if (this->GetFloat32Vectors() != nullptr) {
        copy_dataset->Float32Vectors(
            allocate_and_copy(this->GetFloat32Vectors(), num_elements * dim, allocator_ref));
    }
    if (this->GetFloat16Vectors() != nullptr) {
        copy_dataset->Float16Vectors(
            allocate_and_copy(this->GetFloat16Vectors(), num_elements * dim, allocator_ref));
    }

    if (this->GetVectorCounts() != nullptr) {
        copy_dataset->VectorCounts(
            allocate_and_copy(this->GetVectorCounts(), num_elements, allocator_ref));
    }

    if (this->GetExtraInfoSize() != 0) {
        copy_dataset->ExtraInfoSize(this->GetExtraInfoSize());
        copy_dataset->ExtraInfos(allocate_and_copy(
            this->GetExtraInfos(), num_elements * this->GetExtraInfoSize(), allocator_ref));
    }
    if (this->GetSparseVectors() != nullptr) {
        copy_dataset->SparseVectors(allocate_and_copy_sparse_vectors(
            this->GetSparseVectors(), num_elements, allocator_ref));
    }
    if (this->GetMultiVectors() != nullptr) {
        int64_t mv_dim = this->GetMultiVectorDim();
        copy_dataset->MultiVectorDim(mv_dim);
        copy_dataset->MultiVectors(allocate_and_copy_multi_vectors(
            this->GetMultiVectors(), num_elements, mv_dim, allocator_ref));
    }
    if (this->GetPaths() != nullptr) {
        copy_dataset->Paths(
            allocate_and_copy_paths(this->GetPaths(), static_cast<uint64_t>(num_elements)));
    }
    for (const auto& [key, value] : this->data_) {
        if (IsHierarchyPathsKey(key)) {
            copy_dataset->Paths(HierarchyNameFromPathsKey(key),
                                allocate_and_copy_paths(std::get<const std::string*>(value),
                                                        static_cast<uint64_t>(num_elements)));
        }
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

    // check paths
    if (this->data_.find(DATASET_PATHS) != this->data_.end() && other->GetPaths() == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Cannot append dataset without paths to dataset with paths");
    }
    std::vector<std::string> hierarchy_path_keys;
    for (const auto& [key, value] : this->data_) {
        if (not IsHierarchyPathsKey(key)) {
            continue;
        }
        const auto* paths = std::get<const std::string*>(value);
        if (paths == nullptr) {
            continue;
        }
        const auto hierarchy_name = HierarchyNameFromPathsKey(key);
        if (other->GetPaths(hierarchy_name) == nullptr) {
            std::string error_message = "Cannot append dataset without paths for hierarchy ";
            error_message.append(hierarchy_name)
                .append(" to dataset with paths for hierarchy ")
                .append(hierarchy_name);
            throw VsagException(ErrorType::INVALID_ARGUMENT, error_message);
        }
        hierarchy_path_keys.push_back(key);
    }
    auto other_impl = std::dynamic_pointer_cast<DatasetImpl>(other);
    if (other_impl != nullptr) {
        for (const auto& [key, value] : other_impl->data_) {
            if (not IsHierarchyPathsKey(key)) {
                continue;
            }
            if (std::get<const std::string*>(value) == nullptr) {
                continue;
            }
            const auto hierarchy_name = HierarchyNameFromPathsKey(key);
            if (this->GetPaths(hierarchy_name) == nullptr) {
                std::string error_message = "Cannot append dataset with paths for hierarchy ";
                error_message.append(hierarchy_name)
                    .append(" to dataset without paths for hierarchy ")
                    .append(hierarchy_name);
                throw VsagException(ErrorType::INVALID_ARGUMENT, error_message);
            }
        }
    }

    // check sparse-vectors
    if (this->data_.find(SPARSE_VECTORS) != this->data_.end() &&
        other->GetSparseVectors() == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "Cannot append dataset without sparse vectors to dataset with sparse vectors");
    }

    // check multi-vectors
    if (this->data_.find(MULTI_VECTORS) != this->data_.end()) {
        if (other->GetMultiVectors() == nullptr) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "Cannot append dataset without multi vectors to dataset with multi vectors");
        }
        int64_t mv_dim = this->GetMultiVectorDim();
        if (mv_dim <= 0 || other->GetMultiVectorDim() != mv_dim) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Cannot append datasets with different multi vector dimensions");
        }
    }

    // check attribute-sets
    if (this->data_.find(ATTRIBUTE_SETS) != this->data_.end() &&
        other->GetAttributeSets() == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "Cannot append dataset without attribute sets to dataset with attribute sets");
    }

    // all validation passed; safe to mutate state (destructor relies on NumElements for cleanup)
    this->NumElements(old_num_elements + new_num_elements);

    // append contiguous arrays via realloc-and-copy
    APPEND_DATA(IDS, int64_t*, Ids, 1);
    APPEND_DATA(DISTS, float*, Distances, dim);
    APPEND_DATA(INT8_VECTORS, int8_t*, Int8Vectors, dim);
    APPEND_DATA(FLOAT16_VECTORS, uint16_t*, Float16Vectors, dim);
    APPEND_DATA(FLOAT32_VECTORS, float*, Float32Vectors, dim);
    APPEND_DATA(VECTOR_COUNTS, uint32_t*, VectorCounts, 1);
    if (this->GetExtraInfoSize() != 0) {
        APPEND_DATA(EXTRA_INFOS, char*, ExtraInfos, this->GetExtraInfoSize());
    }

    // append paths
    std::unordered_set<const std::string*> replaced_paths;
    auto append_paths = [&](const std::string* current_paths, const std::string* other_paths) {
        auto* paths_copy = new std::string[old_num_elements + new_num_elements];
        for (int64_t i = 0; i < old_num_elements; ++i) {
            paths_copy[i] = current_paths[i];
        }
        for (int64_t i = 0; i < new_num_elements; ++i) {
            paths_copy[old_num_elements + i] = other_paths[i];
        }
        replaced_paths.insert(current_paths);
        return paths_copy;
    };
    if (auto iter = this->data_.find(DATASET_PATHS); iter != this->data_.end()) {
        this->Paths(append_paths(std::get<const std::string*>(iter->second), other->GetPaths()));
    }
    for (const auto& key : hierarchy_path_keys) {
        auto iter = this->data_.find(key);
        if (iter == this->data_.end()) {
            continue;
        }
        auto hierarchy_name = HierarchyNameFromPathsKey(key);
        this->Paths(hierarchy_name,
                    append_paths(std::get<const std::string*>(iter->second),
                                 other->GetPaths(hierarchy_name)));
    }
    for (const auto* paths : replaced_paths) {
        delete[] paths;
    }

    // append sparse-vectors
    if (auto iter = this->data_.find(SPARSE_VECTORS); iter != this->data_.end()) {
        auto* ptr = const_cast<SparseVector*>(std::get<const SparseVector*>(iter->second));
        this->SparseVectors(allocate_and_copy_sparse_vectors(
            other->GetSparseVectors(), new_num_elements, this->allocator_, ptr, old_num_elements));
    }

    // append multi-vectors
    if (auto iter = this->data_.find(MULTI_VECTORS); iter != this->data_.end()) {
        int64_t mv_dim = this->GetMultiVectorDim();
        auto* ptr = const_cast<MultiVector*>(std::get<const MultiVector*>(iter->second));
        this->MultiVectors(allocate_and_copy_multi_vectors(other->GetMultiVectors(),
                                                           new_num_elements,
                                                           mv_dim,
                                                           this->allocator_,
                                                           ptr,
                                                           old_num_elements));
    }

    // append attribute-sets
    if (auto iter = this->data_.find(ATTRIBUTE_SETS); iter != this->data_.end()) {
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
