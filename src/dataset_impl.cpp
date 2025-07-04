
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

DatasetImpl::~DatasetImpl() noexcept {
    if (not owner_) {
        return;
    }

    if (allocator_ != nullptr) {
        allocator_->Deallocate((void*)(DatasetImpl::GetIds()));
        allocator_->Deallocate((void*)(DatasetImpl::GetDistances()));
        allocator_->Deallocate((void*)(DatasetImpl::GetInt8Vectors()));
        allocator_->Deallocate((void*)(DatasetImpl::GetFloat32Vectors()));
        allocator_->Deallocate((void*)(DatasetImpl::GetPaths()));
        allocator_->Deallocate((void*)(DatasetImpl::GetExtraInfos()));

        if (DatasetImpl::GetSparseVectors() != nullptr) {
            for (int i = 0; i < DatasetImpl::GetNumElements(); i++) {
                allocator_->Deallocate((void*)DatasetImpl::GetSparseVectors()[i].ids_);
                allocator_->Deallocate((void*)DatasetImpl::GetSparseVectors()[i].vals_);
            }
            allocator_->Deallocate((void*)DatasetImpl::GetSparseVectors());
        }

    } else {
        delete[] DatasetImpl::GetIds();
        delete[] DatasetImpl::GetDistances();
        delete[] DatasetImpl::GetInt8Vectors();
        delete[] DatasetImpl::GetFloat32Vectors();
        delete[] DatasetImpl::GetPaths();
        delete[] DatasetImpl::GetExtraInfos();

        if (DatasetImpl::GetSparseVectors() != nullptr) {
            for (int i = 0; i < DatasetImpl::GetNumElements(); i++) {
                delete[] DatasetImpl::GetSparseVectors()[i].ids_;
                delete[] DatasetImpl::GetSparseVectors()[i].vals_;
            }
            delete[] DatasetImpl::GetSparseVectors();
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

};  // namespace vsag
