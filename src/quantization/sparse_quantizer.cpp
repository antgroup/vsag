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

#include "sparse_quantizer.h"

#include <iostream>

#include "simd/sparse_simd.h"
#include "sparse_computer.h"

namespace vsag {

void
SparseQuantizer::ComputeDist(SparseComputer& computer, const uint8_t* codes, float* dists)const{
    *dists = this->ComputeDistImpl(computer.buf_, codes);
}

float
SparseQuantizer::ComputeDist(SparseComputer& computer, const uint8_t* codes)const{
    return this->ComputeDistImpl(computer.buf_, codes);
}

std::shared_ptr<SparseComputer>
SparseQuantizer::FactoryComputer() {
    return std::make_shared<SparseComputer>(static_cast<SparseQuantizer*>(this));
}

void
SparseQuantizer::ReleaseComputer(SparseComputer& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

void
SparseQuantizer::ProcessQuery(const SDataType* query, SparseComputer& computer) const{
    try {
        uint64_t code_size = query->num * (sizeof(uint32_t) + sizeof(float)) + sizeof(uint32_t);
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(code_size));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }
    this->EncodeOne(query, computer.buf_);
}

bool
SparseQuantizer::TrainImpl(const SDataType* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

bool
SparseQuantizer::EncodeOneImpl(uint32_t nnz,
                               const uint32_t* ids,
                               const float* vals,
                               uint8_t* codes) const{
    size_t offset = 0;

    // Write nnz to codes
    std::memcpy(codes + offset, &nnz, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Create a vector of pairs to sort ids and corresponding vals
    std::vector<std::pair<uint32_t, float>> id_val_pairs(nnz);
    for (uint32_t i = 0; i < nnz; ++i) {
        id_val_pairs[i] = {ids[i], vals[i]};
    }

    // Sort the vector by ids
    std::sort(id_val_pairs.begin(),
              id_val_pairs.end(),
              [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                  return a.first < b.first;
              });

    // Write sorted ids and vals to codes
    for (uint32_t i = 0; i < nnz; ++i) {
        std::memcpy(codes + offset, &id_val_pairs[i].first, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(codes + offset, &id_val_pairs[i].second, sizeof(float));
        offset += sizeof(float);
    }
    return true;
}

bool
SparseQuantizer::EncodeOne(const SDataType* data, uint8_t* codes)const {
    uint32_t nnz = data->offsets[1] - data->offsets[0];
    return this->EncodeOneImpl(nnz, data->ids, data->vals, codes);
}

bool
SparseQuantizer::EncodeBatch(const SDataType* data, uint8_t* codes, uint64_t count)const {
    size_t offset = 0;

    for (uint64_t i = 0; i < count; ++i) {
        uint32_t nnz = data->offsets[i + 1] - data->offsets[i];

        if (!EncodeOneImpl(
                nnz, data->ids + data->offsets[i], data->vals + data->offsets[i], codes + offset)) {
            return false;
        }

        offset += sizeof(uint32_t) + nnz * (sizeof(uint32_t) + sizeof(float));
    }
    return true;
}

bool
SparseQuantizer::DecodeOneImpl(uint32_t nnz, uint32_t* ids, float* vals, const uint8_t* codes)const {
    size_t offset = 0;

    offset += sizeof(uint32_t);

    for (uint32_t i = 0; i < nnz; ++i) {
        std::memcpy(&ids[i], codes + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        std::memcpy(&vals[i], codes + offset, sizeof(float));
        offset += sizeof(float);
    }
    return true;
}

bool
SparseQuantizer::DecodeOne(const uint8_t* codes, SDataType* data)const {
    data->num = 1;
    try {
        data->offsets = reinterpret_cast<uint32_t*>(allocator_->Allocate(2 * sizeof(uint32_t)));
    } catch (const std::bad_alloc& e) {
        data->offsets = nullptr;
        logger::error("bad alloc when decode");
        throw std::bad_alloc();
    }
    data->offsets[0] = 0;
    std::memcpy(&data->offsets[1], codes, sizeof(uint32_t));

    try {
        data->ids =
            reinterpret_cast<uint32_t*>(allocator_->Allocate(data->offsets[1] * sizeof(uint32_t)));
    } catch (const std::bad_alloc& e) {
        data->ids = nullptr;
        this->allocator_->Deallocate(data->offsets);
        logger::error("bad alloc when decode");
        throw std::bad_alloc();
    }

    try {
        data->vals =
            reinterpret_cast<float*>(allocator_->Allocate(data->offsets[1] * sizeof(float)));
    } catch (const std::bad_alloc& e) {
        data->vals = nullptr;
        this->allocator_->Deallocate(data->offsets);
        this->allocator_->Deallocate(data->ids);
        logger::error("bad alloc when decode");
        throw std::bad_alloc();
    }

    if (!this->DecodeOneImpl(data->offsets[1], data->ids, data->vals, codes)) {
        this->allocator_->Deallocate(data->offsets);
        this->allocator_->Deallocate(data->ids);
        this->allocator_->Deallocate(data->vals);
        return false;  // Return false if decoding fails
    }
    return true;
}

bool
SparseQuantizer::DecodeBatch(const uint8_t* codes, SDataType* data, uint64_t count) {
    size_t offset_codes = 0;

    try {
        data->offsets =
            reinterpret_cast<uint32_t*>(allocator_->Allocate((count + 1) * sizeof(uint32_t)));
    } catch (const std::bad_alloc& e) {
        data->offsets = nullptr;
        logger::error("bad alloc when decode batch");
        throw std::bad_alloc();
    }

    data->num = count;
    data->offsets[0] = 0;

    // First pass: calculate total nnz to allocate space for ids and vals
    uint32_t total_nnz = 0;
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t nnz;
        std::memcpy(&nnz, codes + offset_codes, sizeof(uint32_t));
        total_nnz += nnz;
        offset_codes += sizeof(uint32_t) + nnz * (sizeof(uint32_t) + sizeof(float));
        data->offsets[i + 1] = data->offsets[i] + nnz;
    }

    try {
        data->ids = reinterpret_cast<uint32_t*>(allocator_->Allocate(total_nnz * sizeof(uint32_t)));
    } catch (const std::bad_alloc& e) {
        data->ids = nullptr;
        this->allocator_->Deallocate(data->offsets);
        logger::error("bad alloc when decode batch");
        throw std::bad_alloc();
    }

    try {
        data->vals = reinterpret_cast<float*>(allocator_->Allocate(total_nnz * sizeof(float)));
    } catch (const std::bad_alloc& e) {
        data->vals = nullptr;
        this->allocator_->Deallocate(data->offsets);
        this->allocator_->Deallocate(data->ids);
        logger::error("bad alloc when decode batch");
        throw std::bad_alloc();
    }

    // Second pass: actually decode the data
    offset_codes = 0;
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t nnz = data->offsets[i + 1] - data->offsets[i];

        if (!DecodeOneImpl(nnz,
                           data->ids + data->offsets[i],
                           data->vals + data->offsets[i],
                           codes + offset_codes)) {
            this->allocator_->Deallocate(data->offsets);
            this->allocator_->Deallocate(data->ids);
            this->allocator_->Deallocate(data->vals);
            return false;  // Return false if decoding fails
        }

        offset_codes += sizeof(uint32_t) + nnz * (sizeof(uint32_t) + sizeof(float));
    }

    return true;
}

float
SparseQuantizer::ComputeDistImpl(const uint8_t* codes1, const uint8_t* codes2) const{
    SDataType* sv1;
    SDataType* sv2;
    this->DecodeOne(codes1, sv1);
    this->DecodeOne(codes2, sv2);

    return 1 - SparseComputeIP(sv1->num, sv1->ids, sv1->vals, sv2->num, sv2->ids, sv2->vals);
}
}  // namespace vsag