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

#include "sparse_dmq_datacell.h"

#include <cstring>
#include <limits>

#include "common.h"

namespace vsag {
namespace {

constexpr uint32_t K_SPARSE_DMQ_DATACELL_MAGIC = 0x53444D51U;
constexpr uint32_t K_SPARSE_DMQ_DATACELL_VERSION = 5;

}  // namespace

SparseDmqDataCell::SparseDmqDataCell(uint32_t term_id_limit, const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      quantizer_(std::make_shared<SparseDmqQuantizer>(term_id_limit, allocator_)),
      offsets_(allocator_),
      codes_(allocator_) {
    offsets_.push_back(0);
}

const uint8_t*
SparseDmqDataCell::GetCode(InnerIdType id) const {
    CHECK_ARGUMENT(id < this->total_count_, "SparseDmqDataCell inner id is out of range");
    return codes_.data() + offsets_[id];
}

void
SparseDmqDataCell::Query(float* result_dists,
                         const ComputerInterfacePtr& computer,
                         const InnerIdType* idx,
                         InnerIdType id_count,
                         QueryContext* ctx) {
    (void)ctx;
    CHECK_ARGUMENT(result_dists != nullptr, "SparseDmqDataCell result buffer is null");
    if (id_count != 0) {
        CHECK_ARGUMENT(idx != nullptr, "SparseDmqDataCell ids are null");
    }
    auto dmq_computer = std::dynamic_pointer_cast<Computer<SparseDmqQuantizer>>(computer);
    CHECK_ARGUMENT(dmq_computer != nullptr, "SparseDmqDataCell computer type mismatch");
    std::shared_lock lock(this->mutex_);
    for (InnerIdType index = 0; index < id_count; ++index) {
        dmq_computer->ComputeDist(GetCode(idx[index]), result_dists + index);
    }
}

ComputerInterfacePtr
SparseDmqDataCell::FactoryComputer(const void* query) {
    CHECK_ARGUMENT(query != nullptr, "SparseDmqDataCell query is null");
    std::shared_lock lock(this->mutex_);
    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(reinterpret_cast<const float*>(query));
    return computer;
}

void
SparseDmqDataCell::Train(const void* data, uint64_t count) {
    CHECK_ARGUMENT(data != nullptr, "SparseDmqDataCell training data is null");
    CHECK_ARGUMENT(count > 0, "SparseDmqDataCell training count must be positive");
    std::unique_lock lock(this->mutex_);
    quantizer_->TrainImpl(reinterpret_cast<const float*>(data), count);
}

void
SparseDmqDataCell::InsertVector(const void* vector, InnerIdType idx) {
    CHECK_ARGUMENT(vector != nullptr, "SparseDmqDataCell insertion vector is null");
    if (idx == std::numeric_limits<InnerIdType>::max()) {
        idx = this->TotalCount();
    }
    BatchInsertVector(vector, 1, &idx);
}

void
SparseDmqDataCell::BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) {
    CHECK_ARGUMENT(vectors != nullptr, "SparseDmqDataCell insertion vectors are null");
    CHECK_ARGUMENT(count > 0, "SparseDmqDataCell insertion count must be positive");
    const auto* sparse_vectors = static_cast<const SparseVector*>(vectors);
    std::unique_lock lock(this->mutex_);
    if (this->total_count_ != 0) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "SparseDmqDataCell does not support incremental insertion");
    }
    quantizer_->TrainImpl(reinterpret_cast<const float*>(vectors), count);
    offsets_.reserve(offsets_.size() + count);
    for (InnerIdType index = 0; index < count; ++index) {
        const InnerIdType expected_id = this->total_count_ + index;
        if (idx_vec != nullptr) {
            CHECK_ARGUMENT(idx_vec[index] == expected_id,
                           "SparseDmqDataCell only supports sequential insertion");
        }
        const uint64_t code_size = quantizer_->GetEncodedSize(sparse_vectors[index]);
        const uint64_t offset = codes_.size();
        codes_.resize(offset + code_size);
        quantizer_->EncodeOneImpl(reinterpret_cast<const float*>(sparse_vectors + index),
                                  codes_.data() + offset);
        offsets_.push_back(codes_.size());
    }
    this->total_count_ += count;
    this->max_capacity_ = std::max(this->max_capacity_, this->total_count_);
}

float
SparseDmqDataCell::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    std::shared_lock lock(this->mutex_);
    return quantizer_->Compute(GetCode(id1), GetCode(id2));
}

void
SparseDmqDataCell::GetSparseVectorByInnerId(InnerIdType inner_id,
                                            SparseVector* data,
                                            Allocator* specified_allocator) const {
    CHECK_ARGUMENT(data != nullptr, "SparseDmqDataCell output vector is null");
    std::shared_lock lock(this->mutex_);
    SparseVector decoded;
    quantizer_->DecodeOneImpl(GetCode(inner_id), reinterpret_cast<float*>(&decoded));
    Allocator* target = specified_allocator == nullptr ? allocator_ : specified_allocator;
    if (target == allocator_) {
        *data = decoded;
        return;
    }
    data->len_ = decoded.len_;
    data->ids_ = static_cast<uint32_t*>(target->Allocate(sizeof(uint32_t) * decoded.len_));
    data->vals_ = static_cast<float*>(target->Allocate(sizeof(float) * decoded.len_));
    std::memcpy(data->ids_, decoded.ids_, sizeof(uint32_t) * decoded.len_);
    std::memcpy(data->vals_, decoded.vals_, sizeof(float) * decoded.len_);
    allocator_->Deallocate(decoded.ids_);
    allocator_->Deallocate(decoded.vals_);
}

void
SparseDmqDataCell::Prefetch(InnerIdType id) {
    if (id < this->total_count_) {
        __builtin_prefetch(codes_.data() + offsets_[id], 0, 1);
    }
}

std::string
SparseDmqDataCell::GetQuantizerName() {
    return quantizer_->Name();
}

MetricType
SparseDmqDataCell::GetMetricType() {
    return quantizer_->Metric();
}

void
SparseDmqDataCell::Resize(InnerIdType capacity) {
    std::unique_lock lock(this->mutex_);
    offsets_.reserve(static_cast<uint64_t>(capacity) + 1);
    this->max_capacity_ = std::max(this->max_capacity_, capacity);
}

void
SparseDmqDataCell::ExportModel(const FlattenInterfacePtr& other) const {
    auto target = std::dynamic_pointer_cast<SparseDmqDataCell>(other);
    CHECK_ARGUMENT(target != nullptr, "SparseDmqDataCell export target type mismatch");
    if (target.get() == this) {
        return;
    }
    std::scoped_lock lock(this->mutex_, target->mutex_);
    target->quantizer_->ExportModel(*quantizer_);
}

bool
SparseDmqDataCell::Decode(const uint8_t* codes, float* vector) {
    return quantizer_->DecodeOne(codes, vector);
}

bool
SparseDmqDataCell::Encode(const float* vector, uint8_t* codes) {
    return quantizer_->EncodeOne(vector, codes);
}

const uint8_t*
SparseDmqDataCell::GetCodesById(InnerIdType id, bool& need_release) const {
    std::shared_lock lock(this->mutex_);
    need_release = false;
    return GetCode(id);
}

void
SparseDmqDataCell::Release(const uint8_t* data) const {
    (void)data;
}

bool
SparseDmqDataCell::GetCodesById(InnerIdType id, uint8_t* codes) const {
    CHECK_ARGUMENT(codes != nullptr, "SparseDmqDataCell output codes are null");
    std::shared_lock lock(this->mutex_);
    CHECK_ARGUMENT(id < this->total_count_, "SparseDmqDataCell inner id is out of range");
    std::memcpy(codes, GetCode(id), offsets_[id + 1] - offsets_[id]);
    return true;
}

void
SparseDmqDataCell::Serialize(StreamWriter& writer) {
    std::shared_lock lock(this->mutex_);
    StreamWriter::WriteObj(writer, K_SPARSE_DMQ_DATACELL_MAGIC);
    StreamWriter::WriteObj(writer, K_SPARSE_DMQ_DATACELL_VERSION);
    StreamWriter::WriteObj(writer, this->total_count_);
    StreamWriter::WriteVector(writer, offsets_);
    StreamWriter::WriteVector(writer, codes_);
    quantizer_->Serialize(writer);
}

void
SparseDmqDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    std::unique_lock lock(this->mutex_);
    uint32_t magic = 0;
    uint32_t version = 0;
    StreamReader::ReadObj(reader, magic);
    StreamReader::ReadObj(reader, version);
    CHECK_ARGUMENT(magic == K_SPARSE_DMQ_DATACELL_MAGIC,
                   "serialized DMQ datacell has invalid magic");
    CHECK_ARGUMENT(version == K_SPARSE_DMQ_DATACELL_VERSION,
                   fmt::format("unsupported sparse DMQ datacell version {}", version));
    StreamReader::ReadObj(reader, this->total_count_);
    StreamReader::ReadVector(reader, offsets_);
    StreamReader::ReadVector(reader, codes_);
    quantizer_->Deserialize(reader);
    CHECK_ARGUMENT(offsets_.size() == static_cast<uint64_t>(this->total_count_) + 1,
                   "serialized DMQ offset count is inconsistent");
    CHECK_ARGUMENT(not offsets_.empty(), "serialized DMQ offsets are empty");
    CHECK_ARGUMENT(offsets_.front() == 0, "serialized DMQ offsets must start at zero");
    CHECK_ARGUMENT(offsets_.back() == codes_.size(),
                   "serialized DMQ offsets do not match code size");
    for (InnerIdType id = 0; id < this->total_count_; ++id) {
        CHECK_ARGUMENT(offsets_[id] <= offsets_[id + 1], "serialized DMQ offsets are not ordered");
        CHECK_ARGUMENT(offsets_[id + 1] - offsets_[id] >= sizeof(SparseDmqQuantizer::EncodedHeader),
                       "serialized DMQ code is smaller than its header");
        SparseVector vector;
        vector.len_ = quantizer_->GetEncodedLength(codes_.data() + offsets_[id]);
        CHECK_ARGUMENT(offsets_[id + 1] - offsets_[id] == quantizer_->GetEncodedSize(vector),
                       "serialized DMQ code size is inconsistent");
    }
    this->max_capacity_ = this->total_count_;
}

uint64_t
SparseDmqDataCell::GetMemoryUsage() const {
    std::shared_lock lock(this->mutex_);
    return sizeof(*this) + offsets_.capacity() * sizeof(uint64_t) +
           codes_.capacity() * sizeof(uint8_t) + quantizer_->GetMemoryUsage();
}

}  // namespace vsag
