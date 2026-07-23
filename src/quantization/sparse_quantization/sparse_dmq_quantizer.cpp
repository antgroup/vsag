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

#include "sparse_dmq_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <tuple>
#include <utility>

#include "common.h"

namespace vsag {
namespace {

constexpr float K_MIN_DENOMINATOR = 1e-12F;
constexpr uint64_t K_MAX_LOOKUP_VALUES = 1'000'000;
constexpr uint32_t K_INVALID_INDEX = std::numeric_limits<uint32_t>::max();

uint32_t
get_bits_for_value_limit(uint32_t value_limit) {
    if (value_limit <= 1) {
        return 1;
    }
    uint32_t max_value = value_limit;
    uint32_t bits = 0;
    do {
        ++bits;
        max_value >>= 1;
    } while (max_value > 0);
    return bits;
}

uint64_t
get_packed_size(uint64_t count, uint32_t bits) {
    return (count * bits + 7) / 8;
}

void
store_packed(uint8_t* bytes, uint64_t index, uint32_t bits, uint32_t value) {
    const uint64_t bit_offset = index * bits;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        const uint64_t target_bit = bit_offset + bit;
        const uint64_t byte_index = target_bit / 8;
        const auto mask = static_cast<uint8_t>(1U << (target_bit % 8));
        if ((value & (1U << bit)) != 0) {
            bytes[byte_index] |= mask;
        } else {
            bytes[byte_index] &= static_cast<uint8_t>(~mask);
        }
    }
}

uint32_t
load_packed(const uint8_t* bytes, uint64_t index, uint32_t bits) {
    const uint64_t bit_offset = index * bits;
    uint32_t result = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        const uint64_t target_bit = bit_offset + bit;
        if ((bytes[target_bit / 8] & (1U << (target_bit % 8))) != 0) {
            result |= 1U << bit;
        }
    }
    return result;
}

std::tuple<Vector<uint32_t>, Vector<float>>
sort_sparse_vector(const SparseVector& vector, Allocator* allocator) {
    Vector<uint32_t> indexes(vector.len_, allocator);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(), indexes.end(), [&vector](uint32_t left, uint32_t right) {
        return vector.ids_[left] < vector.ids_[right];
    });
    Vector<uint32_t> ids(vector.len_, allocator);
    Vector<float> values(vector.len_, allocator);
    for (uint32_t index = 0; index < vector.len_; ++index) {
        ids[index] = vector.ids_[indexes[index]];
        values[index] = vector.vals_[indexes[index]];
    }
    return {std::move(ids), std::move(values)};
}

}  // namespace

struct SparseDmqQuantizer::QueryData {
    explicit QueryData(Allocator* allocator)
        : ids(allocator), values(allocator), term_to_index(allocator), code_lut(allocator) {
    }

    Vector<uint32_t> ids;
    Vector<float> values;
    Vector<uint32_t> term_to_index;
    Vector<float> code_lut;
    bool has_lookup{false};
};

SparseDmqQuantizer::SparseDmqQuantizer(uint32_t term_id_limit, Allocator* allocator)
    : Quantizer<SparseDmqQuantizer>(0, allocator),
      codebook_term_ids_(allocator),
      codebooks_(allocator),
      codebook_index_by_term_id_(allocator),
      codebook_index_lookup_(allocator),
      id_bits_(get_bits_for_value_limit(term_id_limit)) {
    this->metric_ = MetricType::METRIC_TYPE_IP;
    this->code_size_ = 0;
}

uint64_t
SparseDmqQuantizer::GetEncodedSize(const SparseVector& vector) const {
    return sizeof(EncodedHeader) + get_packed_size(vector.len_, id_bits_) + vector.len_;
}

uint32_t
SparseDmqQuantizer::GetEncodedLength(const uint8_t* codes) {
    EncodedHeader header;
    std::memcpy(&header, codes, sizeof(header));
    return header.len;
}

uint32_t
SparseDmqQuantizer::GetCodebookIndex(uint32_t term_id) const {
    if (term_id < codebook_index_lookup_.size() &&
        codebook_index_lookup_[term_id] != K_INVALID_INDEX) {
        return codebook_index_lookup_[term_id];
    }
    auto iterator = codebook_index_by_term_id_.find(term_id);
    CHECK_ARGUMENT(iterator != codebook_index_by_term_id_.end(),
                   fmt::format("missing DMQ codebook for term id {}", term_id));
    return iterator->second;
}

void
SparseDmqQuantizer::RebuildCodebookLookup() {
    codebook_index_lookup_.clear();
    for (uint32_t index = 0; index < codebook_term_ids_.size(); ++index) {
        AddCodebookLookup(codebook_term_ids_[index], index);
    }
}

void
SparseDmqQuantizer::AddCodebookLookup(uint32_t term_id, uint32_t codebook_index) {
    if (term_id >= K_MAX_LOOKUP_VALUES) {
        return;
    }
    if (term_id >= codebook_index_lookup_.size()) {
        codebook_index_lookup_.resize(static_cast<uint64_t>(term_id) + 1, K_INVALID_INDEX);
    }
    codebook_index_lookup_[term_id] = codebook_index;
}

void
SparseDmqQuantizer::BuildCodebook(float* values, uint32_t length, Codebook* codebook) {
    codebook->thresholds.fill(0.0F);
    codebook->values.fill(0.0F);
    if (length == 0) {
        return;
    }
    std::sort(values, values + length);
    double sum = 0.0;
    double square_sum = 0.0;
    for (uint32_t index = 0; index < length; ++index) {
        sum += values[index];
        square_sum += static_cast<double>(values[index]) * values[index];
    }
    double total_weight = 0.0;
    for (uint32_t index = 0; index < length; ++index) {
        const double value = values[index];
        total_weight += value * value * length + square_sum - 2.0 * value * sum;
    }
    if (total_weight <= K_MIN_DENOMINATOR) {
        codebook->thresholds.fill(values[0]);
        codebook->values.fill(values[0]);
        return;
    }
    uint32_t partition = 1;
    double current_weight = 0.0;
    for (uint32_t index = 0; index < length && partition < CODEBOOK_SIZE * 2; ++index) {
        const double value = values[index];
        current_weight += value * value * length + square_sum - 2.0 * value * sum;
        while (current_weight * (CODEBOOK_SIZE * 2) + 1e-7 >= total_weight * partition) {
            if (((partition - 1) & 1U) != 0U) {
                codebook->thresholds[(partition - 1) / 2] = values[index];
            } else {
                codebook->values[(partition - 1) / 2] = values[index];
            }
            if (++partition == CODEBOOK_SIZE * 2) {
                break;
            }
        }
    }
    for (; partition < CODEBOOK_SIZE * 2; ++partition) {
        if (((partition - 1) & 1U) != 0U) {
            codebook->thresholds[(partition - 1) / 2] = values[length - 1];
        } else {
            codebook->values[(partition - 1) / 2] = values[length - 1];
        }
    }
}

uint8_t
SparseDmqQuantizer::EncodeResidual(float residual, const Codebook& codebook) {
    return static_cast<uint8_t>(
        std::lower_bound(codebook.thresholds.begin(), codebook.thresholds.end(), residual) -
        codebook.thresholds.begin());
}

float
SparseDmqQuantizer::DecodeValue(const VectorFactors& factors,
                                const Codebook& codebook,
                                uint8_t code) {
    return factors.mean + factors.alpha * codebook.values[code];
}

bool
SparseDmqQuantizer::TrainImpl(const float* data, uint64_t count) {
    CHECK_ARGUMENT(data != nullptr, "SparseDmqQuantizer training data is null");
    const auto* vectors = reinterpret_cast<const SparseVector*>(data);
    Vector<float> means(count, this->allocator_);
    UnorderedMap<uint32_t, uint32_t> term_indexes(this->allocator_);
    Vector<uint32_t> term_ids(this->allocator_);
    Vector<uint32_t> term_counts(this->allocator_);
    for (uint64_t vector_index = 0; vector_index < count; ++vector_index) {
        double sum = 0.0;
        for (uint32_t index = 0; index < vectors[vector_index].len_; ++index) {
            sum += vectors[vector_index].vals_[index];
        }
        means[vector_index] = vectors[vector_index].len_ == 0
                                  ? 0.0F
                                  : static_cast<float>(sum / vectors[vector_index].len_);
        for (uint32_t index = 0; index < vectors[vector_index].len_; ++index) {
            const uint32_t term_id = vectors[vector_index].ids_[index];
            if (codebook_index_by_term_id_.find(term_id) != codebook_index_by_term_id_.end()) {
                continue;
            }
            auto [iterator, inserted] =
                term_indexes.emplace(term_id, static_cast<uint32_t>(term_ids.size()));
            if (inserted) {
                term_ids.push_back(term_id);
                term_counts.push_back(0);
            }
            ++term_counts[iterator->second];
        }
    }
    Vector<uint64_t> offsets(term_ids.size() + 1, 0, this->allocator_);
    for (uint32_t index = 0; index < term_ids.size(); ++index) {
        offsets[index + 1] = offsets[index] + term_counts[index];
    }
    Vector<uint64_t> cursors(offsets.begin(), offsets.end(), this->allocator_);
    Vector<float> residuals(offsets.back(), this->allocator_);
    for (uint64_t vector_index = 0; vector_index < count; ++vector_index) {
        for (uint32_t index = 0; index < vectors[vector_index].len_; ++index) {
            auto iterator = term_indexes.find(vectors[vector_index].ids_[index]);
            if (iterator != term_indexes.end()) {
                residuals[cursors[iterator->second]++] =
                    vectors[vector_index].vals_[index] - means[vector_index];
            }
        }
    }
    for (uint32_t index = 0; index < term_ids.size(); ++index) {
        const auto codebook_index = static_cast<uint32_t>(codebooks_.size());
        codebook_index_by_term_id_[term_ids[index]] = codebook_index;
        codebook_term_ids_.push_back(term_ids[index]);
        codebooks_.emplace_back();
        BuildCodebook(residuals.data() + offsets[index], term_counts[index], &codebooks_.back());
        AddCodebookLookup(term_ids[index], codebook_index);
    }
    this->is_trained_ = true;
    return true;
}

bool
SparseDmqQuantizer::EncodeOneImpl(const float* data, uint8_t* codes) const {
    const auto& vector = *reinterpret_cast<const SparseVector*>(data);
    auto [ids, values] = sort_sparse_vector(vector, this->allocator_);
    EncodedHeader header;
    header.len = vector.len_;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    header.factors.mean = vector.len_ == 0 ? 0.0F : static_cast<float>(sum / vector.len_);
    auto* packed_ids = codes + sizeof(header);
    const uint64_t id_bytes = get_packed_size(vector.len_, id_bits_);
    std::fill_n(packed_ids, id_bytes, 0);
    auto* value_codes = packed_ids + id_bytes;
    double numerator = 0.0;
    double denominator = 0.0;
    for (uint32_t index = 0; index < vector.len_; ++index) {
        store_packed(packed_ids, index, id_bits_, ids[index]);
        const auto& codebook = codebooks_[GetCodebookIndex(ids[index])];
        const float residual = values[index] - header.factors.mean;
        value_codes[index] = EncodeResidual(residual, codebook);
        numerator += static_cast<double>(residual) * residual;
        denominator += static_cast<double>(codebook.values[value_codes[index]]) * residual;
    }
    header.factors.alpha = std::abs(denominator) <= K_MIN_DENOMINATOR
                               ? 0.0F
                               : static_cast<float>(numerator / denominator);
    std::memcpy(codes, &header, sizeof(header));
    return true;
}

bool
SparseDmqQuantizer::DecodeOneImpl(const uint8_t* codes, float* data) const {
    auto* vector = reinterpret_cast<SparseVector*>(data);
    EncodedHeader header;
    std::memcpy(&header, codes, sizeof(header));
    vector->len_ = header.len;
    vector->ids_ =
        static_cast<uint32_t*>(this->allocator_->Allocate(sizeof(uint32_t) * header.len));
    vector->vals_ = static_cast<float*>(this->allocator_->Allocate(sizeof(float) * header.len));
    const auto* packed_ids = codes + sizeof(header);
    const auto* value_codes = packed_ids + get_packed_size(header.len, id_bits_);
    for (uint32_t index = 0; index < header.len; ++index) {
        vector->ids_[index] = load_packed(packed_ids, index, id_bits_);
        vector->vals_[index] = DecodeValue(
            header.factors, codebooks_[GetCodebookIndex(vector->ids_[index])], value_codes[index]);
    }
    return true;
}

bool
SparseDmqQuantizer::EncodeBatchImpl(const float* /*data*/, uint8_t* /*codes*/, uint64_t /*count*/) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "variable-length DMQ codes do not support EncodeBatchImpl");
}

bool
SparseDmqQuantizer::DecodeBatchImpl(const uint8_t* /*codes*/, float* /*data*/, uint64_t /*count*/) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "variable-length DMQ codes do not support DecodeBatchImpl");
}

float
SparseDmqQuantizer::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    EncodedHeader left;
    EncodedHeader right;
    std::memcpy(&left, codes1, sizeof(left));
    std::memcpy(&right, codes2, sizeof(right));
    const auto* left_ids = codes1 + sizeof(left);
    const auto* right_ids = codes2 + sizeof(right);
    const auto* left_codes = left_ids + get_packed_size(left.len, id_bits_);
    const auto* right_codes = right_ids + get_packed_size(right.len, id_bits_);
    uint32_t left_index = 0;
    uint32_t right_index = 0;
    float product = 0.0F;
    while (left_index < left.len && right_index < right.len) {
        const uint32_t left_id = load_packed(left_ids, left_index, id_bits_);
        const uint32_t right_id = load_packed(right_ids, right_index, id_bits_);
        if (left_id < right_id) {
            ++left_index;
        } else if (left_id > right_id) {
            ++right_index;
        } else {
            const auto& codebook = codebooks_[GetCodebookIndex(left_id)];
            product += DecodeValue(left.factors, codebook, left_codes[left_index]) *
                       DecodeValue(right.factors, codebook, right_codes[right_index]);
            ++left_index;
            ++right_index;
        }
    }
    return 1.0F - product;
}

void
SparseDmqQuantizer::ProcessQueryImpl(const float* query,
                                     Computer<SparseDmqQuantizer>& computer) const {
    void* memory = this->allocator_->Allocate(sizeof(QueryData));
    auto* query_data = new (memory) QueryData(this->allocator_);
    auto [ids, values] =
        sort_sparse_vector(*reinterpret_cast<const SparseVector*>(query), this->allocator_);
    query_data->ids = std::move(ids);
    query_data->values = std::move(values);
    if (!query_data->ids.empty() && query_data->ids.back() < K_MAX_LOOKUP_VALUES) {
        query_data->term_to_index.assign(query_data->ids.back() + 1, K_INVALID_INDEX);
        for (uint32_t index = 0; index < query_data->ids.size(); ++index) {
            query_data->term_to_index[query_data->ids[index]] = index;
        }
        query_data->has_lookup = true;
    }
    query_data->code_lut.resize(static_cast<uint64_t>(query_data->ids.size()) * CODEBOOK_SIZE,
                                0.0F);
    for (uint32_t index = 0; index < query_data->ids.size(); ++index) {
        auto iterator = codebook_index_by_term_id_.find(query_data->ids[index]);
        if (iterator == codebook_index_by_term_id_.end()) {
            continue;
        }
        const auto& codebook = codebooks_[iterator->second];
        for (uint32_t code = 0; code < CODEBOOK_SIZE; ++code) {
            query_data->code_lut[static_cast<uint64_t>(index) * CODEBOOK_SIZE + code] =
                query_data->values[index] * codebook.values[code];
        }
    }
    computer.buf_ = reinterpret_cast<uint8_t*>(query_data);
}

void
SparseDmqQuantizer::ComputeDistImpl(Computer<SparseDmqQuantizer>& computer,
                                    const uint8_t* codes,
                                    float* dists) const {
    const auto& query = *reinterpret_cast<const QueryData*>(computer.buf_);
    EncodedHeader header;
    std::memcpy(&header, codes, sizeof(header));
    const auto* packed_ids = codes + sizeof(header);
    const auto* value_codes = packed_ids + get_packed_size(header.len, id_bits_);
    float query_sum = 0.0F;
    float qualifier_product = 0.0F;
    uint32_t query_index = 0;
    for (uint32_t base_index = 0; base_index < header.len; ++base_index) {
        const uint32_t id = load_packed(packed_ids, base_index, id_bits_);
        uint32_t matched = K_INVALID_INDEX;
        if (query.has_lookup) {
            if (id < query.term_to_index.size()) {
                matched = query.term_to_index[id];
            }
        } else {
            while (query_index < query.ids.size() && query.ids[query_index] < id) {
                ++query_index;
            }
            if (query_index < query.ids.size() && query.ids[query_index] == id) {
                matched = query_index;
            }
        }
        if (matched != K_INVALID_INDEX) {
            query_sum += query.values[matched];
            qualifier_product += query.code_lut[static_cast<uint64_t>(matched) * CODEBOOK_SIZE +
                                                value_codes[base_index]];
        }
    }
    dists[0] = 1.0F - (header.factors.mean * query_sum + header.factors.alpha * qualifier_product);
}

void
SparseDmqQuantizer::ReleaseComputerImpl(Computer<SparseDmqQuantizer>& computer) const {
    if (computer.buf_ != nullptr) {
        auto* query = reinterpret_cast<QueryData*>(computer.buf_);
        query->~QueryData();
        this->allocator_->Deallocate(query);
        computer.buf_ = nullptr;
    }
}

void
SparseDmqQuantizer::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, id_bits_);
    StreamWriter::WriteVector(writer, codebook_term_ids_);
    StreamWriter::WriteVector(writer, codebooks_);
}

void
SparseDmqQuantizer::DeserializeImpl(StreamReader& reader) {
    uint32_t serialized_id_bits = 0;
    StreamReader::ReadObj(reader, serialized_id_bits);
    CHECK_ARGUMENT(serialized_id_bits == id_bits_, "serialized DMQ id width does not match");
    StreamReader::ReadVector(reader, codebook_term_ids_);
    StreamReader::ReadVector(reader, codebooks_);
    CHECK_ARGUMENT(codebook_term_ids_.size() == codebooks_.size(),
                   "serialized DMQ codebook metadata is inconsistent");
    codebook_index_by_term_id_.clear();
    for (uint32_t index = 0; index < codebook_term_ids_.size(); ++index) {
        codebook_index_by_term_id_[codebook_term_ids_[index]] = index;
    }
    RebuildCodebookLookup();
}

std::string
SparseDmqQuantizer::NameImpl() {
    return "dmq8";
}

uint64_t
SparseDmqQuantizer::GetMemoryUsage() const {
    return sizeof(*this) + codebook_term_ids_.capacity() * sizeof(uint32_t) +
           codebooks_.capacity() * sizeof(Codebook) +
           codebook_index_by_term_id_.size() * sizeof(std::pair<uint32_t, uint32_t>) +
           codebook_index_lookup_.capacity() * sizeof(uint32_t);
}

void
SparseDmqQuantizer::ExportModel(const SparseDmqQuantizer& other) {
    CHECK_ARGUMENT(id_bits_ == other.id_bits_, "DMQ export target id width mismatch");
    codebook_term_ids_ = other.codebook_term_ids_;
    codebooks_ = other.codebooks_;
    codebook_index_by_term_id_ = other.codebook_index_by_term_id_;
    RebuildCodebookLookup();
    this->is_trained_ = other.is_trained_;
}

}  // namespace vsag
