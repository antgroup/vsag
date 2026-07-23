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

uint32_t
get_bits_for_value_count(uint64_t value_count) {
    if (value_count <= 1) {
        return 1;
    }
    uint64_t max_value = value_count - 1;
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

SparseDmqQuantizer::SparseDmqQuantizer(uint32_t term_id_limit,
                                       Allocator* allocator,
                                       uint32_t shared_codebook_threshold)
    : Quantizer<SparseDmqQuantizer>(0, allocator),
      term_ids_(allocator),
      codebooks_(allocator),
      codebook_index_by_compact_id_(allocator),
      compact_id_by_term_id_(allocator),
      compact_id_lookup_(allocator),
      id_bits_(get_bits_for_value_limit(term_id_limit)),
      shared_codebook_threshold_(shared_codebook_threshold) {
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
SparseDmqQuantizer::GetCompactId(uint32_t term_id) const {
    if (term_id < compact_id_lookup_.size() && compact_id_lookup_[term_id] != K_INVALID_INDEX) {
        return compact_id_lookup_[term_id];
    }
    auto iterator = compact_id_by_term_id_.find(term_id);
    CHECK_ARGUMENT(iterator != compact_id_by_term_id_.end(),
                   fmt::format("missing DMQ compact ID for term ID {}", term_id));
    return iterator->second;
}

uint32_t
SparseDmqQuantizer::GetCodebookIndex(uint32_t compact_id) const {
    CHECK_ARGUMENT(compact_id < codebook_index_by_compact_id_.size(),
                   fmt::format("invalid DMQ compact ID {}", compact_id));
    const uint32_t codebook_index = codebook_index_by_compact_id_[compact_id];
    CHECK_ARGUMENT(codebook_index < codebooks_.size(),
                   fmt::format("invalid DMQ codebook index {}", codebook_index));
    return codebook_index;
}

void
SparseDmqQuantizer::RebuildCompactIdLookup() {
    compact_id_by_term_id_.clear();
    compact_id_lookup_.clear();
    for (uint32_t compact_id = 0; compact_id < term_ids_.size(); ++compact_id) {
        const auto [iterator, inserted] =
            compact_id_by_term_id_.emplace(term_ids_[compact_id], compact_id);
        (void)iterator;
        CHECK_ARGUMENT(inserted,
                       fmt::format("duplicate serialized DMQ term ID {}", term_ids_[compact_id]));
        AddCompactIdLookup(term_ids_[compact_id], compact_id);
    }
}

void
SparseDmqQuantizer::AddCompactIdLookup(uint32_t term_id, uint32_t compact_id) {
    if (term_id >= K_MAX_LOOKUP_VALUES) {
        return;
    }
    if (term_id >= compact_id_lookup_.size()) {
        compact_id_lookup_.resize(static_cast<uint64_t>(term_id) + 1, K_INVALID_INDEX);
    }
    compact_id_lookup_[term_id] = compact_id;
}

void
SparseDmqQuantizer::BuildCodebook(float* values, uint64_t length, Codebook* codebook) {
    codebook->thresholds.fill(0.0F);
    codebook->values.fill(0.0F);
    if (length == 0) {
        return;
    }
    std::sort(values, values + length);
    double sum = 0.0;
    double square_sum = 0.0;
    for (uint64_t index = 0; index < length; ++index) {
        sum += values[index];
        square_sum += static_cast<double>(values[index]) * values[index];
    }
    double total_weight = 0.0;
    for (uint64_t index = 0; index < length; ++index) {
        const double value = values[index];
        total_weight +=
            value * value * static_cast<double>(length) + square_sum - 2.0 * value * sum;
    }
    if (total_weight <= K_MIN_DENOMINATOR) {
        codebook->thresholds.fill(values[0]);
        codebook->values.fill(values[0]);
        return;
    }
    uint32_t partition = 1;
    double current_weight = 0.0;
    for (uint64_t index = 0; index < length && partition < CODEBOOK_SIZE * 2; ++index) {
        const double value = values[index];
        current_weight +=
            value * value * static_cast<double>(length) + square_sum - 2.0 * value * sum;
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
    Vector<uint64_t> term_counts(this->allocator_);
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
            if (compact_id_by_term_id_.find(term_id) != compact_id_by_term_id_.end()) {
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

    CHECK_ARGUMENT(
        term_ids_.empty() or term_ids.empty(),
        "SparseDmqQuantizer does not support adding new terms after compact IDs are assigned");
    Vector<uint32_t> sorted_term_indexes(term_ids.size(), this->allocator_);
    std::iota(sorted_term_indexes.begin(), sorted_term_indexes.end(), 0);
    std::sort(
        sorted_term_indexes.begin(),
        sorted_term_indexes.end(),
        [&term_ids](uint32_t left, uint32_t right) { return term_ids[left] < term_ids[right]; });

    bool has_shared_codebook = false;
    for (const uint32_t term_index : sorted_term_indexes) {
        if (term_counts[term_index] <= shared_codebook_threshold_) {
            has_shared_codebook = true;
            break;
        }
    }
    uint32_t training_bucket_count = has_shared_codebook ? 1 : 0;
    Vector<uint32_t> training_bucket_by_term(term_ids.size(), this->allocator_);
    for (const uint32_t term_index : sorted_term_indexes) {
        if (term_counts[term_index] <= shared_codebook_threshold_) {
            training_bucket_by_term[term_index] = 0;
        } else {
            training_bucket_by_term[term_index] = training_bucket_count++;
        }
    }

    Vector<uint64_t> bucket_counts(training_bucket_count, 0, this->allocator_);
    for (uint32_t term_index = 0; term_index < term_ids.size(); ++term_index) {
        bucket_counts[training_bucket_by_term[term_index]] += term_counts[term_index];
    }
    Vector<uint64_t> offsets(static_cast<uint64_t>(training_bucket_count) + 1, 0, this->allocator_);
    for (uint32_t bucket = 0; bucket < training_bucket_count; ++bucket) {
        offsets[bucket + 1] = offsets[bucket] + bucket_counts[bucket];
    }
    Vector<uint64_t> cursors(offsets.begin(), offsets.end(), this->allocator_);
    Vector<float> residuals(offsets.back(), this->allocator_);
    for (uint64_t vector_index = 0; vector_index < count; ++vector_index) {
        for (uint32_t index = 0; index < vectors[vector_index].len_; ++index) {
            auto iterator = term_indexes.find(vectors[vector_index].ids_[index]);
            if (iterator == term_indexes.end()) {
                continue;
            }
            const uint32_t bucket = training_bucket_by_term[iterator->second];
            residuals[cursors[bucket]++] = vectors[vector_index].vals_[index] - means[vector_index];
        }
    }

    const auto first_codebook_index = static_cast<uint32_t>(codebooks_.size());
    for (uint32_t bucket = 0; bucket < training_bucket_count; ++bucket) {
        codebooks_.emplace_back();
        BuildCodebook(
            residuals.data() + offsets[bucket], bucket_counts[bucket], &codebooks_.back());
    }
    for (const uint32_t term_index : sorted_term_indexes) {
        const auto compact_id = static_cast<uint32_t>(term_ids_.size());
        const uint32_t codebook_index = first_codebook_index + training_bucket_by_term[term_index];
        term_ids_.push_back(term_ids[term_index]);
        codebook_index_by_compact_id_.push_back(codebook_index);
        compact_id_by_term_id_[term_ids[term_index]] = compact_id;
        AddCompactIdLookup(term_ids[term_index], compact_id);
    }
    id_bits_ = get_bits_for_value_count(term_ids_.size());
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
        const uint32_t compact_id = GetCompactId(ids[index]);
        const uint32_t codebook_index = GetCodebookIndex(compact_id);
        store_packed(packed_ids, index, id_bits_, compact_id);
        const auto& codebook = codebooks_[codebook_index];
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
        const uint32_t compact_id = load_packed(packed_ids, index, id_bits_);
        CHECK_ARGUMENT(compact_id < term_ids_.size(),
                       fmt::format("invalid DMQ compact ID {}", compact_id));
        const uint32_t codebook_index = GetCodebookIndex(compact_id);
        vector->ids_[index] = term_ids_[compact_id];
        vector->vals_[index] =
            DecodeValue(header.factors, codebooks_[codebook_index], value_codes[index]);
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
            const uint32_t codebook_index = GetCodebookIndex(left_id);
            const auto& codebook = codebooks_[codebook_index];
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
    Vector<uint32_t> compact_ids(this->allocator_);
    Vector<float> matched_values(this->allocator_);
    compact_ids.reserve(ids.size());
    matched_values.reserve(values.size());
    for (uint32_t index = 0; index < ids.size(); ++index) {
        auto iterator = compact_id_by_term_id_.find(ids[index]);
        if (iterator != compact_id_by_term_id_.end()) {
            compact_ids.push_back(iterator->second);
            matched_values.push_back(values[index]);
        }
    }
    query_data->ids = std::move(compact_ids);
    query_data->values = std::move(matched_values);
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
        const uint32_t codebook_index = GetCodebookIndex(query_data->ids[index]);
        const auto& codebook = codebooks_[codebook_index];
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
    StreamWriter::WriteVector(writer, term_ids_);
    StreamWriter::WriteVector(writer, codebook_index_by_compact_id_);
    StreamWriter::WriteVector(writer, codebooks_);
}

void
SparseDmqQuantizer::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, id_bits_);
    StreamReader::ReadVector(reader, term_ids_);
    StreamReader::ReadVector(reader, codebook_index_by_compact_id_);
    StreamReader::ReadVector(reader, codebooks_);
    CHECK_ARGUMENT(term_ids_.size() == codebook_index_by_compact_id_.size(),
                   "serialized DMQ term and codebook mappings are inconsistent");
    CHECK_ARGUMENT(id_bits_ == get_bits_for_value_count(term_ids_.size()),
                   "serialized DMQ compact ID width is inconsistent");
    CHECK_ARGUMENT(term_ids_.empty() == codebooks_.empty(),
                   "serialized DMQ term and codebook counts are inconsistent");
    for (uint32_t compact_id = 0; compact_id < term_ids_.size(); ++compact_id) {
        CHECK_ARGUMENT(codebook_index_by_compact_id_[compact_id] < codebooks_.size(),
                       fmt::format("invalid serialized DMQ codebook index {}",
                                   codebook_index_by_compact_id_[compact_id]));
        if (compact_id != 0) {
            CHECK_ARGUMENT(term_ids_[compact_id - 1] < term_ids_[compact_id],
                           "serialized DMQ term IDs are not strictly ordered");
        }
    }
    RebuildCompactIdLookup();
}

std::string
SparseDmqQuantizer::NameImpl() {
    return "dmq8";
}

uint64_t
SparseDmqQuantizer::GetMemoryUsage() const {
    return sizeof(*this) + term_ids_.capacity() * sizeof(uint32_t) +
           codebooks_.capacity() * sizeof(Codebook) +
           codebook_index_by_compact_id_.capacity() * sizeof(uint32_t) +
           compact_id_by_term_id_.size() * sizeof(std::pair<uint32_t, uint32_t>) +
           compact_id_lookup_.capacity() * sizeof(uint32_t);
}

void
SparseDmqQuantizer::ExportModel(const SparseDmqQuantizer& other) {
    id_bits_ = other.id_bits_;
    shared_codebook_threshold_ = other.shared_codebook_threshold_;
    term_ids_ = other.term_ids_;
    codebooks_ = other.codebooks_;
    codebook_index_by_compact_id_ = other.codebook_index_by_compact_id_;
    RebuildCompactIdLookup();
    this->is_trained_ = other.is_trained_;
}

}  // namespace vsag
