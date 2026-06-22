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

#include "sindi_rerank_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "common.h"
#include "impl/label_table/label_table.h"

namespace vsag {
namespace {

constexpr uint32_t kSparseDmqBackendMagic = 0x53444D51U;
constexpr uint32_t kSparseDmqBackendVersion = 4;
constexpr uint64_t kMaxDmqQueryLookupValues = 1'000'000;
constexpr uint32_t kInvalidCodebookIndex = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kDmqDecodeBlockSize = 32;

uint64_t
get_packed_code_size(uint64_t len, uint32_t bits) {
    return (len * bits + 7) / 8;
}

uint32_t
get_bits_for_value_limit(uint32_t value_limit) {
    if (value_limit <= 1) {
        return 1;
    }

    uint32_t max_value = value_limit - 1;
    uint32_t bits = 0;
    do {
        ++bits;
        max_value >>= 1;
    } while (max_value > 0);
    return bits;
}

void
store_packed_value(Vector<uint8_t>& bytes, uint64_t value_index, uint32_t bits, uint32_t value) {
    uint64_t bit_offset = value_index * bits;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        uint64_t target_bit = bit_offset + bit;
        auto byte_index = static_cast<uint64_t>(target_bit / 8);
        auto bit_index = static_cast<uint8_t>(target_bit % 8);
        auto bit_mask = static_cast<uint8_t>(1U << bit_index);
        if ((value & (1U << bit)) != 0) {
            bytes[byte_index] |= bit_mask;
        } else {
            bytes[byte_index] &= static_cast<uint8_t>(~bit_mask);
        }
    }
}

uint32_t
load_packed_value(const Vector<uint8_t>& bytes, uint64_t value_index, uint32_t bits) {
    uint64_t bit_offset = value_index * bits;
    auto byte_offset = static_cast<uint64_t>(bit_offset / 8);
    auto bit_shift = static_cast<uint32_t>(bit_offset % 8);
    auto byte_count = static_cast<uint32_t>((bit_shift + bits + 7) / 8);

    uint64_t packed = 0;
    for (uint32_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        packed |= static_cast<uint64_t>(bytes[byte_offset + byte_index]) << (byte_index * 8);
    }
    auto mask = bits == 32 ? std::numeric_limits<uint32_t>::max() : ((1U << bits) - 1U);
    return static_cast<uint32_t>((packed >> bit_shift) & mask);
}

std::tuple<Vector<uint32_t>, Vector<float>>
sort_sparse_vector(const SparseVector& vector, Allocator* allocator) {
    Vector<uint32_t> indices(vector.len_, allocator);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
        return vector.ids_[a] < vector.ids_[b];
    });
    Vector<uint32_t> sorted_ids(vector.len_, allocator);
    Vector<float> sorted_vals(vector.len_, allocator);
    for (uint64_t j = 0; j < vector.len_; ++j) {
        sorted_ids[j] = vector.ids_[indices[j]];
        sorted_vals[j] = vector.vals_[indices[j]];
    }
    return std::make_tuple(sorted_ids, sorted_vals);
}

class SINDIDmqRerankQueryContext : public SINDIRerankQueryContext {
public:
    SINDIDmqRerankQueryContext(Vector<uint32_t> sorted_ids,
                               Vector<float> sorted_vals,
                               Allocator* allocator)
        : sorted_ids_(std::move(sorted_ids)),
          sorted_vals_(std::move(sorted_vals)),
          lookup_values_(allocator) {
        if (sorted_ids_.empty()) {
            return;
        }
        uint64_t lookup_size = static_cast<uint64_t>(sorted_ids_.back()) + 1;
        if (lookup_size > kMaxDmqQueryLookupValues) {
            return;
        }

        lookup_values_.resize(lookup_size, 0.0F);
        for (uint32_t term_index = 0; term_index < sorted_ids_.size(); ++term_index) {
            lookup_values_[sorted_ids_[term_index]] = sorted_vals_[term_index];
        }
        has_lookup_ = true;
    }

    Vector<uint32_t> sorted_ids_;
    Vector<float> sorted_vals_;
    Vector<float> lookup_values_;
    bool has_lookup_{false};
};

}  // namespace

SINDIDmqRerankBackend::SINDIDmqRerankBackend(uint32_t bits,
                                             uint32_t term_id_limit,
                                             std::shared_ptr<LabelTable> label_table,
                                             const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      label_table_(std::move(label_table)),
      encoded_vectors_(allocator_),
      id_codes_(allocator_),
      value_codes_(allocator_),
      codebook_term_ids_(allocator_),
      codebooks_(allocator_),
      codebook_index_by_term_id_(allocator_),
      codebook_index_lookup_(allocator_),
      total_bits_(bits),
      id_bits_(get_bits_for_value_limit(term_id_limit)) {
    CHECK_ARGUMENT(bits == sindi_dmq::kDirectDmqBits,
                   fmt::format("dmq_bits must be 8 for direct DMQ rerank, got {}", bits));
    CHECK_ARGUMENT(label_table_ != nullptr, "label table is null for dmq rerank backend");
}

void
SINDIDmqRerankBackend::StoreCode(uint64_t term_offset, uint32_t term_index, uint8_t code) {
    if (total_bits_ == sindi_dmq::kDirectDmqBits) {
        value_codes_[term_offset + term_index] = code;
        return;
    }
    store_packed_value(value_codes_, term_offset + term_index, total_bits_, code);
}

uint8_t
SINDIDmqRerankBackend::LoadCode(uint64_t term_offset, uint32_t term_index) const {
    if (total_bits_ == sindi_dmq::kDirectDmqBits) {
        return value_codes_[term_offset + term_index];
    }
    return static_cast<uint8_t>(
        load_packed_value(value_codes_, term_offset + term_index, total_bits_));
}

void
SINDIDmqRerankBackend::StoreId(uint64_t term_offset, uint32_t term_index, uint32_t id) {
    store_packed_value(id_codes_, term_offset + term_index, id_bits_, id);
}

uint32_t
SINDIDmqRerankBackend::LoadId(uint64_t term_offset, uint32_t term_index) const {
    return load_packed_value(id_codes_, term_offset + term_index, id_bits_);
}

void
SINDIDmqRerankBackend::LoadIdBlock(uint64_t term_offset,
                                   uint32_t term_index,
                                   uint32_t count,
                                   uint32_t* ids) const {
    uint64_t bit_offset = (term_offset + term_index) * id_bits_;
    uint64_t byte_offset = bit_offset / 8;
    uint32_t skip_bits = static_cast<uint32_t>(bit_offset % 8);
    const auto* cursor = id_codes_.data() + byte_offset;
    uint64_t bit_buffer = 0;
    uint32_t available_bits = 0;
    auto append_byte = [&]() {
        bit_buffer |= static_cast<uint64_t>(*cursor++) << available_bits;
        available_bits += 8;
    };

    while (available_bits < skip_bits) {
        append_byte();
    }
    bit_buffer >>= skip_bits;
    available_bits -= skip_bits;

    uint32_t mask = id_bits_ == 32 ? std::numeric_limits<uint32_t>::max() : ((1U << id_bits_) - 1U);
    for (uint32_t block_index = 0; block_index < count; ++block_index) {
        while (available_bits < id_bits_) {
            append_byte();
        }
        ids[block_index] = static_cast<uint32_t>(bit_buffer) & mask;
        bit_buffer >>= id_bits_;
        available_bits -= id_bits_;
    }
}

uint32_t
SINDIDmqRerankBackend::GetCodebookIndex(uint32_t term_id) const {
    if (term_id < codebook_index_lookup_.size()) {
        uint32_t index = codebook_index_lookup_[term_id];
        if (index != kInvalidCodebookIndex) {
            return index;
        }
    }
    auto iterator = codebook_index_by_term_id_.find(term_id);
    CHECK_ARGUMENT(iterator != codebook_index_by_term_id_.end(),
                   fmt::format("missing direct DMQ codebook for term id {}", term_id));
    return iterator->second;
}

void
SINDIDmqRerankBackend::RebuildCodebookLookup() {
    uint32_t lookup_size = 0;
    for (uint32_t term_id : codebook_term_ids_) {
        if (term_id <= kMaxDmqQueryLookupValues) {
            lookup_size = std::max<uint32_t>(lookup_size, term_id + 1);
        }
    }

    codebook_index_lookup_.assign(lookup_size, kInvalidCodebookIndex);
    for (uint32_t codebook_index = 0; codebook_index < codebook_term_ids_.size();
         ++codebook_index) {
        uint32_t term_id = codebook_term_ids_[codebook_index];
        if (term_id < codebook_index_lookup_.size()) {
            codebook_index_lookup_[term_id] = codebook_index;
        }
    }
}

void
SINDIDmqRerankBackend::TrainCodebooks(const DatasetPtr& base, bool train_missing_only) {
    const auto* sparse_vectors = base->GetSparseVectors();
    auto data_num = base->GetNumElements();
    Vector<float> means(data_num, allocator_);
    UnorderedMap<uint32_t, uint32_t> local_term_indexes(allocator_);
    Vector<uint32_t> local_term_ids(allocator_);
    Vector<uint32_t> local_term_counts(allocator_);

    for (int64_t vector_index = 0; vector_index < data_num; ++vector_index) {
        const auto& vector = sparse_vectors[vector_index];
        double value_sum = 0.0;
        for (uint32_t term_index = 0; term_index < vector.len_; ++term_index) {
            value_sum += vector.vals_[term_index];
        }
        means[vector_index] = static_cast<float>(value_sum / vector.len_);

        for (uint32_t term_index = 0; term_index < vector.len_; ++term_index) {
            uint32_t term_id = vector.ids_[term_index];
            if (train_missing_only &&
                codebook_index_by_term_id_.find(term_id) != codebook_index_by_term_id_.end()) {
                continue;
            }
            auto iterator = local_term_indexes.find(term_id);
            if (iterator == local_term_indexes.end()) {
                uint32_t local_index = static_cast<uint32_t>(local_term_ids.size());
                local_term_indexes[term_id] = local_index;
                local_term_ids.push_back(term_id);
                local_term_counts.push_back(0);
                iterator = local_term_indexes.find(term_id);
            }
            ++local_term_counts[iterator->second];
        }
    }

    if (local_term_ids.empty()) {
        return;
    }

    Vector<uint64_t> offsets(local_term_ids.size() + 1, 0, allocator_);
    for (uint32_t local_index = 0; local_index < local_term_ids.size(); ++local_index) {
        offsets[local_index + 1] = offsets[local_index] + local_term_counts[local_index];
    }
    Vector<uint64_t> cursors(offsets.begin(), offsets.end(), allocator_);
    Vector<float> residual_samples(offsets.back(), allocator_);

    for (int64_t vector_index = 0; vector_index < data_num; ++vector_index) {
        const auto& vector = sparse_vectors[vector_index];
        for (uint32_t term_index = 0; term_index < vector.len_; ++term_index) {
            auto iterator = local_term_indexes.find(vector.ids_[term_index]);
            if (iterator == local_term_indexes.end()) {
                continue;
            }
            uint32_t local_index = iterator->second;
            residual_samples[cursors[local_index]++] =
                vector.vals_[term_index] - means[vector_index];
        }
    }

    codebook_term_ids_.reserve(codebook_term_ids_.size() + local_term_ids.size());
    codebooks_.reserve(codebooks_.size() + local_term_ids.size());
    for (uint32_t local_index = 0; local_index < local_term_ids.size(); ++local_index) {
        auto codebook_index = static_cast<uint32_t>(codebooks_.size());
        codebook_index_by_term_id_[local_term_ids[local_index]] = codebook_index;
        codebook_term_ids_.push_back(local_term_ids[local_index]);
        codebooks_.emplace_back();
        auto* samples = residual_samples.data() + offsets[local_index];
        sindi_dmq::BuildDirectCodebook(samples, local_term_counts[local_index], &codebooks_.back());
    }
    RebuildCodebookLookup();
}

void
SINDIDmqRerankBackend::Add(const DatasetPtr& base) {
    const auto* sparse_vectors = base->GetSparseVectors();
    auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num > 0, "data_num is zero when add vectors");

    TrainCodebooks(base, !codebooks_.empty());

    uint64_t total_new_len = 0;
    for (int64_t i = 0; i < data_num; ++i) {
        total_new_len += sparse_vectors[i].len_;
    }
    uint64_t final_term_count = total_term_count_ + total_new_len;
    encoded_vectors_.reserve(encoded_vectors_.size() + data_num);
    id_codes_.resize(get_packed_code_size(final_term_count, id_bits_), 0);
    if (total_bits_ == sindi_dmq::kDirectDmqBits) {
        value_codes_.resize(final_term_count, 0);
    } else {
        value_codes_.resize(get_packed_code_size(final_term_count, total_bits_), 0);
    }

    for (int64_t i = 0; i < data_num; ++i) {
        const auto& vector = sparse_vectors[i];
        auto [sorted_ids, sorted_vals] = sort_sparse_vector(vector, allocator_);

        EncodedVector encoded;
        encoded.term_offset = total_term_count_;
        encoded.len = vector.len_;

        double value_sum = 0.0;
        for (uint32_t term_index = 0; term_index < encoded.len; ++term_index) {
            value_sum += sorted_vals[term_index];
        }
        encoded.factors.mean = static_cast<float>(value_sum / encoded.len);

        Vector<uint8_t> unpacked_codes(encoded.len, allocator_);
        double numerator = 0.0;
        double denominator = 0.0;
        for (uint32_t term_index = 0; term_index < encoded.len; ++term_index) {
            const auto& codebook = codebooks_[GetCodebookIndex(sorted_ids[term_index])];
            float residual = sorted_vals[term_index] - encoded.factors.mean;
            uint8_t code = sindi_dmq::EncodeDirectResidual(residual, codebook);
            float qualifier = codebook.values[code];
            unpacked_codes[term_index] = code;
            numerator += static_cast<double>(residual) * residual;
            denominator += static_cast<double>(qualifier) * residual;
        }
        encoded.factors.alpha = 0.0F;
        if (std::abs(denominator) > 1e-12) {
            encoded.factors.alpha = static_cast<float>(numerator / denominator);
        }

        uint64_t next_term_count = total_term_count_ + encoded.len;
        for (uint32_t term_index = 0; term_index < encoded.len; ++term_index) {
            StoreId(encoded.term_offset, term_index, sorted_ids[term_index]);
            StoreCode(encoded.term_offset, term_index, unpacked_codes[term_index]);
        }

        encoded_vectors_.push_back(encoded);
        total_term_count_ = next_term_count;
        ++cur_element_count_;
    }
}

std::unique_ptr<SINDIRerankQueryContext>
SINDIDmqRerankBackend::PrepareQuery(const SparseVector& query) const {
    auto [sorted_ids, sorted_vals] = sort_sparse_vector(query, allocator_);
    return std::make_unique<SINDIDmqRerankQueryContext>(
        std::move(sorted_ids), std::move(sorted_vals), allocator_);
}

float
SINDIDmqRerankBackend::CalDistanceByInnerId(const SINDIRerankQueryContext& query_context,
                                            InnerIdType inner_id) const {
    auto& dmq_context = dynamic_cast<const SINDIDmqRerankQueryContext&>(query_context);
    const auto& encoded = encoded_vectors_[inner_id];

    float query_sum = 0.0F;
    float qualifier_product = 0.0F;
    if (dmq_context.has_lookup_) {
        std::array<uint32_t, kDmqDecodeBlockSize> id_block{};
        uint32_t base_offset = 0;
        while (base_offset < encoded.len) {
            uint32_t block_count =
                std::min<uint32_t>(kDmqDecodeBlockSize, encoded.len - base_offset);
            LoadIdBlock(encoded.term_offset, base_offset, block_count, id_block.data());
            const auto* code_block = value_codes_.data() + encoded.term_offset + base_offset;

            for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
                auto base_id = id_block[block_index];
                if (static_cast<uint64_t>(base_id) >= dmq_context.lookup_values_.size()) {
                    continue;
                }
                float query_value = dmq_context.lookup_values_[base_id];
                if (query_value == 0.0F) {
                    continue;
                }
                const auto& codebook = codebooks_[GetCodebookIndex(base_id)];
                query_sum += query_value;
                qualifier_product += query_value * codebook.values[code_block[block_index]];
            }
            base_offset += block_count;
        }
        float inner_product =
            encoded.factors.mean * query_sum + encoded.factors.alpha * qualifier_product;
        return 1.0F - inner_product;
    }

    uint32_t query_offset = 0;
    uint32_t base_offset = 0;
    std::array<uint32_t, kDmqDecodeBlockSize> id_block{};
    while (query_offset < dmq_context.sorted_ids_.size() && base_offset < encoded.len) {
        uint32_t block_count = std::min<uint32_t>(kDmqDecodeBlockSize, encoded.len - base_offset);
        LoadIdBlock(encoded.term_offset, base_offset, block_count, id_block.data());
        const auto* code_block = value_codes_.data() + encoded.term_offset + base_offset;

        for (uint32_t block_index = 0;
             block_index < block_count && query_offset < dmq_context.sorted_ids_.size();) {
            auto base_id = id_block[block_index];
            if (dmq_context.sorted_ids_[query_offset] < base_id) {
                ++query_offset;
            } else if (dmq_context.sorted_ids_[query_offset] > base_id) {
                ++block_index;
            } else {
                const auto& codebook = codebooks_[GetCodebookIndex(base_id)];
                float query_value = dmq_context.sorted_vals_[query_offset];
                query_sum += query_value;
                qualifier_product += query_value * codebook.values[code_block[block_index]];
                ++query_offset;
                ++block_index;
            }
        }
        base_offset += block_count;
    }

    float inner_product =
        encoded.factors.mean * query_sum + encoded.factors.alpha * qualifier_product;
    return 1.0F - inner_product;
}

float
SINDIDmqRerankBackend::CalcDistanceById(const DatasetPtr& vector, int64_t id) const {
    auto [success, inner_id] = label_table_->TryGetIdByLabel(id);
    if (not success) {
        return -1.0F;
    }
    auto context = PrepareQuery(vector->GetSparseVectors()[0]);
    return CalDistanceByInnerId(*context, inner_id);
}

DatasetPtr
SINDIDmqRerankBackend::CalDistanceById(const DatasetPtr& query,
                                       const int64_t* ids,
                                       int64_t count) const {
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
    result->Distances(distances);

    auto context = PrepareQuery(query->GetSparseVectors()[0]);
    for (int64_t i = 0; i < count; ++i) {
        auto [success, inner_id] = label_table_->TryGetIdByLabel(ids[i]);
        distances[i] = success ? CalDistanceByInnerId(*context, inner_id) : -1.0F;
    }
    return result;
}

void
SINDIDmqRerankBackend::GetSparseVectorByInnerId(InnerIdType inner_id,
                                                SparseVector* data,
                                                Allocator* specified_allocator) const {
    Allocator* allocator = specified_allocator != nullptr ? specified_allocator : allocator_;
    const auto& encoded = encoded_vectors_[inner_id];

    data->len_ = encoded.len;
    data->ids_ = nullptr;
    data->vals_ = nullptr;
    if (encoded.len == 0) {
        return;
    }

    data->ids_ = static_cast<uint32_t*>(allocator->Allocate(sizeof(uint32_t) * encoded.len));
    data->vals_ = static_cast<float*>(allocator->Allocate(sizeof(float) * encoded.len));

    for (uint32_t term_index = 0; term_index < encoded.len; ++term_index) {
        data->ids_[term_index] = LoadId(encoded.term_offset, term_index);
        auto code = LoadCode(encoded.term_offset, term_index);
        const auto& codebook = codebooks_[GetCodebookIndex(data->ids_[term_index])];
        data->vals_[term_index] = sindi_dmq::DecodeDirectValue(encoded.factors, codebook, code);
    }
}

void
SINDIDmqRerankBackend::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, kSparseDmqBackendMagic);
    StreamWriter::WriteObj(writer, kSparseDmqBackendVersion);
    StreamWriter::WriteObj(writer, total_bits_);
    StreamWriter::WriteObj(writer, id_bits_);
    StreamWriter::WriteObj(writer, cur_element_count_);
    StreamWriter::WriteObj(writer, total_term_count_);
    StreamWriter::WriteVector(writer, encoded_vectors_);
    StreamWriter::WriteVector(writer, id_codes_);
    StreamWriter::WriteVector(writer, value_codes_);
    StreamWriter::WriteVector(writer, codebook_term_ids_);
    StreamWriter::WriteVector(writer, codebooks_);
}

void
SINDIDmqRerankBackend::Deserialize(StreamReader& reader) {
    uint32_t serialized_magic = 0;
    StreamReader::ReadObj(reader, serialized_magic);
    CHECK_ARGUMENT(serialized_magic == kSparseDmqBackendMagic,
                   "serialized dmq backend is not sparse-DMQ format");
    uint32_t serialized_version = 0;
    StreamReader::ReadObj(reader, serialized_version);
    CHECK_ARGUMENT(serialized_version == kSparseDmqBackendVersion,
                   fmt::format("unsupported sparse-DMQ version {}", serialized_version));
    uint32_t serialized_bits = 0;
    StreamReader::ReadObj(reader, serialized_bits);
    CHECK_ARGUMENT(serialized_bits == total_bits_,
                   fmt::format("serialized dmq_bits {} does not match current dmq_bits {}",
                               serialized_bits,
                               total_bits_));
    uint32_t serialized_id_bits = 0;
    StreamReader::ReadObj(reader, serialized_id_bits);
    CHECK_ARGUMENT(serialized_id_bits == id_bits_,
                   fmt::format("serialized dmq id bits {} does not match current id bits {}",
                               serialized_id_bits,
                               id_bits_));
    StreamReader::ReadObj(reader, cur_element_count_);
    StreamReader::ReadObj(reader, total_term_count_);
    StreamReader::ReadVector(reader, encoded_vectors_);
    StreamReader::ReadVector(reader, id_codes_);
    StreamReader::ReadVector(reader, value_codes_);
    StreamReader::ReadVector(reader, codebook_term_ids_);
    StreamReader::ReadVector(reader, codebooks_);
    CHECK_ARGUMENT(codebook_term_ids_.size() == codebooks_.size(),
                   "serialized direct DMQ codebook metadata is inconsistent");
    codebook_index_by_term_id_.clear();
    codebook_index_by_term_id_.reserve(codebook_term_ids_.size());
    for (uint32_t codebook_index = 0; codebook_index < codebook_term_ids_.size();
         ++codebook_index) {
        codebook_index_by_term_id_[codebook_term_ids_[codebook_index]] = codebook_index;
    }
    RebuildCodebookLookup();
}

int64_t
SINDIDmqRerankBackend::GetMemoryUsage() const {
    uint64_t memory = sizeof(SINDIDmqRerankBackend);
    memory += encoded_vectors_.capacity() * sizeof(EncodedVector);
    memory += id_codes_.capacity() * sizeof(uint8_t);
    memory += value_codes_.capacity() * sizeof(uint8_t);
    memory += codebook_term_ids_.capacity() * sizeof(uint32_t);
    memory += codebooks_.capacity() * sizeof(sindi_dmq::DirectDmqCodebook);
    memory += codebook_index_by_term_id_.size() * (sizeof(uint32_t) * 2);
    memory += codebook_index_lookup_.capacity() * sizeof(uint32_t);
    return static_cast<int64_t>(memory);
}

void
SINDIDmqRerankBackend::GetMemoryUsageDetail(JsonType& memory_usage) const {
    memory_usage["backend_type"].SetString("sparse_dmq_direct8");
    memory_usage["total_bits"].SetInt(total_bits_);
    memory_usage["id_bits"].SetInt(id_bits_);
    memory_usage["num_vectors"].SetInt(static_cast<uint64_t>(cur_element_count_));
    memory_usage["num_terms"].SetInt(total_term_count_);
    memory_usage["num_codebooks"].SetInt(static_cast<uint64_t>(codebooks_.size()));
    memory_usage["object"].SetInt(sizeof(SINDIDmqRerankBackend));
    memory_usage["encoded_vectors"].SetInt(encoded_vectors_.capacity() * sizeof(EncodedVector));
    memory_usage["id_codes"].SetInt(id_codes_.capacity() * sizeof(uint8_t));
    memory_usage["value_codes"].SetInt(value_codes_.capacity() * sizeof(uint8_t));
    memory_usage["codebook_term_ids"].SetInt(codebook_term_ids_.capacity() * sizeof(uint32_t));
    memory_usage["codebooks"].SetInt(codebooks_.capacity() * sizeof(sindi_dmq::DirectDmqCodebook));
    memory_usage["codebook_map_estimate"].SetInt(codebook_index_by_term_id_.size() *
                                                 sizeof(uint32_t) * 2);
    memory_usage["codebook_lookup"].SetInt(codebook_index_lookup_.capacity() * sizeof(uint32_t));
    memory_usage["total"].SetInt(static_cast<uint64_t>(GetMemoryUsage()));
}

}  // namespace vsag
