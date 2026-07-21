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

#include "immutable_sindi_term_datacell.h"

#include <cmath>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("ImmutableSindiTermDataCell term-first memory load uses exact capacity",
          "[ut][SINDIV2][ImmutableSindiTermDataCell]") {
    constexpr uint32_t term_id_limit = 16;
    constexpr uint32_t window_size = 3;

    const auto remap_term_ids = GENERATE(false, true);
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    DYNAMIC_SECTION("remap=" << remap_term_ids
                             << ", quantization=" << static_cast<int>(quantization)) {
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto quantization_params = std::make_shared<QuantizationParams>();
        quantization_params->min_val = 0.0F;
        quantization_params->max_val = 2.0F;
        quantization_params->diff = 2.0F;

        MutableSindiTermDataCell source(
            term_id_limit, window_size, allocator.get(), quantization, quantization_params);
        std::vector<std::vector<uint32_t>> document_terms = {
            {1, 7}, {3}, {1, 5, 9}, {1, 5}, {7, 9}, {3, 9}, {5}};
        std::vector<std::vector<float>> document_values = {{0.25F, 1.25F},
                                                           {0.5F},
                                                           {0.75F, 1.5F, 0.125F},
                                                           {1.0F, 0.375F},
                                                           {0.625F, 1.75F},
                                                           {1.125F, 0.875F},
                                                           {2.0F}};
        for (uint32_t document = 0; document < document_terms.size(); ++document) {
            SparseVector vector{static_cast<uint32_t>(document_terms[document].size()),
                                document_terms[document].data(),
                                document_values[document].data()};
            source.InsertVector(vector, document);
        }
        source.Finalize();

        const auto term_dict_count = source.GetTermDictCount();
        REQUIRE(term_dict_count == 10);
        std::stringstream stream;
        IOStreamWriter writer(stream);
        source.SerializeTermLayout(writer, term_dict_count);

        ImmutableSindiTermDataCell restored(term_id_limit,
                                            window_size,
                                            remap_term_ids,
                                            quantization,
                                            quantization_params,
                                            allocator.get());
        stream.seekg(0, std::ios::beg);
        IOStreamReader reader(stream);
        restored.DeserializeTermLayout(reader, 3, document_terms.size());
        REQUIRE(reader.GetCursor() == writer.GetCursor());
        REQUIRE(restored.GetWindowCount() == 3);

        for (const auto& window : restored.GetWindows()) {
            REQUIRE(window.sorted_global_terms.capacity() == window.sorted_global_terms.size());
            REQUIRE(window.offsets.capacity() == window.offsets.size());
            REQUIRE(window.id_payloads.capacity() == window.id_payloads.size());
            REQUIRE(window.value_payloads.capacity() == window.value_payloads.size());
        }

        std::stringstream restored_stream;
        IOStreamWriter restored_writer(restored_stream);
        restored.SerializeTermLayout(restored_writer, term_dict_count);
        REQUIRE(restored_stream.str() == stream.str());

        std::vector<uint32_t> query_terms = {1, 3, 5, 7, 9};
        std::vector<float> query_values = {0.5F, 0.75F, 1.0F, 1.25F, 1.5F};
        SparseVector query{
            static_cast<uint32_t>(query_terms.size()), query_terms.data(), query_values.data()};
        SINDISearchParameter search_parameter;
        search_parameter.query_prune_ratio = 0.0F;
        search_parameter.term_prune_ratio = 0.0F;
        auto source_computer =
            std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
        auto restored_computer =
            std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
        QueryTermBuffers query_buffers(allocator.get());
        SindiQueryContext source_context(allocator.get());
        SindiQueryContext restored_context(allocator.get());
        source_context.mapped_query_terms.reserve(query_terms.size());
        restored_context.mapped_query_terms.reserve(query_terms.size());
        const uint64_t source_mapped_capacity = source_context.mapped_query_terms.capacity();
        const uint64_t restored_mapped_capacity = restored_context.mapped_query_terms.capacity();

        for (uint32_t window_id = 0; window_id < restored.GetWindowCount(); ++window_id) {
            std::vector<float> source_distances(window_size, 0.0F);
            std::vector<float> restored_distances(window_size, 0.0F);
            source.QueryWindow(
                source_distances.data(), window_id, source_computer, false, source_context);
            restored.QueryWindow(
                restored_distances.data(), window_id, restored_computer, false, restored_context);
            REQUIRE(source_context.mapped_query_terms.capacity() == source_mapped_capacity);
            REQUIRE(restored_context.mapped_query_terms.capacity() == restored_mapped_capacity);
            for (uint32_t local_id = 0; local_id < window_size; ++local_id) {
                REQUIRE(std::abs(source_distances[local_id] - restored_distances[local_id]) <=
                        1e-6F);
            }
        }

        for (uint32_t document = 0; document < document_terms.size(); ++document) {
            const auto source_distance =
                source.CalcDistanceByInnerId(source_computer, document, query_buffers);
            const auto restored_distance =
                restored.CalcDistanceByInnerId(restored_computer, document, query_buffers);
            REQUIRE(std::abs(source_distance - restored_distance) <= 1e-6F);

            SparseVector restored_vector;
            restored.GetSparseVector(document, &restored_vector, allocator.get());
            REQUIRE(restored_vector.len_ == document_terms[document].size());
            for (uint32_t position = 0; position < restored_vector.len_; ++position) {
                REQUIRE(restored_vector.ids_[position] == document_terms[document][position]);
            }
            allocator->Deallocate(restored_vector.ids_);
            allocator->Deallocate(restored_vector.vals_);
        }
    }
}
