// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "hgraph_pipnn_add.h"

#include <array>
#include <future>

#include "datacell/flatten_code_writer.h"
#include "hgraph_fast_build.h"
#include "pipnn_storage_build.h"
#include "utils/byte_buffer.h"

namespace vsag {

FlattenInterfacePtr
HGraph::PiPNNAdd::DistanceCodes(const HGraph& graph) {
    return DistanceCodes(graph,
                         graph.basic_flatten_codes_,
                         graph.has_precise_reorder() ? graph.high_precise_codes_ : nullptr,
                         graph.raw_vector_);
}

FlattenInterfacePtr
HGraph::PiPNNAdd::DistanceCodes(const HGraph& graph,
                                const FlattenInterfacePtr& base,
                                const FlattenInterfacePtr& precise,
                                const FlattenInterfacePtr& raw) {
    if (graph.build_by_base_) {
        return base;
    }
    if (precise) {
        return precise;
    }
    if (graph.graph_type_ == GRAPH_TYPE_VALUE_PIPNN && graph.pipnn_param_.add_use_codes && raw &&
        raw->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32 &&
        !base->SupportSplitCodeStorage() &&
        GetBaseQuantizerName(base->GetQuantizerName()) == QUANTIZATION_TYPE_VALUE_RABITQ) {
        return raw;
    }
    return base;
}

FlattenInterfacePtr
HGraph::PiPNNAdd::ExactStore(const HGraph& graph, int64_t source) {
    if (graph.support_duplicate_) {
        return nullptr;
    }
    // A cosine FP32 code without its mold contains normalized coordinates.
    const auto exact = [&](const FlattenInterfacePtr& store) {
        return store && (!graph.ignore_reorder_ || store != graph.high_precise_codes_) &&
               (source || store->CanBorrowRowsUntilResize()) &&
               store->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32 &&
               (graph.metric_ != MetricType::METRIC_TYPE_COSINE ||
                store->GetQuantizerCodeSize() == (graph.dim_ + 1) * sizeof(float));
    };
    if (((source == 0) || source == BASE_CODE) && exact(graph.basic_flatten_codes_)) {
        return graph.basic_flatten_codes_;
    }
    if (((source == 0) || source == PRECISE_CODE) && graph.has_precise_reorder() &&
        exact(graph.high_precise_codes_)) {
        return graph.high_precise_codes_;
    }
    if (((source == 0) || source == RAW_CODE) && exact(graph.raw_vector_)) {
        return graph.raw_vector_;
    }
    return nullptr;
}

bool
HGraph::PiPNNAdd::TryWriteRaBitQCodes(HGraph& graph,
                                      const Vector<InnerIdType>& ids,
                                      const Vector<const float*>& vectors) {
    constexpr size_t block_size = 64;
    const size_t workers =
        std::min<size_t>(graph.build_thread_count_, (ids.size() + block_size - 1) / block_size);
    if (workers < 2 || graph.thread_pool_ == nullptr || graph.thread_pool_->IsCurrentThread()) {
        return false;
    }

    // Tune takes graph.add_mutex_ exclusively. Pin the trained encoders and their layout until
    // every block has committed, without serializing the expensive Encode calls.
    std::shared_lock add_lock(graph.add_mutex_);
    if (GetBaseQuantizerName(graph.basic_flatten_codes_->GetQuantizerName()) !=
        QUANTIZATION_TYPE_VALUE_RABITQ) {
        return false;
    }
    struct Store {
        FlattenInterface* cell;
        FlattenCodeWriter* writer;
        size_t width;
    };
    std::array<Store, 3> stores{};
    size_t count = 0;
    auto add_store = [&](const FlattenInterfacePtr& cell) {
        auto* writer = dynamic_cast<FlattenCodeWriter*>(cell.get());
        if (writer == nullptr || !writer->CanInsertEncodedRange()) {
            return false;
        }
        stores[count++] = {cell.get(), writer, cell->GetQuantizerCodeSize()};
        return true;
    };
    if (!add_store(graph.basic_flatten_codes_) ||
        (graph.has_precise_reorder() && !add_store(graph.high_precise_codes_)) ||
        (graph.create_new_raw_vector_ && !add_store(graph.raw_vector_))) {
        return false;
    }

    auto run = [&](size_t worker) {
        std::array<std::unique_ptr<ByteBuffer>, 3> buffers;
        for (size_t j = 0; j < count; ++j) {
            buffers[j] =
                std::make_unique<ByteBuffer>(block_size * stores[j].width, graph.allocator_);
        }
        for (size_t begin = worker * block_size; begin < ids.size();
             begin += workers * block_size) {
            const size_t end = std::min(begin + block_size, ids.size());
            for (size_t j = 0; j < count; ++j) {
                for (size_t i = begin; i < end; ++i) {
                    CHECK_ARGUMENT(
                        stores[j].cell->Encode(vectors[i],
                                               buffers[j]->data + (i - begin) * stores[j].width),
                        "failed to encode PiPNN Add vector");
                }
            }
            std::unique_lock codes_lock(graph.persistent_codes_mutex_);
            for (size_t first = begin; first < end;) {
                size_t next = first + 1;
                while (next < end && ids[next] == ids[next - 1] + 1) {
                    ++next;
                }
                for (size_t j = 0; j < count; ++j) {
                    stores[j].writer->InsertEncodedRange(
                        ids[first],
                        next - first,
                        buffers[j]->data + (first - begin) * stores[j].width);
                }
                first = next;
            }
            for (size_t i = begin; i < end; ++i) {
                graph.finish_code_write(ids[i], BASE_CODE | PRECISE_CODE | RAW_CODE);
            }
        }
    };
    std::vector<std::future<void>> futures;
    HGraphBuildTaskGuard guard(futures, workers);
    for (size_t i = 0; i < workers; ++i) {
        futures.emplace_back(graph.thread_pool_->GeneralEnqueue(run, i));
    }
    for (auto& future : futures) {
        future.get();
    }
    return true;
}

std::vector<int64_t>
HGraph::PiPNNAdd::Add(HGraph& graph, const DatasetPtr& data) {
    graph.validate_add_data(data);
    const bool supports_incremental_mutation = !graph.support_force_remove() &&
                                               !graph.rabitq_fused_datacell_ &&
                                               !graph.use_conjugate_graph_;
    CHECK_ARGUMENT(supports_incremental_mutation,
                   "unsupported mutation mode for incremental PiPNN");
    auto batch = graph.prepare_add_batch(data, true);
    if (batch.rows.empty()) {
        return batch.failed_ids;
    }
    Vector<InnerIdType> ids(graph.allocator_);
    Vector<const float*> vectors(graph.allocator_);
    graph.prepare_build_codes(data, batch.rows);
    for (const auto& row : batch.rows) {
        const auto* v = static_cast<const float*>(graph.get_data(data, row.input_idx));
        ids.push_back(row.inner_id);
        vectors.push_back(v);
        if (graph.extra_infos_) {
            graph.extra_infos_->InsertExtraInfo(
                data->GetExtraInfos() + row.input_idx * graph.extra_info_size_, row.inner_id);
        }
        if (graph.use_attribute_filter_ && (data->GetAttributeSets() != nullptr)) {
            graph.attr_filter_index_->Insert(data->GetAttributeSets()[row.input_idx], row.inner_id);
        }
    }
    if (!graph.using_dedup_storage() && !TryWriteRaBitQCodes(graph, ids, vectors)) {
        for (size_t i = 0; i < ids.size(); ++i) {
            graph.insert_persistent_codes(vectors[i], ids[i]);
        }
    }
    std::unique_lock global(graph.global_mutex_);
    auto parameters = graph.pipnn_param_;
    parameters.alpha = graph.alpha_;
    PiPNNGraphBuilder builder(parameters,
                              graph.dim_,
                              graph.metric_,
                              graph.allocator_,
                              graph.thread_pool_.get(),
                              graph.build_thread_count_);
    auto distance_codes = DistanceCodes(graph);
    // ReaderSet attaches its reader after deserialization. A v3 snapshot may
    // refer to that store; fetch old rows on the first writable Add if necessary.
    if (exact_rows && !exact_rows->Store()->CanBorrowRowsUntilResize()) {
        exact_rows->Materialize(state, graph.dim_, graph.allocator_);
        exact_rows.reset();
    }
    if (parameters.add_use_codes) {
        builder.UpdateWithCodes(graph.bottom_graph_, ids, vectors, state, *distance_codes);
    } else if (exact_rows) {
        auto rows = exact_rows->Prepare(state, ids, graph.max_capacity_.load());
        builder.UpdateFromSharedRows(graph.bottom_graph_, ids, rows, state);
    } else {
        builder.Update(graph.bottom_graph_, ids, vectors, state);
    }
    if (graph.using_dedup_storage()) {
        Vector<int64_t> positions(graph.allocator_);
        for (const auto& row : batch.rows) {
            positions.push_back(row.input_idx);
        }
        materialize_pipnn_storage(
            ids,
            positions,
            graph.bottom_graph_,
            *graph.code_slot_map_,
            graph.allocator_,
            graph.thread_pool_.get(),
            graph.build_thread_count_,
            [&](auto n) { graph.ensure_physical_code_capacity_unlocked(n); },
            [&](auto i, auto slot) {
                graph.insert_persistent_codes_to_slot(graph.get_data(data, i), slot);
            });
    }
    AddContext context;
    if (parameters.add_use_codes) {
        context.graph_read_codes = distance_codes;
    } else {
        context.need_temporary_sq8_build_data = graph.need_temporary_sq8_build_data_for_add();
        graph.prepare_graph_read_codes(data, context);
    }
    for (const auto& row : batch.rows) {
        if (row.level >= 0 && graph.bottom_graph_->GetGroupId(row.inner_id) == row.inner_id) {
            bool highest = row.level >= static_cast<int>(graph.route_graphs_.size());
            graph.ensure_route_graphs_for_level(row.level);
            InnerSearchParam param;
            param.ep = graph.entry_point_id_;
            param.ef = graph.ef_construct_;
            param.topk = static_cast<int64_t>(graph.ef_construct_);
            graph.publish_unique_to_route_graphs(graph.get_data(data, row.input_idx),
                                                 row.level,
                                                 row.inner_id,
                                                 param,
                                                 context.graph_read_codes);
            if (highest) {
                graph.entry_point_id_ = row.inner_id;
            }
        }
        const auto* source_ids = data->GetSourceID();
        {
            std::scoped_lock label_lock(graph.label_lookup_mutex_);
            graph.label_table_->Insert(row.inner_id, data->GetIds()[row.input_idx]);
            if ((source_ids != nullptr) && !source_ids[row.input_idx].empty()) {
                graph.label_table_->InsertSourceId(row.inner_id, source_ids[row.input_idx]);
            }
        }
        graph.PublishNode(row.inner_id);
    }
    graph.mark_memory_usage_dirty();
    return batch.failed_ids;
}

}  // namespace vsag
