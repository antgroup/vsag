
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

#pragma once

#include "hgraph.h"
#include "vsag/binaryset.h"

namespace vsag {

// Forward declarations for DiskANN internal structures
namespace diskann {
template <typename T, typename TagT, typename LabelT>
class Index;
}

/**
 * HGraphDiskANNLoader - A specialized HGraph index that can load existing DiskANN format data.
 *
 * This class inherits from HGraph and provides the ability to deserialize DiskANN index
 * format and convert it to HGraph's in-memory format. It enables migration from DiskANN
 * to HGraph without rebuilding indexes.
 *
 * Key mappings:
 * - DiskANN PQ compressed vectors -> HGraph basic_flatten_codes_ (PQ quantizer)
 * - DiskANN Vamana graph -> HGraph bottom_graph_ (single-level index)
 * - DiskANN tags -> HGraph label_table_
 * - DiskANN precise vectors -> HGraph high_precise_codes_ (if available)
 */
class HGraphDiskANNLoader : public HGraph {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    HGraphDiskANNLoader(const HGraphParameterPtr& param, const IndexCommonParam& common_param);

    HGraphDiskANNLoader(const ParamPtr& param, const IndexCommonParam& common_param)
        : HGraphDiskANNLoader(std::dynamic_pointer_cast<HGraphParameter>(param), common_param){};

    ~HGraphDiskANNLoader() override = default;

    /**
     * Override Deserialize to handle DiskANN BinarySet format
     */
    void
    Deserialize(const BinarySet& binary_set) override;

    /**
     * Override Deserialize to handle DiskANN ReaderSet format
     */
    void
    Deserialize(const ReaderSet& reader_set) override;

    /**
     * Get the name of this index type
     */
    [[nodiscard]] std::string
    GetName() const override {
        return INDEX_TYPE_HGRAPH_DISKANN_LOADER;
    }

    /**
     * Get the index type
     */
    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::HGRAPH_DISKANN_LOADER;
    }

public:
    // DiskANN-specific parameters extracted during loading (public for testing)
    uint64_t diskann_num_points_{0};
    uint64_t diskann_dim_{0};
    uint64_t diskann_max_degree_{0};
    bool has_precise_vectors_{false};
    bool loaded_from_diskann_{false};

    // Store common param for creating FlattenInterface instances
    IndexCommonParam common_param_;

    /**
     * Load PQ data (pivots and compressed vectors) from DiskANN format
     */
    void
    load_diskann_pq_data(std::istream& pq_stream,
                         std::istream& compressed_stream,
                         uint64_t num_points);

    /**
     * Load graph structure from DiskANN format
     */
    void
    load_diskann_graph(std::istream& graph_stream, uint64_t num_points);

    /**
     * Extract precise vectors from DiskANN disk layout
     */
    void
    load_diskann_precise_vectors(std::istream& layout_stream,
                                 uint64_t num_points,
                                 uint64_t dim,
                                 uint64_t max_degree);

    // Accessor methods for HGraph members
    FlattenInterfacePtr
    get_basic_flatten_codes() const {
        return this->basic_flatten_codes_;
    }

    GraphInterfacePtr
    get_bottom_graph() const {
        return this->bottom_graph_;
    }

    InnerIdType
    get_entry_point_id() const {
        return this->entry_point_id_;
    }

    FlattenInterfacePtr
    get_high_precise_codes() const {
        return this->high_precise_codes_;
    }

    void
    set_use_reorder(bool use_reorder) {
        this->use_reorder_ = use_reorder;
    }

private:
    /**
     * Build HGraph parameters from DiskANN format parameters
     */
    static ParamPtr
    build_hgraph_param_from_diskann_param(const JsonType& external_param,
                                          const IndexCommonParam& common_param);

    /**
     * Main entry point for loading DiskANN format from BinarySet
     */
    void
    load_from_diskann(const BinarySet& binary_set);

    /**
     * Main entry point for loading DiskANN format from ReaderSet
     */
    void
    load_from_diskann(const ReaderSet& reader_set);

    /**
     * Load tags (ID mapping) from DiskANN format
     */
    void
    load_diskann_tags(std::istream& tag_stream);

    /**
     * Parse PQ pivot file header and setup ProductQuantizer
     */
    void
    setup_product_quantizer(std::istream& pq_stream,
                            uint32_t& num_subspaces,
                            uint32_t& num_centroids_per_subspace,
                            uint32_t& subspace_dim);

    /**
     * Read compressed vectors from DiskANN format
     */
    void
    read_compressed_vectors(std::istream& compressed_stream,
                            uint64_t num_points,
                            uint32_t num_subspaces);

    /**
     * Finalize loading: validate and initialize structures
     */
    void
    finalize_loading();
};

}  // namespace vsag
