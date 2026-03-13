
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
    // Test-only accessor for unit testing
    void
    TestLoadDiskANNPQData(std::istream& pq_stream,
                          std::istream& compressed_stream,
                          uint64_t num_points) {
        LoadDiskANNPQData(pq_stream, compressed_stream, num_points);
    }

    // Test-only accessor to get the flatten codes interface
    FlattenInterfacePtr
    TestGetFlattenCodes() const {
        return this->basic_flatten_codes_;
    }

    // Test-only accessor for LoadDiskANNGraph
    void
    TestLoadDiskANNGraph(std::istream& graph_stream, uint64_t num_points) {
        LoadDiskANNGraph(graph_stream, num_points);
    }

    // Test-only accessor to get the bottom graph
    GraphInterfacePtr
    TestGetBottomGraph() const {
        return this->bottom_graph_;
    }

    // Test-only accessor to get entry point
    InnerIdType
    TestGetEntryPoint() const {
        return this->entry_point_id_;
    }

    // Test-only accessor to get max degree from loading
    uint64_t
    TestGetDiskANNMaxDegree() const {
        return this->diskann_max_degree_;
    }

    // Test-only accessor for LoadDiskANNPreciseVectors
    void
    TestLoadDiskANNPreciseVectors(std::istream& layout_stream,
                                  uint64_t num_points,
                                  uint64_t dim,
                                  uint64_t max_degree) {
        LoadDiskANNPreciseVectors(layout_stream, num_points, dim, max_degree);
    }

    // Test-only accessor to get precise codes
    FlattenInterfacePtr
    TestGetPreciseCodes() const {
        return this->high_precise_codes_;
    }

    // Test-only accessor to set use_reorder flag
    void
    TestSetUseReorder(bool use_reorder) {
        this->use_reorder_ = use_reorder;
    }

private:
    /**
     * Build HGraph parameters from DiskANN format parameters
     */
    static ParamPtr
    BuildHGraphParamFromDiskANNParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param);

    /**
     * Main entry point for loading DiskANN format from BinarySet
     */
    void
    LoadFromDiskANN(const BinarySet& binary_set);

    /**
     * Main entry point for loading DiskANN format from ReaderSet
     */
    void
    LoadFromDiskANN(const ReaderSet& reader_set);

    /**
     * Load PQ data (pivots and compressed vectors) from DiskANN format
     */
    void
    LoadDiskANNPQData(std::istream& pq_stream,
                      std::istream& compressed_stream,
                      uint64_t num_points);

    /**
     * Load graph structure from DiskANN format
     */
    void
    LoadDiskANNGraph(std::istream& graph_stream, uint64_t num_points);

    /**
     * Load tags (ID mapping) from DiskANN format
     */
    void
    LoadDiskANNTags(std::istream& tag_stream);

    /**
     * Extract precise vectors from DiskANN disk layout
     */
    void
    LoadDiskANNPreciseVectors(std::istream& layout_stream,
                              uint64_t num_points,
                              uint64_t dim,
                              uint64_t max_degree);

    /**
     * Parse PQ pivot file header and setup ProductQuantizer
     */
    void
    SetupProductQuantizer(std::istream& pq_stream,
                          uint32_t& num_subspaces,
                          uint32_t& num_centroids_per_subspace,
                          uint32_t& subspace_dim);

    /**
     * Read compressed vectors from DiskANN format
     */
    void
    ReadCompressedVectors(std::istream& compressed_stream,
                          uint64_t num_points,
                          uint32_t num_subspaces);

private:
    // DiskANN-specific parameters extracted during loading
    uint64_t diskann_num_points_{0};
    uint64_t diskann_dim_{0};
    uint64_t diskann_max_degree_{0};
    bool has_precise_vectors_{false};

    // Track whether this was loaded from DiskANN format
    bool loaded_from_diskann_{false};

    // Store common param for creating FlattenInterface instances
    IndexCommonParam common_param_;
};

}  // namespace vsag
