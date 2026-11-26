// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hgraph.h"
#include "hgraph_pipnn_rows.h"

namespace vsag {

// Exists only for graphs built or restored with incremental PiPNN state.
struct HGraph::PiPNNAdd {
    PiPNNIncrementalState state;
    std::unique_ptr<HGraphPiPNNRows> exact_rows;

    std::vector<int64_t>
    Add(HGraph& graph, const DatasetPtr& data);

    static FlattenInterfacePtr
    DistanceCodes(const HGraph& graph);
    static FlattenInterfacePtr
    DistanceCodes(const HGraph& graph,
                  const FlattenInterfacePtr& base,
                  const FlattenInterfacePtr& precise,
                  const FlattenInterfacePtr& raw);
    static FlattenInterfacePtr
    ExactStore(const HGraph& graph, int64_t source = 0);

    static std::unique_ptr<PiPNNAdd>
    Read(const HGraph& graph,
         StreamReader& reader,
         const MetadataPtr& metadata,
         uint64_t count,
         bool forward_only = false);
    void
    Write(StreamWriter& writer, bool use_codes) const;
    void
    WriteMetadata(Metadata& metadata, const HGraph& graph) const;

private:
    int64_t
    SnapshotVersion(bool use_codes) const;
    static bool
    TryWriteRaBitQCodes(HGraph& graph,
                        const Vector<InnerIdType>& ids,
                        const Vector<const float*>& vectors);
};
}  // namespace vsag
