// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mci_searcher.h"

namespace vsag {

class MCILiveSearcher {
public:
    explicit MCILiveSearcher(const IndexCommonParam& common_param);

    DistHeapPtr
    Search(const CliqueDataCellPtr& cliques,
           const FlattenInterfacePtr& flatten,
           const void* query,
           const InnerSearchParam& inner_search_param,
           const MCISearcherParam& mci_param,
           QueryContext* ctx) const;

    explicit MCILiveSearcher(Allocator* allocator) : allocator_(allocator) {
    }

private:
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
