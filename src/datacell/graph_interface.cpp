
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

#include "graph_interface.h"

#include "compressed_graph_datacell.h"
#include "graph_datacell.h"
#include "io/cache_io/cache_io_parameter.h"
#include "io/io_headers.h"
#include "sparse_graph_datacell.h"

namespace vsag {

void
GraphInterface::UpdateReverseEdges(InnerIdType id,
                                   const Vector<InnerIdType>& old_neighbors,
                                   const Vector<InnerIdType>& new_neighbors) {
    if (reverse_edges_) {
        UnorderedSet<InnerIdType> old_set(allocator_);
        UnorderedSet<InnerIdType> new_set(allocator_);
        for (const auto& n : old_neighbors) {
            old_set.insert(n);
        }
        for (const auto& n : new_neighbors) {
            new_set.insert(n);
        }
        for (const auto& old_n : old_neighbors) {
            if (new_set.find(old_n) == new_set.end()) {
                reverse_edges_->RemoveReverseEdge(id, old_n);
            }
        }
        for (const auto& new_n : new_neighbors) {
            if (old_set.find(new_n) == old_set.end()) {
                reverse_edges_->AddReverseEdge(id, new_n);
            }
        }
    }
}

GraphInterfacePtr
GraphInterface::MakeInstance(const GraphInterfaceParamPtr& graph_param,
                             const IndexCommonParam& common_param) {
    switch (graph_param->graph_storage_type_) {
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_SPARSE:
            return std::make_shared<SparseGraphDataCell>(graph_param, common_param);
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED:
            return std::make_shared<CompressedGraphDataCell>(graph_param, common_param);
        case GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT:
            auto io_string = std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param)
                                 ->io_parameter_->GetTypeName();
            if (io_string == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
                return std::make_shared<GraphDataCell<MemoryBlockIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_MEMORY_IO) {
                return std::make_shared<GraphDataCell<MemoryIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_MMAP_IO) {
                return std::make_shared<GraphDataCell<MMapIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_BUFFER_IO) {
                return std::make_shared<GraphDataCell<BufferIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_ASYNC_IO) {
                return std::make_shared<GraphDataCell<AsyncIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_URING_IO) {
                return std::make_shared<GraphDataCell<UringIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_READER_IO) {
                return std::make_shared<GraphDataCell<ReaderIO>>(graph_param, common_param);
            }
            if (io_string == IO_TYPE_VALUE_CACHE_IO) {
                auto graph_dc_param =
                    std::dynamic_pointer_cast<GraphDataCellParameter>(graph_param);
                if (not graph_dc_param) {
                    return nullptr;
                }
                auto cache_param =
                    std::dynamic_pointer_cast<CacheIOParameter>(graph_dc_param->io_parameter_);
                if (cache_param) {
                    auto inner_type = cache_param->inner_io_type_;
                    if (inner_type == IO_TYPE_VALUE_MMAP_IO) {
                        return std::make_shared<GraphDataCell<CacheIO<MMapIO>>>(graph_param,
                                                                                common_param);
                    }
                    if (inner_type == IO_TYPE_VALUE_BUFFER_IO) {
                        return std::make_shared<GraphDataCell<CacheIO<BufferIO>>>(graph_param,
                                                                                  common_param);
                    }
                    if (inner_type == IO_TYPE_VALUE_ASYNC_IO) {
#if HAVE_LIBAIO
                        return std::make_shared<GraphDataCell<CacheIO<AsyncIO>>>(graph_param,
                                                                                 common_param);
#else
                        return std::make_shared<GraphDataCell<CacheIO<BufferIO>>>(graph_param,
                                                                                  common_param);
#endif
                    }
                    if (inner_type == IO_TYPE_VALUE_MEMORY_IO) {
                        return std::make_shared<GraphDataCell<CacheIO<MemoryIO>>>(graph_param,
                                                                                  common_param);
                    }
                    if (inner_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
                        return std::make_shared<GraphDataCell<CacheIO<MemoryBlockIO>>>(
                            graph_param, common_param);
                    }
                    if (inner_type == IO_TYPE_VALUE_READER_IO) {
                        return std::make_shared<GraphDataCell<CacheIO<ReaderIO>>>(graph_param,
                                                                                  common_param);
                    }
                    throw VsagException(
                        ErrorType::INVALID_ARGUMENT,
                        std::string("Unsupported CacheIO inner_io_type: ") + inner_type);
                }
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    "CacheIO requires CacheIOParameter");
            }
            return nullptr;
    }
    return nullptr;
}
}  // namespace vsag
