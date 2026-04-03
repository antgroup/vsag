
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

#include "hgraph_diskann_loader.h"

#include <fmt/format.h>

#include <cstring>
#include <memory>

#include "common.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "hgraph_parameter.h"
#include "impl/logger/logger.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/memory_io.h"
#include "io/memory_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "quantization/product_quantization/product_quantizer_parameter.h"
#include "vsag/constants.h"

namespace vsag {

// DiskANN file format constants
constexpr uint32_t DISKANN_PQ_HEADER_VERSION = 1;
constexpr uint32_t DISKANN_GRAPH_SLACK = 1;

// Default sector size used by DiskANN
constexpr uint64_t DISKANN_SECTOR_LEN = 4096;

namespace {

// Helper: convert Binary to stringstream
std::stringstream
binary_to_stream(const Binary& binary) {
    std::stringstream stream;
    stream.write(reinterpret_cast<const char*>(binary.data.get()),
                 static_cast<std::streamsize>(binary.size));
    stream.seekg(0);
    return stream;
}

// Helper: convert Reader to stringstream
std::stringstream
reader_to_stream(const ReaderPtr& reader) {
    auto data = std::make_unique<char[]>(reader->Size());
    reader->Read(0, reader->Size(), data.get());
    std::stringstream stream;
    stream.write(data.get(), static_cast<std::streamsize>(reader->Size()));
    stream.seekg(0);
    return stream;
}

// Helper: read num_points from tag stream
int32_t
read_num_points_from_tag_stream(std::istream& tag_stream) {
    int32_t num_points = 0;
    int32_t dim = 0;
    tag_stream.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    tag_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    tag_stream.seekg(0);
    return num_points;
}

// Helper template for PQ quantizer operations
template <MetricType metric>
ProductQuantizer<metric>*
GetPQQuantizer(void* quantizer_ptr) {
    return static_cast<ProductQuantizer<metric>*>(quantizer_ptr);
}

}  // namespace

ParamPtr
HGraphDiskANNLoader::CheckAndMappingExternalParam(const JsonType& external_param,
                                                  const IndexCommonParam& common_param) {
    // Check if this is a pure DiskANN format parameter (has "diskann" key but no "index_param" key)
    if (external_param.Contains(INDEX_DISKANN) && !external_param.Contains(INDEX_PARAM)) {
        const auto& diskann_json = external_param[INDEX_DISKANN];

        // Check use_opq - not supported
        if (diskann_json.Contains(DISKANN_PARAMETER_USE_OPQ) &&
            diskann_json[DISKANN_PARAMETER_USE_OPQ].GetBool()) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "OPQ (Optimized Product Quantization) is not supported in HGraphDiskANNLoader");
        }

        // Check use_bsa - not supported
        if (diskann_json.Contains(DISKANN_PARAMETER_USE_BSA) &&
            diskann_json[DISKANN_PARAMETER_USE_BSA].GetBool()) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "BSA (Balanced Sampling Algorithm) is not supported in HGraphDiskANNLoader");
        }

        // Build HGraph parameters from DiskANN parameters
        return build_hgraph_param_from_diskann_param(external_param, common_param);
    }

    // For HGraph format parameters (with "index_param" key or without "diskann" key),
    // extract the "index_param" content and pass to parent class
    if (external_param.Contains(INDEX_PARAM)) {
        return HGraph::CheckAndMappingExternalParam(external_param[INDEX_PARAM], common_param);
    }

    // If neither "diskann" nor "index_param" key exists, pass full parameters to parent
    return HGraph::CheckAndMappingExternalParam(external_param, common_param);
}

ParamPtr
HGraphDiskANNLoader::build_hgraph_param_from_diskann_param(const JsonType& external_param,
                                                           const IndexCommonParam& common_param) {
    const auto& diskann_json = external_param[INDEX_DISKANN];

    // Get DiskANN parameters with defaults
    int64_t max_degree = 64;
    if (diskann_json.Contains(DISKANN_PARAMETER_R)) {
        max_degree = diskann_json[DISKANN_PARAMETER_R].GetInt();
    }

    int64_t ef_construction = 200;
    if (diskann_json.Contains(DISKANN_PARAMETER_L)) {
        ef_construction = diskann_json[DISKANN_PARAMETER_L].GetInt();
    }

    int64_t pq_dims = 16;
    if (diskann_json.Contains(DISKANN_PARAMETER_DISK_PQ_DIMS)) {
        pq_dims = diskann_json[DISKANN_PARAMETER_DISK_PQ_DIMS].GetInt();
    }

    std::string graph_type = GRAPH_TYPE_VALUE_NSW;
    if (diskann_json.Contains(DISKANN_PARAMETER_GRAPH_TYPE)) {
        graph_type = diskann_json[DISKANN_PARAMETER_GRAPH_TYPE].GetString();
    }

    // Validate parameters
    auto max_degree_threshold = std::max<int64_t>(common_param.dim_, 128);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    CHECK_ARGUMENT(
        max_degree >= 4 && max_degree <= max_degree_threshold,
        fmt::format("max_degree({}) must be in range [4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold =
        std::max<uint64_t>(1000UL, 100 * static_cast<uint64_t>(max_degree));
    // NOLINTNEXTLINE(readability-simplify-boolean-expr)
    CHECK_ARGUMENT(max_degree <= ef_construction &&
                       ef_construction <= static_cast<int64_t>(construction_threshold),
                   fmt::format("ef_construction({}) must be in range [{}, {}]",
                               ef_construction,
                               max_degree,
                               construction_threshold));

    // Validate pq_dims divides dim
    if (common_param.dim_ > 0 && common_param.dim_ % pq_dims != 0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("pq_dims({}) must divide dim({})", pq_dims, common_param.dim_));
    }

    // Build HGraph parameter JSON
    JsonType hgraph_json;

    // Base codes: PQ quantization with memory I/O
    hgraph_json[BASE_CODES_KEY][CODES_TYPE_KEY].SetString(FLATTEN_CODES);
    hgraph_json[BASE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetString(IO_TYPE_VALUE_MEMORY_IO);
    hgraph_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetString(
        QUANTIZATION_TYPE_VALUE_PQ);
    hgraph_json[BASE_CODES_KEY][QUANTIZATION_PARAMS_KEY][PRODUCT_QUANTIZATION_DIM_KEY].SetInt(
        pq_dims);

    // Graph: flat storage with memory I/O
    hgraph_json[GRAPH_KEY][GRAPH_STORAGE_TYPE_KEY].SetString(GRAPH_STORAGE_TYPE_VALUE_FLAT);
    hgraph_json[GRAPH_KEY][IO_PARAMS_KEY][TYPE_KEY].SetString(IO_TYPE_VALUE_MEMORY_IO);
    hgraph_json[GRAPH_KEY][GRAPH_PARAM_MAX_DEGREE_KEY].SetInt(max_degree);
    hgraph_json[GRAPH_KEY][GRAPH_TYPE_KEY].SetString(graph_type);

    // Precise codes: FP32 quantization with memory I/O (DiskANN always has precise vectors)
    hgraph_json[PRECISE_CODES_KEY][CODES_TYPE_KEY].SetString(FLATTEN_CODES);
    hgraph_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetString(IO_TYPE_VALUE_MEMORY_IO);
    hgraph_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetString(
        QUANTIZATION_TYPE_VALUE_FP32);

    // HGraph-specific parameters
    hgraph_json[USE_REORDER_KEY].SetBool(
        true);  // DiskANN format always has precise vectors for reorder
    hgraph_json[EF_CONSTRUCTION_KEY].SetInt(ef_construction);

    // Create HGraphParameter from the constructed JSON
    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(hgraph_json);

    return hgraph_parameter;
}

HGraphDiskANNLoader::HGraphDiskANNLoader(const HGraphParameterPtr& param,
                                         const IndexCommonParam& common_param)
    : HGraph(param, common_param), common_param_(common_param) {
    // Mark as not loaded yet
    loaded_from_diskann_ = false;
}

void
HGraphDiskANNLoader::Deserialize(const BinarySet& binary_set) {
    // Check if this is a DiskANN format
    if (binary_set.Contains(DISKANN_PQ) || binary_set.Contains(DISKANN_LAYOUT_FILE)) {
        load_from_diskann(binary_set);
        loaded_from_diskann_ = true;
        return;
    }
    // This loader is specifically for DiskANN format
    // For HGraph format, use the regular HGraph index instead
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        "HGraphDiskANNLoader only supports DiskANN format data. "
                        "Use 'hgraph' index type for HGraph format data.");
}

void
HGraphDiskANNLoader::Deserialize(const ReaderSet& reader_set) {
    // Check if this is a DiskANN format
    if (reader_set.Contains(DISKANN_PQ) || reader_set.Contains(DISKANN_LAYOUT_FILE)) {
        load_from_diskann(reader_set);
        loaded_from_diskann_ = true;
        return;
    }
    // This loader is specifically for DiskANN format
    // For HGraph format, use the regular HGraph index instead
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        "HGraphDiskANNLoader only supports DiskANN format data. "
                        "Use 'hgraph' index type for HGraph format data.");
}

void
HGraphDiskANNLoader::load_from_diskann(const BinarySet& binary_set) {
    // Validate required components
    if (!binary_set.Contains(DISKANN_PQ) || !binary_set.Contains(DISKANN_COMPRESSED_VECTOR) ||
        !binary_set.Contains(DISKANN_LAYOUT_FILE) || !binary_set.Contains(DISKANN_TAG_FILE)) {
        throw VsagException(
            ErrorType::MISSING_FILE,
            "DiskANN format requires PQ, compressed vectors, layout file, and tags");
    }

    // Get the binary data and convert to streams
    auto pq_stream = binary_to_stream(binary_set.Get(DISKANN_PQ));
    auto compressed_stream = binary_to_stream(binary_set.Get(DISKANN_COMPRESSED_VECTOR));
    auto layout_stream = binary_to_stream(binary_set.Get(DISKANN_LAYOUT_FILE));
    auto tag_stream = binary_to_stream(binary_set.Get(DISKANN_TAG_FILE));

    // Read num_points from tags file
    diskann_num_points_ = static_cast<uint64_t>(read_num_points_from_tag_stream(tag_stream));
    this->use_reorder_ = true;

    // Load components
    load_diskann_pq_data(pq_stream, compressed_stream, diskann_num_points_);
    load_diskann_tags(tag_stream);

    // Load graph FIRST to get diskann_max_degree_ before loading precise vectors
    if (binary_set.Contains(DISKANN_GRAPH)) {
        auto graph_stream = binary_to_stream(binary_set.Get(DISKANN_GRAPH));
        load_diskann_graph(graph_stream, diskann_num_points_);
    }

    load_diskann_precise_vectors(
        layout_stream, diskann_num_points_, diskann_dim_, diskann_max_degree_);

    finalize_loading();
}

void
HGraphDiskANNLoader::load_from_diskann(const ReaderSet& reader_set) {
    // Validate required components
    if (!reader_set.Contains(DISKANN_PQ) || !reader_set.Contains(DISKANN_COMPRESSED_VECTOR) ||
        !reader_set.Contains(DISKANN_LAYOUT_FILE) || !reader_set.Contains(DISKANN_TAG_FILE)) {
        throw VsagException(
            ErrorType::MISSING_FILE,
            "DiskANN format requires PQ, compressed vectors, layout file, and tags");
    }

    // Load tags first and reuse the stream for num_points
    auto tag_stream = reader_to_stream(reader_set.Get(DISKANN_TAG_FILE));
    diskann_num_points_ = static_cast<uint64_t>(read_num_points_from_tag_stream(tag_stream));
    this->use_reorder_ = true;

    // Load PQ data
    {
        auto pq_stream = reader_to_stream(reader_set.Get(DISKANN_PQ));
        auto compressed_stream = reader_to_stream(reader_set.Get(DISKANN_COMPRESSED_VECTOR));
        load_diskann_pq_data(pq_stream, compressed_stream, diskann_num_points_);
    }

    // Load tags (reuse the already-read tag_stream)
    load_diskann_tags(tag_stream);

    // Load graph FIRST to get diskann_max_degree_
    if (reader_set.Contains(DISKANN_GRAPH)) {
        auto graph_stream = reader_to_stream(reader_set.Get(DISKANN_GRAPH));
        load_diskann_graph(graph_stream, diskann_num_points_);
    }

    // Load precise vectors
    {
        auto layout_stream = reader_to_stream(reader_set.Get(DISKANN_LAYOUT_FILE));
        load_diskann_precise_vectors(
            layout_stream, diskann_num_points_, diskann_dim_, diskann_max_degree_);
    }

    finalize_loading();
}

void
HGraphDiskANNLoader::finalize_loading() {
    this->total_count_ = diskann_num_points_;

    // Validate max_degree consistency
    if (diskann_max_degree_ > 0 && this->bottom_graph_ != nullptr) {
        uint64_t configured_max_degree = this->bottom_graph_->MaximumDegree();
        if (diskann_max_degree_ > configured_max_degree) {
            logger::warn(
                "max_degree mismatch: DiskANN index has max_degree={}, but config "
                "expects max_degree={}. Neighbors will be truncated to fit the configuration.",
                diskann_max_degree_,
                configured_max_degree);
        }
        if (diskann_max_degree_ < configured_max_degree) {
            logger::info(
                "DiskANN index max_degree ({}) is less than configured max_degree ({}). "
                "The index will work but with potentially lower recall.",
                diskann_max_degree_,
                configured_max_degree);
        }
    }

    // Validate entry point
    if (this->entry_point_id_ >= static_cast<InnerIdType>(diskann_num_points_)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Invalid entry point {}: exceeds num_points {}",
                                        this->entry_point_id_,
                                        diskann_num_points_));
    }

    // Initialize structures
    this->init_resize_bit_and_reorder();
    this->resize(this->bottom_graph_->max_capacity_);
    this->neighbors_mutex_->Resize(this->bottom_graph_->max_capacity_);
    pool_ = std::make_shared<VisitedListPool>(1, allocator_, diskann_num_points_, allocator_);
    this->mult_ = 1.0 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));
    loaded_from_diskann_ = true;
}

void
HGraphDiskANNLoader::load_diskann_pq_data(std::istream& pq_stream,
                                          std::istream& compressed_stream,
                                          uint64_t num_points) {
    // DiskANN PQ pivot file format (from generate_pq.cpp):
    // The file is organized in bin format sections using save_bin():
    //
    // Section 0: cumul_bytes table at offset 0 (METADATA_SIZE = 4096 bytes reserved)
    //   - Stored as: uint64[4] with bin header
    //   - cumul_bytes[0] = 4096 (start of codebook)
    //   - cumul_bytes[1] = offset to centroid section
    //   - cumul_bytes[2] = offset to chunk_offsets section
    //   - cumul_bytes[3] = total file size
    //
    // Section 1: full_pivot_data (codebook) at offset 4096
    //   - Stored as: float[num_centers=256, dim] with bin header
    //   - This is the PQ codebook with 256 centroids per subspace
    //
    // Section 2: centroid vector
    //   - Stored as: float[dim, 1] with bin header
    //
    // Section 3: chunk_offsets
    //   - Stored as: uint32[num_pq_chunks+1, 1] with bin header
    //   - Dimension boundaries for each PQ chunk
    //
    // DiskANN compressed vectors file:
    // Uses bin format: npts (int32), nchunks (int32), compressed_data (uint8[npts * nchunks])

    // Reset streams to beginning
    pq_stream.seekg(0, std::ios::beg);
    compressed_stream.seekg(0, std::ios::beg);

    // Read cumul_bytes table (first bin section in PQ file)
    int32_t cb_npts = 0;
    int32_t cb_dim = 0;
    pq_stream.read(reinterpret_cast<char*>(&cb_npts), sizeof(int32_t));
    pq_stream.read(reinterpret_cast<char*>(&cb_dim), sizeof(int32_t));

    // Validate cumul_bytes header: expects [n, 1] where n >= 4
    if (cb_npts < 4 || cb_dim != 1) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "Invalid PQ file header: expected [n>=4, 1], got [{}, {}]", cb_npts, cb_dim));
    }

    // Read the data based on the header
    size_t offset_count = static_cast<size_t>(cb_npts) * cb_dim;
    Vector<uint64_t> cumul_bytes(offset_count, allocator_);
    pq_stream.read(reinterpret_cast<char*>(cumul_bytes.data()),
                   static_cast<std::streamsize>(offset_count * sizeof(uint64_t)));

    // The codebook section starts at cumul_bytes[0]
    // For standard format: cumul_bytes[0] = 4096
    // For compact format: cumul_bytes[0] could be a small value like 40
    uint64_t codebook_offset = cumul_bytes[0];

    // Read codebook metadata from codebook section
    pq_stream.seekg(static_cast<std::streamoff>(codebook_offset), std::ios::beg);
    int32_t pivot_npts = 0;
    int32_t pivot_dim = 0;
    pq_stream.read(reinterpret_cast<char*>(&pivot_npts), sizeof(int32_t));
    pq_stream.read(reinterpret_cast<char*>(&pivot_dim), sizeof(int32_t));

    // pivot_npts = 256 (num_centers), pivot_dim = original vector dimension
    if (pivot_npts != 256) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Expected 256 PQ centroids, got {}", pivot_npts));
    }

    diskann_dim_ = static_cast<uint32_t>(pivot_dim);

    // Validate dimension consistency with configuration
    if (common_param_.dim_ > 0 && diskann_dim_ != static_cast<uint64_t>(common_param_.dim_)) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("Dimension mismatch: DiskANN index has dim={}, but config expects dim={}",
                        diskann_dim_,
                        common_param_.dim_));
    }

    // Calculate PQ parameters by reading chunk_offsets
    // chunk_offsets is at cumul_bytes[2] (or we can calculate from dimension)
    uint64_t chunk_offset = cumul_bytes.size() > 2
                                ? cumul_bytes[2]
                                : (codebook_offset + 8 +
                                   static_cast<uint64_t>(pivot_npts) *
                                       static_cast<uint64_t>(pivot_dim) * sizeof(float));
    pq_stream.seekg(static_cast<std::streamoff>(chunk_offset), std::ios::beg);
    int32_t chunk_npts = 0;
    int32_t chunk_dim = 0;
    pq_stream.read(reinterpret_cast<char*>(&chunk_npts), sizeof(int32_t));
    pq_stream.read(reinterpret_cast<char*>(&chunk_dim), sizeof(int32_t));

    // chunk_npts = num_pq_chunks + 1, chunk_dim = 1
    uint32_t num_pq_chunks = static_cast<uint32_t>(std::max(1, chunk_npts - 1));

    // Use the num_points passed from tag file (authoritative count)
    diskann_num_points_ = num_points;

    // Setup ProductQuantizer for HGraph - pass the dimension info
    uint32_t subspace_dim = diskann_dim_ / num_pq_chunks;
    if (subspace_dim == 0) {
        subspace_dim = 1;  // Minimum subspace dimension
    }
    uint32_t num_centroids_per_subspace = 256;
    setup_product_quantizer(pq_stream, num_pq_chunks, num_centroids_per_subspace, subspace_dim);

    // Validate PQ dimension consistency with configuration
    void* quantizer_ptr = this->basic_flatten_codes_->GetQuantizer();
    if (quantizer_ptr != nullptr) {
        int64_t configured_pq_dim = -1;
#define GET_PQ_DIM(METRIC) GetPQQuantizer<METRIC>(quantizer_ptr)->pq_dim_
        switch (common_param_.metric_) {
            case MetricType::METRIC_TYPE_L2SQR:
                configured_pq_dim = GET_PQ_DIM(MetricType::METRIC_TYPE_L2SQR);
                break;
            case MetricType::METRIC_TYPE_IP:
                configured_pq_dim = GET_PQ_DIM(MetricType::METRIC_TYPE_IP);
                break;
            case MetricType::METRIC_TYPE_COSINE:
                configured_pq_dim = GET_PQ_DIM(MetricType::METRIC_TYPE_COSINE);
                break;
            default:
                break;
        }
#undef GET_PQ_DIM
        if (configured_pq_dim > 0 && num_pq_chunks != static_cast<uint32_t>(configured_pq_dim)) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format(
                    "PQ dimension mismatch: DiskANN index has pq_dim={}, but config expects "
                    "pq_dim={}",
                    num_pq_chunks,
                    configured_pq_dim));
        }
    }

    // Read compressed vectors into basic_flatten_codes_
    read_compressed_vectors(compressed_stream, diskann_num_points_, num_pq_chunks);
}

void
HGraphDiskANNLoader::setup_product_quantizer(std::istream& pq_stream,
                                             uint32_t& num_subspaces,
                                             uint32_t& num_centroids_per_subspace,
                                             uint32_t& subspace_dim) {
    // The codebook is essential for computing distances with PQ compressed vectors.

    // Rewind to start of PQ file
    pq_stream.seekg(0, std::ios::beg);

    // Read cumul_bytes table to find codebook location
    int32_t cb_npts = 0;
    int32_t cb_dim = 0;
    pq_stream.read(reinterpret_cast<char*>(&cb_npts), sizeof(int32_t));
    pq_stream.read(reinterpret_cast<char*>(&cb_dim), sizeof(int32_t));

    if (cb_npts < 4 || cb_dim != 1) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Invalid cumul_bytes header: [{}, {}]", cb_npts, cb_dim));
    }

    Vector<uint64_t> cumul_bytes(cb_npts, allocator_);
    pq_stream.read(reinterpret_cast<char*>(cumul_bytes.data()),
                   static_cast<std::streamsize>(cb_npts * sizeof(uint64_t)));

    // Seek to codebook section (at offset cumul_bytes[0])
    pq_stream.seekg(static_cast<std::streamoff>(cumul_bytes[0]), std::ios::beg);

    // Read codebook header: num_centers (256), dim (original vector dim)
    int32_t num_centers = 0;
    int32_t dim = 0;
    pq_stream.read(reinterpret_cast<char*>(&num_centers), sizeof(int32_t));
    pq_stream.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    // Validate that dim is evenly divisible by num_subspaces
    if (static_cast<uint32_t>(dim) % num_subspaces != 0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("Dimension ({}) is not evenly divisible by num_subspaces ({}). "
                        "Non-uniform chunk offsets are not supported.",
                        dim,
                        num_subspaces));
    }

    // Calculate subspace_dim
    subspace_dim = static_cast<uint32_t>(dim) / num_subspaces;

    // Read the entire DiskANN codebook
    // DiskANN format: float[num_centers][dim] where each row is a full-dim centroid
    Vector<float> diskann_codebook(static_cast<uint64_t>(num_centers) * dim, allocator_);
    pq_stream.read(reinterpret_cast<char*>(diskann_codebook.data()),
                   static_cast<std::streamsize>(diskann_codebook.size() * sizeof(float)));

    // Convert DiskANN codebook format to VSAG format
    // VSAG expects: codebooks_[subspace][centroid][subspace_dim]
    // DiskANN has: codebook[centroid][dim] where dim = pq_dim * subspace_dim
    Vector<float> vsag_codebook(
        static_cast<uint64_t>(num_subspaces) * num_centroids_per_subspace * subspace_dim,
        allocator_);

    for (uint32_t s = 0; s < num_subspaces; ++s) {
        for (uint32_t c = 0; c < num_centroids_per_subspace; ++c) {
            for (uint32_t d = 0; d < subspace_dim; ++d) {
                // DiskANN: codebook[c * dim + s * subspace_dim + d]
                uint64_t src_idx = static_cast<uint64_t>(c) * static_cast<uint64_t>(dim) +
                                   static_cast<uint64_t>(s) * subspace_dim + d;
                // VSAG: codebooks_[s * 256 * subspace_dim + c * subspace_dim + d]
                uint64_t dst_idx =
                    (static_cast<uint64_t>(s) * num_centroids_per_subspace + c) * subspace_dim + d;
                vsag_codebook[dst_idx] = diskann_codebook[src_idx];
            }
        }
    }

    // Create flatten parameter with PQ
    auto pq_param = std::make_shared<ProductQuantizerParameter>();
    pq_param->pq_dim_ = static_cast<int64_t>(num_subspaces);

    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->quantizer_parameter = pq_param;
    flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();

    // Create the flatten interface
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param_);
    this->basic_flatten_codes_ = flatten;

    // Load the codebook into ProductQuantizer based on metric type
    void* quantizer_ptr = flatten->GetQuantizer();
    if (quantizer_ptr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Failed to get quantizer pointer");
    }

    // Use macro to simplify switch cases
#define LOAD_CODEBOOK(METRIC) \
    GetPQQuantizer<METRIC>(quantizer_ptr)->LoadCodebook(vsag_codebook.data())

    switch (common_param_.metric_) {
        case MetricType::METRIC_TYPE_L2SQR:
            LOAD_CODEBOOK(MetricType::METRIC_TYPE_L2SQR);
            break;
        case MetricType::METRIC_TYPE_IP:
            LOAD_CODEBOOK(MetricType::METRIC_TYPE_IP);
            break;
        case MetricType::METRIC_TYPE_COSINE:
            LOAD_CODEBOOK(MetricType::METRIC_TYPE_COSINE);
            break;
        default:
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("Unsupported metric type for PQ: {}",
                                            static_cast<int>(common_param_.metric_)));
    }
#undef LOAD_CODEBOOK
}

void
HGraphDiskANNLoader::read_compressed_vectors(std::istream& compressed_stream,
                                             uint64_t num_points,
                                             uint32_t num_subspaces) {
    // DiskANN compressed vectors file format (standard bin format):
    // [npts (int32), nchunks (int32), data (uint8[npts * nchunks])]

    // Read header
    int32_t file_npts = 0;
    int32_t file_nchunks = 0;
    compressed_stream.read(reinterpret_cast<char*>(&file_npts), sizeof(int32_t));
    compressed_stream.read(reinterpret_cast<char*>(&file_nchunks), sizeof(int32_t));

    if (file_npts < 0 || file_nchunks < 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Invalid compressed vectors file header");
    }

    // Validate consistency
    if (static_cast<uint64_t>(file_npts) != num_points) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "Compressed vectors file has {} points, expected {}", file_npts, num_points));
    }
    if (static_cast<uint32_t>(file_nchunks) != num_subspaces) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "Compressed vectors file has {} chunks, expected {}", file_nchunks, num_subspaces));
    }

    // Allocate buffer for compressed vectors
    uint64_t code_size = num_subspaces;  // One byte per subspace
    auto codes = std::make_unique<uint8_t[]>(num_points * code_size);

    // Read all compressed vectors
    compressed_stream.read(reinterpret_cast<char*>(codes.get()),
                           static_cast<std::streamsize>(num_points * code_size));

    // Resize and populate the flatten codes
    this->basic_flatten_codes_->Resize(static_cast<InnerIdType>(num_points));

    // Insert pre-encoded PQ codes directly without re-encoding
    for (uint64_t i = 0; i < num_points; ++i) {
        this->basic_flatten_codes_->InsertCodes(codes.get() + i * code_size,
                                                static_cast<InnerIdType>(i));
    }
}

void
HGraphDiskANNLoader::load_diskann_tags(std::istream& tag_stream) {
    // DiskANN tag file format (from save_bin, standard bin format):
    // [npts (int32), dim (int32), tags (int64[npts * dim])]
    // Note: TagT is int64_t in VSAG's DiskANN wrapper

    // Reset stream to beginning
    tag_stream.seekg(0, std::ios::beg);

    int32_t num_points = 0;
    int32_t dim = 0;

    tag_stream.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    tag_stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));

    if (num_points < 0 || dim < 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Invalid tag file header");
    }

    // Read tags as int64_t (VSAG's DiskANN uses int64_t for labels/ids)
    int64_t total_tags = static_cast<int64_t>(num_points) * dim;
    Vector<int64_t> tags(total_tags, allocator_);
    tag_stream.read(reinterpret_cast<char*>(tags.data()),
                    static_cast<std::streamsize>(total_tags * sizeof(int64_t)));

    // Populate label table
    // DiskANN uses tags directly as external IDs (1-to-1 mapping)
    this->label_table_->Resize(num_points);
    for (int32_t i = 0; i < num_points; ++i) {
        // The tag at position i corresponds to inner_id i
        // In standard DiskANN, tags[i] is typically the original dataset ID
        int64_t tag = (dim > 0) ? tags[static_cast<int64_t>(i) * dim] : static_cast<int64_t>(i);
        this->label_table_->Insert(static_cast<InnerIdType>(i), tag);
    }
}

void
HGraphDiskANNLoader::load_diskann_graph(std::istream& graph_stream, uint64_t num_points) {
    // Parse DiskANN graph format (from Index::save_graph)
    // Format:
    //   offset 0: index_size (uint64_t) - total file size (written as 24 initially, updated at end)
    //   offset 8: max_observed_degree (uint32_t)
    //   offset 12: start/ep (uint32_t) - entry point
    //   offset 16: num_frozen_pts (uint64_t)
    //   offset 24+: for each node: GK (uint32_t), then GK neighbor ids (uint32_t each)

    uint64_t index_size = 0;
    uint32_t max_degree = 0;
    uint32_t entry_point = 0;
    uint64_t num_frozen_pts = 0;

    graph_stream.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
    graph_stream.read(reinterpret_cast<char*>(&max_degree), sizeof(max_degree));
    graph_stream.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
    graph_stream.read(reinterpret_cast<char*>(&num_frozen_pts), sizeof(num_frozen_pts));

    diskann_max_degree_ = max_degree;
    this->entry_point_id_ = static_cast<InnerIdType>(entry_point);

    // Use the minimum of file's max_degree and graph's max_degree for truncation
    // Do not expand beyond graph's configured max_degree
    uint32_t effective_max_degree = std::min(max_degree, this->bottom_graph_->MaximumDegree());
    max_degree = effective_max_degree;

    // Calculate number of nodes from expected size
    // index_size = 24 + sum over all nodes of (sizeof(uint32_t) * (GK + 1))
    // num_points passed from caller already includes frozen points if applicable
    uint64_t expected_nodes = num_points;

    // Resize graph to accommodate all points
    this->bottom_graph_->Resize(static_cast<InnerIdType>(expected_nodes));

    // Parse graph structure for each node
    for (uint64_t i = 0; i < expected_nodes; ++i) {
        uint32_t num_neighbors = 0;
        graph_stream.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

        if (num_neighbors > 0) {
            Vector<uint32_t> temp_neighbors(num_neighbors, allocator_);
            graph_stream.read(reinterpret_cast<char*>(temp_neighbors.data()),
                              static_cast<std::streamsize>(num_neighbors * sizeof(uint32_t)));

            // Limit neighbors to max_degree (some nodes may have more neighbors due to consolidation)
            uint32_t neighbors_to_insert = std::min(num_neighbors, max_degree);

            // Convert to InnerIdType and insert into graph
            Vector<InnerIdType> neighbor_ids(neighbors_to_insert, allocator_);
            for (uint32_t j = 0; j < neighbors_to_insert; ++j) {
                neighbor_ids[j] = static_cast<InnerIdType>(temp_neighbors[j]);
            }
            this->bottom_graph_->InsertNeighborsById(static_cast<InnerIdType>(i), neighbor_ids);
        } else {
            // Explicitly insert empty neighbors for nodes with 0 neighbors
            Vector<InnerIdType> empty_neighbors(0, allocator_);
            this->bottom_graph_->InsertNeighborsById(static_cast<InnerIdType>(i), empty_neighbors);
        }
    }

    // Set total count to expected nodes (includes nodes with 0 neighbors)
    this->bottom_graph_->SetTotalCount(static_cast<InnerIdType>(expected_nodes));
}

void
HGraphDiskANNLoader::load_diskann_precise_vectors(std::istream& layout_stream,
                                                  uint64_t num_points,
                                                  uint64_t dim,
                                                  uint64_t max_degree) {
    // The disk layout contains the full-precision vectors interleaved with graph structure
    // We need to extract them for high_precise_codes_

    if (!this->use_reorder_) {
        return;
    }

    if (!this->high_precise_codes_) {
        // Create FP32 quantizer for high precision codes
        auto fp32_param = std::make_shared<FP32QuantizerParameter>();
        auto flatten_param = std::make_shared<FlattenDataCellParameter>();
        flatten_param->quantizer_parameter = fp32_param;
        flatten_param->io_parameter = std::make_shared<MemoryIOParameter>();
        this->high_precise_codes_ = FlattenInterface::MakeInstance(flatten_param, common_param_);
    }

    // Calculate layout parameters
    uint64_t max_node_len =
        dim * sizeof(float) + (max_degree * DISKANN_GRAPH_SLACK + 1) * sizeof(uint32_t);
    if (max_node_len == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Invalid layout parameters: max_node_len is zero");
    }
    uint64_t sector_len =
        std::max(static_cast<uint64_t>(DISKANN_SECTOR_LEN),
                 (max_node_len + DISKANN_SECTOR_LEN - 1) & ~(DISKANN_SECTOR_LEN - 1));
    uint64_t nnodes_per_sector = sector_len / max_node_len;

    // Resize high_precise_codes_ to accommodate all points
    this->high_precise_codes_->Resize(static_cast<InnerIdType>(num_points));

    // Read and extract vectors
    auto vector_buffer = std::make_unique<float[]>(dim);

    for (uint64_t i = 0; i < num_points; ++i) {
        // sector_id has +1 because sector 0 is metadata, data starts from sector 1
        uint64_t sector_id = i / nnodes_per_sector + 1;
        uint64_t offset_in_sector = (i % nnodes_per_sector) * max_node_len;
        uint64_t file_offset = sector_id * sector_len + offset_in_sector;

        // Seek to position
        layout_stream.seekg(static_cast<std::streamoff>(file_offset));

        // DiskANN layout format: [float*dim][uint32_t nnbrs][uint32_t*neighbors]
        layout_stream.read(reinterpret_cast<char*>(vector_buffer.get()),
                           static_cast<std::streamsize>(dim * sizeof(float)));

        // Insert into high_precise_codes_
        this->high_precise_codes_->InsertVector(vector_buffer.get(), static_cast<InnerIdType>(i));
    }

    has_precise_vectors_ = true;
}

}  // namespace vsag
