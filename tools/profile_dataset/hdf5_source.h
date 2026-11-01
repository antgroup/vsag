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

#include <H5Cpp.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

// Readers for the HDF5 layouts documented in docs/docs/en/src/resources/dataset_format.md.
// Unlike EvalDataset::Load they stream bounded row ranges, so profiling memory does not grow
// with the file size, and they treat ground truth as optional.

struct SplitInfo {
    std::string name;
    bool present{false};
    /// Dense rows, or sparse records (known up front only when offsets are stored).
    uint64_t rows{0};
    /// Dense dimensionality; 0 for sparse splits.
    uint64_t dim{0};
    /// On-disk element type, e.g. "float32", "int8", "uint8", or "sparse byte stream".
    std::string storage_type;
    uint64_t stored_bytes{0};
    bool has_offsets{false};
};

struct DatasetInfo {
    std::string path;
    std::string file_name;
    uint64_t file_size_bytes{0};
    VectorKind kind{VectorKind::DENSE};
    /// Raw `type` and `distance` attributes; empty when missing.
    std::string type_attribute;
    std::string distance_attribute;
    std::vector<std::string> dataset_names;
    SplitInfo train;
    SplitInfo test;
    bool has_neighbors{false};
    bool has_distances{false};
    uint64_t ground_truth_rows{0};
    uint64_t ground_truth_k{0};
    bool has_train_labels{false};
    bool has_test_labels{false};
    bool has_valid_ratios{false};
    /// Differences from what `eval_performance` accepts; profiling continues regardless.
    std::vector<std::string> eval_compatibility_issues;

    [[nodiscard]] const SplitInfo&
    Split(const std::string& name) const {
        return name == "test" ? test : train;
    }
};

/// Reads the file structure and attributes without loading vectors. Throws ProfileError
/// for missing files, non-HDF5 files, unsupported layouts and malformed shapes.
DatasetInfo
inspect_dataset(const std::string& path);

/// Streams row ranges of a dense (N, D) split, converting elements to float32.
class DenseSplitReader {
public:
    DenseSplitReader(const std::string& path, const std::string& name);

    [[nodiscard]] uint64_t
    Rows() const {
        return rows_;
    }

    [[nodiscard]] uint64_t
    Dim() const {
        return dim_;
    }

    /// Reads rows [begin, begin + count) into `out` (count * Dim() floats).
    void
    ReadRows(uint64_t begin, uint64_t count, float* out);

private:
    H5::H5File file_;
    H5::DataSet dataset_;
    uint64_t rows_{0};
    uint64_t dim_{0};
    bool unsigned_bytes_{false};
    bool signed_bytes_{false};
    std::vector<uint8_t> byte_buffer_;
};

/// A batch of parsed sparse records. Ids are sorted ascending per row (values follow), as
/// `parse_sparse_vectors` in tools/eval/eval_dataset.cpp does.
struct SparseBatch {
    uint64_t first_row{0};
    /// Row r spans [offsets[r], offsets[r + 1]) in ids/values.
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> ids;
    std::vector<float> values;
    /// Rows whose stored ids were not strictly ascending / contained repeated ids.
    uint64_t unsorted_rows{0};
    uint64_t duplicate_id_rows{0};

    [[nodiscard]] uint64_t
    Rows() const {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    void
    Clear();
};

/// Streams records of a sparse byte-stream split in bounded chunks. Records may straddle
/// chunk boundaries; truncated or oversized records raise ProfileError with the byte offset.
class SparseSplitReader {
public:
    SparseSplitReader(const std::string& path, const std::string& name, uint64_t chunk_bytes);

    /// Parses up to `max_rows` records into `batch`; returns false at the end of the stream.
    bool
    NextBatch(uint64_t max_rows, SparseBatch& batch);

    [[nodiscard]] uint64_t
    TotalBytes() const {
        return total_bytes_;
    }

private:
    bool
    ensure_available(uint64_t bytes);

    H5::H5File file_;
    H5::DataSet dataset_;
    uint64_t total_bytes_{0};
    uint64_t chunk_bytes_{0};
    std::vector<char> buffer_;
    uint64_t buffer_stream_offset_{0};
    uint64_t buffer_begin_{0};
    uint64_t buffer_end_{0};
    uint64_t next_row_{0};
    std::vector<uint32_t> scratch_ids_;
    std::vector<float> scratch_values_;
    std::vector<uint32_t> order_;
};

/// Counts the records of a sparse split by walking the length headers.
uint64_t
count_sparse_rows(const std::string& path, const std::string& name, uint64_t chunk_bytes);

/// Visits a one-dimensional INT64 dataset in chunks of at most `chunk_rows` values.
void
for_each_int64_chunk(const std::string& path,
                     const std::string& name,
                     uint64_t chunk_rows,
                     const std::function<void(uint64_t, const std::vector<int64_t>&)>& visit);

/// Reads selected rows of the (Q, K) /neighbors and /distances datasets.
void
read_ground_truth_rows(const std::string& path,
                       const std::vector<uint64_t>& rows,
                       uint64_t k,
                       std::vector<int64_t>& neighbors,
                       std::vector<float>& distances);

/// Reads a whole one-dimensional FLOAT32 dataset (used for /valid_ratios).
std::vector<float>
read_float_vector(const std::string& path, const std::string& name);

}  // namespace vsag::dataset_profile
