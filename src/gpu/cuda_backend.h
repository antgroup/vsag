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

#include <cstdint>

/// The optional CUDA backend: finding a device, binding one, the three
/// compute-heavy passes of a k-means build, and a device-resident PiPNN build.
///
/// With ENABLE_CUDA off a stub exporting the same symbols is compiled instead and
/// every entry point returns false, so callers link unconditionally and decide at
/// runtime. No CUDA type appears here.
///
/// One contract for every entry point:
///
///   * `false` means "not done, use the CPU path" -- compiled out, no device, too
///     small, does not fit, or the device failed. The caller's output buffers are
///     untouched in all of those.
///   * `budget_bytes` bounds the device working set; pass what
///     CudaSuggestedBudget returns. 0 is refused, which is how a caller with no
///     device says so.
///   * `min_work` is the smallest `rows * k * dim` worth offloading; 0 selects
///     the calibrated default.
namespace vsag::gpu {

enum class DistanceType : int32_t { L2Sqr = 0, InnerProduct = 1, Cosine = 2 };

/// True when this build has the CUDA backend and the machine has at least one
/// usable device.
bool
CudaAvailable();

/// Binds the calling thread to a device.
///
/// False for an ordinal the machine does not have, so the caller keeps its CPU
/// path instead of a device it did not ask for, and false for one that exists but
/// cannot be used. The runtime does not promise where the thread is left then, so
/// select again before issuing more work.
///
/// The binding outlives the call; prefer CudaDeviceScope, which puts it back.
bool
CudaSelectDevice(int32_t device_id);

/// Binds the calling thread to a device for as long as the object is in scope,
/// and puts it back, since the thread belongs to the caller.
///
/// A previous ordinal of 0 is left alone: the runtime reports 0 both for a
/// thread bound to device 0 and for one that has never bound any, and no
/// runtime call tells them apart (measured on CUDA 12.4), so restoring the
/// second kind would create a context nothing asked for. What that costs is the
/// first kind: a thread that really was on device 0 is left on this scope's
/// device once it closes. The driver API's cuCtxGetCurrent would separate the
/// two, at the price of libcuda in DT_NEEDED, and a build with the backend on
/// currently names no CUDA library there at all, which is what lets it load
/// where no driver is installed.
///
/// A negative ordinal binds nothing, and Bound() is then false, as it is
/// whenever there is no device to use.
class CudaDeviceScope {
public:
    explicit CudaDeviceScope(int32_t device_id);

    // Declared, not defaulted: with CUDA on, the definition puts the previous
    // device back. The stub's is empty, which is the translation unit clang-tidy
    // sees, and there the class does look trivially destructible.
    // NOLINTNEXTLINE(performance-trivially-destructible)
    ~CudaDeviceScope();

    CudaDeviceScope(const CudaDeviceScope&) = delete;
    CudaDeviceScope&
    operator=(const CudaDeviceScope&) = delete;

    /// True when the thread is bound to the requested device.
    [[nodiscard]] bool
    Bound() const {
        return bound_;
    }

private:
    int32_t previous_{0};
    bool bound_{false};
};

/// Nearest-centroid assignment.
///
/// For every row of `query` (query_count x dim, row-major) writes the nearest row
/// of `centroids` (k x dim, row-major) under squared L2 into `labels`, and the
/// mean squared distance into `error` when it is not null. Both operands are
/// streamed in chunks, so neither has to fit in device memory.
bool
CudaAssignNearest(const float* query,
                  uint64_t query_count,
                  const float* centroids,
                  uint64_t k,
                  int32_t dim,
                  int32_t* labels,
                  double* error,
                  uint64_t budget_bytes,
                  uint64_t min_work);

/// k-means++ seeding.
///
/// Draws k rows of `datas` (count x dim, row-major) with probability proportional
/// to the squared distance to the nearest row drawn so far, into `centroids_out`
/// (k x dim).
///
/// `uniforms` holds SeedUniformCount(k) values in [0, 1). `uniforms[0]` picks the
/// first row outright, there being nothing yet to weight it against, and step c in
/// 1..k-1 takes `uniforms[2 * c]` for the weighted draw and `uniforms[2 * c + 1]`
/// for the uniform fallback for when every remaining row coincides with a chosen
/// centroid. `uniforms[1]` is therefore never read: the count is what the indexing
/// reaches, not what the pass consumes. Supplying them leaves the random source
/// with the caller, as the CPU routine does.
///
/// The dataset stays resident for the pass, so it must fit the budget; unlike
/// assignment this one is not chunked.
bool
CudaKMeansPlusPlusInit(const float* datas,
                       uint64_t count,
                       int32_t dim,
                       uint32_t k,
                       const float* uniforms,
                       float* centroids_out,
                       uint64_t budget_bytes,
                       uint64_t min_work);

/// Per-cluster sums and point counts.
///
/// Adds each row of `datas` (count x dim, row-major) into `sums` (k x dim) at its
/// label and counts the rows per label into `counts` (k). Both are overwritten,
/// not accumulated into; labels outside [0, k) are skipped. Points are streamed
/// in chunks, so the dataset need not fit in device memory.
bool
CudaAccumulateCentroids(const float* datas,
                        uint64_t count,
                        int32_t dim,
                        const int32_t* labels,
                        uint32_t k,
                        float* sums,
                        int32_t* counts,
                        uint64_t budget_bytes,
                        uint64_t min_work);

/// Parameters of the device-resident PiPNN build. Fields mirror the PiPNN builder's
/// parameters; the stage names refer to the passes of CudaBuildPiPNNGraph.
struct CudaPiPNNConfig {
    uint64_t max_leaf_size{1024};
    uint64_t min_leaf_size{64};
    float leader_sample_rate{0.005F};
    uint32_t fanout_l1{8};
    uint32_t fanout_l2{2};
    uint32_t leaf_neighbor_count{5};
    uint32_t hash_plane_count{12};
    uint32_t max_degree{32};
    DistanceType distance_type{DistanceType::L2Sqr};
    uint64_t seed{1000};
    // Candidates kept per vertex, nearest first, before the alpha occlusion rule
    // picks the final neighbours from them; at least max_degree are kept.
    uint32_t reservoir_size{64};
    float alpha{1.2F};
};

/// Builds a PiPNN graph entirely on the device: the rows are uploaded once and
/// quantized to a transient INT8 copy used only for scoring, partitioned into
/// overlapping leaves, every leaf member's nearest leaf neighbours are selected and
/// folded into sign-hash buckets, and the bidirectional candidates are pruned to at
/// most `max_degree` neighbours per vertex. The INT8 copy is not a VSAG quantizer:
/// it is never stored and does not depend on the index's quantization type.
///
/// `vectors` holds `count` contiguous rows of `dim` floats. On success
/// `out_neighbors` (count * max_degree slots) and `out_degrees` (count slots) hold
/// each vertex's neighbours nearest first; slots past a vertex's degree are not
/// specified. L2, inner product and cosine are built: cosine normalizes the rows
/// and is then exactly L2, inner product is scored as -dot on the uncentred rows.
/// A configuration the kernels do not cover, or a budget too small for the working
/// set, returns false.
/// The covered range is leaf_neighbor_count, fanouts and max_degree up to 16, 16
/// and 128, hash_plane_count up to 12, reservoir_size up to 128 and alpha >= 1.
/// `budget_bytes` bounds the device working set: when the candidate edges do not
/// fit they are sorted and pruned one slab of source vertices at a time.
bool
CudaBuildPiPNNGraph(const float* vectors,
                    uint64_t count,
                    int32_t dim,
                    const CudaPiPNNConfig& config,
                    uint32_t* out_neighbors,
                    uint32_t* out_degrees,
                    uint64_t budget_bytes);

/// A working-set budget from what the device has free, capped at `cap_bytes`
/// (0 means no cap). Returns 0 with no usable device, which every entry point
/// above reads as "do not offload".
///
/// Call it from a thread that already holds a CudaDeviceScope: it reports the
/// bound device's memory, and on an unbound thread it binds device 0 to answer.
uint64_t
CudaSuggestedBudget(uint64_t cap_bytes);

}  // namespace vsag::gpu
