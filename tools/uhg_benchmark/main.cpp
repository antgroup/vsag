
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

#include <cblas.h>
#include <hdf5.h>
#include <omp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "algorithm/uhg/uhg.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace vsag::benchmark {

double
Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void
Log(const std::string& text) {
    std::cerr << text << std::endl;
}
void
Require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class T>
T
Take(tl::expected<T, Error> result) {
    if (!result.has_value())
        throw std::runtime_error(result.error().message);
    return std::move(result.value());
}
void
Check(tl::expected<void, Error> result) {
    if (!result.has_value())
        throw std::runtime_error(result.error().message);
}

struct Data {
    uint64_t n{0}, nq{0}, dim{0};
    uint32_t vocabulary{0};
    std::vector<float> train, test;
    std::vector<int64_t> ids;
    std::vector<uint8_t> train_bytes, test_bytes;
    std::vector<SparseVector> base_sparse, query_sparse;
    std::vector<double> norms;
    DatasetPtr
    Base() {
        return Dataset::Make()
            ->Owner(false)
            ->NumElements(n)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float32Vectors(train.data())
            ->SparseVectors(base_sparse.data());
    }
    DatasetPtr
    Query(uint64_t q) const {
        return Dataset::Make()
            ->Owner(false)
            ->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(test.data() + q * dim)
            ->SparseVectors(query_sparse.data() + q);
    }
};

std::vector<hsize_t>
Shape(hid_t file, const char* name) {
    hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    Require(dataset >= 0, std::string("missing HDF5 dataset ") + name);
    hid_t space = H5Dget_space(dataset);
    std::vector<hsize_t> dims(H5Sget_simple_extent_ndims(space));
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    H5Dclose(dataset);
    return dims;
}

template <class T>
std::vector<T>
Read(hid_t file, const char* name, hid_t type, uint64_t rows = 0) {
    auto dims = Shape(file, name);
    Require(dims.size() == 1 || dims.size() == 2, "unsupported dataset rank");
    if (rows)
        dims[0] = std::min<hsize_t>(rows, dims[0]);
    uint64_t count = dims[0] * (dims.size() == 2 ? dims[1] : 1);
    std::vector<T> values(count);
    hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    hid_t source = H5Dget_space(dataset);
    std::vector<hsize_t> start(dims.size(), 0);
    H5Sselect_hyperslab(source, H5S_SELECT_SET, start.data(), nullptr, dims.data(), nullptr);
    hid_t target = H5Screate_simple(dims.size(), dims.data(), nullptr);
    Require(H5Dread(dataset, type, target, source, H5P_DEFAULT, values.data()) >= 0,
            std::string("failed to read ") + name);
    H5Sclose(target);
    H5Sclose(source);
    H5Dclose(dataset);
    return values;
}

std::vector<SparseVector>
Parse(std::vector<uint8_t>& bytes, uint64_t rows, uint32_t& vocabulary) {
    std::vector<SparseVector> result;
    result.reserve(rows);
    uint64_t offset = 0;
    for (uint64_t i = 0; i < rows; ++i) {
        Require(offset + 4 <= bytes.size(), "truncated sparse record");
        uint32_t length;
        std::memcpy(&length, bytes.data() + offset, 4);
        offset += 4;
        Require(length > 0 && offset + 8ULL * length <= bytes.size(),
                "invalid sparse length (base/query requires nonempty sparse vectors)");
        auto* ids = reinterpret_cast<uint32_t*>(bytes.data() + offset);
        auto* vals = reinterpret_cast<float*>(bytes.data() + offset + 4ULL * length);
        for (uint32_t t = 0; t < length; ++t) {
            Require(t == 0 || ids[t - 1] < ids[t], "sparse IDs must be sorted and unique");
            Require(std::isfinite(vals[t]) && vals[t] >= 0, "invalid sparse value");
            vocabulary = std::max(vocabulary, ids[t] + 1);
        }
        result.push_back({length, ids, vals});
        offset += 8ULL * length;
    }
    return result;
}

Data
Load(const std::string& path, uint64_t max_base, uint64_t max_queries) {
    auto start = Clock::now();
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    Require(file >= 0, "cannot open dataset");
    auto train_shape = Shape(file, "train"), test_shape = Shape(file, "test");
    Data data;
    data.n = max_base ? std::min(max_base, uint64_t(train_shape[0])) : train_shape[0];
    data.nq = max_queries ? std::min(max_queries, uint64_t(test_shape[0])) : test_shape[0];
    data.dim = train_shape[1];
    Require(data.n >= 100 && test_shape[1] == data.dim, "invalid dense shapes");
    Log("Loading dense base: " + std::to_string(data.n) + " x " + std::to_string(data.dim));
    data.train = Read<float>(file, "train", H5T_NATIVE_FLOAT, data.n);
    data.test = Read<float>(file, "test", H5T_NATIVE_FLOAT, data.nq);
    data.ids = Read<int64_t>(file, "train_labels", H5T_NATIVE_INT64, data.n);
    // Identity labels allow direct row lookup without a multi-million-entry hash map.
    for (uint64_t i = 0; i < data.n; ++i)
        Require(data.ids[i] == int64_t(i), "identity labels required");
    Log("Loading paired sparse vectors");
    data.train_bytes = Read<uint8_t>(file, "train_sparse", H5T_NATIVE_UINT8);
    data.test_bytes = Read<uint8_t>(file, "test_sparse", H5T_NATIVE_UINT8);
    data.base_sparse = Parse(data.train_bytes, data.n, data.vocabulary);
    data.query_sparse = Parse(data.test_bytes, data.nq, data.vocabulary);
    H5Fclose(file);
    data.norms.resize(data.n);
    for (uint64_t i = 0; i < data.n; ++i) {
        double squared = 0;
        for (uint32_t t = 0; t < data.base_sparse[i].len_; ++t) {
            double value = data.base_sparse[i].vals_[t];
            squared += value * value;
        }
        data.norms[i] = std::sqrt(squared);
    }
    Log("Loaded dataset in " + std::to_string(Seconds(start)) +
        " seconds; vocabulary=" + std::to_string(data.vocabulary));
    return data;
}

float
SparseDot(const SparseVector& a, const SparseVector& b) {
    uint32_t i = 0, j = 0;
    float score = 0;
    while (i < a.len_ && j < b.len_) {
        if (a.ids_[i] == b.ids_[j]) {
            score += a.vals_[i++] * b.vals_[j++];
        } else if (a.ids_[i] < b.ids_[j])
            ++i;
        else
            ++j;
    }
    return score;
}

struct Scored {
    double score;
    int64_t id;
    bool
    operator<(const Scored& other) const {
        return score == other.score ? id > other.id : score < other.score;
    }
};

// Gold standard: all base items, complete dense and sparse scores, no candidate truncation.
std::vector<std::vector<int64_t>>
GroundTruth(const Data& data,
            const std::vector<float>& alphas,
            const fs::path& directory,
            int threads) {
    fs::path cache = directory / "ground_truth.bin";
    uint64_t scenarios = alphas.size();
    std::vector<std::vector<int64_t>> truth(scenarios * data.nq, std::vector<int64_t>(100));
    if (fs::exists(cache)) {
        std::ifstream in(cache, std::ios::binary);
        for (auto& row : truth) in.read(reinterpret_cast<char*>(row.data()), 800);
        Require(bool(in), "invalid ground-truth cache");
        return truth;
    }
    auto start = Clock::now();
    constexpr uint64_t block_size = 32768;
    std::vector<float> dense_scores(data.nq * block_size), sparse_scores(data.nq * block_size);
    std::vector<std::vector<std::pair<uint32_t, float>>> query_terms(data.vocabulary);
    for (uint32_t q = 0; q < data.nq; ++q) {
        const auto& vector = data.query_sparse[q];
        for (uint32_t t = 0; t < vector.len_; ++t)
            query_terms[vector.ids_[t]].emplace_back(q, vector.vals_[t]);
    }
    std::vector<std::vector<Scored>> heaps(scenarios * data.nq);
    for (auto& heap : heaps) heap.reserve(101);
    for (uint64_t begin = 0; begin < data.n; begin += block_size) {
        uint64_t count = std::min(block_size, data.n - begin);
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    data.nq,
                    count,
                    data.dim,
                    1,
                    data.test.data(),
                    data.dim,
                    data.train.data() + begin * data.dim,
                    data.dim,
                    0,
                    dense_scores.data(),
                    count);
        std::fill(sparse_scores.begin(), sparse_scores.begin() + data.nq * count, 0);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t j = 0; j < int64_t(count); ++j) {
            const auto& vector = data.base_sparse[begin + j];
            for (uint32_t t = 0; t < vector.len_; ++t)
                for (auto [q, value] : query_terms[vector.ids_[t]])
                    sparse_scores[uint64_t(q) * count + j] += value * vector.vals_[t];
        }
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t q = 0; q < int64_t(data.nq); ++q) {
            for (uint64_t s = 0; s < scenarios; ++s) {
                auto& heap = heaps[s * data.nq + q];
                float alpha = alphas[s];
                for (uint64_t j = 0; j < count; ++j) {
                    Scored value{double(alpha) * dense_scores[q * count + j] +
                                     (1.0 - alpha) * sparse_scores[q * count + j],
                                 int64_t(begin + j)};
                    // Comparator creates a min heap: retain largest similarity scores.
                    auto greater = [](const Scored& a, const Scored& b) { return b < a; };
                    if (heap.size() < 100) {
                        heap.push_back(value);
                        std::push_heap(heap.begin(), heap.end(), greater);
                    } else if (heap.front() < value) {
                        std::pop_heap(heap.begin(), heap.end(), greater);
                        heap.back() = value;
                        std::push_heap(heap.begin(), heap.end(), greater);
                    }
                }
            }
        }
        if (begin % (block_size * 16) == 0 || begin + count == data.n)
            Log("Exact gold " + std::to_string(begin + count) + "/" + std::to_string(data.n) +
                " elapsed=" + std::to_string(Seconds(start)));
    }
    for (uint64_t r = 0; r < heaps.size(); ++r) {
        std::sort(heaps[r].begin(), heaps[r].end(), [](const Scored& a, const Scored& b) {
            return b < a;
        });
        for (uint64_t j = 0; j < 100; ++j) truth[r][j] = heaps[r][j].id;
    }
    std::ofstream out(cache.string() + ".partial", std::ios::binary);
    for (const auto& row : truth) out.write(reinterpret_cast<const char*>(row.data()), 800);
    out.close();
    Require(bool(out), "failed to save ground truth");
    fs::rename(cache.string() + ".partial", cache);
    Log("Exact gold finished in " + std::to_string(Seconds(start)) + " seconds");
    return truth;
}

Json
DenseParams(uint64_t ef) {
    return {{"hgraph", {{"ef_search", ef}}}};
}
Json
SparseParams(uint64_t candidates, float pruning) {
    return {
        {"sindi",
         {{"n_candidate", candidates}, {"query_prune_ratio", pruning}, {"term_prune_ratio", 0.0}}}};
}

IndexPtr
Guide(const std::string& type,
      const Json& params,
      const Data& data,
      const fs::path& path,
      Json& metadata) {
    auto index = Take(Factory::CreateIndex(type, params.dump()));
    auto start = Clock::now();
    if (fs::exists(path)) {
        std::ifstream in(path, std::ios::binary);
        Check(index->Deserialize(in));
        metadata[type + "_loaded_seconds"] = Seconds(start);
        Log("Loaded " + type + " in " + std::to_string(Seconds(start)));
    } else {
        Log("Building " + type);
        auto failed = Take(index->Build(const_cast<Data&>(data).Base()));
        Require(failed.empty(), "guide rejected document IDs");
        metadata[type + "_build_seconds"] = Seconds(start);
        Log("Built " + type + " in " + std::to_string(Seconds(start)) + "; saving");
        std::ofstream out(path.string() + ".partial", std::ios::binary);
        Check(index->Serialize(out));
        out.close();
        Require(bool(out), "failed to save guide archive");
        fs::rename(path.string() + ".partial", path);
    }
    metadata[type + "_index_bytes"] = index->GetMemoryUsage();
    return index;
}

// Parallel offline candidate generation, with the same alpha-frequency rule as UHG::Build.
// The resulting graph is loaded by the production UHG deserializer/search implementation.
void
AssembleUHG(const Data& data,
            const IndexPtr& dense,
            const IndexPtr& sparse,
            const Json& build,
            const fs::path& directory,
            int threads,
            Json& metadata) {
    fs::path archive = directory / "uhg.index";
    if (fs::exists(archive))
        return;
    auto start = Clock::now();
    uint64_t m = build["construction_candidates"], degree = build["max_degree"];
    uint64_t kg = build["neighbors_per_alpha"], intervals = build["alpha_intervals"];
    std::vector<std::vector<uint32_t>> graph(data.n);
    std::string dense_search = DenseParams(std::max(uint64_t{200}, m + 1)).dump();
    std::string sparse_search = SparseParams(m + 1, 0.9F).dump();
    std::atomic<uint64_t> completed{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    Log("Building alpha-cover graph in parallel");
#pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int64_t i = 0; i < int64_t(data.n); ++i) {
        try {
            auto query = Dataset::Make()
                             ->Owner(false)
                             ->NumElements(1)
                             ->Dim(data.dim)
                             ->Float32Vectors(data.train.data() + i * data.dim)
                             ->SparseVectors(data.base_sparse.data() + i);
            std::vector<int64_t> pool;
            auto collect = [&](const DatasetPtr& result) {
                uint64_t retained = 0;
                for (int64_t j = 0; j < result->GetDim() && retained < m; ++j) {
                    auto id = result->GetIds()[j];
                    if (id != i) {
                        pool.push_back(id);
                        ++retained;
                    }
                }
            };
            collect(Take(dense->KnnSearch(query, std::min(data.n, m + 1), dense_search)));
            collect(Take(sparse->KnnSearch(query, std::min(data.n, m + 1), sparse_search)));
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
            struct Candidate {
                uint32_t id;
                float dd, ds;
                uint64_t frequency{0};
            };
            std::vector<Candidate> candidates;
            for (auto id : pool) {
                float dd = 1 - cblas_sdot(data.dim,
                                          data.train.data() + i * data.dim,
                                          1,
                                          data.train.data() + id * data.dim,
                                          1);
                float ds = 1 - SparseDot(data.base_sparse[i], data.base_sparse[id]);
                candidates.push_back({uint32_t(id), dd, ds});
            }
            std::vector<uint64_t> order(candidates.size());
            std::iota(order.begin(), order.end(), 0);
            uint64_t keep = std::min(kg, uint64_t(order.size()));
            for (uint64_t a = 0; a <= intervals; ++a) {
                double alpha = double(a) / intervals;
                auto distance = [&](uint64_t j) {
                    return alpha * candidates[j].dd + (1 - alpha) * candidates[j].ds;
                };
                std::partial_sort(
                    order.begin(), order.begin() + keep, order.end(), [&](auto x, auto y) {
                        auto dx = distance(x), dy = distance(y);
                        return dx == dy ? candidates[x].id < candidates[y].id : dx < dy;
                    });
                for (uint64_t j = 0; j < keep; ++j) ++candidates[order[j]].frequency;
            }
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.frequency == b.frequency ? a.id < b.id : a.frequency > b.frequency;
            });
            for (const auto& c : candidates) {
                if (!c.frequency || graph[i].size() == degree)
                    break;
                graph[i].push_back(c.id);
            }
            auto done = ++completed;
            if (done % 10000 == 0 || done == data.n)
                Log("Alpha-cover " + std::to_string(done) + "/" + std::to_string(data.n) +
                    " elapsed=" + std::to_string(Seconds(start)));
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error)
                error = std::current_exception();
        }
    }
    if (error)
        std::rethrow_exception(error);
    metadata["uhg_graph_build_seconds"] = Seconds(start);
    std::ofstream out(archive.string() + ".partial", std::ios::binary);
    IOStreamWriter writer(out);
    StreamWriter::WriteObj(writer, uint32_t{0x55484701});
    UHGParameter param;
    param.FromJson(JsonType::Parse(build.dump()));
    StreamWriter::WriteString(writer, param.ToString());
    StreamWriter::WriteObj(writer, int64_t(data.dim));
    StreamWriter::WriteVector(writer, data.ids);
    StreamWriter::WriteVector(writer, data.norms);
    for (const auto& edges : graph) StreamWriter::WriteVector(writer, edges);
    for (const char* name : {"hgraph", "sindi"}) {
        fs::path source = directory / (std::string(name) + ".index");
        StreamWriter::WriteObj(writer, uint64_t(fs::file_size(source)));
        std::ifstream in(source, std::ios::binary);
        std::vector<char> buffer(16 * 1024 * 1024);
        while (in) {
            in.read(buffer.data(), buffer.size());
            out.write(buffer.data(), in.gcount());
        }
    }
    out.close();
    Require(bool(out), "failed to save UHG archive");
    fs::rename(archive.string() + ".partial", archive);
    Log("UHG alpha-cover graph saved");
}

// Add forward-vector cells to an existing graph without rebuilding its topology or guides.
void
UpgradeForwardVectors(const Data& data, const fs::path& archive, Json& metadata) {
    std::ifstream source(archive, std::ios::binary);
    uint32_t magic = 0;
    source.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic == 0x55484702)
        return;
    Require(magic == 0x55484701, "invalid graph archive to upgrade");
    Log("Preparing UHG forward vectors; retaining the original graph and guides");
    auto start = Clock::now();
    IndexCommonParam common;
    common.dim_ = data.dim;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto dp = FlattenDataCellParameter::CreateDefault("fp32", "memory_io");
    auto sp = std::make_shared<SparseVectorDataCellParameter>();
    sp->FromJson(JsonType::Parse(
        R"({"io_params":{"type":"memory_io"},"quantization_params":{"type":"sparse"}})"));
    auto dense = FlattenInterface::MakeInstance(dp, common);
    auto sparse = FlattenInterface::MakeInstance(sp, common);
    dense->Train(data.train.data(), data.n);
    sparse->Train(data.base_sparse.data(), data.n);
    dense->BatchInsertVector(data.train.data(), data.n);
    sparse->BatchInsertVector(data.base_sparse.data(), data.n);
    std::ofstream out(archive.string() + ".upgrade", std::ios::binary);
    magic = 0x55484702;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    std::vector<char> buffer(16 * 1024 * 1024);
    while (source) {
        source.read(buffer.data(), buffer.size());
        out.write(buffer.data(), source.gcount());
    }
    Require(source.eof(), "failed to copy original graph archive");
    IOStreamWriter writer(out);
    StreamWriter::WriteObj(writer, true);
    dense->Serialize(writer);
    sparse->Serialize(writer);
    out.close();
    Require(bool(out), "failed to save upgraded graph archive");
    fs::rename(archive.string() + ".upgrade", archive);
    metadata["forward_vector_prepare_seconds"] = Seconds(start);
    metadata["forward_vector_bytes"] = dense->GetMemoryUsage() + sparse->GetMemoryUsage();
    metadata["query_implementation"] = "batched forward-vector distances and pooled visited list";
}

struct Retrieved {
    std::vector<int64_t> ids;
    uint64_t union_size{0};
};
Retrieved
TwoRoute(const IndexPtr& dense,
         const IndexPtr& sparse,
         const DatasetPtr& query,
         uint64_t m,
         float alpha) {
    auto d = Take(dense->KnnSearch(query, m, DenseParams(m).dump()));
    auto s = Take(sparse->KnnSearch(query, m, SparseParams(m, 0.0F).dump()));
    std::vector<int64_t> ids(d->GetIds(), d->GetIds() + d->GetDim());
    ids.insert(ids.end(), s->GetIds(), s->GetIds() + s->GetDim());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    auto dd = Take(dense->CalcDistancesById(query, ids.data(), ids.size()));
    auto ds = Take(sparse->CalcDistancesById(query, ids.data(), ids.size()));
    std::vector<Scored> scores;
    for (uint64_t j = 0; j < ids.size(); ++j) {
        // Exact full hybrid distance for every item in the union (both components).
        double distance =
            double(alpha) * dd->GetDistances()[j] + (1.0 - alpha) * ds->GetDistances()[j];
        scores.push_back({-distance, ids[j]});
    }
    uint64_t k = std::min(uint64_t{100}, uint64_t(scores.size()));
    std::partial_sort(
        scores.begin(), scores.begin() + k, scores.end(), [](const Scored& a, const Scored& b) {
            return b < a;
        });
    Retrieved result;
    result.union_size = ids.size();
    for (uint64_t j = 0; j < k; ++j) result.ids.push_back(scores[j].id);
    return result;
}

Json
Measure(const std::string& method,
        uint64_t budget,
        const std::string& scenario,
        const Data& data,
        const std::vector<std::vector<int64_t>>& truth,
        const std::vector<float>& alphas,
        const std::vector<uint64_t>& assignment,
        const IndexPtr& dense,
        const IndexPtr& sparse,
        const IndexPtr& uhg,
        int repeats) {
    std::vector<DatasetPtr> queries;
    for (uint64_t q = 0; q < data.nq; ++q) queries.push_back(data.Query(q));
    auto retrieve = [&](uint64_t q) {
        float alpha = alphas[assignment[q]];
        if (method == "two_route")
            return TwoRoute(dense, sparse, queries[q], budget, alpha);
        Json params = {{"uhg",
                        {{"alpha", alpha},
                         {"ef_search", budget},
                         {"entry_points", budget},
                         {"guide_ef_search", budget},
                         {"pruning_scale", 0.3},
                         {"use_pruning", true}}}};
        auto result = Take(uhg->KnnSearch(queries[q], 100, params.dump()));
        return Retrieved{{result->GetIds(), result->GetIds() + result->GetDim()}, 0};
    };
    // Warmup is outside timing; all timed queries run sequentially on this thread.
    for (uint64_t q = 0; q < std::min(uint64_t{20}, data.nq); ++q) retrieve(q);
    std::vector<double> total_times, recalls, union_sizes;
    std::vector<double> latencies;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        double total = 0, hit_sum = 0, union_sum = 0;
        for (uint64_t q = 0; q < data.nq; ++q) {
            auto start = Clock::now();
            auto result = retrieve(q);
            double seconds = Seconds(start);
            total += seconds;
            latencies.push_back(seconds * 1000);
            union_sum += result.union_size;
            const auto& gold = truth[assignment[q] * data.nq + q];
            std::unordered_set<int64_t> expected(gold.begin(), gold.end());
            for (auto id : result.ids)
                if (expected.count(id))
                    hit_sum += 1;
        }
        total_times.push_back(total);
        recalls.push_back(hit_sum / (100 * data.nq));
        union_sizes.push_back(union_sum / data.nq);
    }
    std::sort(total_times.begin(), total_times.end());
    std::sort(latencies.begin(), latencies.end());
    Json row = {{"method", method},
                {"budget", budget},
                {"scenario", scenario},
                {"queries", data.nq},
                {"repeats", repeats},
                {"recall_at_100", recalls[0]},
                {"qps", data.nq / total_times[total_times.size() / 2]},
                {"latency_p50_ms", latencies[latencies.size() / 2]},
                {"latency_p95_ms", latencies[uint64_t(0.95 * (latencies.size() - 1))]},
                {"average_union_size", union_sizes[0]},
                {"total_seconds_repeats", total_times}};
    Log(row.dump());
    return row;
}

int
Run(int argc, char** argv) {
    Require(argc >= 3,
            "usage: uhg_benchmark DATASET OUTPUT_DIR [MAX_BASE=0] [MAX_QUERIES=0] "
            "[BUILD_THREADS=64] [REPEATS=3]");
    uint64_t max_base = argc > 3 ? std::stoull(argv[3]) : 0;
    uint64_t max_queries = argc > 4 ? std::stoull(argv[4]) : 0;
    int threads = argc > 5 ? std::stoi(argv[5]) : 64;
    int repeats = argc > 6 ? std::stoi(argv[6]) : 3;
    Require(threads > 0 && repeats > 0, "positive threads/repeats required");
    fs::path directory(argv[2]);
    fs::create_directories(directory);
    auto data = Load(argv[1], max_base, max_queries);
    Json metadata = {
        {"dataset", argv[1]},
        {"base_count", data.n},
        {"query_count", data.nq},
        {"dense_dim", data.dim},
        {"model", "BAAI/bge-m3"},
        {"build_threads", threads},
        {"query_threads", 1},
        {"ground_truth", "exhaustive complete hybrid top-100"},
        {"weights", {0.3, 0.5, 0.7}},
        {"sparse_query_prune_ratio", 0.0},
        {"offline_sparse_query_prune_ratio", 0.9},
        {"offline_dense_ef_search", 200},
        {"uhg_entry_points", "equal to ef_search"},
        {"uhg_pruning_scale", 0.3},
        {"graph_build",
         "parallel alpha-frequency construction; production UHG deserialization/search"},
        {"subset", max_base != 0}};
    fs::path manifest = directory / "metadata.json";
    if (fs::exists(manifest)) {
        Json old;
        std::ifstream(manifest) >> old;
        Require(old["base_count"] == data.n && old["query_count"] == data.nq,
                "cache dataset size mismatch");
        for (auto it = old.begin(); it != old.end(); ++it)
            if (it.key().find("seconds") != std::string::npos)
                metadata[it.key()] = it.value();
    }
    auto save_metadata = [&]() { std::ofstream(manifest) << metadata.dump(2) << '\n'; };
    save_metadata();
    std::vector<float> alphas{0.3F, 0.5F, 0.7F};
    auto gold_start = Clock::now();
    auto truth = GroundTruth(data, alphas, directory, threads);
    metadata["ground_truth_build_or_load_seconds"] = Seconds(gold_start);
    save_metadata();
    Json common = {{"dtype", "float32"}, {"metric_type", "ip"}, {"dim", data.dim}};
    Json hgraph = {{"max_degree", 64},
                   {"ef_construction", 200},
                   {"build_thread_count", threads},
                   {"base_quantization_type", "fp32"},
                   {"use_reorder", false}};
    Json sindi = {{"term_id_limit", data.vocabulary},
                  {"window_size", 10000},
                  {"use_quantization", false},
                  {"use_reorder", false},
                  {"doc_prune_ratio", 0.0}};
    auto dense_params = common;
    dense_params["index_param"] = hgraph;
    auto sparse_params = common;
    sparse_params["repr"] = "sparse";
    sparse_params["index_param"] = sindi;
    metadata["hgraph_build_parameters"] = dense_params;
    metadata["sindi_build_parameters"] = sparse_params;
    auto dense = Guide("hgraph", dense_params, data, directory / "hgraph.index", metadata);
    save_metadata();
    auto sparse = Guide("sindi", sparse_params, data, directory / "sindi.index", metadata);
    save_metadata();
    Json build = {{"max_degree", 64},
                  {"neighbors_per_alpha", 32},
                  {"construction_candidates", 100},
                  {"alpha_intervals", 10},
                  {"hgraph", hgraph},
                  {"sindi", sindi}};
    metadata["uhg_build_parameters"] = build;
    AssembleUHG(data, dense, sparse, build, directory, threads, metadata);
    UpgradeForwardVectors(data, directory / "uhg.index", metadata);
    save_metadata();
    auto uhg_params = common;
    uhg_params["index_param"] = build;
    auto uhg = Take(Factory::CreateIndex("uhg", uhg_params.dump()));
    Log("Loading production UHG index");
    std::ifstream in(directory / "uhg.index", std::ios::binary);
    Check(uhg->Deserialize(in));
    metadata["uhg_index_bytes"] = uhg->GetMemoryUsage();
    save_metadata();
    // Base data is no longer required: exact reranking uses both guide indexes' FP32 storage.
    data.train.clear();
    data.train.shrink_to_fit();
    data.train_bytes.clear();
    data.train_bytes.shrink_to_fit();
    data.base_sparse.clear();
    data.base_sparse.shrink_to_fit();
    std::vector<uint64_t> mixed(data.nq);
    for (uint64_t q = 0; q < data.nq; ++q) mixed[q] = q % alphas.size();
    std::mt19937 random(42);
    std::shuffle(mixed.begin(), mixed.end(), random);
    metadata["mixed_alpha_assignment"] = mixed;
    save_metadata();
    Json results = Json::array();
    for (uint64_t scenario = 0; scenario < 4; ++scenario) {
        std::vector<uint64_t> assignment(data.nq, scenario < 3 ? scenario : 0);
        if (scenario == 3)
            assignment = mixed;
        std::string name =
            scenario < 3 ? "alpha_" + std::to_string(alphas[scenario]) : "mixed_discrete_seed42";
        for (uint64_t m : {100, 150, 200, 300, 500}) {
            results.push_back(Measure("two_route",
                                      m,
                                      name,
                                      data,
                                      truth,
                                      alphas,
                                      assignment,
                                      dense,
                                      sparse,
                                      uhg,
                                      repeats));
            std::ofstream(directory / "results.json") << results.dump(2) << '\n';
        }
        for (uint64_t ef : {100, 200, 300, 500, 1000}) {
            results.push_back(Measure(
                "uhg", ef, name, data, truth, alphas, assignment, dense, sparse, uhg, repeats));
            std::ofstream(directory / "results.json") << results.dump(2) << '\n';
        }
    }
    Log("Benchmark complete");
    return 0;
}
}  // namespace vsag::benchmark

int
main(int argc, char** argv) {
    try {
        return vsag::benchmark::Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
