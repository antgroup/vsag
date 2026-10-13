
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

#include "uhg.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <unordered_set>

#include "algorithm/hgraph/hgraph.h"
#include "algorithm/sindi/sindi.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "index_feature_list.h"
#include "utils/search_threshold.h"

namespace vsag {
namespace {
uint64_t
positive_integer(const JsonType& json, const char* key, uint64_t fallback) {
    if (!json.Contains(key)) {
        return fallback;
    }
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(json[key].IsNumberInteger(), std::string(key) + " must be an integer");
    auto value = json[key].GetInt();
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(value > 0 && value <= 1000000, std::string(key) + " must be in [1, 1000000]");
    return static_cast<uint64_t>(value);
}

float
unit_float(const JsonType& json, const char* key, float fallback) {
    if (!json.Contains(key)) {
        return fallback;
    }
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(json[key].IsNumber(), std::string(key) + " must be a number");
    auto value = json[key].GetFloat();
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(std::isfinite(value) && value >= 0 && value <= 1,
                   std::string(key) + " must be finite and in [0, 1]");
    return value;
}

std::string
dense_search_parameters(uint64_t beam) {
    JsonType json;
    json["hgraph"]["ef_search"].SetUint64(beam);
    return json.Dump();
}

std::string
sparse_search_parameters(uint64_t count) {
    JsonType json;
    json["sindi"]["n_candidate"].SetUint64(count);
    json["sindi"]["query_prune_ratio"].SetFloat(0);
    json["sindi"]["term_prune_ratio"].SetFloat(0);
    return json.Dump();
}

double
sparse_norm(const SparseVector& vector) {
    double squared = 0;
    for (uint32_t i = 0; i < vector.len_; ++i) {
        squared += static_cast<double>(vector.vals_[i]) * vector.vals_[i];
    }
    return std::sqrt(squared);
}

template <typename T>
void
read_vector(StreamReader& reader, std::vector<T>& values, uint64_t maximum) {
    uint64_t count = 0;
    StreamReader::ReadObj(reader, count);
    auto cursor = reader.GetCursor();
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(count <= maximum && cursor <= reader.Length() &&
                       count <= (reader.Length() - cursor) / sizeof(T),
                   "invalid UHG serialized vector length");
    values.resize(count);
    if (count > 0) {
        reader.Read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    }
}

// Component serializers see a cursor starting at zero, just as in a standalone index.
class GuideStreamWriter : public StreamWriter {
public:
    explicit GuideStreamWriter(StreamWriter& parent) : parent_(parent) {
    }
    void
    Write(const char* data, uint64_t size) override {
        parent_.Write(data, size);
        bytes_written_ += size;
    }

private:
    StreamWriter& parent_;
};

void
write_guide(StreamWriter& writer, const InnerIndexPtr& guide) {
    CountingStreamWriter counter;
    guide->Serialize(counter);
    StreamWriter::WriteObj(writer, counter.GetCursor());
    GuideStreamWriter guide_writer(writer);
    guide->Serialize(guide_writer);
}

void
read_guide(StreamReader& reader, const InnerIndexPtr& guide) {
    uint64_t length = 0;
    StreamReader::ReadObj(reader, length);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(
        reader.GetCursor() <= reader.Length() && length <= reader.Length() - reader.GetCursor(),
        "invalid UHG guide length");
    auto begin = reader.GetCursor();
    auto slice = reader.Slice(begin, length);
    guide->Deserialize(slice);
    reader.Seek(begin + length);
}
}  // namespace

void
UHGParameter::FromJson(const JsonType& json) {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(json.IsObject(), "UHG parameters must be an object");
    const std::unordered_set<std::string> allowed{"type",
                                                  "max_degree",
                                                  "neighbors_per_alpha",
                                                  "construction_candidates",
                                                  "alpha_intervals",
                                                  "hgraph",
                                                  "sindi"};
    for (const auto& item : json.GetInnerJson()->items()) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(allowed.count(item.key()) != 0,
                       "unsupported UHG build parameter: " + item.key());
    }
    max_degree = positive_integer(json, "max_degree", 64);
    neighbors_per_alpha = positive_integer(json, "neighbors_per_alpha", 32);
    construction_candidates = positive_integer(json, "construction_candidates", 200);
    alpha_intervals = positive_integer(json, "alpha_intervals", 10);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(alpha_intervals <= 1000, "alpha_intervals must not exceed 1000");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(construction_candidates >= neighbors_per_alpha,
                   "construction_candidates must be at least neighbors_per_alpha");
    hgraph = json.Contains("hgraph") ? json["hgraph"] : JsonType::Parse("{}");
    sindi = json.Contains("sindi") ? json["sindi"] : JsonType::Parse("{}");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(hgraph.IsObject() && sindi.IsObject(), "guide parameters must be objects");
    const std::unordered_set<std::string> allowed_hgraph{"max_degree",
                                                         "ef_construction",
                                                         "build_thread_count",
                                                         "base_quantization_type",
                                                         "use_reorder"};
    for (const auto& item : hgraph.GetInnerJson()->items()) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(allowed_hgraph.count(item.key()) != 0,
                       "unsupported UHG hgraph parameter: " + item.key());
    }
    const std::unordered_set<std::string> allowed_sindi{"term_id_limit",
                                                        "window_size",
                                                        "remap_term_ids",
                                                        "use_quantization",
                                                        "doc_prune_ratio",
                                                        "use_reorder"};
    for (const auto& item : sindi.GetInnerJson()->items()) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(allowed_sindi.count(item.key()) != 0,
                       "unsupported UHG sindi parameter: " + item.key());
    }
    // Keep exact component distances and the norms used by the pruning bound consistent.
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!hgraph.Contains("base_quantization_type") ||
                       hgraph["base_quantization_type"].GetString() == "fp32",
                   "UHG requires fp32 hgraph base quantization");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!hgraph.Contains("use_reorder") || !hgraph["use_reorder"].GetBool(),
                   "UHG uses the fp32 hgraph base without reorder");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!sindi.Contains("use_quantization") ||
                       (sindi["use_quantization"].IsBool() && !sindi["use_quantization"].GetBool()),
                   "UHG requires unquantized SINDI values");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!sindi.Contains("doc_prune_ratio") || sindi["doc_prune_ratio"].GetFloat() == 0,
                   "UHG requires doc_prune_ratio=0");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!sindi.Contains("use_reorder") || !sindi["use_reorder"].GetBool(),
                   "UHG uses exact SINDI term values without reorder");
    hgraph["base_quantization_type"].SetString("fp32");
    hgraph["use_reorder"].SetBool(false);
    sindi["use_quantization"].SetBool(false);
    sindi["doc_prune_ratio"].SetFloat(0);
    sindi["use_reorder"].SetBool(false);
}

JsonType
UHGParameter::ToJson() const {
    auto json = JsonType::Parse("{}");
    json["type"].SetString("uhg");
    json["max_degree"].SetUint64(max_degree);
    json["neighbors_per_alpha"].SetUint64(neighbors_per_alpha);
    json["construction_candidates"].SetUint64(construction_candidates);
    json["alpha_intervals"].SetUint64(alpha_intervals);
    json["hgraph"].SetJson(hgraph);
    json["sindi"].SetJson(sindi);
    return json;
}

ParamPtr
UHG::CheckAndMappingExternalParam(const JsonType& external, const IndexCommonParam& common) {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(common.metric_ == MetricType::METRIC_TYPE_IP, "UHG supports metric_type=ip");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(common.data_type_ == DataTypes::DATA_TYPE_FLOAT &&
                       common.repr_ == RecordRepr::DENSE && common.dim_ > 0,
                   "UHG requires float32 dense vectors with a positive dimension");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(common.extra_info_size_ == 0, "UHG does not support extra info");
    auto param = std::make_shared<UHGParameter>();
    param->FromJson(external.GetInnerJson()->is_null() ? JsonType::Parse("{}") : external);
    // Validate both guides at creation time, before constructing the index.
    HGraph::CheckAndMappingExternalParam(param->hgraph, common);
    auto sparse_common = common;
    sparse_common.repr_ = RecordRepr::SPARSE;
    SINDI::CheckAndMappingExternalParam(param->sindi, sparse_common);
    return param;
}

UHG::UHG(const ParamPtr& param, const IndexCommonParam& common)
    : InnerIndexInterface(std::dynamic_pointer_cast<UHGParameter>(param), common),
      param_(std::dynamic_pointer_cast<UHGParameter>(param)),
      common_(common) {
    auto guides = make_guides();
    dense_ = std::move(guides.first);
    sparse_ = std::move(guides.second);
    UHG::InitFeatures();
}

std::pair<InnerIndexPtr, InnerIndexPtr>
UHG::make_guides() const {
    auto dense = std::make_shared<HGraph>(
        HGraph::CheckAndMappingExternalParam(param_->hgraph, common_), common_);
    auto sparse_common = common_;
    sparse_common.repr_ = RecordRepr::SPARSE;
    auto sparse = std::make_shared<SINDI>(
        SINDI::CheckAndMappingExternalParam(param_->sindi, sparse_common), sparse_common);
    return {dense, sparse};
}

void
UHG::InitFeatures() {
    index_feature_list_->SetFeatures({SUPPORT_BUILD,
                                      SUPPORT_KNN_SEARCH,
                                      SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
                                      SUPPORT_SEARCH_CONCURRENT,
                                      SUPPORT_CHECK_ID_EXIST,
                                      SUPPORT_GET_MEMORY_USAGE,
                                      SUPPORT_SERIALIZE_BINARY_SET,
                                      SUPPORT_SERIALIZE_FILE,
                                      SUPPORT_SERIALIZE_WRITE_FUNC,
                                      SUPPORT_DESERIALIZE_BINARY_SET,
                                      SUPPORT_DESERIALIZE_FILE,
                                      SUPPORT_DESERIALIZE_READER_SET});
}

std::pair<FlattenInterfacePtr, FlattenInterfacePtr>
UHG::make_distance_cells() const {
    auto dense = FlattenDataCellParameter::CreateDefault("fp32", "memory_io");
    auto sparse = std::make_shared<SparseVectorDataCellParameter>();
    sparse->FromJson(JsonType::Parse(
        R"({"io_params":{"type":"memory_io"},"quantization_params":{"type":"sparse"}})"));
    return {FlattenInterface::MakeInstance(dense, common_),
            FlattenInterface::MakeInstance(sparse, common_)};
}

int64_t
UHG::GetNumElements() const {
    std::shared_lock lock(mutex_);
    return static_cast<int64_t>(ids_.size());
}

bool
UHG::CheckIdExist(int64_t id) const {
    std::shared_lock lock(mutex_);
    return positions_.count(id) != 0;
}

uint64_t
UHG::GetMemoryUsage() const {
    std::shared_lock lock(mutex_);
    uint64_t memory = dense_->GetMemoryUsage() + sparse_->GetMemoryUsage();
    memory += ids_.capacity() * sizeof(int64_t) + norms_.capacity() * sizeof(double);
    memory += graph_.capacity() * sizeof(std::vector<uint32_t>);
    memory += positions_.size() * (sizeof(int64_t) + sizeof(uint32_t) + 2 * sizeof(void*));
    for (const auto& edges : graph_) {
        memory += edges.capacity() * sizeof(uint32_t);
    }
    if (dense_codes_) {
        memory += dense_codes_->GetMemoryUsage() + sparse_codes_->GetMemoryUsage();
    }
    if (visited_pool_) {
        memory += visited_pool_->GetMemoryUsage();
    }
    return memory;
}

void
UHG::validate(const DatasetPtr& data, bool base) const {
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(data != nullptr, "UHG dataset must not be null");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(data->GetNumElements() > 0 && (!base || data->GetIds() != nullptr),
                   "UHG requires nonempty data and IDs for base vectors");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(base || data->GetNumElements() == 1, "UHG requires one query");
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(data->GetDim() == dim_ && data->GetFloat32Vectors() != nullptr &&
                       data->GetSparseVectors() != nullptr,
                   "UHG requires matching dense dimensions and paired dense/sparse vectors");
    for (int64_t i = 0; i < data->GetNumElements(); ++i) {
        const auto& sparse = data->GetSparseVectors()[i];
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(
            (!base || sparse.len_ > 0) &&
                (sparse.len_ == 0 || (sparse.ids_ != nullptr && sparse.vals_ != nullptr)),
            "UHG sparse buffers must not be null");
        for (uint32_t t = 0; t < sparse.len_; ++t) {
            // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
            CHECK_ARGUMENT(std::isfinite(sparse.vals_[t]) && sparse.vals_[t] >= 0,
                           "UHG sparse values must be finite and nonnegative");
            // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
            CHECK_ARGUMENT(t == 0 || sparse.ids_[t - 1] < sparse.ids_[t],
                           "UHG sparse term IDs must be sorted and unique");
        }
        for (int64_t d = 0; d < dim_; ++d) {
            // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
            CHECK_ARGUMENT(std::isfinite(data->GetFloat32Vectors()[i * dim_ + d]),
                           "UHG dense values must be finite");
        }
    }
}

bool
UHG::UpdateId(int64_t /*old_id*/, int64_t /*new_id*/) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "UHG has a fixed topology and does not support UpdateId");
}

std::vector<int64_t>
UHG::Add(const DatasetPtr& /*base*/) {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        "UHG has a fixed topology; use Build on an empty index");
}

std::vector<int64_t>
UHG::Build(const DatasetPtr& base) {
    std::unique_lock lock(mutex_);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(ids_.empty(), "UHG Build requires an empty index");
    validate(base, true);
    auto count = static_cast<uint64_t>(base->GetNumElements());
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(count < std::numeric_limits<uint32_t>::max(), "too many UHG vectors");
    std::vector<int64_t> ids(base->GetIds(), base->GetIds() + count);
    std::unordered_map<int64_t, uint32_t> positions;
    std::vector<double> norms(count);
    for (uint32_t i = 0; i < count; ++i) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(ids[i] != -1 && positions.emplace(ids[i], i).second,
                       "UHG base IDs must be unique and must not be -1");
        norms[i] = sparse_norm(base->GetSparseVectors()[i]);
    }
    auto [dense, sparse] = make_guides();
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(dense->Build(base).empty() && sparse->Build(base).empty(),
                   "UHG guide build rejected base IDs");
    std::vector<std::vector<uint32_t>> graph(count);
    auto route_count = std::min(count, param_->construction_candidates + 1);
    for (uint32_t i = 0; i < count; ++i) {
        auto query = Dataset::Make()
                         ->Owner(false)
                         ->NumElements(1)
                         ->Dim(dim_)
                         ->Float32Vectors(base->GetFloat32Vectors() + i * dim_)
                         ->SparseVectors(base->GetSparseVectors() + i);
        std::unordered_set<uint32_t> pool;
        auto collect = [&](const DatasetPtr& result) {
            for (int64_t j = 0; j < result->GetDim(); ++j) {
                auto found = positions.find(result->GetIds()[j]);
                if (found != positions.end() && found->second != i) {
                    pool.insert(found->second);
                }
            }
        };
        collect(dense->KnnSearch(query,
                                 static_cast<int64_t>(route_count),
                                 dense_search_parameters(route_count),
                                 FilterPtr{}));
        if (query->GetSparseVectors()->len_ > 0) {
            collect(sparse->KnnSearch(query,
                                      static_cast<int64_t>(route_count),
                                      sparse_search_parameters(route_count),
                                      FilterPtr{}));
        }
        struct Candidate {
            uint32_t node;
            float dense;
            float sparse;
            uint64_t frequency{0};
        };
        std::vector<Candidate> candidates;
        candidates.reserve(pool.size());
        for (auto node : pool) {
            candidates.push_back({node,
                                  dense->CalcDistanceById(query, ids[node]),
                                  sparse->CalcDistanceById(query, ids[node])});
        }
        std::vector<uint64_t> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        auto keep = std::min(param_->neighbors_per_alpha, uint64_t(order.size()));
        for (uint64_t a = 0; a <= param_->alpha_intervals; ++a) {
            double alpha = static_cast<double>(a) / static_cast<double>(param_->alpha_intervals);
            auto distance = [&](uint64_t j) {
                return alpha * candidates[j].dense + (1 - alpha) * candidates[j].sparse;
            };
            std::partial_sort(
                order.begin(),
                order.begin() + static_cast<int64_t>(keep),
                order.end(),
                [&](uint64_t x, uint64_t y) {
                    auto dx = distance(x);
                    auto dy = distance(y);
                    return dx == dy ? ids[candidates[x].node] < ids[candidates[y].node] : dx < dy;
                });
            for (uint64_t j = 0; j < keep; ++j) {
                ++candidates[order[j]].frequency;
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](const auto& x, const auto& y) {
            return x.frequency == y.frequency ? ids[x.node] < ids[y.node]
                                              : x.frequency > y.frequency;
        });
        for (const auto& candidate : candidates) {
            if (candidate.frequency == 0 || graph[i].size() == param_->max_degree) {
                break;
            }
            graph[i].push_back(candidate.node);
        }
    }
    auto [dense_codes, sparse_codes] = make_distance_cells();
    dense_codes->Train(base->GetFloat32Vectors(), count);
    sparse_codes->Train(base->GetSparseVectors(), count);
    dense_codes->BatchInsertVector(base->GetFloat32Vectors(), count);
    sparse_codes->BatchInsertVector(base->GetSparseVectors(), count);
    auto visited_pool = std::make_shared<VisitedListPool>(1, allocator_, count, allocator_);
    // Publish only after both guides and the entire graph have been built successfully.
    dense_codes_ = std::move(dense_codes);
    sparse_codes_ = std::move(sparse_codes);
    visited_pool_ = std::move(visited_pool);
    dense_ = std::move(dense);
    sparse_ = std::move(sparse);
    ids_ = std::move(ids);
    positions_ = std::move(positions);
    norms_ = std::move(norms);
    graph_ = std::move(graph);
    label_table_ = dense_->label_table_;
    return {};
}

DatasetPtr
UHG::KnnSearch(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter) const {
    std::shared_lock lock(mutex_);
    validate(query, false);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(k > 0, "UHG k must be positive");
    const auto root = parameters.empty() ? JsonType::Parse("{}") : JsonType::Parse(parameters);
    auto json = root.Contains("uhg") ? root["uhg"] : root;
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(json.IsObject(), "UHG search parameters must be an object");
    double alpha = unit_float(json, "alpha", 0.5F);
    double gamma = unit_float(json, "pruning_scale", 1.0F);
    auto ef = positive_integer(json, "ef_search", 100);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(ef >= static_cast<uint64_t>(k), "UHG ef_search must be at least k");
    auto entry_count = positive_integer(json, "entry_points", 8);
    auto guide_ef = positive_integer(json, "guide_ef_search", 100);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(!json.Contains("use_pruning") || json["use_pruning"].IsBool(),
                   "use_pruning must be boolean");
    bool pruning = !json.Contains("use_pruning") || json["use_pruning"].GetBool();
    if (ids_.empty()) {
        return make_empty_result();
    }
    ef = std::min(ef, uint64_t(ids_.size()));
    entry_count = std::min(entry_count, ef);
    DatasetPtr entries;
    if (alpha <= 0.5 && query->GetSparseVectors()->len_ > 0) {
        entries = sparse_->KnnSearch(query,
                                     static_cast<int64_t>(entry_count),
                                     sparse_search_parameters(std::max(entry_count, guide_ef)),
                                     FilterPtr{});
    } else {
        entries = dense_->KnnSearch(query,
                                    static_cast<int64_t>(entry_count),
                                    dense_search_parameters(std::max(entry_count, guide_ef)),
                                    FilterPtr{});
    }
    using Node = std::pair<double, uint32_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<>> pending;
    std::priority_queue<Node> beam;
    std::priority_queue<Node> output;
    auto visited = visited_pool_->TakeOne();
    struct ReturnVisited {
        std::shared_ptr<VisitedListPool> pool;
        VisitedListPtr visited;
        ~ReturnVisited() {
            pool->ReturnOne(visited);
        }
    } return_visited{visited_pool_, visited};
    auto dense_computer = dense_codes_ && alpha != 0
                              ? dense_codes_->FactoryComputer(query->GetFloat32Vectors())
                              : nullptr;
    auto sparse_computer = sparse_codes_ && alpha != 1
                               ? sparse_codes_->FactoryComputer(query->GetSparseVectors())
                               : nullptr;
    std::vector<uint32_t> batch;
    std::vector<uint32_t> selected;
    std::vector<uint64_t> selected_positions;
    std::vector<float> dense_distances;
    std::vector<float> sparse_distances;
    batch.reserve(std::max(entry_count, param_->max_degree));
    selected.reserve(batch.capacity());
    selected_positions.reserve(batch.capacity());
    double query_norm = sparse_norm(*query->GetSparseVectors());
    auto evaluate = [&]() {
        if (batch.empty()) {
            return;
        }
        dense_distances.assign(batch.size(), 0);
        if (alpha != 0) {
            if (dense_codes_) {
                dense_codes_->Query(
                    dense_distances.data(), dense_computer, batch.data(), batch.size());
            } else {
                // Legacy archives remain readable without duplicating/reconstructing vectors.
                for (uint64_t i = 0; i < batch.size(); ++i) {
                    dense_distances[i] = dense_->CalcDistanceById(query, ids_[batch[i]]);
                }
            }
        }
        selected.clear();
        selected_positions.clear();
        const bool bounded = pruning && beam.size() == ef;
        const double worst = bounded ? beam.top().first : 0;
        for (uint64_t i = 0; i < batch.size(); ++i) {
            auto node = batch[i];
            if (bounded) {
                double bound = alpha * dense_distances[i] +
                               (1 - alpha) * (1 - gamma * query_norm * norms_[node]);
                double margin = 1e-5 * (1 + std::abs(bound) + query_norm * norms_[node]);
                if (bound > worst + margin) {
                    continue;
                }
            }
            selected.push_back(node);
            selected_positions.push_back(i);
        }
        sparse_distances.assign(selected.size(), 0);
        if (alpha != 1 && !selected.empty()) {
            if (sparse_codes_) {
                sparse_codes_->Query(
                    sparse_distances.data(), sparse_computer, selected.data(), selected.size());
            } else {
                for (uint64_t i = 0; i < selected.size(); ++i) {
                    sparse_distances[i] = sparse_->CalcDistanceById(query, ids_[selected[i]]);
                }
            }
        }
        for (uint64_t i = 0; i < selected.size(); ++i) {
            auto node = selected[i];
            Node value{
                alpha * dense_distances[selected_positions[i]] + (1 - alpha) * sparse_distances[i],
                node};
            // Only filtered searches need a separate result heap: excluded nodes remain bridges.
            if (filter && filter->CheckValid(ids_[node])) {
                output.push(value);
                if (output.size() > static_cast<uint64_t>(k)) {
                    output.pop();
                }
            }
            if (beam.size() < ef || value < beam.top()) {
                beam.push(value);
                pending.push(value);
                if (beam.size() > ef) {
                    beam.pop();
                }
            }
        }
        batch.clear();
    };
    for (int64_t j = 0; j < entries->GetDim(); ++j) {
        auto found = positions_.find(entries->GetIds()[j]);
        if (found != positions_.end() && !visited->TestAndSet(found->second)) {
            batch.push_back(found->second);
        }
    }
    evaluate();
    if (pending.empty()) {
        visited->Set(0);
        batch.push_back(0);
        evaluate();
    }
    while (!pending.empty()) {
        auto current = pending.top();
        pending.pop();
        if (beam.size() == ef && current.first > beam.top().first) {
            break;
        }
        for (auto neighbor : graph_[current.second]) {
            if (!visited->TestAndSet(neighbor)) {
                batch.push_back(neighbor);
            }
        }
        evaluate();
    }
    if (!filter) {
        while (beam.size() > static_cast<uint64_t>(k)) {
            beam.pop();
        }
        output = std::move(beam);
    }
    if (output.empty()) {
        return make_empty_result();
    }
    auto result = Dataset::Make()
                      ->Owner(true, allocator_)
                      ->NumElements(1)
                      ->Dim(static_cast<int64_t>(output.size()));
    auto* result_ids = AllocateThresholdArray<int64_t>(output.size(), allocator_);
    result->Ids(result_ids);
    auto* distances = AllocateThresholdArray<float>(output.size(), allocator_);
    result->Distances(distances);
    for (uint64_t j = output.size(); j > 0; --j) {
        result_ids[j - 1] = ids_[output.top().second];
        distances[j - 1] = static_cast<float>(output.top().first);
        output.pop();
    }
    return FilterDatasetByThreshold(result, ParseSearchThreshold(parameters), allocator_);
}

DatasetPtr
UHG::RangeSearch(const DatasetPtr& /*query*/,
                 float /*radius*/,
                 const std::string& /*parameters*/,
                 const FilterPtr& /*filter*/,
                 int64_t /*limited_size*/) const {
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "UHG supports KNN search only");
}

void
UHG::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    StreamWriter::WriteObj(writer, uint32_t{0x55484702});
    StreamWriter::WriteString(writer, param_->ToString());
    StreamWriter::WriteObj(writer, dim_);
    StreamWriter::WriteVector(writer, ids_);
    StreamWriter::WriteVector(writer, norms_);
    for (const auto& edges : graph_) {
        StreamWriter::WriteVector(writer, edges);
    }
    write_guide(writer, dense_);
    write_guide(writer, sparse_);
    const bool forward_vectors = dense_codes_ != nullptr;
    StreamWriter::WriteObj(writer, forward_vectors);
    if (forward_vectors) {
        dense_codes_->Serialize(writer);
        sparse_codes_->Serialize(writer);
    }
}

void
UHG::Deserialize(StreamReader& reader) {
    std::unique_lock lock(mutex_);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(ids_.empty(), "UHG Deserialize requires an empty index");
    uint32_t magic = 0;
    StreamReader::ReadObj(reader, magic);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(magic == 0x55484701 || magic == 0x55484702, "invalid UHG serialization version");
    auto stored = std::make_shared<UHGParameter>();
    std::vector<char> parameter_bytes;
    read_vector(reader, parameter_bytes, uint64_t{1024} * 1024);
    stored->FromString(std::string(parameter_bytes.begin(), parameter_bytes.end()));
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(param_->CheckCompatibility(stored), "incompatible UHG build parameters");
    int64_t dim = 0;
    StreamReader::ReadObj(reader, dim);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(dim == dim_, "incompatible UHG dimension");
    std::vector<int64_t> ids;
    std::vector<double> norms;
    read_vector(reader, ids, UINT32_MAX - 1);
    read_vector(reader, norms, ids.size());
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(ids.size() == norms.size() && ids.size() < UINT32_MAX,
                   "invalid UHG vector count");
    std::vector<std::vector<uint32_t>> graph(ids.size());
    std::unordered_map<int64_t, uint32_t> positions;
    for (uint32_t i = 0; i < ids.size(); ++i) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(ids[i] != -1 && positions.emplace(ids[i], i).second &&
                           std::isfinite(norms[i]) && norms[i] >= 0,
                       "invalid UHG ID or sparse norm");
        read_vector(reader, graph[i], param_->max_degree);
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(graph[i].size() <= param_->max_degree, "invalid UHG degree");
        std::unordered_set<uint32_t> unique;
        for (auto neighbor : graph[i]) {
            // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
            CHECK_ARGUMENT(neighbor < ids.size() && neighbor != i && unique.insert(neighbor).second,
                           "invalid UHG edge");
        }
    }
    auto [dense, sparse] = make_guides();
    read_guide(reader, dense);
    read_guide(reader, sparse);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
    CHECK_ARGUMENT(dense->GetNumElements() == static_cast<int64_t>(ids.size()) &&
                       sparse->GetNumElements() == static_cast<int64_t>(ids.size()),
                   "inconsistent UHG guide sizes");
    for (auto id : ids) {
        // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
        CHECK_ARGUMENT(dense->CheckIdExist(id) && sparse->CheckIdExist(id),
                       "inconsistent UHG guide IDs");
    }
    FlattenInterfacePtr dense_codes;
    FlattenInterfacePtr sparse_codes;
    if (magic == 0x55484702) {
        bool forward_vectors = false;
        StreamReader::ReadObj(reader, forward_vectors);
        if (forward_vectors) {
            std::tie(dense_codes, sparse_codes) = make_distance_cells();
            dense_codes->Deserialize(reader);
            sparse_codes->Deserialize(reader);
            // NOLINTNEXTLINE(readability-simplify-boolean-expr): CHECK_ARGUMENT negates its condition.
            CHECK_ARGUMENT(
                dense_codes->TotalCount() == ids.size() && sparse_codes->TotalCount() == ids.size(),
                "inconsistent UHG forward vector counts");
        }
    }
    auto visited_pool = std::make_shared<VisitedListPool>(1, allocator_, ids.size(), allocator_);
    dense_codes_ = std::move(dense_codes);
    sparse_codes_ = std::move(sparse_codes);
    visited_pool_ = std::move(visited_pool);
    dense_ = std::move(dense);
    sparse_ = std::move(sparse);
    ids_ = std::move(ids);
    norms_ = std::move(norms);
    graph_ = std::move(graph);
    positions_ = std::move(positions);
    label_table_ = dense_->label_table_;
}
}  // namespace vsag
