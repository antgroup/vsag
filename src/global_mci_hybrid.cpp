// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/global_mci_hybrid.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>
#include <set>

#include "algorithm/hgraph/hgraph.h"
#include "impl/heap/standard_heap.h"
#include "impl/searcher/mci_distance_sampling.h"
#include "impl/searcher/mci_neighborhood_probe.h"
#include "index/index_impl.h"
#include "simd/fp32_simd.h"
#include "utils/search_lock.h"

namespace vsag {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void
require(bool value, const std::string& message) {
    if (not value) {
        throw std::invalid_argument(message);
    }
}

bool
finite(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

double
elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Parameters {
    Parameters() noexcept(false) = default;
    Parameters(Parameters&&) noexcept(false) = default;
    Json backend;
    double threshold{0.05};
    double negative_ratio{0.5};
    double positive_ratio{2.0};
    double min_expected{1.0};
    uint64_t navigation_ef{80};
    uint64_t max_cliques{256};
    uint64_t member_budget{8192};
    std::string method{"clique_probe"};
    uint64_t global_samples{512};
    uint64_t valid_samples{256};
    uint64_t sample_seed{42};
    double quantile{0.1};
    double confidence_z{1.96};
    bool enabled{true};
    bool probe_only{false};
};

Parameters
parse(const std::string& text) {
    auto j = Json::parse(text.empty() ? "{}" : text);
    require(j.is_object(), "global_mci_hybrid parameters must be an object");
    if (j.contains("global_mci_hybrid")) {
        require(j.size() == 1, "unexpected top-level search parameters");
        j = j.at("global_mci_hybrid");
    }
    const std::set<std::string> keys{"hgraph_valid_ratio_threshold",
                                     "ef_search",
                                     "mci_ef_search",
                                     "hgraph_ef_search",
                                     "mci_seed_ratio",
                                     "mci_seed_coverage",
                                     "mci_seed_max_count",
                                     "mci_hops_limit",
                                     "mci_expansion_idle_window",
                                     "mci_expansion_lower_bound",
                                     "search_no_lock",
                                     "parallelism",
                                     "correlation_routing",
                                     "correlation_probe_only",
                                     "correlation_navigation_ef",
                                     "correlation_max_cliques",
                                     "correlation_member_budget",
                                     "correlation_negative_ratio",
                                     "correlation_positive_ratio",
                                     "correlation_min_expected",
                                     "correlation_method",
                                     "correlation_global_samples",
                                     "correlation_valid_samples",
                                     "correlation_sample_seed",
                                     "correlation_radius_quantile",
                                     "correlation_confidence_z"};
    require(j.is_object(), "global_mci_hybrid parameters must be an object");
    for (const auto& entry : j.items()) {
        require(keys.count(entry.key()) != 0,
                "unknown global_mci_hybrid parameter: " + entry.key());
    }
    require(j.value("search_no_lock", true), "global_mci_hybrid requires search_no_lock=true");
    require(j.value("parallelism", 1) == 1, "global_mci_hybrid requires parallelism=1");
    Parameters p;
    p.threshold = j.value("hgraph_valid_ratio_threshold", p.threshold);
    p.negative_ratio = j.value("correlation_negative_ratio", p.negative_ratio);
    p.positive_ratio = j.value("correlation_positive_ratio", p.positive_ratio);
    p.min_expected = j.value("correlation_min_expected", p.min_expected);
    p.enabled = j.value("correlation_routing", p.enabled);
    p.probe_only = j.value("correlation_probe_only", p.probe_only);
    auto integer = [&](const char* name, uint64_t fallback, uint64_t lower, uint64_t upper) {
        if (not j.contains(name)) {
            return fallback;
        }
        require(j[name].is_number_integer(), std::string(name) + " must be an integer");
        const auto v = j[name].get<int64_t>();
        require(v >= static_cast<int64_t>(lower) and uint64_t(v) <= upper,
                std::string("invalid ") + name);
        return static_cast<uint64_t>(v);
    };
    p.navigation_ef = integer("correlation_navigation_ef", p.navigation_ef, 10, 1000000);
    p.max_cliques = integer("correlation_max_cliques", p.max_cliques, 1, 65536);
    p.member_budget = integer("correlation_member_budget", p.member_budget, 1, 1000000);
    p.method = j.value("correlation_method", p.method);
    require(
        p.method == "clique_probe" or p.method == "distance_samples" or p.method == "sample_seeds",
        "unknown correlation_method");
    p.global_samples = integer("correlation_global_samples", p.global_samples, 1, 65536);
    p.valid_samples = integer("correlation_valid_samples", p.valid_samples, 1, 1024);
    p.sample_seed = integer("correlation_sample_seed", p.sample_seed, 0, 1000000000);
    p.quantile = j.value("correlation_radius_quantile", p.quantile);
    p.confidence_z = j.value("correlation_confidence_z", p.confidence_z);
    require(finite(p.quantile) and p.quantile > 0 and p.quantile <= 1, "invalid radius quantile");
    require(finite(p.confidence_z) and p.confidence_z > 0, "invalid confidence z");
    require(finite(p.threshold) and p.threshold >= 0 and p.threshold <= 1, "invalid threshold");
    require(finite(p.negative_ratio) and p.negative_ratio >= 0 and p.negative_ratio < 1,
            "correlation_negative_ratio must be in [0,1)");
    require(finite(p.positive_ratio) and p.positive_ratio > 1, "invalid positive ratio");
    require(finite(p.min_expected) and p.min_expected > 0, "invalid minimum expected matches");
    p.backend = j;
    for (const auto& key : keys) {
        if (key.find("correlation_") == 0) {
            p.backend.erase(key);
        }
    }
    p.backend["search_no_lock"] = true;
    p.backend["parallelism"] = 1;
    p.backend["use_mci"] = true;
    p.backend["use_hybrid_traversal"] = false;
    p.backend["brute_force_threshold"] = 0.0;
    p.backend["hgraph_valid_ratio_threshold"] = 1.0;
    p.backend["mci_seed_ratio"] = j.value("mci_seed_ratio", 0.1);
    p.backend["mci_seed_coverage"] = j.value("mci_seed_coverage", 0.0);
    p.backend["mci_seed_max_count"] = j.value("mci_seed_max_count", 10000);
    require(finite(p.backend["mci_seed_ratio"].get<double>()) and
                p.backend["mci_seed_ratio"].get<double>() >= 0,
            "invalid mci_seed_ratio");
    require(finite(p.backend["mci_seed_coverage"].get<double>()) and
                p.backend["mci_seed_coverage"].get<double>() >= 0 and
                p.backend["mci_seed_coverage"].get<double>() <= 1,
            "invalid mci_seed_coverage");
    p.backend["mci_seed_max_count"] = integer("mci_seed_max_count", 10000, 0, 1000000000);
    p.backend["ef_search"] = integer("ef_search", 40, 1, 1000000);
    p.backend["mci_ef_search"] = integer("mci_ef_search", p.backend["ef_search"], 1, 1000000);
    p.backend["hgraph_ef_search"] = integer("hgraph_ef_search", p.backend["ef_search"], 1, 1000000);
    return p;
}

double
distance_count(const Json& stats) {
    return stats.value("distance_evaluations", stats.value("dist_cmp", 0.0));
}
}  // namespace

struct GlobalMCIHybrid::State {
    IndexPtr index;
    std::shared_ptr<HGraph> graph;
    FlattenInterfacePtr codes;
    CliqueDataCellBaseView view;
    const float* vectors{nullptr};
    uint64_t stride{0};
    uint64_t count{0};
    uint64_t dim{0};
};

GlobalMCIHybrid::GlobalMCIHybrid(std::shared_ptr<State> state) : state_(std::move(state)) {
}

tl::expected<std::shared_ptr<GlobalMCIHybrid>, Error>
GlobalMCIHybrid::Create(IndexPtr index) {
    try {
        const auto impl = std::dynamic_pointer_cast<IndexImpl<HGraph>>(index);
        require(impl != nullptr, "global_mci_hybrid requires hgraph");
        const auto& common = impl->GetCommonParam();
        require(common.metric_ == MetricType::METRIC_TYPE_L2SQR and
                    common.data_type_ == DataTypes::DATA_TYPE_FLOAT,
                "global_mci_hybrid requires FP32/L2");
        auto state = std::make_shared<State>();
        state->index = std::move(index);
        state->graph = std::dynamic_pointer_cast<HGraph>(impl->GetInnerIndex());
        state->count = impl->GetNumElements();
        state->dim = common.dim_;
        require(state->count > 0 and state->graph->GetNumberRemoved() == 0,
                "global_mci_hybrid requires a nonempty immutable index without deleted rows");
        require(state->graph->mci_cliques_ != nullptr, "missing global MCI companion");
        const auto info = state->graph->mci_cliques_->CollectStats(state->count);
        require(info.delta_clique_count == 0 and info.delta_extra_membership_count == 0,
                "global_mci_hybrid requires static CSR without delta memberships");
        const SearchLockScope scope(true);
        require(state->graph->mci_cliques_->TryGetBaseViewNoLock(state->count, state->view),
                "global MCI CSR is unavailable");
        state->codes = state->graph->get_precise_codes();
        require(state->codes != nullptr and
                    state->codes->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32,
                "global_mci_hybrid requires precise FP32 vectors");
        state->vectors = state->codes->TryGetContiguousRawFloatData(&state->stride);
        require(state->vectors == nullptr or state->stride >= state->dim,
                "invalid FP32 vector stride");
        return std::shared_ptr<GlobalMCIHybrid>(new GlobalMCIHybrid(std::move(state)));
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, e.what()));
    } catch (const std::exception& e) {
        return tl::unexpected(Error(ErrorType::INVALID_ARGUMENT, e.what()));
    }
}

tl::expected<DatasetPtr, Error>
GlobalMCIHybrid::KnnSearch(const DatasetPtr& query,
                           int64_t k,
                           const std::string& parameters,
                           const FilterPtr& filter) const {
    try {
        const auto p = parse(parameters);
        const SearchLockScope scope(true);
        require(query != nullptr and query->GetNumElements() == 1 and
                    query->GetDim() == static_cast<int64_t>(state_->dim) and
                    query->GetFloat32Vectors() != nullptr and k > 0,
                "expected one FP32 query and positive k");
        for (uint64_t d = 0; d < state_->dim; ++d) {
            require(finite(query->GetFloat32Vectors()[d]), "query contains non-finite values");
        }
        require(state_->index->GetNumElements() == static_cast<int64_t>(state_->count) and
                    state_->graph->GetNumberRemoved() == 0,
                "global index changed after router creation");
        const double global_ratio = filter == nullptr ? 1.0 : filter->ValidRatio();
        require(finite(global_ratio) and global_ratio >= 0 and global_ratio <= 1,
                "invalid filter retention");
        const bool low = float(global_ratio) < float(p.threshold);
        Json diagnostics{{"retained_fraction", global_ratio},
                         {"threshold", p.threshold},
                         {"prediction", "disabled"},
                         {"navigation_distances", 0},
                         {"navigation_ms", 0.0},
                         {"probe_ms", 0.0},
                         {"member_slots", 0},
                         {"considered", 0},
                         {"satisfied", 0},
                         {"cliques", 0},
                         {"navigation_seeds", 0},
                         {"search_no_lock", true},
                         {"partition_dependency", false}};
        auto backend = p.backend;
        std::vector<int64_t> extra_seeds;
        bool scan = false;
        std::vector<std::pair<InnerIdType, float>> known_samples;
        diagnostics["correlation_method"] = p.method;
        diagnostics["sample_distances"] = 0;
        diagnostics["reused_sample_distances"] = 0;
        if (low and p.enabled and global_ratio > 0 and p.method != "clique_probe") {
            const auto sample_start = Clock::now();
            const int64_t* labels = nullptr;
            int64_t count = -1;
            filter->GetValidIds(&labels, count);
            require(count > 0 and labels != nullptr,
                    "distance sampling requires complete GetValidIds");
            require(p.backend["mci_seed_max_count"].get<uint64_t>() == 0 or
                        p.backend["mci_seed_max_count"].get<uint64_t>() >= p.valid_samples,
                    "sample seeds exceed mci_seed_max_count");
            uint64_t query_seed = p.sample_seed;
            for (uint64_t d = 0; d < state_->dim; ++d) {
                uint32_t bits;
                std::memcpy(&bits, query->GetFloat32Vectors() + d, sizeof(bits));
                query_seed = (query_seed ^ bits) * 1099511628211ULL;
            }
            auto computer = state_->codes->FactoryComputer(query->GetFloat32Vectors());
            auto compute = [&](const std::vector<InnerIdType>& ids, std::vector<float>& values) {
                values.resize(ids.size());
                for (uint64_t begin = 0; begin < ids.size(); begin += 64) {
                    state_->codes->Query(values.data() + begin,
                                         computer,
                                         ids.data() + begin,
                                         std::min<uint64_t>(64, ids.size() - begin));
                }
                diagnostics["sample_distances"] =
                    diagnostics["sample_distances"].get<uint64_t>() + ids.size();
            };
            std::vector<InnerIdType> global_ids;
            std::vector<float> global_distances;
            if (p.method == "distance_samples") {
                const auto selected =
                    mci_sample_positions(state_->count, p.global_samples, p.sample_seed);
                global_ids.assign(selected.begin(), selected.end());
                compute(global_ids, global_distances);
            }
            std::vector<InnerIdType> valid_ids;
            const auto positions =
                mci_sample_positions(static_cast<uint64_t>(count), p.valid_samples, query_seed);
            for (auto position : positions) {
                const auto label = labels[position];
                const auto [found, inner] = state_->graph->label_table_->TryGetIdByLabel(label);
                require(found and filter->CheckValid(label), "invalid sampled legal ID");
                valid_ids.push_back(inner);
                extra_seeds.push_back(label);
            }
            std::vector<float> valid_distances(valid_ids.size());
            std::vector<InnerIdType> pending;
            std::vector<uint64_t> pending_positions;
            for (uint64_t i = 0; i < valid_ids.size(); ++i) {
                auto it = std::lower_bound(global_ids.begin(), global_ids.end(), valid_ids[i]);
                if (it != global_ids.end() and *it == valid_ids[i]) {
                    valid_distances[i] = global_distances[it - global_ids.begin()];
                } else {
                    pending.push_back(valid_ids[i]);
                    pending_positions.push_back(i);
                }
            }
            std::vector<float> pending_distances;
            compute(pending, pending_distances);
            for (uint64_t i = 0; i < pending.size(); ++i) {
                valid_distances[pending_positions[i]] = pending_distances[i];
            }
            for (uint64_t i = 0; i < valid_ids.size(); ++i) {
                known_samples.emplace_back(valid_ids[i], valid_distances[i]);
            }
            std::sort(known_samples.begin(), known_samples.end());
            require(std::adjacent_find(known_samples.begin(),
                                       known_samples.end(),
                                       [](const auto& a, const auto& b) {
                                           return a.first == b.first;
                                       }) == known_samples.end(),
                    "GetValidIds contains duplicate sampled IDs");
            diagnostics["global_sample_count"] = global_ids.size();
            diagnostics["valid_sample_count"] = valid_ids.size();
            if (p.method == "distance_samples") {
                const auto measured = mci_classify_distance_samples(global_distances,
                                                                    valid_distances,
                                                                    p.quantile,
                                                                    p.negative_ratio,
                                                                    p.positive_ratio,
                                                                    p.confidence_z);
                diagnostics["prediction"] = measured.prediction;
                diagnostics["radius"] = measured.radius;
                diagnostics["global_inside"] = measured.global_inside;
                diagnostics["valid_inside"] = measured.valid_inside;
                diagnostics["enrichment"] = measured.enrichment;
                diagnostics["enrichment_lower"] = measured.lower;
                diagnostics["enrichment_upper"] = measured.upper;
                scan = std::string(measured.prediction) == "negative" and not p.probe_only;
            }
            backend["mci_seed_ids"] = extra_seeds;
            backend["mci_seed_distances"] = valid_distances;
            backend["mci_seed_replace"] = true;
            diagnostics["probe_ms"] = elapsed(sample_start);
        } else if (low and p.enabled and global_ratio > 0) {
            const Json nav{{"hgraph",
                            {{"use_mci", false},
                             {"use_hybrid_traversal", false},
                             {"brute_force_threshold", 0.0},
                             {"parallelism", 1},
                             {"search_no_lock", true},
                             {"ef_search", p.navigation_ef}}}};
            auto start = Clock::now();
            auto navigated = state_->index->KnnSearch(
                query,
                static_cast<int64_t>(std::min<uint64_t>(10, state_->count)),
                nav.dump(),
                FilterPtr{});
            require(navigated.has_value(), navigated ? "" : navigated.error().message);
            diagnostics["navigation_ms"] = elapsed(start);
            const auto nav_stats = Json::parse(navigated.value()->GetStatistics());
            const auto nav_route = nav_stats.value("mci_hybrid_route", "");
            require(nav_route == "hgraph" or nav_route == "disabled",
                    "unfiltered navigation did not use full HGraph");
            diagnostics["navigation_distances"] = distance_count(nav_stats);
            diagnostics["navigation_hops"] = nav_stats.value("hops", 0.0);
            std::vector<InnerIdType> anchors;
            for (int64_t i = 0; i < navigated.value()->GetDim(); ++i) {
                const auto label = navigated.value()->GetIds()[i];
                const auto [found, inner] = state_->graph->label_table_->TryGetIdByLabel(label);
                require(found, "navigation returned an unknown ID");
                anchors.push_back(inner);
                if (filter->CheckValid(label)) {
                    extra_seeds.push_back(label);
                }
            }
            diagnostics["navigation_top10"] = navigated.value()->GetDim();
            diagnostics["navigation_seeds"] = extra_seeds.size();
            start = Clock::now();
            uint64_t bitmap_size = 0;
            const auto* bitmap = filter->GetValidBitmap(&bitmap_size);
            auto valid = [&](InnerIdType inner) {
                const auto label = state_->graph->label_table_->GetLabelById(inner);
                return bitmap != nullptr and label >= 0 and uint64_t(label) < bitmap_size
                           ? bitmap[label] != 0
                           : filter->CheckValid(label);
            };
            thread_local MCINeighborhoodProbeScratch scratch;
            auto measured = probe_mci_neighborhood(state_->view,
                                                   state_->count,
                                                   anchors,
                                                   p.max_cliques,
                                                   p.member_budget,
                                                   valid,
                                                   scratch);
            classify_mci_neighborhood(
                measured, global_ratio, p.negative_ratio, p.positive_ratio, p.min_expected);
            diagnostics["probe_ms"] = elapsed(start);
            diagnostics["prediction"] = measured.prediction;
            diagnostics["cliques"] = measured.cliques;
            diagnostics["member_slots"] = measured.member_slots;
            diagnostics["considered"] = measured.considered;
            diagnostics["satisfied"] = measured.satisfied;
            diagnostics["local_ratio"] = measured.local_ratio;
            diagnostics["enrichment"] = measured.enrichment;
            scan = std::string(measured.prediction) == "negative" and not p.probe_only;
            backend["mci_seed_ids"] = extra_seeds;
        }
        DatasetPtr result;
        double search_distances = 0;
        const auto start = Clock::now();
        if (scan or (low and global_ratio == 0)) {
            const int64_t* valid_labels = nullptr;
            int64_t valid_count = -1;
            filter->GetValidIds(&valid_labels, valid_count);
            std::vector<InnerIdType> ids;
            if (valid_count >= 0 and (valid_count == 0 or valid_labels != nullptr)) {
                ids.reserve(valid_count);
                for (int64_t i = 0; i < valid_count; ++i) {
                    const auto [found, inner] =
                        state_->graph->label_table_->TryGetIdByLabel(valid_labels[i]);
                    if (found) {
                        ids.push_back(inner);
                    }
                }
                if (not std::is_sorted(ids.begin(), ids.end())) {
                    std::sort(ids.begin(), ids.end());
                }
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            } else {
                for (uint64_t inner = 0; inner < state_->count; ++inner) {
                    if (filter->CheckValid(state_->graph->label_table_->GetLabelById(inner))) {
                        ids.push_back(inner);
                    }
                }
            }
            auto heap = DistanceHeap::MakeInstanceBySize<true, true>(state_->graph->allocator_, k);
            const auto scanned_count = ids.size();
            if (not known_samples.empty()) {
                std::vector<InnerIdType> pending;
                pending.reserve(ids.size());
                uint64_t cached = 0;
                for (auto id : ids) {
                    while (cached < known_samples.size() and known_samples[cached].first < id) {
                        ++cached;
                    }
                    if (cached < known_samples.size() and known_samples[cached].first == id) {
                        heap->Push(known_samples[cached].second, id);
                    } else {
                        pending.push_back(id);
                    }
                }
                diagnostics["reused_sample_distances"] = ids.size() - pending.size();
                ids.swap(pending);
            }
            diagnostics["scan_candidates"] = scanned_count;
            if (state_->vectors != nullptr) {
                for (const auto id : ids) {
                    heap->Push(FP32ComputeL2Sqr(query->GetFloat32Vectors(),
                                                state_->vectors + id * state_->stride,
                                                state_->dim),
                               id);
                }
            } else {
                auto computer = state_->codes->FactoryComputer(query->GetFloat32Vectors());
                float batch_distances[64];
                for (uint64_t begin = 0; begin < ids.size(); begin += 64) {
                    const auto count = std::min<uint64_t>(64, ids.size() - begin);
                    state_->codes->Query(batch_distances, computer, ids.data() + begin, count);
                    for (uint64_t i = 0; i < count; ++i) {
                        heap->Push(batch_distances[i], ids[begin + i]);
                    }
                }
            }
            const auto count = heap->Size();
            auto labels = std::make_unique<int64_t[]>(count);
            auto distances = std::make_unique<float[]>(count);
            for (uint64_t i = count; i > 0; --i) {
                distances[i - 1] = heap->Top().first;
                labels[i - 1] = state_->graph->label_table_->GetLabelById(heap->Top().second);
                heap->Pop();
            }
            result = Dataset::Make()
                         ->NumElements(1)
                         ->Dim(static_cast<int64_t>(count))
                         ->Ids(labels.release())
                         ->Distances(distances.release())
                         ->Owner(true);
            search_distances = static_cast<double>(ids.size());
            result->Statistics(Json{
                {"mci_hybrid_route", "correlation_scan"},
                {"distance_evaluations", search_distances},
                {"mci_seed_count", 0},
                {"mci_expanded_cliques",
                 0}}.dump());
            diagnostics["route"] = "scan";
        } else {
            if (not low) {
                backend["use_mci"] = false;
                backend["ef_search"] = backend["hgraph_ef_search"];
            }
            auto got = state_->index->KnnSearch(query, k, Json{{"hgraph", backend}}.dump(), filter);
            require(got.has_value(), got ? "" : got.error().message);
            result = got.value();
            const auto stats = Json::parse(result->GetStatistics());
            const auto actual_route = stats.value("mci_hybrid_route", "");
            require(low ? actual_route == "mci"
                        : (actual_route == "hgraph" or actual_route == "disabled"),
                    "unexpected backend route");
            search_distances = distance_count(stats);
            diagnostics["route"] = low ? "mci" : "hgraph";
            if (low and not known_samples.empty()) {
                require(stats.value("mci_seed_count", uint64_t{0}) == known_samples.size(),
                        "sample seed replacement was not honored");
                diagnostics["reused_sample_distances"] = known_samples.size();
            }
        }
        diagnostics["search_ms"] = elapsed(start);
        diagnostics["search_distances"] = search_distances;
        diagnostics["total_distances"] = search_distances +
                                         diagnostics["navigation_distances"].get<double>() +
                                         diagnostics["sample_distances"].get<double>();
        auto stats = Json::parse(result->GetStatistics());
        stats["global_mci_hybrid"] = diagnostics;
        result->Statistics(stats.dump());
        return result;
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, e.what()));
    } catch (const std::exception& e) {
        return tl::unexpected(Error(ErrorType::INVALID_ARGUMENT, e.what()));
    }
}

}  // namespace vsag
