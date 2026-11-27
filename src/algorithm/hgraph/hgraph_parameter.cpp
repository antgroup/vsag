
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

#include "hgraph_parameter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string_view>

#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface_parameter.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "hgraph_mci_parameter_table.h"
#include "impl/logger/logger.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "inner_string_params.h"
#include "quantization/rabitq_quantization/rabitq_quantizer_parameter.h"
#include "utils/float_utils.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    if (json.Contains("merge_strategy")) {
        CHECK_ARGUMENT(json["merge_strategy"].IsString(), "merge_strategy must be a string");
        merge_strategy = json["merge_strategy"].GetString();
    }
    static constexpr std::array<std::string_view, 2> merge_strategies{"odescent", "fgim"};
    CHECK_ARGUMENT(std::find(merge_strategies.begin(), merge_strategies.end(), merge_strategy) !=
                       merge_strategies.end(),
                   "merge_strategy must be odescent or fgim");
    if (json.Contains("fgim_refinement_iterations")) {
        CHECK_ARGUMENT(json["fgim_refinement_iterations"].IsNumberUnsigned(),
                       "fgim_refinement_iterations must be a non-negative integer");
        fgim_refinement_iterations = json["fgim_refinement_iterations"].GetUint64();
    }

    if (json.Contains(RESIZE_INCREASE_COUNT_BIT)) {
        CHECK_ARGUMENT(json[RESIZE_INCREASE_COUNT_BIT].IsNumberInteger(),
                       "resize_increase_count_bit must be an integer (log2 of slot count)");
        CHECK_ARGUMENT(json[RESIZE_INCREASE_COUNT_BIT].IsNumberUnsigned(),
                       "resize_increase_count_bit must be in range [1, 31] (log2 of slot count)");
        this->resize_increase_count_bit = json[RESIZE_INCREASE_COUNT_BIT].GetUint64();
        CHECK_ARGUMENT(
            this->resize_increase_count_bit >= MIN_RESIZE_INCREASE_COUNT_BIT,
            fmt::format("resize_increase_count_bit must be in range [{}, {}] (log2 of slot count)",
                        MIN_RESIZE_INCREASE_COUNT_BIT,
                        MAX_RESIZE_INCREASE_COUNT_BIT));
        CHECK_ARGUMENT(
            this->resize_increase_count_bit <= MAX_RESIZE_INCREASE_COUNT_BIT,
            fmt::format("resize_increase_count_bit must be in range [{}, {}] (log2 of slot count)",
                        MIN_RESIZE_INCREASE_COUNT_BIT,
                        MAX_RESIZE_INCREASE_COUNT_BIT));
    }

    if (json.Contains(HGRAPH_USE_ELP_OPTIMIZER_KEY)) {
        this->use_elp_optimizer = json[HGRAPH_USE_ELP_OPTIMIZER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY)) {
        this->build_by_base = json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].GetBool();
    }
    if (json.Contains(HGRAPH_RABITQ_FUSED_DATACELL_KEY)) {
        this->rabitq_fused_datacell = json[HGRAPH_RABITQ_FUSED_DATACELL_KEY].GetBool();
    }
    external_fused_supplement = false;
    if (json.Contains(HGRAPH_RABITQ_FUSED_LAYOUT_KEY)) {
        const auto layout = json[HGRAPH_RABITQ_FUSED_LAYOUT_KEY].GetString();
        CHECK_ARGUMENT(layout == "inline" or layout == "external_supplement",
                       "unknown rabitq_fused_layout");
        this->external_fused_supplement = layout == "external_supplement";
    }
    if (external_fused_supplement) {
        CHECK_ARGUMENT(rabitq_fused_datacell, "external_supplement requires rabitq_fused_datacell");
    }

    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);

    if (use_reorder && this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
        CHECK_ARGUMENT(json.Contains(PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[PRECISE_CODES_KEY];
        this->precise_codes_param = CreateFlattenParam(precise_codes_json);
    }

    CHECK_ARGUMENT(json.Contains(GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", GRAPH_KEY));
    const auto& graph_json = json[GRAPH_KEY];

    GraphStorageTypes graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT;
    if (graph_json.Contains(GRAPH_STORAGE_TYPE_KEY)) {
        const auto graph_storage_type_str = graph_json[GRAPH_STORAGE_TYPE_KEY].GetString();
        if (graph_storage_type_str == GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
            graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED;
        }

        if (graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_COMPRESSED &&
            graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_FLAT) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid graph_storage_type: {}", graph_storage_type_str));
        }
    }
    this->bottom_graph_param =
        GraphInterfaceParameter::GetGraphParameterByJson(graph_storage_type, graph_json);

    hierarchical_graph_param = std::make_shared<SparseGraphDatacellParameter>();
    hierarchical_graph_param->max_degree_ = this->bottom_graph_param->max_degree_ / 2;
    if (graph_storage_type == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(this->bottom_graph_param);
        if (graph_param != nullptr) {
            hierarchical_graph_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            hierarchical_graph_param->support_delete_ = graph_param->support_remove_;
            hierarchical_graph_param->use_reverse_edges_ = graph_param->use_reverse_edges_;
        } else {
            hierarchical_graph_param->support_delete_ = false;
        }
    } else {
        hierarchical_graph_param->support_delete_ = false;
    }

    if (json.Contains(EF_CONSTRUCTION_KEY)) {
        this->ef_construction = json[EF_CONSTRUCTION_KEY].GetUint64();
        CHECK_ARGUMENT(this->ef_construction > 0, "ef_construction must be positive");
    }

    // FromJson replaces the whole sub-struct, so it must run before any field that lives on
    // AdaptivePruningParameter is parsed. alpha stays on HGraphParameter and is unaffected.
    adaptive_pruning.FromJson(json);
    if (json.Contains(ALPHA_KEY)) {
        this->alpha = json[ALPHA_KEY].GetFloat();
    }

    if (json.Contains(BUILD_THREAD_COUNT_KEY)) {
        this->build_thread_count = json[BUILD_THREAD_COUNT_KEY].GetUint64();
    }

    if (graph_json.Contains(GRAPH_TYPE_KEY)) {
        graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
        CHECK_ARGUMENT(
            graph_type == GRAPH_TYPE_VALUE_NSW or graph_type == GRAPH_TYPE_VALUE_ODESCENT or
                graph_type == GRAPH_TYPE_VALUE_ALPHA_CNG or graph_type == GRAPH_TYPE_VALUE_PIPNN,
            fmt::format("invalid graph_type: {}", graph_type));
        if (graph_type != GRAPH_TYPE_VALUE_NSW) {
            odescent_param = std::make_shared<ODescentParameter>();
            odescent_param->FromJson(graph_json);
        }
        if (graph_type == GRAPH_TYPE_VALUE_PIPNN) {
            pipnn_param.FromJson(graph_json);
            pipnn_param.alpha = this->alpha;
            pipnn_param.Validate(this->bottom_graph_param->max_degree_);
        }
    }

    if (graph_json.Contains(PIPNN_PARAMETER_ADAPTIVE_PRUNING) and
        graph_json[PIPNN_PARAMETER_ADAPTIVE_PRUNING].GetBool()) {
        CHECK_ARGUMENT(graph_type == GRAPH_TYPE_VALUE_PIPNN,
                       "pipnn_adaptive_pruning requires graph_type=pipnn");
    }

    if (graph_json.Contains(ALPHA_CNG_TAU) and graph_type != GRAPH_TYPE_VALUE_PIPNN) {
        CHECK_ARGUMENT(graph_type == GRAPH_TYPE_VALUE_ALPHA_CNG,
                       "alpha_cng_tau requires graph_type=alpha_cng or pipnn");
        alpha_cng_tau = graph_json[ALPHA_CNG_TAU].GetFloat();
        CHECK_ARGUMENT(std::isfinite(alpha_cng_tau) and alpha_cng_tau >= 0,
                       "alpha_cng_tau must be finite and nonnegative");
    }
    if (graph_type == GRAPH_TYPE_VALUE_ALPHA_CNG) {
        const auto& base = base_codes_param->quantizer_parameter->GetTypeName();
        const bool sq8_rerank =
            reorder_source != HGRAPH_REORDER_SOURCE_BASE and precise_codes_param != nullptr and
            precise_codes_param->quantizer_parameter->GetTypeName() == QUANTIZATION_TYPE_VALUE_SQ8;
        const bool rabitq_reorder =
            base == QUANTIZATION_TYPE_VALUE_RABITQ and use_reorder and not rabitq_fused_datacell and
            (base_codes_param->name == RABITQ_SPLIT_DATA_CELL or sq8_rerank);
        CHECK_ARGUMENT((base == QUANTIZATION_TYPE_VALUE_FP32 and not use_reorder) or rabitq_reorder,
                       "alpha_cng supports FP32, RaBitQ split, or RaBitQ with SQ8 reranking");
        CHECK_ARGUMENT(
            odescent_param->turn > 0 and odescent_param->block_size > 0 and
                odescent_param->sample_rate > 0 and odescent_param->sample_rate <= 1,
            "alpha_cng needs positive seed iterations, block size and sampling rate <= 1");
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
        if (this->bottom_graph_param != nullptr) {
            this->bottom_graph_param->support_duplicate_ = this->support_duplicate;
        }
    }
    if (json.Contains(DEDUPLICATE_STORAGE)) {
        this->deduplicate_storage = json[DEDUPLICATE_STORAGE].GetBool();
    }
    if (this->deduplicate_storage && not this->support_duplicate) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "deduplicate_storage requires support_duplicate to be true");
    }
    if (json.Contains(DUPLICATE_DISTANCE_THRESHOLD)) {
        this->duplicate_distance_threshold = json[DUPLICATE_DISTANCE_THRESHOLD].GetFloat();
    }
    if (json.Contains(SUPPORT_FORCE_REMOVE)) {
        this->support_force_remove = json[SUPPORT_FORCE_REMOVE].GetBool();
    }
    if (this->deduplicate_storage && this->support_force_remove) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "deduplicate_storage does not support force remove because duplicate groups share "
            "physical vector slots");
    }
    if (json.Contains(HGRAPH_PERSIST_SOURCE_ID_KEY)) {
        this->persist_source_id = json[HGRAPH_PERSIST_SOURCE_ID_KEY].GetBool();
    }
    if (json.Contains(PARAMETER_USE_CONJUGATE_GRAPH)) {
        this->use_conjugate_graph = json[PARAMETER_USE_CONJUGATE_GRAPH].GetBool();
    }
    // MCI opt-in: any MCI key present is enough, plus the legacy explicit use_mci flag.
    this->mci_parameters.enabled =
        HGraphMCIFieldsPresent(json) or
        (json.Contains(HGRAPH_USE_MCI) and json[HGRAPH_USE_MCI].GetBool());
    if (this->mci_parameters.enabled) {
        for (const auto& field : kHGraphMCIFields) {
            ReadHGraphMCIField(this->mci_parameters, json, field);
        }
        ValidateHGraphMCIParameters(this->mci_parameters);
    }
    if (this->mci_parameters.enabled and this->support_force_remove) {
        CHECK_ARGUMENT(not this->use_attribute_filter,
                       "MCI force remove does not support attribute storage remapping");
        CHECK_ARGUMENT(not this->support_duplicate,
                       "MCI force remove does not support duplicate groups");
        auto graph_param = std::dynamic_pointer_cast<GraphDataCellParameter>(bottom_graph_param);
        CHECK_ARGUMENT(graph_param != nullptr, "MCI force remove requires a flat graph");
        if (not graph_param->use_reverse_edges_ or
            not hierarchical_graph_param->use_reverse_edges_) {
            logger::warn("MCI force remove enables reverse edges for safe physical ID compaction");
        }
        graph_param->use_reverse_edges_ = true;
        hierarchical_graph_param->use_reverse_edges_ = true;
    }
    if (this->rabitq_fused_datacell) {
        CHECK_ARGUMENT(not this->mci_parameters.enabled,
                       "rabitq_fused_datacell does not support MCI");
        CHECK_ARGUMENT(not this->deduplicate_storage,
                       "rabitq_fused_datacell does not support deduplicate_storage");
        CHECK_ARGUMENT(not this->support_force_remove,
                       "rabitq_fused_datacell does not support force remove");
        CHECK_ARGUMENT(this->base_codes_param->name == RABITQ_SPLIT_DATA_CELL,
                       "rabitq_fused_datacell requires RaBitQ split codes");
        const auto rabitq_param = std::dynamic_pointer_cast<RaBitQuantizerParameter>(
            this->base_codes_param->quantizer_parameter);
        CHECK_ARGUMENT(rabitq_param != nullptr,
                       "rabitq_fused_datacell requires RaBitQ quantization");
        CHECK_ARGUMENT(rabitq_param->pca_dim_ == 0,
                       "rabitq_fused_datacell v1 does not support PCA; "
                       "rabitq_pca_dim must be omitted or set to 0");
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            rabitq_param->num_bits_per_dim_filter_ >= 1 and
                rabitq_param->num_bits_per_dim_filter_ <= 4 and
                rabitq_param->num_bits_per_dim_base_ > rabitq_param->num_bits_per_dim_filter_ and
                rabitq_param->num_bits_per_dim_base_ <= 8,
            "rabitq_fused_datacell requires split x+y with x in [1, 4], y >= 1, "
            "and x+y <= 8");
        const auto io_type = this->base_codes_param->io_parameter->GetTypeName();
        CHECK_ARGUMENT(
            io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO or io_type == IO_TYPE_VALUE_MEMORY_IO,
            "rabitq_fused_datacell only supports memory IO");
        const auto supplement_io = this->base_codes_param->supplement_io_parameter;
        const auto supplement_io_type =
            supplement_io == nullptr ? io_type : supplement_io->GetTypeName();
        const bool in_memory = supplement_io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO or
                               supplement_io_type == IO_TYPE_VALUE_MEMORY_IO;
        const bool external_disk = external_fused_supplement and
                                   io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO and
                                   (supplement_io_type == IO_TYPE_VALUE_ASYNC_IO
#if !HAVE_LIBAIO
                                    or supplement_io_type == IO_TYPE_VALUE_BUFFER_IO
#endif
                                   );
        CHECK_ARGUMENT(in_memory or external_disk, "unsupported fused RaBitQ supplement IO");
    }
}

JsonType
HGraphParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json["merge_strategy"].SetString(merge_strategy);
    json["fgim_refinement_iterations"].SetUint64(fgim_refinement_iterations);
    json[TYPE_KEY].SetString(INDEX_TYPE_HGRAPH);

    json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetBool(this->use_elp_optimizer);
    json[HGRAPH_IGNORE_REORDER_KEY].SetBool(this->ignore_reorder);
    json[HGRAPH_RABITQ_FUSED_DATACELL_KEY].SetBool(this->rabitq_fused_datacell);
    if (external_fused_supplement) {
        json[HGRAPH_RABITQ_FUSED_LAYOUT_KEY].SetString("external_supplement");
    }
    json[REORDER_SOURCE_KEY].SetString(this->reorder_source);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    auto graph_json = this->bottom_graph_param->ToJson();
    if (graph_type == GRAPH_TYPE_VALUE_ALPHA_CNG) {
        graph_json.UpdateJson(odescent_param->ToJson());
        graph_json[GRAPH_TYPE_KEY].SetString(graph_type);
        graph_json[ALPHA_CNG_TAU].SetFloat(alpha_cng_tau);
    }
    if (this->graph_type == GRAPH_TYPE_VALUE_PIPNN) {
        graph_json[GRAPH_TYPE_KEY].SetString(this->graph_type);
        graph_json.UpdateJson(this->pipnn_param.ToJson());
    }
    json[GRAPH_KEY].SetJson(graph_json);
    json[EF_CONSTRUCTION_KEY].SetUint64(this->ef_construction);
    json[RESIZE_INCREASE_COUNT_BIT].SetUint64(this->resize_increase_count_bit);
    json[ALPHA_KEY].SetFloat(this->alpha);
    json.UpdateJson(adaptive_pruning.ToJson());
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate);
    json[DEDUPLICATE_STORAGE].SetBool(this->deduplicate_storage);
    json[DUPLICATE_DISTANCE_THRESHOLD].SetFloat(this->duplicate_distance_threshold);
    json[SUPPORT_FORCE_REMOVE].SetBool(this->support_force_remove);
    json[HGRAPH_PERSIST_SOURCE_ID_KEY].SetBool(this->persist_source_id);
    json[PARAMETER_USE_CONJUGATE_GRAPH].SetBool(this->use_conjugate_graph);
    if (this->mci_parameters.enabled) {
        json[HGRAPH_USE_MCI].SetBool(true);
        for (const auto& field : kHGraphMCIFields) {
            WriteHGraphMCIField(this->mci_parameters, json, field);
        }
    }
    json[TRAIN_SAMPLE_COUNT_KEY].SetInt(this->train_sample_count);
    return json;
}

bool
HGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(HGraphParameter, p, other);
    CHECK_FIELD_EQ(*this, *p, adaptive_pruning.enabled);
    if (adaptive_pruning.enabled) {
        CHECK_FIELD_NEAR(*this, *p, alpha, 1e-6F);
        CHECK_FIELD_NEAR(*this, *p, adaptive_pruning.adjust_step, 1e-6F);
        CHECK_FIELD_EQ(*this, *p, adaptive_pruning.apply_to_reverse);
        CHECK_FIELD_EQ(*this, *p, adaptive_pruning.apply_to_upper);
    }
    auto have_reorder = this->use_reorder && not this->ignore_reorder;
    auto have_reorder_other = p->use_reorder && not p->ignore_reorder;
    if (have_reorder != have_reorder_other) {
        logger::error(
            "HGraphParameter::CheckCompatibility: use_reorder and ignore_reorder must be the same");
        return false;
    }
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    if (have_reorder) {
        CHECK_FIELD_EQ(*this, *p, reorder_source);
        if (this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
            if (not this->precise_codes_param ||
                not this->precise_codes_param->CheckCompatibility(p->precise_codes_param)) {
                logger::error(
                    "HGraphParameter::CheckCompatibility: precise_codes_param is not compatible");
                return false;
            }
        }
    }
    CHECK_SUB_PARAM(*this, *p, bottom_graph_param);
    CHECK_FIELD_EQ(*this, *p, use_attribute_filter);
    CHECK_FIELD_EQ(*this, *p, support_duplicate);
    CHECK_FIELD_EQ(*this, *p, deduplicate_storage);
    CHECK_FIELD_EQ(*this, *p, duplicate_distance_threshold);
    CHECK_FIELD_EQ(*this, *p, support_force_remove);
    CHECK_FIELD_EQ(*this, *p, rabitq_fused_datacell);
    CHECK_FIELD_EQ(*this, *p, external_fused_supplement);
    CHECK_FIELD_EQ(*this, *p, pipnn_param.adaptive_pruning);
    if (this->pipnn_param.adaptive_pruning) {
        CHECK_FIELD_EQ(*this, *p, pipnn_param.alpha_cng_tau);
        CHECK_FIELD_EQ(*this, *p, graph_type);
        CHECK_FIELD_EQ(*this, *p, alpha);
        CHECK_FIELD_EQ(*this, *p, pipnn_param.adaptive_pruning_adjust_step);
    }
    // A conjugate-enabled reader can load an older index without the optional graph and start
    // with an empty one. The reverse direction would discard serialized enhancement data.
    if (not this->use_conjugate_graph and p->use_conjugate_graph) {
        logger::error(
            "HGraphParameter::CheckCompatibility: serialized conjugate graph is not supported");
        return false;
    }
    if (not HGraphMCIParametersCompatible(this->mci_parameters, p->mci_parameters)) {
        return false;
    }
    return true;
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    return FromParsedJson(JsonType::Parse(json_string));
}

float
HGraphSearchParameters::DistanceEarlyStopAlpha(uint64_t ef, int64_t k) const {
    if (not early_stop or
        static_cast<double>(ef) <= DISTANCE_EARLY_STOP_MIN_EF_TO_K * static_cast<double>(k)) {
        return 0.0F;
    }
    return early_stop_alpha > 0.0F ? early_stop_alpha : DefaultDistanceEarlyStopAlpha(k);
}

float
HGraphSearchParameters::DefaultDistanceEarlyStopAlpha(int64_t k) {
    // The line through the values at k = 10 and k = 100 (as floats), clamped to its values at
    // k = 100 and at k = 1, the smallest k.
    constexpr double at_k10 = 1.5F;
    constexpr double at_k100 = 1.15F;
    constexpr double at_k1 = at_k10 + (at_k10 - at_k100);
    static_assert(at_k100 <= at_k10, "the default early-stop alpha must not grow with k");
    const double decades = std::log10(static_cast<double>(std::max<int64_t>(k, 1)) / 10.0);
    const double alpha = at_k10 - (at_k10 - at_k100) * decades;
    return static_cast<float>(std::clamp(alpha, at_k100, at_k1));
}

namespace {

// Presence-guarded readers for optional search parameters.
//
// Every optional hgraph search parameter follows the same three steps: skip when the key is
// absent, reject a JSON value of the wrong type, then apply the field-specific range rule. These
// helpers own the first two steps so that the range rule is the only thing left at each call site.
// They return true when the key was present and `value` was written.
bool
read_optional_bool(const JsonType& json, const char* key, bool& value) {
    if (not json.Contains(key)) {
        return false;
    }
    CHECK_ARGUMENT(json[key].IsBool(), fmt::format("{} must be boolean", key));
    value = json[key].GetBool();
    return true;
}

// Unsigned budgets are range-checked here rather than at the call site: a JSON value that does not
// fit the field is a rejection, never a truncation.
bool
read_optional_uint32(const JsonType& json, const char* key, uint32_t& value) {
    if (not json.Contains(key)) {
        return false;
    }
    CHECK_ARGUMENT(json[key].IsNumberInteger(), fmt::format("{} must be integer", key));
    if (json[key].IsNumberUnsigned()) {
        CHECK_ARGUMENT(json[key].GetUint64() <= UINT32_MAX,
                       fmt::format("{} exceeds uint32 range", key));
        value = static_cast<uint32_t>(json[key].GetUint64());
    } else {
        const auto parsed = json[key].GetInt();
        const bool valid_budget = parsed >= 0 and static_cast<uint64_t>(parsed) <= UINT32_MAX;
        CHECK_ARGUMENT(valid_budget, fmt::format("{} exceeds uint32 range", key));
        value = static_cast<uint32_t>(parsed);
    }
    return true;
}

bool
read_optional_int64(const JsonType& json, const char* key, int64_t& value) {
    if (not json.Contains(key)) {
        return false;
    }
    CHECK_ARGUMENT(json[key].IsNumberInteger(), fmt::format("{} must be integer", key));
    if (json[key].IsNumberUnsigned()) {
        CHECK_ARGUMENT(
            json[key].GetUint64() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            fmt::format("{} exceeds int64 range", key));
        value = json[key].GetInt();
    } else {
        value = json[key].GetInt();
    }
    return true;
}

bool
read_optional_float(const JsonType& json, const char* key, float& value) {
    if (not json.Contains(key)) {
        return false;
    }
    CHECK_ARGUMENT(json[key].IsNumber(), fmt::format("{} must be number", key));
    value = json[key].GetFloat();
    return true;
}

// The hybrid traversal costs are stored as double but the JSON layer exposes float only, so the
// value widens exactly once, here.
bool
read_optional_hybrid_cost(const JsonType& json, const char* key, const char* name, double& target) {
    if (not json.Contains(key)) {
        return false;
    }
    CHECK_ARGUMENT(json[key].IsNumber(), fmt::format("{} must be number", key));
    const auto value = json[key].GetFloat();
    CHECK_ARGUMENT(std::isfinite(value), fmt::format("hgraph {} must be finite", name));
    CHECK_ARGUMENT(value >= 0.0, fmt::format("hgraph {} must be non-negative", name));
    target = value;
    return true;
}

}  // namespace

HGraphSearchParameters
HGraphSearchParameters::FromParsedJson(const JsonType& params) {
    HGraphSearchParameters obj;

    // Validate structure before indexing: malformed requests remain recoverable.
    CHECK_ARGUMENT(params.IsObject(), "search parameters must be an object");
    CHECK_ARGUMENT(params.Contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    CHECK_ARGUMENT(params[INDEX_TYPE_HGRAPH].IsObject(), "hgraph parameters must be an object");
    obj.IndexSearchParameter::FromJson(params[INDEX_TYPE_HGRAPH]);

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    const auto& ef_search_json = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT(ef_search_json.IsNumberInteger(), "ef_search must be an integer");
    if (ef_search_json.IsNumberUnsigned()) {
        CHECK_ARGUMENT(ef_search_json.GetUint64() <=
                           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                       "ef_search exceeds int64_t range");
    }
    obj.ef_search = ef_search_json.GetInt();

    // From here on every parameter is optional and goes through the same presence-guarded reader,
    // so the only per-key code left is the range rule that applies to that one field.
    const auto& hgraph = params[INDEX_TYPE_HGRAPH];

    if (hgraph.Contains(HGRAPH_PARAMETER_EARLY_STOP)) {
        const auto& early_stop = hgraph[HGRAPH_PARAMETER_EARLY_STOP];
        CHECK_ARGUMENT(early_stop.IsObject(), "hgraph early_stop must be an object");
        // A misspelt key must not silently fall back to the default.
        for (const auto& item : early_stop.GetInnerJson()->items()) {
            CHECK_ARGUMENT(item.key() == HGRAPH_PARAMETER_EARLY_STOP_TYPE or
                               item.key() == HGRAPH_PARAMETER_EARLY_STOP_ALPHA,
                           fmt::format("hgraph early_stop has unknown key {}", item.key()));
        }
        CHECK_ARGUMENT(early_stop.Contains(HGRAPH_PARAMETER_EARLY_STOP_TYPE) and
                           early_stop[HGRAPH_PARAMETER_EARLY_STOP_TYPE].IsString() and
                           early_stop[HGRAPH_PARAMETER_EARLY_STOP_TYPE].GetString() ==
                               HGRAPH_PARAMETER_EARLY_STOP_TYPE_DISTANCE,
                       "hgraph early_stop type must be distance");
        obj.early_stop = true;
        if (read_optional_float(
                early_stop, HGRAPH_PARAMETER_EARLY_STOP_ALPHA, obj.early_stop_alpha)) {
            const bool valid_alpha =
                IsFiniteFloatBits(obj.early_stop_alpha) and obj.early_stop_alpha >= 1.0F;
            CHECK_ARGUMENT(valid_alpha, "hgraph early_stop alpha must be finite and at least 1");
        }
    }

    read_optional_uint32(hgraph, HGRAPH_PARAMETER_HOPS_LIMIT, obj.hops_limit);
    read_optional_bool(hgraph, HGRAPH_USE_EXTRA_INFO_FILTER, obj.use_extra_info_filter);
    obj.has_rabitq_one_bit_search = read_optional_bool(
        hgraph, HGRAPH_PARAMETER_RABITQ_ONE_BIT_SEARCH, obj.rabitq_one_bit_search);
    if (read_optional_uint32(hgraph, "rabitq_rerank_candidates", obj.rabitq_rerank_candidates)) {
        CHECK_ARGUMENT(obj.rabitq_rerank_candidates > 0,
                       "rabitq_rerank_candidates must be positive");
    }
    read_optional_bool(hgraph, HGRAPH_USE_MCI, obj.use_mci);
    read_optional_bool(hgraph, HGRAPH_USE_HYBRID_TRAVERSAL, obj.use_hybrid_traversal);

    read_optional_bool(hgraph, SEARCH_NO_LOCK_KEY, obj.search_no_lock);
    read_optional_bool(hgraph, MCI_PACKED_FILTER_KEY, obj.mci_packed_filter);
    read_optional_bool(hgraph, MCI_BATCH_DISTANCE_KEY, obj.mci_batch_distance);
    read_optional_uint32(hgraph, MCI_ACTIVATION_CAP_KEY, obj.mci_activation_cap);
    read_optional_uint32(hgraph, MCI_HOPS_LIMIT_KEY, obj.mci_hops_limit);
    // Stored as uint64_t but deliberately limited to 32 bits: that is far more than any useful
    // window, and it keeps the accepted range identical to the other budget parameters.
    uint32_t expansion_idle_window = 0;
    if (read_optional_uint32(hgraph, MCI_EXPANSION_IDLE_WINDOW_KEY, expansion_idle_window)) {
        obj.mci_expansion_idle_window = expansion_idle_window;
    }
    if (obj.search_no_lock) {
        CHECK_ARGUMENT(obj.parallel_search_thread_count == 1,
                       "search_no_lock requires parallelism=1 (concurrent queries are supported)");
    }

    read_optional_hybrid_cost(hgraph, HGRAPH_HYBRID_VOB, HGRAPH_HYBRID_VOB, obj.hybrid_vob);
    read_optional_hybrid_cost(hgraph,
                              HGRAPH_HYBRID_FILTER_COST_RATIO,
                              HGRAPH_HYBRID_FILTER_COST_RATIO,
                              obj.hybrid_filter_cost_ratio);
    read_optional_bool(
        hgraph, PARAMETER_USE_CONJUGATE_GRAPH_SEARCH, obj.use_conjugate_graph_search);

    CHECK_ARGUMENT(not hgraph.Contains(HGRAPH_MCI_SEED_COUNT_KEY),
                   "hgraph mci_seed_count is not supported; use mci_seed_ratio");
    if (read_optional_float(hgraph, HGRAPH_MCI_SEED_RATIO, obj.mci_seed_ratio)) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            std::isfinite(obj.mci_seed_ratio) and obj.mci_seed_ratio >= 0.0F,
            "hgraph mci_seed_ratio must be finite and non-negative");
    }
    if (read_optional_float(hgraph, HGRAPH_MCI_SEED_COVERAGE, obj.mci_seed_coverage)) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            std::isfinite(obj.mci_seed_coverage) and obj.mci_seed_coverage >= 0.0F,
            "hgraph mci_seed_coverage must be finite and non-negative");
    }
    if (read_optional_int64(hgraph, HGRAPH_MCI_SEED_MAX_COUNT, obj.mci_seed_max_count)) {
        CHECK_ARGUMENT(obj.mci_seed_max_count >= 0,
                       "hgraph mci_seed_max_count must be non-negative");
    }
    if (read_optional_float(hgraph,
                            HGRAPH_MCI_HGRAPH_VALID_RATIO_THRESHOLD,
                            obj.mci_hgraph_valid_ratio_threshold)) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            obj.mci_hgraph_valid_ratio_threshold >= 0.0F and
                obj.mci_hgraph_valid_ratio_threshold <= 1.0F,
            "hgraph mci hgraph_valid_ratio_threshold must be in range [0, 1]");
    }
    if (read_optional_float(
            hgraph, HGRAPH_PARAMETER_BRUTE_FORCE_THRESHOLD, obj.brute_force_threshold)) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            0.0F <= obj.brute_force_threshold and obj.brute_force_threshold <= 1.0F,
            fmt::format("brute_force_threshold({}) must in range[0.0, 1.0]",
                        obj.brute_force_threshold));
    }
    if (read_optional_float(hgraph, RABITQ_QUANTIZATION_ERROR_RATE_KEY, obj.rabitq_error_rate)) {
        // One message for both checks: a NaN and a zero are the same operator mistake, and tests
        // assert on the text.
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            std::isfinite(obj.rabitq_error_rate) and obj.rabitq_error_rate > 0.0F,
            fmt::format("rabitq_error_rate must be finite and positive, got {}",
                        obj.rabitq_error_rate));
    }
    if (read_optional_float(hgraph, HGRAPH_PARAMETER_SKIP_RATIO, obj.skip_ratio)) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            0.0F <= obj.skip_ratio and obj.skip_ratio <= 1.0F,
            fmt::format("skip_ratio({}) must be in range [0.0, 1.0]", obj.skip_ratio));
    }
    if (hgraph.Contains(HGRAPH_PARAMETER_SKIP_STRATEGY)) {
        CHECK_ARGUMENT(
            hgraph[HGRAPH_PARAMETER_SKIP_STRATEGY].IsString(),
            fmt::format("parameters[{}] must be string type", HGRAPH_PARAMETER_SKIP_STRATEGY));
        obj.skip_strategy_type = parse_filter_search_skip_strategy_type(
            hgraph[HGRAPH_PARAMETER_SKIP_STRATEGY].GetString());
    }

    return obj;
}
}  // namespace vsag
