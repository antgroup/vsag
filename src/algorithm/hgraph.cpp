
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

#include "hgraph.h"

#include <datacell/compressed_graph_datacell_parameter.h>
#include <fmt/format.h>

#include <atomic>
#include <memory>
#include <stdexcept>

#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "attr/argparse.h"
#include "common.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "index/index_impl.h"
#include "index/iterator_filter.h"
#include "io/reader_io_parameter.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/options.h"

namespace vsag {

class HGraphAnalyzer;

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      ef_construct_(hgraph_param->ef_construction),
      alpha_(hgraph_param->alpha),
      odescent_param_(hgraph_param->odescent_param),
      graph_type_(hgraph_param->graph_type),
      hierarchical_datacell_param_(hgraph_param->hierarchical_graph_param),
      use_old_serial_format_(common_param.use_old_serial_format_) {
    this->label_table_->compress_duplicate_data_ = hgraph_param->support_duplicate;
    this->label_table_->support_tombstone_ = hgraph_param->support_tombstone;
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (use_reorder_) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);

    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    init_resize_bit_and_reorder();

    this->parallel_searcher_ =
        std::make_shared<ParallelSearcher>(common_param, thread_pool_, neighbors_mutex_);

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    check_and_init_raw_vector(hgraph_param->raw_vector_param, common_param);
    resize(bottom_graph_->max_capacity_);
}
void
HGraph::Train(const DatasetPtr& base) {
    int64_t total_elements = base->GetNumElements();
    int64_t dim = base->GetDim();
    DatasetPtr train_data =
        vsag::sample_train_data(base, total_elements, dim, train_sample_count_, allocator_);

    const auto* data_ptr = get_data(train_data);
    this->basic_flatten_codes_->Train(data_ptr, train_data->GetNumElements());
    if (use_reorder_) {
        this->high_precise_codes_->Train(data_ptr, train_data->GetNumElements());
    }
    if (create_new_raw_vector_) {
        // nothing to do since raw_vector_ is fp32
        this->raw_vector_->Train(data_ptr, train_data->GetNumElements());
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    this->Train(data);
    std::vector<int64_t> ret;
    if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
        ret = this->Add(data);
    } else {
        ret = this->build_by_odescent(data);
    }
    if (use_elp_optimizer_) {
        elp_optimize();
    }
    return ret;
}

JsonType
HGraph::map_hgraph_param(const JsonType& hgraph_json) {
    static const ConstParamMap external_mapping = {
        {
            HGRAPH_USE_REORDER,
            {
                USE_REORDER_KEY,
            },
        },
        {
            HGRAPH_USE_ELP_OPTIMIZER,
            {
                HGRAPH_USE_ELP_OPTIMIZER_KEY,
            },
        },
        {
            HGRAPH_IGNORE_REORDER,
            {
                HGRAPH_IGNORE_REORDER_KEY,
            },
        },
        {
            HGRAPH_BUILD_BY_BASE_QUANTIZATION,
            {
                HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY,
            },
        },
        {
            USE_ATTRIBUTE_FILTER,
            {
                USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            HGRAPH_BASE_QUANTIZATION_TYPE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            HGRAPH_BASE_IO_TYPE,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_PRECISE_IO_TYPE,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_BASE_FILE_PATH,
            {
                BASE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_FILE_PATH,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_PRECISE_QUANTIZATION_TYPE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_IO_TYPE,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_FILE_PATH,
            {
                GRAPH_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                HOLD_MOLDS,
            },
        },
        {
            STORE_RAW_VECTOR,
            {
                STORE_RAW_VECTOR_KEY,
            },
        },
        {
            RAW_VECTOR_IO_TYPE,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            RAW_VECTOR_FILE_PATH,
            {
                RAW_VECTOR_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            HGRAPH_GRAPH_MAX_DEGREE,
            {
                GRAPH_KEY,
                GRAPH_PARAM_MAX_DEGREE_KEY,
            },
        },
        {
            HGRAPH_BUILD_EF_CONSTRUCTION,
            {
                EF_CONSTRUCTION_KEY,
            },
        },
        {
            HGRAPH_BUILD_ALPHA,
            {
                ALPHA_KEY,
            },
        },
        {
            HGRAPH_INIT_CAPACITY,
            {
                GRAPH_KEY,
                GRAPH_PARAM_INIT_MAX_CAPACITY_KEY,
            },
        },
        {
            HGRAPH_GRAPH_TYPE,
            {
                GRAPH_KEY,
                GRAPH_TYPE_KEY,
            },
        },
        {
            HGRAPH_GRAPH_STORAGE_TYPE,
            {
                GRAPH_KEY,
                GRAPH_STORAGE_TYPE_KEY,
            },
        },
        {
            ODESCENT_PARAMETER_ALPHA,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_ALPHA,
            },
        },
        {
            ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_GRAPH_ITER_TURN,
            },
        },
        {
            ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
            },
        },
        {
            ODESCENT_PARAMETER_MIN_IN_DEGREE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_MIN_IN_DEGREE,
            },
        },
        {
            ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            {
                GRAPH_KEY,
                ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
            },
        },
        {
            HGRAPH_BUILD_THREAD_COUNT,
            {
                BUILD_THREAD_COUNT_KEY,
            },
        },
        {
            SQ4_UNIFORM_TRUNC_RATE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY,
            },
        },
        {
            RABITQ_PCA_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PCA_DIM_KEY,
            },
        },
        {
            INDEX_TQ_CHAIN,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TQ_CHAIN_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_QUERY,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_BASE,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY,
            },
        },
        {
            HGRAPH_BASE_PQ_DIM,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM_KEY,
            },
        },
        {
            RABITQ_USE_FHT,
            {
                BASE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                USE_FHT_KEY,
            },
        },
        {
            HGRAPH_SUPPORT_REMOVE,
            {GRAPH_KEY, GRAPH_SUPPORT_REMOVE},
        },
        {
            HGRAPH_REMOVE_FLAG_BIT,
            {GRAPH_KEY, REMOVE_FLAG_BIT},
        },
        {
            HGRAPH_SUPPORT_DUPLICATE,
            {
                SUPPORT_DUPLICATE,
            },
        },
        {
            HGRAPH_SUPPORT_TOMBSTONE,
            {
                SUPPORT_TOMBSTONE,
            },
        }};
    const std::string hgraph_params_template =
        R"(
    {
        "{TYPE_KEY}": "{INDEX_TYPE_HGRAPH}",
        "{USE_REORDER_KEY}": false,
        "{HGRAPH_USE_ENV_OPTIMIZER}": false,
        "{HGRAPH_IGNORE_REORDER_KEY}": false,
        "{HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY}": false,
        "{HGRAPH_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_VALUE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_VALUE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE_KEY}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY_KEY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8
        },
        "{BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{TQ_CHAIN_KEY}": "",
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{STORE_RAW_VECTOR_KEY}": false,
        "{RAW_VECTOR_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{CODES_TYPE_KEY}": "flatten",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{HOLD_MOLDS}": true
            }
        },
        "{BUILD_THREAD_COUNT_KEY}": 100,
        "{EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        },
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": false
        },
        "{HGRAPH_SUPPORT_DUPLICATE}": false,
        "{HGRAPH_SUPPORT_TOMBSTONE}": false,
        "{EF_CONSTRUCTION_KEY}": 400
    })";

    std::string str = format_map(hgraph_params_template, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(hgraph_json, external_mapping, inner_json);

    return inner_json;
}

bool
HGraph::Tune(const std::string& parameters, bool disable_future_tuning) {
    if (not this->index_feature_list_->CheckFeature(IndexFeature::SUPPORT_TUNE) or
        not this->has_raw_vector_) {
        return false;
    }

    // parse
    auto parsed_params = JsonType::Parse(parameters);
    JsonType hgraph_json;
    if (parsed_params.Contains(INDEX_PARAM)) {
        hgraph_json = parsed_params[INDEX_PARAM];
    }

    // map
    auto inner_json = map_hgraph_param(hgraph_json);

    // construct param obj
    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->FromJson(inner_json);
    auto inner_parameter = std::make_shared<InnerIndexParameter>();
    inner_parameter->FromJson(inner_json);

    // init new_basic_code obj
    auto common_param = this->basic_flatten_codes_->ExportCommonParam();
    auto new_basic_code =
        FlattenInterface::MakeInstance(hgraph_parameter->base_codes_param, common_param);
    FlattenInterfacePtr new_precise_code;
    if (inner_parameter->use_reorder) {
        new_precise_code =
            FlattenInterface::MakeInstance(hgraph_parameter->precise_codes_param, common_param);
    }

    std::scoped_lock lock(this->add_mutex_);

    // check which code need to tune and update create_param_ptr_
    bool is_tune_base_code = false;
    bool is_tune_precise_code = false;
    auto param = std::dynamic_pointer_cast<HGraphParameter>(create_param_ptr_);
    if (basic_flatten_codes_->GetQuantizerName() != new_basic_code->GetQuantizerName()) {
        // [case 1] base_code is not same
        is_tune_base_code = true;
    }
    if (use_reorder_ and inner_parameter->use_reorder and
        this->high_precise_codes_->GetQuantizerName() != new_precise_code->GetQuantizerName()) {
        // [case 2] precise code is not same
        is_tune_precise_code = true;
    }
    if (not inner_parameter->use_reorder) {
        // [case 3] drop precise_code
        use_reorder_ = false;
        this->high_precise_codes_.reset();
        param->precise_codes_param.reset();
        is_tune_precise_code = false;
    }
    if (not use_reorder_ and inner_parameter->use_reorder) {
        // [case 4] assign new precise_code
        use_reorder_ = true;
        is_tune_precise_code = true;
    }

    // update create_param_ptr_
    if (is_tune_base_code) {
        param->base_codes_param = hgraph_parameter->base_codes_param;
    }
    if (is_tune_precise_code) {
        param->precise_codes_param = hgraph_parameter->precise_codes_param;
    }
    param->use_reorder = use_reorder_;

    // export train data and train new_basic_code
    auto train_count = std::min(this->train_sample_count_, this->GetNumElements());
    Vector<float> train_data(train_count * dim_, 0, allocator_);
    if (is_tune_base_code or is_tune_precise_code) {
        for (InnerIdType i = 0; i < train_count; i++) {
            this->GetVectorByInnerId(i, (train_data.data() + i * dim_));
        }
    }

    auto tune_and_rebuild =
        [&](bool need_tune, FlattenInterfacePtr old_code, FlattenInterfacePtr new_code) {
            if (not need_tune) {
                return old_code;
            }

            new_code->Train(train_data.data(), train_count);

            Vector<float> insert_buffer(dim_, 0, allocator_);
            for (int64_t i = 0; i < total_count_; ++i) {
                GetVectorByInnerId(i, insert_buffer.data());
                new_code->InsertVector(static_cast<const void*>(insert_buffer.data()), i);
            }
            return new_code;
        };

    basic_flatten_codes_ =
        tune_and_rebuild(is_tune_base_code, basic_flatten_codes_, new_basic_code);
    high_precise_codes_ =
        tune_and_rebuild(is_tune_precise_code, high_precise_codes_, new_precise_code);

    check_and_init_raw_vector(param->raw_vector_param, common_param, false);
    init_resize_bit_and_reorder();

    // set status
    if (disable_future_tuning) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_TUNE, false);
        this->raw_vector_.reset();
        has_raw_vector_ = false;
        create_new_raw_vector_ = false;
    }
    return true;
}

std::vector<int64_t>
HGraph::build_by_odescent(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* extra_infos = data->GetExtraInfos();
    auto inner_ids = this->get_unique_inner_ids(total);
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    InnerIdType cur_size = 0;
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }
        InnerIdType inner_id = inner_ids.at(cur_size);
        cur_size++;
        this->label_table_->Insert(inner_id, label);
        this->basic_flatten_codes_->InsertVector(vectors + dim_ * i, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(vectors + dim_ * i, inner_id);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->InsertVector(vectors + dim_ * i, inner_id);
        }
        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(allocator_);
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    this->resize(total_count_);
    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    return failed_ids;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data, AddMode mode) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    }
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.float_vector is nullptr");

    {
        std::scoped_lock lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const void* data,
                        int level,
                        InnerIdType inner_id,
                        const char* extra_info,
                        const AttributeSet* attrs) -> void {
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_info, inner_id);
        }
        if (attrs != nullptr and this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(*attrs, inner_id);
        }
        this->add_one_point(data, level, inner_id);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        InnerIdType inner_id;

        // try recover tombstone
        if (this->data_type_ != DataTypes::DATA_TYPE_SPARSE) {
            auto one_base = get_single_dataset(data, j);
            bool is_process_finished = try_recover_tombstone(one_base, failed_ids);
            if (is_process_finished) {
                continue;
            }
        }

        {
            std::scoped_lock lock(this->add_mutex_);
            inner_id = this->get_unique_inner_ids(1).at(0);
            uint64_t new_count = total_count_;
            this->resize(new_count);
        }

        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            this->label_table_->Insert(inner_id, labels[j]);
            inner_ids.emplace_back(inner_id, j);
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        int level;
        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            level = this->get_random_level() - 1;
        }
        const auto* extra_info = extra_infos + local_idx * extra_info_size_;
        const AttributeSet* cur_attr_set = nullptr;
        if (attr_sets != nullptr) {
            cur_attr_set = attr_sets + local_idx;
        }
        if (this->thread_pool_ != nullptr) {
            auto future = this->thread_pool_->GeneralEnqueue(
                add_func, get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
            futures.emplace_back(std::move(future));
        } else {
            add_func(get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
        }
    }
    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return failed_ids;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, nullptr);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.filter_ = filter;
    req.params_str_ = parameters;
    req.search_allocator_ = allocator;
    return this->SearchWithRequest(req);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator,
                  IteratorContext*& iter_ctx,
                  bool is_last_filter) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = allocator_, .stats = &stats};
    if (allocator != nullptr) {
        ctx.alloc = allocator;
    }

    if (GetNumElements() == 0) {
        return DatasetImpl::MakeEmptyDataset();
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(parameters);
    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock shared_lock(this->global_mutex_);
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (iter_ctx == nullptr) {
        auto cur_count = this->bottom_graph_->TotalCount();

        if (cur_count == 0) {
            SearchStatistics stats;
            auto dataset_result = DatasetImpl::MakeEmptyDataset();
            dataset_result->Statistics(stats.Dump());
            return dataset_result;
        }
        auto* new_ctx = new IteratorFilterContext();
        if (auto ret = new_ctx->init(cur_count, params.ef_search, ctx.alloc); not ret.has_value()) {
            delete new_ctx;
            throw vsag::VsagException(ErrorType::INTERNAL_ERROR,
                                      "failed to init IteratorFilterContext");
        }
        iter_ctx = new_ctx;
    }

    auto* iter_filter_ctx = static_cast<IteratorFilterContext*>(iter_ctx);
    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(ctx.alloc, k);
    const auto* query_data = get_data(query);
    if (is_last_filter) {
        while (!iter_filter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_filter_ctx->GetTopID();
            float cur_dist = iter_filter_ctx->GetTopDist();
            search_result->Push(cur_dist, cur_inner_id);
            iter_filter_ctx->PopDiscard();
        }
    } else {
        InnerSearchParam search_param;
        search_param.ep = this->entry_point_id_;
        search_param.topk = 1;
        search_param.ef = 1;
        search_param.is_inner_id_allowed = nullptr;
        if (iter_filter_ctx->IsFirstUsed()) {
            for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                auto result = this->search_one_graph(query_data,
                                                     this->route_graphs_[i],
                                                     this->basic_flatten_codes_,
                                                     search_param,
                                                     (VisitedListPtr) nullptr,
                                                     &ctx);
                search_param.ep = result->Top().second;
            }
        }

        search_param.ef = std::max(params.ef_search, k);
        search_param.is_inner_id_allowed = ft;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        search_param.parallel_search_thread_count = params.parallel_search_thread_count;

        search_result = this->search_one_graph(query_data,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               iter_filter_ctx,
                                               &ctx);
    }

    if (use_reorder_) {
        this->reorder(
            query_data, this->high_precise_codes_, search_result, k, iter_filter_ctx, ctx);
    }

    while (search_result->Size() > k) {
        auto curr = search_result->Top();
        iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        iter_filter_ctx->SetPoint(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    iter_filter_ctx->SetOFFFirstUsed();

    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto block_memory_ceil = [](uint64_t memory, uint64_t block_size) -> uint64_t {
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(memory) / static_cast<double>(block_size)) *
            static_cast<double>(block_size));
    };

    if (this->basic_flatten_codes_->InMemory()) {
        auto base_memory = this->basic_flatten_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(base_memory, block_size);
    }

    if (bottom_graph_->InMemory()) {
        auto bottom_graph_memory =
            (this->bottom_graph_->maximum_degree_ + 1) * sizeof(InnerIdType) * element_count;
        estimate_memory += block_memory_ceil(bottom_graph_memory, block_size);
    }

    if (use_reorder_ && this->high_precise_codes_->InMemory() && not this->ignore_reorder_) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        auto extra_info_memory = this->extra_infos_->ExtraInfoSize() * element_count;
        estimate_memory += block_memory_ceil(extra_info_memory, block_size);
    }

    auto label_map_memory =
        element_count * (sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory += label_map_memory;

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    estimate_memory += static_cast<uint64_t>(sparse_graph_memory);

    auto other_memory = element_count * (sizeof(LabelType) + sizeof(std::shared_mutex) +
                                         sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory += other_memory;

    return estimate_memory;
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(hierarchical_datacell_param_, this->allocator_);
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         const VisitedListPtr& vt,
                         QueryContext* ctx) const {
    bool new_visited_list = vt == nullptr;
    VisitedListPtr visited_list;
    if (new_visited_list) {
        visited_list = this->pool_->TakeOne();
    } else {
        visited_list = vt;
        visited_list->Reset();
    }
    DistHeapPtr result = nullptr;
    if (inner_search_param.parallel_search_thread_count > 1) {
        result = this->parallel_searcher_->Search(
            graph, flatten, visited_list, query, inner_search_param);
    } else {
        result = this->searcher_->Search(
            graph, flatten, visited_list, query, inner_search_param, this->label_table_, ctx);
    }
    if (new_visited_list) {
        this->pool_->ReturnOne(visited_list);
    }
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         IteratorFilterContext* iter_ctx,
                         QueryContext* ctx) const {
    auto visited_list = this->pool_->TakeOne();
    auto result = this->searcher_->Search(
        graph, flatten, visited_list, query, inner_search_param, iter_ctx, ctx);
    this->pool_->ReturnOne(visited_list);
    return result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    SearchStatistics stats;
    QueryContext ctx{.stats = &stats};

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (filter != nullptr) {
        combined_filter->AppendFilter(
            std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_));
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(raw_query,
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param,
                                             (VisitedListPtr) nullptr,
                                             &ctx);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    CHECK_ARGUMENT((1 <= params.ef_search) and (params.ef_search <= 1000),  // NOLINT
                   fmt::format("ef_search({}) must in range[1, 1000]", params.ef_search));
    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    search_param.consider_duplicate = true;
    search_param.range_search_limit_size = static_cast<int>(limited_size);
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    auto search_result = this->search_one_graph(raw_query,
                                                this->bottom_graph_,
                                                this->basic_flatten_codes_,
                                                search_param,
                                                (VisitedListPtr) nullptr,
                                                &ctx);

    if (use_reorder_) {
        this->reorder(
            raw_query, this->high_precise_codes_, search_result, limited_size, nullptr, ctx);
    }

    if (limited_size > 0) {
        while (search_result->Size() > limited_size) {
            search_result->Pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, allocator_);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos =
            static_cast<char*>(allocator_->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }

    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

void
HGraph::serialize_basic_info_v0_14(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    uint64_t max_level = this->route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    auto capacity = this->max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);

    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_basic_info_v0_14(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    this->max_capacity_.store(capacity);
    StreamReader::ReadVector(reader, this->label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
}

#define TO_JSON_BASE64(json_obj, var) json_obj[#var].SetString(base64_encode_obj(this->var##_));

JsonType
HGraph::serialize_basic_info() const {
    JsonType jsonify_basic_info;
    jsonify_basic_info["use_reorder"].SetBool(this->use_reorder_);
    jsonify_basic_info["dim"].SetInt(this->dim_);
    jsonify_basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));
    jsonify_basic_info["entry_point_id"].SetInt(this->entry_point_id_);
    jsonify_basic_info["ef_construct"].SetInt(this->ef_construct_);
    jsonify_basic_info["extra_info_size"].SetInt(this->extra_info_size_);
    jsonify_basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    // logger::debug("mult: {}", this->mult_);
    TO_JSON_BASE64(jsonify_basic_info, mult);
    jsonify_basic_info["max_capacity"].SetInt(this->max_capacity_.load());
    jsonify_basic_info["max_level"].SetInt(this->route_graphs_.size());
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());

    return jsonify_basic_info;
}

#define FROM_JSON(json_obj, var, type)                   \
    do {                                                 \
        if ((json_obj).Contains(#var)) {                 \
            this->var##_ = (json_obj)[#var].Get##type(); \
        }                                                \
    } while (0)

#define FROM_JSON_BASE64(json_obj, var) \
    base64_decode_obj((json_obj)[#var].GetString(), this->var##_);

void
HGraph::deserialize_basic_info(const JsonType& jsonify_basic_info) {
    logger::debug("jsonify_basic_info: {}", jsonify_basic_info.Dump());
    FROM_JSON(jsonify_basic_info, use_reorder, Bool);
    FROM_JSON(jsonify_basic_info, dim, Int);
    if (jsonify_basic_info.Contains("metric")) {
        this->metric_ = static_cast<MetricType>(jsonify_basic_info["metric"].GetInt());
    }
    FROM_JSON(jsonify_basic_info, entry_point_id, Int);
    FROM_JSON(jsonify_basic_info, ef_construct, Int);
    FROM_JSON(jsonify_basic_info, extra_info_size, Int);
    if (jsonify_basic_info.Contains("data_type")) {
        this->data_type_ = static_cast<DataTypes>(jsonify_basic_info["data_type"].GetInt());
    }
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    // logger::debug("mult: {}", this->mult_);
    this->max_capacity_.store(jsonify_basic_info["max_capacity"].GetInt());

    auto max_level = jsonify_basic_info["max_level"].GetInt();
    for (int64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    if (jsonify_basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = jsonify_basic_info[INDEX_PARAM].GetString();
        HGraphParameterPtr index_param = std::make_shared<HGraphParameter>();
        index_param->data_type = this->data_type_;
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("HGraph index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
}

void
HGraph::serialize_label_info(StreamWriter& writer) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Serialize(writer);
        return;
    }
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);
    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::deserialize_label_info(StreamReader& reader) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Deserialize(reader);
        return;
    }
    StreamReader::ReadVector(reader, this->label_table_->label_table_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
}

void
HGraph::Serialize(StreamWriter& writer) const {
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }

    // FIXME(wxyu): this option is used for special purposes, like compatibility testing
    if (this->use_old_serial_format_) {
        this->serialize_basic_info_v0_14(writer);
        this->basic_flatten_codes_->Serialize(writer);
        this->bottom_graph_->Serialize(writer);
        if (this->use_reorder_) {
            this->high_precise_codes_->Serialize(writer);
        }
        for (const auto& route_graph : this->route_graphs_) {
            route_graph->Serialize(writer);
        }
        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Serialize(writer);
        }
        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Serialize(writer);
        }
        return;
    }

    this->serialize_label_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->use_reorder_) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : this->route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    auto jsonify_basic_info = this->serialize_basic_info();
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    logger::debug(jsonify_basic_info.Dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
HGraph::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        this->deserialize_basic_info_v0_14(reader);

        this->basic_flatten_codes_->Deserialize(reader);
        this->bottom_graph_->Deserialize(reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        BufferStreamReader buffer_reader(
            &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

        auto metadata = footer->GetMetadata();
        // metadata should NOT be nullptr if footer is not nullptr
        this->deserialize_basic_info(metadata->Get(BASIC_INFO));
        this->deserialize_label_info(buffer_reader);

        this->basic_flatten_codes_->Deserialize(buffer_reader);
        this->bottom_graph_->Deserialize(buffer_reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(buffer_reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }

        if (create_new_raw_vector_) {
            this->raw_vector_->Deserialize(buffer_reader);
        }
        if (this->raw_vector_ != nullptr) {
            this->has_raw_vector_ = true;
        }
    }
    this->cal_memory_usage();

    // post serialize procedure
    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

std::string
HGraph::GetMemoryUsageDetail() const {
    JsonType memory_usage;
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }
    memory_usage["basic_flatten_codes"].SetInt(this->basic_flatten_codes_->CalcSerializeSize());
    memory_usage["bottom_graph"].SetInt(this->bottom_graph_->CalcSerializeSize());
    if (this->use_reorder_) {
        memory_usage["high_precise_codes"].SetInt(this->high_precise_codes_->CalcSerializeSize());
    }
    uint64_t route_graph_size = 0;
    for (const auto& route_graph : this->route_graphs_) {
        route_graph_size += route_graph->CalcSerializeSize();
    }
    memory_usage["route_graph"].SetInt(route_graph_size);
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        memory_usage["extra_infos"].SetInt(this->extra_infos_->CalcSerializeSize());
    }
    memory_usage["__total_size__"].SetInt(this->CalSerializeSize());
    return memory_usage.Dump();
}

float
HGraph::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
HGraph::CalDistanceById(const float* query,
                        const int64_t* ids,
                        int64_t count,
                        bool calculate_precise_distance) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_ && calculate_precise_distance) {
        flat = this->high_precise_codes_;
    }
    if (create_new_raw_vector_ && calculate_precise_distance) {
        flat = this->raw_vector_;
    }
    return InnerIndexInterface::cal_distance_by_id(query, ids, count, flat);
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (this->label_table_->IsRemoved(i)) {
            continue;
        }
        auto label = this->label_table_->GetLabelById(i);
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id) {
    {
        std::shared_lock add_lock(add_mutex_);
        this->basic_flatten_codes_->InsertVector(data, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(data, inner_id);
        }
        if (create_new_raw_vector_) {
            raw_vector_->InsertVector(data, inner_id);
        }
    }
    std::unique_lock add_lock(add_mutex_);
    if (level >= static_cast<int>(this->route_graphs_.size()) || bottom_graph_->TotalCount() == 0) {
        std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
        // level maybe a negative number(-1)
        for (auto j = static_cast<int>(this->route_graphs_.size()); j <= level; ++j) {
            this->route_graphs_.emplace_back(this->generate_one_route_graph());
        }
        auto insert_success = this->graph_add_one(data, level, inner_id);
        if (insert_success) {
            entry_point_id_ = inner_id;
        } else {
            this->route_graphs_.pop_back();
        }
        add_lock.unlock();
    } else {
        add_lock.unlock();
        std::shared_lock rlock(this->global_mutex_);
        this->graph_add_one(data, level, inner_id);
    }
}

bool
HGraph::graph_add_one(const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param;
    param.topk = 1;
    param.ep = this->entry_point_id_;
    param.ef = 1;
    param.is_inner_id_allowed = nullptr;

    LockGuard cur_lock(neighbors_mutex_, inner_id);
    auto flatten_codes = basic_flatten_codes_;
    if (use_reorder_ and not build_by_base_) {
        flatten_codes = high_precise_codes_;
    }

    for (auto j = this->route_graphs_.size() - 1; j > level; --j) {
        result = search_one_graph(
            data, route_graphs_[j], flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
        param.ep = result->Top().second;
    }

    param.ef = this->ef_construct_;
    param.topk = static_cast<int64_t>(ef_construct_);
    if (this->label_table_->CompressDuplicateData()) {
        param.find_duplicate = true;
    }

    if (bottom_graph_->TotalCount() != 0) {
        result = search_one_graph(data,
                                  this->bottom_graph_,
                                  flatten_codes,
                                  param,
                                  // to specify which overloaded function to call
                                  (VisitedListPtr) nullptr,
                                  nullptr);
        if (this->label_table_->CompressDuplicateData() && param.duplicate_id >= 0) {
            std::unique_lock lock(this->label_lookup_mutex_);
            label_table_->SetDuplicateId(static_cast<InnerIdType>(param.duplicate_id), inner_id);
            return false;
        }
        mutually_connect_new_element(inner_id,
                                     result,
                                     this->bottom_graph_,
                                     flatten_codes,
                                     neighbors_mutex_,
                                     allocator_,
                                     alpha_);
    } else {
        bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    }

    for (int64_t j = 0; j <= level; ++j) {
        if (route_graphs_[j]->TotalCount() != 0) {
            result = search_one_graph(data,
                                      route_graphs_[j],
                                      flatten_codes,
                                      param,
                                      // to specify which overloaded function to call
                                      (VisitedListPtr) nullptr,
                                      nullptr);
            mutually_connect_new_element(inner_id,
                                         result,
                                         route_graphs_[j],
                                         flatten_codes,
                                         neighbors_mutex_,
                                         allocator_,
                                         alpha_);
        } else {
            route_graphs_[j]->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        }
    }
    return true;
}

void
HGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_.load();
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::scoped_lock lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size_power_2, allocator_);
        this->label_table_->Resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);
        this->basic_flatten_codes_->Resize(new_size_power_2);
        if (use_reorder_) {
            this->high_precise_codes_->Resize(new_size_power_2);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->Resize(new_size_power_2);
        }
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
    }
}
void
HGraph::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_MERGE_INDEX,
    });
    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_KNN_ITERATOR_FILTER_SEARCH,
    });
    // update
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT});
    }
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT});
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_DELETE_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });
    // other
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_ESTIMATE_MEMORY,
                                            IndexFeature::SUPPORT_GET_MEMORY_USAGE,
                                            IndexFeature::SUPPORT_CHECK_ID_EXIST,
                                            IndexFeature::SUPPORT_CLONE,
                                            IndexFeature::SUPPORT_EXPORT_MODEL,
                                            IndexFeature::SUPPORT_TUNE});

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    bool have_fp32 = false;
    bool hold_molds = false;
    if (name == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->basic_flatten_codes_->HoldMolds();
    }
    if (use_reorder_ and not ignore_reorder_ and
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->high_precise_codes_->HoldMolds();
    }
    if (have_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
        if (metric_ != MetricType::METRIC_TYPE_COSINE || hold_molds) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
        }
    }

    if (raw_vector_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
    }

    // metric
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }

    if (this->extra_infos_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT);
    }
}

void
HGraph::elp_optimize() {
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 80;
    param.topk = 10;
    param.is_inner_id_allowed = nullptr;
    searcher_->SetMockParameters(bottom_graph_, basic_flatten_codes_, pool_, param, dim_);
    // TODO(ZXY): optimize PREFETCH_DEPTH_CODE and add default value for the others
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    optimizer_->Optimize(searcher_);
}

void
HGraph::reorder(const void* query,
                const FlattenInterfacePtr& flatten,
                DistHeapPtr& candidate_heap,
                int64_t k,
                IteratorFilterContext* iter_ctx,
                QueryContext& ctx) const {
    uint64_t size = candidate_heap->Size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    auto reorder_heap =
        reorder_->Reorder(candidate_heap, static_cast<const float*>(query), k, ctx, iter_ctx);
    candidate_heap = reorder_heap;
}

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    auto inner_json = map_hgraph_param(external_param);
    if (common_param.data_type_ == DataTypes::DATA_TYPE_SPARSE) {
        inner_json[BASE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[PRECISE_CODES_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
        inner_json[RAW_VECTOR_KEY][CODES_TYPE_KEY].SetString(SPARSE_CODES);
    }

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(inner_json);
    uint64_t max_degree = hgraph_parameter->bottom_graph_param->max_degree_;

    auto max_degree_threshold = std::max<int64_t>(common_param.dim_, 128);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= max_degree) and (max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold = std::max<uint64_t>(1000UL, AMPLIFICATION_FACTOR * max_degree);
    CHECK_ARGUMENT((max_degree <= hgraph_parameter->ef_construction) and  // NOLINT
                       (hgraph_parameter->ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               hgraph_parameter->ef_construction,
                               max_degree,
                               construction_threshold));
    return hgraph_parameter;
}
InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (use_reorder_) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}
void
HGraph::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    if (raw_vector_ != nullptr) {
        raw_vector_->GetCodesById(inner_id, data);
        return;
    }

    if (use_reorder_) {
        high_precise_codes_->GetCodesById(inner_id, data);
    } else {
        basic_flatten_codes_->GetCodesById(inner_id, data);
    }
}

uint32_t
HGraph::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
        return delete_count;
    }
    for (const auto& id : ids) {
        InnerIdType inner_id;
        {
            std::shared_lock lock(this->label_lookup_mutex_);
            inner_id = this->label_table_->GetIdByLabel(id);
        }
        if (inner_id == this->entry_point_id_) {
            bool find_new_ep = false;
            while (not route_graphs_.empty()) {
                auto& upper_graph = route_graphs_.back();
                Vector<InnerIdType> neighbors(allocator_);
                upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
                for (const auto& nb_id : neighbors) {
                    if (inner_id == nb_id) {
                        continue;
                    }
                    this->entry_point_id_ = nb_id;
                    find_new_ep = true;
                    break;
                }
                if (find_new_ep) {
                    break;
                }
                route_graphs_.pop_back();
            }
        }
        {
            {
                std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
                for (int level = static_cast<int>(route_graphs_.size()) - 1; level >= 0; --level) {
                    this->route_graphs_[level]->DeleteNeighborsById(inner_id);
                }
                this->bottom_graph_->DeleteNeighborsById(inner_id);
            }
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            this->label_table_->MarkRemove(id);
            delete_count++;
        }
    }
    return delete_count;
}

void
HGraph::recover_remove(int64_t id) {
    // note:
    // 1. this function doesn't recover entry_point and route_graphs caused by Remove()
    // 2. use this function only when is_tombstone is checked

    std::shared_lock label_lock(this->label_lookup_mutex_);
    auto inner_id = this->label_table_->GetIdByLabel(id, true);
    this->bottom_graph_->RecoverDeleteNeighborsById(inner_id);
    this->label_table_->RecoverRemove(id);
    delete_count_--;
}

DatasetPtr
HGraph::get_single_dataset(const DatasetPtr& data, uint32_t j) {
    void* vectors = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, data, &vectors, &data_size);
    const auto* labels = data->GetIds();
    auto one_data = Dataset::Make();
    one_data->Ids(labels + j)
        ->Float32Vectors((float*)((char*)vectors + data_size * j))
        ->Int8Vectors((int8_t*)((char*)vectors + data_size * j))
        ->NumElements(1)
        ->Owner(false);
    return one_data;
}

bool
HGraph::try_recover_tombstone(const DatasetPtr& data, std::vector<int64_t>& failed_ids) {
    /*
     * return:
     *      True : No processing required — data already exists or was recovered successfully
     *      False: Processing required — data not found or recovery failed
     *
     *
     * [case 1] fail to insert -> continue + record failed id
     * exist + not delete : is_label_valid = true, is_tombstone = false
     *
     * [case 2] fail to recovery -> add process
     * exist + delete + not recovery: is_label_valid = false, is_tombstone = ture, is_recovered = false
     *
     * [case 3] tombstone recovery -> continue
     * exist + delete + recovery: is_label_valid = false, is_tombstone = ture, is_recovered = true
     *
     * [case 4] no old point -> add process
     * not exists + not delete: is_label_valid = false, is_tombstone = false
     *
     * [case 5] error
     * exists + deleted: is_label_valid = true, is_tombstone = true
     */

    auto label = data->GetIds()[0];

    bool is_label_valid = false;
    bool is_tombstone = false;
    bool is_recovered = false;
    {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        is_label_valid = this->label_table_->CheckLabel(label);
        if (not is_label_valid) {
            is_tombstone = this->label_table_->IsTombstoneLabel(label);
        }
    }

    if (is_tombstone) {
        try {
            // try recover and update
            recover_remove(label);
            auto update_res = UpdateVector(label, data, false);
            if (update_res) {
                // [case 3]
                is_recovered = true;
                return is_recovered;
            }
            // recover failed: roll back
            Remove({label});
        } catch (std::runtime_error& e) {
            // recover failed: roll back
            Remove({label});
        }
    }

    // is_recovered = false
    if (is_label_valid) {
        // [case 1]
        failed_ids.emplace_back(label);
        return true;
    }

    // [case 2, 4]
    return false;
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, this->total_count_);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (use_reorder_) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, this->total_count_);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, this->total_count_);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < other_index->route_graphs_.size(); ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], this->total_count_);
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        Vector<InnerIdType> neighbors(this->allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        neighbors.resize(neighbors.size() / 2);
        this->bottom_graph_->InsertNeighborsById(inner_id, neighbors);
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}

void
HGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    auto codes = (use_reorder_) ? high_precise_codes_ : basic_flatten_codes_;
    codes = (create_new_raw_vector_) ? raw_vector_ : codes;
    bool release;
    const auto* buffer = codes->GetCodesById(inner_id, release);
    codes->Decode(buffer, data);
    if (release) {
        codes->Release(buffer);
    }
}

void
HGraph::SetImmutable() {
    if (this->immutable_) {
        return;
    }
    std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
    this->neighbors_mutex_.reset();
    this->neighbors_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->neighbors_mutex_);
    this->immutable_ = true;
}

void
HGraph::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    if (use_reorder_) {
        high_precise_codes_->InitIO(reader_param);
    }
    basic_flatten_codes_->InitIO(reader_param);
    bottom_graph_->InitIO(reader_param);
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = this->allocator_, .stats = &stats};
    if (request.search_allocator_ != nullptr) {
        ctx.alloc = request.search_allocator_;
    }

    const auto& query = request.query_;
    int64_t query_dim = query->GetDim();
    auto k = request.topk_;
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);

    auto ef_search_threshold = std::max<int64_t>(AMPLIFICATION_FACTOR * k, 1000);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    std::shared_lock shared_lock(this->global_mutex_);
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;

    if (search_param.ep == INVALID_ENTRY_POINT) {
        SearchStatistics stats;
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(stats.Dump());
        return dataset_result;
    }

    auto vt = this->pool_->TakeOne();

    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param, vt, &ctx);
        search_param.ep = result->Top().second;
    }

    auto combined_filter = std::make_shared<CombinedFilter>();
    combined_filter->AppendFilter(this->label_table_->GetDeletedIdsFilter());
    if (request.filter_ != nullptr) {
        if (params.use_extra_info_filter) {
            combined_filter->AppendFilter(
                std::make_shared<ExtraInfoWrapperFilter>(request.filter_, this->extra_infos_));
        } else {
            combined_filter->AppendFilter(
                std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_));
        }
    }
    FilterPtr ft = nullptr;
    if (not combined_filter->IsEmpty()) {
        ft = combined_filter;
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    search_param.topk = static_cast<int64_t>(search_param.ef);
    if (params.topk_factor > 1.0F) {
        search_param.topk = std::min(
            search_param.topk, static_cast<int64_t>(static_cast<float>(k) * params.topk_factor));
    }
    search_param.consider_duplicate = true;
    if (params.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(params.timeout_ms);
        stats.is_timeout.store(false, std::memory_order_relaxed);
    }
    search_param.parallel_search_thread_count = params.parallel_search_thread_count;

    auto search_result = this->search_one_graph(
        raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param, vt, &ctx);

    this->pool_->ReturnOne(vt);

    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, k, nullptr, ctx);
    }

    while (search_result->Size() > k) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        auto dataset_result = DatasetImpl::MakeEmptyDataset();
        dataset_result->Statistics(stats.Dump());
        return dataset_result;
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, ctx.alloc);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        extra_infos =
            static_cast<char*>(ctx.alloc->Allocate(extra_info_size_ * search_result->Size()));
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    dataset_results->Statistics(stats.Dump());
    return std::move(dataset_results);
}

void
HGraph::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraph::UpdateAttribute(int64_t id,
                        const AttributeSet& new_attrs,
                        const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

const static uint64_t QUERY_SAMPLE_SIZE = 10;
const static int64_t DEFAULT_TOPK = 100;

std::string
HGraph::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = DEFAULT_TOPK;
    analyzer_param.base_sample_size = std::min(QUERY_SAMPLE_SIZE, this->total_count_.load());
    analyzer_param.search_params =
        fmt::format(R"({{"hgraph": {{"ef_search": {}}}}})", ef_construct_);
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->GetStats();
    return stats.Dump(4);
}

void
HGraph::init_resize_bit_and_reorder() {
    auto step_block_size = Options::Instance().block_size_limit();
    auto block_size_per_vector = this->basic_flatten_codes_->code_size_;
    block_size_per_vector =
        std::max(block_size_per_vector,
                 static_cast<uint32_t>(this->bottom_graph_->maximum_degree_ * sizeof(InnerIdType)));
    if (use_reorder_) {
        block_size_per_vector =
            std::max(block_size_per_vector, this->high_precise_codes_->code_size_);
        reorder_ = std::make_shared<FlattenReorder>(this->high_precise_codes_, allocator_);
    }
    if (this->extra_infos_ != nullptr) {
        block_size_per_vector =
            std::max<int64_t>(block_size_per_vector, static_cast<uint32_t>(this->extra_info_size_));
    }
    auto increase_count = step_block_size / block_size_per_vector;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
}

void
HGraph::check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                                  const IndexCommonParam& common_param,
                                  bool is_create_new) {
    if (raw_vector_param == nullptr) {
        return;
    }

    if (is_create_new) {
        raw_vector_ = FlattenInterface::MakeInstance(raw_vector_param, common_param);
    }

    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ == nullptr) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }
    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    auto io_type_name = raw_vector_param->io_parameter->GetTypeName();
    if (io_type_name != IO_TYPE_VALUE_BLOCK_MEMORY_IO and io_type_name != IO_TYPE_VALUE_MEMORY_IO) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    if (basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = basic_flatten_codes_;
        has_raw_vector_ = true;
        return;
    }

    if (high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = high_precise_codes_;
        has_raw_vector_ = true;
        return;
    }
}

bool
HGraph::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    // check if id exists and get copied base data
    uint32_t inner_id = 0;
    {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        inner_id = this->label_table_->GetIdByLabel(id);
    }

    // the validation of the new vector
    void* new_base_vec = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, new_base, &new_base_vec, &data_size);

    if (not force_update) {
        std::shared_lock label_lock(this->label_lookup_mutex_);

        // 1. check whether vectors are same
        Vector<int8_t> base_data(data_size, allocator_);
        GetVectorByInnerId(inner_id, (float*)base_data.data());
        float old_self_dist = this->CalcDistanceById((float*)base_data.data(), id);
        float self_dist = this->CalcDistanceById((float*)new_base_vec, id);
        if (std::abs(old_self_dist - self_dist) < 1e-3) {
            return true;
        }

        // 2. check whether the neighborhood relationship is same
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        for (auto neighbor_inner_id : neighbors) {
            // don't compare with itself
            if (neighbor_inner_id == inner_id) {
                continue;
            }

            float neighbor_dist = 0;
            try {
                neighbor_dist =
                    this->CalcDistanceById(static_cast<float*>(new_base_vec),
                                           this->label_table_->GetLabelById(neighbor_inner_id));
            } catch (const std::runtime_error& e) {
                // incase that neighbor has been deleted
                continue;
            }
            if (neighbor_dist < self_dist) {
                return false;
            }
        }
    }

    // note that only modify vector need to obtain unique lock
    // and the lock has been obtained inside datacell
    auto codes = (use_reorder_) ? high_precise_codes_ : basic_flatten_codes_;
    bool update_status = basic_flatten_codes_->UpdateVector(new_base_vec, inner_id);
    if (use_reorder_) {
        update_status = update_status && high_precise_codes_->UpdateVector(new_base_vec, inner_id);
    }
    return update_status;
}

std::string
HGraph::AnalyzeIndexBySearch(const SearchRequest& request) {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->AnalyzeIndexBySearch(request);
    return stats.Dump(4);
}

void
HGraph::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
HGraph::cal_memory_usage() {
    auto memory = sizeof(HGraph);
    memory += this->neighbors_mutex_->GetMemoryUsage();
    memory += this->pool_->GetMemoryUsage();
    memory += this->label_table_->GetMemoryUsage();
    memory += this->basic_flatten_codes_->GetMemoryUsage();
    memory += this->bottom_graph_->GetMemoryUsage();
    for (auto& graph : this->route_graphs_) {
        memory += graph->GetMemoryUsage();
    }
    if (use_reorder_) {
        memory += this->high_precise_codes_->GetMemoryUsage();
    }

    if (this->extra_infos_ != nullptr and this->extra_info_size_ > 0) {
        memory += this->extra_infos_->GetMemoryUsage();
    }

    if (this->create_new_raw_vector_ and this->raw_vector_ != nullptr) {
        memory += raw_vector_->GetMemoryUsage();
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

}  // namespace vsag
