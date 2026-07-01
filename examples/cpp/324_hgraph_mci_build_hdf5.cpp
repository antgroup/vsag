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

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string data_path{"/root/data/wufufilter-5m-128-euclidean.h5"};
    std::string index_path{};
    std::string metric{};
    std::string quantization{"sq8"};
    std::string base_io_type{"block_memory_io"};
    uint64_t limit{0};
    uint64_t max_degree{32};
    uint64_t ef_construction{400};
    uint64_t build_thread_count{100};
    uint64_t graph_iter_turn{30};
    uint64_t mci_mcs{200};
    uint64_t mci_clique_max{50};
    float mci_alpha{1.2F};
    float hgraph_valid_ratio_threshold{1.0F};
    std::string mci_knng_path{};
    bool store_raw_vector{true};
};

void
check(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename E>
void
check_expected(const tl::expected<T, E>& result, const std::string& prefix) {
    if (not result.has_value()) {
        throw std::runtime_error(prefix + result.error().message);
    }
}

uint64_t
parse_u64(const std::string& value) {
    return static_cast<uint64_t>(std::stoull(value));
}

float
parse_float(const std::string& value) {
    return std::stof(value);
}

Options
parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) {
            check(i + 1 < argc, "missing value for " + name);
            return std::string(argv[++i]);
        };
        if (key == "--data") {
            options.data_path = need_value(key);
        } else if (key == "--index") {
            options.index_path = need_value(key);
        } else if (key == "--metric") {
            options.metric = need_value(key);
        } else if (key == "--quantization") {
            options.quantization = need_value(key);
        } else if (key == "--base-io-type") {
            options.base_io_type = need_value(key);
        } else if (key == "--limit") {
            options.limit = parse_u64(need_value(key));
        } else if (key == "--max-degree") {
            options.max_degree = parse_u64(need_value(key));
        } else if (key == "--ef-construction") {
            options.ef_construction = parse_u64(need_value(key));
        } else if (key == "--build-thread-count") {
            options.build_thread_count = parse_u64(need_value(key));
        } else if (key == "--graph-iter-turn") {
            options.graph_iter_turn = parse_u64(need_value(key));
        } else if (key == "--mci-mcs") {
            options.mci_mcs = parse_u64(need_value(key));
        } else if (key == "--mci-clique-max") {
            options.mci_clique_max = parse_u64(need_value(key));
        } else if (key == "--mci-alpha") {
            options.mci_alpha = parse_float(need_value(key));
        } else if (key == "--mci-knng-path") {
            options.mci_knng_path = need_value(key);
        } else if (key == "--hgraph-valid-ratio-threshold") {
            options.hgraph_valid_ratio_threshold = parse_float(need_value(key));
        } else if (key == "--no-store-raw-vector") {
            options.store_raw_vector = false;
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return options;
}

std::string
infer_metric(const std::string& path, const std::string& override_metric) {
    if (not override_metric.empty()) {
        return override_metric;
    }
    if (path.find("angular") != std::string::npos) {
        return "cosine";
    }
    return "l2";
}

std::string
default_index_path(const std::string& data_path) {
    auto slash = data_path.find_last_of('/');
    auto name = slash == std::string::npos ? data_path : data_path.substr(slash + 1);
    return "/tmp/" + name + ".hgraph_mci.index";
}

std::pair<uint64_t, uint64_t>
get_matrix_shape(const H5::H5File& file, const std::string& dataset_name) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    check(dataspace.getSimpleExtentNdims() == 2, dataset_name + " must be a 2-D dataset");
    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims, nullptr);
    return {static_cast<uint64_t>(dims[0]), static_cast<uint64_t>(dims[1])};
}

std::vector<float>
read_float_matrix_prefix(const H5::H5File& file,
                         const std::string& dataset_name,
                         uint64_t rows,
                         uint64_t dim) {
    auto dataset = file.openDataSet(dataset_name);
    auto dataspace = dataset.getSpace();
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {rows, dim};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memspace(2, count);
    std::vector<float> values(rows * dim);
    dataset.read(values.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
    return values;
}

std::string
make_hgraph_mci_params(const Options& options, uint64_t dim, const std::string& metric) {
    std::ostringstream builder;
    builder << R"({
    "dtype": "float32",
    "metric_type": ")"
            << metric << R"(",
    "dim": )"
            << dim << R"(,
    "index_param": {
        "base_quantization_type": ")"
            << options.quantization << R"(",
        "base_io_type": ")"
            << options.base_io_type << R"(",
        "store_raw_vector": )"
            << (options.store_raw_vector ? "true" : "false") << R"(,
        "raw_vector_io_type": "memory_io",
        "graph_type": "odescent",
        "max_degree": )"
            << options.max_degree << R"(,
        "ef_construction": )"
            << options.ef_construction << R"(,
        "build_thread_count": )"
            << options.build_thread_count << R"(,
        "alpha": 1.2,
        "graph_iter_turn": )"
            << options.graph_iter_turn << R"(,
        "neighbor_sample_rate": 0.2,
        "mci": {
            "use_mci": true,
            "mcs": )"
            << options.mci_mcs << R"(,
            "clique_max": )"
            << options.mci_clique_max << R"(,
            "alpha": )"
            << options.mci_alpha << R"(,
            "hgraph_valid_ratio_threshold": )"
            << options.hgraph_valid_ratio_threshold << R"(
)";
    if (not options.mci_knng_path.empty()) {
        builder << R"(,
            "knng_path": ")"
                << options.mci_knng_path << R"(")";
    }
    builder << R"(
        }
    }
})";
    return builder.str();
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::init();
        auto options = parse_args(argc, argv);
        if (options.index_path.empty()) {
            options.index_path = default_index_path(options.data_path);
        }

        H5::H5File file(options.data_path, H5F_ACC_RDONLY);
        auto [total_rows, dim] = get_matrix_shape(file, "/train");
        auto rows = options.limit == 0 ? total_rows : std::min<uint64_t>(options.limit, total_rows);
        auto metric = infer_metric(options.data_path, options.metric);
        std::cout << "loading train vectors: rows=" << rows << ", dim=" << dim
                  << ", metric=" << metric << std::endl;
        auto train = read_float_matrix_prefix(file, "/train", rows, dim);

        std::vector<int64_t> ids(rows);
        for (uint64_t i = 0; i < rows; ++i) {
            ids[i] = static_cast<int64_t>(i);
        }

        auto base = vsag::Dataset::Make();
        base->NumElements(static_cast<int64_t>(rows))
            ->Dim(static_cast<int64_t>(dim))
            ->Ids(ids.data())
            ->Float32Vectors(train.data())
            ->Owner(false);

        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto params = make_hgraph_mci_params(options, dim, metric);
        auto index_result = engine.CreateIndex("hgraph", params);
        check_expected(index_result, "create hgraph index failed: ");
        auto index = index_result.value();

        std::cout << "building HGraph index with MCI clique datacell..." << std::endl;
        auto build_result = index->Build(base);
        check_expected(build_result, "build index failed: ");
        std::cout << "index contains vectors: " << index->GetNumElements() << std::endl;

        std::ofstream output(options.index_path, std::ios::binary);
        check(output.good(), "failed to open index output: " + options.index_path);
        auto serialize_result = index->Serialize(output);
        check_expected(serialize_result, "serialize index failed: ");
        std::cout << "saved index to: " << options.index_path << std::endl;
        engine.Shutdown();
    } catch (const std::exception& err) {
        std::cerr << "error: " << err.what() << std::endl;
        return 1;
    }
    return 0;
}
