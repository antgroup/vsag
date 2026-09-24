// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "eval_dataset.h"
#include "impl/allocator/safe_allocator.h"
#include "nlohmann/json.hpp"
#include "quantization/rabitq_quantization/rabitq_quantizer.h"
#include "quantization/saq_quantization/saq_quantizer.h"

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

struct DistancePair {
    uint64_t query_id;
    uint64_t base_id;
};

double
ElapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename Quantizer>
Json
RunQuantizer(const std::string& name,
             Quantizer& quantizer,
             const float* data,
             const float* queries,
             const std::vector<DistancePair>& distance_pairs,
             uint64_t dim,
             uint64_t train_count,
             uint64_t encode_count) {
    std::cerr << "benchmark_method_started=" << name << '\n';
    const auto train_begin = Clock::now();
    if (not quantizer.Train(data, train_count)) {
        throw std::runtime_error(name + " training failed");
    }
    const auto train_end = Clock::now();

    const uint64_t code_size = quantizer.GetCodeSize();
    std::vector<uint8_t> codes(code_size * encode_count);
    const auto encode_begin = Clock::now();
    if (not quantizer.EncodeBatch(data, codes.data(), encode_count)) {
        throw std::runtime_error(name + " encoding failed");
    }
    const auto encode_end = Clock::now();

    const uint64_t distance_count = distance_pairs.size();
    if (distance_count == 0) {
        throw std::runtime_error("distance evaluation requires at least one valid pair");
    }
    double distance_squared_error = 0.0;
    double exact_distance_squared_sum = 0.0;
    double distance_relative_error_sum = 0.0;
    auto computer = quantizer.FactoryComputer();
    uint64_t active_query_id = std::numeric_limits<uint64_t>::max();
    for (uint64_t i = 0; i < distance_count; ++i) {
        const auto& pair = distance_pairs[i];
        const uint64_t base_id = pair.base_id;
        const float* query = queries + pair.query_id * dim;
        const float* base = data + base_id * dim;
        if (pair.query_id != active_query_id) {
            computer->SetQuery(query);
            active_query_id = pair.query_id;
        }
        const double approximate =
            quantizer.ComputeDist(*computer, codes.data() + base_id * code_size);
        double exact = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            const double delta = static_cast<double>(query[d]) - base[d];
            exact += delta * delta;
        }
        const double error = approximate - exact;
        distance_squared_error += error * error;
        exact_distance_squared_sum += exact * exact;
        if (exact > std::numeric_limits<double>::epsilon()) {
            distance_relative_error_sum += std::abs(error) / exact;
        }
    }

    double timed_distance_checksum = 0.0;
    uint64_t timed_query_setup_count = 0;
    active_query_id = std::numeric_limits<uint64_t>::max();
    const auto query_distance_begin = Clock::now();
    for (uint64_t i = 0; i < distance_count; ++i) {
        const auto& pair = distance_pairs[i];
        const uint64_t base_id = pair.base_id;
        if (pair.query_id != active_query_id) {
            computer->SetQuery(queries + pair.query_id * dim);
            active_query_id = pair.query_id;
            ++timed_query_setup_count;
        }
        timed_distance_checksum +=
            quantizer.ComputeDist(*computer, codes.data() + base_id * code_size);
    }
    const auto query_distance_end = Clock::now();
    const double query_distance_total_ms =
        ElapsedMilliseconds(query_distance_begin, query_distance_end);

    const uint64_t scan_count = std::min<uint64_t>(encode_count, 100'000);
    computer->SetQuery(queries);
    double distance_checksum = 0.0;
    const auto scan_begin = Clock::now();
    for (uint64_t i = 0; i < scan_count; ++i) {
        distance_checksum += quantizer.ComputeDist(*computer, codes.data() + i * code_size);
    }
    const auto scan_end = Clock::now();
    const double scan_ms = ElapsedMilliseconds(scan_begin, scan_end);

    const uint64_t batch4_scan_count = scan_count - scan_count % 4;
    double batch4_distance_checksum = 0.0;
    const auto batch4_scan_begin = Clock::now();
    for (uint64_t i = 0; i < batch4_scan_count; i += 4) {
        float distances[4];
        quantizer.ComputeDistsBatch4(*computer,
                                     codes.data() + i * code_size,
                                     codes.data() + (i + 1) * code_size,
                                     codes.data() + (i + 2) * code_size,
                                     codes.data() + (i + 3) * code_size,
                                     distances[0],
                                     distances[1],
                                     distances[2],
                                     distances[3]);
        batch4_distance_checksum += distances[0] + distances[1] + distances[2] + distances[3];
    }
    const auto batch4_scan_end = Clock::now();
    const double batch4_scan_ms = ElapsedMilliseconds(batch4_scan_begin, batch4_scan_end);

    const uint64_t code_pair_count = std::min<uint64_t>(encode_count, 100'000);
    const uint64_t code_pair_offset = std::max<uint64_t>(1, encode_count / 2);
    bool code_pair_supported = true;
    std::string code_pair_error;
    double code_pair_checksum = 0.0;
    const auto code_pair_begin = Clock::now();
    try {
        for (uint64_t i = 0; i < code_pair_count; ++i) {
            const uint64_t other = (i + code_pair_offset) % encode_count;
            code_pair_checksum +=
                quantizer.Compute(codes.data() + i * code_size, codes.data() + other * code_size);
        }
    } catch (const std::exception& exception) {
        code_pair_supported = false;
        code_pair_error = exception.what();
    }
    const auto code_pair_end = Clock::now();
    const double code_pair_ms = ElapsedMilliseconds(code_pair_begin, code_pair_end);

    constexpr uint64_t maximum_reconstruction_count = 10'000;
    const uint64_t reconstruction_count =
        std::min<uint64_t>(encode_count, maximum_reconstruction_count);
    std::vector<float> decoded(dim);
    const bool reconstruction_supported = quantizer.DecodeOne(codes.data(), decoded.data());
    double reconstruction_ms = 0.0;
    double reconstruction_squared_error = 0.0;
    double reconstruction_squared_norm = 0.0;
    if (reconstruction_supported) {
        decoded.resize(dim * reconstruction_count);
        const auto decode_begin = Clock::now();
        if (not quantizer.DecodeBatch(codes.data(), decoded.data(), reconstruction_count)) {
            throw std::runtime_error(name + " batch decoding failed");
        }
        const auto decode_end = Clock::now();
        reconstruction_ms = ElapsedMilliseconds(decode_begin, decode_end);
        for (uint64_t i = 0; i < dim * reconstruction_count; ++i) {
            const double value = data[i];
            const double delta = value - decoded[i];
            reconstruction_squared_error += delta * delta;
            reconstruction_squared_norm += value * value;
        }
    }

    Json result;
    result["name"] = name;
    result["train_ms"] = ElapsedMilliseconds(train_begin, train_end);
    result["encode_ms"] = ElapsedMilliseconds(encode_begin, encode_end);
    const double encode_ms = result["encode_ms"].get<double>();
    result["encode_vectors_per_second"] =
        encode_ms == 0.0 ? 0.0 : static_cast<double>(encode_count) * 1000.0 / encode_ms;
    result["code_size_bytes_per_vector"] = code_size;
    result["code_bits_per_vector"] = code_size * 8;
    result["distance_pair_count"] = distance_count;
    result["distance_pair_protocol"] = "query_to_ground_truth_neighbor";
    result["distance_mse"] = distance_squared_error / static_cast<double>(distance_count);
    result["distance_relative_mse"] = exact_distance_squared_sum == 0.0
                                          ? 0.0
                                          : distance_squared_error / exact_distance_squared_sum;
    result["distance_mean_relative_error"] =
        distance_relative_error_sum / static_cast<double>(distance_count);
    result["query_setup_and_single_distance_ms"] =
        query_distance_total_ms / static_cast<double>(distance_count);
    result["query_setup_count"] = timed_query_setup_count;
    result["query_setup_and_distance_total_ms"] = query_distance_total_ms;
    result["query_setup_and_distance_checksum"] = timed_distance_checksum;
    result["distance_scan_count"] = scan_count;
    result["distance_scan_ms"] = scan_ms;
    result["distance_scans_per_second"] =
        scan_ms == 0.0 ? 0.0 : static_cast<double>(scan_count) * 1000.0 / scan_ms;
    result["distance_batch4_scan_count"] = batch4_scan_count;
    result["distance_batch4_scan_ms"] = batch4_scan_ms;
    result["distance_batch4_scans_per_second"] =
        batch4_scan_ms == 0.0 ? 0.0
                              : static_cast<double>(batch4_scan_count) * 1000.0 / batch4_scan_ms;
    result["distance_batch4_checksum"] = batch4_distance_checksum;
    result["code_pair_supported"] = code_pair_supported;
    if (code_pair_supported) {
        result["code_pair_count"] = code_pair_count;
        result["code_pair_ms"] = code_pair_ms;
        result["code_pairs_per_second"] =
            code_pair_ms == 0.0 ? 0.0
                                : static_cast<double>(code_pair_count) * 1000.0 / code_pair_ms;
        result["code_pair_checksum"] = code_pair_checksum;
    } else {
        result["code_pair_error"] = code_pair_error;
    }
    result["distance_checksum"] = distance_checksum;
    result["reconstruction_supported"] = reconstruction_supported;
    if (reconstruction_supported) {
        result["reconstruction_count"] = reconstruction_count;
        result["decode_ms"] = reconstruction_ms;
        result["decode_vectors_per_second"] =
            reconstruction_ms == 0.0
                ? 0.0
                : static_cast<double>(reconstruction_count) * 1000.0 / reconstruction_ms;
        result["reconstruction_mse"] =
            reconstruction_squared_error / static_cast<double>(dim * reconstruction_count);
        result["reconstruction_relative_mse"] =
            reconstruction_squared_norm == 0.0
                ? 0.0
                : reconstruction_squared_error / reconstruction_squared_norm;
    }
    std::cerr << "benchmark_method_finished=" << name << '\n';
    return result;
}

std::vector<DistancePair>
BuildDistancePairs(const vsag::eval::EvalDataset& dataset,
                   const float* data,
                   uint64_t dim,
                   uint64_t encode_count) {
    constexpr uint64_t max_pair_count = 100'000;
    constexpr uint64_t ground_truth_ranks_per_query = 10;
    std::vector<DistancePair> pairs;
    pairs.reserve(max_pair_count);
    const uint64_t query_count = dataset.GetNumberOfQuery();
    const uint64_t ground_truth_k = dataset.GetGroundTruthK();
    const uint64_t ranks = std::min(ground_truth_k, ground_truth_ranks_per_query);
    for (uint64_t query_id = 0; query_id < query_count and pairs.size() < max_pair_count;
         ++query_id) {
        const int64_t* neighbors = dataset.GetNeighbors(query_id);
        if (neighbors == nullptr) {
            break;
        }
        for (uint64_t rank = 0; rank < ranks and pairs.size() < max_pair_count; ++rank) {
            const auto* base = static_cast<const float*>(dataset.GetOneTrainById(neighbors[rank]));
            if (base == nullptr or base < data) {
                continue;
            }
            const uint64_t element_offset = static_cast<uint64_t>(base - data);
            if (element_offset % dim != 0) {
                continue;
            }
            const uint64_t base_id = element_offset / dim;
            if (base_id < encode_count) {
                pairs.push_back({query_id, base_id});
            }
        }
    }

    if (pairs.empty()) {
        const uint64_t fallback_count = std::min<uint64_t>(encode_count, 10'000);
        for (uint64_t i = 0; i < fallback_count; ++i) {
            pairs.push_back({i % query_count, i});
        }
    }
    return pairs;
}

uint64_t
ParseCount(const char* value, const std::string& name) {
    const auto count = std::stoull(value);
    if (count == 0) {
        throw std::invalid_argument(name + " must be positive");
    }
    return count;
}

Json
RunSAQ(const std::string& name,
       const float* data,
       const float* queries,
       const std::vector<DistancePair>& distance_pairs,
       uint64_t dim,
       uint64_t train_count,
       uint64_t encode_count,
       float avg_bits,
       uint64_t segment_count,
       uint64_t adjustment_rounds,
       bool use_pca,
       bool random_rotation,
       vsag::Allocator* allocator) {
    vsag::SAQQuantizer<vsag::MetricType::METRIC_TYPE_L2SQR> quantizer(
        dim, avg_bits, segment_count, adjustment_rounds, use_pca, random_rotation, allocator);
    Json result = RunQuantizer(
        name, quantizer, data, queries, distance_pairs, dim, train_count, encode_count);
    result["parameters"] = {
        {"avg_bits", avg_bits},
        {"segment_count", segment_count},
        {"adjustment_rounds", adjustment_rounds},
        {"use_pca", use_pca},
        {"random_rotation", random_rotation},
    };
    if (random_rotation) {
        result["parameters"]["random_rotation_seed"] =
            vsag::SAQQuantizerParameter::DEFAULT_RANDOM_ROTATION_SEED;
    }
    result["trained_segments"] = Json::array();
    for (const auto& segment : quantizer.GetSegments()) {
        result["trained_segments"].push_back(
            {{"begin", segment.begin}, {"length", segment.length}, {"bits", segment.bits}});
    }
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 3 or argc > 8) {
        std::cerr << "Usage: " << argv[0]
                  << " DATASET.hdf5 OUTPUT.json [AVG_BITS=4] [TRAIN_COUNT=10000] "
                     "[ENCODE_COUNT=100000] [--ablations] [--exact-rabitq]\n";
        return 2;
    }

    try {
        const std::string dataset_path = argv[1];
        const std::string output_path = argv[2];
        const float avg_bits = argc >= 4 ? std::stof(argv[3]) : 4.0F;
        if (not std::isfinite(avg_bits) or avg_bits < 1.0F or avg_bits > 8.0F or
            std::abs(avg_bits - std::round(avg_bits)) > 1e-6F) {
            throw std::invalid_argument(
                "AVG_BITS must be an integer in [1, 8] for an equal-length RaBitQ comparison");
        }
        const uint64_t requested_train_count =
            argc >= 5 ? ParseCount(argv[4], "TRAIN_COUNT") : 10'000;
        const uint64_t requested_encode_count =
            argc >= 6 ? ParseCount(argv[5], "ENCODE_COUNT") : 100'000;
        bool run_ablations = false;
        bool run_exact_rabitq = false;
        for (int argument = 6; argument < argc; ++argument) {
            const std::string flag = argv[argument];
            if (flag == "--ablations") {
                run_ablations = true;
            } else if (flag == "--exact-rabitq") {
                run_exact_rabitq = true;
            } else {
                throw std::invalid_argument("unsupported optional flag: " + flag);
            }
        }

        const auto dataset = vsag::eval::EvalDataset::Load(dataset_path);
        if (dataset->GetVectorType() != vsag::eval::DENSE_VECTORS or
            dataset->GetTrainDataType() != vsag::DATATYPE_FLOAT32 or
            dataset->GetTestDataType() != vsag::DATATYPE_FLOAT32) {
            throw std::invalid_argument("the quantization benchmark requires dense float32 data");
        }
        if (dataset->GetMetric() != "euclidean") {
            throw std::invalid_argument("the quantization benchmark currently requires L2 data");
        }

        const uint64_t base_count = dataset->GetNumberOfBase();
        const uint64_t train_count = std::min(requested_train_count, base_count);
        const uint64_t encode_count = std::min(requested_encode_count, base_count);
        if (train_count < 2) {
            throw std::invalid_argument("the dataset must contain at least two base vectors");
        }
        if (dataset->GetNumberOfQuery() <= 0) {
            throw std::invalid_argument("the dataset must contain at least one query vector");
        }
        const uint64_t dim = dataset->GetDim();
        const auto* data = static_cast<const float*>(dataset->GetTrain());
        const auto* queries = static_cast<const float*>(dataset->GetTest());
        const auto distance_pairs = BuildDistancePairs(*dataset, data, dim, encode_count);
        auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

        vsag::RaBitQuantizer<vsag::MetricType::METRIC_TYPE_L2SQR> rabitq(
            dim,
            dim,
            32,
            static_cast<uint64_t>(std::lround(avg_bits)),
            false,
            false,
            allocator.get());

        Json output;
        output["benchmark_schema_version"] = 4;
        output["dataset"] = dataset_path;
        output["metric"] = dataset->GetMetric();
        output["dimension"] = dim;
        output["base_count"] = base_count;
        output["train_count"] = train_count;
        output["encode_count"] = encode_count;
        output["query_count"] = dataset->GetNumberOfQuery();
        output["ground_truth_k"] = dataset->GetGroundTruthK();
        output["distance_pair_protocol"] = "up_to_10_ground_truth_neighbors_per_query";
        output["distance_pair_count"] = distance_pairs.size();
        output["reconstruction_protocol"] = "first_up_to_10000_encoded_base_vectors";
        output["requested_average_bits_per_dimension"] = avg_bits;
        output["ablations_enabled"] = run_ablations;
        output["exact_rabitq_enabled"] = run_exact_rabitq;
        output["methods"] = Json::array();
        output["methods"].push_back(RunSAQ("saq",
                                           data,
                                           queries,
                                           distance_pairs,
                                           dim,
                                           train_count,
                                           encode_count,
                                           avg_bits,
                                           0,
                                           6,
                                           true,
                                           true,
                                           allocator.get()));
        output["methods"].push_back(RunQuantizer(
            "rabitq", rabitq, data, queries, distance_pairs, dim, train_count, encode_count));
        output["methods"].back()["parameters"] = {
            {"bits_per_dimension", static_cast<uint64_t>(std::lround(avg_bits))},
            {"fast_encode_rabitq", true},
            {"fast_encode_rabitq_rounds", 6},
            {"use_fht", false},
            {"use_mrq", false},
        };
        if (run_exact_rabitq) {
            vsag::RaBitQuantizer<vsag::MetricType::METRIC_TYPE_L2SQR> exact_rabitq(
                dim,
                dim,
                32,
                static_cast<uint64_t>(std::lround(avg_bits)),
                false,
                false,
                allocator.get(),
                vsag::RaBitQuantizerParameter::DEFAULT_RABITQ_VERSION,
                vsag::RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
                vsag::RaBitQuantizerParameter::DEFAULT_RABITQ_BITS_PER_DIM_FILTER,
                false,
                vsag::RaBitQuantizerParameter::DEFAULT_FAST_ENCODE_RABITQ_ROUNDS);
            output["methods"].push_back(RunQuantizer("rabitq_exact",
                                                     exact_rabitq,
                                                     data,
                                                     queries,
                                                     distance_pairs,
                                                     dim,
                                                     train_count,
                                                     encode_count));
            output["methods"].back()["parameters"] = {
                {"bits_per_dimension", static_cast<uint64_t>(std::lround(avg_bits))},
                {"fast_encode_rabitq", false},
                {"use_fht", false},
                {"use_mrq", false},
            };
        }
        output["equal_code_length"] = output["methods"][0]["code_size_bytes_per_vector"] ==
                                      output["methods"][1]["code_size_bytes_per_vector"];
        if (not output["equal_code_length"].get<bool>()) {
            throw std::runtime_error("SAQ and RaBitQ produced different stored code lengths");
        }
        if (run_ablations) {
            output["methods"].push_back(RunSAQ("saq_no_pca",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               0,
                                               6,
                                               false,
                                               true,
                                               allocator.get()));
            output["methods"].push_back(RunSAQ("saq_no_pca_no_rotation",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               0,
                                               6,
                                               false,
                                               false,
                                               allocator.get()));
            output["methods"].push_back(RunSAQ("saq_no_random_rotation",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               0,
                                               6,
                                               true,
                                               false,
                                               allocator.get()));
            output["methods"].push_back(RunSAQ("saq_adjustment_rounds_0_no_rotation",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               0,
                                               0,
                                               true,
                                               false,
                                               allocator.get()));
            output["methods"].push_back(RunSAQ("saq_fixed_1_segment_no_rotation",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               1,
                                               6,
                                               true,
                                               false,
                                               allocator.get()));
            constexpr uint64_t fixed_segment_count = 2;
            output["methods"].push_back(RunSAQ("saq_fixed_2_segments_no_rotation",
                                               data,
                                               queries,
                                               distance_pairs,
                                               dim,
                                               train_count,
                                               encode_count,
                                               avg_bits,
                                               fixed_segment_count,
                                               6,
                                               true,
                                               false,
                                               allocator.get()));
        }
        output["all_equal_code_length"] = std::all_of(
            output["methods"].begin(), output["methods"].end(), [&output](const Json& method) {
                return method["code_size_bytes_per_vector"] ==
                       output["methods"][0]["code_size_bytes_per_vector"];
            });
        if (not output["all_equal_code_length"].get<bool>()) {
            throw std::runtime_error("an ablation produced a different stored code length");
        }

        std::ofstream stream(output_path);
        if (not stream) {
            throw std::runtime_error("cannot open output file: " + output_path);
        }
        stream << output.dump(2) << '\n';
        std::cout << output.dump(2) << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "saq_quantization_benchmark: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
