#include <filesystem>
#include "../src/utils.h"
#include "constants_path.h"
#include "vsag/vsag.h"
#include <nlohmann/json.hpp>
#include <unordered_set>
#include "fmt/format.h"
#include "data_loader.h"
#include "omp.h"

std::string dataset = "gist-960-euclidean";
int target_npts = -1;
bool use_static = false;
int sq_num_bits = -1;
int gt_dim = 100;


int get_data(vsag::DatasetPtr& data, uint32_t expected_dim, std::string data_path_fmt)
{
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    int64_t *base_id;
    float *base_vec;
    uint32_t base_npts, base_dim;
    auto base_path = fmt::format(data_path_fmt, dataset);
    if (not std::filesystem::exists(base_path)) {
        logger->Debug(fmt::format("Error: file not exist({})", base_path));
        return -1;
    }
    vsag::load_aligned_fvecs(base_path, base_vec, base_npts, base_dim);
    if (expected_dim != base_dim) {
        logger->Debug(
            fmt::format("Error: expected_dim({}) != dim({})", expected_dim, base_dim));
        return -1;
    } else {
        logger->Debug(fmt::format("npts: {}, dim: {}", base_npts, base_dim));
    }

    base_id = new int64_t[base_npts];
    for (int64_t i = 0; i < base_npts; i++) {
        base_id[i] = i;
    }

    data->NumElements(base_npts)
        ->Dim(base_dim)
        ->Ids(base_id)
        ->Float32Vectors(base_vec)
        ->Owner(true);
    return base_npts;
}


int build() {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    logger->Debug(fmt::format("====Metadata===="));
    size_t pos1 = dataset.find_first_of("0123456789");
    size_t pos2 = dataset.find_first_not_of("0123456789", pos1) + 1;
    if (pos1 == std::string::npos || pos2 == std::string::npos) {
        logger->Error("dataset does not match the expected format.");
        return -1;
    }
    std::string dataset_name = dataset.substr(0, pos1 - 1);
    std::uint32_t expected_dim = std::stoul(dataset.substr(pos1, pos2 - pos1 - 1));
    std::string metric_name = dataset.substr(pos2);
    logger->Debug(fmt::format("dataset: {}, expected_dim: {}, metric: {}", dataset_name, expected_dim, metric_name));

    // data preparation
    logger->Debug(fmt::format("====Start load data===="));
    int64_t *base_id;
    float *base_vec;
    uint32_t base_npts, base_dim;
    auto base_path = fmt::format(BENCHMARK_BASE_PATH_FMT, dataset);
    if (not std::filesystem::exists(base_path)) {
        logger->Debug(fmt::format("Error: file not exist({})", base_path));
        return -1;
    }
    vsag::load_aligned_fvecs(base_path, base_vec, base_npts, base_dim);
    if (expected_dim != base_dim) {
        logger->Debug(fmt::format("Error: expected_dim({}) != dim({})", expected_dim, base_dim));
        return -1;
    } else {
        logger->Debug(fmt::format("npts: {}, dim: {}", base_npts, base_dim));
    }

    //base id
    base_id = new int64_t[base_npts];
    for (int64_t i = 0; i < base_npts; i++) {
        base_id[i] = i;
    }

    // metric
    std::string metric_type;
    if (metric_name == "euclidean") {
        metric_type = vsag::METRIC_L2;
    } else if (metric_name == "angular" or metric_name == "dot"){
        metric_type = vsag::METRIC_IP;
    } else {
        logger->Error(fmt::format("unsupported metric: {}", metric_name));
    }

    // data
    if (target_npts > 0) {
        base_npts = std::min((uint32_t)target_npts, base_npts);
        logger->Debug(fmt::format("target npts: {}", base_npts));
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(base_npts)
        ->Dim(base_dim)
        ->Ids(base_id)
        ->Float32Vectors(base_vec)
        ->Owner(true);

    // index build
    std::string index_path = fmt::format(INDEX_PATH_FMT,
                                         workspace, algo_name, dataset_name,
                                         base_npts, BL, BR,
                                         use_static ? "static" : "pure");
    auto build_parameters = fmt::format(BUILD_PARAM_FMT, metric_type, base_dim, BR, BL, sq_num_bits, use_static);
    auto index = vsag::Factory::CreateIndex(algo_name, build_parameters).value();
    if (std::filesystem::exists(index_path)) {
        logger->Debug(fmt::format("====Index Path Exists===="));
        logger->Debug(fmt::format("Index path: {}", index_path));
    } else {
        logger->Debug(fmt::format("====Start build===="));
        if (const auto build_result = index->Build(base); build_result.has_value()) {
            logger->Debug(fmt::format("After Build(), Index constains: {} elements", index->GetNumElements()));
        } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            logger->Error(fmt::format( "Failed to build index: internalError"));
            return -1;
        }

        // serialize
        logger->Debug(fmt::format("====Start serialize===="));
        logger->Debug(fmt::format("serialize with space cost: {:.3f} MB", index->GetMemoryUsage() / (1024.0 * 1024.0)));
        vsag::serialize(index, index_path);

        // deserialize test
        logger->Debug(fmt::format("====Start deserialize check===="));
        auto another_index = vsag::Factory::CreateIndex(algo_name, build_parameters).value();
        vsag::deserialize(another_index, index_path);
        if (another_index->GetNumElements() != index->GetNumElements() or
            another_index->GetMemoryUsage() != index->GetMemoryUsage()) {
            logger->Error(fmt::format( "Failed to check serialize result {}=={}, {}=={}",
                                      another_index->GetNumElements(), index->GetNumElements(),
                                      another_index->GetMemoryUsage(), index->GetMemoryUsage()));
            return -1;
        }

        logger->Debug(fmt::format("====Build finished in {} ====", index_path));
    }

    return 0;
}


int calculate_gt() {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kDEBUG);

    logger->Debug(fmt::format("====Metadata===="));
    size_t pos1 = dataset.find_first_of("0123456789");
    size_t pos2 = dataset.find_first_not_of("0123456789", pos1) + 1;
    if (pos1 == std::string::npos || pos2 == std::string::npos) {
        logger->Error("dataset does not match the expected format.");
        return -1;
    }
    std::string dataset_name = dataset.substr(0, pos1 - 1);
    std::uint32_t expected_dim = std::stoul(dataset.substr(pos1, pos2 - pos1 - 1));
    std::string metric_name = dataset.substr(pos2);
    logger->Debug(fmt::format("dataset: {}, expected_dim: {}, metric: {}", dataset_name, expected_dim, metric_name));

    // metric
    std::string metric_type;
    if (metric_name == "euclidean") {
        metric_type = vsag::METRIC_L2;
    } else if (metric_name == "angular" or metric_name == "dot"){
        metric_type = vsag::METRIC_IP;
    } else {
        logger->Error(fmt::format("unsupported metric: {}", metric_name));
    }

    // data preparation
    logger->Debug(fmt::format("====Start load data===="));
    auto base = vsag::Dataset::Make();
    auto query = vsag::Dataset::Make();
    int base_npts = get_data(base, expected_dim, BENCHMARK_BASE_PATH_FMT);
    int query_npts = get_data(query, expected_dim, BENCHMARK_QUERY_PATH_FMT);
    if (target_npts > 0) {
        base_npts = std::min(target_npts, base_npts);
        logger->Debug(fmt::format("target npts: {}", base_npts));
        base->NumElements(base_npts);
    }


    // index load
    logger->Debug(fmt::format("====Start create===="));
    auto build_parameters = fmt::format(BUILD_PARAM_FMT, metric_type, expected_dim, BR, BL, sq_num_bits, false);
    auto index = vsag::Factory::CreateIndex(algo_name, build_parameters).value();
    std::string index_path = fmt::format(INDEX_PATH_FMT,
                                         workspace, algo_name, dataset_name,
                                         base_npts, BL, BR,
                                         "pure");

    logger->Debug(fmt::format("====Start deserialize from {}====", index_path));
    vsag::deserialize(index, index_path);

    // calculate and store
    bool validate_gt = false;
    auto gt_path = fmt::format(BENCHMARK_GT_PATH_FMT, dataset, base_npts, gt_dim);
    if (std::filesystem::exists(gt_path)) {
        logger->Debug(fmt::format("====GT already exists===="));

        int32_t* gt_data;
        uint32_t gt_npts, gt_valid_dim;
        vsag::load_aligned_fvecs(gt_path, gt_data, gt_npts, gt_valid_dim);
        if ((gt_npts != query_npts) or (gt_valid_dim != gt_dim)) {
            logger->Debug(fmt::format("{} != asked {}, or {} != asked {}", gt_npts, query_npts, gt_valid_dim, gt_dim));
            logger->Debug(fmt::format("====GT validate: no===="));
            validate_gt = false;
        } else {
            logger->Debug(fmt::format("====GT validate: yes===="));
            validate_gt = true;
        }
        delete[] gt_data;
    }

    if (not validate_gt){
        logger->Debug(fmt::format("====GT calculation start===="));
        std::fstream out_file(gt_path, std::ios::out | std::ios::binary);

        std::vector<int32_t*> gt_results(query_npts);

        omp_set_num_threads(24);
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < query_npts; i++) {
            if (i % 100 == 0) {
                logger->Debug(fmt::format("calculated on {}", i));
            }
            auto single_query = vsag::Dataset::Make();
            single_query->NumElements(1)->Dim(expected_dim)->Owner(false);
            single_query->Float32Vectors(query->GetFloat32Vectors() + i * expected_dim);
            auto knn_result = *index->BruteForce(single_query, gt_dim);
            if (gt_dim != knn_result->GetDim()) {
                logger->Error(fmt::format("gt_dim({}) != knn_result_dim({})", gt_dim, knn_result->GetDim()));
            }

            int32_t* data32 = new int32_t[gt_dim];
            for (int j = 0; j < gt_dim; j++) {
                data32[j] = static_cast<int32_t>(knn_result->GetIds()[j]);
            }

            gt_results[i] = data32;
        }

        for (int i = 0; i < query_npts; i++) {
            out_file.write((char *)&gt_dim, sizeof(gt_dim));
            out_file.write((char *)gt_results[i], gt_dim * sizeof(int32_t));
            delete[] gt_results[i];
        }

        out_file.close();
        logger->Debug(fmt::format("====GT calculation finished===="));
    }

    return 0;

}


float calculate_recall(vsag::DatasetPtr ann_result, int32_t* gt_data, uint32_t k) {

    std::unordered_set<int32_t> searched;
    std::unordered_set<int32_t> gt;
    for (int i = 0; i < k; i++) {
        searched.insert((int32_t)ann_result->GetIds()[i]);
        gt.insert(gt_data[i]);
    }

    int count_successful_searched = 0;
    for (const auto& item : gt) {
        if (searched.find(item) != searched.end()) {
            count_successful_searched++;
        }
    }

    return (count_successful_searched * 1.0) / k;
}


int search(std::vector<uint32_t> efs, uint32_t k = 10) {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::kINFO);

    #ifdef NDEBUG
        logger->Info("Release mode");
    #else
        logger->Info("Debug mode");
    #endif

    logger->Debug(fmt::format("====Metadata===="));
    size_t pos1 = dataset.find_first_of("0123456789");
    size_t pos2 = dataset.find_first_not_of("0123456789", pos1) + 1;
    if (pos1 == std::string::npos || pos2 == std::string::npos) {
        logger->Error("dataset does not match the expected format.");
        return -1;
    }
    std::string dataset_name = dataset.substr(0, pos1 - 1);
    std::uint32_t expected_dim = std::stoul(dataset.substr(pos1, pos2 - pos1 - 1));
    std::string metric_name = dataset.substr(pos2);
    logger->Debug(fmt::format("dataset: {}, expected_dim: {}, metric: {}", dataset_name, expected_dim, metric_name));

    // metric
    std::string metric_type;
    if (metric_name == "euclidean") {
        metric_type = vsag::METRIC_L2;
    } else if (metric_name == "angular" or metric_name == "dot"){
        metric_type = vsag::METRIC_IP;
    } else {
        logger->Error(fmt::format("unsupported metric: {}", metric_name));
    }

    // index load
    logger->Debug(fmt::format("====Start create===="));
    auto base = vsag::Dataset::Make();
    int base_npts = get_data(base, expected_dim, BENCHMARK_BASE_PATH_FMT);
    if (target_npts > 0) {
        base_npts = std::min(target_npts, base_npts);
        logger->Debug(fmt::format("target npts: {}", base_npts));
        base->NumElements(base_npts);
    }
    auto build_parameters = fmt::format(BUILD_PARAM_FMT, metric_type, expected_dim, BR, BL, sq_num_bits, use_static);
    auto index = vsag::Factory::CreateIndex(algo_name, build_parameters).value();
    std::string index_path = fmt::format(INDEX_PATH_FMT,
                                         workspace, algo_name, dataset_name,
                                         base_npts, BL, BR,
                                         use_static ? "static" : "pure");

    logger->Debug(fmt::format("====Start deserialize from {}====", index_path));
    vsag::deserialize(index, index_path);

    // query and gt
    logger->Debug(fmt::format("====Load query and GT===="));
    auto query = vsag::Dataset::Make();
    int query_npts = get_data(query, expected_dim, BENCHMARK_QUERY_PATH_FMT);

    auto gt_path = fmt::format(BENCHMARK_GT_PATH_FMT, dataset, base_npts, gt_dim);
    int32_t* gt_data;
    uint32_t gt_npts, gt_valid_dim;
    vsag::load_aligned_fvecs(gt_path, gt_data, gt_npts, gt_valid_dim);
    if ((gt_npts != query_npts) or (gt_valid_dim != gt_dim)) {
        logger->Debug(fmt::format("{} != asked {}, or {} != asked {}", gt_npts, query_npts, gt_valid_dim, gt_dim));
        logger->Debug(fmt::format("====GT validate: no===="));
        return -1;
    }

    //search
    constexpr auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }}
        )";

    auto single_query = vsag::Dataset::Make();
    single_query->NumElements(1)->Dim(expected_dim)->Owner(false);
    vsag::DatasetPtr ann_result;
    single_query->Float32Vectors(query->GetFloat32Vectors() + 100 * expected_dim);
    auto result = index->Test(single_query);
    logger->Info(fmt::format("sq: {}, time: {}", sq_num_bits, result->first));

    for (auto ef_search : efs) {
        logger->Debug(fmt::format("====Search with ef {}====", ef_search));
        auto search_parameters = fmt::format(search_parameters_json, ef_search);


        double total_time_cost = 0;
        double recall = 0;
        double avg_dist_cmp = 0, avg_hop = 0;
        for (int i = 0; i < query_npts; i++) {
            auto single_query = vsag::Dataset::Make();
            single_query->NumElements(1)->Dim(expected_dim)->Owner(false);
            double time_cost = 0;
            vsag::DatasetPtr ann_result;
            single_query->Float32Vectors(query->GetFloat32Vectors() + i * expected_dim);
            {
                vsag::Timer t(time_cost);
                ann_result = *index->KnnSearch(single_query, k, search_parameters);
            }
            total_time_cost += time_cost;
            recall += calculate_recall(ann_result, gt_data + i * gt_dim, k);
            assert(ann_result->GetDim() == k + 2);
            assert(ann_result->GetDistances()[k] - 10000000 < 1e4);
            assert(ann_result->GetDistances()[k + 1] - 20000000 < 1e4);
            avg_dist_cmp += ann_result->GetIds()[k];
            avg_hop += ann_result->GetIds()[k + 1];
        }
        recall /= query_npts;
        logger->Info(fmt::format("recall: {:.4f}, QPS: {:.1f}, time cost: {:.2f} ms, avg_dist_cmp: {:.2f}, avg_hop: {:.2f}",
                                 recall,
                                 query_npts / (total_time_cost / 1000),
                                 total_time_cost / query_npts,
                                 avg_dist_cmp / query_npts,
                                 avg_hop / query_npts));
    }


    delete[] gt_data;

    return 0;
}

int main() {
    // metadata
    dataset = "gist-960-euclidean";
    target_npts = -1;
    use_static = false;
    sq_num_bits = -1;
    gt_dim = 100;

    // prepare index and ground_truth
    // build();
    calculate_gt();

    // search
    std::vector<uint32_t> efs = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    efs.resize(100, 80);
    for (int round = 0; round < 2; round++) {
         if (round == 0) {
             sq_num_bits = 4;
         } else if (round == 1) {
             sq_num_bits = 8;
         } else {
             sq_num_bits = -1;
         }
         use_static = (round == 2);
        search(efs);
    }
}
