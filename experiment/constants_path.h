#pragma once

// dataset name reference

//inline std::set<std::string> benchmark_dataset = {
//    "deep-image-96-angular",
//    "fashion-mnist-784-euclidean",
//    "gist-960-euclidean",
//    "glove-25-angular",
//    "glove-50-angular",
//    "glove-100-angular",
//    "glove-200-angular",
//    "mnist-784-euclidean",
//    "nytimes-256-angular",
//    "sift-128-euclidean",
//    "lastfm-64-dot"
//};


// benchmark experiment
inline std::string algo_name = "hnsw";
inline std::string workspace = "/tbase-project";

constexpr const char* BENCHMARK_GT_PATH_FMT = "/tbase-project/vsag/data/{}/gt_N{}_K{}.fvecs";
constexpr const char* BENCHMARK_QUERY_PATH_FMT = "/tbase-project/vsag/data/{}/query.fvecs";
constexpr const char* BENCHMARK_BASE_PATH_FMT = "/tbase-project/vsag/data/{}/learn.fvecs";
constexpr const char* INDEX_PATH_FMT = "{}/vsag/data/{}_{}_N{}_BL{}_BR{}_{}.index";

constexpr const char* BUILD_PARAM_FMT = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": {},
            "ef_construction": {},
            "use_conjugate_graph": false,
            "sq_num_bits": {},
            "use_static": {},
            "redundant_rate": {},
            "extra_file": "123"
        }}
    }}
    )";



// face data experiment
constexpr const char* DATA_PATH_PREFIX_FMT = "{}/dataset/{}/{}";

constexpr const char* BASE_VEC_PATH_FMT = "{}_learn.fbin";
constexpr const char* BASE_ID_PATH_FMT = "{}_learn_id.fbin";

constexpr const char* TRAIN_VEC_PATH_FMT = "{}_train.fbin";
constexpr const char* TRAIN_ID_PATH_FMT = "{}_train_id.fbin";
constexpr const char* TRAIN_GT_PATH_FMT = "{}_train_learn_gt100";

constexpr const char* FAILED_TRAIN_ID_PATH_FMT = "{}_failed_train_id.fbin";
constexpr const char* FAILED_BASE_TAG_PATH_FMT = "{}_failed_base_tag.fbin";
constexpr const char* FAILED_BASE_ID_PATH_FMT = "{}_failed_base_id.fbin";

inline std::string face_dataset_name = "face_b1t10q10_50000000_float";
inline int BL = 500, BR = 36;
inline int npts_index = 835770;