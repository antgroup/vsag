# VSAG

VSAG is a vector indexing library used for similarity search. The indexing algorithm allows users to search through various sizes of vector sets, especially those that cannot fit in memory. The library also provides methods for generating parameters based on vector dimensions and data scale, allowing developers to use it without understanding the algorithm’s principles. VSAG is written in C++ and provides a Python wrapper package called pyvsag.

## Performance
The VSAG algorithm achieves a significant boost of efficiency and outperforms the previous **state-of-the-art (SOTA)** by a clear margin. Specifically, VSAG's QPS exceeds that of the previous SOTA algorithm, Glass, by over 100%, and the baseline algorithm, HNSWLIB, by over 300% according to the ann-benchmark result on the GIST dataset at 90% recall.
The test in [ann-benchmarks](https://ann-benchmarks.com/) is running on an r6i.16xlarge machine on AWS with `--parallelism 31`, single-CPU, and hyperthreading disabled.
The result is as follows:

### gist-960-euclidean
![](./docs/gist-960-euclidean_10_euclidean.png)

## Getting Started
### Integrate with CMake
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.11)

project (myproject)

set (CMAKE_CXX_STANDARD 11)

# download and compile vsag
include (FetchContent)
FetchContent_Declare (
  vsag
  GIT_REPOSITORY https://github.com/alipay/vsag
  GIT_TAG main
)
FetchContent_MakeAvailable (vsag)
include_directories (vsag-cmake-example PRIVATE ${vsag_SOURCE_DIR}/include)

# compile executable and link to vsag
add_executable (vsag-cmake-example src/main.cpp)
target_link_libraries (vsag-cmake-example PRIVATE vsag)

# add dependency
add_dependencies (vsag-cmake-example vsag)
```
### Try the Example
```cpp
#include <vsag/vsag.h>

#include <iostream>

int
main(int argc, char** argv) {
    vsag::init();

    int64_t num_vectors = 10000;
    int64_t dim = 128;

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    // create index
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors)->Owner(false);
    index->Build(base);

    // prepare a query vector
    auto query_vector = new float[dim];  // memory will be released by query the dataset
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // search on the index
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);
    auto result = index->KnnSearch(query, topk, hnsw_search_parameters).value();

    // print the results
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }

    // free memory
    delete[] ids;
    delete[] vectors;

    return 0;
}
```

## Building from Source
Please read the [DEVELOPMENT](./DEVELOPMENT.md) guide for instructions on how to build.

## Who's Using VSAG
- [OceanBase](https://github.com/oceanbase/oceanbase)
- [TuGraph](https://github.com/TuGraph-family/tugraph-db)
- [GreptimeDB](https://github.com/GreptimeTeam/greptimedb)

If your system uses VSAG, then feel free to make a pull request to add it to the list.

## How to Contribute

Although VSAG is initially developed by the Vector Database Team at Ant Group, it's the work of
the [community](https://github.com/alipay/vsag/graphs/contributors), and contributions are always welcome!
See [CONTRIBUTING](./CONTRIBUTING.md) for ways to get started.

## Community
Thrive together in VSAG community with users and developers from all around the world.
- Discuss at [discord](https://discord.com/invite/JyDmUzuhrp).
- Follow us on [Weixin Official Accounts](./docs/weixin-qr.jpg)（微信公众平台）to get the latest news.

## Roadmap
- v0.12 (ETA: Oct. 2024)
  - introduce datacell as the new index framework
  - support pluggable scalar quantization(known as SQ) in datacell
  - implement a new Hierarchical Graph(named HGraph) index based on datacell
  - support INT8 datatype on HNSW Index

- v0.13 (ETA: Nov. 2024)
  - support inverted index(be like IVFFlat) based on datacell
  - introduce pluggable product quantization(known as PQ) in datacell
  - support extrainfo storage within vector

- v0.14 (ETA: Dec. 2024)
  - implement a new MultiIndex that supports efficient pre-filtering on enumerable tags
  - support automated parameter
  - support sparse vector searching

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=alipay/vsag&type=Date)](https://star-history.com/#alipay/vsag&Date)

## License
[Apache License 2.0](./LICENSE)
