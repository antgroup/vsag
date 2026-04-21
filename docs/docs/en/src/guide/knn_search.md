# k-Nearest Neighbor Search

This page assumes VSAG is already installed. Examples are available in C++, Python, and TypeScript
under the [`examples/`](https://github.com/antgroup/vsag/tree/main/examples) directory. This page
uses the C++ `BruteForce` index for illustration; the full source is at
[`examples/cpp/105_index_brute_force.cpp`](https://github.com/antgroup/vsag/blob/main/examples/cpp/105_index_brute_force.cpp).

> In most cases, your program should call `vsag::init()` once at startup to perform one-time
> initialization (global logger, allocator, etc.). The snippets below omit boilerplate to focus on
> the essential steps.

## Prepare Vectors

VSAG operates on collections of fixed-dimensional vectors (typically a few hundred to a few
thousand dimensions). Vectors are laid out row-major, equivalent to `vector[num_vectors][dim]` in
C++. The API only requires a pointer (`const float*`) to the first element, so you can use a raw
array, `std::vector<float>`, or a custom buffer.

> VSAG currently supports 32-bit float vectors for the public API. Other dtypes are available
> internally via the `dtype` option.

A k-NN search needs two datasets:

- **base**: all vectors in the database; size = `num_vectors * dim`.
- **query**: the query vector(s) for which to find nearest neighbors; size = `num_queries * dim`.
  Currently the public `KnnSearch` API processes one query at a time.

```cpp
int64_t num_vectors = 10000;
int64_t dim = 128;
int64_t* ids = new int64_t[num_vectors];
float* datas = new float[num_vectors * dim];
std::mt19937 rng(47);
std::uniform_real_distribution<float> distrib;
for (int64_t i = 0; i < num_vectors; ++i) ids[i] = i;
for (int64_t i = 0; i < dim * num_vectors; ++i) datas[i] = distrib(rng);

float* query_vector = new float[dim];
for (int64_t i = 0; i < dim; ++i) query_vector[i] = distrib(rng);
```

## Create an Index and Insert Vectors

The `Index` interface is the central abstraction. Multiple implementations exist; `brute_force` is
the simplest (exhaustive comparison, used as a baseline).

All indexes must be created explicitly, specifying dimension and metric:

```cpp
std::string build_params = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128
}
)";
auto index = vsag::Factory::CreateIndex("brute_force", build_params).value();
```

`Build` performs any required training; `Add` appends vectors. `BruteForce` supports both:

```cpp
auto base = vsag::Dataset::Make();
base->NumElements(num_vectors)
    ->Dim(dim)
    ->Ids(ids)
    ->Float32Vectors(datas)
    ->Owner(false);
index->Add(base);
```

## Search

`KnnSearch` takes the query, `k`, and a JSON search-params string. `BruteForce` has no tunable
search params, so an empty object is passed.

```cpp
auto query = vsag::Dataset::Make();
query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);

int64_t topk = 10;
auto result = index->KnnSearch(query, topk, R"({})").value();

for (int64_t i = 0; i < result->GetDim(); ++i) {
    std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
}
```

The result contains up to `k` neighbors sorted by ascending distance to the query.
