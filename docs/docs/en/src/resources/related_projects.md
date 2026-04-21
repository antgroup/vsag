# Related Projects

This page lists upstream and downstream projects related to or integrating with VSAG, making it
easier to assemble complete stacks.

## Projects Using VSAG

- **[OceanBase](https://github.com/oceanbase/oceanbase)** — Ant Group's open-source distributed
  relational database; its vector search is powered by VSAG.
- **Other vector databases / integrations** — if you maintain an integration, feel free to open a
  PR to list it here.

## Dependencies and Inspirations

- **[hnswlib](https://github.com/nmslib/hnswlib)** — the canonical HNSW implementation; VSAG's
  HNSW interface and algorithms were influenced by it.
- **[DiskANN](https://github.com/microsoft/DiskANN)** — Microsoft Research's large-scale on-disk
  vector search work; VSAG's `diskann` index is based on this approach.
- **[Faiss](https://github.com/facebookresearch/faiss)** — Meta's vector search library; VSAG
  borrows ideas in IVF and quantization.
- **[SPANN / SPTAG](https://github.com/microsoft/SPTAG)** — Microsoft's large-scale retrieval
  system; shaped our hybrid-index approach.

## Ecosystem Tooling

- **[ann-benchmarks](https://github.com/erikbern/ann-benchmarks)** — the de-facto ANN benchmark
  harness; VSAG's [performance evaluation tool](eval.md) is compatible with its dataset format.
- **[pybind11](https://github.com/pybind/pybind11)** — powers the `pyvsag` Python binding.
- **[napi-rs](https://napi.rs/)** — powers the Node.js binding under `typescript/`.

## Bindings / Language Support

- **C++** (native)
- **Python** — `pyvsag`, source under `python_bindings/` and `python/`.
- **Node.js / TypeScript** — source under `typescript/`, npm package name `vsag`.

Pull requests to extend this list are welcome.
