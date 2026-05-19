# Summary

[Introduction](README.md)

# User Guide

- [Installation](guide/installation.md)
- [Creating an Index](guide/create_index.md)
- [k-Nearest Neighbor Search](guide/knn_search.md)
- [pyvsag](guide/pyvsag.md)

# Indexes

- [Overview](indexes/README.md)
- [Index Parameters](resources/index_parameters.md)
- [HGraph](indexes/hgraph.md)
- [IVF](indexes/ivf.md)
- [SINDI](indexes/sindi.md)
- [Pyramid](indexes/pyramid.md)
- [BruteForce](indexes/brute_force.md)

# Quantization

- [Overview](quantization/README.md)
- [FP32 (Baseline)](quantization/fp32.md)
- [Half-Precision (FP16 / BF16)](quantization/fp16_bf16.md)
- [Scalar Quantization (SQ4 / SQ8)](quantization/sq.md)
- [Scalar Uniform (SQ4 / SQ8 Uniform)](quantization/sq_uniform.md)
- [Product Quantization (PQ)](quantization/pq.md)
- [PQ FastScan](quantization/pqfs.md)
- [RaBitQ](quantization/rabitq.md)
- [Transform Quantizer (TQ)](advanced/quantization_transform.md)

# Developer Guide

- [Code Structure](development/code_structure.md)
- [Building](development/building.md)
- [Running Tests](development/testing.md)
- [Contributing](development/contributing.md)

# Advanced Features

- [Build and Train](advanced/build_and_train.md)
- [Range Search](advanced/range_search.md)
- [Calculate Distance by ID](advanced/calc_distance_by_id.md)
- [Filtered Search](advanced/filtered_search.md)
- [Attribute Filter (Hybrid Search)](advanced/attribute_filter.md)
- [Serialization](advanced/serialization.md)
- [Memory Management](advanced/memory.md)
- [Per-Search Allocator](advanced/search_allocator.md)
- [Index Introspection](advanced/introspection.md)
- [Extensibility](advanced/extensibility.md)
- [Graph Enhancement](advanced/enhance_graph.md)
- [Hybrid Memory-Disk Index](advanced/hybrid_index.md)
- [Extra Info](advanced/extra_info.md)
- [Index Lifecycle Management](advanced/index_lifecycle.md)

# Performance and Tuning

- [Best Practices](resources/best_practices.md)
- [Metric Semantics](resources/metric_semantics.md)
- [Optimizer (Tune)](advanced/optimizer.md)
- [Benchmarks](resources/performance.md)
- [Evaluation Tool](resources/eval.md)
- [HDF5 Dataset Format](resources/dataset_format.md)
- [Index Analysis Tool](resources/analyze_index.md)

# Resources

- [Release Notes](resources/release_notes.md)
- [Roadmap](resources/roadmap_2025.md)
- [Community](resources/community.md)
- [Related Projects](resources/related_projects.md)
- [Research Papers](resources/research_papers.md)
- [Contributors](misc/contributors.md)
