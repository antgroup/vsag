# visualize_index

`visualize_index` inspects a streaming-serialized VSAG index file and shows each
part's byte-size proportion.

```bash
./build/tools/visualize_index/visualize_index --index_path ./index.bin
./build/tools/visualize_index/visualize_index --index_path ./index.bin --html ./index.html
```

The terminal output remains a raw/exploded layout summary. The logical block
view groups every TLV header with its payload, so blocks such as `base_codes`,
`bottom_graph`, `ivf_bucket`, `sindi_windows`, and `pyramid_hierarchies` are
shown as single size-proportional units.

The HTML output uses a grouped exploded horizontal layout: related small segments
are merged into logical blocks, very large blocks are visually folded with
stripes, and tables keep exact byte sizes and real percentages.

The tool currently supports the streaming format whose magic is `vsagstm0`,
including streaming BruteForce, HGraph, IVF, SINDI, and Pyramid index files.
Footer-based legacy index files are reported as unsupported.
