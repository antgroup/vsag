# Independent VSAG Lite v0.1

This experimental entry point starts with exact FP32 squared-L2 BruteForce and
selective source reuse, rather than conditionally compiling the Full library.
It currently targets Linux x86_64, C++17 and the repository compiler baseline.
The default Full build is unchanged. Run these commands from the repository root:

```sh
cmake -S lite -B build-lite-first -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build-lite-first -j2
./build-lite-first/lite_tests '[lite]'
./build-lite-first/lite_tests '[lite-scale]'
cmake --install build-lite-first --prefix "$PWD/install-lite-first"
cmake -S lite/example -B build-lite-consumer -DCMAKE_PREFIX_PATH="$PWD/install-lite-first"
cmake --build build-lite-consumer -j2
./build-lite-consumer/lite_example "$PWD/new-example.lite"
```

Unlike the Full `make` targets, `cmake -S lite` is a separate dependency-isolated
entry point. Tests use the existing pinned Catch2 v3.7.1 configuration. For an
offline build, pass `-DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/catch2-v3.7.1`.
Tests are invoked directly through Catch2; no parallel custom testing framework
or CTest suite is introduced. With tests disabled, no Catch2 download is needed.

## API and ownership

`vsag::lite::Index::Create(dim)` creates a fixed-dimension index. `Add`, `Update`,
`Search`, `Save`, and static `Load` use the existing `tl::expected` / `vsag::Error`
contract without calling Full's global logger. `Remove` returns a boolean.
Inputs must contain the specified number of finite float32 values. Calls must
be externally serialized. Results own their memory; internal slots are not exposed.

- Single-record Add rejects duplicate IDs; Update rejects absent IDs.
- Remove returns false for absent IDs. A removed external ID can be inserted again.
- Add failure leaves logical records unchanged, though reserved capacity may grow.
- Search returns `min(k, Size())` entries sorted by squared-L2 then ascending ID.
  Valid queries with k=0 or an empty index return an empty result.
- FP32 accumulation can overflow to positive infinity for extreme finite inputs;
  those distances tie and are ordered by ID. This is not arbitrary-precision L2.
- NaN/Inf inputs and dimension mismatches are rejected even for k=0.
- Memory allocation failures are translated to Error where caught; constructing
  an error itself may still allocate. No hard real-time or no-throw OOM guarantee.

## Data organization and reuse

The private implementation owns contiguous FP32 records, external IDs, and an
ID-to-slot map. BruteForce scans use the official generic `ComputeL2SqrImpl`
kernel. Physical removal uses the same last-record-to-hole principle as Full
BruteForce, but retains capacity for reuse. It does not promise immediate RSS
reduction. The minimal implementation intentionally avoids Full Factory,
InnerIndexInterface, attribute and multi-vector dependencies.

The first version keeps these small responsibilities in one private translation
unit rather than creating unused abstract interfaces. Before adding graph search,
extract the store boundary and select a graph-safe slot/deletion strategy.
Reference LazyHGraph's build-before-publish phase transition rather than inventing
a new ANN algorithm. Quantization should separate candidate traversal from code
encoding/distance. mmap requires explicit mapping lifetimes and read/write policy.
None of these future capabilities is implemented in v0.1.

## Snapshot v1

The binary format is separate from Full VSAG. All numeric fields are little endian:

| Offset | Field |
| --- | --- |
| 0 | 8 bytes `VSAGLT01` |
| 8 | uint64 version = 1 |
| 16 | uint64 dimension |
| 24 | uint64 record count |
| 32 | uint64 payload bytes = count * (8 + 4 * dimension) |
| 40 | uint64 representation = 1 (IEEE754 FP32, squared-L2) |
| 48 | count signed int64 IDs encoded as two's-complement bits |
| 48 + 8 * count | contiguous row-major FP32 values |

Load requires a seekable stream and consumes its remaining bytes, rejecting
truncation, trailing bytes, invalid sizes/versions, duplicate IDs and non-finite
values before publishing a new index. Loading does not mutate an existing index.
There is no checksum: valid-looking bit corruption cannot always be detected.
Save writes at the current output position; the caller owns flushing/closing,
atomic replacement, permissions and crash durability. Partial output may remain
after failure. Write failures currently use existing `READ_ERROR` because the
shared ErrorType has no separate write-error code.

## Verification and limitations

`[lite]` covers CRUD, independent-reference search and malformed snapshots.
The opt-in `[lite-scale]` case runs a 100,000 x 128 synthetic end-to-end workflow;
it is a correctness smoke test, not a realistic retrieval benchmark or a claim
of performance improvement. Full/Lite comparisons on at least two datasets or
scales, cold/warm loading, CRUD latency/throughput, recall and memory remain work.

For source coverage add `-DENABLE_COVERAGE=ON` to a Debug build, run the tests,
and collect gcov results. Only report actually measured coverage; Full coverage
is not implied. The installed consumer above checks that the new target can be
used without linking `libvsag.so`. Full API/ABI replacement, bindings, graph
search, sparse vectors, WARP, quantization, mmap and concurrent calls are out of scope.
