# Lite 8-bit RaBitQ design and delivery boundary

Status (2026-10-09): implementation-aligned design for the opt-in personal development branch, audited at `bcaf9c31e89abe811de84ffad3e1f63dadaa6882`. The public candidate is implemented; performance acceptance and upstream promotion are not complete. This supersedes the earlier RABITQ enum / VSAGLT01 v4 proposal. It does not assert maintainer or mentor approval of the candidate format.

## Decisions

| Area | First delivery decision |
| --- | --- |
| Entry | `BuildGraph(VectorStorage::RABITQ8, max_degree, ef_search)` from nonempty BruteForce; `ENABLE_RABITQ_LITE_BACKEND=ON` is required and defaults OFF |
| Metric/profile | Squared L2 estimates; 8 bits = 3 filter + 5 supplement; four transform rounds, six coordinate-adjustment rounds, error rate 1.9, training seed 47 |
| Lifetime | One fixed centroid and flip model, owned codes and IDs; no online retraining, no persistent raw FP32 vectors |
| Queries | Existing Search/filter contract plus per-call SearchWithOptions; no concurrent calls |
| Mutations | Prepared encoding and local rollback journals; identical complete encodings skip topology changes after validation |
| Persistence | Keep candidate `VSAGLQ01` version 1 separate from FP32/FP16 `VSAGLT01` versions 1/2/3 and experimental `VSLRBQ01` |
| Promotion | Keep candidate on the personal branch until acceptance gaps below close; do not merge benchmark archives into the feature PR |

These are engineering decisions for this candidate, not new externally approved acceptance thresholds. No automatic BruteForce-to-graph switch is added: the caller explicitly pays the construction cost.

## Source evidence and reuse limits

| Source | Reused contract or implementation |
| --- | --- |
| `include/vsag/lite/index.h`, `src/lite/index.cpp` | Owned Index, tl::expected, explicit graph conversion with publication after success; disabled RABITQ8 returns UNSUPPORTED_INDEX_OPERATION |
| `src/quantization/rabitq_quantization/rabitq_quantizer.cpp` | FastEncodeRaBitQ coordinate adjustment; centered split-code IP and distance math |
| `src/simd/kernels/rabitq_pack.h`, `src/simd/kernels/rabitq_compute.h` | Official plane packing and isolated filter kernels; no Full allocator/datacell dependency imported |
| `src/lite/rabitq_codec.h` | Fixed model, transform/inverse, prepare_encoding, split EncodedRecords; caller-owned decode scratch |
| `src/lite/rabitq_filter_ip*.cpp` | Generic fallback; runtime AVX2+FMA or complete AVX512 feature checks with per-file ISA flags |
| `src/lite/rabitq_graph_state.h` | Search/filter traversal, frozen-model mutable state and transactional Add/Update/Remove |
| `src/lite/rabitq_backend.cpp` | Public error translation, model construction, code ownership, identity update check |
| `src/lite/rabitq_snapshot.h` | Actual little-endian format, layout and topology validation |
| `src/lite/rabitq_backend_test.cpp`, `lite/example/rabitq_example.cpp` | API lifecycle, corruption and allocation-fault regressions; executable public usage |

Lite borrows simple official algorithms and kernels. It does not claim the Full fused 16-cluster model, graph implementation or storage ABI is reused unchanged. Quantized estimates and reconstructed vectors are not exact input vectors.

## Model, construction and memory

Build requires `1 <= count <= 1000000`, `0 < dim <= 2^20`, `2 <= degree <= 64`, and `ef >= degree`, subject to container/stream capacity and finite arithmetic checks. Add cannot exceed the candidate one-million-record cap. These are safety limits, not validated practical capacity guarantees.

Current construction scans the source backend row by row to train the centroid/flip model and encode vectors, calls the existing FP32 graph builder for initial topology, then constructs RaBitQ state and publishes it on complete success. Training preserves separate finite-value and centroid passes in the original row order. It needs at most a single decoded row of scratch instead of a copied N-by-D matrix. Add/Update subsequently use the frozen model and quantized neighbor discovery. Initial construction retains the source raw vectors and temporary floating adjacency, but the build-only graph reads source rows rather than owning a second FP32 matrix; its peak still needs separate measurement. Model drift from later insertions remains a quality risk.

Training row pointers are consumed before requesting the next scratch-backed row. Topology construction reuses the existing FP32 graph algorithm with a synchronous read-only source: pair distances use separate row scratch buffers, and no borrowed graph or row pointer escapes the factory. Only owned adjacency is returned and converted to topology; temporary adjacency is released before mutable RaBitQ state construction. The original source backend remains intact until successful publication, including when any intermediate allocation fails.

For dimension D and N records, encoded plane payload per record is `8 * ceil(D/8)` bytes, with six float metadata fields (24 bytes). The model contains D floats and `4 * ceil(D/8)` flip bytes. Runtime additionally owns 8-byte external IDs, an ID/slot hash map, adjacency allocations, capacities and temporary query/journal buffers. Neither the bit count nor snapshot size represents whole-process RSS. Large candidate construction and resident mutation costs must be measured separately.

Decoding uses caller-owned scratch and returns an approximate original-space vector. Save retains model/codes directly; decoding and re-encoding is forbidden as a persistence shortcut. Seed reproducibility is bounded to the recorded implementation/toolchain: exported exact flips and centroid, rather than a promise about standard-library random distribution portability, provide Load consistency.

## Query contract and cost

1. Validate finite FP32 query and dimension. Normalize/transform using the immutable model.
2. Traverse with 3-bit filter estimates. Keep the current uniform entry points and implicit slot ring; visited checks prevent duplicates.
3. Without a filter, reorder retained traversal candidates using the complete 3+5 estimate. With a filter, rank visited accepted external IDs with that estimate; rejected nodes remain traversable.
4. Return quantized squared-L2 estimates ordered by distance, then external ID. These are approximate, not exact FP32 distances. The heuristic lower bound is not a universal recall guarantee.

SearchWithOptions zero uses the stored budget; the effective budget is bounded by the live count and requested result count. Query overrides do not modify construction/mutation defaults or saved settings. k=0 and an empty index after removals return no results. Filters may return fewer than k candidates.

Transform/normalization costs O(D) per query; each filter and supplement evaluation also scales with D, while visited storage scales with N. Graph search caches query_sum once and reuses the coarse filter inner product during complete 3+5-bit scoring. Before each four-record adjacency batch, read-only cache hints prefetch the filter plane cache lines and metadata, following native HGraph RaBitQ filter prefetching. These hints do not change scoring, traversal order, visited state, filtering or persistence, and do not allocate shared query scratch. Benefits depend on dimension, cache and hardware; this is not a blanket latency guarantee. Higher ef cannot be described as free performance optimization.

## CRUD and failure boundaries

| Operation | Current behavior | Limit |
| --- | --- | --- |
| Add | Validate ID/vector; prepare code/query, reserve storage, journal changed rows, publish record; roll back on exceptions | Growth, neighbor discovery and journal allocation still cost work |
| Update | Validate existing ID, prepare once, compare all code planes and metadata; skip identical encoding; journal code/changed rows before relinking | Equal encoding does not mean raw inputs are identical; real distribution changes can degrade graph/model quality |
| Remove | Physical last-slot compaction; repair every incoming reference even for asymmetric graphs; journal code/ID/rows and retain removed map node for rollback | Default has no incoming index and may scan O(total_edges); it is not O(degree) |
| Incoming experiment | Separate optional internal path | Still uses clone fallback for transactions; not enabled through this public delivery |

Public Add/Update/Search return tl::expected. Allocation/capacity failures map to NO_ENOUGH_MEMORY; normal vector/ID validation to INVALID_ARGUMENT; guarded unexpected operation failures to INTERNAL_ERROR. Snapshot parse failures map to INVALID_BINARY and stream write failures use the existing READ_ERROR convention. Remove's bool conflates missing IDs and transaction failure: false preserves original state but cannot report the cause. Document this; do not silently change its public signature.

Build failure keeps the old backend. Failed transactional mutation keeps logical records/topology/model intact. Save failure can leave a partially written stream, so callers needing atomic file replacement must write a temporary file and replace after success. No concurrent-call or atomic filesystem guarantee is added. Allocation fault tests support the rollback contract; they do not prove every external allocator/platform case.

## Exact candidate snapshot layout

All integer fields are little-endian u64 except the 8-byte magic. Floats are IEEE float32 bit patterns in little-endian order. IDs persist signed int64 bit patterns.

| Order | Contents |
| --- | --- |
| 1 | magic VSAGLQ01, version=1, D, N, centroid_count=D, flip_count=4*ceil(D/8) |
| 2 | D centroid floats; flip_count bytes |
| 3 | Per record: norm, code_norm, error, filter_norm, filter_error, lower_bound_error (six floats), 3*ceil(D/8) filter bytes, 5*ceil(D/8) supplement bytes |
| 4 | max_degree, ef_search, ID count=N; N external IDs |
| 5 | offset_count=N+1, neighbor_count=E; N+1 offsets and E neighbor slots |

Total snapshot bytes are `96 + 4D + 4*ceil(D/8) + N*(40 + 8*ceil(D/8)) + 8E`. This excludes filesystem and runtime hash-map/capacity overhead. The serialized records are interleaved; the in-memory filter/supplement arrays are separate contiguous arrays.

Version 1 fixes profile constants implicitly. It does NOT store seed/error rate/bit identifiers as separate fields, contrary to the old proposal. Exact model bytes suffice for Load without retraining. Any future change to fixed codec/transform semantics must use a new recognized format version with explicit compatibility tests; do not reinterpret v1. No migration to VSAGLT01 v4 is scheduled merely to unify magic strings.

Load verifies seekable remaining length before encoded allocations, dimension/count/model layout, finite and valid metadata, unique IDs, valid offsets, degree bounds, neighbor range, no self-links/duplicates, truncation and trailing bytes. It restores codes and topology, not original vectors. Existing floating formats remain separate and unchanged. Disabled-backend behavior and malformed-magic error classification must follow Index::Load as implemented, not a speculative unified loader.

## Measured evidence and remaining acceptance

Evidence comes from the committed studies under `lite/benchmark/results/`; do not label this document a new benchmark run.

| Study | Observation | What it does not establish |
| --- | --- | --- |
| integrated-load-final-20261009, 10k | Fresh-process loaded RSS about 41–66% lower than FP32 across SIFT/GIST/Cohere | Cold I/O, build peak, long changed CRUD or universal latency improvement |
| integrated-100k-pilot-20261009 | Fixed degree16/ef128 RaBitQ Recall@10 SIFT .946 / GIST .748 / Cohere .889; lower loaded RSS | Quality floor compliance at this stored budget |
| integrated-100k-budget-grid-20261009 | Observed-query selection: SIFT ef512 .975; GIST ef2048 .924; Cohere ef2048 .967 | Blind acceptance, exact matched recall, repeated stable timing or optimality |
| CRUD journal regressions | Public lifecycle, snapshot roundtrip and bounded allocation-fault rollback tests pass in prior evidence | Long real whole-vector mutation performance at accepted query settings |

The study floors (.95 SIFT/Cohere, .90 GIST) are declared experiment floors, not community-wide final requirements. Larger ef satisfies those observed floors with slower queries than floating storage. RAM-backed warm uncontrolled loading is not cold-start storage benchmarking. 100k grid stores aggregated quality, so final acceptance must add returned-neighbor evidence rather than pretending old ef128 query files prove high-ef results.

## Remaining work in dependency order

1. Finish an auditable acceptance runner using existing cached inputs: fixed single core, SIFT128/GIST960/Cohere768, 10k and 100k, exact source identity and configuration. Store per-query IDs/distances/hits and CPU/wall samples; independent truth reflects each changed live vector set.
2. Run long mixed Add/changed Update/Remove, including whole-vector changes, fixed-model drift, repeated ID churn, filtered search and post-mutation persistence. No concurrent guarantee is in scope. Record throughput, P50/P99 and quality checkpoints, plus transaction failures and fallback costs. No default budget increase during a run.
3. Compare Lite FP32/FP16/RABITQ8 with Full at declared quality targets under the same single-core protocol. Separate single-insert/update/remove cost from initial build; report build/query CPU, package/snapshot size, load-only RSS and peak separately. Use repeated fresh processes for timing claims. A 1M run is optional capacity characterization after these required small-scale gates, not a prerequisite for the Lite 10k/100k decision.
4. Optimize only identified costs: first a prepared query context caching query_sum and filter results for reorder, then scalar-vs-SIMD differential tests and measured three-distribution comparison. If build peak dominates, evaluate encoding and transferring topology without keeping duplicate graphs, keeping failure publication semantics. Incoming indexing needs its own memory/Remove tradeoff review. These are proposed changes, not implemented by this design update.
5. Freeze documentation, installed example, format compatibility and default-OFF behavior; rerun Release, ASan/UBSan, format/tidy15, candidate-scoped coverage >=90%, corruption/fault and dependency checks. Promote feature-only code to the PR source branch only after explicit authorization.

## Questions to settle with the mentor

- What minimum Recall@10, allowable P99 regression, and memory reduction justify the quantized tradeoff? Compare quality-aligned settings, rather than assuming minimum RSS wins.
- Is the first release explicitly single-threaded, with frozen-model Add/Update and bool Remove limits documented?
- Is the separate candidate format and default-OFF opt-in acceptable for upstream delivery, or should a maintained format be agreed before promotion?

Work on existing CRUD correctness, single-core evidence and query context optimization can proceed without inventing new features while these acceptance choices are discussed. Deferred features: IP/cosine, PCA/MRQ, disk supplement, mmap/zero-copy, online retraining, raw-vector exact reorder, fused clusters and ARM-specific SIMD. No completion date or global optimum is claimed until the acceptance work is measured.

### 2026-10-10 exclusive mutation-stage diagnosis

An isolated single-thread RAM profiler at source bc3e9bd measures changed-vector CRUD without changing production. Fixed 10k/600-query,degree16/maintenance128/query512,three full-ID Update/Remove/Add passes: GIST/Cohere exclusive CPU shares are nearest (including route/scoring)29.20/30.91%, incoming scans19.08/22.21%, neighbor score/sort20.15/17.90%, transformed-code reconstruction13.77/11.93%, encoding6.28/5.97%. Clocks perturb costs; these are diagnostic shares,not production improvements or a Full comparison. Ordered results/snapshot SHA match prior NONE controls; Recall remains.811167/.908167. See `lite/benchmark/results/rabitq-mutation-profile-20261010` for compiled sources,receipts,scope/calibration fixtures and audit180000operations/2400truthstates. Production tests/coverage are inherited,not rerun. Prioritize measured route/scan costs at fixed quality/budget;100k/interleaved CRUD,fresh Full alignment,cold I/O remain open.

### 2026-10-10 rejected blocked transformed-code decoder

A real production candidate adapted native RecoverOrderSQ shift/mask recovery to eight coordinates per plane byte in MutableGraphState::decode_query. Original float division,graph policy,budgets,API/model/snapshot stayed unchanged. Exhaustive256code/tail/unaligned/four-norm bitwise regression and candidate Release6/SAN5 passed. Fixed10k GIST/Cohere600queries,NONE/NONE,three alternating pairs,three full-ID changed UpdateRemoveAdd passes: CRUD median+0.18484%/+1.57367%,all pairs slower in this sample,no gain established. Candidate header/test were restored byte-exact to675aed7; restoredRelease6/SAN5 and final format/tidy15 passed,defaultlibrary SHA1a18379a unchanged. No fresh coverage/OFF/wholeFull/Node result. Evidence `lite/benchmark/results/rabitq-block-decode-20261010` independently audits1080000operations/14400truthstates,paired ordered results/snapshotSHA,183rawmembers30852726bytes and10validation exits. DefaultRecall.811167/.908167 unchanged. Next inspect existing native Batch4 inner-product scoring for graph-maintenance reuse; not yet implemented or measured. Project acceptance remains incomplete; PR2904/2926 unchanged.


### Maintenance Batch4 reuse (2026-10-10)

The existing native three-bit centered-IP Batch4 dispatch now scores four eligible link/repair neighbors together, with scalar tails and unchanged supplementary-distance formula, candidate order, tie breaking, graph budget/policy, API, snapshot and transaction rollback. A scalar golden test covers counts0..65 and dimensional tails. Three alternating fixed-budget 10k GIST/Cohere pairs observe CRUD median reductions of1.17%/1.95%; all six pairs improve and snapshot SHA/ordered IDs/hex distances/hits remain exact. This is small workload-specific evidence, not a universal/statistically established gain: GIST Update/Remove P50 slightly increase, final recall .811167/.908167 remains insufficient for the research quality floors. No query or Full advantage is claimed. Release6, ASan/UBSan5, fresh scoped coverage5, disabled4 and format/tidy15 passed; scoped Lite line coverage2862/3007=95.18%, not whole Full coverage. Initial test-only float-memcmp lint failure was repaired with explicit byte representation and all validation rerun, without retiming unchanged production. Evidence: `lite/benchmark/results/rabitq-maintenance-batch4-20261010`. Remaining 100k, interleaved true-value CRUD, fresh aligned native Full and cold-I/O/package acceptance are not completed.


### Rejected grouped incoming-row scan (2026-10-10)

A four-at-a-time integer membership gate replaced only Update/Remove's scalar row scan in an isolated production candidate. Three alternating fixed-budget 10k GIST/Cohere pairs regressed CRUD median by11.50%/14.05%; every pair was slower. Header and test were restored byte-exact to194bf86, retaining the prior Batch4 optimization. Initial/final IDs, hex distances, hits and snapshot SHA were identical; recall .811167/.908167 was not repaired. Candidate and restored Release6/SAN5 passed; final restored format/tidy15 passed. No fresh coverage/OFF/Full/Node checker result is claimed. The test-only compile/expectation failures were fixed before timing and their logs retained. An inactive1.30GB old Full static archive was removed only after all219 missing objects' recipes/source/include directories/compiler were verified; shared libraries, data and raw evidence remain. Evidence: `lite/benchmark/results/rabitq-grouped-scan-20261010`. Future structural work should examine reverse-row journaling with explicit memory costs instead of enabling the existing full-copy reverse transaction or retrying this grouped comparator.

### Internal reverse Update transaction journal (2026-10-10)

Reverse-adjacency Update now snapshots only the replaced code and affected outgoing/incoming
rows instead of copying the whole graph. Rows are saved before cache deltas and incoming edge
mutations; failure restores vectors by swaps and valid counts from restored incoming cardinality.
CACHED's first cold rebuild retains full-copy isolation, as do reverse Add/Remove. The public
constructor still disables reverse adjacency; no new public option, format or scoring change.
Known memory totals now include the reverse vectors' payload/capacity, not allocator overhead or
transient transaction storage. Snapshot rollback preserves logical state, not exact vector capacity.

Three alternating fixed-budget 10k/600-query pairs per dataset, degree16/maintenance128/query512,
CPU0/one thread and three full-ID true changed-vector Update passes observe median Update-block
reductions of 82.92% (GIST) and 85.15% (Cohere), versus the same reverse-enabled
NONE-policy full-copy parent. Ordered IDs/hex distances/hits and snapshot SHA stay identical.
This is not a public-default, mixed CRUD, query or native Full speedup. Update-only final recall
remains .858667/.935500, below research floors; quality is not repaired. Release6, sanitizer5,
fresh scoped coverage5, disabled4 and format/tidy15 passed. Internal fault injection covers2000
positions across NONE/RECOUNT/CACHED cold/warm (347 observed failures, no state/rebuild/retry
mismatch). See `lite/benchmark/results/rabitq-reverse-update-20261010` for raw data, reproducible
configuration and independent audit. Reverse Add/Remove journaling, public-path memory/time
tradeoffs,100k,interleaved CRUD,fresh aligned Full,cold I/O and complete package acceptance remain open.

### Internal reverse Add transaction journal (2026-10-10)

Reverse Add now uses the existing append/truncate transaction with snapshots of affected
outgoing and incoming rows, excluding the appended slot via row_limit. Failure restores old
reverse rows and valid counts before truncating the new slot. Cold CACHED Add retains copy
isolation; reverse Remove is not changed. Public reverse default remains false, and scoring,
model, budget, API and snapshot stay unchanged. Rollback preserves logical state, not capacities.

Three alternating fixed-budget10k/600-query pairs,degree16/maintenance128/query512,CPU0/one thread,
three full-ID Remove/true-changed-vector Add passes per builder: whole Remove/Add block medians
GIST156625.917 to65534.431ms (-58.16%),Cohere204033.785 to62390.095ms (-69.42%). Add P50 medians
2667.221 to560.127us and3437.696 to541.119us. All six pairs improve; absolute control timings vary
substantially. Remove implementation is identical; its observed timing changes are not claimed
as a separate algorithmic gain. Paired snapshots/ordered IDs/hex distances/hits remain exact;
final recall .8495/.923333 is this replace-only workload, not repaired quality or an Update-only
comparison. This is an internal-path gain, not public-default or native Full advantage.

Release6,sanitizer5,fresh scoped coverage5,disabled4,format/tidy15 passed. Internal allocation
injection covers4000 positions across Add/Update and four policies (306/363 observed failures,
zero state/rebuild/retry mismatches); preparation allocations and protection flags are now also
covered. Lite2906/3047=95.37% is scoped,not whole Full coverage. Evidence and independent720000
operation/14400 truth-state audit are in `lite/benchmark/results/rabitq-reverse-add-20261010`.
Next: reverse Remove local journaling,public-path memory/time tradeoffs and fixed-budget quality.
100k,interleaved CRUD,fresh aligned Full,cold I/O/complete deployment acceptance remain open.

### Internal reverse Remove transaction journal (2026-10-10)

Reverse Remove journals affected outgoing/incoming rows instead of copying the whole state.
Snapshot deleted/last/removed-target incoming rows before count deltas; snapshot affected rows
before erase/renumber. Rollback restores sizes,rows,valid counts using retained capacity,and
returns the extracted map node without allocation. Cold CACHED Remove retains copy isolation.
Public reverse=false/NONE,scoring,model,order,degree/ef,API,snapshot remain unchanged.
Official HGraph forward/reverse neighborhood logic is source evidence,not transplanted removal:
its label/tombstone/code-slot semantics differ. Logical state,not exact capacities,is preserved.

Three alternating10k/600-query pairs,degree16/maintenance128/query512,CPU0/thread1,three full-ID
Remove/true-vector Add passes: block medians GIST65928.356 to12563.964ms(-80.94%),Cohere62249.974
to10340.707ms(-83.39%). Remove P50/P99:1571.716/2730.809 to38.520/1169.654us and1489.659/2506.485
to30.089/1042.167us. Six pairs improve. Add unchanged; its timing is not an independent gain.
Paired snapshots,ordered IDs/hex distances/hits stay exact;replace-only recall .8495/.923333
is unchanged,below quality floors. Internal-path evidence,not public-default or Full gain.

Release6,sanitizer5,fresh coverage5,disabled4,format/tidy15 pass;scoped Lite2924/3065=95.40%.
Remove6000 fault positions cover middle,last,single-node deletion,NONE/RECOUNT/CACHED cold/warm:
692 actual failures,no state/cache/retry/continued-CRUD mismatch. Test-only narrowing warnings
fixed before timing;failed lint log retained.199 raw members,720000-operation/14400-truth audit:
lite/benchmark/results/rabitq-reverse-remove-20261010. Next public-path memory/time tradeoff,
quality,100k/interleaved CRUD,fresh aligned Full,cold I/O/complete package acceptance remain open.

### Current reverse off/on acceptance tradeoff (2026-10-10)

Same06ab591 production state and current public adapter,NONE protection;only the isolated
constructor-default overlay enables reverse adjacency. No production code/default changed.
Three alternating10k/600-query pairs,degree16/maintenance128/query512,CPU0/thread1,three
full-ID true Update/Remove/Add passes (1080000 operations): median block GIST25007.966 to
20511.163ms(-17.98%),Cohere22422.324 to17637.928ms(-21.34%). Add P50 slightly increases:
267.354 to270.933us and238.555 to240.845us. Initial/final ordered IDs/hex distances/hits
are identical for all pairs;recall .811167/.908167 remains below quality floors.

72 fresh internal-load probes load EACH identical snapshot with both flags.10000 nodes and
160000 edges add1520000B incoming logical/capacity storage. Known capacity goes11444320 to
12964320B (GIST),9523456 to11043456B (Cohere);excludes map-node/allocator/transient overhead.
Median baseline-adjusted resident KiB13766 to16776 and11970 to14900;warm internal load
6.122 to8.637ms and5.355 to7.978ms. RAM/page-cache warm,not cold I/O. Probe ru_maxrss can
include launcher inherited HWM. Whole builder peak includes matrices and simultaneous old/new
indexes;its tiny changes do not prove lower mutation memory. Retain public reverse=false.

Evidence:lite/benchmark/results/rabitq-reverse-tradeoff-20261010,independent1080000-operation/
14400-truth/72-load audit. Release6 repeated;new probe format/tidy15,syntax,malformed-input
checks pass. Production sanitizer/coverage/OFF inherited from06ab591,not rerun. Setup mode
spelling fixed before timing;post-run absent RAM probe executables rebuilt only for error checks.
Next investigate incoming allocations/layout while preserving journal/rollback and snapshot
compatibility;quality,100k/interleaved CRUD,fresh aligned Full,cold I/O/complete package remain open.

### Checked 32-bit internal incoming slots (2026-10-11)

Following native src/basic_types.h uint32_t InnerIdType,reverse source slots and their
rollback backups now use uint32_t. External IDs,outgoing slots,counts,scoring,model,budgets,
API and persisted snapshot keep their types/semantics. Checked conversion rejects overflow
before adding a reverse slot;max/max+1 regression needs no huge allocation. Public reverse=false.

Both overlays enable reverse/NONE. Control5d633c9 uses64-bit slots,candidate32-bit. Three
alternating10k/600-query pairs per dataset,degree16/maintenance128/query512,CPU0/thread1,
three full-ID true Update/Remove/Add passes: block medians20327.354 to20116.230ms(-1.04%)
GIST,17470.224 to17346.610ms(-0.71%)Cohere. Six pairs faster,small changes,no universal or
statistical speedup claim. Ordered IDs/hex distances/hits and snapshot SHA exact across pairs;
mixed recall .811167/.908167 remains below quality floors,no quality fix or Full advantage.

72 fresh same-snapshot warm loaders:160000 edge payload1280000 to640000B(-50%);unchanged
240000B vector headers,total1520000 to880000B(-42.11%). Known capacity12964320 to12324320B
GIST,11043456 to10403456B Cohere(excludes map/allocator/transients). Baseline-adjusted
resident16808 to15532KiB and14880 to13662KiB;warm load8.949 to8.434ms and8.341 to7.805ms.
Not cold I/O;probe HWM may include launcher inheritance,builder peak includes input matrices
and two indexes. Snapshot size unchanged,no package or mutation-only peak improvement claim.

Release6,sanitizer5,fresh coverage5,disabled4,format/tidy15 pass;scoped Lite2936/3077=95.42%,
not whole Full. Existing allocation-failure journal/retry/continued-CRUD and48 mixed-cache
configurations remain passing. Evidence:lite/benchmark/results/rabitq-incoming32-20261011,
independent1080000 operations/14400 truth states/72 load probes/fresh coverage/hash audit.
Next incoming allocation/layout overhead and fixed-budget quality;public reverse remains off.
100k/interleaved CRUD,fresh aligned native Full,cold I/O and complete package acceptance stay open.

### Query-owned visited state for large RaBitQ graphs

Mutable RaBitQ routing selects its visited representation once per search: below65536 slots it
retains the byte representation; at or above65536 it uses64-bit words with native VisitedList
TestAndSet mask semantics. Compile-time specialization avoids a per-node representation branch.
The buffer belongs to the individual call, with no epoch pool or retained mutable workspace.
FP32/FP16 production paths, scoring, graph maintenance, public options and snapshots are unchanged.

At100k the visited payload is100000 to12504 bytes; recorded cumulative ordinary-new query
requests are170160 to82664 bytes (-51.42%), not loaded RSS or whole-process peak memory.
Fixed-budget ordered results remain exact, but final exploratory P50 changes do not prove a
query speedup. The broader all-storage bitmap and per-node dynamic-policy prototypes were not
adopted. See `lite/benchmark/results/query-visited-specialized-20261011` for raw comparisons,
limitations, fresh scoped coverage and threshold-crossing CRUD/filter/SaveLoad sanitizer checks.

### Small-graph query heap replacement

For the byte-visited RaBitQ routing specialization, an admitted candidate replaces the full
best heap's root with a single sift-down instead of push followed by pop. The existing comparator
and candidate queue order are retained; the packed-visited and reference specializations keep
their previous operations. This extends Lite's existing heap, not a native StandardHeap root-
replacement implementation. FP32/FP16 search, public options and snapshot bytes are unchanged.

Fixed-budget public API measurements on three10k inputs reduce P50 by about3.15%/1.80%/2.40%
(SIFT/GIST/Cohere), and avoid one16384-byte allocation request per query. These are seven paired
fresh processes over100 historical queries each, not general tail-latency acceptance. GIST100k
still regresses about2.20% in P50 despite keeping the prior packed-path heap operations; the
cause is unverified and must not be dismissed as noise. This change targets small-graph CPU
and workspace cost, does not fix100k recall and does not establish superiority to native Full.
See `lite/benchmark/results/rabitq-root-public-20261011` for exact results, negative candidates
and validation scope.

### Routing-only estimate and large-graph no-filter specialization

RaBitQ graph routing now requests only the coarse distance and centered inner product.
The error/lower-bound calculation remains enabled by default for callers such as filtered
full scans that actually consume it. For at least65536 slots, an empty user filter selects
a compile-time no-filter routing path once per call; nonempty filters retain their prior
callback/ranking behavior. This does not change the distance formula, graph, search budget,
snapshot format or FP32/FP16 paths.

On GIST100k, seven rotating triples of current parent9c44640/new library/referencec55466b,
each repeating100 historical queries ten times, show mean query latency1092.762→1072.303us
(-1.87%,7/7 paired wins) and P501100.791→1082.083us(-1.70%,6/7 wins). The new P50 is2.38%
belowc55466b in the same run (7/7 wins). This supports removal of the earlier small regression
for this workload, not all100k performance acceptance. The original100-query windows give
only0.21% P50 improvement; repetitions are not1000 unique queries or a new quality cohort.
Recall remains.859, which is still an open graph-quality issue. Shared-library file size
increases395648→412512 bytes; no peak/RSS/Full/package gain is claimed here.
See `lite/benchmark/results/rabitq-routing-only-public-20261011` and
`lite/benchmark/results/rabitq-routing-repeat-20261011` for all comparisons and limits.

## Promoted implementation and historical evidence

The implementation promoted on 2026-10-12 is derived from personal source `a62e62e08db3f0d7da2e4439785feed6fcf90257`. Experimental diversity switches and diagnostic benchmark targets are excluded. Benchmark commands and `lite/benchmark/` paths below refer to that pinned experimental repository, not targets shipped in this standalone PR. The reports retain each measured source version: numbers from different stages are not a single latest-version acceptance matrix. See [the pinned reproducibility sources](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark) and [promotion evidence](https://github.com/nw9663644-eng/vsag/blob/a62e62e08db3f0d7da2e4439785feed6fcf90257/lite/benchmark/results). Current CPU-time/utilization, quality-matched native Full comparisons, cold I/O and hours-long interleaved changed-vector CRUD remain incomplete. Externally concurrent calls require caller serialization.
