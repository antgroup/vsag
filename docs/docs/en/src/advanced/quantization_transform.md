# Quantization Transform

The **Transform Quantizer** (`base_quantization_type: "tq"`) chains one or more vector
transformations in front of a final quantizer. Transformations reshape vectors so a downstream
quantizer can encode them more accurately or compactly — for example, rotate vectors so their
energy is spread across dimensions (RaBitQ / SQ benefit greatly), or reduce dimensionality with
PCA before storing them.

> Runnable example: `examples/cpp/501_quantization_transform.cpp`.

## Why a transform layer

A pure quantizer compresses vectors directly. With low-bit quantizers (e.g. `sq4`,
`sq*_uniform`, `rabitq`) accuracy depends heavily on the **distribution** of vector
coordinates: heavy-tailed or anisotropic dimensions waste code bits. A transform layer
mitigates this:

- **Random rotations** (`rom`, `fht`) decorrelate coordinates so a uniform/scalar quantizer
  works better on each axis.
- **PCA** (`pca`) reduces dimensions while keeping most of the variance — code size shrinks
  proportionally.
- **MRLE** (`mrle`) is a metric-recoverable low-rank encoding tailored to L2/IP search.

The transform output then feeds a standard quantizer (`fp32`, `sq8`, `sq8_uniform`, `rabitq`,
…), which actually stores the codes. The whole chain is referred to as **`tq` (Transform
Quantizer)**.

## Quick start

`tq` is currently exposed as a **public, externally configurable** quantization type only by
**HGraph**. HGraph maps the top-level keys `tq_chain` and `rabitq_pca_dim` into the nested
`base_codes.quantization_params` JSON via its external-parameter mapping
(`src/algorithm/hgraph.cpp:370-385`). IVF, BruteForce, Pyramid and WARP all internally render
a `tq_chain` field into their inner JSON template, but none of them expose `tq_chain` (or any
other TQ parameter) in their external mapping today. `CheckAndMappingExternalParam` rejects
unknown external keys with `invalid config param`
(`src/utils/util_functions.cpp:50-53`), so passing `tq_chain` in the `index_param` JSON of
those indexes will fail at index construction. Configuring TQ on non-HGraph indexes
therefore requires code-side changes to add the external mapping.

```cpp
std::string params = R"({
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 128,
    "index_param": {
        "base_quantization_type": "tq",
        "tq_chain": "pca, rom, sq8_uniform",
        "rabitq_pca_dim": 64,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
})";

vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
vsag::Engine engine(&resource);
auto index = engine.CreateIndex("hgraph", params).value();
index->Build(base);
auto result = index->KnnSearch(query, topk, search_params).value();
```

In the example above, base vectors are first projected from 128 to 64 dimensions (`pca`),
randomly rotated (`rom`), then quantized with `sq8_uniform`. Reordering is enabled, so HGraph
keeps an `fp32` precise copy and re-ranks the top candidates returned by the graph search
(`include/vsag/index.h`; see [Memory Management](memory.md) for the storage implications).

## `tq_chain` syntax

`tq_chain` is a **comma-separated string**: one or more transformer names followed by exactly
one final quantizer name. Whitespace around tokens is trimmed
(`src/quantization/transform_quantization/transform_quantizer_parameter.cpp:53-74`).

```
"<transform1>, <transform2>, ..., <quantizer>"
```

Examples:

| Chain | Effect |
|---|---|
| `"rom, fp32"` | Random rotation, then store as fp32 (used for tests / sanity baselines). |
| `"fht, sq8_uniform"` | Fast Hadamard rotation, then 8-bit uniform scalar quantization. |
| `"pca, rom, sq8_uniform"` | PCA reduction, random rotation, then 8-bit uniform — the example chain. |
| `"pca, rom, rabitq"` | PCA + rotation feeding the RaBitQ binary quantizer. |
| `"mrle, fp32"` | MRLE projection then store as fp32 (MRLE must be first). |

Constraints (`transform_quantizer_parameter.cpp:33-45`):

- The chain must contain **at least one transformer + one quantizer** (length ≥ 2). An empty
  or single-token chain raises `INVALID_ARGUMENT`.
- The **last token must be a quantizer that the TQ flatten path can dispatch**: one of
  `fp32`, `sq8`, `sq8_uniform`, `sq4`, `sq4_uniform`, `bf16`, `fp16`, `pq`, `pqfs`, `rabitq`
  (`src/datacell/flatten_interface.cpp:126-164`). `TransformQuantizerParameter` parses a
  slightly wider set of names (it also accepts `sparse`, `int8`, `tq`), but the flatten
  factory does not have a dispatch branch for `int8`/`tq` and explicitly rejects `sparse`
  when `is_transform_quantizer` is true (`src/datacell/flatten_interface.cpp:166`), so using
  any of those three as the terminal quantizer fails at index construction with an
  "unsupported quantization type" error.
- Any unrecognized transformer name raises `INVALID_ARGUMENT: invalid transformer name`
  (`transform_quantizer.h:225-227`).

## Supported transformers

The factory at `src/quantization/transform_quantization/transform_quantizer.h:192-227`
recognizes four transformer names today:

| Name | Output dim | Description | Implementation |
|---|---|---|---|
| `pca` | `pca_dim` if set, else input dim | Principal-Component-Analysis projection; reduces dim while keeping variance. | `src/impl/transform/pca_transformer.h` |
| `rom` | input dim | Random Orthogonal Matrix; rotates vectors to decorrelate dimensions. | `src/impl/transform/random_orthogonal_transformer.h` |
| `fht` | input dim | Fast Hadamard / KAC random rotation; cheaper variant of `rom`. | `src/impl/transform/fht_kac_rotate_transformer.h` |
| `mrle` | `mrle_dim` (≤ input dim) | Metric-Recoverable Low-rank Encoding; **must be the first transformer in the chain**. | `src/impl/transform/mrle_transformer.h` |

Notes:

- `mrle` placement is enforced at `transform_quantizer.h:155-159` and `mrle_dim ≤ input_dim`
  at `transform_quantizer.h:217-220`.
- Other strings declared in headers (`residual`, `normalize`) are **not** wired into the
  factory and will be rejected.

## Transformer parameters

The transformer JSON is read by `VectorTransformerParameter::FromJson`
(`src/impl/transform/vector_transformer_parameter.cpp:22-35`):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `pca_dim` | int | `0` (= input dim) | Output dim of the `pca` transformer. |
| `mrle_dim` | int | `0` (= input dim) | Output dim of the `mrle` transformer. |
| `input_dim` | int | auto | Auto-populated by the chain — do not set manually. |

### HGraph external mapping

When using HGraph, two top-level shortcuts are mapped into the nested quantizer params
(`src/algorithm/hgraph.cpp:370-385`):

- `tq_chain` → `base_codes.quantization_params.tq_chain`
- `rabitq_pca_dim` → `base_codes.quantization_params.pca_dim`

The name `rabitq_pca_dim` predates Transform Quantizer; when the chain includes `pca`, it
drives the **`pca` transformer's output dim** (it is not RaBitQ-specific). When the chain
ends in `rabitq` without `pca`, the same key configures RaBitQ's own PCA preprocessing
(`src/quantization/rabitq_quantization/rabitq_quantizer_parameter.cpp:30`).

## Reordering and the precise codes store

Transform chains lose some information by design (rotation is lossless, but `pca` /
`sq*_uniform` / `rabitq` are not). Combining `tq` with **reorder** — keep a precise (typically
`fp32`) copy of every vector and re-rank the top candidates — restores accuracy with a
modest memory cost:

- `use_reorder: true` makes HGraph keep a second flatten store, the **precise codes store**
  (`src/algorithm/hgraph.cpp:76-79`).
- `precise_quantization_type` selects its quantizer (`fp32` default; can be `fp16` / `bf16` /
  `sq8` if you want to trade memory for accuracy).
- At search time the graph walk uses the cheap `tq` base codes, then the top-K are re-scored
  against the precise codes (`hgraph.cpp:978-981` and surrounding sites).

`use_reorder` and `precise_quantization_type` are not specific to `tq` — they also apply when
`base_quantization_type` is `sq8`, `pq`, `rabitq`, etc. See the table in
[HGraph index](../indexes/hgraph.md) for the full per-index parameter list.

## Choosing a chain

A pragmatic rule of thumb:

| Goal | Suggested chain | Notes |
|---|---|---|
| Memory-aggressive, accuracy-restored | `"pca, rom, sq8_uniform"` + `use_reorder: true`, `precise_quantization_type: "fp32"` | Example 501 baseline. |
| Maximum compression | `"pca, rom, rabitq"` + reorder | 1-bit quantization with rotation cleanup; expect noticeable accuracy loss without reorder. |
| Anisotropic data, no dim reduction | `"rom, sq8_uniform"` or `"fht, sq8_uniform"` | Use `fht` for lower build cost on high dim. |
| Distance-preserving low-rank | `"mrle, fp32"` | Metric-aware reduction, no further quantization. |

Always benchmark on your own data — the right tradeoff between `tq` aggressiveness and
`use_reorder` depends on dataset distribution, target recall, and memory budget.

## Compatibility and merge

Two `tq` configurations are considered compatible only when the chain length, every
transformer name, and the final quantizer all match
(`src/quantization/transform_quantization/transform_quantizer_parameter.cpp:99-117`). This
matters for serialization round-trips and for any future merge / clone operations across
indexes — keep the chain string stable across builds you intend to combine.

> **Chain string equality is necessary but not sufficient.** The `tq_chain` token list does
> not encode transformer parameters such as `pca_dim` / `mrle_dim` (read as separate sibling
> JSON keys at `src/quantization/transform_quantization/transform_quantizer.h:200-216`) or the
> internal parameters of the terminal quantizer (e.g. `pq` subspace count, `rabitq` rotation
> seed). These parameters change the effective code dimension and layout, so for two builds
> to be practically merge-/clone-compatible you must keep the **entire** transform + quantizer
> parameter set consistent, not just the chain string.

## Related pages

- [HGraph index](../indexes/hgraph.md) — parameter reference for `base_quantization_type`,
  `use_reorder`, `precise_quantization_type`.
- [Memory Management](memory.md) — memory cost of base + precise stores.
