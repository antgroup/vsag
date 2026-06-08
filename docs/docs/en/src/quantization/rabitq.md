# RaBitQ

`rabitq` is VSAG's binary / low-bit quantizer. In its default mode each
coordinate is encoded with **1 bit**, giving the highest compression ratio
of any built-in quantizer. A second mode (`rabitq_version =
"split_1bit_7bit"`) splits the representation into a 1-bit base and a
7-bit refinement to recover much of the accuracy at ~8 bits/dim, while
preserving the 1-bit fast distance kernel.

![RaBitQ: encode each coordinate by its sign relative to a random hyperplane](../figures/quantization/rabitq-hyperplane.svg)

> Implementation: `src/quantization/rabitq_quantization/rabitq_quantizer.cpp`,
> parameter file `rabitq_quantizer_parameter.cpp`.
> Design notes: `docs/rabitq_1xbit_new_repo_guide.md`,
> `docs/rabitq_split_1bit_7bit.md`.

## When to use it

- **Maximum compression.** 1-bit codes are the smallest possible storage
  for dense vectors.
- **High-dim embeddings** where rotation + binarization preserves enough
  geometry for nearest-neighbor search.
- Combined with a precise reorder store (`fp16` / `fp32`) — the standard
  recipe is "RaBitQ + reorder", because the binary distance is noisy on
  its own.

For best accuracy, also enable `rabitq_use_fht: true` or wrap with a
[Transform Quantizer](../advanced/quantization_transform.md) chain such
as `"pca, rom, rabitq"`.

## Memory cost (codes only)

- `rabitq_bits_per_dim_base = 1`: `ceil(dim / 8)` bytes per vector. With
  `dim = 768` that is 96 bytes (vs 3072 for fp32 → 32× smaller).
- `rabitq_bits_per_dim_base = 8` (split-1+7 mode stores additional bits):
  ~`dim` bytes per vector.

## Parameters

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pca_dim` | int | `0` (= input dim) | Optional PCA preprocessing dimension applied inside RaBitQ. `0` means no PCA reduction (`rabitq_quantizer_parameter.cpp:30-32`). |
| `rabitq_bits_per_dim_query` | int | `32` | Bits per dimension used to encode the **query** during search. Allowed values: `4` or `32` (`rabitq_quantizer_parameter.cpp:38-43`). |
| `rabitq_bits_per_dim_base` | int | `1` | Bits per dimension for the **base** (stored) codes. Allowed range `[1, 8]` (`rabitq_quantizer_parameter.cpp:45-54`). Use `1` for pure 1-bit RaBitQ. |
| `rabitq_version` | string | `"standard"` | One of `"standard"` (1-bit) or `"split_1bit_7bit"`. The split version requires `rabitq_bits_per_dim_query = 32` (`rabitq_quantizer_parameter.cpp:55-67`). |
| `rabitq_error_rate` | float | `1.9` | Controls the error budget of the encoder; must be finite and positive (`rabitq_quantizer_parameter.cpp:68-75`). |
| `use_fht` | bool | `false` | If `true`, applies a Fast Hadamard Transform rotation before binarization. Improves accuracy on anisotropic data with cheap O(dim log dim) cost (`rabitq_quantizer_parameter.cpp:76-78`). |

On HGraph these are exposed as the top-level keys `rabitq_pca_dim`,
`rabitq_bits_per_dim_query`, `rabitq_bits_per_dim_base`, `rabitq_version`,
`rabitq_error_rate`, and `rabitq_use_fht` — the last one is the HGraph
alias for the quantizer's `use_fht` key and is rewritten by the index
layer (`src/algorithm/hgraph.cpp:473-480`, names defined in
`src/constants.cpp:142-148`). Pyramid exposes the same `rabitq_*` keys
(`src/algorithm/pyramid.cpp:698-699`).

```json
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 768,
    "index_param": {
        "base_quantization_type": "rabitq",
        "rabitq_use_fht": true,
        "rabitq_pca_dim": 0,
        "rabitq_bits_per_dim_base": 1,
        "rabitq_bits_per_dim_query": 32,
        "max_degree": 32,
        "ef_construction": 300,
        "use_reorder": true,
        "precise_quantization_type": "fp32"
    }
}
```

Swap to the higher-accuracy split mode. The split layout is selected by a
combination of two keys — `rabitq_version: "split_1bit_7bit"` selects the
1+7 RaBitQ encoding, and `base_codes_type: "rabitq_split"` switches the
storage datacell. Setting `rabitq_version` alone does **not** activate the
split datacell path; both keys must be set together (see
`docs/rabitq_split_1bit_7bit.md`):

```json
{
    "base_quantization_type": "rabitq",
    "base_codes_type": "rabitq_split",
    "rabitq_version": "split_1bit_7bit",
    "rabitq_bits_per_dim_base": 8,
    "rabitq_bits_per_dim_query": 32,
    "rabitq_use_fht": true
}
```

## Training

`NEED_TRAIN` is set. Training learns the rotation and per-dimension
statistics that make the 1-bit encoding well-balanced. The optional FHT
rotation is fixed (not learned), so it adds no extra training cost; PCA
preprocessing (when `pca_dim > 0`) trains a projection matrix.

## Metric compatibility

`l2`, `ip`, `cosine` — all supported. The binary distance kernel is a
popcount over XORed code words; for `ip` / `cosine` the implementation
also tracks a residual norm so the inner-product estimate is unbiased.

## Tips

- **Always enable reorder** unless you have validated that 1-bit recall
  is acceptable on your data. `use_reorder: true` +
  `precise_quantization_type: "fp32"` is the safe default.
- **Rotate first.** For un-normalized data, set `rabitq_use_fht: true` or
  use a `tq` chain that includes `rom` / `fht`.
- **Split mode for accuracy.** `rabitq_version: "split_1bit_7bit"` keeps
  the 1-bit fast path for graph traversal and adds a 7-bit refinement
  for re-ranking; expect significantly higher recall at ~8× the code
  size of pure 1-bit.

## Related pages

- [Transform Quantizer](../advanced/quantization_transform.md)
- [HGraph index](../indexes/hgraph.md)
- Design notes: `docs/rabitq_1xbit_new_repo_guide.md`,
  `docs/rabitq_split_1bit_7bit.md`
- [Quantization overview](README.md)
