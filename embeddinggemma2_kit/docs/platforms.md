# Platform guide

The checkpoint is the same everywhere; what changes is the **device** and
**dtype** you should give it. Getting this wrong is not merely slow — on
Apple Silicon the default combination is ~12x slower than the fastest one.

## At a glance

| Host | Recommended | Why |
|---|---|---|
| Apple Silicon (M1–M5) | `device="mps", dtype="bfloat16"` | GPU is ~8x faster than the same machine's CPU |
| Apple Silicon, CPU only | `device="cpu", dtype="float32"` | arm64 has no bf16 hardware path; bf16 emulation is 12x slower |
| x86 with AVX512-BF16 (EPYC / Xeon) | `device="cpu", dtype="bfloat16"` | bf16 is the hardware fast path here (~1.9x over fp32) |
| x86 without AVX512-BF16 | `device="cpu", dtype="float32"` | measure; bf16 may fall back to emulation |
| Anything you are unsure about | measure it — see [examples/06](../examples/06_apple_silicon.py) | |

`Embedder` defaults to `device="cpu", dtype="bfloat16"`. That default is tuned
for AVX512-BF16 servers. It is *not* the right default on arm64.

## Apple Silicon (macOS, arm64)

### Measured

Apple M5 Pro, 18 cores, 24 GB unified memory, macOS 26.4.1, torch 2.14.1,
one image, warm, medians:

| Device | dtype | ms/image | vs fastest |
|---|---|---|---|
| `mps` | `bfloat16` | **128** | 1.0x |
| `mps` | `float32` | 262 | 2.0x |
| `cpu` | `float32` | 1000 | 7.8x |
| `cpu` | `bfloat16` | 12144 | **95x** |

Text, `role="document"`, same host:

| Batch | ms/sequence | sequences/s |
|---|---|---|
| 1 | 19.5 | 51 |
| 8 | **3.1** | **327** |
| 32 | 3.6 | 275 |

Image throughput with one process per shard drops as workers contend for the
single GPU: 7.9 img/s (1 process), 10.6 (2), 11.9 (4). Beyond ~4 workers the
GPU is saturated.

### Why bf16 is the wrong default here

`bfloat16` needs hardware support to be worth anything. On arm64 PyTorch has no
bf16 instruction path, so it emulates — slower than plain `float32`, by a lot.
On EPYC/Xeon with `avx512_bf16` the same dtype hits a dedicated kernel and wins.
The dtype that is "fast" is therefore a property of the CPU, not of the model.

### Recommended

```python
from embeddinggemma2_kit import Embedder

emb = Embedder(device="mps", dtype="bfloat16")          # fastest on Apple Silicon
# emb = Embedder(device="cpu", dtype="float32")         # if you must stay on CPU
```

### Pick the device safely

`device` is passed straight through to PyTorch, so asking for `"mps"` on a host
without it fails inside torch with an unhelpful message. Check first:

```python
import torch

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"                       # not used by this kit's baselines
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def best_dtype(device: str) -> str:
    # bf16 only pays off with AVX512-BF16; on Apple Silicon's CPU it is a trap.
    if device == "cpu" and torch.backends.mps.is_available():
        return "float32"                    # arm64 CPU
    return "bfloat16"

emb = Embedder(device=best_device(), dtype=best_dtype(best_device()))
```

### Notes specific to MPS

* **`float16` is unsupported** regardless of device — the checkpoint's
  activations do not tolerate it. Use `bfloat16` or `float32`.
* **Accuracy is unaffected.** Against a `float32` CPU reference the minimum
  cosine is 0.99994 for `bfloat16` and 1.000000 for `float32`, and a 10-image
  text-to-image check scores 10/10 with both `cpu/float32` and `mps/bfloat16`.
  The choice is purely about speed.
* **First pass is slower.** The first time each distinct image is seen costs
  ~1.7x (decode paths, buffer allocation). Steady state is reached after one
  pass over the corpus.
* **Startup costs ~7 s per process** (imports, checkpoint load, MPS context).
  For fewer than ~50 images, a single process is faster than a pool.
* **Memory.** 1.5 GB per process. With 24 GB unified memory, keep worker pools
  to a few processes — the GPU is the bottleneck anyway.
* **Use `spawn`, not `fork`,** if you build a process pool: forking a process
  that has already initialised OpenMP/MPS can deadlock.

### Verify on your machine

```bash
# Benchmark device/dtype combinations and run a retrieval check.
python examples/06_apple_silicon.py

# Set the corpus location first if it is not in the default place:
EG2_DATA=/path/to/flickr30k-subset python examples/06_apple_silicon.py
```

Sample output on an M5 Pro:

```text
mps/bfloat16     128 ms/image
mps/float32      262 ms/image
cpu/float32     1000 ms/image

recommended: Embedder(device="mps", dtype="bfloat16")
```

## x86 servers

### Measured

AMD EPYC 9T24, 192 cores, 1132 GB, torch 2.14.1+cpu, one image, medians:

| dtype | 16 threads | 32 threads | 64 threads |
|---|---|---|---|
| `bfloat16` | **806** | 854 | 706 |
| `float32` | — | 1606 | — |

bf16 is ~1.9x faster than fp32 here because the CPU has `avx512_bf16`; oneDNN
dispatches a dedicated kernel (`brg_matmul:avx512_core_bf16`) for all 332
matmuls of an image encode.

### Threads

Scaling flattens early — 1 thread is 9435 ms, 16 is 806 ms (11.7x), 64 is
706 ms (13.4x). **16–32 threads is the safe range**; going wider buys little and
makes latency less predictable on a shared host (measured jitter 15–78%).

```python
Embedder(device="cpu", dtype="bfloat16", num_threads=16)
```

### Batch indexing

One encode cannot use 192 cores, so add *processes* rather than threads:

| Processes × threads | Throughput |
|---|---|
| 1 × 32 | 1.15 img/s |
| 4 × 16 | 2.64 img/s |
| 8 × 16 | 4.27 img/s |
| **16 × 8** | **4.83 img/s** |

Fewer threads per process, more processes. Splitting shards as `items[i::n]` and
reassembling in the same order keeps the output bit-identical to a serial run.

## Measured comparison

Same script, same checkpoint, same images, both hosts:

| | Apple M5 Pro (MPS) | EPYC 9T24 (192-core CPU) |
|---|---|---|
| one image | **128 ms** | 854 ms |
| image throughput (peak) | **11.9 img/s** | 4.8 img/s |
| text, batch 8 | **327 seq/s** | 68 seq/s |

A 24 GB laptop beats the 192-core server on both latency and throughput for this
model, because an image becomes a 2520-token sequence and the dense matmuls suit
a GPU. The server's advantages are memory capacity and the ability to run many
independent processes — not single-stream speed.

## Measuring on your own hardware

`examples/06_apple_silicon.py` benchmarks the combinations on whatever machine
it runs on and prints a recommendation, so it is the authority for your host
rather than the tables above. To run the same measurements non-interactively:

```bash
python examples/06_apple_silicon.py
```
