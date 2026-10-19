# Dependencies

## Where they are declared

**`pyproject.toml` is the only source of truth.** There is deliberately no
`requirements.txt` and no lock file inside the package:

* It is a *library*, not an application. A pinned requirements file would
  conflict with `pyproject.toml`, and a lock file belongs to whoever deploys it,
  not to the package.
* The repository already carries unrelated dependency files for other
  components (`python/setup.py`, `scripts/perf_reports/requirements.txt`).
  Adding a third convention next to the package would invite confusion about
  which one is authoritative.

To install:

```bash
pip install -e embeddinggemma2_kit             # core: text + image
pip install -e 'embeddinggemma2_kit[media]'    # + audio/video decoding
pip install -e 'embeddinggemma2_kit[finetune]' # + PEFT adapter interfaces
pip install -e 'embeddinggemma2_kit[dev]'      # + pytest / ruff
```

## Declared requirements

| Requirement | Floor | Why that floor |
|---|---|---|
| `python` | `>=3.10` | `X \| None` types in annotations are evaluated at import time |
| `numpy` | `>=1.24` | `np.errstate` / `np.stack` behaviour used by the index backends |
| `torch` | `>=2.0` | `torch.inference_mode` and the SDPA attention path |
| `torchvision` | `>=0.15` | required **even for text-only use** — loading the checkpoint builds the multimodal processors and the video processor imports torchvision at construction |
| `transformers` | `>=5.19` | first release line that knows the `embedding_gemma2` architecture |
| `sentence-transformers` | `>=6.1.0` | **hard floor**: earlier versions ignore the ordering of multimodal input dicts |
| `pillow` | `>=10.0` | image decoding |
| `av` `soundfile` `soxr` | `>=12.0` `>=0.12` `>=0.3` | `[media]` extra: video demux, audio decode, resampling to 16 kHz |
| `peft` | `>=0.13` | `[finetune]` extra: adapter interfaces |

`pyvsag` is **not** a dependency. It is an optional runtime backend; a missing
one raises `IndexBackendUnavailable` rather than silently degrading. See
[design.md](design.md#decoupling-from-pyvsag).

## Verified versions

The repository does not pin these — they are the versions the results in this PR
were produced and checked with, recorded so a reviewer can reproduce or explain a
difference.

| | macOS arm64 | Linux x86_64 |
|---|---|---|
| Python | 3.10.22 | 3.14.4 |
| torch | 2.14.1 | 2.14.1+cpu |
| torchvision | 0.29.1 | 0.29.1+cpu |
| transformers | **5.20.1** | **5.19.0.dev0** |
| sentence-transformers | 6.1.0 | 6.1.0 |
| numpy | 2.2.6 | 2.5.3 |
| pillow | 12.3.0 | 12.3.0 |
| av | 17.1.0 | 19.0.1 |
| soundfile | 0.14.0 | 0.14.0 |
| soxr | 1.1.0 | 1.1.0 |
| peft | 0.21.2 | not installed |

Index environment (separate, only for pyvsag): Python 3.10.22, `pyvsag`
1.1.0.dev159, **numpy 1.26.4**.

### One caveat: a pre-release on the x86 host

The x86 host runs `transformers 5.19.0.dev0`, a build from main made before
5.19.0 was released. Under PEP 440 a `.devN` release sorts *before* the release
it precedes, so it does not satisfy the declared `>=5.19` floor and
`pip check` reports:

```
embeddinggemma2-kit 0.1.0 has requirement transformers>=5.19,
but you have transformers 5.19.0.dev0.
```

The floor is deliberately **not** lowered to admit the pre-release: users
installing this package should get a release, not a snapshot. A clean
`pip install` resolves to a release — on macOS that is 5.20.1 — and passes
`pip check`. Both environments were exercised, so the code is known to work on a
release and on the pre-release: 26 tests plus the text-to-image retrieval check
pass on each.

If you need to reproduce the x86 environment exactly:

```bash
pip install 'transformers==5.19.0.dev0'   # needs the pre-release index
```

## Weights are not a dependency

The checkpoint (~1.5 GB) is never installed or bundled; it is governed by the
Gemma terms of use, not this repository's license. It is fetched once by
`scripts/fetch_weights.py`, which **symlinks** the result into `weights/` and
never copies it into the tree.

`EG2_WEIGHTS_SOURCE` picks the origin, so the same checkout works on a laptop, in
CI, and behind a corporate mirror without touching the code:

| Value | Origin |
|---|---|
| `huggingface` (default) | `google/embeddinggemma-2` via `huggingface_hub`, so each user accepts the Gemma terms themselves |
| `local` | a checkpoint you provide (`--src` / `$EMBEDDINGGEMMA2_SRC`); no network |
| `http(s)://…/` | any static host — an internal OSS/S3 mirror, verified against `weights.sha256` |

The required file set is 11 entries, not just the obvious config and weights.
Three are easy to leave out and each one breaks something specific:

| File | If missing |
|---|---|
| `sentence_bert_config.json` | model silently loads **text-only**; image/audio/video raise "Modality is not supported" |
| `chat_template.jinja` | text processor fails with "does not have a chat template" |
| `1_Pooling/config.json` | loading fails with `Pooling.__init__() missing ... 'embedding_dimension'` |

`2_Normalize/config.json`, `tokenizer.model` and `README.md` are optional — the
first falls back to defaults and the other two are unused by the kit.

To publish an internal mirror, generate the manifest where the checkpoint lives
and upload it alongside:

```bash
python scripts/fetch_weights.py --src /path/to/checkpoint --write-manifest
```

`weights.sha256` is both the file list and the integrity check, so downloads from
a URL source are verified by default. See `NOTICE.md`.

## Reproducing an environment

```bash
# 1. core + optional extras
pip install -e 'embeddinggemma2_kit[media,finetune,dev]'

# 2. check nothing conflicts
python -m pip check

# 3. provision the weights and run the tests
python embeddinggemma2_kit/scripts/fetch_weights.py --src /path/to/embeddinggemma-2
python -m pytest embeddinggemma2_kit/tests -q
```

On a host where the resolved torch has no bf16 hardware path (Apple Silicon CPU),
pick the dtype explicitly — see [platforms.md](platforms.md).