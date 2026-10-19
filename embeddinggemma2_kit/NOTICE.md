# Notice on model weights

This package contains **no model weights**.

The EmbeddingGemma 2 checkpoint (~1.5 GB) is distributed under the **Gemma
Terms of Use**, not under this repository's Apache-2.0 license. It must be
obtained separately by the user.

`scripts/fetch_weights.py` **symlinks** the checkpoint into `weights/`; it never
copies it into the repository. Where the checkpoint comes from is chosen at run
time via `EG2_WEIGHTS_SOURCE`:

* `huggingface` (default) downloads `google/embeddinggemma-2` from the hub, so
  the user accepts the Gemma terms themselves rather than inheriting them from
  this repository;
* `local` uses a checkpoint the user provisioned (`--src` / `$EMBEDDINGGEMMA2_SRC`)
  and touches no network;
* an `http(s)` base URL fetches from any static host, which is how an internal
  OSS/S3 mirror is used, verified against `weights.sha256` when present.

No internal host or credential is baked into this repository: the default is the
public upstream, and every alternative is supplied by the caller at run time.

Required checkpoint contents (see `REQUIRED_FILES` in the script; the last
three are easy to overlook but each one is load-bearing):

```
config.json                        text/vision/audio tower configuration
model.safetensors                  ~1.5 GB of weights (744M parameters, bf16)
modules.json                       sentence-transformers pipeline
tokenizer.json, tokenizer.model   tokenizer
tokenizer_config.json
processor_config.json              multimodal processor settings
preprocessor_config.json           audio feature extractor settings
config_sentence_transformers.json  task prompts (query/document prefixes)
sentence_bert_config.json          modality support; without it the model
                                   silently degrades to text-only
chat_template.jinja                text processor template
1_Pooling/config.json              pooling dimension; loading fails without it
```

`weights/` and the common model file patterns (`*.safetensors`, `*.bin`,
`*.pt`, `*.pth`, `*.onnx`, `*.gguf`) are git-ignored.
