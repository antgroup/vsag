# Notice on model weights

This package contains **no model weights**.

The EmbeddingGemma 2 checkpoint (~1.5 GB) is distributed under the **Gemma
Terms of Use**, not under this repository's Apache-2.0 license. It must be
obtained separately by the user.

`scripts/fetch_weights.py` only creates a **local symlink** to a checkpoint you
have already provisioned. It deliberately performs no download, so that no
internal artifact host, mirror, or credential is referenced from this public
repository.

Expected checkpoint contents:

```
config.json                        text/vision/audio tower configuration
model.safetensors                  ~1.5 GB of weights (744M parameters, bf16)
modules.json                       sentence-transformers pipeline
tokenizer.json, tokenizer.model   tokenizer
tokenizer_config.json
processor_config.json              multimodal processor settings
preprocessor_config.json           audio feature extractor settings
config_sentence_transformers.json  task prompts (query/document prefixes)
```

`weights/` and the common model file patterns (`*.safetensors`, `*.bin`,
`*.pt`, `*.pth`, `*.onnx`, `*.gguf`) are git-ignored.
