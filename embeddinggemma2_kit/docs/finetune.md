# Fine-tuning

## Status

The **interface layer** is implemented and tested on CPU. The **training loop is
not implemented in this version**, because the development host exposes no GPU.
`FinetuneTrainer.fit` raises a clear `FinetuneError` rather than starting a run
that would take days on CPU and produce an adapter of unknown quality.

Implemented and verified on CPU:

* `LoRAConfig` validation (including the dtype constraint below).
* `PairDataset` validation of `(media, caption)` pairs, with file checks.
* Loading an existing PEFT adapter and reporting its configuration.
* `verify_merged_checkpoint` — loads a merged checkpoint and checks it still
  produces `(1, 768)` embeddings.

## Interface

```python
from embeddinggemma2_kit.finetune import LoRAConfig, FinetuneTrainer, PairDataset

config = LoRAConfig(r=8, alpha=16, target="audio_attention", epochs=3, learning_rate=1e-4)
dataset = PairDataset.from_jsonl("pairs.jsonl")
train, valid = dataset.split(validation_fraction=0.1)

trainer = FinetuneTrainer(embedder=emb, config=config, output_dir="out/")
trainer.fit(train, valid)                  # requires a CUDA GPU
trainer.save_adapter("out/adapter")        # adapter directory
trainer.merge_and_export("out/merged")     # loadable by Embedder(model_path=...)
```

Training pairs are one JSON object per line:

```json
{"media": "/data/clip.wav", "caption": "a dog barking", "media_modality": "audio"}
```

## Constraints for a GPU run

These come from the previously verified audio fine-tuning experiment on this
project and are encoded in `LoRAConfig`:

* **Only the audio attention LoRA parameters and the existing audio projection
  are updated.** The rest of the model stays frozen, and the model remains
  multimodal.
* **No FP16 activations.** `LoRAConfig` rejects `dtype="float16"`. Use
  `float32`. On Turing-class GPUs (compute capability 7.5) there is no native
  BF16 either, so `float32` is the correct choice there.
* **Hold out the evaluation set** from optimization and model selection. In the
  earlier experiment a 289-item validation set was a seeded holdout from the
  2,893-item development set, and the 1,045 evaluation recordings were never
  used for selection.
* **Training data must be public datasets** (Clotho and similar). Nothing
  private is bundled with this kit.

## Environment

PEFT is an optional extra so CPU inference users never install it:

```bash
pip install -e 'embeddinggemma2_kit[finetune]'
```

The earlier GPU environment used Python 3.10.12, PyTorch 2.7.1+cu118 and PEFT
0.21.2, with Transformers from a pinned source checkout.

## Verifying a merged checkpoint without a GPU

`verify_merged_checkpoint` runs on CPU and catches the failure that matters
most — a merge that produced an unloadable or wrong-dimensioned model:

```python
trainer.verify_merged_checkpoint("out/merged")
# -> {"path": ..., "parameters": 744371488, "embedding_dim": 768, "probe_shape": [1, 768]}
```