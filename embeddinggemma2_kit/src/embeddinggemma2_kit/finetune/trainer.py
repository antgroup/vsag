"""Fine-tuning interfaces.

Scope note
----------
This module provides the *interface layer* for adapting EmbeddingGemma 2. The
actual training run requires a CUDA GPU, which the current development host
does not expose, so ``FinetuneTrainer.fit`` deliberately raises a clear error
rather than attempting something slow and misleading.

What is fully implemented and tested on CPU:

* ``LoRAConfig`` validation.
* ``PairDataset`` validation of ``(media, caption)`` training pairs.
* Loading an already-produced adapter or merged checkpoint and verifying it.

Design constraints carried over from the verified audio fine-tuning experiment
(``embeddinggemma2-finetune-20261015`` on this host), recorded here so a future
GPU run does not have to rediscover them:

* Only the **audio attention LoRA parameters and the existing audio
  projection** are updated; the rest of the model stays frozen.
* The model does **not support FP16 activations**. On a Turing-class GPU
  (compute capability 7.5, no native BF16) train in **FP32**.
* The 1,045 evaluation recordings used for reporting are held out from
  optimization and model selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import ConfigError, FinetuneError

# Adapter targets that the verified experiment exercised. Kept as a whitelist so
# a typo cannot silently produce a no-op adapter that trains nothing.
SUPPORTED_TARGETS = ("audio_attention", "audio_projection")


def _require_peft():
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        from ..errors import MissingDependencyError

        raise MissingDependencyError("peft", "finetune", "adapter training") from exc
    return peft


@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters for the audio tower."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target: str = "audio_attention"
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 8
    warmup_ratio: float = 0.0
    seed: int = 20261015
    # Turing (7.5) has no native BF16 and the model forbids FP16 activations.
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ConfigError(f"r must be positive, got {self.r}")
        if self.alpha <= 0:
            raise ConfigError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.target not in SUPPORTED_TARGETS:
            raise ConfigError(
                f"target must be one of {list(SUPPORTED_TARGETS)}, got {self.target!r}"
            )
        if self.epochs <= 0:
            raise ConfigError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if self.dtype not in ("float32", "bfloat16"):
            raise ConfigError(
                "dtype must be 'float32' or 'bfloat16'. EmbeddingGemma 2 does not "
                "support float16 activations; GPUs without native bfloat16 "
                "(e.g. Turing, compute capability 7.5) must use float32."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingPair:
    """One ``(media, caption)`` training example."""

    media: str
    caption: str
    media_modality: str = "audio"
    metadata: dict[str, Any] = field(default_factory=dict)


class PairDataset:
    """Validated collection of training pairs.

    Validation is strict and up-front: a typo in a path or an empty caption
    would otherwise surface hours into a GPU run.
    """

    def __init__(self, pairs: Sequence[TrainingPair], check_files: bool = True) -> None:
        if not pairs:
            raise FinetuneError("training set is empty")
        self._pairs = list(pairs)
        for index, pair in enumerate(self._pairs):
            if not isinstance(pair, TrainingPair):
                raise FinetuneError(
                    f"pair {index} is {type(pair).__name__}, expected TrainingPair"
                )
            if not pair.caption or not pair.caption.strip():
                raise FinetuneError(f"pair {index} ({pair.media}) has an empty caption")
            if check_files and not Path(pair.media).exists():
                raise FinetuneError(f"pair {index}: media file not found: {pair.media}")

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    @property
    def pairs(self) -> list[TrainingPair]:
        return list(self._pairs)

    @classmethod
    def from_jsonl(cls, path: str | Path, check_files: bool = True) -> "PairDataset":
        """Read ``{"media": ..., "caption": ..., "media_modality": ...}`` lines."""
        p = Path(path)
        if not p.exists():
            raise FinetuneError(f"training file not found: {p}")
        pairs = []
        for line_no, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinetuneError(f"line {line_no}: invalid JSON: {exc}") from exc
            if "media" not in raw or "caption" not in raw:
                raise FinetuneError(f"line {line_no}: needs 'media' and 'caption'")
            pairs.append(
                TrainingPair(
                    media=raw["media"],
                    caption=raw["caption"],
                    media_modality=raw.get("media_modality", "audio"),
                    metadata=raw.get("metadata") or {},
                )
            )
        return cls(pairs, check_files=check_files)

    def split(self, validation_fraction: float = 0.1, seed: int = 20261015):
        """Deterministic train/validation split."""
        if not 0.0 < validation_fraction < 1.0:
            raise FinetuneError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}"
            )
        import random

        shuffled = list(self._pairs)
        random.Random(seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * validation_fraction))
        return (
            PairDataset(shuffled[cut:], check_files=False),
            PairDataset(shuffled[:cut], check_files=False),
        )


class FinetuneTrainer:
    """Adapter training entry point.

    On a CPU-only host ``fit`` raises :class:`FinetuneError` explaining what is
    required, instead of running a training loop that would take days and
    produce an adapter of unknown quality.
    """

    def __init__(
        self,
        embedder: Any = None,
        config: LoRAConfig | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or LoRAConfig()
        self.output_dir = Path(output_dir) if output_dir else None
        self._adapter_path: Path | None = None

    @staticmethod
    def require_cuda() -> None:
        """Raise unless a usable CUDA device is present."""
        import torch

        if not torch.cuda.is_available():
            raise FinetuneError(
                "fine-tuning requires a CUDA GPU, and none is available in this "
                "environment.\n"
                "The development host for this kit exposes no GPU; run training on "
                "a GPU machine instead.\n"
                "Constraints to respect there: train in float32 unless the GPU has "
                "native bfloat16 (Turing/7.5 does not), keep the base checkpoint "
                "frozen, and only adapt the audio attention LoRA plus the audio "
                "projection. See docs/finetune.md."
            )

    def build_adapter_config(self) -> Any:
        """Create the PEFT LoRA config for this target (requires ``peft``)."""
        peft = _require_peft()
        return peft.LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            bias="none",
            task_type=None,
        )

    def fit(self, dataset: PairDataset, validation: PairDataset | None = None) -> None:
        """Train the adapter. Requires a GPU."""
        if not isinstance(dataset, PairDataset):
            raise FinetuneError(
                f"fit expects a PairDataset, got {type(dataset).__name__}"
            )
        self.require_cuda()
        raise FinetuneError(
            "the training loop is intentionally not implemented in this version; "
            "the interface, config validation and dataset handling are in place, "
            "and the audio LoRA / projection adaptation was validated in the "
            "earlier standalone experiment. See docs/finetune.md."
        )

    # ------------------------------------------------------- adapter loading

    def load_adapter(self, path: str | Path) -> dict[str, Any]:
        """Load an existing adapter directory and report what it contains."""
        _require_peft()
        from peft import PeftConfig

        src = Path(path)
        if not src.exists():
            raise FinetuneError(f"adapter path not found: {src}")
        try:
            config = PeftConfig.from_pretrained(str(src))
        except Exception as exc:
            raise FinetuneError(f"not a readable PEFT adapter at {src}: {exc}") from exc
        self._adapter_path = src
        return {
            "path": str(src),
            "peft_type": str(getattr(config, "peft_type", "")),
            "r": getattr(config, "r", None),
            "lora_alpha": getattr(config, "lora_alpha", None),
            "target_modules": list(getattr(config, "target_modules", []) or []),
        }

    def verify_merged_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Check a merged checkpoint is loadable and produces the right shape.

        Useful after ``merge_and_export``: a merged model that loads is not
        necessarily a *correct* one, but one that fails to load is definitely
        wrong, and the embedding dimension must still match.
        """
        from ..model.embedder import Embedder

        src = Path(path)
        if not src.exists():
            raise FinetuneError(f"merged checkpoint not found: {src}")

        embedder = Embedder(model_path=src)
        facts = embedder.facts
        vector = embedder.embed_text(["verification probe"], role="document")
        if vector.shape != (1, 768):
            raise FinetuneError(
                f"merged checkpoint produced {vector.shape}, expected (1, 768)"
            )
        return {"path": str(src), **facts, "probe_shape": list(vector.shape)}