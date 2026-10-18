"""Fine-tuning interfaces (CPU-side scaffolding; training requires a GPU)."""

from .trainer import (
    SUPPORTED_TARGETS,
    FinetuneTrainer,
    LoRAConfig,
    PairDataset,
    TrainingPair,
)

__all__ = [
    "LoRAConfig",
    "PairDataset",
    "TrainingPair",
    "FinetuneTrainer",
    "SUPPORTED_TARGETS",
]
