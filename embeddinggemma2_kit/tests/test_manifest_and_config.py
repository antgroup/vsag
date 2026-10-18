"""Manifest, prompt-role and fine-tuning config validation."""

from __future__ import annotations

import json
import pytest

from embeddinggemma2_kit.errors import ConfigError, FinetuneError, MediaDecodeError
from embeddinggemma2_kit.finetune import LoRAConfig, PairDataset, TrainingPair
from embeddinggemma2_kit.model.prompts import ROLE_DOCUMENT, ROLE_QUERY, prompt_for_role
from embeddinggemma2_kit.store.manifest import CorpusItem, Manifest


def test_role_mapping_is_distinct():
    """Queries and documents must not share a prompt prefix."""
    assert prompt_for_role(ROLE_QUERY) != prompt_for_role(ROLE_DOCUMENT)


def test_unknown_role_rejected():
    with pytest.raises(ConfigError):
        prompt_for_role("querry")


def test_manifest_rejects_duplicate_ids():
    with pytest.raises(ConfigError, match="duplicate"):
        Manifest([CorpusItem(id="a", text="x"), CorpusItem(id="a", text="y")])


def test_manifest_rejects_item_without_modality():
    with pytest.raises(ConfigError):
        Manifest([CorpusItem(id="a")])


def test_manifest_rejects_unknown_field(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"id": "a", "img": "x.jpg"}) + "\n")
    with pytest.raises(ConfigError, match="unknown field"):
        Manifest.from_jsonl(path)


def test_manifest_primary_modality_prefers_media():
    """A caption-bearing image is an image item, not a text item."""
    item = CorpusItem(id="a", image="x.jpg", text="a caption")
    assert item.primary_modality == "image"


def test_lora_config_rejects_float16():
    """The model does not support FP16 activations."""
    with pytest.raises(ConfigError, match="float16"):
        LoRAConfig(dtype="float16")


def test_lora_config_rejects_unknown_target():
    with pytest.raises(ConfigError, match="target"):
        LoRAConfig(target="text_tower")


def test_pair_dataset_rejects_empty_caption():
    with pytest.raises(FinetuneError, match="empty caption"):
        PairDataset([TrainingPair(media="/tmp/x.wav", caption="   ")], check_files=False)


def test_pair_dataset_checks_files():
    with pytest.raises(FinetuneError, match="not found"):
        PairDataset([TrainingPair(media="/nonexistent/a.wav", caption="hi")])


def test_finetune_requires_gpu():
    """fit must refuse on a CPU-only host with an actionable message."""
    from embeddinggemma2_kit.finetune import FinetuneTrainer

    trainer = FinetuneTrainer()
    dataset = PairDataset(
        [TrainingPair(media="/tmp/a.wav", caption="hello")], check_files=False
    )
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        pytest.skip("CUDA present; the refusal path is not exercised")

    with pytest.raises(FinetuneError, match="CUDA"):
        trainer.fit(dataset)
