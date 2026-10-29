"""Provisioning logic: source resolution, file lists, and manifest integrity.

The download paths themselves are exercised manually against a real HTTP server
and the hub; these tests cover the decisions that are easy to regress and cheap
to check offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_weights.py"


def load_script():
    """Load scripts/fetch_weights.py as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location("fetch_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fw():
    return load_script()


def args(src=None, source=None, repo=None):
    return SimpleNamespace(src=src, source=source, repo=repo)


# --------------------------------------------------------------- source choice

def test_default_source_is_huggingface(fw, monkeypatch):
    monkeypatch.delenv(fw.SOURCE_ENV, raising=False)
    assert fw.resolve_source(args()) == ("huggingface", fw.DEFAULT_REPO_ID)


def test_env_var_selects_local(fw, monkeypatch):
    monkeypatch.setenv(fw.SOURCE_ENV, "local")
    monkeypatch.setenv(fw.LOCAL_ENV, "/some/checkpoint")
    assert fw.resolve_source(args()) == ("local", "/some/checkpoint")


def test_env_var_url_is_normalised(fw, monkeypatch):
    monkeypatch.setenv(fw.SOURCE_ENV, "https://oss.example.com/models/embeddinggemma-2/v1")
    kind, value = fw.resolve_source(args())
    assert kind == "url"
    assert value.endswith("/"), "base URL must end with a slash for joining"


def test_src_flag_beats_env(fw, monkeypatch):
    monkeypatch.setenv(fw.SOURCE_ENV, "huggingface")
    assert fw.resolve_source(args(src=Path("/explicit")))[0] == "local"


def test_source_flag_beats_env(fw, monkeypatch):
    monkeypatch.setenv(fw.SOURCE_ENV, "local")
    assert fw.resolve_source(args(source="huggingface"))[0] == "huggingface"


def test_unknown_source_is_rejected(fw, monkeypatch):
    monkeypatch.setenv(fw.SOURCE_ENV, "s3-bucket-typo")
    with pytest.raises(ValueError, match="not understood"):
        fw.resolve_source(args())


# ------------------------------------------------------------------ file lists

def test_required_files_cover_multimodal_loading(fw):
    """These three are not obvious and were each found by removing them.

    Dropping sentence_bert_config.json degrades the model to text-only, dropping
    chat_template.jinja breaks the text processor, and dropping
    1_Pooling/config.json fails loading outright. A checkpoint that passes
    verification but cannot encode an image is the worst outcome, so pin it.
    """
    for name in ("sentence_bert_config.json", "chat_template.jinja",
                 "1_Pooling/config.json", "model.safetensors"):
        assert name in fw.REQUIRED_FILES


def test_verify_checkpoint_reports_missing_including_subdirs(fw, tmp_path):
    for name in fw.REQUIRED_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    assert fw.verify_checkpoint(tmp_path) == []

    (tmp_path / "sentence_bert_config.json").unlink()
    assert fw.verify_checkpoint(tmp_path) == ["sentence_bert_config.json"]


# ------------------------------------------------------------------- manifests

def test_manifest_roundtrip_and_tamper_detection(fw, tmp_path):
    for name in fw.REQUIRED_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"contents of {name}")

    manifest = fw.write_manifest(tmp_path)
    entries = fw.read_manifest(manifest)
    assert set(entries) >= set(fw.REQUIRED_FILES)
    assert fw.verify_manifest(tmp_path, entries) == []

    (tmp_path / "config.json").write_text("tampered")
    problems = fw.verify_manifest(tmp_path, entries)
    assert len(problems) == 1
    assert "checksum mismatch" in problems[0]


def test_manifest_detects_missing_file(fw, tmp_path):
    (tmp_path / "config.json").write_text("x")
    entries = {"config.json": fw.sha256_of(tmp_path / "config.json"),
               "modules.json": "0" * 64}
    problems = fw.verify_manifest(tmp_path, entries)
    assert problems == ["missing: modules.json"]


def test_malformed_manifest_line_is_rejected(fw, tmp_path):
    manifest = tmp_path / "weights.sha256"
    manifest.write_text("not-a-valid-line\n")
    with pytest.raises(ValueError, match="malformed manifest"):
        fw.read_manifest(manifest)

def test_huggingface_provision_requests_known_files(fw, tmp_path, monkeypatch):
    """The hub download must be scoped to the files the kit reads.

    Without allow_patterns, snapshot_download pulls every alternative weight
    format in the repo -- several gigabytes of files this package never opens.
    """
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        for name in fw.REQUIRED_FILES:
            target = Path(kwargs["local_dir"]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    produced = fw.provision_huggingface("google/embeddinggemma-2", "v1", tmp_path)

    assert calls["repo_id"] == "google/embeddinggemma-2"
    assert calls["revision"] == "v1"
    assert set(fw.REQUIRED_FILES) <= set(calls["allow_patterns"])
    assert fw.verify_checkpoint(produced) == []
