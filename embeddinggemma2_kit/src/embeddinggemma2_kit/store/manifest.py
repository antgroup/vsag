"""Corpus manifests: a single description of a searchable collection.

One manifest drives every retrieval direction. A text->image search and an
image->text search over the same collection use the same manifest; which side
is the query and which is the corpus is decided by the search call, not by the
file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..constants import SUPPORTED_MODALITIES
from ..errors import ConfigError


@dataclass
class CorpusItem:
    """One searchable entry.

    ``id`` is caller-defined and must be unique. Exactly one primary modality
    key should be present; ``text`` may additionally accompany a media item
    (for example a caption for an image, or the modality marker for
    video+audio).
    """

    id: str
    text: str | None = None
    image: str | None = None
    audio: str | None = None
    video: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[str]:
        return [m for m in SUPPORTED_MODALITIES if getattr(self, m) is not None]

    @property
    def primary_modality(self) -> str:
        """The modality used to encode this item (text only if nothing else)."""
        for m in ("image", "video", "audio", "text"):
            if getattr(self, m) is not None:
                return m
        raise ConfigError(f"item {self.id!r} has no usable modality")

    def to_model_input(self) -> dict[str, Any]:
        """Build the encoder's mixed-modality dict for this item."""
        payload: dict[str, Any] = {}
        for m in ("image", "video", "audio", "text"):
            value = getattr(self, m)
            if value is not None:
                payload[m] = value
        return payload


def _coerce_item(raw: dict[str, Any], line_no: int) -> CorpusItem:
    if not isinstance(raw, dict):
        raise ConfigError(f"manifest line {line_no}: expected a JSON object")
    if "id" not in raw:
        raise ConfigError(f"manifest line {line_no}: missing required 'id'")

    known = {"id", *SUPPORTED_MODALITIES, "metadata"}
    item = CorpusItem(
        id=str(raw["id"]),
        text=raw.get("text"),
        image=raw.get("image"),
        audio=raw.get("audio"),
        video=raw.get("video"),
        metadata=raw.get("metadata") or {},
    )
    unknown = set(raw) - known
    if unknown:
        # Surface typos like "img" instead of silently ignoring the field.
        raise ConfigError(
            f"manifest line {line_no}: unknown field(s) {sorted(unknown)}; "
            f"expected any of {sorted(known)}"
        )
    if not item.modalities:
        raise ConfigError(
            f"manifest line {line_no}: item {item.id!r} has none of "
            f"{list(SUPPORTED_MODALITIES)}"
        )
    return item


class Manifest:
    """An ordered, validated collection of :class:`CorpusItem`."""

    def __init__(self, items: Iterable[CorpusItem]) -> None:
        self._items = list(items)
        seen: set[str] = set()
        for item in self._items:
            if item.id in seen:
                raise ConfigError(f"duplicate id in manifest: {item.id!r}")
            seen.add(item.id)
            # Fail here rather than when encoding: an item with no content would
            # otherwise reach the encoder and produce a confusing error, or
            # worse, a valid-looking embedding of nothing.
            if not item.modalities:
                raise ConfigError(
                    f"item {item.id!r} has no content; expected at least one of "
                    f"{list(SUPPORTED_MODALITIES)}"
                )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[CorpusItem]:
        return iter(self._items)

    def __getitem__(self, index: int) -> CorpusItem:
        return self._items[index]

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self._items]

    def by_id(self, item_id: str) -> CorpusItem:
        for item in self._items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def filter_modality(self, modality: str) -> "Manifest":
        """Subset of items whose primary modality is ``modality``."""
        return Manifest([i for i in self._items if i.primary_modality == modality])

    # ------------------------------------------------------------ loading

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Manifest":
        """Read one JSON object per line."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"manifest not found: {p}")
        items = []
        for line_no, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"manifest line {line_no}: invalid JSON: {exc}") from exc
            items.append(_coerce_item(raw, line_no))
        if not items:
            raise ConfigError(f"manifest is empty: {p}")
        return cls(items)

    @classmethod
    def from_json(cls, path: str | Path) -> "Manifest":
        """Read a JSON array of objects."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"manifest not found: {p}")
        raw = json.loads(p.read_text())
        if not isinstance(raw, list):
            raise ConfigError(f"manifest {p} must contain a JSON array")
        return cls(_coerce_item(item, i + 1) for i, item in enumerate(raw))

    @classmethod
    def from_flickr30k(cls, path: str | Path, images_dir: str | Path) -> "Manifest":
        """Build from the Flickr30k manifest used by the baseline experiments.

        That file is a JSON array of ``{id, image, captions: [...]}``. Each
        caption becomes its own corpus item so image->text search can return a
        specific caption rather than an image's whole caption set.
        """
        p = Path(path)
        raw = json.loads(p.read_text())
        if not isinstance(raw, list):
            raise ConfigError(f"{p} must contain a JSON array")
        image_root = Path(images_dir)

        items: list[CorpusItem] = []
        for entry in raw:
            image_id = str(entry["id"])
            image_path = entry.get("image")
            if not image_path:
                candidate = image_root / image_id
                image_path = str(candidate)
            items.append(
                CorpusItem(id=image_id, image=str(image_path), metadata={"kind": "image"})
            )
            for j, caption in enumerate(entry.get("captions", [])):
                items.append(
                    CorpusItem(
                        id=f"{image_id}::caption:{j}",
                        text=caption,
                        metadata={"kind": "caption", "image_id": image_id, "index": j},
                    )
                )
        return cls(items)

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as handle:
            for item in self._items:
                record: dict[str, Any] = {"id": item.id}
                for m in SUPPORTED_MODALITIES:
                    value = getattr(item, m)
                    if value is not None:
                        record[m] = value
                if item.metadata:
                    record["metadata"] = item.metadata
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")