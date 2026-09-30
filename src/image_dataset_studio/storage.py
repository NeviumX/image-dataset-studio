"""Sidecar compatibility, atomic writes and optimistic external-change checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from .core import Dataset, ImageRecord, Tag, normalize_caption, parse_tags, tag_key
from .translations import msg, tr

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
METADATA = ".image-dataset-studio.json"
READ_MODES = {"tags", "caption", "mixed"}


def hash_bytes(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def render_text(tags: list[Tag], caption: str) -> bytes:
    tag_text = ", ".join(tag.name for tag in tags)
    caption = normalize_caption(caption)
    if tag_text and caption:
        value = tag_text + "\n\n" + caption
    else:
        value = tag_text or caption
    return value.encode("utf-8")


def parse_content(text: str, mode: str, existing: list[Tag] | None = None) -> tuple[list[Tag], str]:
    if mode not in READ_MODES:
        raise ValueError(f"Unknown caption read mode: {mode}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "caption":
        return [], normalize_caption(text)
    if mode == "mixed":
        boundary = re.search(r"\n[ \t]*\n", text)
        if boundary:
            return parse_tags(text[:boundary.start()], existing or []), normalize_caption(
                text[boundary.end():])
    return parse_tags(text, existing or []), ""


def entry(tags: list[Tag], caption: str, source: str | None,
          text_digest: str | None) -> dict:
    return {"tags": [asdict(tag) for tag in tags], "caption": caption,
            "caption_source": source, "text_digest": text_digest}


def entry_tags(value: dict | list | None) -> list[Tag]:
    return [Tag(**tag) for tag in (value.get("tags", []) if isinstance(value, dict) else value or [])]


def enrich_tags(tags: list[Tag], known_tags: list[Tag]) -> list[Tag]:
    known = {tag_key(tag.name): tag for tag in known_tags}
    return [Tag(tag.name, known[tag_key(tag.name)].category,
                known[tag_key(tag.name)].score, known[tag_key(tag.name)].source)
            if tag_key(tag.name) in known else tag for tag in tags]


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_dataset(root: Path, recursive: bool = True, read_mode: str | None = None) -> Dataset:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(tr(msg.error_FolderMissing))
    if read_mode is not None and read_mode not in READ_MODES:
        raise ValueError(f"Unknown caption read mode: {read_mode}")
    metadata_path = root / METADATA
    metadata_bytes = metadata_path.read_bytes() if metadata_path.exists() else None
    meta = json.loads(metadata_bytes) if metadata_bytes else {"version": 1, "images": {}}
    if meta.get("version") not in {1, 2} or not isinstance(meta.get("images"), dict):
        raise ValueError(tr(msg.error_MetadataFormat))
    pending = meta.get("pending", {})
    if not isinstance(pending, dict):
        raise ValueError(tr(msg.error_MetadataFormat))
    paths = sorted((root.rglob("*") if recursive else root.glob("*")), key=lambda p: str(p).casefold())
    records, seen, warnings = [], {}, []
    mixed_without_boundary = 0
    for path in paths:
        if path.suffix.lower() not in EXTENSIONS or not path.is_file() or path.is_symlink():
            continue
        sidecar = path.with_suffix(".txt")
        key = str(sidecar).casefold()
        if key in seen:
            raise ValueError(tr(msg.error_SidecarConflict, first=seen[key].name, second=path.name))
        seen[key] = path
        raw = sidecar.read_bytes() if sidecar.exists() else None
        try:
            text = raw.decode("utf-8-sig") if raw is not None else ""
        except UnicodeDecodeError as exc:
            raise ValueError(tr(msg.error_SidecarDecode, path=sidecar)) from exc
        relative = path.relative_to(root).as_posix()
        actual_digest = hash_bytes(raw)
        if read_mode == "mixed" and raw is not None and not re.search(
                r"\n[ \t]*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
            mixed_without_boundary += 1
        saved = meta["images"].get(relative)
        known_tags = entry_tags(saved)
        base = saved if isinstance(saved, dict) else None
        if relative in pending:
            transaction = pending[relative]
            before, after = transaction["before"], transaction["after"]
            if actual_digest == after["text_digest"]:
                base = after
            elif actual_digest == before["text_digest"]:
                base = before
            else:
                raise ValueError(tr(msg.error_PendingConflict, filename=path.name))
            known_tags = entry_tags(after)
        if read_mode is not None:
            tags, caption = parse_content(text, read_mode, known_tags)
            tags = enrich_tags(tags, known_tags)
            current_source = (base.get("caption_source") if base and caption == base.get("caption")
                              else None)
        elif relative in pending:
            tags, caption = entry_tags(base), base["caption"]
            current_source = base.get("caption_source")
        elif base and actual_digest == base.get("text_digest"):
            tags, caption = entry_tags(base), base["caption"]
            current_source = base.get("caption_source")
        elif base:
            previous_mode = "mixed" if base["tags"] and base["caption"] else (
                "caption" if base["caption"] else "tags")
            if raw is None or (previous_mode == "mixed" and not re.search(
                    r"\n[ \t]*\n", text.replace("\r\n", "\n"))) or (
                    previous_mode == "tags" and re.search(
                        r"\n[ \t]*\n", text.replace("\r\n", "\n"))):
                raise ValueError(tr(msg.error_CaptionAmbiguous, filename=path.name))
            tags, caption = parse_content(text, previous_mode, known_tags)
            tags = enrich_tags(tags, known_tags)
            current_source = base.get("caption_source") if caption == base["caption"] else None
        else:
            tags, caption = parse_content(text, "tags", known_tags)
            tags = enrich_tags(tags, known_tags)
            current_source = None
        if base:
            saved_tags = entry_tags(base)
            saved_caption = base["caption"]
            saved_source = base.get("caption_source")
            metadata_dirty = relative in pending or actual_digest != base.get("text_digest")
            if metadata_dirty and relative not in pending:
                # A changed sidecar is the transaction's true on-disk "before" state.
                saved_tags, saved_caption, saved_source = list(tags), caption, current_source
        else:
            saved_tags = enrich_tags(parse_tags(text), known_tags)
            saved_caption = ""
            saved_source = None
            metadata_dirty = False
        records.append(ImageRecord(path, tags, saved_tags, actual_digest, caption,
                                   saved_caption, current_source, saved_source,
                                   metadata_dirty))
    if set(pending) - {r.path.relative_to(root).as_posix() for r in records}:
        raise ValueError(tr(msg.error_PendingScope))
    dataset = Dataset(root, records)
    dataset.metadata_digest = hash_bytes(metadata_bytes)
    dataset.warnings = warnings
    dataset.mixed_without_boundary = mixed_without_boundary
    return dataset


def save_dataset(dataset: Dataset) -> int:
    dirty = [r for r in dataset.records if r.dirty]
    if not dirty:
        return 0
    metadata_path = dataset.root / METADATA
    if digest(metadata_path) != dataset.metadata_digest:
        raise ValueError(tr(msg.error_MetadataChanged))
    for record in dirty:
        if digest(record.path.with_suffix(".txt")) != record.text_digest:
            raise ValueError(tr(msg.error_SidecarChanged, filename=record.path.name))
    # Keep metadata of subfolders omitted by a non-recursive load.
    old_meta = json.loads(metadata_path.read_text("utf-8")) if metadata_path.exists() else {"images": {}}
    images = dict(old_meta["images"])
    pending = {}
    data_by_path = {}
    for record in dataset.records:
        relative = record.path.relative_to(dataset.root).as_posix()
        data = render_text(record.tags, record.caption) if record.dirty else None
        new_digest = hash_bytes(data) if data is not None else record.text_digest
        images[relative] = entry(record.tags, record.caption, record.caption_source, new_digest)
        if record.dirty:
            pending[relative] = {
                "before": entry(record.saved_tags, record.saved_caption,
                                record.saved_caption_source, record.text_digest),
                "after": images[relative],
            }
            data_by_path[record.path] = data
    staged = json.dumps({"version": 2, "images": old_meta["images"], "pending": pending},
                        ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write(metadata_path, staged)
    dataset.metadata_digest = hash_bytes(staged)
    for record in dirty:
        data = data_by_path[record.path]
        if hash_bytes(data) != record.text_digest:
            atomic_write(record.path.with_suffix(".txt"), data)
        record.text_digest = hash_bytes(data)
    metadata = json.dumps({"version": 2, "images": images},
                          ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write(metadata_path, metadata)
    dataset.metadata_digest = hash_bytes(metadata)
    for record in dirty:
        record.saved_tags = list(record.tags)
        record.saved_caption = record.caption
        record.saved_caption_source = record.caption_source
        record.metadata_dirty = False
    return len(dirty)
