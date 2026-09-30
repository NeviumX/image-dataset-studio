"""GUI-independent tag editing and bounded undo history."""

from __future__ import annotations

import fnmatch
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

from .translations import msg, tr

CATEGORIES = {
    "general": msg.category_general, "character": msg.category_character,
    "copyright": msg.category_copyright, "style": msg.category_style,
    "artist": msg.category_artist, "meta": msg.category_meta,
    "rating": msg.category_rating, "year": msg.category_year,
    "unknown": msg.category_unknown,
}


def tag_key(name: str) -> str:
    return name.strip().replace(r"\(", "(").replace(r"\)", ")").replace("_", " ").casefold()


@dataclass(frozen=True)
class Tag:
    name: str
    category: str = "unknown"
    score: float | None = None
    source: str | None = None


def unique_tags(tags: Iterable[Tag]) -> list[Tag]:
    result: dict[str, Tag] = {}
    for tag in tags:
        name = tag.name.strip()
        if name and tag_key(name) not in result:
            result[tag_key(name)] = replace(tag, name=name)
    return list(result.values())


def parse_tags(text: str, existing: Iterable[Tag] = ()) -> list[Tag]:
    known = {tag_key(t.name): t for t in existing}
    return unique_tags(
        replace(known.get(tag_key(n), Tag(n.strip())), name=n.strip())
        for n in re.split(r"[,\n\r]+", text) if n.strip()
    )


def normalize_caption(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def insert_tags(old: list[Tag], incoming: list[Tag], position: str = "after",
                anchor: str | None = None, skip_existing: bool = True,
                custom_index: int = 0) -> list[Tag]:
    """Insert tags relative to an anchor while preserving the unique-tag invariant."""
    incoming = unique_tags(incoming)
    known = {tag_key(tag.name): tag for tag in old}
    keys = {tag_key(tag.name) for tag in incoming}
    additions = [known.get(tag_key(tag.name), tag) for tag in incoming
                 if not skip_existing or tag_key(tag.name) not in known]
    base = old if skip_existing else [tag for tag in old if tag_key(tag.name) not in keys]
    anchor_index = next((i for i, tag in enumerate(base)
                         if anchor and tag_key(tag.name) == tag_key(anchor)), -1)
    if position == "start":
        index = 0
    elif position == "end":
        index = len(base)
    elif position == "middle":
        index = len(base) // 2
    elif position == "custom":
        index = max(0, min(custom_index, len(base)))
    elif position == "before":
        index = anchor_index if anchor_index >= 0 else 0
    else:
        index = anchor_index + 1 if anchor_index >= 0 else len(base)
    return base[:index] + additions + base[index:]


@dataclass
class ImageRecord:
    path: Path
    tags: list[Tag] = field(default_factory=list)
    saved_tags: list[Tag] = field(default_factory=list)
    text_digest: str | None = None
    caption: str = ""
    saved_caption: str = ""
    caption_source: str | None = None
    saved_caption_source: str | None = None
    metadata_dirty: bool = False

    @property
    def dirty(self) -> bool:
        return (self.tags != self.saved_tags or self.caption != self.saved_caption or
                self.caption_source != self.saved_caption_source or self.metadata_dirty)

    def state(self) -> RecordState:
        return RecordState(tuple(self.tags), self.caption, self.caption_source)


@dataclass(frozen=True)
class RecordState:
    tags: tuple[Tag, ...]
    caption: str = ""
    caption_source: str | None = None


@dataclass
class Edit:
    label: str
    before: dict[Path, RecordState]
    after: dict[Path, RecordState]
    merge_key: str | None = None


class Dataset:
    def __init__(self, root: Path, records: list[ImageRecord]):
        self.root = root
        self.records = records
        self.by_path = {r.path: r for r in records}
        self.undo_stack: list[Edit] = []
        self.redo_stack: list[Edit] = []
        self.metadata_digest: str | None = None
        self.warnings: list[str] = []
        self.mixed_without_boundary = 0

    @property
    def dirty(self) -> bool:
        return any(r.dirty for r in self.records)

    def apply(self, label: str, updates: dict[Path, list[Tag]]) -> int:
        return self.apply_content(label, {
            p: RecordState(tuple(tags), self.by_path[p].caption, self.by_path[p].caption_source)
            for p, tags in updates.items()
        })

    def apply_captions(self, label: str, updates: dict[Path, tuple[str, str | None]],
                       merge_key: str | None = None) -> int:
        return self.apply_content(label, {
            p: RecordState(tuple(self.by_path[p].tags),
                           normalize_caption(caption), source)
            for p, (caption, source) in updates.items()
        }, merge_key)

    def apply_content(self, label: str, updates: dict[Path, RecordState],
                      merge_key: str | None = None) -> int:
        # Resolve all paths and validate all states before changing any record.
        before = {p: self.by_path[p].state() for p in updates}
        after = {p: RecordState(tuple(unique_tags(value.tags)), value.caption, value.caption_source)
                 for p, value in updates.items()}
        after = {p: value for p, value in after.items() if value != before[p]}
        if not after:
            return 0
        before = {p: before[p] for p in after}
        self._restore(after)
        if (merge_key and self.undo_stack and self.undo_stack[-1].merge_key == merge_key and
                self.undo_stack[-1].after == before):
            self.undo_stack[-1].after = after
        else:
            self.undo_stack.append(Edit(label, before, after, merge_key))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()
        return len(after)

    def transform(self, label: str, records: list[ImageRecord], fn: Callable) -> int:
        return self.apply(label, {r.path: fn(list(r.tags)) for r in records})

    def _restore(self, values: dict[Path, RecordState]) -> None:
        for path, value in values.items():
            record = self.by_path[path]
            record.tags = list(value.tags)
            record.caption = value.caption
            record.caption_source = value.caption_source

    def undo(self) -> None:
        if self.undo_stack:
            edit = self.undo_stack.pop()
            self._restore(edit.before)
            self.redo_stack.append(edit)

    def redo(self) -> None:
        if self.redo_stack:
            edit = self.redo_stack.pop()
            self._restore(edit.after)
            self.undo_stack.append(edit)


def tag_counts(records: Iterable[ImageRecord]) -> Counter:
    return Counter(t.name for r in records for t in r.tags)


def remove_category(tags: list[Tag], category: str,
                    vocabulary: dict[str, str] | None = None) -> list[Tag]:
    vocabulary = vocabulary or {}
    return [t for t in tags if t.category != category and not (
        t.category == "unknown" and vocabulary.get(tag_key(t.name)) == category
    )]


def remove_characters(tags: list[Tag], vocabulary: dict[str, str] | None = None) -> list[Tag]:
    return remove_category(tags, "character", vocabulary)


def rename_tags(tags: list[Tag], old: str, new: str, regex: bool = False) -> list[Tag]:
    pattern = re.compile(old) if regex else None
    result = []
    for tag in tags:
        name = pattern.sub(new, tag.name) if pattern else (
            new if tag_key(tag.name) == tag_key(old) else tag.name
        )
        # Renamed user tags must not inherit a stale character classification.
        result.append(tag if name == tag.name else Tag(name))
    return unique_tags(result)


def compile_filter(query: str) -> Callable[[ImageRecord], bool]:
    """Whitespace = AND, | = OR, - prefix = NOT; tag/name/path/category/count filters."""
    if not query.strip():
        return lambda record: True
    lexer = shlex.shlex(query, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    groups: list[list[Callable]] = [[]]
    for token in lexer:
        if token == "|":
            if not groups[-1]:
                raise ValueError(tr(msg.error_SearchOrAround))
            groups.append([])
            continue
        negate = token.startswith("-")
        token = token[1:] if negate else token
        prefix, sep, value = token.partition(":")
        if not sep:
            value, prefix = prefix, "text"
        if prefix == "untagged":
            pred = lambda r: not r.tags
        elif prefix == "tags":
            match = re.fullmatch(r"(>=|<=|!=|=|>|<)(\d+)", value)
            if not match:
                raise ValueError(tr(msg.error_SearchCount))
            op, count = match[1], int(match[2])
            funcs = {">=": lambda n, c: n >= c, "<=": lambda n, c: n <= c,
                     "!=": lambda n, c: n != c, "=": lambda n, c: n == c,
                     ">": lambda n, c: n > c, "<": lambda n, c: n < c}
            pred = lambda r, f=funcs[op], c=count: f(len(r.tags), c)
        elif prefix == "tag":
            pred = lambda r, v=tag_key(value): any(
                fnmatch.fnmatchcase(tag_key(t.name), v) for t in r.tags
            )
        elif prefix == "category":
            pred = lambda r, v=value: any(t.category == v for t in r.tags)
        elif prefix in {"name", "path", "text"}:
            def pred(r, v=value.casefold(), p=prefix):
                haystack = r.path.name if p == "name" else str(r.path)
                if p == "text":
                    haystack += " " + " ".join(t.name for t in r.tags) + " " + r.caption
                return v in haystack.casefold()
        else:
            raise ValueError(tr(msg.error_SearchUnknown, term=prefix))
        groups[-1].append((lambda r, f=pred: not f(r)) if negate else pred)
    if not groups[-1]:
        raise ValueError(tr(msg.error_SearchOrAfter))
    return lambda r: any(all(f(r) for f in group) for group in groups)
