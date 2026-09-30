"""Persisted model registry. Only known input/output contracts can be registered."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .storage import atomic_write
from .translations import msg, tr

PIXAI = "pixai-labs/pixai-tagger-v1.0"
FLORENCE = "microsoft/Florence-2-large-ft"
QWEN_4B = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_8B = "Qwen/Qwen3-VL-8B-Instruct"
JOYCAPTION = "fancyfeast/llama-joycaption-beta-one-hf-llava"
CAPTION_BACKENDS = {"florence", "qwen3_vl", "joycaption"}
PIXAI_THRESHOLDS = {"general": .17, "character": .27, "style": .15,
                    "copyright": .24, "meta": .17, "rating": .41}


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    backend: str
    revision: str = "main"
    model_file: str = "model.onnx"
    labels_file: str = "selected_tags.csv"

    @property
    def title(self) -> str:
        return self.repo_id

    @property
    def kind(self) -> str:
        return "caption" if self.backend in CAPTION_BACKENDS else "tags"

    @property
    def thresholds(self) -> dict[str, float]:
        if self.backend == "pixai":
            return dict(PIXAI_THRESHOLDS)
        if self.backend == "camie":
            return {"general": .5, "character": .5, "copyright": .5,
                    "artist": .5, "meta": .5, "rating": .5, "year": .5}
        return {"general": .35, "character": .85, "rating": .5}


def presets() -> list[ModelSpec]:
    wd = ["wd-swinv2-tagger-v3", "wd-convnext-tagger-v3", "wd-vit-tagger-v3",
          "wd-vit-large-tagger-v3", "wd-eva02-large-tagger-v3",
          "wd-v1-4-moat-tagger-v2", "wd-v1-4-swinv2-tagger-v2",
          "wd-v1-4-convnext-tagger-v2", "wd-v1-4-convnextv2-tagger-v2",
          "wd-v1-4-vit-tagger-v2"]
    return [ModelSpec(PIXAI, "pixai", model_file="model.safetensors", labels_file="config.json"),
            *[ModelSpec(f"SmilingWolf/{name}", "wd") for name in wd],
            ModelSpec("Camais03/camie-tagger-v2", "camie", model_file="camie-tagger-v2.onnx",
                      labels_file="camie-tagger-v2-metadata.json"),
            ModelSpec("deepghs/idolsankaku-swinv2-tagger-v1", "wd"),
            ModelSpec("deepghs/idolsankaku-eva02-large-tagger-v1", "wd"),
            ModelSpec(FLORENCE, "florence", model_file="", labels_file=""),
            ModelSpec(QWEN_4B, "qwen3_vl", model_file="", labels_file=""),
            ModelSpec(QWEN_8B, "qwen3_vl", model_file="", labels_file=""),
            ModelSpec(JOYCAPTION, "joycaption", model_file="", labels_file="")]


def parse_hf_url(value: str) -> tuple[str, str]:
    value = value.strip().rstrip("/")
    revision = "main"
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError(tr(msg.error_ModelUrl))
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            revision = unquote(parts[3])
        elif len(parts) != 2:
            raise ValueError(tr(msg.error_ModelUrlPath))
        value = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(tr(msg.error_ModelId))
    if ".." in value or "--" in value:
        raise ValueError(tr(msg.error_ModelIdInvalid))
    return value, revision


def detect_model(repo_id: str, revision: str, files: set[str]) -> ModelSpec:
    caption_presets = {spec.repo_id: spec for spec in presets() if spec.kind == "caption"}
    if repo_id in caption_presets and "config.json" in files and (
            "model.safetensors" in files or "model.safetensors.index.json" in files):
        spec = caption_presets[repo_id]
        return ModelSpec(repo_id, spec.backend, revision, "", "")
    if {"model.onnx", "selected_tags.csv"} <= files:
        return ModelSpec(repo_id, "wd", revision)
    if {"camie-tagger-v2.onnx", "camie-tagger-v2-metadata.json"} <= files:
        return ModelSpec(repo_id, "camie", revision, "camie-tagger-v2.onnx",
                         "camie-tagger-v2-metadata.json")
    if {"tagger_pipeline.py", "config.json", "preprocessor_config.json",
        "model.safetensors"} <= files:
        return ModelSpec(repo_id, "pixai", revision, "model.safetensors", "config.json")
    raise ValueError(tr(msg.error_ModelUnsupported))


def inspect_model(value: str) -> ModelSpec:
    from huggingface_hub import HfApi

    repo, revision = parse_hf_url(value)
    info = HfApi().model_info(repo, revision=revision)
    return detect_model(repo, info.sha, {f.rfilename for f in info.siblings})


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.models = presets()
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            for entry in data:
                spec = ModelSpec(**entry)
                parse_hf_url(spec.repo_id)
                if spec.backend not in {"pixai", "wd", "camie", *CAPTION_BACKENDS}:
                    raise ValueError(tr(msg.error_ModelRegistry))
                self._put(spec)

    def _put(self, spec: ModelSpec) -> None:
        for index, model in enumerate(self.models):
            if model.repo_id == spec.repo_id:
                self.models[index] = spec
                return
        self.models.append(spec)

    def add(self, spec: ModelSpec) -> None:
        proposed = [m for m in self.models if m.repo_id != spec.repo_id] + [spec]
        atomic_write(self.path, json.dumps([asdict(m) for m in proposed], indent=2).encode())
        self._put(spec)
