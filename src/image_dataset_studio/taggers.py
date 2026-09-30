"""Lazy-loaded model adapters; GUI startup does not load ML libraries."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .core import Tag, tag_key
from .models import ModelSpec
from .translations import msg, tr


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGBA")


def wd_input(image: Image.Image, size: int) -> np.ndarray:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    rgb = canvas.convert("RGB")
    side = max(rgb.size)
    square = Image.new("RGB", (side, side), "white")
    square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    square = square.resize((size, size), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(square, dtype=np.float32)[None, :, :, ::-1])


def camie_input(image: Image.Image, size: int) -> np.ndarray:
    rgb = image.convert("RGB")
    ratio = size / max(rgb.size)
    resized = rgb.resize((max(1, int(rgb.width * ratio)), max(1, int(rgb.height * ratio))),
                         Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (124, 116, 104))
    square.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    values = np.asarray(square, dtype=np.float32) / np.float32(255)
    values = (values - np.array([.485, .456, .406], dtype=np.float32)) / np.array(
        [.229, .224, .225], dtype=np.float32
    )
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def decode_scores(names, categories, scores, thresholds, source) -> list[Tag]:
    if len(names) != len(scores):
        raise ValueError(tr(msg.error_OutputCount))
    if not np.isfinite(scores).all():
        raise ValueError(tr(msg.error_OutputNonfinite))
    return sorted([
        Tag(name, category, float(score), source)
        for name, category, score in zip(names, categories, scores, strict=True)
        if float(score) >= thresholds.get(category, .5)
    ], key=lambda t: -(t.score or 0))


def pixai_vocabulary(config: dict) -> dict[str, str]:
    tags = config["tags"]
    result, start = {}, 0
    for category, count in config["tags_split"]:
        for name in tags[start:start + count]:
            result[tag_key(name)] = category
        start += count
    if start != len(tags):
        raise ValueError(tr(msg.error_PixaiTagsSplit))
    return result


class Tagger:
    def __init__(self, spec: ModelSpec, device: str = "auto", offline: bool = False,
                 allow_code: bool = False, report=None):
        self.spec = spec
        self.device = device
        self.offline = offline
        self.allow_code = allow_code
        self.vocabulary: dict[str, str] = {}
        self.runtime = ""
        self.report = report or (lambda message: None)

    def download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        self.report(tr(msg.log_CheckCache if self.offline else msg.log_Download,
                       filename=filename))
        path = hf_hub_download(self.spec.repo_id, filename, revision=self.spec.revision,
                               local_files_only=self.offline)
        self.report(tr(msg.log_FileReady, filename=filename))
        return path

    def load_vocabulary(self) -> None:
        path = Path(self.download(self.spec.labels_file))
        self.report(tr(msg.log_LoadDictionary))
        # A Hub download resolves a branch to snapshots/<commit>. Pin the remaining
        # files to that same commit so a model update cannot mix labels and weights.
        if re.fullmatch(r"[0-9a-f]{40}", path.parent.name):
            self.spec = replace(self.spec, revision=path.parent.name)
        if self.spec.backend == "pixai":
            self.vocabulary = pixai_vocabulary(json.loads(path.read_text("utf-8")))
        elif self.spec.backend == "wd":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            mapping = {"0": "general", "4": "character", "9": "rating",
                       "1": "artist", "3": "copyright", "5": "meta"}
            self.names = [r["name"] for r in rows]
            self.categories = [mapping.get(r["category"], "unknown") for r in rows]
            self.vocabulary = {tag_key(n): c for n, c in zip(self.names, self.categories)}
        else:
            metadata = json.loads(path.read_text("utf-8"))
            mapping = metadata["dataset_info"]["tag_mapping"]
            indexed = mapping["idx_to_tag"]
            self.names = [indexed[str(i)] for i in range(len(indexed))]
            self.categories = [mapping["tag_to_category"].get(n, "unknown") for n in self.names]
            self.vocabulary = {tag_key(n): c for n, c in zip(self.names, self.categories)}
            self.size = int(metadata["model_info"]["img_size"])
        self.report(tr(msg.log_DictionaryLoaded, count=len(self.vocabulary)))

    def load(self) -> None:
        self.load_vocabulary()
        if self.spec.backend == "pixai":
            if not self.allow_code:
                raise ValueError(tr(msg.error_PixaiCode))
            import torch
            from huggingface_hub import snapshot_download
            from transformers import pipeline

            if self.device == "cuda" and not torch.cuda.is_available():
                raise ValueError(tr(msg.error_CudaTorch))
            device = 0 if self.device != "cpu" and torch.cuda.is_available() else -1
            self.report(tr(msg.log_PixaiDownload))
            folder = snapshot_download(self.spec.repo_id, revision=self.spec.revision,
                                       allow_patterns=["*.json", "*.py", "*.safetensors"],
                                       local_files_only=self.offline)
            self.report(tr(msg.log_ModelDownloadDone))
            self.pipeline = pipeline(model=folder, image_processor=folder,
                                     trust_remote_code=True, device=device)
            self.runtime = "PyTorch CUDA" if device == 0 else "PyTorch CPU"
        else:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if self.device == "cuda" and "CUDAExecutionProvider" not in available:
                raise ValueError(tr(msg.error_CudaOnnx))
            providers = ["CPUExecutionProvider"]
            if self.device != "cpu" and "CUDAExecutionProvider" in available:
                try:
                    ort.preload_dlls()
                except Exception as exc:
                    if self.device == "cuda":
                        raise RuntimeError(tr(msg.log_CudaDllFailure, error=exc)) from exc
                    self.report(tr(msg.log_CudaFallback, error=exc))
                else:
                    providers.insert(0, "CUDAExecutionProvider")
            self.report(tr(msg.log_OnnxLoading))
            model_path = self.download(self.spec.model_file)
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
            except Exception as exc:
                if self.device != "auto" or "CUDAExecutionProvider" not in providers:
                    raise
                self.report(tr(msg.log_CudaModelFallback, error=exc))
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            if self.device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError(tr(msg.error_CudaProvider))
            if (self.device == "auto" and "CUDAExecutionProvider" in providers and
                    "CUDAExecutionProvider" not in self.session.get_providers()):
                self.report(tr(msg.log_CudaProviderFallback))
            shape = self.session.get_inputs()[0].shape
            if self.spec.backend == "wd":
                if len(shape) != 4 or shape[3] != 3 or not isinstance(shape[1], int) or shape[1] != shape[2]:
                    raise ValueError(tr(msg.error_WdInput))
                self.size = shape[1]
            elif len(shape) != 4 or shape[1] != 3 or shape[2:] != [self.size, self.size]:
                raise ValueError(tr(msg.error_CamieInput))
            self.runtime = self.session.get_providers()[0]
        self.report(tr(msg.log_ModelLoaded, runtime=self.runtime))

    def predict(self, path: Path, thresholds: dict[str, float]) -> list[Tag]:
        image = read_image(path)
        source = f"{self.spec.repo_id}@{self.spec.revision}"
        if self.spec.backend == "pixai":
            output = self.pipeline(image, threshold=thresholds)["results"]
            if not isinstance(output, dict):
                raise ValueError(tr(msg.error_PixaiResponse))
            return sorted([Tag(name, category, float(score), source)
                           for category, values in output.items() for name, score in values.items()],
                          key=lambda t: -(t.score or 0))
        data = wd_input(image, self.size) if self.spec.backend == "wd" else camie_input(image, self.size)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: data})
        scores = outputs[0][0]
        if self.spec.backend == "camie":
            logits = (outputs[1] if len(outputs) > 1 else outputs[0])[0]
            scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        return decode_scores(self.names, self.categories, scores, thresholds, source)
