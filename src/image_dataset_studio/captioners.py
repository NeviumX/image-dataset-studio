"""Lazy-loaded natural-language image caption adapters."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .models import ModelSpec
from .translations import msg, tr

FLORENCE_TASKS = {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"}


class Captioner:
    def __init__(self, spec: ModelSpec, device: str = "auto", offline: bool = False,
                 allow_code: bool = False, report=None):
        self.spec = spec
        self.requested_device = device
        self.offline = offline
        self.allow_code = allow_code
        self.report = report or (lambda _message: None)
        self.runtime = ""
        self.source = ""

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor

        if self.spec.backend == "florence" and not self.allow_code:
            raise ValueError(tr(msg.error_CaptionCode))
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError(tr(msg.error_CudaTorch))
        self.device = "cuda" if self.requested_device != "cpu" and torch.cuda.is_available() else "cpu"
        if self.spec.backend == "florence":
            self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        else:
            self.dtype = torch.bfloat16
        self.report(tr(msg.log_CaptionDownload, model=self.spec.repo_id))
        folder = Path(snapshot_download(self.spec.repo_id, revision=self.spec.revision,
                                        local_files_only=self.offline))
        self.source = f"{self.spec.repo_id}@{folder.name}"
        self.processor = AutoProcessor.from_pretrained(
            folder, trust_remote_code=self.spec.backend == "florence" and self.allow_code,
            local_files_only=True)
        if self.spec.backend == "florence":
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                folder, torch_dtype=self.dtype, trust_remote_code=True,
                attn_implementation="eager", local_files_only=True).to(self.device)
        elif self.spec.backend == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                folder, torch_dtype=self.dtype, local_files_only=True).to(self.device)
        elif self.spec.backend == "joycaption":
            from transformers import LlavaForConditionalGeneration

            self.model = LlavaForConditionalGeneration.from_pretrained(
                folder, torch_dtype=self.dtype, local_files_only=True).to(self.device)
        else:
            raise ValueError(tr(msg.error_ModelUnsupported))
        self.model.eval()
        self.runtime = f"PyTorch {self.device.upper()}"
        self.report(tr(msg.log_ModelLoaded, runtime=self.runtime))

    def predict(self, path: Path, options: dict) -> str:
        import torch

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        max_tokens = int(options.get("max_new_tokens", 256))
        if not 1 <= max_tokens <= 2048:
            raise ValueError(tr(msg.error_CaptionTokens))
        with torch.inference_mode():
            if self.spec.backend == "florence":
                task = options.get("task", "<MORE_DETAILED_CAPTION>")
                if task not in FLORENCE_TASKS:
                    raise ValueError(tr(msg.error_CaptionTask))
                inputs = self.processor(text=task, images=image, return_tensors="pt")
                inputs = inputs.to(self.device, self.dtype)
                output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                             do_sample=False, num_beams=3, use_cache=False)
                decoded = self.processor.batch_decode(output, skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    decoded, task=task, image_size=image.size)
                caption = parsed.get(task, "")
            elif self.spec.backend == "qwen3_vl":
                prompt = options.get("prompt") or tr(msg.default_CaptionPrompt)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(self.device)
                output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                             do_sample=False)
                generated = output[0][inputs["input_ids"].shape[-1]:]
                caption = self.processor.decode(generated, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            else:
                prompt = options.get("prompt") or tr(msg.default_JoyCaptionPrompt)
                messages = [{"role": "system", "content": "You are a helpful image captioner."},
                            {"role": "user", "content": prompt}]
                formatted = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[formatted], images=[image], return_tensors="pt").to(
                    self.device)
                if self.device == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                             do_sample=True, temperature=.6, top_p=.9)[0]
                generated = output[inputs["input_ids"].shape[1]:]
                caption = self.processor.tokenizer.decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(tr(msg.error_CaptionEmpty))
        return caption.strip()
