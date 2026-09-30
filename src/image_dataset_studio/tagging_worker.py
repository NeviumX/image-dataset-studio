"""Isolated model runner. Communicates through temporary files, never the dataset."""

from __future__ import annotations

import json
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path


class EventStream:
    encoding = "utf-8"

    def __init__(self, emit):
        self.emit = emit
        self.pending = ""

    def write(self, text):
        self.pending += text
        parts = re.split(r"[\r\n]", self.pending)
        self.pending = parts.pop()
        for part in parts:
            if part.strip():
                self.emit("log", message=part.strip()[:1000])
        return len(text)

    def flush(self):
        if self.pending.strip():
            self.emit("log", message=self.pending.strip()[:1000])
        self.pending = ""

    def isatty(self):
        return True


def run(input_path: Path, output_path: Path, events_path: Path) -> int:
    with events_path.open("w", encoding="utf-8", buffering=1) as events:
        def emit(kind, **data):
            events.write(json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n")

        stream = EventStream(emit)
        with redirect_stderr(stream), redirect_stdout(stream):
            try:
                from .models import ModelSpec
                from .translations import msg, set_language, tr

                request = json.loads(input_path.read_text(encoding="utf-8"))
                set_language(request.get("language", "JP"))
                spec = ModelSpec(**request["spec"])
                paths = [Path(path) for path in request["paths"]]
                total = len(paths)

                def report(message):
                    emit("log", message=message)

                emit("progress", current=0, total=total,
                     message=tr(msg.progress_ModelPreparing, model=spec.repo_id))
                if spec.kind == "caption":
                    from .captioners import Captioner

                    captioner = Captioner(spec, request["device"], request["offline"],
                                          request["allow_code"], report)
                    captioner.load()
                    captions, errors = {}, []
                    for index, path in enumerate(paths, 1):
                        emit("progress", current=index - 1, total=total,
                             message=tr(msg.progress_ImageAnalyzing, index=index,
                                        total=total, filename=path.name))
                        try:
                            caption = captioner.predict(path, request.get("options", {}))
                            captions[str(path)] = caption
                            emit("image_result", path=str(path), caption=caption,
                                 source=captioner.source)
                        except Exception as exc:
                            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                            emit("log", message=errors[-1])
                            emit("image_error", message=errors[-1])
                        emit("progress", current=index, total=total,
                             message=tr(msg.progress_ImageDone, index=index, total=total))
                    result = {"captions": captions, "source": captioner.source,
                              "errors": errors, "runtime": captioner.runtime,
                              "vocabulary": {}, "predictions": {}, "cancelled": False}
                else:
                    from .taggers import Tagger

                    tagger = Tagger(spec, request["device"], request["offline"],
                                    request["allow_code"], report)
                    if request["vocabulary_only"]:
                        tagger.load_vocabulary()
                        result = {"vocabulary": tagger.vocabulary, "predictions": {}}
                    else:
                        tagger.load()
                        predictions, errors = {}, []
                        for index, path in enumerate(paths, 1):
                            emit("progress", current=index - 1, total=total,
                                 message=tr(msg.progress_ImageAnalyzing, index=index,
                                            total=total, filename=path.name))
                            try:
                                tags = [asdict(tag) for tag in tagger.predict(
                                    path, request["thresholds"])]
                                predictions[str(path)] = tags
                                emit("image_result", path=str(path), tags=tags)
                            except Exception as exc:
                                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                                emit("log", message=errors[-1])
                                emit("image_error", message=errors[-1])
                            emit("progress", current=index, total=total,
                                 message=tr(msg.progress_ImageDone, index=index, total=total))
                        result = {"predictions": predictions, "errors": errors,
                                  "vocabulary": tagger.vocabulary, "runtime": tagger.runtime,
                                  "cancelled": False}
                output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                emit("log", message=tr(msg.log_WorkerDone))
                return 0
            except Exception as exc:
                stream.flush()
                emit("log", message=traceback.format_exc())
                emit("error", message=f"{type(exc).__name__}: {exc}")
                return 1


if __name__ == "__main__":
    raise SystemExit(run(*(Path(path) for path in sys.argv[1:4])))
