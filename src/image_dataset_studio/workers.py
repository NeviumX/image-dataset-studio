"""Background jobs return results; all dataset mutations happen in the GUI thread."""

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QThread, QTimer, pyqtSignal

from .core import Tag
from .models import inspect_model
from .storage import load_dataset
from .translations import get_language, msg, tr


class Job(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action

    def run(self):
        try:
            result = self.action(self)
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


def _cleanup_temporary(temporary, retry_delays=(500, 2000, 5000)):
    """Windows can briefly keep a killed worker's event file open."""
    try:
        temporary.cleanup()
    except OSError:
        if retry_delays:
            QTimer.singleShot(retry_delays[0],
                              lambda: _cleanup_temporary(temporary, retry_delays[1:]))


def load_job(root: Path, recursive: bool, read_mode: str | None = None):
    return lambda job: load_dataset(root, recursive, read_mode)


def inspect_job(url: str):
    return lambda job: inspect_model(url)


class TaggingJob(QObject):
    """Run model downloads and inference in a process that can be killed promptly."""

    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, spec, paths, thresholds, device, offline, allow_code,
                 vocabulary_only=False, options=None):
        super().__init__()
        self._temporary = tempfile.TemporaryDirectory(prefix="ids-tagger-")
        root = Path(self._temporary.name)
        self.input_path = root / "input.json"
        self.output_path = root / "output.json"
        self.events_path = root / "events.jsonl"
        self.input_path.write_text(json.dumps({
            "spec": asdict(spec), "paths": [str(path) for path in paths],
            "thresholds": thresholds, "device": device, "offline": offline,
            "allow_code": allow_code, "vocabulary_only": vocabulary_only,
            "options": options or {},
            "language": get_language(),
        }, ensure_ascii=False), encoding="utf-8")
        self.process = QProcess(self)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._read_events)
        self._offset = 0
        self._pending = b""
        self._cancelled = False
        self._done = False
        self._last_error = ""
        self._partial_predictions = {}
        self._partial_captions = {}
        self._partial_errors = []
        self._partial_source = None

    def start(self):
        executable, args = self._command()
        self.log.emit(tr(msg.log_WorkerStarting))
        self.timer.start()
        self.process.start(str(executable), args)

    def _command(self):
        executable = Path(sys.executable)
        if getattr(sys, "frozen", False):
            args = ["--tagging-worker"]
        else:
            if executable.name.lower() == "pythonw.exe":
                executable = executable.with_name("python.exe")
            args = ["-m", "image_dataset_studio.app", "--tagging-worker"]
        args.extend([str(self.input_path), str(self.output_path), str(self.events_path)])
        return executable, args

    def requestInterruption(self):
        self._cancelled = True
        self.log.emit(tr(msg.log_WorkerStopping))
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _read_events(self):
        if not self.events_path.exists():
            return
        with self.events_path.open("rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
            self._offset = stream.tell()
        self._pending += chunk
        *lines, self._pending = self._pending.split(b"\n")
        for line in lines:
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.log.emit(line.decode("utf-8", errors="replace"))
                continue
            if event["type"] == "progress":
                self.progress.emit(event["current"], event["total"], event["message"])
            elif event["type"] == "error":
                self._last_error = event["message"]
            elif event["type"] == "log":
                self.log.emit(event["message"])
            elif event["type"] == "image_result":
                path = Path(event["path"])
                if "tags" in event:
                    self._partial_predictions[path] = [Tag(**tag) for tag in event["tags"]]
                else:
                    self._partial_captions[path] = event["caption"]
                    self._partial_source = event.get("source")
            elif event["type"] == "image_error":
                self._partial_errors.append(event["message"])

    def _read_stderr(self):
        for line in bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").splitlines():
            if line.strip():
                self.log.emit(line.strip())

    def _read_stdout(self):
        for line in bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").splitlines():
            if line.strip():
                self.log.emit(line.strip())

    def _process_error(self, code):
        if code == QProcess.ProcessError.FailedToStart:
            self._finish(tr(msg.error_WorkerStart, error=self.process.errorString()))

    def _process_finished(self, exit_code, _status):
        self._read_events()
        self._read_stderr()
        self._read_stdout()
        if self._cancelled:
            self.log.emit(tr(msg.log_WorkerStopped))
            if self._partial_predictions or self._partial_captions:
                self.result.emit({
                    "predictions": self._partial_predictions,
                    "captions": self._partial_captions,
                    "errors": self._partial_errors,
                    "vocabulary": None,
                    "source": self._partial_source,
                    "cancelled": True,
                })
            self._finish()
        elif exit_code or not self.output_path.exists():
            self._finish(self._last_error or tr(msg.error_WorkerExit, code=exit_code))
        else:
            try:
                data = json.loads(self.output_path.read_text(encoding="utf-8"))
                data["predictions"] = {
                    Path(path): [Tag(**tag) for tag in tags]
                    for path, tags in data.get("predictions", {}).items()
                }
                data["captions"] = {
                    Path(path): caption for path, caption in data.get("captions", {}).items()
                }
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self._finish(tr(msg.error_WorkerResult, error=exc))
                return
            _cleanup_temporary(self._temporary)
            self.timer.stop()
            self._done = True
            self.result.emit(data)
            self.finished.emit()

    def _finish(self, message=""):
        if self._done:
            return
        self._done = True
        self.timer.stop()
        _cleanup_temporary(self._temporary)
        if message:
            self.error.emit(message)
        self.finished.emit()


def tagging_job(spec, paths, thresholds, device, offline, allow_code, vocabulary_only=False,
                options=None):
    return TaggingJob(spec, paths, thresholds, device, offline, allow_code, vocabulary_only,
                      options)
