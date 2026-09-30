import json
import sys
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QProcess

from image_dataset_studio.core import Tag
from image_dataset_studio.models import ModelSpec
from image_dataset_studio.tagging_worker import run
from image_dataset_studio.translations import set_language
from image_dataset_studio.workers import TaggingJob


def test_isolated_worker_writes_progress_and_results(tmp_path, monkeypatch):
    class FakeTagger:
        def __init__(self, spec, device, offline, allow_code, report):
            self.report = report
            self.runtime = 'fake CPU'
            self.vocabulary = {'solo': 'general'}

        def load(self):
            self.report('モデルの読み込みが完了しました。')

        def predict(self, path, thresholds):
            return [Tag('solo', 'general', .9)]

    monkeypatch.setattr('image_dataset_studio.taggers.Tagger', FakeTagger)
    source, output, events = (tmp_path / name for name in ('input.json', 'output.json', 'events.jsonl'))
    image = tmp_path / 'a.png'
    source.write_text(json.dumps({
        'spec': asdict(ModelSpec('u/m', 'wd')), 'paths': [str(image)],
        'thresholds': {}, 'device': 'cpu', 'offline': True, 'allow_code': False,
        'vocabulary_only': False,
    }), encoding='utf-8')
    assert run(source, output, events) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['predictions'][str(image)] == [asdict(Tag('solo', 'general', .9))]
    assert result['runtime'] == 'fake CPU'
    progress = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    assert [event['current'] for event in progress if event['type'] == 'progress'] == [0, 0, 1]
    assert [event['tags'] for event in progress if event['type'] == 'image_result'] == [
        [asdict(Tag('solo', 'general', .9))]
    ]


def test_isolated_caption_worker_returns_text_without_tag_parsing(tmp_path, monkeypatch):
    class FakeCaptioner:
        def __init__(self, spec, device, offline, allow_code, report):
            self.source = f'{spec.repo_id}@abc'
            self.runtime = 'fake CPU'

        def load(self):
            pass

        def predict(self, path, options):
            assert options['max_new_tokens'] == 100
            return 'A woman, by a window.'

    monkeypatch.setattr('image_dataset_studio.captioners.Captioner', FakeCaptioner)
    source, output, events = (tmp_path / name for name in ('input.json', 'output.json', 'events.jsonl'))
    image = tmp_path / 'a.png'
    source.write_text(json.dumps({
        'spec': asdict(ModelSpec('Qwen/Qwen3-VL-4B-Instruct', 'qwen3_vl')),
        'paths': [str(image)], 'thresholds': {}, 'device': 'cpu', 'offline': True,
        'allow_code': False, 'vocabulary_only': False,
        'options': {'max_new_tokens': 100},
    }), encoding='utf-8')
    assert run(source, output, events) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['captions'][str(image)] == 'A woman, by a window.'
    assert result['predictions'] == {}
    progress = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
    assert [event['caption'] for event in progress if event['type'] == 'image_result'] == [
        'A woman, by a window.'
    ]


def test_worker_request_and_progress_follow_selected_language(tmp_path, monkeypatch):
    class FakeTagger:
        def __init__(self, spec, device, offline, allow_code, report):
            self.vocabulary = {'solo': 'general'}

        def load_vocabulary(self):
            pass

    monkeypatch.setattr('image_dataset_studio.taggers.Tagger', FakeTagger)
    set_language('EN')
    try:
        job = TaggingJob(ModelSpec('u/m', 'wd'), [], {}, 'cpu', True, False, True)
        request = json.loads(job.input_path.read_text(encoding='utf-8'))
        assert request['language'] == 'EN'
        output, events = tmp_path / 'output.json', tmp_path / 'events.jsonl'
        assert run(job.input_path, output, events) == 0
        messages = [json.loads(line)['message'] for line in events.read_text(encoding='utf-8').splitlines()
                    if 'message' in json.loads(line)]
        assert any(message.startswith('Preparing model:') for message in messages)
        job._temporary.cleanup()
    finally:
        set_language('JP')


def test_tagging_process_reports_offline_download_failure(qtbot):
    job = TaggingJob(ModelSpec('missing/model', 'pixai', labels_file='config.json'),
                     [], {}, 'cpu', True, True, True)
    logs, errors = [], []
    job.log.connect(logs.append)
    job.progress.connect(lambda _current, _total, message: logs.append(message))
    job.error.connect(errors.append)
    with qtbot.waitSignal(job.finished, timeout=20000):
        job.start()
    assert errors
    assert any('config.json' in line for line in logs)
    assert any('missing/model' in line for line in logs)
    assert job.process.state() == QProcess.ProcessState.NotRunning
    assert not job.input_path.exists()


def test_tagging_process_can_cancel_blocked_child(qtbot, monkeypatch):
    job = TaggingJob(ModelSpec('missing/model', 'wd'), [], {}, 'cpu', True, False)
    monkeypatch.setattr(job, '_command', lambda: (
        Path(sys.executable), ['-c', 'import time; time.sleep(60)']))
    errors, results, logs = [], [], []
    job.error.connect(errors.append)
    job.result.connect(results.append)
    job.log.connect(logs.append)
    with qtbot.waitSignal(job.process.started, timeout=5000):
        job.start()
    with qtbot.waitSignal(job.finished, timeout=5000):
        job.requestInterruption()
    assert job.process.state() == QProcess.ProcessState.NotRunning
    assert not errors and not results
    assert any('停止しました' in line for line in logs)
    assert not job.input_path.exists()


def test_cancel_finishes_when_windows_temporarily_locks_event_file(qtbot, monkeypatch):
    job = TaggingJob(ModelSpec('missing/model', 'wd'), [], {}, 'cpu', True, False)
    monkeypatch.setattr(job, '_command', lambda: (
        Path(sys.executable), ['-c', 'import time; time.sleep(60)']))
    original_cleanup = job._temporary.cleanup
    attempts = []

    def locked_once():
        attempts.append(True)
        if len(attempts) == 1:
            raise PermissionError('event file is still open')
        original_cleanup()

    monkeypatch.setattr(job._temporary, 'cleanup', locked_once)
    with qtbot.waitSignal(job.process.started, timeout=5000):
        job.start()
    with qtbot.waitSignal(job.finished, timeout=5000):
        job.requestInterruption()
    assert len(attempts) == 1
    qtbot.waitUntil(lambda: len(attempts) == 2, timeout=3000)
    assert not job.input_path.exists()
