import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from image_dataset_studio.models import ModelRegistry, ModelSpec, detect_model, parse_hf_url
from image_dataset_studio.taggers import (
    Tagger,
    camie_input,
    decode_scores,
    pixai_vocabulary,
    wd_input,
)


def test_wd_input_bgr_range_and_alpha():
    values = wd_input(Image.new('RGBA', (4, 4), (255, 0, 0, 255)), 4)
    assert values.shape == (1, 4, 4, 3) and values.dtype == np.float32
    np.testing.assert_array_equal(values[0, 0, 0], [0, 0, 255])
    transparent = wd_input(Image.new('RGBA', (4, 2), (0, 0, 0, 0)), 4)
    assert np.all(transparent == 255)


def test_camie_input_normalization_and_padding():
    values = camie_input(Image.new('RGB', (4, 2), (255, 0, 0)), 4)
    assert values.shape == (1, 3, 4, 4)
    np.testing.assert_allclose(values[0, :, 1, 0], [(1 - .485) / .229, -.456 / .224, -.406 / .225], rtol=1e-5)
    assert np.max(np.abs(values[0, :, 0, 0])) < .02


def test_threshold_categories_and_output_validation():
    tags = decode_scores(['solo', 'alice', 'bob'], ['general', 'character', 'character'],
                         np.array([.4, .9, .8]), {'general': .35, 'character': .85}, 'model')
    assert [t.name for t in tags] == ['alice', 'solo']
    assert tags[0].category == 'character'
    with pytest.raises(ValueError):
        decode_scores(['a'], ['general'], np.array([.4, .5]), {}, 'model')


def test_pixai_vocab():
    assert pixai_vocabulary({'tags': ['long_hair', 'alice'], 'tags_split': [['general', 1], ['character', 1]]}) == {
        'long hair': 'general', 'alice': 'character'}


@pytest.mark.parametrize('url', ['https://evil.test/u/m', 'https://huggingface.co/u/m/blob/main/file',
                                  'https://huggingface.co/spaces/u/m', '../x', 'u/m?x=y'])
def test_invalid_url(url):
    with pytest.raises(ValueError):
        parse_hf_url(url)


def test_detection_and_persisted_models(tmp_path):
    assert parse_hf_url('https://huggingface.co/u/m/tree/v1') == ('u/m', 'v1')
    assert parse_hf_url('u/m') == ('u/m', 'main')
    spec = detect_model('u/m', 'abc123', {'model.onnx', 'selected_tags.csv'})
    assert spec.backend == 'wd'
    registry = ModelRegistry(tmp_path / 'models.json')
    registry.add(spec)
    assert ModelRegistry(registry.path).models[-1] == spec
    with pytest.raises(ValueError, match='未対応'):
        detect_model('u/m', 'main', {'random.bin'})


def test_pixai_defaults():
    assert ModelSpec('u/m', 'pixai').thresholds['character'] == .27


def test_caption_presets_and_known_url_detection(tmp_path):
    registry = ModelRegistry(tmp_path / 'models.json')
    caption_models = [spec for spec in registry.models if spec.kind == 'caption']
    assert {spec.backend for spec in caption_models} == {'florence', 'qwen3_vl', 'joycaption'}
    assert len(caption_models) == 4
    detected = detect_model('Qwen/Qwen3-VL-4B-Instruct', 'abc123',
                            {'config.json', 'model.safetensors.index.json'})
    assert detected.kind == 'caption' and detected.revision == 'abc123'
    with pytest.raises(ValueError):
        detect_model('unknown/vlm', 'main', {'config.json', 'model.safetensors'})


def test_explicit_cuda_does_not_silently_fall_back_to_cpu(monkeypatch):
    preloaded = []
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(shape=[1, 448, 448, 3])],
        get_providers=lambda: ['CPUExecutionProvider'],
    )
    runtime = SimpleNamespace(
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        preload_dlls=lambda: preloaded.append(True),
        InferenceSession=lambda *_args, **_kwargs: session,
    )
    monkeypatch.setitem(sys.modules, 'onnxruntime', runtime)
    monkeypatch.setattr(Tagger, 'load_vocabulary', lambda self: None)
    monkeypatch.setattr(Tagger, 'download', lambda self, _filename: 'model.onnx')
    tagger = Tagger(ModelSpec('u/m', 'wd'), device='cuda')
    with pytest.raises(RuntimeError, match='CUDA実行プロバイダー'):
        tagger.load()
    assert preloaded


def test_auto_retries_onnx_model_on_cpu_when_cuda_fails(monkeypatch):
    attempts = []

    def create_session(_path, *, providers):
        attempts.append(providers)
        if providers[0] == 'CUDAExecutionProvider':
            raise RuntimeError('CUDA driver unavailable')
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(shape=[1, 448, 448, 3])],
            get_providers=lambda: providers,
        )

    runtime = SimpleNamespace(
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        preload_dlls=lambda: None,
        InferenceSession=create_session,
    )
    monkeypatch.setitem(sys.modules, 'onnxruntime', runtime)
    monkeypatch.setattr(Tagger, 'load_vocabulary', lambda self: None)
    monkeypatch.setattr(Tagger, 'download', lambda self, _filename: 'model.onnx')
    logs = []
    tagger = Tagger(ModelSpec('u/m', 'wd'), device='auto', report=logs.append)
    tagger.load()
    assert attempts == [['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']]
    assert tagger.runtime == 'CPUExecutionProvider'
    assert any('CPU' in message for message in logs)
