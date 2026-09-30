import json

import pytest
from PIL import Image

from image_dataset_studio.core import Tag, compile_filter
from image_dataset_studio.storage import METADATA, load_dataset, save_dataset


def image(path):
    Image.new('RGB', (16, 12)).save(path)


def test_roundtrip_and_undo_after_save(tmp_path):
    image(tmp_path / 'test.png')
    (tmp_path / 'test.txt').write_text('solo, alice', encoding='utf-8')
    ds = load_dataset(tmp_path)
    record = ds.records[0]
    tags = [Tag('solo', 'general', .8), Tag('bob', 'character', .9, 'model@revision')]
    ds.apply('classify', {record.path: tags})
    assert save_dataset(ds) == 1
    assert not ds.dirty
    reloaded = load_dataset(tmp_path)
    assert reloaded.records[0].tags == tags
    assert (tmp_path / 'test.txt').read_text() == 'solo, bob'
    assert not list(tmp_path.glob('*.bak'))
    ds.undo()
    assert ds.dirty
    save_dataset(ds)
    assert (tmp_path / 'test.txt').read_text() == 'solo, alice'
    ds.redo()
    assert ds.dirty
    save_dataset(ds)
    assert (tmp_path / 'test.txt').read_text() == 'solo, bob'
    assert load_dataset(tmp_path).records[0].tags == tags
    assert not list(tmp_path.glob('*.bak'))


def test_external_change_aborts_all_writes(tmp_path):
    for name in ['a', 'b']:
        image(tmp_path / f'{name}.png')
        (tmp_path / f'{name}.txt').write_text('solo')
    ds = load_dataset(tmp_path)
    ds.transform('change', ds.records, lambda t: [Tag('new')])
    (tmp_path / 'b.txt').write_text('external')
    with pytest.raises(ValueError, match='外部'):
        save_dataset(ds)
    assert (tmp_path / 'a.txt').read_text() == 'solo'
    assert (tmp_path / 'b.txt').read_text() == 'external'


def test_stem_collision(tmp_path):
    image(tmp_path / 'a.png')
    image(tmp_path / 'a.jpg')
    with pytest.raises(ValueError, match='同じタグ'):
        load_dataset(tmp_path)


def test_empty_file_creation_and_unknown_tags(tmp_path):
    image(tmp_path / 'a.png')
    ds = load_dataset(tmp_path)
    assert ds.records[0].tags == []
    ds.apply('add', {ds.records[0].path: [Tag('trigger')]})
    save_dataset(ds)
    assert load_dataset(tmp_path).records[0].tags == [Tag('trigger')]


def test_corrupt_metadata_is_not_overwritten(tmp_path):
    (tmp_path / METADATA).write_text('{invalid')
    with pytest.raises(json.JSONDecodeError):
        load_dataset(tmp_path)
    assert (tmp_path / METADATA).read_text() == '{invalid'


def test_nonrecursive_save_preserves_nested_metadata(tmp_path):
    (tmp_path / 'sub').mkdir()
    image(tmp_path / 'a.png')
    image(tmp_path / 'sub' / 'b.png')
    ds = load_dataset(tmp_path)
    ds.transform('add', ds.records, lambda t: [Tag('alice', 'character')])
    save_dataset(ds)
    ds = load_dataset(tmp_path, False)
    ds.transform('edit', ds.records, lambda t: [Tag('solo')])
    save_dataset(ds)
    reloaded = load_dataset(tmp_path)
    assert reloaded.records[1].tags == [Tag('alice', 'character')]


def test_tags_caption_mixed_and_empty_roundtrip(tmp_path):
    for name in ('tags', 'caption', 'mixed', 'empty'):
        image(tmp_path / f'{name}.png')
    ds = load_dataset(tmp_path)
    paths = {record.path.stem: record.path for record in ds.records}
    ds.apply('tags', {paths['tags']: [Tag('solo')], paths['mixed']: [Tag('blue hair')]})
    ds.apply_captions('captions', {
        paths['caption']: ('A woman stands by a window, looking outside.', None),
        paths['mixed']: (' \nA person stands outside.\n\nThe sky is blue, with clouds.\n\t',
                         'model@abc'),
    })
    assert save_dataset(ds) == 3
    assert (tmp_path / 'tags.txt').read_text() == 'solo'
    assert (tmp_path / 'caption.txt').read_text() == 'A woman stands by a window, looking outside.'
    assert (tmp_path / 'mixed.txt').read_text() == (
        'blue hair\n\nA person stands outside.\n\nThe sky is blue, with clouds.')
    assert not (tmp_path / 'empty.txt').exists()
    reloaded = load_dataset(tmp_path)
    by_name = {record.path.stem: record for record in reloaded.records}
    assert by_name['caption'].tags == []
    assert by_name['caption'].caption == 'A woman stands by a window, looking outside.'
    assert by_name['mixed'].tags == [Tag('blue hair')]
    assert by_name['mixed'].caption_source == 'model@abc'
    assert compile_filter('text:clouds')(by_name['mixed'])
    assert not compile_filter('tag:clouds')(by_name['mixed'])
    assert by_name['empty'].tags == [] and by_name['empty'].caption == ''


def test_explicit_reload_modes_and_caption_only_without_metadata(tmp_path):
    image(tmp_path / 'a.png')
    sidecar = tmp_path / 'a.txt'
    sidecar.write_text('A woman, by a window.', encoding='utf-8')
    initial = load_dataset(tmp_path).records[0]
    assert [tag.name for tag in initial.tags] == ['A woman', 'by a window.']
    reloaded = load_dataset(tmp_path, read_mode='caption')
    assert reloaded.records[0].tags == []
    assert reloaded.records[0].caption == 'A woman, by a window.'
    assert reloaded.dirty
    save_dataset(reloaded)
    assert load_dataset(tmp_path).records[0].caption == 'A woman, by a window.'
    as_tags = load_dataset(tmp_path, read_mode='tags')
    assert [tag.name for tag in as_tags.records[0].tags] == ['A woman', 'by a window.']
    assert as_tags.dirty


def test_mixed_reload_uses_only_first_blank_line(tmp_path):
    image(tmp_path / 'a.png')
    (tmp_path / 'a.txt').write_text('solo, blue hair\n\nA person, outside.\n\nMore text.',
                                    encoding='utf-8')
    record = load_dataset(tmp_path, read_mode='mixed').records[0]
    assert [tag.name for tag in record.tags] == ['solo', 'blue hair']
    assert record.caption == 'A person, outside.\n\nMore text.'


def test_pending_metadata_recovers_caption_after_partial_save(tmp_path, monkeypatch):
    from image_dataset_studio import storage

    image(tmp_path / 'a.png')
    ds = load_dataset(tmp_path)
    ds.apply_captions('caption', {ds.records[0].path: ('A person, outside.', None)})
    original_write = storage.atomic_write
    metadata_writes = []

    def fail_final_metadata(path, data):
        if path.name == METADATA:
            metadata_writes.append(True)
            if len(metadata_writes) == 2:
                raise OSError('simulated final metadata write failure')
        original_write(path, data)

    monkeypatch.setattr(storage, 'atomic_write', fail_final_metadata)
    with pytest.raises(OSError, match='simulated'):
        save_dataset(ds)
    monkeypatch.setattr(storage, 'atomic_write', original_write)
    recovered = load_dataset(tmp_path)
    assert recovered.records[0].tags == []
    assert recovered.records[0].caption == 'A person, outside.'
    assert recovered.dirty
    save_dataset(recovered)
    assert not load_dataset(tmp_path).dirty


def test_pending_metadata_recovers_before_sidecar_write(tmp_path, monkeypatch):
    from image_dataset_studio import storage

    image(tmp_path / 'a.png')
    (tmp_path / 'a.txt').write_text('solo', encoding='utf-8')
    ds = load_dataset(tmp_path)
    ds.apply_captions('caption', {ds.records[0].path: ('A person, outside.', None)})
    original_write = storage.atomic_write

    def fail_sidecar(path, data):
        if path.name == 'a.txt':
            raise OSError('simulated sidecar write failure')
        original_write(path, data)

    monkeypatch.setattr(storage, 'atomic_write', fail_sidecar)
    with pytest.raises(OSError, match='simulated'):
        save_dataset(ds)
    monkeypatch.setattr(storage, 'atomic_write', original_write)
    recovered = load_dataset(tmp_path)
    assert recovered.records[0].tags == [Tag('solo')]
    assert recovered.records[0].caption == ''
    assert recovered.dirty
    save_dataset(recovered)
    assert not load_dataset(tmp_path).dirty
    assert (tmp_path / 'a.txt').read_text(encoding='utf-8') == 'solo'


def test_external_sidecar_then_partial_save_keeps_external_state(tmp_path, monkeypatch):
    from image_dataset_studio import storage

    image(tmp_path / 'a.png')
    (tmp_path / 'a.txt').write_text('solo', encoding='utf-8')
    initial = load_dataset(tmp_path)
    initial.apply('classify', {initial.records[0].path: [Tag('solo', 'general')]})
    save_dataset(initial)
    (tmp_path / 'a.txt').write_text('external, tag', encoding='utf-8')
    ds = load_dataset(tmp_path)
    assert [tag.name for tag in ds.records[0].tags] == ['external', 'tag']
    ds.apply_captions('caption', {ds.records[0].path: ('A person.', None)})
    original_write = storage.atomic_write

    def fail_sidecar(path, data):
        if path.name == 'a.txt':
            raise OSError('simulated sidecar write failure')
        original_write(path, data)

    monkeypatch.setattr(storage, 'atomic_write', fail_sidecar)
    with pytest.raises(OSError, match='simulated'):
        save_dataset(ds)
    monkeypatch.setattr(storage, 'atomic_write', original_write)
    recovered = load_dataset(tmp_path)
    assert [tag.name for tag in recovered.records[0].tags] == ['external', 'tag']
    assert recovered.records[0].caption == ''
