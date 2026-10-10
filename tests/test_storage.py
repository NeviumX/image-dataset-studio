import json

import pytest
from PIL import Image

from image_dataset_studio.core import Tag, compile_filter
from image_dataset_studio.storage import METADATA, hash_bytes, load_dataset, save_dataset


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


def test_metadata_contains_only_sidecar_attributes(tmp_path):
    image(tmp_path / 'a.png')
    ds = load_dataset(tmp_path)
    path = ds.records[0].path
    tags = [Tag('solo'), Tag('blue hair', 'general', 0.0, 'tagger@abc')]
    ds.apply('tags', {path: tags})
    ds.apply_captions('caption', {path: ('A person, outside.', 'captioner@abc')})
    save_dataset(ds)
    metadata = json.loads((tmp_path / METADATA).read_text(encoding='utf-8'))
    assert metadata['version'] == 3
    saved = metadata['images']['a.png']
    assert set(saved) == {'read_mode', 'text_digest', 'tag_metadata',
                          'caption_source', 'caption_digest'}
    assert saved['read_mode'] == 'mixed'
    assert saved['tag_metadata'] == {
        'blue hair': {'category': 'general', 'score': 0.0, 'source': 'tagger@abc'}}
    assert 'A person, outside.' not in json.dumps(metadata)
    assert load_dataset(tmp_path).records[0].tags == tags


def test_external_sidecar_controls_tag_content_order_and_caption(tmp_path):
    image(tmp_path / 'a.png')
    ds = load_dataset(tmp_path)
    path = ds.records[0].path
    ds.apply('tags', {path: [Tag('solo', 'general', .8), Tag('blue hair', 'general', .9)]})
    ds.apply_captions('caption', {path: ('Original caption.', 'captioner@abc')})
    save_dataset(ds)
    (tmp_path / 'a.txt').write_text('blue hair, new tag\n\nOriginal caption.', encoding='utf-8')
    reloaded = load_dataset(tmp_path)
    assert reloaded.records[0].tags == [Tag('blue hair', 'general', .9), Tag('new tag')]
    assert reloaded.records[0].caption_source == 'captioner@abc'
    save_dataset(reloaded)
    saved = json.loads((tmp_path / METADATA).read_text(encoding='utf-8'))['images']['a.png']
    assert set(saved['tag_metadata']) == {'blue hair'}
    (tmp_path / 'a.txt').write_text('blue hair, new tag\n\nEdited caption.', encoding='utf-8')
    record = load_dataset(tmp_path).records[0]
    assert record.caption == 'Edited caption.'
    assert record.caption_source is None


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_metadata_migrates_on_save_without_rewriting_sidecars(tmp_path, version):
    image(tmp_path / 'a.png')
    (tmp_path / 'sub').mkdir()
    image(tmp_path / 'sub' / 'b.png')
    raw = b'solo, blue hair\r\n\r\nA person, outside.\r\n'
    (tmp_path / 'a.txt').write_bytes(raw)
    (tmp_path / 'sub' / 'b.txt').write_text('Nested caption.', encoding='utf-8')
    metadata = {'version': version, 'images': {
        'a.png': {'tags': [{'name': 'solo', 'category': 'general', 'score': .8,
                            'source': 'tagger@abc'}, {'name': 'blue hair'}],
                  'caption': 'A person, outside.', 'caption_source': 'captioner@abc',
                  'text_digest': hash_bytes(raw)},
        'sub/b.png': {'tags': [], 'caption': 'Nested caption.',
                      'text_digest': hash_bytes(b'Nested caption.')},
    }}
    (tmp_path / METADATA).write_text(json.dumps(metadata), encoding='utf-8')
    ds = load_dataset(tmp_path, recursive=False)
    assert ds.dirty
    assert ds.records[0].caption == 'A person, outside.'
    assert ds.records[0].tags[0] == Tag('solo', 'general', .8, 'tagger@abc')
    assert save_dataset(ds) == 1
    assert (tmp_path / 'a.txt').read_bytes() == raw
    migrated = json.loads((tmp_path / METADATA).read_text(encoding='utf-8'))
    assert migrated['version'] == 3
    assert all('tags' not in value and 'caption' not in value
               for value in migrated['images'].values())
    reloaded = load_dataset(tmp_path)
    assert not reloaded.dirty
    assert reloaded.records[0].caption_source == 'captioner@abc'
    assert reloaded.records[1].caption == 'Nested caption.'


def test_legacy_tag_list_metadata_keeps_scores(tmp_path):
    image(tmp_path / 'a.png')
    (tmp_path / 'a.txt').write_text('solo, new tag', encoding='utf-8')
    (tmp_path / METADATA).write_text(json.dumps({'version': 1, 'images': {
        'a.png': [{'name': 'solo', 'category': 'general', 'score': .8},
                  {'name': 'removed tag', 'score': .9}],
    }}), encoding='utf-8')
    ds = load_dataset(tmp_path)
    assert ds.records[0].tags == [Tag('solo', 'general', .8), Tag('new tag')]
    save_dataset(ds)
    assert not load_dataset(tmp_path).dirty
    saved = json.loads((tmp_path / METADATA).read_text(encoding='utf-8'))['images']['a.png']
    assert saved['tag_metadata'] == {'solo': {'category': 'general', 'score': .8}}


@pytest.mark.parametrize('written', [False, True])
def test_pending_recovery_uses_attributes_of_the_actual_sidecar(tmp_path, monkeypatch, written):
    from image_dataset_studio import storage

    image(tmp_path / 'a.png')
    (tmp_path / 'a.txt').write_text('solo', encoding='utf-8')
    ds = load_dataset(tmp_path)
    path = ds.records[0].path
    ds.apply('score', {path: [Tag('solo', 'general', .8)]})
    save_dataset(ds)
    ds.apply('new tags', {path: [Tag('solo', 'general', .9), Tag('new tag')]})
    ds.apply_captions('caption', {path: ('New caption.', 'captioner@abc')})
    original_write = storage.atomic_write
    metadata_writes = 0

    def fail_write(path, data):
        nonlocal metadata_writes
        if path.name == METADATA:
            metadata_writes += 1
        if (written and metadata_writes == 2) or (not written and path.suffix == '.txt'):
            raise OSError('simulated interruption')
        original_write(path, data)

    monkeypatch.setattr(storage, 'atomic_write', fail_write)
    with pytest.raises(OSError, match='simulated'):
        save_dataset(ds)
    staged = json.loads((tmp_path / METADATA).read_text(encoding='utf-8'))
    for value in [*staged['images'].values(), *staged['pending']['a.png'].values()]:
        assert 'tags' not in value and 'caption' not in value
    monkeypatch.setattr(storage, 'atomic_write', original_write)
    recovered = load_dataset(tmp_path)
    record = recovered.records[0]
    assert record.tags[0].score == (.9 if written else .8)
    assert record.caption == ('New caption.' if written else '')
    assert record.caption_source == ('captioner@abc' if written else None)
    assert recovered.dirty
    save_dataset(recovered)
    assert not load_dataset(tmp_path).dirty
