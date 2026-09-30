from pathlib import Path

import pytest

from image_dataset_studio.core import (
    Dataset,
    ImageRecord,
    Tag,
    compile_filter,
    insert_tags,
    parse_tags,
    remove_category,
    remove_characters,
    rename_tags,
)


def test_batch_history_preserves_metadata_and_saved_state():
    a, b = ImageRecord(Path('a.png'), [Tag('alice', 'character', .9)]), ImageRecord(Path('b.png'))
    a.saved_tags = list(a.tags)
    dataset = Dataset(Path('.'), [a, b])
    assert dataset.transform('add', [a, b], lambda tags: tags + [Tag('solo')]) == 2
    assert a.dirty and b.dirty
    dataset.undo()
    assert not a.dirty and not b.dirty
    assert a.tags[0].score == .9
    dataset.redo()
    assert len(a.tags) == 2 and len(b.tags) == 1


def test_character_removal_is_category_based():
    tags = [Tag('alice', 'character'), Tag('original', 'copyright'), Tag('long_hair'),
            Tag('bob'), Tag('custom_character'), Tag('charlie', 'general')]
    result = remove_characters(tags, {'bob': 'character', 'charlie': 'character'})
    assert [t.name for t in result] == ['original', 'long_hair', 'custom_character', 'charlie']


@pytest.mark.parametrize('category', ['character', 'copyright', 'style', 'artist', 'meta'])
def test_category_removal_uses_metadata_and_dictionary_for_unknown_tags(category):
    tags = [Tag('direct', category), Tag('lookup'), Tag('protected', 'general'), Tag('keep')]
    vocabulary = {'lookup': category, 'protected': category}
    assert remove_category(tags, category, vocabulary) == [
        Tag('protected', 'general'), Tag('keep')
    ]


def test_parse_and_rename_preserve_or_clear_metadata():
    tag = Tag('alice', 'character', .8)
    assert parse_tags('alice, solo, solo', [tag]) == [tag, Tag('solo')]
    assert rename_tags([tag], 'alice', 'my_trigger') == [Tag('my_trigger')]
    assert rename_tags([Tag('red_hair'), Tag('blue_hair')], '.*_hair', 'hair', True) == [Tag('hair')]


def test_insert_tags_keeps_position_and_handles_existing_tags():
    old = [Tag('first'), Tag('anchor'), Tag('last')]
    assert [tag.name for tag in insert_tags(old, [Tag('new')], 'after', 'anchor')] == [
        'first', 'anchor', 'new', 'last'
    ]
    assert insert_tags(old, [Tag('first')], 'after', 'anchor') == old
    assert [tag.name for tag in insert_tags(old, [Tag('first')], 'end', skip_existing=False)] == [
        'anchor', 'last', 'first'
    ]
    assert [tag.name for tag in insert_tags(old, [Tag('new')], 'before', 'missing')] == [
        'new', 'first', 'anchor', 'last'
    ]


@pytest.mark.parametrize(('query', 'expected'), [
    ('tag:solo -tag:hat', True), ('tag:hat | name:a.png', True),
    ('tag:"blue hair" tags:>=2', True), ('category:character', False),
    ('tags:<2', False), ('untagged:', False), ('-tag:solo', False),
])
def test_filter(query, expected):
    record = ImageRecord(Path('a.png'), [Tag('solo'), Tag('blue_hair')])
    assert compile_filter(query)(record) is expected


@pytest.mark.parametrize('query', ['tags:hello', 'foo:bar', 'tag:solo |', '| tag:solo', 'tag:"oops'])
def test_invalid_filter(query):
    with pytest.raises(ValueError):
        compile_filter(query)


def test_failed_batch_does_not_partially_mutate():
    a = ImageRecord(Path('a.png'))
    dataset = Dataset(Path('.'), [a])
    with pytest.raises(KeyError):
        dataset.apply('invalid', {a.path: [Tag('solo')], Path('missing.png'): []})
    assert not a.tags


def test_caption_edit_coalesces_undo_and_tag_edit_keeps_caption():
    record = ImageRecord(Path('a.png'), [Tag('solo')])
    record.saved_tags = list(record.tags)
    ds = Dataset(Path('.'), [record])
    ds.apply_captions('caption', {record.path: ('A person', None)}, merge_key='typing:1')
    ds.apply_captions('caption', {record.path: ('A person, outside.', None)},
                      merge_key='typing:1')
    assert len(ds.undo_stack) == 1
    ds.apply('tags', {record.path: [Tag('blue hair')]})
    assert record.caption == 'A person, outside.'
    ds.undo()
    assert record.tags == [Tag('solo')] and record.caption == 'A person, outside.'
    ds.undo()
    assert record.caption == '' and not record.dirty
