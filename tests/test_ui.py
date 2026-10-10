import json
import sys
from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QMessageBox,
    QToolBar,
    QToolButton,
    QTreeWidget,
)

from image_dataset_studio.core import Tag
from image_dataset_studio.models import ModelSpec
from image_dataset_studio.storage import load_dataset
from image_dataset_studio.translations import set_language
from image_dataset_studio.ui.main_window import TAG_CATEGORY_COLORS, MainWindow
from image_dataset_studio.ui.widgets import AddTagDialog, RenameTagDialog, SortTagsDialog
from image_dataset_studio.workers import TaggingJob


@pytest.fixture(autouse=True)
def prevent_model_downloads(monkeypatch):
    monkeypatch.setattr(MainWindow, 'load_vocabulary', lambda self: None)
    yield
    set_language('JP')


def click_tool(window, name, qtbot):
    button = next(button for button in window.findChildren(QToolButton)
                  if button.accessibleName() == name)
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)


def test_compact_initial_layout_and_reload_toolbar_order(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    assert window.height() <= 900
    assert window.minimumSizeHint().height() <= 900
    toolbar = window.findChild(QToolBar)
    labels = [toolbar.widgetForAction(action).text() for action in toolbar.actions()]
    assert labels[labels.index('やり直す') + 1:labels.index('タグ辞書を読込')] == ['再読み込み']
    assert [action.text() for action in window.reload_menu.actions()] == [
        '自然言語として再読み込み', 'タグとして再読み込み',
        '自然言語とタグ混合として再読み込み'
    ]
    window.resize(1480, 1100)
    window.close()

    reopened = MainWindow(config)
    qtbot.addWidget(reopened)
    reopened.show()
    assert reopened.height() <= 900
    reopened.close()


def test_undo_redo_shortcuts_follow_focus_and_repeat(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(tmp_path))
    record = window.dataset.records[0]
    for name in ('first', 'second', 'third'):
        window.dataset.apply('add', {record.path: record.tags + [Tag(name)]})
    window.refresh_all()

    window.search.setFocus()
    for expected in (['first', 'second'], ['first'], []):
        QTest.keyClick(window.search, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert [tag.name for tag in record.tags] == expected
    for expected in (['first'], ['first', 'second'], ['first', 'second', 'third']):
        QTest.keyClick(window.search, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert [tag.name for tag in record.tags] == expected

    window.tag_search.setFocus()
    QTest.keyClicks(window.tag_search, 'test')
    QTest.keyClick(window.tag_search, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.tag_search.text() == ''
    assert len(record.tags) == 3
    QTest.keyClick(window.tag_search, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert window.tag_search.text() == 'test'
    assert len(record.tags) == 3

    window.log_view.setFocus()
    QTest.keyClick(window.log_view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert [tag.name for tag in record.tags] == ['first', 'second']
    assert next(action for action in window.actions if action.text() == 'やり直す').shortcut().toString() == 'Ctrl+Y'
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()


def test_caption_interpret_button_shares_right_aligned_heading(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    label = window.caption_label.geometry()
    clear_caption = window.clear_caption_button.geometry()
    button = window.interpret_button.geometry()
    editor = window.caption_editor.geometry()
    assert abs(label.center().y() - button.center().y()) <= 4
    assert abs(label.center().y() - clear_caption.center().y()) <= 4
    assert label.right() < clear_caption.left() < button.left()
    assert button.right() <= window.splitter.widget(1).width()
    assert editor.top() > button.bottom()
    tag_label = window.selection_label.geometry()
    clear_tags = window.clear_tags_button.geometry()
    assert abs(tag_label.center().y() - clear_tags.center().y()) <= 4
    assert clear_tags.left() > tag_label.right()
    window.close()


def test_clear_one_images_tags_and_caption_with_undo_redo(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    for name, content in [('a', 'first, second\n\nA sentence, with a comma.'),
                          ('b', 'other\n\nAnother sentence.')]:
        Image.new('RGB', (20, 20)).save(tmp_path / f'{name}.png')
        (tmp_path / f'{name}.txt').write_text(content, encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(tmp_path, read_mode='mixed'))
    first = window.dataset.by_path[tmp_path / 'a.png']
    second = window.dataset.by_path[tmp_path / 'b.png']
    assert window.clear_tags_button.isEnabled()
    assert window.clear_caption_button.isEnabled()
    clear_icon = window.clear_tags_button.icon().pixmap(28, 28).toImage()
    assert clear_icon == window.clear_caption_button.icon().pixmap(28, 28).toImage()
    assert clear_icon.pixelColor(14, 14).name() == '#8c9ab0'

    qtbot.mouseClick(window.clear_tags_button, Qt.MouseButton.LeftButton)
    assert first.tags == [] and first.caption == 'A sentence, with a comma.'
    assert [tag.name for tag in second.tags] == ['other']
    window.undo()
    assert [tag.name for tag in first.tags] == ['first', 'second']
    window.redo()
    assert first.tags == []
    window.undo()

    qtbot.mouseClick(window.clear_caption_button, Qt.MouseButton.LeftButton)
    assert first.caption == '' and [tag.name for tag in first.tags] == ['first', 'second']
    assert second.caption == 'Another sentence.'
    window.undo()
    assert first.caption == 'A sentence, with a comma.'
    window.redo()
    assert first.caption == ''
    assert (tmp_path / 'a.txt').read_text(encoding='utf-8') == (
        'first, second\n\nA sentence, with a comma.')
    window.images.selectionModel().select(window.image_model.index(1),
                                          QItemSelectionModel.SelectionFlag.Select)
    assert not window.clear_tags_button.isEnabled()
    assert not window.clear_caption_button.isEnabled()
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()


@pytest.mark.parametrize(('mode', 'expected'), [
    (0, [['alpha', 'beta', 'zebra'], ['beta', 'zebra']]),
    (1, [['beta', 'zebra', 'alpha'], ['beta', 'zebra']]),
    (2, [['beta', 'alpha', 'zebra'], ['zebra', 'beta']]),
    (3, [['alpha', 'zebra', 'beta'], ['beta', 'zebra']]),
])
def test_sort_dialog_applies_to_selected_images_with_undo_redo(
        qtbot, tmp_path, monkeypatch, mode, expected):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    original = [['zebra', 'alpha', 'beta'], ['beta', 'zebra'], ['beta', 'alpha']]
    for name, tags in zip(('a', 'b', 'c'), original, strict=True):
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text(', '.join(tags), encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    window.images.selectionModel().select(window.image_model.index(1),
                                          QItemSelectionModel.SelectionFlag.Select)
    for row in range(window.tags.topLevelItemCount()):
        item = window.tags.topLevelItem(row)
        item.setSelected(item.data(0, Qt.ItemDataRole.UserRole) == 'alpha')
    monkeypatch.setattr('image_dataset_studio.ui.main_window.random.shuffle',
                        lambda tags: tags.reverse())

    def accept(dialog):
        assert [dialog.mode.itemText(i) for i in range(dialog.mode.count())] == [
            'アルファベット順', '頻度順', 'ランダム', '選択タグを先頭へ']
        dialog.mode.setCurrentIndex(mode)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SortTagsDialog, 'exec', accept)
    click_tool(window, '選択画像のタグを並べ替え', qtbot)

    def orders():
        return [[tag.name for tag in record.tags] for record in window.dataset.records]

    assert orders() == expected + [original[2]]
    window.undo()
    assert orders() == original
    window.redo()
    assert orders() == expected + [original[2]]
    assert (images / 'a.txt').read_text(encoding='utf-8') == ', '.join(original[0])
    history_length = len(window.dataset.undo_stack)

    def cancel(dialog):
        assert dialog.mode.currentIndex() == mode
        dialog.mode.setCurrentIndex((mode + 1) % 4)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(SortTagsDialog, 'exec', cancel)
    click_tool(window, '選択画像のタグを並べ替え', qtbot)
    assert orders() == expected + [original[2]]
    assert len(window.dataset.undo_stack) == history_length
    assert window._sort_mode == mode
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()


def test_dropped_tag_order_is_saved_and_undoable(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    image.with_suffix('.txt').write_text('first, second, third', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(tmp_path))
    record = window.dataset.by_path[image]
    monkeypatch.setattr(window.tags, 'dropIndicatorPosition',
                        lambda: QAbstractItemView.DropIndicatorPosition.BelowItem)
    monkeypatch.setattr(QTreeWidget, 'dropEvent',
                        lambda tree, _event: tree.insertTopLevelItem(
                            3, tree.topLevelItem(0).clone()))
    window.tags.dropEvent(None)
    window.tags.takeTopLevelItem(0)
    qtbot.waitUntil(lambda: [tag.name for tag in record.tags] == ['second', 'third', 'first'])
    window.tags.setFocus()
    QTest.keyClick(window.tags, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert [tag.name for tag in record.tags] == ['first', 'second', 'third']
    QTest.keyClick(window.tags, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert [tag.name for tag in record.tags] == ['second', 'third', 'first']
    assert window.save()
    assert image.with_suffix('.txt').read_text(encoding='utf-8') == 'second, third, first'
    window.close()


def test_edit_save_filter_undo(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    Image.new('RGB', (20, 20)).save(images / 'b.png')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    assert len(window.selected_records()) == 1
    def add_exec(dialog):
        dialog.tag_input.setText('solo, alice')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', add_exec)
    click_tool(window, '選択画像にタグを追加', qtbot)
    assert len(window.dataset.records[0].tags) == 2
    assert not (images / 'a.txt').exists()
    window.vocabulary = {'alice': 'character'}
    window.strip_characters()
    assert window.dataset.records[0].tags == [Tag('solo')]
    window.undo()
    assert len(window.dataset.records[0].tags) == 2
    assert not (images / 'a.txt').exists()
    next(action for action in window.actions if action.text() == 'すべて保存').trigger()
    assert (images / 'a.txt').read_text() == 'solo, alice'
    window.search.setText('tag:alice')
    window.refresh_images()
    assert window.image_model.rowCount() == 1
    assert len(window.target_records()) == 1
    window.generation_scope.setCurrentIndex(2)
    assert window.generation_scope.currentText() == '全画像'
    window.close()


def test_four_column_tag_scopes_and_filenames(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name in ('a', 'b'):
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(images))
    assert [window.left_columns.widget(i).objectName() for i in range(2)] == [
        'imageListPane', 'previewPane'
    ]
    assert [window.splitter.widget(i).objectName() for i in (1, 2)] == [
        'selectedTagsPane', 'globalTagsPane'
    ]
    widths = window.left_columns.sizes() + window.splitter.sizes()[1:]
    assert widths[1] > widths[2] > widths[3]
    def global_exec(dialog):
        dialog.tag_input.setText('shared')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', global_exec)
    click_tool(window, '全画像にタグを追加', qtbot)
    assert all([t.name for t in record.tags] == ['shared'] for record in window.dataset.records)
    assert not list(images.glob('*.txt'))

    def selected_exec(dialog):
        dialog.tag_input.setText('only_a')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', selected_exec)
    click_tool(window, '選択画像にタグを追加', qtbot)
    assert [t.name for t in window.dataset.records[0].tags] == ['shared', 'only_a']
    assert [t.name for t in window.dataset.records[1].tags] == ['shared']
    window.images.selectionModel().select(window.image_model.index(1),
                                          QItemSelectionModel.SelectionFlag.Select)
    window.refresh_tags()
    items = {window.tags.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole):
             window.tags.topLevelItem(i).text(1)
             for i in range(window.tags.topLevelItemCount())}
    assert items['shared'] == 'a.png, b.png'
    assert items['only_a'] == 'a.png'
    assert not window.tags.isColumnHidden(1)
    def rename_exec(dialog):
        assert 'shared' in dialog.tag_input.suggestions.model().stringList()
        dialog.tag_input.setText('renamed')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(RenameTagDialog, 'exec', rename_exec)
    window.stats.setCurrentRow(next(i for i in range(window.stats.count())
                                    if window.stats.item(i).data(Qt.ItemDataRole.UserRole) == 'shared'))
    window.rename_global_tag()
    assert all('renamed' in [t.name for t in r.tags] for r in window.dataset.records)
    window.stats.setCurrentRow(next(i for i in range(window.stats.count())
                                    if window.stats.item(i).data(Qt.ItemDataRole.UserRole) == 'renamed'))
    window.delete_global_tags()
    assert all('renamed' not in [t.name for t in r.tags] for r in window.dataset.records)
    assert window.save()
    window.close()


def test_generated_tags_apply_unsaved_and_show_all_tags(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    a, b, failed = (tmp_path / f'{name}.png' for name in ('a', 'b', 'failed'))
    for image in (a, b, failed):
        Image.new('RGB', (20, 20)).save(image)
    b.with_suffix('.txt').write_text('old', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(tmp_path))
    assert window.replace_underscores.isChecked()
    result = {'vocabulary': {'blue hair': 'general'}, 'errors': [],
              'predictions': {a: [Tag('blue_hair', 'general', .9)],
                              b: [Tag('green_eyes', 'general', .8)]}}
    result['errors'] = ['failed.png: inference failed']
    monkeypatch.setattr(QDialog, 'exec', lambda _: pytest.fail('unexpected dialog'))
    window.global_tabs.setCurrentIndex(1)
    window.tag_search.setText('no match')
    window.predictions_ready(result, 0, set(), True, window.replace_underscores.isChecked())
    assert window.global_tabs.currentIndex() == 0
    assert window.tag_search.text() == ''
    assert window.stats.count() == 3
    assert window.dataset.by_path[a].tags == [Tag('blue hair', 'general', .9)]
    assert window.dataset.by_path[b].tags == [Tag('old'), Tag('green eyes', 'general', .8)]
    assert not window.dataset.by_path[failed].tags
    assert window.dataset.dirty
    assert not a.with_suffix('.txt').exists()
    assert b.with_suffix('.txt').read_text(encoding='utf-8') == 'old'
    assert '1 枚失敗' in window.log_view.toPlainText()
    window.undo()
    assert not window.dataset.by_path[a].tags
    assert window.dataset.by_path[b].tags == [Tag('old')]

    window.replace_underscores.setChecked(False)
    window.predictions_ready(result, 1, set(), True, window.replace_underscores.isChecked())
    assert window.dataset.by_path[a].tags == [Tag('blue_hair', 'general', .9)]
    assert window.dataset.by_path[b].tags == [Tag('green_eyes', 'general', .8)]
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()


def test_reload_menu_caption_mode_then_normal_open_preserves_type(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    (images / 'a.txt').write_text('A woman, by a window.', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    assert [tag.name for tag in window.dataset.records[0].tags] == ['A woman', 'by a window.']
    monkeypatch.setattr(window, 'start_job', lambda action, callback: callback(action(None)))
    window.reload_actions['caption'].trigger()
    record = window.dataset.records[0]
    assert record.tags == [] and record.caption == 'A woman, by a window.'
    assert window.dataset.dirty
    assert window.save()
    assert load_dataset(images).records[0].caption == 'A woman, by a window.'
    window.close()


def test_mixed_reload_and_individual_caption_override(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name, text in [('mixed', 'solo, blue hair\n\nA woman, outside.\n\nMore text.'),
                       ('caption', 'A person, by a window.')]:
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text(text, encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    monkeypatch.setattr(window, 'start_job', lambda action, callback: callback(action(None)))
    window.reload_actions['mixed'].trigger()
    mixed = window.dataset.by_path[images / 'mixed.png']
    assert [tag.name for tag in mixed.tags] == ['solo', 'blue hair']
    assert mixed.caption == 'A woman, outside.\n\nMore text.'
    assert '境界なし 1 枚' in window.statusBar().currentMessage()
    caption = window.dataset.by_path[images / 'caption.png']
    assert caption.caption == ''
    row = window.image_model.records.index(caption)
    window.images.setCurrentIndex(window.image_model.index(row))
    window.interpret_selected('caption')
    assert caption.tags == []
    assert caption.caption == 'A person, by a window.'
    assert '自然文あり' in window.image_model.data(window.image_model.index(row))
    assert window.save()
    assert load_dataset(images).by_path[caption.path].caption == caption.caption
    window.close()


def test_caption_model_ui_and_generation_preserve_tags(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    image.with_suffix('.txt').write_text('solo', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(tmp_path))
    assert window.start_generation_button.text() == 'タグ付け開始'
    assert window.start_generation_button.font().bold()
    index = next(i for i in range(window.model_combo.count())
                 if window.model_combo.itemData(i).repo_id == 'Qwen/Qwen3-VL-4B-Instruct')
    assert window.model_combo.itemText(index).startswith('[NL]')
    assert window.model_combo.itemData(index, Qt.ItemDataRole.ForegroundRole) is not None
    window.model_combo.setCurrentIndex(index)
    assert window.settings_stack.currentIndex() == 1
    assert window.start_generation_button.text() == 'キャプション開始'
    assert window.start_generation_button.font().bold()
    window.captions_ready({'captions': {image: 'A person stands outside, smiling.'},
                           'source': 'Qwen/Qwen3-VL-4B-Instruct@abc', 'errors': []})
    record = window.dataset.records[0]
    assert record.tags == [Tag('solo')]
    assert record.caption == 'A person stands outside, smiling.'
    assert record.caption_source == 'Qwen/Qwen3-VL-4B-Instruct@abc'
    window.caption_editor.setPlainText('A person by a window, smiling.')
    assert record.caption == 'A person by a window, smiling.'
    assert record.caption_source is None
    assert window.save()
    assert image.with_suffix('.txt').read_text(encoding='utf-8') == (
        'solo\n\nA person by a window, smiling.')
    window.close()


def test_tagger_settings_height_follows_selected_model(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.global_tabs.setCurrentIndex(1)
    heights = {}
    for backend in ('pixai', 'florence', 'qwen3_vl', 'joycaption', 'florence'):
        index = next(i for i in range(window.model_combo.count())
                     if window.model_combo.itemData(i).backend == backend)
        window.model_combo.setCurrentIndex(index)
        qtbot.waitUntil(lambda: window.settings_stack.height() ==
                        window.settings_stack.currentWidget().sizeHint().height())
        heights[backend] = window.settings_stack.height()
        assert window.start_generation_button.y() - window.settings_stack.geometry().bottom() < 80
    assert heights['florence'] < heights['qwen3_vl'] < heights['joycaption'] < heights['pixai']
    window.close()


def test_add_dialog_single_row_and_multi_scope(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name, text in [('a', 'first, anchor, last'), ('b', 'first, last')]:
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text(text, encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    window.vocabulary = {'blue hair': 'general'}
    anchor = next(window.tags.topLevelItem(i) for i in range(window.tags.topLevelItemCount())
                  if window.tags.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == 'anchor')
    window.tags.setCurrentItem(anchor)

    def single_exec(dialog):
        assert window.tags.topLevelItem(window.tags.indexOfTopLevelItem(anchor) + 1).text(0) == ''
        assert dialog.layout().getWidgetPosition(dialog.position)[0] == -1
        dialog.tag_input.suggestions.setCompletionPrefix('blue')
        assert dialog.tag_input.suggestions.completionModel().index(0, 0).data() == 'blue hair'
        dialog.tag_input.setText('new')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', single_exec)
    click_tool(window, '選択画像にタグを追加', qtbot)
    assert [tag.name for tag in window.dataset.records[0].tags] == [
        'first', 'anchor', 'new', 'last'
    ]
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'first, anchor, last'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'first, last'
    assert window.tags.topLevelItemCount() == 4

    def cancel_exec(dialog):
        assert window.tags.topLevelItemCount() == 5
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', cancel_exec)
    click_tool(window, '選択画像にタグを追加', qtbot)
    assert window.tags.topLevelItemCount() == 4
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'first, anchor, last'

    window.images.selectionModel().select(window.image_model.index(1),
                                          QItemSelectionModel.SelectionFlag.Select)
    anchor = next(window.tags.topLevelItem(i) for i in range(window.tags.topLevelItemCount())
                  if window.tags.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == 'anchor')
    window.tags.setCurrentItem(anchor)

    def multi_exec(dialog):
        assert dialog.layout().getWidgetPosition(dialog.position)[0] >= 0
        dialog.tag_input.setText('batch')
        dialog.position.setCurrentIndex(1)  # Before the selected tag.
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', multi_exec)
    click_tool(window, '選択画像にタグを追加', qtbot)
    assert [tag.name for tag in window.dataset.records[0].tags] == [
        'first', 'batch', 'anchor', 'new', 'last'
    ]
    assert [tag.name for tag in window.dataset.records[1].tags] == [
        'batch', 'first', 'last'
    ]
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'first, anchor, last'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'first, last'
    next(action for action in window.actions if action.text() == 'すべて保存').trigger()
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'first, batch, anchor, new, last'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'batch, first, last'
    window.close()


def test_global_add_dialog_and_character_removal(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name, text in [('a', 'first, anchor, alice'), ('b', 'first, alice')]:
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text(text, encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    window.stats.setCurrentRow(next(i for i in range(window.stats.count())
                                    if window.stats.item(i).data(Qt.ItemDataRole.UserRole) == 'anchor'))

    def global_exec(dialog):
        dialog.tag_input.setText('shared')
        assert dialog.skip_existing.isChecked()
        assert [dialog.position.itemText(i) for i in range(dialog.position.count())] == [
            '先頭', '末尾', '真ん中', 'カスタム'
        ]
        dialog.position.setCurrentIndex(1)  # End of each caption.
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', global_exec)
    click_tool(window, '全画像にタグを追加', qtbot)
    assert [tag.name for tag in window.dataset.records[0].tags] == [
        'first', 'anchor', 'alice', 'shared'
    ]
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'first, anchor, alice'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'first, alice'
    assert any(window.stats.item(i).data(Qt.ItemDataRole.UserRole) == 'shared'
               for i in range(window.stats.count()))
    window.vocabulary = {'alice': 'character'}
    click_tool(window, '全画像からキャラクタータグを除去', qtbot)
    assert all('alice' not in [tag.name for tag in record.tags] for record in window.dataset.records)
    assert 'alice' in (images / 'a.txt').read_text(encoding='utf-8')
    next(action for action in window.actions if action.text() == 'すべて保存').trigger()
    assert all('alice' not in (images / f'{name}.txt').read_text(encoding='utf-8')
               for name in ('a', 'b'))
    window.close()


@pytest.mark.parametrize(('category', 'label'), [
    ('character', 'キャラクター'), ('copyright', '作品'),
    ('style', 'スタイル'), ('artist', '作者'), ('meta', 'メタ'),
])
@pytest.mark.parametrize('global_scope', [False, True])
def test_category_removal_buttons_keep_edits_unsaved(qtbot, tmp_path, category, label,
                                                     global_scope):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name in ('a', 'b'):
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text('direct, lookup, protected', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    for record in window.dataset.records:
        record.tags = [Tag('direct', category), Tag('lookup'), Tag('protected', 'general')]
        record.saved_tags = list(record.tags)
    window.vocabulary = {'lookup': category, 'protected': category}
    window.refresh_all()

    scope = '全画像' if global_scope else '選択画像'
    click_tool(window, f'{scope}から{label}タグを除去', qtbot)
    assert [tag.name for tag in window.dataset.records[0].tags] == ['protected']
    expected_b = ['protected'] if global_scope else ['direct', 'lookup', 'protected']
    assert [tag.name for tag in window.dataset.records[1].tags] == expected_b
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'direct, lookup, protected'
    window.undo()
    assert all([tag.name for tag in record.tags] == ['direct', 'lookup', 'protected']
               for record in window.dataset.records)
    window.close()


@pytest.mark.parametrize('excluded_category', ['character', 'copyright', 'style', 'artist', 'meta'])
def test_generation_exclusion_checkboxes_filter_each_category(qtbot, tmp_path, monkeypatch,
                                                               excluded_category):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(tmp_path))
    window.dataset.records[0].tags = [Tag('existing', excluded_category)]
    window.dataset.records[0].saved_tags = list(window.dataset.records[0].tags)
    window.allow_code.setChecked(True)
    assert all(not checkbox.isChecked() for checkbox in window.exclude_categories.values())
    window.exclude_categories[excluded_category].setChecked(True)
    predicted = [Tag(category, category) for category in window.exclude_categories]
    predicted.append(Tag('keep', 'general'))
    result = {'vocabulary': {}, 'errors': [], 'predictions': {image: predicted}}
    monkeypatch.setattr('image_dataset_studio.ui.main_window.tagging_job', lambda *args: object())
    monkeypatch.setattr(window, 'start_job', lambda _job, callback: callback(result))
    window.generate()
    assert [tag.name for tag in window.dataset.records[0].tags] == [
        'existing', *(tag.name for tag in predicted if tag.category != excluded_category)
    ]
    assert not image.with_suffix('.txt').exists()
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()


def test_category_colors_match_in_selected_and_all_tags(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(tmp_path))
    record = window.dataset.records[0]
    record.tags = [Tag(f'{category}_tag', category) for category in TAG_CATEGORY_COLORS]
    record.tags.append(Tag('from_dictionary'))
    record.saved_tags = list(record.tags)
    window.vocabulary = {'from dictionary': 'meta'}
    window.refresh_all()
    assert len(set(TAG_CATEGORY_COLORS.values())) == len(TAG_CATEGORY_COLORS)
    selected = {window.tags.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole):
                window.tags.topLevelItem(i) for i in range(window.tags.topLevelItemCount())}
    all_tags = {window.stats.item(i).data(Qt.ItemDataRole.UserRole): window.stats.item(i)
                for i in range(window.stats.count())}
    for category, color in TAG_CATEGORY_COLORS.items():
        name = f'{category}_tag'
        assert selected[name].foreground(0).color().name() == color
        assert all_tags[name].foreground().color().name() == color
    assert selected['from_dictionary'].foreground(0).color().name() == TAG_CATEGORY_COLORS['meta']
    assert all_tags['from_dictionary'].foreground().color().name() == TAG_CATEGORY_COLORS['meta']
    window.close()


def test_tag_lists_share_row_height_and_draw_grid(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    path = tmp_path / 'a.png'
    Image.new('RGB', (20, 20)).save(path)
    path.with_suffix('.txt').write_text('first, second, third', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(tmp_path))
    selected_rect = window.tags.visualItemRect(window.tags.topLevelItem(0))
    global_rect = window.stats.visualItemRect(window.stats.item(0))
    assert selected_rect.height() == global_rect.height() > window.tags.fontMetrics().height()

    border_colors = []
    for view, rect in ((window.tags, selected_rect), (window.stats, global_rect)):
        image = view.viewport().grab().toImage()
        border = image.pixelColor(5, rect.bottom())
        assert border != image.pixelColor(5, rect.bottom() - 1)
        assert border == image.pixelColor(rect.right(), rect.center().y())
        border_colors.append(border)
    assert border_colors[0] == border_colors[1]
    window.close()


@pytest.mark.parametrize(('position', 'index', 'expected_a', 'expected_b'), [
    ('start', 0, 'insert, a, b, c', 'insert, a'),
    ('end', 0, 'a, b, c, insert', 'a, insert'),
    ('middle', 0, 'a, insert, b, c', 'insert, a'),
    ('custom', 2, 'a, b, insert, c', 'a, insert'),
])
def test_global_insert_positions_are_per_image_and_unsaved(
    qtbot, tmp_path, monkeypatch, position, index, expected_a, expected_b
):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    for name, text in [('a', 'a, b, c'), ('b', 'a')]:
        Image.new('RGB', (20, 20)).save(images / f'{name}.png')
        (images / f'{name}.txt').write_text(text, encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))

    def add_exec(dialog):
        dialog.tag_input.setText('insert')
        dialog.position.setCurrentIndex(next(i for i in range(dialog.position.count())
                                             if dialog.position.itemData(i) == position))
        dialog.custom_index.setValue(index)
        assert dialog.custom_index.isHidden() == (position != 'custom')
        assert dialog.layout().labelForField(dialog.custom_index).isHidden() == (position != 'custom')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', add_exec)
    click_tool(window, '全画像にタグを追加', qtbot)
    assert ', '.join(tag.name for tag in window.dataset.records[0].tags) == expected_a
    assert ', '.join(tag.name for tag in window.dataset.records[1].tags) == expected_b
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'a, b, c'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'a'

    responses = iter([QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Discard])

    def discard_question(box):
        assert not box.standardButtons() & QMessageBox.StandardButton.Save
        assert box.defaultButton() == box.button(QMessageBox.StandardButton.Cancel)
        return next(responses, QMessageBox.StandardButton.Discard)

    monkeypatch.setattr('image_dataset_studio.ui.main_window.QMessageBox.exec',
                        discard_question)
    assert not window.close()
    assert window.dataset.dirty
    assert window.close()
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'a, b, c'
    assert (images / 'b.txt').read_text(encoding='utf-8') == 'a'


def test_add_dialog_tab_completes_first_dictionary_suggestion(qtbot):
    dialog = AddTagDialog(tag_names=['blue hair', 'blue eyes', 'black hair'])
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.tag_input.setFocus()
    qtbot.keyClicks(dialog.tag_input, 'solo, blu')
    assert dialog.tag_input.suggestions.popup().isVisible()
    assert dialog.tag_input.suggestions.completionModel().index(0, 0).data() == 'blue eyes'
    qtbot.keyClick(dialog.tag_input, Qt.Key.Key_Tab)
    assert dialog.tag_input.text() == 'solo, blue eyes'
    assert dialog.tags() == [Tag('solo'), Tag('blue eyes')]
    dialog.close()


def test_rename_dialog_tab_completes_model_vocabulary(qtbot):
    dialog = RenameTagDialog('タグ変更', 'old', ['blue hair', 'blue eyes', 'black hair'])
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.tag_input.setFocus()
    qtbot.keyClicks(dialog.tag_input, 'blu')
    assert dialog.tag_input.suggestions.popup().isVisible()
    qtbot.keyClick(dialog.tag_input, Qt.Key.Key_Tab)
    assert dialog.tag_name() == 'blue eyes'
    dialog.close()


def test_model_vocabulary_is_default_and_import_extends_it(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    (images / 'a.txt').write_text('old', encoding='utf-8')
    dictionary = tmp_path / 'extra.csv'
    dictionary.write_text('blue_hair,4\nextra_tag,0\n', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    started = []
    monkeypatch.setattr(window, 'load_vocabulary', lambda: started.append(True))
    window.dataset_loaded(load_dataset(images))
    assert started == [True]
    window.vocabulary_loaded({'vocabulary': {'blue hair': 'general', 'model tag': 'style'}})
    monkeypatch.setattr('image_dataset_studio.ui.main_window.QFileDialog.getOpenFileName',
                        lambda *args: (str(dictionary), ''))
    window.import_dictionary()
    assert window.vocabulary['blue hair'] == 'general'
    assert window.vocabulary['extra tag'] == 'general'
    assert {'blue hair', 'model tag', 'extra tag'} <= set(window.tag_suggestions())
    def inspect_rename(dialog):
        dialog.tag_input.suggestions.setCompletionPrefix('model')
        assert dialog.tag_input.suggestions.completionModel().index(0, 0).data() == 'model tag'
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(RenameTagDialog, 'exec', inspect_rename)
    window.rename_selected(window.tags.topLevelItem(0))
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'old'
    window.model_combo.setCurrentIndex(1)
    assert 'model tag' not in window.vocabulary
    assert window.vocabulary['extra tag'] == 'general'
    window.close()


def test_imported_dictionary_is_offered_in_both_add_dialogs(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    dictionary = tmp_path / 'danbooru.csv'
    dictionary.write_text('blue_hair,0,123,"azure_hair"\n', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.dataset_loaded(load_dataset(images))
    monkeypatch.setattr('image_dataset_studio.ui.main_window.QFileDialog.getOpenFileName',
                        lambda *args: (str(dictionary), ''))
    window.import_dictionary()
    assert window.vocabulary['blue hair'] == 'general'

    def inspect_dialog(dialog):
        dialog.tag_input.suggestions.setCompletionPrefix('blue')
        assert dialog.tag_input.suggestions.completionModel().index(0, 0).data() == 'blue hair'
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr('image_dataset_studio.ui.main_window.AddTagDialog.exec', inspect_dialog)
    click_tool(window, '選択画像にタグを追加', qtbot)
    click_tool(window, '全画像にタグを追加', qtbot)
    assert not (images / 'a.txt').exists()
    window.close()


def test_stop_button_ends_tagging_process_and_logs_status(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(images))
    assert window.device.currentText() == 'auto'
    window.allow_code.setChecked(True)
    monkeypatch.setattr('image_dataset_studio.workers.TaggingJob._command', lambda _self: (
        Path(sys.executable), ['-c', 'import time; time.sleep(60)']))
    window.generate()
    assert window.job.parent() is window
    qtbot.waitUntil(lambda: window.job.process.state().name == 'Running', timeout=5000)
    with qtbot.waitSignal(window.job.finished, timeout=5000):
        window.cancel_action.trigger()
    assert window.job is None
    assert '停止しました' in window.log_view.toPlainText()
    assert not (images / 'a.txt').exists()
    qtbot.wait(50)
    assert window.isVisible()
    assert window.centralWidget().isEnabled()
    window.generate()
    qtbot.waitUntil(lambda: window.job.process.state().name == 'Running', timeout=5000)
    with qtbot.waitSignal(window.job.finished, timeout=5000):
        window.cancel_action.trigger()
    assert window.isVisible()
    assert window.job is None
    window.close()


@pytest.mark.parametrize('backend', ['wd', 'qwen3_vl'])
def test_stop_keeps_completed_image_result(qtbot, tmp_path, monkeypatch, backend):
    config = tmp_path / 'settings'
    config.mkdir()
    image = tmp_path / 'a.png'
    pending_image = tmp_path / 'b.png'
    Image.new('RGB', (20, 20)).save(image)
    Image.new('RGB', (20, 20)).save(pending_image)
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    window.dataset_loaded(load_dataset(tmp_path))
    index = next(i for i in range(window.model_combo.count())
                 if window.model_combo.itemData(i).backend == backend)
    window.model_combo.setCurrentIndex(index)
    window.generation_scope.setCurrentIndex(2)
    spec = ModelSpec('u/m', backend)
    job = TaggingJob(spec, [image, pending_image], {}, 'cpu', True, False)
    event = {'type': 'image_result', 'path': str(image)}
    if backend == 'wd':
        event['tags'] = [{'name': 'finished', 'category': 'general'}]
    else:
        event.update(caption='A completed caption, with punctuation.', source='u/m@abc')
    script = ('import sys, time; from pathlib import Path; '
              'Path(sys.argv[1]).write_text(sys.argv[2] + "\\n", encoding="utf-8"); '
              'time.sleep(60)')
    monkeypatch.setattr(job, '_command', lambda: (
        Path(sys.executable), ['-c', script, str(job.events_path), json.dumps(event)]))
    monkeypatch.setattr('image_dataset_studio.ui.main_window.tagging_job',
                        lambda *args, **kwargs: job)
    window.generate()
    qtbot.waitUntil(lambda: job.events_path.exists() and job.events_path.stat().st_size > 0,
                    timeout=5000)
    with qtbot.waitSignal(job.finished, timeout=5000):
        window.cancel_action.trigger()
    record = window.dataset.by_path[image]
    pending = window.dataset.by_path[pending_image]
    if backend == 'wd':
        assert [tag.name for tag in record.tags] == ['finished']
    else:
        assert record.caption == 'A completed caption, with punctuation.'
        assert record.caption_source == 'u/m@abc'
    assert record.dirty
    assert not pending.dirty
    assert not image.with_suffix('.txt').exists()
    assert window.save()
    assert image.with_suffix('.txt').exists()
    assert not pending_image.with_suffix('.txt').exists()
    window.close()


def test_process_log_starts_at_minimum_height(qtbot, tmp_path):
    config = tmp_path / 'settings'
    config.mkdir()
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(
        lambda: window.log_dock.height() == window.log_dock.minimumSizeHint().height(),
        timeout=3000,
    )
    assert window.log_view.height() == window.log_view.minimumHeight()
    preview = window.left_columns.widget(1)
    selected = window.splitter.widget(1)
    global_tabs = window.splitter.widget(2)
    assert abs(window.log_dock.mapToGlobal(window.log_dock.rect().topRight()).x() -
               preview.mapToGlobal(preview.rect().topRight()).x()) <= 1
    assert abs(selected.mapToGlobal(selected.rect().bottomLeft()).y() -
               window.log_dock.mapToGlobal(window.log_dock.rect().bottomLeft()).y()) <= 1
    assert abs(global_tabs.mapToGlobal(global_tabs.rect().bottomLeft()).y() -
               window.log_dock.mapToGlobal(window.log_dock.rect().bottomLeft()).y()) <= 1
    window.close()


def test_language_switch_updates_ui_and_keeps_unsaved_tags(qtbot, tmp_path, monkeypatch):
    config = tmp_path / 'settings'
    config.mkdir()
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 20)).save(images / 'a.png')
    (images / 'a.txt').write_text('old', encoding='utf-8')
    window = MainWindow(config)
    qtbot.addWidget(window)
    assert window.dataset is None
    assert not window.caption_editor.isEnabled()
    assert window.caption_editor.placeholderText() == '画像を1枚選択すると自然文を編集できます'
    window.language_actions['EN'].trigger()
    assert window.caption_editor.placeholderText() == 'Select one image to edit its caption'
    assert window.dataset is None and not window.caption_editor.isEnabled()
    window.language_actions['JP'].trigger()
    assert window.caption_editor.placeholderText() == '画像を1枚選択すると自然文を編集できます'
    assert window.dataset is None and not window.caption_editor.isEnabled()
    window.dataset_loaded(load_dataset(images))
    record = window.dataset.records[0]
    window.dataset.transform('test', [record], lambda tags: tags + [Tag('new')])
    window.refresh_all()
    selected_path = window.selected_records()[0].path

    assert window.language_button.text() == 'LANGUAGE'
    assert window.language_button.popupMode().name == 'InstantPopup'
    assert [action.text() for action in window.language_menu.actions()] == ['日本語', 'English']
    assert window.language_actions['JP'].isChecked()
    window.language_actions['EN'].trigger()
    assert window.language == 'EN'
    assert window.language_actions['EN'].isChecked()
    assert window.actions[0].text() == 'Open Folder'
    assert window.global_tabs.tabText(0) == 'All Image Tags'
    assert window.tags.headerItem().text(1) == 'Filename'
    assert window.exclude_categories['character'].text() == 'Exclude Character from generated tags'
    assert window.image_model.data(window.image_model.index(0)).endswith('2 tags')
    assert 'Unsaved 1 images' in window.statusBar().currentMessage()
    assert window.selected_records()[0].path == selected_path
    assert [tag.name for tag in record.tags] == ['old', 'new']
    assert (images / 'a.txt').read_text(encoding='utf-8') == 'old'
    dialog = AddTagDialog(global_positions=True)
    qtbot.addWidget(dialog)
    assert [dialog.position.itemText(i) for i in range(dialog.position.count())] == [
        'Start', 'End', 'Middle', 'Custom'
    ]
    dialog.close()

    window.language_actions['JP'].trigger()
    assert window.actions[0].text() == 'フォルダーを開く'
    window.language_actions['EN'].trigger()
    monkeypatch.setattr(window, 'confirm_unsaved', lambda: True)
    window.close()
    reopened = MainWindow(config)
    qtbot.addWidget(reopened)
    assert reopened.language == 'EN'
    assert reopened.actions[0].text() == 'Open Folder'
    reopened.close()
