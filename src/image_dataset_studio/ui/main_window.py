from __future__ import annotations

import csv
import random
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QEvent, QSettings, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import (
    CATEGORIES,
    Dataset,
    RecordState,
    Tag,
    compile_filter,
    insert_tags,
    parse_tags,
    remove_category,
    rename_tags,
    tag_counts,
    tag_key,
    unique_tags,
)
from ..models import PIXAI, ModelRegistry
from ..storage import enrich_tags, parse_content, save_dataset
from ..translations import msg, set_language, tr
from ..workers import Job, TaggingJob, inspect_job, load_job, tagging_job
from .icons import tool_button
from .widgets import (
    AddTagDialog,
    ImageListModel,
    Preview,
    RenameTagDialog,
    SortTagsDialog,
    TagGridDelegate,
    TagInput,
    TagTree,
    TextPromptDialog,
    dialog_buttons,
)

TAG_CATEGORY_COLORS = {
    "general": "#dde7f2", "character": "#e5b3ff", "copyright": "#ffca85",
    "style": "#84d8ff", "artist": "#ff9fb9", "meta": "#a8e6a1",
    "rating": "#ffe48a", "year": "#abbcff", "unknown": "#a8b1bf",
}
MODEL_KIND_COLORS = {"tags": "#9cc9ff", "caption": "#b9e6a3"}


def tag_categories(records, vocabulary):
    """Prefer saved/model classifications, then use the loaded dictionary for unknown tags."""
    categories = {}
    for record in records:
        for tag in record.tags:
            if tag.category != "unknown":
                categories.setdefault(tag_key(tag.name), tag.category)
    for record in records:
        for tag in record.tags:
            key = tag_key(tag.name)
            categories.setdefault(key, vocabulary.get(key, "unknown"))
    return categories


class CurrentPageStack(QStackedWidget):
    def fitCurrentPage(self):
        page = self.currentWidget()
        if page is None:
            return
        page.layout().activate()
        height = page.sizeHint().height()
        if page.hasHeightForWidth() and self.contentsRect().width() > 0:
            height = max(height, page.heightForWidth(self.contentsRect().width()))
        if self.minimumHeight() != height or self.maximumHeight() != height:
            self.setFixedHeight(height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            QTimer.singleShot(0, self.fitCurrentPage)


class MainWindow(QMainWindow):
    def __init__(self, config_dir: Path):
        super().__init__()
        self.resize(1480, 900)
        self.settings = QSettings(str(config_dir / "settings.ini"), QSettings.Format.IniFormat)
        self.language = self.settings.value("language", "JP")
        if self.language not in {"JP", "EN"}:
            self.language = "JP"
        set_language(self.language)
        self._language_callbacks = []
        self.registry = ModelRegistry(config_dir / "models.json")
        self.dataset: Dataset | None = None
        self.job = None
        self.vocabulary = {}
        self.additional_vocabulary = {}
        self._vocabulary_model_key = None
        self._loaded_vocabulary_model_key = None
        self._pending_vocabulary_load = False
        self._caption_edit_session = 0
        self._updating_caption_editor = False
        self._sort_mode = 0
        self._nl_settings = {}
        self._current_model_spec = None
        self.refreshing = False
        self._log_initial_size_set = False
        self._history_shortcuts_installed = False
        self.build_ui()
        self.build_toolbar()
        self.populate_models(self.settings.value("model", PIXAI))
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            if not self.isMaximized() and self.height() > 900:
                self.resize(self.width(), 900)
        self.update_status()

    def bind(self, widget, entry, setter="setText"):
        def update():
            getattr(widget, setter)(tr(entry))
        self._language_callbacks.append(update)
        update()
        return widget

    def label(self, entry):
        return self.bind(QLabel(), entry)

    def button(self, entry, callback, layout):
        button = self.bind(QPushButton(), entry)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def bind_combo(self, combo, entries):
        for entry in entries:
            index = combo.count()
            combo.addItem(tr(entry))
            self._language_callbacks.append(
                lambda c=combo, i=index, e=entry: c.setItemText(i, tr(e)))

    def tool(self, kind, entry, callback, parent):
        button = tool_button(kind, tr(entry), callback, parent)
        self._language_callbacks.append(
            lambda b=button, e=entry: (b.setToolTip(tr(e)), b.setAccessibleName(tr(e))))
        return button

    def change_language(self, language):
        if language == self.language:
            return
        self.language = language
        set_language(self.language)
        self.settings.setValue("language", self.language)
        self.language_actions[language].setChecked(True)
        for update in self._language_callbacks:
            update()
        self.preview.retranslate()
        self.update_model_info()
        self.settings_stack.fitCurrentPage()
        if self.dataset:
            self.refresh_all()
        else:
            self.update_status()

    def build_toolbar(self):
        toolbar = self.bind(QToolBar(), msg.toolbar_Operations, "setWindowTitle")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.reload_menu = QMenu(self)
        self.reload_button = self.bind(QToolButton(), msg.menu_Reload)
        self.reload_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.reload_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.reload_button.setMenu(self.reload_menu)
        self.reload_button.setEnabled(False)
        self.reload_actions = {}
        for mode, entry in (("caption", msg.action_ReloadCaption),
                            ("tags", msg.action_ReloadTags),
                            ("mixed", msg.action_ReloadMixed)):
            action = self.bind(QAction(self.reload_menu), entry)
            action.triggered.connect(lambda _checked, value=mode: self.reload_dataset(value))
            action.setEnabled(False)
            self.reload_menu.addAction(action)
            self.reload_actions[mode] = action
        self.actions = []
        for entry, shortcut, callback in [
            (msg.action_OpenFolder, "Ctrl+O", self.open_folder),
            (msg.action_SaveAll, "Ctrl+S", self.save),
            (msg.action_Undo, "Ctrl+Z", self.undo),
            (msg.action_Redo, "Ctrl+Y", self.redo),
            (msg.action_ImportDictionary, "", self.import_dictionary),
            (msg.action_Help, "F1", self.help),
        ]:
            action = self.bind(QAction(self), entry)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            toolbar.addAction(action)
            self.actions.append(action)
            if entry == msg.action_Redo:
                toolbar.addWidget(self.reload_button)
        self.cancel_action = self.bind(QAction(self), msg.action_Stop)
        self.cancel_action.triggered.connect(self.cancel_job)
        self.cancel_action.setEnabled(False)
        toolbar.addAction(self.cancel_action)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    def build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        header_row = QHBoxLayout()
        header = self.label(msg.label_AppTitle)
        header.setStyleSheet("font-size: 20px; font-weight: 600; padding: 8px 0;")
        header_row.addWidget(header)
        header_row.addStretch()
        self.language_button = self.bind(QToolButton(), msg.button_Language)
        self.bind(self.language_button, msg.tooltip_Language, "setToolTip")
        self.language_button.setMinimumWidth(140)
        self.language_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.language_menu = QMenu(self.language_button)
        language_group = QActionGroup(self.language_menu)
        language_group.setExclusive(True)
        self.language_actions = {}
        for language, entry in (("JP", msg.menu_LanguageJapanese),
                                ("EN", msg.menu_LanguageEnglish)):
            action = self.bind(QAction(self.language_menu), entry)
            action.setCheckable(True)
            action.setChecked(language == self.language)
            action.triggered.connect(
                lambda _checked, code=language: self.change_language(code))
            language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[language] = action
        self.language_button.setMenu(self.language_menu)
        header_row.addWidget(self.language_button)
        outer.addLayout(header_row)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.bind(self.search, msg.placeholder_Search, "setPlaceholderText")
        self.search.setClearButtonEnabled(True)
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(250)
        self.search.textChanged.connect(lambda: self.search_timer.start())
        self.search_timer.timeout.connect(self.refresh_images)
        row.addWidget(self.search, 1)
        self.untagged = self.bind(QCheckBox(), msg.checkbox_Untagged)
        self.untagged.toggled.connect(self.refresh_images)
        row.addWidget(self.untagged)
        self.recursive = self.bind(QCheckBox(), msg.checkbox_Recursive)
        self.recursive.setChecked(True)
        row.addWidget(self.recursive)
        outer.addLayout(row)
        self.filter_error = QLabel()
        self.filter_error.setStyleSheet("color: #ffbdad")
        self.filter_error.hide()
        outer.addWidget(self.filter_error)
        self.splitter = QSplitter()
        outer.addWidget(self.splitter, 1)
        self.left_stack = QSplitter(Qt.Orientation.Vertical)
        self.left_columns = QSplitter(Qt.Orientation.Horizontal)
        self.left_stack.addWidget(self.left_columns)
        self.splitter.addWidget(self.left_stack)

        left = QWidget()
        left.setObjectName("imageListPane")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.image_count = self.label(msg.label_ImageCountEmpty)
        ll.addWidget(self.image_count)
        self.image_model = ImageListModel(self)
        self.images = QListView()
        self.images.setModel(self.image_model)
        self.images.setIconSize(QSize(112, 84))
        self.images.setUniformItemSizes(True)
        self.images.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.images.selectionModel().selectionChanged.connect(self.selection_changed)
        ll.addWidget(self.images, 1)
        left_buttons = QHBoxLayout()
        self.button(msg.button_SelectVisible, self.images.selectAll, left_buttons)
        self.button(msg.button_InvertSelection, self.invert_selection, left_buttons)
        ll.addLayout(left_buttons)
        self.left_columns.addWidget(left)

        preview_pane = QWidget()
        preview_pane.setObjectName("previewPane")
        pl = QVBoxLayout(preview_pane)
        pl.setContentsMargins(4, 0, 4, 0)
        pl.addWidget(self.label(msg.label_Preview))
        self.preview = Preview()
        pl.addWidget(self.preview, 1)
        self.left_columns.addWidget(preview_pane)

        selected_pane = QWidget()
        selected_pane.setObjectName("selectedTagsPane")
        ml = QVBoxLayout(selected_pane)
        ml.setContentsMargins(4, 0, 4, 0)
        ml.setSpacing(4)
        selection_header = QHBoxLayout()
        self.selection_label = self.label(msg.label_SelectedTagsEmpty)
        selection_header.addWidget(self.selection_label)
        selection_header.addStretch()
        self.clear_tags_button = self.tool(
            "clear", msg.tool_ClearImageTags, self.clear_image_tags, selected_pane)
        selection_header.addWidget(self.clear_tags_button)
        self.clear_tags_button.setEnabled(False)
        ml.addLayout(selection_header)
        scope_row = QHBoxLayout()
        self.common = self.bind(QCheckBox(), msg.checkbox_CommonOnly)
        self.common.toggled.connect(self.refresh_tags)
        scope_row.addWidget(self.common)
        scope_row.addStretch()
        ml.addLayout(scope_row)
        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(0, 0, 0, 0)
        self.tags = TagTree()
        self.tags.setItemDelegate(TagGridDelegate(self.tags))
        self.tags.setHeaderLabels([tr(msg.header_Tag), tr(msg.header_Filename)])
        self._language_callbacks.append(lambda: self.tags.setHeaderLabels(
            [tr(msg.header_Tag), tr(msg.header_Filename)]))
        self.tags.setRootIsDecorated(False)
        self.tags.setColumnWidth(0, 230)
        self.tags.header().setStretchLastSection(True)
        self.tags.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tags.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tags.orderDropped.connect(self.reorder_tags)
        self.tags.itemDoubleClicked.connect(self.rename_selected)
        tag_row.addWidget(self.tags, 1)
        selected_tools = QVBoxLayout()
        selected_tools.setSpacing(2)
        for kind, entry, callback in [
            ("add", msg.tool_SelectedAdd, self.add_selected_tag_dialog),
            ("delete", msg.tool_SelectedDelete, self.delete_tags),
            ("edit", msg.tool_SelectedRename, self.rename_selected_button),
            ("replace", msg.tool_SelectedReplace, self.replace_dialog),
            ("text", msg.tool_SelectedText, self.edit_text),
            ("sort", msg.tool_SelectedSort, self.sort_tags),
            ("character", msg.tool_SelectedRemoveCharacter, self.strip_characters),
            ("copyright", msg.tool_SelectedRemoveCopyright,
             lambda: self.strip_category("copyright")),
            ("style", msg.tool_SelectedRemoveStyle, lambda: self.strip_category("style")),
            ("artist", msg.tool_SelectedRemoveArtist, lambda: self.strip_category("artist")),
            ("meta", msg.tool_SelectedRemoveMeta, lambda: self.strip_category("meta")),
            ("copy", msg.tool_SelectedCopy, self.copy_tags),
            ("paste", msg.tool_SelectedPaste, self.paste_tags),
        ]:
            selected_tools.addWidget(self.tool(kind, entry, callback, selected_pane))
        selected_tools.addStretch()
        tag_row.addLayout(selected_tools)
        ml.addLayout(tag_row, 1)
        caption_header = QHBoxLayout()
        self.caption_label = self.label(msg.label_Caption)
        caption_header.addWidget(self.caption_label)
        caption_header.addStretch()
        self.clear_caption_button = self.tool(
            "clear", msg.tool_ClearCaption, self.clear_caption, selected_pane)
        caption_header.addWidget(self.clear_caption_button)
        self.clear_caption_button.setEnabled(False)
        ml.addLayout(caption_header)
        self.caption_editor = QPlainTextEdit()
        caption_palette = self.caption_editor.palette()
        for role, color in {
            QPalette.ColorRole.Base: "#2b3038",
            QPalette.ColorRole.Text: "#8c9ab0",
            QPalette.ColorRole.PlaceholderText: "#8c9ab0",
        }.items():
            caption_palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))
        self.caption_editor.setPalette(caption_palette)
        def update_caption_placeholder():
            self.caption_editor.setPlaceholderText(tr(msg.placeholder_Caption))
            self.caption_editor.viewport().update()
        self._language_callbacks.append(update_caption_placeholder)
        update_caption_placeholder()
        self.caption_editor.setFixedHeight(90)
        self.caption_editor.textChanged.connect(self.caption_text_changed)
        ml.addWidget(self.caption_editor)
        self.interpret_button = self.bind(QToolButton(), msg.button_InterpretOne)
        self.interpret_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.interpret_menu = QMenu(self.interpret_button)
        for mode, entry in (("caption", msg.button_InterpretCaption),
                            ("tags", msg.button_InterpretTags),
                            ("mixed", msg.button_InterpretMixed)):
            action = self.bind(QAction(self.interpret_menu), entry)
            action.triggered.connect(lambda _checked, value=mode: self.interpret_selected(value))
            self.interpret_menu.addAction(action)
        self.interpret_button.setMenu(self.interpret_menu)
        caption_header.addWidget(self.interpret_button)
        self.caption_editor.setEnabled(False)
        self.interpret_button.setEnabled(False)
        self.splitter.addWidget(selected_pane)

        self.global_tabs = QTabWidget()
        self.global_tabs.setObjectName("globalTagsPane")
        stats = QWidget()
        sl = QVBoxLayout(stats)
        self.tag_search = QLineEdit()
        self.bind(self.tag_search, msg.placeholder_GlobalSearch, "setPlaceholderText")
        self.tag_search.textChanged.connect(self.refresh_stats)
        sl.addWidget(self.tag_search)
        sl.addWidget(self.label(msg.label_GlobalTagsHint))
        global_row = QHBoxLayout()
        global_row.setContentsMargins(0, 0, 0, 0)
        self.stats = QListWidget()
        self.stats.setItemDelegate(TagGridDelegate(self.stats))
        self.stats.setSpacing(0)
        self.stats.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.stats.itemDoubleClicked.connect(self.filter_tag)
        global_row.addWidget(self.stats, 1)
        global_tools = QVBoxLayout()
        global_tools.setSpacing(2)
        for kind, entry, callback in [
            ("refresh", msg.tool_GlobalRefresh, self.refresh_stats),
            ("add", msg.tool_GlobalAdd, self.add_global_tag_dialog),
            ("delete", msg.tool_GlobalDelete, self.delete_global_tags),
            ("edit", msg.tool_GlobalRename, self.rename_global_tag),
            ("character", msg.tool_GlobalRemoveCharacter, self.strip_global_characters),
            ("copyright", msg.tool_GlobalRemoveCopyright,
             lambda: self.strip_global_category("copyright")),
            ("style", msg.tool_GlobalRemoveStyle, lambda: self.strip_global_category("style")),
            ("artist", msg.tool_GlobalRemoveArtist, lambda: self.strip_global_category("artist")),
            ("meta", msg.tool_GlobalRemoveMeta, lambda: self.strip_global_category("meta")),
            ("transfer", msg.tool_GlobalTransfer, self.add_stat_tag),
            ("filter", msg.tool_GlobalFilter, self.filter_selected_global_tag),
        ]:
            global_tools.addWidget(self.tool(kind, entry, callback, stats))
        global_tools.addStretch()
        global_row.addLayout(global_tools)
        sl.addLayout(global_row, 1)
        self.global_tabs.addTab(stats, tr(msg.tab_GlobalTags))
        self.global_tabs.addTab(self.build_tagger_panel(), tr(msg.tab_AutoTagging))
        self._language_callbacks.append(lambda: self.global_tabs.setTabText(0, tr(msg.tab_GlobalTags)))
        self._language_callbacks.append(lambda: self.global_tabs.setTabText(1, tr(msg.tab_AutoTagging)))
        self.splitter.addWidget(self.global_tabs)
        self.splitter.setSizes([700, 400, 320])
        self.left_columns.setSizes([280, 420])
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(1000)
        self.log_view.setMinimumHeight(110)
        self.log_dock = QDockWidget("", self)
        self.log_dock.setObjectName("processLogDock")
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.log_dock.setWidget(self.log_view)
        self.left_stack.addWidget(self.log_dock)
        self.left_stack.setStretchFactor(0, 1)
        self.left_stack.setStretchFactor(1, 0)
        self.setCentralWidget(central)

    def build_tagger_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(22)
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        layout.addWidget(self.label(msg.label_Model))
        layout.addWidget(self.model_combo)
        self.button(msg.button_AddModel, self.add_model, layout)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        layout.addWidget(self.model_info)
        device_row = QHBoxLayout()
        device_row.addWidget(self.label(msg.label_Device))
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.device)
        layout.addLayout(device_row)
        self.offline = self.bind(QCheckBox(), msg.checkbox_Offline)
        layout.addWidget(self.offline)
        self.allow_code = self.bind(QCheckBox(), msg.checkbox_AllowCode)
        self.bind(self.allow_code, msg.tooltip_AllowCode, "setToolTip")
        layout.addWidget(self.allow_code)
        self.settings_stack = CurrentPageStack()
        self.settings_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tag_page = QWidget()
        tag_layout = QVBoxLayout(tag_page)
        threshold_box = self.bind(QGroupBox(), msg.group_Thresholds, "setTitle")
        self.threshold_form = QFormLayout(threshold_box)
        self.thresholds = {}
        for category, name in CATEGORIES.items():
            if category == "unknown":
                continue
            spin = QDoubleSpinBox()
            spin.setRange(0, 1)
            spin.setSingleStep(.01)
            spin.setDecimals(2)
            self.thresholds[category] = spin
            self.threshold_form.addRow(tr(name), spin)
            self._language_callbacks.append(
                lambda s=spin, e=name: self.threshold_form.labelForField(s).setText(tr(e)))
        tag_layout.addWidget(threshold_box)
        self.button(msg.button_ResetThresholds, self.reset_thresholds, tag_layout)
        self.exclude_categories = {}
        for category in ("character", "copyright", "style", "artist", "meta"):
            checkbox = QCheckBox(tr(msg.checkbox_ExcludeCategory,
                                    category=tr(CATEGORIES[category])))
            self._language_callbacks.append(
                lambda c=checkbox, k=category: c.setText(tr(
                    msg.checkbox_ExcludeCategory, category=tr(CATEGORIES[k]))))
            self.exclude_categories[category] = checkbox
            tag_layout.addWidget(checkbox)
        self.replace_underscores = self.bind(QCheckBox(), msg.checkbox_ReplaceUnderscores)
        self.replace_underscores.setChecked(True)
        tag_layout.addWidget(self.replace_underscores)
        self.include_rating = self.bind(QCheckBox(), msg.checkbox_IncludeRating)
        tag_layout.addWidget(self.include_rating)
        tag_layout.addWidget(self.label(msg.label_MergeMode))
        self.merge_mode = QComboBox()
        self.bind_combo(self.merge_mode, [msg.option_MergeAppend, msg.option_MergeReplace,
                                          msg.option_MergeUntagged])
        tag_layout.addWidget(self.merge_mode)
        self.button(msg.button_LoadVocabulary, self.load_vocabulary, tag_layout)
        note = self.label(msg.label_TaggerNote)
        note.setWordWrap(True)
        tag_layout.addWidget(note)
        self.settings_stack.addWidget(tag_page)

        caption_page = QWidget()
        caption_layout = QVBoxLayout(caption_page)
        caption_layout.addWidget(self.label(msg.label_CaptionMode))
        self.caption_mode = QComboBox()
        self.bind_combo(self.caption_mode, [msg.option_CaptionReplace, msg.option_CaptionEmpty])
        caption_layout.addWidget(self.caption_mode)
        self.caption_task_label = self.label(msg.label_CaptionTask)
        caption_layout.addWidget(self.caption_task_label)
        self.caption_task = QComboBox()
        for task, entry in (("<CAPTION>", msg.option_CaptionShort),
                            ("<DETAILED_CAPTION>", msg.option_CaptionDetailed),
                            ("<MORE_DETAILED_CAPTION>", msg.option_CaptionMoreDetailed)):
            index = self.caption_task.count()
            self.caption_task.addItem(tr(entry), task)
            self._language_callbacks.append(
                lambda i=index, e=entry: self.caption_task.setItemText(i, tr(e)))
        self.caption_task.setCurrentIndex(2)
        caption_layout.addWidget(self.caption_task)
        self.joy_style_label = self.label(msg.label_JoyStyle)
        caption_layout.addWidget(self.joy_style_label)
        self.joy_style = QComboBox()
        self.bind_combo(self.joy_style, [msg.option_JoyStraight, msg.option_JoyDescriptive])
        caption_layout.addWidget(self.joy_style)
        self.joy_length_label = self.label(msg.label_JoyLength)
        caption_layout.addWidget(self.joy_length_label)
        self.joy_length = QComboBox()
        self.bind_combo(self.joy_length, [msg.option_JoyShort, msg.option_JoyMedium,
                                          msg.option_JoyLong])
        self.joy_length.setCurrentIndex(1)
        caption_layout.addWidget(self.joy_length)
        self.caption_prompt_label = self.label(msg.label_CaptionPrompt)
        caption_layout.addWidget(self.caption_prompt_label)
        self.caption_prompt = QPlainTextEdit()
        self.caption_prompt.setFixedHeight(90)
        caption_layout.addWidget(self.caption_prompt)
        caption_layout.addWidget(self.label(msg.label_MaxTokens))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 2048)
        self.max_tokens.setValue(256)
        caption_layout.addWidget(self.max_tokens)
        caption_note = self.label(msg.label_CaptionNote)
        caption_note.setWordWrap(True)
        caption_layout.addWidget(caption_note)
        self.settings_stack.addWidget(caption_page)
        layout.addWidget(self.settings_stack)
        layout.addWidget(self.label(msg.label_GenerationScope))
        self.generation_scope = QComboBox()
        self.bind_combo(self.generation_scope, [msg.option_ScopeSelected,
                                                msg.option_ScopeVisible, msg.option_ScopeAll])
        layout.addWidget(self.generation_scope)
        self.start_generation_button = self.button(msg.button_StartTagging, self.generate, layout)
        start_font = self.start_generation_button.font()
        start_font.setBold(True)
        self.start_generation_button.setFont(start_font)
        self.start_generation_button.setStyleSheet(
            "QPushButton { background-color: #3265a5; color: #ffffff; "
            "border: 1px solid #4c7fbd; border-radius: 5px; padding: 8px 12px; }"
            "QPushButton:hover { background-color: #3972b7; }"
            "QPushButton:pressed { background-color: #28558d; }"
            "QPushButton:disabled { background-color: #2b394d; color: #8c9ab0; "
            "border-color: #3a4759; }")
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def populate_models(self, selected):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for spec in self.registry.models:
            prefix = "[NL]" if spec.kind == "caption" else "[Danbooru]"
            self.model_combo.addItem(f"{prefix} {spec.title}", spec)
            self.model_combo.setItemData(self.model_combo.count() - 1,
                                         QColor(MODEL_KIND_COLORS[spec.kind]),
                                         Qt.ItemDataRole.ForegroundRole)
        index = next((i for i, m in enumerate(self.registry.models) if m.repo_id == selected), 0)
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self.model_changed()

    def model_changed(self):
        spec = self.model_combo.currentData()
        if not spec:
            return
        old = self._current_model_spec
        if old and old.kind == "caption":
            self._nl_settings[old.repo_id] = (
                self.caption_task.currentIndex(), self.joy_style.currentIndex(),
                self.joy_length.currentIndex(), self.caption_prompt.toPlainText(),
                self.max_tokens.value(), self.caption_mode.currentIndex())
        self._current_model_spec = spec
        if spec.kind == "tags":
            key = (spec.repo_id, spec.revision, spec.labels_file)
            if key != self._vocabulary_model_key:
                self._vocabulary_model_key = key
                self._loaded_vocabulary_model_key = None
                self.vocabulary = dict(self.additional_vocabulary)
                if self.dataset:
                    self.refresh_tags()
                    self.refresh_stats()
                    self.ensure_vocabulary()
        else:
            settings = self._nl_settings.get(spec.repo_id)
            if settings:
                task, style, length, prompt, tokens, mode = settings
                self.caption_task.setCurrentIndex(task)
                self.joy_style.setCurrentIndex(style)
                self.joy_length.setCurrentIndex(length)
                self.caption_prompt.setPlainText(prompt)
                self.max_tokens.setValue(tokens)
                self.caption_mode.setCurrentIndex(mode)
            else:
                self.caption_task.setCurrentIndex(2)
                self.joy_style.setCurrentIndex(0)
                self.joy_length.setCurrentIndex(1)
                self.caption_prompt.setPlainText(
                    tr(msg.default_CaptionPrompt) if spec.backend == "qwen3_vl" else "")
                self.max_tokens.setValue(256)
                self.caption_mode.setCurrentIndex(0)
        self.settings_stack.setCurrentIndex(1 if spec.kind == "caption" else 0)
        florence = spec.backend == "florence"
        joy = spec.backend == "joycaption"
        for widget in (self.caption_task_label, self.caption_task):
            widget.setVisible(florence)
        for widget in (self.joy_style_label, self.joy_style,
                       self.joy_length_label, self.joy_length):
            widget.setVisible(joy)
        for widget in (self.caption_prompt_label, self.caption_prompt):
            widget.setVisible(not florence)
        self.update_model_info()
        self.model_info.setToolTip(f"https://huggingface.co/{spec.repo_id}/tree/{spec.revision}")
        self.allow_code.setVisible(spec.backend in {"pixai", "florence"})
        self.allow_code.setChecked(False)
        if spec.kind == "tags":
            self.reset_thresholds()
        palette = self.model_combo.palette()
        for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor(MODEL_KIND_COLORS[spec.kind]))
        self.model_combo.setPalette(palette)
        self.settings.setValue("model", spec.repo_id)
        self.settings_stack.fitCurrentPage()
        panel_layout = self.settings_stack.parentWidget().layout()
        panel_layout.invalidate()
        panel_layout.activate()

    def update_model_info(self):
        spec = self.model_combo.currentData()
        if spec:
            note = (msg.label_Qwen8bResources if spec.repo_id == "Qwen/Qwen3-VL-8B-Instruct"
                    else msg.label_JoyResources if spec.backend == "joycaption" else None)
            info = tr(msg.label_ModelInfo, backend=spec.backend, revision=spec.revision[:12])
            self.model_info.setText(info + ("\n" + tr(note) if note else ""))
            self.start_generation_button.setText(tr(
                msg.button_StartCaptioning if spec.kind == "caption" else msg.button_StartTagging))

    def reset_thresholds(self):
        spec = self.model_combo.currentData()
        if not spec:
            return
        for category, spin in self.thresholds.items():
            spin.setValue(spec.thresholds.get(category, .5))
            spin.setEnabled(category in spec.thresholds)

    def selected_records(self):
        return [self.image_model.records[i.row()] for i in self.images.selectionModel().selectedIndexes()]

    def target_records(self):
        return self.selected_records()

    def require_targets(self, records=None):
        records = self.target_records() if records is None else records
        if not records:
            self.statusBar().showMessage(tr(msg.status_NoTargets), 5000)
        return records

    def open_folder(self):
        if not self.confirm_unsaved():
            return
        folder = QFileDialog.getExistingDirectory(
            self, tr(msg.dialog_OpenFolder), self.settings.value("folder", ""))
        if folder:
            self.start_job(load_job(Path(folder), self.recursive.isChecked()), self.dataset_loaded)

    def reload_dataset(self, mode):
        if not self.dataset or self.job is not None or not self.confirm_unsaved():
            return
        state = (self.search.text(), self.untagged.isChecked(),
                 self.global_tabs.currentIndex(),
                 [record.path for record in self.selected_records()])
        root = self.dataset.root
        self.start_job(load_job(root, self.recursive.isChecked(), mode),
                       lambda dataset: self.reloaded_dataset(dataset, state, mode))

    def reloaded_dataset(self, dataset, state, mode):
        self.dataset_loaded(dataset)
        search, untagged, tab, selected_paths = state
        self.search.setText(search)
        self.untagged.setChecked(untagged)
        self.global_tabs.setCurrentIndex(tab)
        self.refresh_images()
        from PyQt6.QtCore import QItemSelectionModel
        selection = self.images.selectionModel()
        selection.clearSelection()
        for row, record in enumerate(self.image_model.records):
            if record.path in selected_paths:
                selection.select(self.image_model.index(row),
                                 QItemSelectionModel.SelectionFlag.Select)
        self.selection_changed()
        count = sum(bool(record.caption) for record in dataset.records)
        tags = sum(len(record.tags) for record in dataset.records)
        boundary = (tr(msg.status_NoBoundary, count=dataset.mixed_without_boundary)
                    if mode == "mixed" else "")
        self.statusBar().showMessage(tr(msg.status_Reloaded, captions=count, tags=tags,
                                        mode=self.reload_actions[mode].text(),
                                        boundary=boundary), 8000)

    def dataset_loaded(self, dataset):
        self.dataset = dataset
        reload_enabled = self.job is None
        self.reload_button.setEnabled(reload_enabled)
        for action in self.reload_actions.values():
            action.setEnabled(reload_enabled)
        self.image_model.cache.clear()
        self.settings.setValue("folder", str(dataset.root))
        self.search.clear()
        self.untagged.setChecked(False)
        self.refresh_all()
        if self.image_model.rowCount():
            self.images.setCurrentIndex(self.image_model.index(0))
        self.ensure_vocabulary()

    def refresh_all(self):
        self.refresh_images()
        self.refresh_stats()
        self.update_status()

    def refresh_images(self):
        if not self.dataset:
            return
        try:
            predicate = compile_filter(self.search.text())
        except ValueError as exc:
            self.filter_error.setText(str(exc))
            self.filter_error.show()
            return
        self.filter_error.hide()
        selected = {r.path for r in self.selected_records()}
        current = self.images.currentIndex()
        current_path = self.image_model.records[current.row()].path if current.isValid() else None
        visible = [r for r in self.dataset.records if predicate(r) and (not self.untagged.isChecked() or not r.tags)]
        self.refreshing = True
        self.image_model.set_records(visible, self.dataset.root)
        selection = self.images.selectionModel()
        from PyQt6.QtCore import QItemSelectionModel
        for row, record in enumerate(visible):
            index = self.image_model.index(row)
            if record.path in selected:
                selection.select(index, QItemSelectionModel.SelectionFlag.Select)
            if record.path == current_path:
                selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.refreshing = False
        self.image_count.setText(tr(msg.label_ImageCount, shown=len(visible),
                                    total=len(self.dataset.records)))
        self.selection_changed()

    def selection_changed(self, *_):
        if self.refreshing:
            return
        records = self.selected_records()
        current = self.images.currentIndex()
        path = self.image_model.records[current.row()].path if records and current.isValid() else None
        self.preview.set_path(path)
        self.refresh_tags()
        self.refresh_caption()
        self.update_status()

    def refresh_caption(self):
        records = self.selected_records()
        self._caption_edit_session += 1
        self._updating_caption_editor = True
        try:
            single = len(records) == 1
            self.caption_editor.setEnabled(single)
            self.interpret_button.setEnabled(single)
            self.clear_caption_button.setEnabled(single and bool(records[0].caption))
            self.caption_editor.setPlainText(records[0].caption if single else "")
        finally:
            self._updating_caption_editor = False

    def caption_text_changed(self):
        if self._updating_caption_editor or not self.dataset:
            return
        records = self.selected_records()
        if len(records) != 1:
            return
        record = records[0]
        self.dataset.apply_captions(
            tr(msg.history_EditCaption), {record.path: (self.caption_editor.toPlainText(), None)},
            merge_key=f"caption:{self._caption_edit_session}")
        index = self.images.currentIndex()
        if index.isValid():
            self.image_model.dataChanged.emit(index, index)
        self.clear_caption_button.setEnabled(bool(record.caption))
        self.update_status()

    def clear_image_tags(self):
        records = self.selected_records()
        if self.dataset and len(records) == 1:
            self.dataset.apply(tr(msg.history_ClearImageTags), {records[0].path: []})
            self.refresh_all()

    def clear_caption(self):
        records = self.selected_records()
        if self.dataset and len(records) == 1:
            self.dataset.apply_captions(
                tr(msg.history_ClearCaption), {records[0].path: ("", None)})
            self.refresh_all()

    def interpret_selected(self, mode):
        records = self.selected_records()
        if len(records) != 1 or not self.dataset:
            return
        record = records[0]
        if record.dirty:
            box = QMessageBox(QMessageBox.Icon.Warning, tr(msg.dialog_Unsaved),
                              tr(msg.dialog_InterpretDiscard),
                              QMessageBox.StandardButton.Discard |
                              QMessageBox.StandardButton.Cancel, self)
            box.button(QMessageBox.StandardButton.Discard).setText(tr(msg.dialog_Discard))
            box.button(QMessageBox.StandardButton.Cancel).setText(tr(msg.dialog_Cancel))
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if box.exec() != QMessageBox.StandardButton.Discard:
                return
        sidecar = record.path.with_suffix(".txt")
        try:
            text = sidecar.read_text(encoding="utf-8-sig") if sidecar.exists() else ""
            tags, caption = parse_content(text, mode, record.saved_tags)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.error(str(exc))
            return
        tags = enrich_tags(tags, record.saved_tags)
        self.dataset.apply_content(tr(msg.history_InterpretCaption), {
            record.path: RecordState(tuple(tags), caption)
        })
        self.refresh_all()

    def refresh_tags(self):
        records = self.selected_records()
        counts = tag_counts(records)
        categories = tag_categories(records, self.vocabulary)
        selected_names = {item.data(0, Qt.ItemDataRole.UserRole) for item in self.tags.selectedItems()}
        self.tags.clear()
        self.tags.setColumnHidden(1, len(records) <= 1)
        representative = {t.name: t for r in records for t in r.tags}
        for name, count in counts.items():
            if self.common.isChecked() and count != len(records):
                continue
            tag = representative[name]
            category = categories.get(tag_key(name), "unknown")
            label = tr(CATEGORIES[category]) if category in CATEGORIES else category
            score = tr(msg.label_TagScore, score=tag.score) if tag.score is not None else ""
            filenames = ", ".join(r.path.name for r in records
                                  if any(tag_key(t.name) == tag_key(name) for t in r.tags))
            item = QTreeWidgetItem([tr(msg.label_TagRow, name=name, count=count,
                                      total=len(records), category=label, score=score),
                                    filenames if len(records) > 1 else ""])
            item.setData(0, Qt.ItemDataRole.UserRole, name)
            item.setToolTip(0, tag.source or tr(msg.tooltip_ManualTag))
            item.setToolTip(1, filenames)
            item.setForeground(0, QColor(TAG_CATEGORY_COLORS.get(
                category, TAG_CATEGORY_COLORS["unknown"])))
            self.tags.addTopLevelItem(item)
            item.setSelected(name in selected_names)
        self.tags.setDragEnabled(len(records) == 1)
        self.tags.setAcceptDrops(len(records) == 1)
        self.selection_label.setText(tr(msg.label_SelectedTagsCount,
                                        images=len(records), tags=len(counts)))
        self.clear_tags_button.setEnabled(len(records) == 1 and bool(records[0].tags))

    def refresh_stats(self):
        self.stats.clear()
        if not self.dataset:
            return
        counts = tag_counts(self.dataset.records)
        categories = tag_categories(self.dataset.records, self.vocabulary)
        query = self.tag_search.text().casefold()
        for name, count in counts.most_common():
            if query not in name.casefold():
                continue
            item = QListWidgetItem(f"{count:>5}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            category = categories.get(tag_key(name), "unknown")
            item.setForeground(QColor(TAG_CATEGORY_COLORS.get(
                category, TAG_CATEGORY_COLORS["unknown"])))
            item.setToolTip(tr(CATEGORIES[category]) if category in CATEGORIES else category)
            self.stats.addItem(item)

    def update_status(self):
        if not hasattr(self, "actions"):
            return
        dirty = sum(r.dirty for r in self.dataset.records) if self.dataset else 0
        folder = tr(msg.label_WindowFolder, folder=self.dataset.root.name) if self.dataset else ""
        self.setWindowTitle(tr(msg.label_WindowTitle, dirty='* ' if dirty else '', folder=folder))
        total = len(self.dataset.records) if self.dataset else 0
        self.statusBar().showMessage(tr(msg.label_Status, selected=len(self.selected_records()),
                                        total=total, dirty=dirty))

    def transform(self, label, fn, records=None):
        targets = self.require_targets(records)
        if targets:
            self.dataset.transform(label, targets, fn)
            self.refresh_all()

    def tag_suggestions(self):
        names = {tag_key(name): name for name in self.vocabulary}
        if self.dataset:
            for record in self.dataset.records:
                for tag in record.tags:
                    names.setdefault(tag_key(tag.name), tag.name)
        return names.values()

    def add_selected_tag_dialog(self):
        records = self.require_targets()
        if not records:
            return
        item = self.tags.currentItem()
        anchor = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        single = len(records) == 1
        placeholder = None
        if single:
            row = self.tags.indexOfTopLevelItem(item) + 1 if item else self.tags.topLevelItemCount()
            placeholder = QTreeWidgetItem(["", ""])
            self.tags.insertTopLevelItem(row, placeholder)
            self.tags.setCurrentItem(placeholder)
        dialog = AddTagDialog(self, show_position=not single,
                              tag_names=self.tag_suggestions())
        try:
            accepted = dialog.exec()
        finally:
            if placeholder is not None:
                row = self.tags.indexOfTopLevelItem(placeholder)
                if row >= 0:
                    self.tags.takeTopLevelItem(row)
                if item is not None:
                    self.tags.setCurrentItem(item)
        if accepted and dialog.tags():
            position = "after" if single else dialog.position.currentData()
            self.dataset.transform(
                tr(msg.history_AddSelected), records,
                lambda old: insert_tags(old, dialog.tags(), position, anchor,
                                        dialog.skip_existing.isChecked()))
            self.refresh_all()

    def add_global_tag_dialog(self):
        if not self.dataset or not self.dataset.records:
            return
        dialog = AddTagDialog(self, global_positions=True,
                              tag_names=self.tag_suggestions())
        if dialog.exec() and dialog.tags():
            self.dataset.transform(
                tr(msg.history_AddGlobal), self.dataset.records,
                lambda old: insert_tags(old, dialog.tags(), dialog.position.currentData(),
                                        skip_existing=dialog.skip_existing.isChecked(),
                                        custom_index=dialog.custom_index.value()))
            self.refresh_all()

    def selected_tag_names(self):
        return [item.data(0, Qt.ItemDataRole.UserRole) for item in self.tags.selectedItems()]

    def delete_tags(self):
        keys = {tag_key(n) for n in self.selected_tag_names()}
        if keys:
            self.transform(tr(msg.history_DeleteSelected),
                           lambda old: [t for t in old if tag_key(t.name) not in keys])

    def rename_selected(self, item):
        old = item.data(0, Qt.ItemDataRole.UserRole)
        dialog = RenameTagDialog(tr(msg.dialog_RenameTag), old, self.tag_suggestions(), self)
        if dialog.exec() and dialog.tag_name():
            self.transform(tr(msg.history_RenameSelected),
                           lambda tags: rename_tags(tags, old, dialog.tag_name()))

    def rename_selected_button(self):
        item = self.tags.currentItem()
        if item:
            self.rename_selected(item)

    def replace_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(tr(msg.dialog_ReplaceTags))
        form = QFormLayout(dialog)
        old, new = QLineEdit(), TagInput(self.tag_suggestions(), dialog)
        regex = QCheckBox(tr(msg.checkbox_Regex))
        form.addRow(tr(msg.label_Find), old)
        form.addRow(tr(msg.label_ReplaceWith), new)
        form.addRow(regex)
        buttons = dialog_buttons(QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() and old.text():
            try:
                self.transform(tr(msg.history_Replace),
                               lambda tags: rename_tags(tags, old.text(), new.text(), regex.isChecked()))
            except re.error as exc:
                self.error(str(exc))

    def edit_text(self):
        records = self.selected_records()
        if len(records) != 1:
            self.error(tr(msg.error_TextSingleImage))
            return
        record = records[0]
        dialog = QDialog(self)
        dialog.setWindowTitle(tr(msg.dialog_EditText, filename=record.path.name))
        dialog.resize(650, 380)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(", ".join(t.name for t in record.tags))
        layout.addWidget(text)
        buttons = dialog_buttons(QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            self.dataset.apply(tr(msg.history_EditText),
                               {record.path: parse_tags(text.toPlainText(), record.tags)})
            self.refresh_all()

    def sort_tags(self):
        records = self.require_targets()
        if not records:
            return
        dialog = SortTagsDialog(self._sort_mode, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mode = dialog.mode.currentIndex()
        self._sort_mode = mode
        counts = tag_counts(self.dataset.records) if self.dataset else {}
        selected = {tag_key(n) for n in self.selected_tag_names()}
        def sort(tags):
            if mode == 0:
                return sorted(tags, key=lambda t: t.name.casefold())
            if mode == 1:
                return sorted(tags, key=lambda t: -counts.get(t.name, 0))
            if mode == 2:
                random.shuffle(tags)
                return tags
            return sorted(tags, key=lambda t: tag_key(t.name) not in selected)
        self.transform(tr(msg.history_Sort), sort, records)

    def reorder_tags(self, *_):
        records = self.selected_records()
        if len(records) != 1:
            return
        record = records[0]
        by_name = {t.name: t for t in record.tags}
        tags = [by_name[self.tags.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)]
                for i in range(self.tags.topLevelItemCount())]
        self.dataset.apply(tr(msg.history_Reorder), {record.path: tags})
        # Keep the drag/drop view intact until Qt finishes dispatching the event.
        QTimer.singleShot(0, self.refresh_all)

    def strip_characters(self):
        self.strip_category("character")

    def strip_global_characters(self):
        self.strip_global_category("character")

    def strip_category(self, category):
        self.transform(tr(msg.history_RemoveCategory, category=tr(CATEGORIES[category])),
                       lambda tags: remove_category(tags, category, self.vocabulary))

    def strip_global_category(self, category):
        if self.dataset:
            self.dataset.transform(
                tr(msg.history_RemoveGlobalCategory, category=tr(CATEGORIES[category])),
                self.dataset.records,
                lambda tags: remove_category(tags, category, self.vocabulary))
            self.refresh_all()

    def copy_tags(self):
        records = self.selected_records()
        names = self.selected_tag_names() or list(tag_counts(records))
        QApplication.clipboard().setText(", ".join(names))

    def paste_tags(self):
        incoming = parse_tags(QApplication.clipboard().text())
        self.transform(tr(msg.history_Paste), lambda tags: tags + incoming)

    def filter_tag(self, item):
        name = item.data(Qt.ItemDataRole.UserRole).replace('"', '\\"')
        self.search.setText(f'tag:"{name}"')

    def filter_selected_global_tag(self):
        item = self.stats.currentItem()
        if item:
            self.filter_tag(item)

    def add_stat_tag(self):
        item = self.stats.currentItem()
        if item:
            tag = Tag(item.data(Qt.ItemDataRole.UserRole))
            self.transform(tr(msg.history_Transfer), lambda tags: tags + [tag])

    def delete_global_tags(self):
        keys = {tag_key(item.data(Qt.ItemDataRole.UserRole)) for item in self.stats.selectedItems()}
        if keys and self.dataset:
            self.transform(tr(msg.history_DeleteGlobal),
                           lambda old: [t for t in old if tag_key(t.name) not in keys],
                           self.dataset.records)

    def rename_global_tag(self):
        item = self.stats.currentItem()
        if not item or not self.dataset:
            return
        old = item.data(Qt.ItemDataRole.UserRole)
        dialog = RenameTagDialog(tr(msg.dialog_RenameGlobal), old,
                                 self.tag_suggestions(), self)
        if dialog.exec() and dialog.tag_name():
            self.transform(tr(msg.history_RenameGlobal),
                           lambda tags: rename_tags(tags, old, dialog.tag_name()), self.dataset.records)

    def invert_selection(self):
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel
        if self.image_model.rowCount():
            selection = QItemSelection(self.image_model.index(0), self.image_model.index(self.image_model.rowCount() - 1))
            self.images.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.Toggle)

    def undo(self):
        if self.dataset:
            self.dataset.undo()
            self.refresh_all()

    def redo(self):
        if self.dataset:
            self.dataset.redo()
            self.refresh_all()

    def eventFilter(self, watched, event):
        if (event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                or not isinstance(watched, QWidget) or watched.window() is not self
                or QApplication.activeModalWidget() is not None or self.job is not None):
            return super().eventFilter(watched, event)
        if event.modifiers() != Qt.KeyboardModifier.ControlModifier:
            return super().eventFilter(watched, event)
        is_redo = event.key() == Qt.Key.Key_Y
        operation = self.redo if is_redo else self.undo if event.key() == Qt.Key.Key_Z else None
        if operation is None or self._focused_text_has_history(watched, is_redo):
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
        else:
            operation()
        return True

    def _focused_text_has_history(self, widget, redo):
        while widget is not None and widget is not self:
            if isinstance(widget, QLineEdit):
                return not widget.isReadOnly() and (
                    widget.isRedoAvailable() if redo else widget.isUndoAvailable())
            if isinstance(widget, (QPlainTextEdit, QTextEdit)):
                document = widget.document()
                return not widget.isReadOnly() and (
                    document.isRedoAvailable() if redo else document.isUndoAvailable())
            widget = widget.parentWidget()
        return False

    def save(self):
        if not self.dataset:
            return True
        try:
            count = save_dataset(self.dataset)
        except (OSError, ValueError) as exc:
            self.error(str(exc))
            return False
        self.refresh_all()
        self.statusBar().showMessage(tr(msg.status_Saved, count=count), 5000)
        return True

    def confirm_unsaved(self):
        if not self.dataset or not self.dataset.dirty:
            return True
        box = QMessageBox(QMessageBox.Icon.Question, tr(msg.dialog_Unsaved),
                          tr(msg.dialog_UnsavedPrompt),
                          QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                          self)
        box.button(QMessageBox.StandardButton.Discard).setText(tr(msg.dialog_Discard))
        box.button(QMessageBox.StandardButton.Cancel).setText(tr(msg.dialog_Cancel))
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Discard

    def add_model(self):
        dialog = TextPromptDialog(tr(msg.dialog_AddModel), tr(msg.dialog_ModelUrlPrompt), self)
        if dialog.exec() and dialog.input.text().strip():
            self.start_job(inspect_job(dialog.input.text()), self.model_added)

    def model_added(self, spec):
        try:
            self.registry.add(spec)
        except OSError as exc:
            self.error(str(exc))
            return
        self.populate_models(spec.repo_id)

    def load_vocabulary(self):
        spec = self.model_combo.currentData()
        if not spec or spec.kind != "tags":
            return
        key = (spec.repo_id, spec.revision, spec.labels_file)
        self.start_job(tagging_job(spec, [], {}, "cpu", self.offline.isChecked(), False, True),
                       lambda result: self.vocabulary_loaded(result, key))

    def ensure_vocabulary(self):
        spec = self.model_combo.currentData()
        if (not self.dataset or not spec or spec.kind != "tags" or
                self._loaded_vocabulary_model_key == self._vocabulary_model_key):
            return
        if self.job is not None:
            self._pending_vocabulary_load = True
            return
        self.load_vocabulary()

    def apply_model_vocabulary(self, vocabulary, key):
        if key != self._vocabulary_model_key:
            return
        self.vocabulary = dict(self.additional_vocabulary)
        self.vocabulary.update(vocabulary)
        self._loaded_vocabulary_model_key = key
        self.refresh_tags()
        self.refresh_stats()

    def vocabulary_loaded(self, result, key=None):
        self.apply_model_vocabulary(result["vocabulary"], key or self._vocabulary_model_key)
        self.statusBar().showMessage(tr(msg.status_VocabularyLoaded,
                                        count=len(self.vocabulary)), 6000)

    def generate(self):
        if not self.dataset:
            return
        records = [self.selected_records(), self.image_model.records,
                   self.dataset.records][self.generation_scope.currentIndex()]
        records = self.require_targets(records)
        if not records:
            return
        spec = self.model_combo.currentData()
        if spec.kind == "caption":
            if self.caption_mode.currentIndex() == 1:
                records = [r for r in records if not r.caption]
            if not records:
                self.error(tr(msg.error_NoCaptionTargets))
                return
            if spec.backend == "florence" and not self.allow_code.isChecked():
                self.error(tr(msg.error_CaptionCode))
                return
            options = {"max_new_tokens": self.max_tokens.value()}
            if spec.backend == "florence":
                options["task"] = self.caption_task.currentData()
            elif spec.backend == "joycaption":
                length = (40, 80, 160)[self.joy_length.currentIndex()]
                default_prompt = (
                    f"Write a straightforward caption for this image within {length} words."
                    if self.joy_style.currentIndex() == 0 else
                    f"Write a detailed description for this image in {length} words or less.")
                options["prompt"] = self.caption_prompt.toPlainText().strip() or default_prompt
            else:
                options["prompt"] = self.caption_prompt.toPlainText().strip() or tr(
                    msg.default_CaptionPrompt)
            self.start_job(tagging_job(
                spec, [r.path for r in records], {}, self.device.currentText(),
                self.offline.isChecked(), self.allow_code.isChecked(), options=options),
                self.captions_ready)
            return
        mode = self.merge_mode.currentIndex()
        if mode == 2:
            records = [r for r in records if not r.tags]
        if not records:
            self.error(tr(msg.error_NoUntagged))
            return
        if spec.backend == "pixai" and not self.allow_code.isChecked():
            self.error(tr(msg.error_PixaiPermission))
            return
        thresholds = {c: s.value() for c, s in self.thresholds.items() if s.isEnabled()}
        excluded_categories = {category for category, checkbox in self.exclude_categories.items()
                               if checkbox.isChecked()}
        replace_underscores = self.replace_underscores.isChecked()
        rating = self.include_rating.isChecked()
        self.start_job(tagging_job(spec, [r.path for r in records], thresholds,
                                  self.device.currentText(), self.offline.isChecked(),
                                  self.allow_code.isChecked()),
                       lambda result: self.predictions_ready(
                           result, mode, excluded_categories, rating, replace_underscores))

    def predictions_ready(self, result, mode, excluded_categories, rating, replace_underscores):
        if result.get("vocabulary") is not None:
            self.apply_model_vocabulary(result["vocabulary"], self._vocabulary_model_key)
        predictions = {}
        for path, tags in result["predictions"].items():
            filtered = [t for t in tags if t.category not in excluded_categories and
                        (rating or t.category != "rating")]
            if replace_underscores:
                filtered = unique_tags(replace(t, name=t.name.replace("_", " ")) for t in filtered)
            predictions[path] = filtered
        updates = {}
        for path, incoming in predictions.items():
            if mode == 1:
                updates[path] = incoming
            else:
                known = {tag_key(t.name): t for t in incoming}
                old = [known.get(tag_key(t.name), t) for t in self.dataset.by_path[path].tags]
                updates[path] = unique_tags(old + incoming)
        changed = self.dataset.apply(tr(msg.history_AutoTag), updates)
        self.tag_search.clear()
        self.refresh_all()
        self.global_tabs.setCurrentIndex(0)
        self.append_log(tr(msg.log_Generated, changed=changed, failed=len(result['errors'])))

    def captions_ready(self, result):
        updates = {path: (caption, result.get("source"))
                   for path, caption in result.get("captions", {}).items() if caption.strip()}
        changed = self.dataset.apply_captions(tr(msg.history_AutoCaption), updates)
        self.refresh_all()
        self.append_log(tr(msg.log_CaptionsGenerated,
                           changed=changed, failed=len(result.get("errors", []))))

    def import_dictionary(self):
        file, _ = QFileDialog.getOpenFileName(
            self, tr(msg.dialog_ImportDictionary), "", tr(msg.dialog_DictionaryFilter))
        if not file:
            return
        try:
            with Path(file).open(encoding="utf-8-sig", newline="") as stream:
                if file.lower().endswith(".csv"):
                    rows = list(csv.reader(stream))
                    header = rows[0] if rows else []
                    if "name" in header and "category" in header:
                        ni, ci, rows = header.index("name"), header.index("category"), rows[1:]
                    else:
                        ni, ci = 0, 1
                    categories = {"0": "general", "1": "artist", "3": "copyright", "4": "character", "5": "meta", "9": "rating"}
                    incoming = {tag_key(r[ni]): categories.get(r[ci], r[ci] if r[ci] in CATEGORIES else "unknown")
                                for r in rows if len(r) > max(ni, ci)}
                else:
                    incoming = {tag_key(line): "unknown" for line in stream if line.strip()}
            self.additional_vocabulary.update(incoming)
            for name, category in incoming.items():
                self.vocabulary.setdefault(name, category)
            self.refresh_tags()
            self.refresh_stats()
        except (OSError, ValueError) as exc:
            self.error(str(exc))

    def start_job(self, action, callback):
        if self.job is not None:
            return
        self.centralWidget().setEnabled(False)
        for action_item in self.actions:
            action_item.setEnabled(False)
        for action_item in self.reload_actions.values():
            action_item.setEnabled(False)
        self.reload_button.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        job = action if isinstance(action, TaggingJob) else Job(action, self)
        if isinstance(job, TaggingJob):
            job.setParent(self)
        self.job = job
        job.result.connect(callback)
        job.error.connect(self.job_error)
        job.progress.connect(self.job_progress)
        job.finished.connect(self.job_finished)
        if isinstance(job, TaggingJob):
            job.log.connect(self.append_log)
        job.start()

    def job_progress(self, current, total, text):
        self.progress.setRange(0, total if current or total == 0 else 0)
        self.progress.setValue(current)
        self.statusBar().showMessage(text)
        self.append_log(text)

    def append_log(self, message):
        self.log_view.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")

    def job_error(self, message):
        self.append_log(tr(msg.log_Error, message=message))
        self.error(message)

    def job_finished(self):
        job = self.job
        self.job = None
        self.centralWidget().setEnabled(True)
        for action in self.actions:
            action.setEnabled(True)
        for action in self.reload_actions.values():
            action.setEnabled(self.dataset is not None)
        self.reload_button.setEnabled(self.dataset is not None)
        self.cancel_action.setEnabled(False)
        self.progress.hide()
        if job:
            job.deleteLater()
        if self._pending_vocabulary_load:
            self._pending_vocabulary_load = False
            self.ensure_vocabulary()

    def cancel_job(self):
        if self.job:
            self.job.requestInterruption()
            self.statusBar().showMessage(tr(msg.status_StopRequested))
            if not isinstance(self.job, TaggingJob):
                self.append_log(tr(msg.log_StopWaiting))

    def error(self, message):
        box = QMessageBox(QMessageBox.Icon.Warning, tr(msg.dialog_Error), message[:2500],
                          QMessageBox.StandardButton.Ok, self)
        box.button(QMessageBox.StandardButton.Ok).setText(tr(msg.dialog_Ok))
        box.exec()

    def help(self):
        box = QMessageBox(QMessageBox.Icon.Information, tr(msg.dialog_Help),
                          tr(msg.dialog_HelpBody), QMessageBox.StandardButton.Ok, self)
        box.button(QMessageBox.StandardButton.Ok).setText(tr(msg.dialog_Ok))
        box.exec()

    def closeEvent(self, event):
        if self.job is not None:
            self.cancel_job()
            self.statusBar().showMessage(tr(msg.status_CloseAfterStop))
            event.ignore()
            return
        if not self.confirm_unsaved():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()
        if self._history_shortcuts_installed:
            QApplication.instance().removeEventFilter(self)
            self._history_shortcuts_installed = False
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._history_shortcuts_installed:
            QApplication.instance().installEventFilter(self)
            self._history_shortcuts_installed = True
        if not self._log_initial_size_set:
            self._log_initial_size_set = True
            QTimer.singleShot(0, self._set_initial_log_size)

    def _set_initial_log_size(self):
        minimum = self.log_dock.minimumSizeHint().height()
        self.left_stack.setSizes([max(0, self.left_stack.height() - minimum), minimum])
