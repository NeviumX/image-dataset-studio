from collections import OrderedDict

from PyQt6.QtCore import QAbstractListModel, QEvent, QModelIndex, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImageReader, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTreeWidget,
)

from ..core import parse_tags
from ..translations import msg, tr


def dialog_buttons(buttons):
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr(msg.dialog_Ok))
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr(msg.dialog_Cancel))
    return buttons


class TagInput(QLineEdit):
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.suggestions = QCompleter(sorted(set(names), key=str.casefold), self)
        self.suggestions.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.suggestions.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.suggestions.setMaxVisibleItems(12)
        self.suggestions.setWidget(self)
        self.suggestions.activated[str].connect(self.complete_tag)
        self.textEdited.connect(self.show_suggestions)
        self.installEventFilter(self)

    def eventFilter(self, watched, event):
        if (watched is self and event.type() == QEvent.Type.KeyPress and
                event.key() == Qt.Key.Key_Tab and self.suggestions.popup().isVisible()):
            first = self.suggestions.completionModel().index(0, 0)
            if first.isValid():
                self.complete_tag(first.data())
                return True
        return super().eventFilter(watched, event)

    def show_suggestions(self, _text):
        start = self.text().rfind(',', 0, self.cursorPosition()) + 1
        prefix = self.text()[start:self.cursorPosition()].strip()
        if not prefix:
            self.suggestions.popup().hide()
            return
        self.suggestions.setCompletionPrefix(prefix)
        first = self.suggestions.completionModel().index(0, 0)
        if first.isValid():
            self.suggestions.complete(self.rect())
            self.suggestions.popup().setCurrentIndex(first)
        else:
            self.suggestions.popup().hide()

    def complete_tag(self, name):
        value = self.text()
        cursor = self.cursorPosition()
        start = value.rfind(',', 0, cursor) + 1
        end = value.find(',', cursor)
        end = len(value) if end < 0 else end
        leading = value[start:end][:len(value[start:end]) - len(value[start:end].lstrip())]
        self.setText(value[:start] + leading + name + value[end:])
        self.setCursorPosition(start + len(leading) + len(name))
        self.suggestions.popup().hide()


class RenameTagDialog(QDialog):
    def __init__(self, title, old_name, tag_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(390)
        form = QFormLayout(self)
        self.tag_input = TagInput(tag_names, self)
        self.tag_input.setText(old_name)
        self.tag_input.selectAll()
        form.addRow(tr(msg.label_RenameArrow, name=old_name), self.tag_input)
        buttons = dialog_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                                  QDialogButtonBox.StandardButton.Cancel))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(old_name.strip()))
        self.tag_input.textChanged.connect(
            lambda text: buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(text.strip())))

    def tag_name(self):
        return self.tag_input.text().strip()


class TextPromptDialog(QDialog):
    def __init__(self, title, prompt, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(390)
        form = QFormLayout(self)
        self.input = QLineEdit(self)
        form.addRow(prompt, self.input)
        buttons = dialog_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                                  QDialogButtonBox.StandardButton.Cancel))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class AddTagDialog(QDialog):
    def __init__(self, parent=None, *, show_position=True, global_positions=False, tag_names=()):
        super().__init__(parent)
        self.setWindowTitle(tr(msg.dialog_AddTag))
        self.setMinimumWidth(390)
        form = QFormLayout(self)
        self.position = QComboBox()
        options = ([(msg.option_PositionStart, 'start'), (msg.option_PositionEnd, 'end'),
                    (msg.option_PositionMiddle, 'middle'), (msg.option_PositionCustom, 'custom')]
                   if global_positions else
                   [(msg.option_PositionAfter, 'after'), (msg.option_PositionBefore, 'before'),
                    (msg.option_PositionStart, 'start'), (msg.option_PositionEnd, 'end')])
        for label, value in options:
            self.position.addItem(tr(label), value)
        if global_positions:
            self.position.setCurrentIndex(1)  # Keep the previous append behavior by default.
        if show_position:
            form.addRow(tr(msg.label_Position), self.position)
        self.custom_index = QSpinBox()
        self.custom_index.setRange(0, 1_000_000)
        self.custom_index.setToolTip(tr(msg.tooltip_CustomIndex))
        if global_positions:
            form.addRow(tr(msg.label_CustomIndex), self.custom_index)
            index_label = form.labelForField(self.custom_index)

            def show_custom_index():
                visible = self.position.currentData() == 'custom'
                index_label.setVisible(visible)
                self.custom_index.setVisible(visible)

            self.position.currentIndexChanged.connect(
                show_custom_index)
            show_custom_index()
        self.skip_existing = QCheckBox(tr(msg.checkbox_SkipExisting))
        self.skip_existing.setChecked(True)
        if show_position:
            form.addRow(self.skip_existing)
        self.tag_input = TagInput(tag_names, self)
        self.tag_input.setPlaceholderText(tr(msg.placeholder_TagInput))
        form.addRow(tr(msg.label_Tag), self.tag_input)
        lower = QPushButton(tr(msg.button_Lowercase))
        lower.clicked.connect(lambda: self.tag_input.setText(self.tag_input.text().lower()))
        form.addRow("", lower)
        buttons = dialog_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                                  QDialogButtonBox.StandardButton.Cancel))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.tag_input.textChanged.connect(lambda text: ok.setEnabled(bool(parse_tags(text))))
        ok.setEnabled(False)

    def tags(self):
        return parse_tags(self.tag_input.text())


class TagTree(QTreeWidget):
    """Allow row reordering without nesting a tag under another tag."""

    orderDropped = pyqtSignal()

    def dropEvent(self, event):
        if self.dropIndicatorPosition() == QAbstractItemView.DropIndicatorPosition.OnItem:
            event.ignore()
            return
        before = [self.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                  for i in range(self.topLevelItemCount())]
        super().dropEvent(event)
        QTimer.singleShot(0, lambda: self._emit_order_change(before))

    def _emit_order_change(self, before):
        after = [self.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                 for i in range(self.topLevelItemCount())]
        if before != after:
            self.orderDropped.emit()


class TagGridDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        hint.setHeight(max(26, option.fontMetrics.height() + 8))
        return hint

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        painter.save()
        painter.setPen(QPen(QColor("#40516a")))
        rect = option.rect
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.restore()


class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []
        self.root = None
        self.cache = OrderedDict()

    def set_records(self, records, root):
        self.beginResetModel()
        self.records = records
        self.root = root
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        record = self.records[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return tr(msg.model_ImageRow, dirty='● ' if record.dirty else '',
                      filename=record.path.name, count=len(record.tags),
                      caption=tr(msg.model_CaptionPresent) if record.caption else '')
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(record.path.relative_to(self.root))
        if role == Qt.ItemDataRole.DecorationRole:
            key = str(record.path)
            if key not in self.cache:
                reader = QImageReader(key)
                reader.setAutoTransform(True)
                size = reader.size()
                if size.isValid():
                    reader.setScaledSize(size.scaled(112, 84, Qt.AspectRatioMode.KeepAspectRatio))
                image = reader.read()
                self.cache[key] = QIcon(QPixmap.fromImage(image)) if not image.isNull() else QIcon()
                if len(self.cache) > 256:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None


class Preview(QLabel):
    def __init__(self):
        super().__init__(tr(msg.preview_OpenFolder))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setStyleSheet("background: #131923; border: 1px solid #344155; border-radius: 6px;")
        self.original = QPixmap()
        self.path = None
        self._folder_opened = False

    def retranslate(self):
        if self.path is None:
            self.setText(tr(msg.preview_SelectImage if self._folder_opened
                            else msg.preview_OpenFolder))
        elif self.original.isNull():
            self.setText(tr(msg.preview_ImageError, filename=self.path.name))

    def set_path(self, path):
        self._folder_opened = True
        if self.path == path:
            return
        self.path = path
        if path is None:
            self.original = QPixmap()
            self.setText(tr(msg.preview_SelectImage))
            return
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(1800, 1800, Qt.AspectRatioMode.KeepAspectRatio))
        self.original = QPixmap.fromImage(reader.read())
        if self.original.isNull():
            self.setText(tr(msg.preview_ImageError, filename=path.name))
        else:
            self.rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale()

    def rescale(self):
        if not self.original.isNull():
            self.setPixmap(self.original.scaled(self.size() - QSize(12, 12),
                           Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
