"""Small vector icons for the two tag toolbars."""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QToolButton


def icon(kind: str) -> QIcon:
    pixmap = QPixmap(28, 28)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#198754" if kind == "add" else "#c6454e" if kind == "delete" else
                   "#8c9ab0" if kind == "clear" else "#3976b6")
    painter.setPen(QPen(color, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                        Qt.PenJoinStyle.RoundJoin))
    if kind == "add":
        painter.drawLine(14, 5, 14, 23)
        painter.drawLine(5, 14, 23, 14)
    elif kind in {"delete", "clear"}:
        painter.drawLine(7, 7, 21, 21)
        painter.drawLine(21, 7, 7, 21)
    elif kind == "edit":
        painter.drawLine(7, 21, 18, 10)
        painter.drawLine(10, 23, 21, 12)
        painter.drawLine(7, 21, 10, 23)
        painter.drawLine(17, 9, 20, 6)
        painter.drawLine(20, 6, 23, 9)
        painter.drawLine(23, 9, 20, 12)
    elif kind == "replace":
        painter.drawLine(5, 9, 22, 9)
        painter.drawLine(17, 5, 22, 9)
        painter.drawLine(17, 13, 22, 9)
        painter.drawLine(23, 19, 6, 19)
        painter.drawLine(11, 15, 6, 19)
        painter.drawLine(11, 23, 6, 19)
    elif kind == "sort":
        painter.drawLine(9, 22, 9, 6)
        painter.drawLine(5, 10, 9, 6)
        painter.drawLine(13, 10, 9, 6)
        painter.drawLine(19, 6, 19, 22)
        painter.drawLine(15, 18, 19, 22)
        painter.drawLine(23, 18, 19, 22)
    elif kind == "copy":
        painter.drawRoundedRect(5, 5, 13, 14, 2, 2)
        painter.drawRoundedRect(10, 10, 13, 14, 2, 2)
    elif kind == "paste":
        painter.drawRoundedRect(5, 6, 18, 18, 2, 2)
        painter.drawLine(10, 4, 18, 4)
        painter.drawLine(10, 9, 18, 9)
        painter.drawLine(9, 15, 19, 15)
        painter.drawLine(9, 19, 17, 19)
    elif kind == "character":
        painter.drawEllipse(5, 5, 9, 9)
        painter.drawArc(3, 15, 15, 10, 0, 180 * 16)
    elif kind == "copyright":
        painter.drawLine(4, 7, 13, 9)
        painter.drawLine(13, 9, 23, 7)
        painter.drawLine(4, 7, 4, 19)
        painter.drawLine(4, 19, 13, 21)
        painter.drawLine(13, 9, 13, 21)
        painter.drawLine(13, 21, 23, 19)
        painter.drawLine(23, 7, 23, 19)
    elif kind == "style":
        painter.drawLine(13, 4, 13, 22)
        painter.drawLine(4, 13, 22, 13)
        painter.drawLine(7, 7, 19, 19)
        painter.drawLine(19, 7, 7, 19)
    elif kind == "artist":
        painter.drawLine(7, 21, 19, 6)
        painter.drawLine(11, 23, 23, 8)
        painter.drawLine(19, 6, 23, 8)
        painter.drawEllipse(4, 20, 7, 5)
    elif kind == "meta":
        painter.drawRect(5, 4, 17, 20)
        painter.drawLine(8, 9, 18, 9)
        painter.drawLine(8, 13, 18, 13)
        painter.drawLine(8, 17, 15, 17)
    elif kind == "filter":
        painter.drawLine(4, 6, 24, 6)
        painter.drawLine(4, 6, 12, 15)
        painter.drawLine(24, 6, 16, 15)
        painter.drawLine(12, 15, 12, 23)
        painter.drawLine(16, 15, 16, 20)
        painter.drawLine(12, 23, 16, 20)
    elif kind == "transfer":
        painter.drawLine(23, 14, 6, 14)
        painter.drawLine(6, 14, 13, 7)
        painter.drawLine(6, 14, 13, 21)
    elif kind == "refresh":
        painter.drawArc(5, 5, 18, 18, 30 * 16, 290 * 16)
        painter.drawLine(21, 5, 22, 13)
        painter.drawLine(22, 13, 16, 10)
    elif kind == "text":
        painter.drawLine(6, 7, 22, 7)
        painter.drawLine(9, 14, 19, 14)
        painter.drawLine(6, 21, 22, 21)
    if kind in {"character", "copyright", "style", "artist", "meta"}:
        painter.setPen(QPen(QColor("#c6454e"), 2.6, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawLine(19, 18, 24, 23)
        painter.drawLine(24, 18, 19, 23)
    painter.end()
    return QIcon(pixmap)


def tool_button(kind: str, label: str, callback, parent=None) -> QToolButton:
    button = QToolButton(parent)
    button.setIcon(icon(kind))
    button.setIconSize(QSize(22, 22))
    button.setToolTip(label)
    button.setAccessibleName(label)
    button.setFixedSize(30, 30)
    button.clicked.connect(callback)
    return button
