"""Render the editable SVG into the PNG and Windows ICO used by the app.

Run with the project's Python environment: python scripts/render_app_icon.py
"""

import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QBuffer, QIODevice, Qt
from PyQt6.QtGui import QGuiApplication, QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(renderer: QSvgRenderer, size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return image


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance() or QGuiApplication([])
    assets = Path(__file__).resolve().parents[1] / "src" / "image_dataset_studio" / "assets"
    renderer = QSvgRenderer(str(assets / "app_icon.svg"))
    if not renderer.isValid():
        raise ValueError("Invalid app_icon.svg")
    if not render(renderer, 1024).save(str(assets / "app_icon.png"), "PNG"):
        raise OSError("Could not save app_icon.png")
    frames = []
    for size in ICON_SIZES:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not render(renderer, size).save(buffer, "PNG"):
            raise OSError(f"Could not render {size}px ICO frame")
        frames.append(Image.open(BytesIO(bytes(buffer.data()))).convert("RGBA"))
    frames[-1].save(
        assets / "app_icon.ico", format="ICO",
        sizes=[(size, size) for size in ICON_SIZES], append_images=frames[:-1],
    )
    app.processEvents()
    print("Updated app_icon.png (1024px) and app_icon.ico (16-256px) from app_icon.svg")


if __name__ == "__main__":
    main()
