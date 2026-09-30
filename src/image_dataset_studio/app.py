from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QStandardPaths
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox

from .translations import msg, tr


def self_check() -> int:
    """Dependency/import check usable from a frozen exe, without model downloads."""
    import einops
    import onnxruntime
    import timm
    import torch
    import torchvision
    import transformers
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        BaseImageProcessor,
        Florence2ForConditionalGeneration,
        LlavaForConditionalGeneration,
        Pipeline,
        Qwen3VLForConditionalGeneration,
    )

    from .captioners import Captioner
    from .taggers import camie_input, wd_input
    from .ui.main_window import MainWindow

    assert all([AutoModel, AutoModelForCausalLM, BaseImageProcessor, Captioner,
                Florence2ForConditionalGeneration, LlavaForConditionalGeneration,
                Pipeline, Qwen3VLForConditionalGeneration, camie_input, wd_input])
    result = {"torch": torch.__version__, "torchvision": torchvision.__version__,
              "einops": einops.__version__,
              "timm": timm.__version__, "transformers": transformers.__version__,
              "onnxruntime": onnxruntime.__version__, "cuda": torch.cuda.is_available()}
    # Exercise the packaged Qt plugin and UI constructors without displaying a window.
    from tempfile import TemporaryDirectory

    app = QApplication(["ImageDatasetStudio", "-platform", "offscreen"])
    with TemporaryDirectory(prefix="ids-check-") as config:
        window = MainWindow(Path(config))
        window.close()
        app.processEvents()
        result["qt_ui_constructed"] = True
    if "--report" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--report") + 1])
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--tagging-worker":
        from .tagging_worker import run

        return run(*(Path(path) for path in sys.argv[2:5]))
    if "--self-check" in sys.argv:
        try:
            return self_check()
        except Exception:
            import traceback

            if "--report" in sys.argv:
                Path(sys.argv[sys.argv.index("--report") + 1]).write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            return 1
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Image Dataset Studio")
    app.setOrganizationName("ImageDatasetStudio")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "app_icon.png")))
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in {
        QPalette.ColorRole.Window: "#1d2532", QPalette.ColorRole.WindowText: "#e4eaf4",
        QPalette.ColorRole.Base: "#151d29", QPalette.ColorRole.AlternateBase: "#222e3e",
        QPalette.ColorRole.Text: "#e4eaf4", QPalette.ColorRole.Button: "#2b394d",
        QPalette.ColorRole.ButtonText: "#e4eaf4", QPalette.ColorRole.Highlight: "#3265a5",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#2b394d",
        QPalette.ColorRole.ToolTipText: "#ffffff", QPalette.ColorRole.PlaceholderText: "#8c9ab0",
    }.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet("QPushButton { padding: 6px 9px; } QLineEdit { padding: 5px; } "
                     "QListView::item { padding: 4px; } QToolBar { spacing: 8px; padding: 5px; }")
    config = Path(os.environ.get("IDS_CONFIG_DIR") or QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation))
    config.mkdir(parents=True, exist_ok=True)
    try:
        window = MainWindow(config)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        box = QMessageBox(QMessageBox.Icon.Critical, tr(msg.dialog_StartupError),
                          tr(msg.error_StartupConfig, config=config, error=exc),
                          QMessageBox.StandardButton.Ok)
        box.button(QMessageBox.StandardButton.Ok).setText(tr(msg.dialog_Ok))
        box.exec()
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
