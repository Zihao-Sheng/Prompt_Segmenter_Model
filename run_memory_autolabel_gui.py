from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from PySide6 import QtWidgets
    from src.memory_autolabel.gui.main_window import MemoryAutolabelWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MemoryAutolabelWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
