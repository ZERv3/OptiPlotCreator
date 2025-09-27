#!/usr/bin/env python3
"""Entry point for the inequality visualizer application."""

import sys

from PyQt5 import QtWidgets

from src.main_window import MainWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
