from __future__ import annotations

import io
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class FloatSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, minimum=-1000.0, maximum=1000.0, step=0.5, decimals=3, parent=None):
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setAlignment(QtCore.Qt.AlignRight)


class LatexLabel(QtWidgets.QLabel):
    def __init__(self, dpi: int = 150, min_height: int = 120, parent=None):
        super().__init__(parent)
        self._dpi = dpi
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setStyleSheet("background-color: white; border: 1px solid #d0d0d0;")

    def set_latex(self, expression: Optional[str]) -> None:
        if not expression:
            self.clear()
            return

        lines = [line.strip() for line in expression.split("\n") if line.strip()]
        if not lines:
            self.clear()
            return

        formatted = [line if (line.startswith("$") and line.endswith("$")) else f"${line}$" for line in lines]
        max_len = max(len(line) for line in formatted)
        width_in = max(2.0, 0.12 * max_len)
        height_in = max(0.6, 0.45 * len(formatted))

        try:
            fig = Figure(figsize=(width_in, height_in), dpi=self._dpi)
            fig.patch.set_facecolor("white")
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            total = len(formatted)
            for idx, line in enumerate(formatted):
                y = 1.0 - (idx + 0.5) / max(total, 1)
                ax.text(0.5, y, line, ha="center", va="center", fontsize=14)
            buffer = io.BytesIO()
            canvas.print_png(buffer)
        except Exception:
            self.clear()
            return

        buffer.seek(0)
        image = QtGui.QImage.fromData(buffer.getvalue(), "PNG")
        if image.isNull():
            self.clear()
            return

        pixmap = QtGui.QPixmap.fromImage(image)
        self.setPixmap(pixmap)
        self.setFixedHeight(max(self.minimumHeight(), pixmap.height() + 10))
from typing import Optional

from .models import Inequality


class InequalityEditor(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)

    def __init__(self, inequality: Optional[Inequality] = None, parent=None):
        super().__init__(parent)
        self._inequality = inequality or Inequality()
        self._build_ui()
        self._set_values(self._inequality)

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.coeff_a = FloatSpinBox()
        self.coeff_b = FloatSpinBox()
        self.constant = FloatSpinBox(step=1.0)

        font = QtGui.QFont()
        font.setPointSize(11)
        for widget in (self.coeff_a, self.coeff_b, self.constant):
            widget.setFont(font)
            widget.valueChanged.connect(self.changed)

        self.operator = QtWidgets.QComboBox()
        self.operator.setFont(font)
        self.operator.addItems(["<", "<=", ">", ">=", "="])
        self.operator.currentIndexChanged.connect(self.changed)

        remove_button = QtWidgets.QToolButton()
        remove_button.setText("✕")
        remove_button.setToolTip("Удалить неравенство")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self))

        layout.addWidget(QtWidgets.QLabel("a:"))
        layout.addWidget(self.coeff_a)
        layout.addWidget(QtWidgets.QLabel("b:"))
        layout.addWidget(self.coeff_b)
        layout.addWidget(self.operator)
        layout.addWidget(QtWidgets.QLabel("c:"))
        layout.addWidget(self.constant)
        layout.addWidget(remove_button)
        layout.addStretch(1)

    def _set_values(self, inequality: Inequality) -> None:
        self.coeff_a.setValue(inequality.a)
        self.coeff_b.setValue(inequality.b)
        self.constant.setValue(inequality.c)
        index = self.operator.findText(inequality.operator)
        if index >= 0:
            self.operator.setCurrentIndex(index)

    def to_inequality(self) -> Inequality:
        return Inequality(
            a=self.coeff_a.value(),
            b=self.coeff_b.value(),
            c=self.constant.value(),
            operator=self.operator.currentText(),
        )
