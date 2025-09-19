#!/usr/bin/env python3
"""
Qt-based visualizer for systems of linear inequalities with matplotlib shading.
"""

import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


TOLERANCE = 1e-6
DEFAULT_RANGE = (-10.0, 10.0)
GRID_POINTS = 400


@dataclass
class Inequality:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    operator: str = "<="

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return boolean mask where the inequality holds."""
        expression = self.a * x + self.b * y
        op = self.operator
        # Use small tolerance so boundaries remain visible.
        if op == "<":
            return expression < self.c - TOLERANCE
        if op == "<=":
            return expression <= self.c + TOLERANCE
        if op == ">":
            return expression > self.c + TOLERANCE
        if op == ">=":
            return expression >= self.c - TOLERANCE
        if op == "=":
            return np.isclose(expression, self.c, atol=max(TOLERANCE, 0.001 * (abs(self.c) + 1)))
        raise ValueError(f"Unsupported operator: {op}")


class FloatSpinBox(QtWidgets.QDoubleSpinBox):
    """Spin box tuned for coefficient editing."""

    def __init__(self, minimum=-1000.0, maximum=1000.0, step=0.5, decimals=3, parent=None):
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)


class InequalityEditor(QtWidgets.QWidget):
    """Row widget that lets the user edit a single inequality."""

    changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal('PyQt_PyObject')

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


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas that renders the inequality system."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.tight_layout()

    def draw_system(
        self,
        inequalities: List[Inequality],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        self.ax.clear()
        self.ax.set_xlabel("x₁")
        self.ax.set_ylabel("x₂")
        self.ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        self.ax.set_xlim(*x_range)
        self.ax.set_ylim(*y_range)
        self.ax.set_aspect("equal", adjustable="box")

        if not inequalities:
            self.ax.set_title("Добавьте неравенства")
            self.draw()
            return

        x_values = np.linspace(x_range[0], x_range[1], GRID_POINTS)
        y_values = np.linspace(y_range[0], y_range[1], GRID_POINTS)
        X, Y = np.meshgrid(x_values, y_values)

        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        colors_cycle = self._build_color_cycle(len(inequalities))
        intersection_mask = np.ones_like(X, dtype=bool)

        for idx, inequality in enumerate(inequalities):
            mask = inequality.evaluate(X, Y)
            color = colors_cycle[idx]

            if inequality.operator == "=":
                self._draw_equality(inequality, x_range, y_range, color)
                intersection_mask &= mask
                continue

            intersection_mask &= mask
            colormap = mcolors.ListedColormap([(0, 0, 0, 0), mcolors.to_rgba(color, alpha=0.35)])
            self.ax.imshow(
                mask.astype(float),
                extent=extent,
                origin="lower",
                cmap=colormap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            # Highlight boundary line for non-equalities.
            self._draw_boundary(inequality, x_range, y_range, color)

        if intersection_mask.any():
            colormap = mcolors.ListedColormap([(0, 0, 0, 0), (0.0, 0.0, 0.0, 0.55)])
            self.ax.imshow(
                intersection_mask.astype(float),
                extent=extent,
                origin="lower",
                cmap=colormap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        else:
            self.ax.set_title("Общая область пуста", fontsize=12, color="crimson")

        self.ax.figure.canvas.draw_idle()

    def _build_color_cycle(self, count: int) -> List[str]:
        base_cmap = mcolors.TABLEAU_COLORS
        keys = list(base_cmap.keys())
        colors_list = [base_cmap[key] for key in keys]
        # Repeat colors if there are more inequalities than unique colors in the base set.
        while len(colors_list) < count:
            colors_list.extend(colors_list)
        return colors_list[:count]

    def _draw_boundary(
        self,
        inequality: Inequality,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        color: str,
    ) -> None:
        a, b, c = inequality.a, inequality.b, inequality.c
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        if abs(b) > TOLERANCE:
            y_vals = (c - a * x_vals) / b
            valid = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
            if np.any(valid):
                linestyle = "--" if inequality.operator in {"<", ">"} else "-"
                self.ax.plot(x_vals[valid], y_vals[valid], color=color, linestyle=linestyle, linewidth=1.6)
        elif abs(a) > TOLERANCE:
            x_line = np.full_like(np.linspace(y_range[0], y_range[1], 400), c / a)
            linestyle = "--" if inequality.operator in {"<", ">"} else "-"
            self.ax.plot(x_line, np.linspace(y_range[0], y_range[1], 400), color=color, linestyle=linestyle, linewidth=1.6)

    def _draw_equality(
        self,
        inequality: Inequality,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        color: str,
    ) -> None:
        a, b, c = inequality.a, inequality.b, inequality.c
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        if abs(b) > TOLERANCE:
            y_vals = (c - a * x_vals) / b
            valid = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
            if np.any(valid):
                self.ax.plot(x_vals[valid], y_vals[valid], color=color, linewidth=2.0)
        elif abs(a) > TOLERANCE:
            x_line = np.full_like(np.linspace(y_range[0], y_range[1], 400), c / a)
            self.ax.plot(x_line, np.linspace(y_range[0], y_range[1], 400), color=color, linewidth=2.0)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Графический редактор неравенств")
        self.resize(1100, 600)
        self._inequality_widgets: list[InequalityEditor] = []

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        self.canvas = PlotCanvas()

        self.controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_inequality_group())
        controls_layout.addWidget(self._build_range_group())
        controls_layout.addStretch(1)

        main_layout.addWidget(self.controls_widget, stretch=2)

        main_layout.addWidget(self.canvas, stretch=5)

        self.setCentralWidget(central)

        self._build_menu()
        self._add_default_inequalities()
        self._refresh_plot()

    # Menu -----------------------------------------------------------------
    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        system_menu = menu_bar.addMenu("Система")

        add_action = QtWidgets.QAction("Добавить неравенство", self)
        add_action.triggered.connect(self._handle_add_inequality)
        system_menu.addAction(add_action)

        clear_action = QtWidgets.QAction("Очистить", self)
        clear_action.triggered.connect(self._handle_clear)
        system_menu.addAction(clear_action)

        system_menu.addSeparator()

        exit_action = QtWidgets.QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtWidgets.QApplication.instance().quit)
        system_menu.addAction(exit_action)

    # Inequality controls --------------------------------------------------
    def _build_inequality_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Неравенства")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.inequalities_container = QtWidgets.QWidget()
        self.inequalities_layout = QtWidgets.QVBoxLayout(self.inequalities_container)
        self.inequalities_layout.setContentsMargins(0, 0, 0, 0)
        self.inequalities_layout.setSpacing(4)
        self.inequalities_layout.addStretch(1)

        self.scroll_area.setWidget(self.inequalities_container)

        buttons_layout = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Добавить")
        add_button.clicked.connect(self._handle_add_inequality)
        buttons_layout.addWidget(add_button)

        reset_button = QtWidgets.QPushButton("Сбросить к примеру")
        reset_button.clicked.connect(self._add_default_inequalities)
        buttons_layout.addWidget(reset_button)

        buttons_layout.addStretch(1)

        layout.addWidget(self.scroll_area)
        layout.addLayout(buttons_layout)
        return group

    def _build_range_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Диапазоны отображения")
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.x_min = FloatSpinBox(minimum=-1000, maximum=1000, step=1.0)
        self.x_max = FloatSpinBox(minimum=-1000, maximum=1000, step=1.0)
        self.y_min = FloatSpinBox(minimum=-1000, maximum=1000, step=1.0)
        self.y_max = FloatSpinBox(minimum=-1000, maximum=1000, step=1.0)

        self.x_min.setValue(DEFAULT_RANGE[0])
        self.x_max.setValue(DEFAULT_RANGE[1])
        self.y_min.setValue(DEFAULT_RANGE[0])
        self.y_max.setValue(DEFAULT_RANGE[1])

        for spin in (self.x_min, self.x_max, self.y_min, self.y_max):
            spin.valueChanged.connect(self._refresh_plot)

        layout.addWidget(QtWidgets.QLabel("x₁ min"), 0, 0)
        layout.addWidget(self.x_min, 0, 1)
        layout.addWidget(QtWidgets.QLabel("x₁ max"), 0, 2)
        layout.addWidget(self.x_max, 0, 3)

        layout.addWidget(QtWidgets.QLabel("x₂ min"), 1, 0)
        layout.addWidget(self.y_min, 1, 1)
        layout.addWidget(QtWidgets.QLabel("x₂ max"), 1, 2)
        layout.addWidget(self.y_max, 1, 3)

        return group

    def _handle_add_inequality(self) -> None:
        self._append_inequality(Inequality())
        self._refresh_plot()

    def _handle_clear(self) -> None:
        for widget in list(self._inequality_widgets):
            self._remove_inequality(widget)
        self._refresh_plot()

    def _append_inequality(self, inequality: Inequality) -> None:
        widget = InequalityEditor(inequality)
        widget.changed.connect(self._refresh_plot)
        widget.remove_requested.connect(self._remove_inequality)

        self.inequalities_layout.insertWidget(self.inequalities_layout.count() - 1, widget)
        self._inequality_widgets.append(widget)

    def _remove_inequality(self, widget: InequalityEditor) -> None:
        if widget in self._inequality_widgets:
            self._inequality_widgets.remove(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._refresh_plot()

    def _collect_inequalities(self) -> List[Inequality]:
        return [widget.to_inequality() for widget in self._inequality_widgets]

    def _add_default_inequalities(self) -> None:
        self._handle_clear()
        examples = [
            Inequality(1, 1, 6, "<="),
            Inequality(1, -1, 2, ">="),
            Inequality(0, 1, 1, ">"),
        ]
        for sample in examples:
            self._append_inequality(sample)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        x_range = (min(self.x_min.value(), self.x_max.value()), max(self.x_min.value(), self.x_max.value()))
        y_range = (min(self.y_min.value(), self.y_max.value()), max(self.y_min.value(), self.y_max.value()))
        inequalities = self._collect_inequalities()
        self.canvas.draw_system(inequalities, x_range, y_range)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
