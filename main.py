#!/usr/bin/env python3
"""
Qt-based visualizer for systems of linear inequalities with matplotlib shading.
"""

import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import io

from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


TOLERANCE = 1e-6
DEFAULT_RANGE = (0.0, 10.0)
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


class LatexLabel(QtWidgets.QLabel):
    """QLabel capable of rendering LaTeX strings via matplotlib."""

    def __init__(self, dpi: int = 150, min_height: int = 120, parent=None):
        super().__init__(parent)
        self._dpi = dpi
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setStyleSheet("background-color: white; border: 1px solid #d0d0d0;")

    def set_latex(self, expression: Optional[str]) -> None:
        if not expression:
            self.clear()
            return

        if "\n" in expression:
            lines = [line.strip() for line in expression.split("\n") if line.strip()]
        else:
            lines = [expression.strip()]

        if not lines:
            self.clear()
            return

        formatted_lines = [line if (line.startswith("$") and line.endswith("$")) else f"${line}$" for line in lines]

        max_len = max(len(line) for line in formatted_lines)
        width_in = max(2.0, 0.12 * max_len)
        height_in = max(0.6, 0.45 * len(formatted_lines))

        try:
            fig = Figure(figsize=(width_in, height_in), dpi=self._dpi)
            fig.patch.set_facecolor("white")
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            total = len(formatted_lines)
            for idx, line in enumerate(formatted_lines):
                y = 1.0 - (idx + 0.5) / max(total, 1)
                ax.text(0.5, y, line, ha="center", va="center", fontsize=14)

            buffer = io.BytesIO()
            canvas.print_png(buffer)
        except Exception:
            self.clear()
            return

        buffer.seek(0)
        data = buffer.getvalue()
        image = QtGui.QImage.fromData(data, "PNG")
        if image.isNull():
            self.clear()
            return

        pixmap = QtGui.QPixmap.fromImage(image)
        self.setPixmap(pixmap)
        self.setFixedHeight(max(self.minimumHeight(), pixmap.height() + 10))


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
        vector: Optional[tuple[float, float]] = None,
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
        intersection_mask = np.ones_like(X, dtype=bool)

        for inequality in inequalities:
            mask = inequality.evaluate(X, Y)

            if inequality.operator == "=":
                self._draw_equality(inequality, x_range, y_range)
                intersection_mask &= mask
                continue

            intersection_mask &= mask
            # Highlight boundary line for non-equalities.
            self._draw_boundary(inequality, x_range, y_range)

        intersection_centroid = None
        if intersection_mask.any():
            intersection_centroid = (
                float(X[intersection_mask].mean()),
                float(Y[intersection_mask].mean()),
            )
            colormap = mcolors.ListedColormap([(0, 0, 0, 0), (0.5, 0.5, 0.5, 0.45)])
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

        base_point = None
        intersection_point = None
        if vector is not None and intersection_mask.any():
            base_point, intersection_point = self._find_support_point(
                vector,
                x_values,
                y_values,
                intersection_mask,
            )

        if vector is not None:
            base_for_perpendicular = base_point if base_point is not None else intersection_centroid
            self._draw_vector(vector, base_for_perpendicular, intersection_point)

        self.ax.figure.canvas.draw_idle()

    def _draw_vector(
        self,
        vector: tuple[float, float],
        perpendicular_base: Optional[tuple[float, float]] = None,
        intersection_point: Optional[tuple[float, float]] = None,
    ) -> None:
        a, b = vector
        color = "red"
        magnitude = np.hypot(a, b)
        if magnitude < TOLERANCE:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        half_length = span
        direction = np.array([a, b]) / magnitude
        line_points = np.outer([-half_length, half_length], direction)

        self.ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

        # Draw perpendicular line through the support point (or origin if absent).
        perp_direction = np.array([-direction[1], direction[0]])
        base_point = np.array([0.0, 0.0]) if perpendicular_base is None else np.array(perpendicular_base)
        perp_points = base_point + np.outer([-half_length, half_length], perp_direction)
        self.ax.plot(
            perp_points[:, 0],
            perp_points[:, 1],
            color=color,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )

        self.ax.arrow(
            0,
            0,
            a,
            b,
            color=color,
            width=0.0,
            length_includes_head=True,
            head_width=max(0.25, 0.02 * max(self.ax.get_xlim()[1] - self.ax.get_xlim()[0], self.ax.get_ylim()[1] - self.ax.get_ylim()[0])),
            head_length=max(0.35, 0.03 * max(self.ax.get_xlim()[1] - self.ax.get_xlim()[0], self.ax.get_ylim()[1] - self.ax.get_ylim()[0])),
            linewidth=2.0,
        )
        self.ax.scatter([a], [b], color=color, s=40, zorder=5)
        self.ax.annotate(
            f"F({a:.2f}, {b:.2f})",
            xy=(a, b),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

        if intersection_point is not None:
            self.ax.scatter([intersection_point[0]], [intersection_point[1]], color="black", s=60, zorder=7)

    def _find_support_point(
        self,
        vector: tuple[float, float],
        x_values: np.ndarray,
        y_values: np.ndarray,
        intersection_mask: np.ndarray,
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        a, b = vector
        magnitude = np.hypot(a, b)
        if magnitude < TOLERANCE:
            return (None, None)

        direction = np.array([a, b]) / magnitude
        dx, dy = direction
        perp = np.array([-dy, dx])

        # Determine maximum positive parameter t before leaving plot bounds.
        t_candidates: list[float] = []

        def append_limit(component: float, lower: float, upper: float) -> None:
            if abs(component) < TOLERANCE:
                if lower <= 0.0 <= upper:
                    t_candidates.append(np.inf)
                return
            if component > 0:
                limit = upper / component
                if limit > 0:
                    t_candidates.append(limit)
            else:
                limit = lower / component
                if limit > 0:
                    t_candidates.append(limit)

        append_limit(dx, x_values[0], x_values[-1])
        append_limit(dy, y_values[0], y_values[-1])

        if not t_candidates:
            return (None, None)

        t_max = min(t_candidates)
        if not np.isfinite(t_max) or t_max <= 0:
            return (None, None)

        # Helper to test whether a point lies inside the intersection mask.
        x_min, x_max = x_values[0], x_values[-1]
        y_min, y_max = y_values[0], y_values[-1]
        x_step = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
        y_step = y_values[1] - y_values[0] if len(y_values) > 1 else 1.0

        max_x_idx = len(x_values) - 1
        max_y_idx = len(y_values) - 1

        def point_inside(px: float, py: float) -> bool:
            if px < x_min or px > x_max or py < y_min or py > y_max:
                return False
            ix = int(round((px - x_min) / x_step))
            iy = int(round((py - y_min) / y_step))
            ix = int(np.clip(ix, 0, max_x_idx))
            iy = int(np.clip(iy, 0, max_y_idx))
            return bool(intersection_mask[iy, ix])

        span = max(x_max - x_min, y_max - y_min)
        s_values = np.linspace(-span, span, 256)

        def sample_intersection(t_val: float) -> tuple[bool, Optional[np.ndarray], Optional[float]]:
            center = direction * t_val
            best_s = None
            min_abs_s = None
            for s in s_values:
                px, py = center + s * perp
                if point_inside(px, py):
                    abs_s = abs(s)
                    if min_abs_s is None or abs_s < min_abs_s:
                        min_abs_s = abs_s
                        best_s = s
            if best_s is None:
                return False, None, None
            boundary_point = self._refine_boundary_point(center, perp, best_s, point_inside, span)
            return True, boundary_point, best_s

        t_values = np.linspace(0.0, t_max, 512)
        inside_flags: list[bool] = []
        intersection_cache: list[Optional[np.ndarray]] = []
        for t_val in t_values:
            flag, point, _ = sample_intersection(t_val)
            inside_flags.append(flag)
            intersection_cache.append(point)

        last_inside_idx = None
        for idx, flag in enumerate(inside_flags):
            if flag:
                last_inside_idx = idx
            elif last_inside_idx is not None:
                break

        if last_inside_idx is None:
            return (None, None)

        # Find first outside point after the last inside to refine boundary.
        t_low = t_values[last_inside_idx]
        t_high = t_max
        best_point = intersection_cache[last_inside_idx]
        for idx in range(last_inside_idx + 1, len(t_values)):
            if not inside_flags[idx]:
                t_high = t_values[idx]
                break

        for _ in range(20):
            t_mid = 0.5 * (t_low + t_high)
            flag, point, _ = sample_intersection(t_mid)
            if flag:
                t_low = t_mid
                best_point = point
            else:
                t_high = t_mid

        base_point = direction * t_low
        if best_point is not None:
            return (
                (float(base_point[0]), float(base_point[1])),
                (float(best_point[0]), float(best_point[1])),
            )
        return (float(base_point[0]), float(base_point[1])), None

    def _refine_boundary_point(
        self,
        center: np.ndarray,
        perp: np.ndarray,
        s_inside: float,
        point_inside_fn,
        span: float,
    ) -> np.ndarray:
        step = max(span / 256.0, 1e-3)

        def binary_search(s_low: float, s_high: float) -> float:
            for _ in range(25):
                s_mid = 0.5 * (s_low + s_high)
                px, py = center + s_mid * perp
                if point_inside_fn(px, py):
                    s_low = s_mid
                else:
                    s_high = s_mid
            return s_low

        if abs(s_inside) < TOLERANCE:
            for sign in (1.0, -1.0):
                s_out = s_inside
                while abs(s_out) <= span:
                    s_out += sign * step
                    px, py = center + s_out * perp
                    if not point_inside_fn(px, py):
                        boundary_s = binary_search(s_inside, s_out)
                        return center + boundary_s * perp
                # fallback to next sign
            return center + s_inside * perp

        sign = 1.0 if s_inside > 0 else -1.0
        s_out = s_inside
        while abs(s_out) <= span:
            s_out += sign * step
            px, py = center + s_out * perp
            if not point_inside_fn(px, py):
                boundary_s = binary_search(s_inside, s_out)
                return center + boundary_s * perp

        return center + s_inside * perp

    def _draw_boundary(
        self,
        inequality: Inequality,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        a, b, c = inequality.a, inequality.b, inequality.c
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        if abs(b) > TOLERANCE:
            y_vals = (c - a * x_vals) / b
            valid = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
            if np.any(valid):
                linestyle = "--" if inequality.operator in {"<", ">"} else "-"
                self.ax.plot(x_vals[valid], y_vals[valid], color="black", linestyle=linestyle, linewidth=1.6)
        elif abs(a) > TOLERANCE:
            x_line = np.full_like(np.linspace(y_range[0], y_range[1], 400), c / a)
            linestyle = "--" if inequality.operator in {"<", ">"} else "-"
            self.ax.plot(x_line, np.linspace(y_range[0], y_range[1], 400), color="black", linestyle=linestyle, linewidth=1.6)

    def _draw_equality(
        self,
        inequality: Inequality,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        a, b, c = inequality.a, inequality.b, inequality.c
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        if abs(b) > TOLERANCE:
            y_vals = (c - a * x_vals) / b
            valid = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
            if np.any(valid):
                self.ax.plot(x_vals[valid], y_vals[valid], color="black", linewidth=2.0)
        elif abs(a) > TOLERANCE:
            x_line = np.full_like(np.linspace(y_range[0], y_range[1], 400), c / a)
            self.ax.plot(x_line, np.linspace(y_range[0], y_range[1], 400), color="black", linewidth=2.0)


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
        controls_layout.addWidget(self._build_function_group())
        controls_layout.addWidget(self._build_latex_group())
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

    def _build_function_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Функция F = a·x₁ + b·x₂")
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.func_a = FloatSpinBox(minimum=-1000, maximum=1000, step=0.5)
        self.func_b = FloatSpinBox(minimum=-1000, maximum=1000, step=0.5)

        self.func_a.setValue(1.0)
        self.func_b.setValue(1.0)

        for spin in (self.func_a, self.func_b):
            spin.valueChanged.connect(self._handle_function_change)

        layout.addWidget(QtWidgets.QLabel("a"), 0, 0)
        layout.addWidget(self.func_a, 0, 1)
        layout.addWidget(QtWidgets.QLabel("b"), 1, 0)
        layout.addWidget(self.func_b, 1, 1)

        self.function_label = QtWidgets.QLabel()
        self.function_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.function_label.font()
        font.setPointSize(11)
        self.function_label.setFont(font)
        layout.addWidget(self.function_label, 2, 0, 1, 2)

        self._update_function_label()

        return group

    def _build_latex_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("LaTeX отображение")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.system_latex_label = LatexLabel(min_height=140)
        self.system_latex_label.setToolTip("Система неравенств")

        self.function_latex_label = LatexLabel(min_height=80)
        self.function_latex_label.setToolTip("Целевая функция")

        layout.addWidget(QtWidgets.QLabel("Система"))
        layout.addWidget(self.system_latex_label)
        layout.addWidget(QtWidgets.QLabel("Функция"))
        layout.addWidget(self.function_latex_label)

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
        vector = (self.func_a.value(), self.func_b.value()) if hasattr(self, "func_a") else None
        self.canvas.draw_system(inequalities, x_range, y_range, vector)
        self._update_latex_display(inequalities)

    def _handle_function_change(self, _value: float) -> None:
        self._update_function_label()
        self._refresh_plot()

    def _update_function_label(self) -> None:
        if not hasattr(self, "function_label"):
            return
        a = self.func_a.value()
        b = self.func_b.value()
        self.function_label.setText(f"F = {a:.2f}·x₁ + {b:.2f}·x₂")

    def _update_latex_display(self, inequalities: List[Inequality]) -> None:
        if not hasattr(self, "system_latex_label"):
            return
        system_expr = self._build_system_latex(inequalities)
        self.system_latex_label.set_latex(system_expr)

        func_expr = self._format_linear_expression(self.func_a.value(), self.func_b.value())
        if func_expr:
            self.function_latex_label.set_latex(f"F = {func_expr}")
        else:
            self.function_latex_label.set_latex("F = 0")

    def _format_number(self, value: float) -> str:
        if abs(value) < TOLERANCE:
            return "0"
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.2f}"

    def _format_linear_expression(self, a: float, b: float) -> str:
        terms: list[str] = []
        for coeff, symbol in ((a, r"x_{1}"), (b, r"x_{2}")):
            if abs(coeff) < TOLERANCE:
                continue
            coeff_sign = 1 if coeff >= 0 else -1
            coeff_abs = abs(coeff)
            is_one = abs(coeff_abs - 1.0) < 1e-9

            if not terms:
                if coeff_sign < 0:
                    prefix = "-"
                else:
                    prefix = ""
            else:
                prefix = "+" if coeff_sign > 0 else "-"

            if is_one:
                term_body = symbol
            else:
                term_body = f"{self._format_number(coeff_abs)}\\,{symbol}"

            if not terms:
                terms.append(f"{prefix}{term_body}" if prefix else term_body)
            else:
                sign = f"{prefix}\\," if prefix else ""
                terms.append(f"{sign}{term_body}")

        if not terms:
            return "0"
        return " ".join(terms)

    def _build_system_latex(self, inequalities: List[Inequality]) -> str:
        if not inequalities:
            return r"\text{Нет неравенств}"

        op_map = {"<": "<", "<=": "\\leq", ">": ">", ">=": "\\geq", "=": "="}
        rows: list[str] = []
        for ineq in inequalities:
            expr = self._format_linear_expression(ineq.a, ineq.b)
            rhs = self._format_number(ineq.c)
            op = op_map.get(ineq.operator, "=")
            rows.append(f"{expr} {op} {rhs}")

        return "\n".join(rows)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
