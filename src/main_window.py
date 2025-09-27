from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

from .config_io import build_config, export_config, import_config
from .models import DEFAULT_RANGE, Inequality, InequalityList
from .plot_canvas import PlotCanvas
from .ui import FloatSpinBox, InequalityEditor, LatexLabel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Графический редактор неравенств")
        self._inequality_widgets: List[InequalityEditor] = []

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.controls_widget = QtWidgets.QWidget()
        self.controls_widget.setMinimumWidth(260)
        controls_layout = QtWidgets.QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_inequality_group())
        controls_layout.addWidget(self._build_range_group())
        controls_layout.addWidget(self._build_function_group())
        controls_layout.addWidget(self._build_latex_group())
        controls_layout.addWidget(self._build_save_group())
        controls_layout.addWidget(self._build_config_group())
        controls_layout.addStretch(1)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.controls_scroll.setWidget(self.controls_widget)
        self.controls_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        main_layout.addWidget(self.controls_scroll, stretch=2)
        main_layout.addWidget(self.canvas, stretch=5)
        self.setCentralWidget(central)

        self._build_menu()
        self._refresh_plot()
        if not self._inequality_widgets:
            self._append_inequality(Inequality())

        self.setMinimumSize(720, 480)
        self.resize(1000, 620)

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

        self.func_a.setValue(0.0)
        self.func_b.setValue(0.0)

        for spin in (self.func_a, self.func_b):
            spin.valueChanged.connect(self._handle_function_change)

        layout.addWidget(QtWidgets.QLabel("a"), 0, 0)
        layout.addWidget(self.func_a, 0, 1)
        layout.addWidget(QtWidgets.QLabel("b"), 1, 0)
        layout.addWidget(self.func_b, 1, 1)

        self.function_label = QtWidgets.QLabel()
        self.function_label.setAlignment(QtCore.Qt.AlignCenter)
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

    def _build_save_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Изображение")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        save_button = QtWidgets.QPushButton("Сохранить график…")
        save_button.clicked.connect(self._handle_save_figure)
        layout.addWidget(save_button)

        info_label = QtWidgets.QLabel("Сохранить текущее изображение графика в файл")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        return group

    def _build_config_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Конфигурация")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        buttons_layout = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Импорт cfg…")
        import_btn.clicked.connect(self._handle_import_config)
        export_btn = QtWidgets.QPushButton("Экспорт cfg…")
        export_btn.clicked.connect(self._handle_export_config)
        buttons_layout.addWidget(import_btn)
        buttons_layout.addWidget(export_btn)
        layout.addLayout(buttons_layout)

        sample_text = (
            '{\n'
            '  "function": {"a": 1.0, "b": 1.0},\n'
            '  "inequalities": [\n'
            '    {"a": 1.0, "b": 1.0, "c": 6.0, "operator": "<="}\n'
            '  ],\n'
            '  "display": {\n'
            '    "x": {"min": 0.0, "max": 10.0},\n'
            '    "y": {"min": 0.0, "max": 10.0}\n'
            '  }\n'
            '}'
        )

        hint = QtWidgets.QTextEdit()
        hint.setReadOnly(True)
        hint.setFontFamily("Courier")
        hint.setMinimumHeight(120)
        hint.setPlainText(sample_text)
        layout.addWidget(hint)
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

    def _collect_inequalities(self) -> InequalityList:
        return [widget.to_inequality() for widget in self._inequality_widgets]

    def _add_default_inequalities(self) -> None:
        self._handle_clear()

    def _refresh_plot(self) -> None:
        x_range = (
            min(self.x_min.value(), self.x_max.value()),
            max(self.x_min.value(), self.x_max.value()),
        )
        y_range = (
            min(self.y_min.value(), self.y_max.value()),
            max(self.y_min.value(), self.y_max.value()),
        )
        inequalities = self._collect_inequalities()
        vector = (self.func_a.value(), self.func_b.value())
        self.canvas.draw_system(inequalities, x_range, y_range, vector)
        self._update_latex_display(inequalities)

    def _handle_function_change(self, _value: float) -> None:
        self._update_function_label()
        self._refresh_plot()

    def _update_function_label(self) -> None:
        a = self.func_a.value()
        b = self.func_b.value()
        self.function_label.setText(f"F = {a:.2f}·x₁ + {b:.2f}·x₂")

    def _update_latex_display(self, inequalities: InequalityList) -> None:
        system_expr = self._build_system_latex(inequalities)
        self.system_latex_label.set_latex(system_expr)

        func_expr = self._format_linear_expression(self.func_a.value(), self.func_b.value())
        self.function_latex_label.set_latex(f"F = {func_expr}" if func_expr else "F = 0")

    def _handle_save_figure(self) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setWindowTitle("Сохранить график")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(["PNG (*.png)", "JPEG (*.jpg *.jpeg)", "PDF (*.pdf)", "SVG (*.svg)"])
        dialog.setDefaultSuffix("png")
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            filename = dialog.selectedFiles()[0]
            if filename:
                try:
                    self.canvas.figure.savefig(filename, dpi=300, bbox_inches="tight")
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось сохранить файл:\n{exc}")

    def _handle_export_config(self) -> None:
        x_range = (
            min(self.x_min.value(), self.x_max.value()),
            max(self.x_min.value(), self.x_max.value()),
        )
        y_range = (
            min(self.y_min.value(), self.y_max.value()),
            max(self.y_min.value(), self.y_max.value()),
        )
        config = build_config(
            self.func_a.value(),
            self.func_b.value(),
            self._collect_inequalities(),
            x_range,
            y_range,
        )
        export_config(self, config)

    def _handle_import_config(self) -> None:
        data = import_config(self)
        if data:
            try:
                self._apply_config_data(data)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Ошибка импорта", str(exc))

    def _apply_config_data(self, data: dict) -> None:
        func = data.get("function")
        ineqs = data.get("inequalities")
        display = data.get("display")
        if not isinstance(func, dict) or not isinstance(ineqs, list):
            raise ValueError("Некорректный формат конфигурации")

        try:
            a_val = float(func.get("a", 0.0))
            b_val = float(func.get("b", 0.0))
        except (TypeError, ValueError):
            raise ValueError("Некорректные коэффициенты функции")

        new_inequalities: InequalityList = []
        for idx, item in enumerate(ineqs):
            if not isinstance(item, dict):
                raise ValueError(f"Неравенство №{idx + 1} имеет неверный формат")
            try:
                a = float(item.get("a", 0.0))
                b = float(item.get("b", 0.0))
                c = float(item.get("c", 0.0))
            except (TypeError, ValueError):
                raise ValueError(f"Неравенство №{idx + 1}: коэффициенты должны быть числами")
            operator = item.get("operator", "<=")
            if operator not in {"<", "<=", ">", ">=", "="}:
                raise ValueError(f"Неравенство №{idx + 1}: неподдерживаемый оператор '{operator}'")
            new_inequalities.append(Inequality(a=a, b=b, c=c, operator=operator))

        self.func_a.blockSignals(True)
        self.func_b.blockSignals(True)
        self.func_a.setValue(a_val)
        self.func_b.setValue(b_val)
        self.func_a.blockSignals(False)
        self.func_b.blockSignals(False)

        if isinstance(display, dict):
            x_disp = display.get("x")
            y_disp = display.get("y")
            if isinstance(x_disp, dict) and isinstance(y_disp, dict):
                try:
                    x_min = float(x_disp.get("min", self.x_min.value()))
                    x_max = float(x_disp.get("max", self.x_max.value()))
                    y_min = float(y_disp.get("min", self.y_min.value()))
                    y_max = float(y_disp.get("max", self.y_max.value()))
                except (TypeError, ValueError):
                    raise ValueError("Некорректные значения диапазона отображения")

                for spin, value in (
                    (self.x_min, x_min),
                    (self.x_max, x_max),
                    (self.y_min, y_min),
                    (self.y_max, y_max),
                ):
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)

        self._handle_clear()
        if new_inequalities:
            for ineq in new_inequalities:
                self._append_inequality(ineq)
        else:
            self._append_inequality(Inequality())

        self._refresh_plot()

    def _format_number(self, value: float) -> str:
        if abs(value) < 1e-9:
            return "0"
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.2f}"

    def _format_linear_expression(self, a: float, b: float) -> str:
        terms = []
        for coeff, symbol in ((a, r"x_{1}"), (b, r"x_{2}")):
            if abs(coeff) < 1e-9:
                continue
            sign = "+" if coeff >= 0 else "-"
            coeff_abs = abs(coeff)
            coeff_str = "" if abs(coeff_abs - 1.0) < 1e-9 else f"{self._format_number(coeff_abs)}\\,"
            term = f"{coeff_str}{symbol}"
            if not terms:
                terms.append(term if coeff >= 0 else f"-{term}")
            else:
                terms.append(f" {sign} {term}")
        return "".join(terms) if terms else "0"

    def _build_system_latex(self, inequalities: InequalityList) -> str:
        if not inequalities:
            return r"\text{Нет неравенств}"
        op_map = {"<": "<", "<=": "\\leq", ">": ">", ">=": "\\geq", "=": "="}
        rows = []
        for ineq in inequalities:
            expr = self._format_linear_expression(ineq.a, ineq.b)
            rhs = self._format_number(ineq.c)
            rows.append(f"{expr} {op_map.get(ineq.operator, '=')} {rhs}")
        return "\n".join(rows)
