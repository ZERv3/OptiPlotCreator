from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtWidgets

from .models import Inequality


def build_config(
    function_a: float,
    function_b: float,
    inequalities: List[Inequality],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Dict:
    return {
        "function": {"a": function_a, "b": function_b},
        "inequalities": [
            {"a": ineq.a, "b": ineq.b, "c": ineq.c, "operator": ineq.operator}
            for ineq in inequalities
        ],
        "display": {
            "x": {"min": x_range[0], "max": x_range[1]},
            "y": {"min": y_range[0], "max": y_range[1]},
        },
    }


def export_config(parent: QtWidgets.QWidget, config: Dict) -> None:
    dialog = QtWidgets.QFileDialog(parent)
    dialog.setWindowTitle("Экспорт конфигурации")
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    dialog.setNameFilters(["CFG (*.cfg)", "JSON (*.json)", "Все файлы (*)"])
    dialog.setDefaultSuffix("cfg")
    if dialog.exec() == QtWidgets.QDialog.Accepted:
        filename = Path(dialog.selectedFiles()[0])
        try:
            with filename.open("w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(parent, "Ошибка экспорта", f"Не удалось сохранить cfg:\n{exc}")


def import_config(parent: QtWidgets.QWidget) -> Optional[Dict]:
    filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Импорт конфигурации",
        "",
        "CFG (*.cfg);;JSON (*.json);;Все файлы (*)",
    )
    if not filename:
        return None
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(parent, "Ошибка импорта", f"Не удалось прочитать cfg:\n{exc}")
        return None
