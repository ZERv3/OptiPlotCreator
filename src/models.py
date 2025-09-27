from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

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
        expression = self.a * x + self.b * y
        op = self.operator
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


InequalityList = List[Inequality]
