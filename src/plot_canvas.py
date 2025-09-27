from __future__ import annotations

from typing import Optional, Tuple
import itertools

import numpy as np
from PyQt5 import QtWidgets
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .models import GRID_POINTS, TOLERANCE, Inequality, InequalityList


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        self.figure = Figure(figsize=(5, 5))
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.tight_layout()

        self._vertices: list[Tuple[float, float]] = []
        self._vector: Optional[Tuple[float, float]] = None
        self._perpendiculars: list[Tuple[Tuple[float, float], object, object]] = []
        self._vertex_highlights: dict[Tuple[float, float], object] = {}
        self._click_cid = self.mpl_connect("button_press_event", self._on_click)

    def draw_system(
        self,
        inequalities: InequalityList,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        vector: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.ax.clear()
        self.ax.set_xlabel("x₁")
        self.ax.set_ylabel("x₂")
        self.ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        self.ax.set_xlim(*x_range)
        self.ax.set_ylim(*y_range)
        self.ax.set_aspect("equal", adjustable="box")

        self._draw_axes_guides()

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
            self._draw_boundary(inequality, x_range, y_range)

        if intersection_mask.any():
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

        feasible_vertices = self._collect_feasible_vertices(inequalities)
        self._vertices = feasible_vertices
        self._vector = vector
        self._clear_perpendiculars()

        intersection_point = None
        if vector is not None and intersection_mask.any():
            _, intersection_point = self._find_support_point(vector, x_values, y_values, intersection_mask)
            self._draw_vector(vector, intersection_point)

        self._draw_vertices(feasible_vertices)

        self.ax.figure.canvas.draw_idle()

    def _draw_axes_guides(self) -> None:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if xlim[0] <= 0 <= xlim[1]:
            self.ax.axvline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3, zorder=1)
        if ylim[0] <= 0 <= ylim[1]:
            self.ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3, zorder=1)

    def _draw_vector(
        self,
        vector: Tuple[float, float],
        intersection_point: Optional[Tuple[float, float]] = None,
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

        perp_direction = np.array([-direction[1], direction[0]])
        base_point = np.array([0.0, 0.0])
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
            head_width=max(0.25, 0.02 * span),
            head_length=max(0.35, 0.03 * span),
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
            self.ax.scatter([intersection_point[0]], [intersection_point[1]], color="black", s=1, alpha=0.0, zorder=7)

    def _draw_boundary(
        self,
        inequality: Inequality,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
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
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
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
    def _find_support_point(
        self,
        vector: Tuple[float, float],
        x_values: np.ndarray,
        y_values: np.ndarray,
        intersection_mask: np.ndarray,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        a, b = vector
        magnitude = np.hypot(a, b)
        if magnitude < TOLERANCE:
            return (None, None)

        direction = np.array([a, b]) / magnitude
        dx, dy = direction
        perp = np.array([-dy, dx])

        t_candidates: list[float] = []

        def append_limit(component: float, lower: float, upper: float) -> None:
            if abs(component) < TOLERANCE:
                if lower <= 0.0 <= upper:
                    t_candidates.append(np.inf)
                return
            limit = (upper if component > 0 else lower) / component
            if limit > 0:
                t_candidates.append(limit)

        append_limit(dx, x_values[0], x_values[-1])
        append_limit(dy, y_values[0], y_values[-1])

        if not t_candidates:
            return (None, None)

        t_max = min(t_candidates)
        if not np.isfinite(t_max) or t_max <= 0:
            return (None, None)

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

        def sample_intersection(t_val: float):
            center = direction * t_val
            chosen_point = None
            min_abs_s = None
            for s in s_values:
                px, py = center + s * perp
                if point_inside(px, py):
                    abs_s = abs(s)
                    if min_abs_s is None or abs_s < min_abs_s:
                        min_abs_s = abs_s
                        chosen_point = np.array([px, py])
            return chosen_point is not None, chosen_point

        t_values = np.linspace(0.0, t_max, 512)
        inside_flags = []
        intersection_cache = []
        for t_val in t_values:
            flag, point = sample_intersection(t_val)
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

        t_low = t_values[last_inside_idx]
        t_high = t_max
        best_point = intersection_cache[last_inside_idx]
        for idx in range(last_inside_idx + 1, len(t_values)):
            if not inside_flags[idx]:
                t_high = t_values[idx]
                break

        for _ in range(20):
            t_mid = 0.5 * (t_low + t_high)
            flag, point = sample_intersection(t_mid)
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

    def _collect_feasible_vertices(self, inequalities: InequalityList) -> list[Tuple[float, float]]:
        vertices: list[Tuple[float, float]] = []
        for first, second in itertools.combinations(inequalities, 2):
            denom = first.a * second.b - second.a * first.b
            if abs(denom) < TOLERANCE:
                continue
            x = (first.c * second.b - second.c * first.b) / denom
            y = (first.a * second.c - second.a * first.c) / denom
            if self._point_satisfies(inequalities, x, y):
                self._add_vertex(vertices, (x, y))
        return vertices

    def _point_satisfies(self, inequalities: InequalityList, x: float, y: float) -> bool:
        for ineq in inequalities:
            lhs = ineq.a * x + ineq.b * y
            rhs = ineq.c
            if ineq.operator == "<":
                if not lhs < rhs + TOLERANCE:
                    return False
            elif ineq.operator == "<=":
                if not lhs <= rhs + TOLERANCE:
                    return False
            elif ineq.operator == ">":
                if not lhs > rhs - TOLERANCE:
                    return False
            elif ineq.operator == ">=":
                if not lhs >= rhs - TOLERANCE:
                    return False
            elif ineq.operator == "=":
                if abs(lhs - rhs) > TOLERANCE:
                    return False
        return True

    def _add_vertex(self, vertices: list[Tuple[float, float]], candidate: Tuple[float, float]) -> None:
        for existing in vertices:
            if abs(existing[0] - candidate[0]) < 1e-6 and abs(existing[1] - candidate[1]) < 1e-6:
                return
        vertices.append(candidate)

    def _draw_vertices(self, vertices: list[Tuple[float, float]]) -> None:
        for x, y in vertices:
            self.ax.scatter([x], [y], color="black", s=1, zorder=7, alpha=0.0, picker=True)

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if not self._vertices or not self._vector:
            return

        clicked_point = (event.xdata, event.ydata)
        nearest = None
        min_dist = None
        for vx, vy in self._vertices:
            dist = np.hypot(vx - clicked_point[0], vy - clicked_point[1])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                nearest = (vx, vy)

        if nearest is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if min_dist is None or min_dist > 0.05 * span:
            return

        if event.button == 1:  # left click
            self._highlight_vertex(nearest)
            self._draw_perpendicular_through(nearest)
        elif event.button == 3:  # right click
            self._remove_perpendicular(nearest)

    def _draw_perpendicular_through(self, point: Tuple[float, float]) -> None:
        vector = self._vector
        if vector is None:
            return
        a, b = vector
        if abs(a) < TOLERANCE and abs(b) < TOLERANCE:
            return
        direction = np.array([a, b], dtype=float)
        norm = np.hypot(*direction)
        if norm < TOLERANCE:
            return
        direction /= norm
        perp = np.array([-direction[1], direction[0]])

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        segment = self._compute_segment(point, perp, xlim, ylim)
        if segment is None:
            return

        (x1, y1), (x2, y2) = segment
        for (vx, vy), line_artist, label_artist in self._perpendiculars:
            if np.hypot(point[0] - vx, point[1] - vy) < 1e-6:
                return

        line, = self.ax.plot([x1, x2], [y1, y2], color="crimson", linestyle="-.", linewidth=1.6, zorder=6)
        label = self.ax.annotate(
            f"({point[0]:.2f}, {point[1]:.2f})",
            xy=point,
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
        self._perpendiculars.append(((point[0], point[1]), line, label))
        self.ax.figure.canvas.draw_idle()

    def _compute_segment(
        self,
        point: Tuple[float, float],
        direction: np.ndarray,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        px, py = point
        dx, dy = direction
        candidates = []

        if abs(dx) > TOLERANCE:
            for x_bound in xlim:
                t = (x_bound - px) / dx
                y = py + t * dy
                if ylim[0] - 1e-9 <= y <= ylim[1] + 1e-9:
                    candidates.append((t, (px + t * dx, y)))
        if abs(dy) > TOLERANCE:
            for y_bound in ylim:
                t = (y_bound - py) / dy
                x = px + t * dx
                if xlim[0] - 1e-9 <= x <= xlim[1] + 1e-9:
                    candidates.append((t, (x, py + t * dy)))

        if not candidates:
            return None

        # remove duplicates based on coordinates
        filtered: list[Tuple[float, Tuple[float, float]]] = []
        for t, point in candidates:
            duplicate = False
            for existing_t, existing_point in filtered:
                if np.hypot(point[0] - existing_point[0], point[1] - existing_point[1]) < 1e-6:
                    duplicate = True
                    if abs(t) > abs(existing_t):
                        filtered.remove((existing_t, existing_point))
                        filtered.append((t, point))
                    break
            if not duplicate:
                filtered.append((t, point))

        if len(filtered) < 2:
            return None

        filtered.sort(key=lambda item: item[0])
        start = filtered[0][1]
        end = filtered[-1][1]
        return start, end

    def _highlight_vertex(self, point: Tuple[float, float]) -> None:
        self._remove_vertex_highlight(point)
        current = self._vertex_highlights.get(point)
        if current is not None:
            try:
                current.remove()
            except Exception:
                pass
        dot = self.ax.scatter([point[0]], [point[1]], color="red", s=65, zorder=8)
        self._vertex_highlights[point] = dot

    def _remove_vertex_highlight(self, point: Tuple[float, float]) -> None:
        artist = None
        for key in list(self._vertex_highlights.keys()):
            if np.hypot(key[0] - point[0], key[1] - point[1]) < 1e-6:
                artist = self._vertex_highlights.pop(key)
                try:
                    artist.remove()
                except Exception:
                    pass
                break
        if artist is not None:
            self.ax.figure.canvas.draw_idle()

    def _clear_vertex_highlights(self) -> None:
        for artist in self._vertex_highlights.values():
            try:
                artist.remove()
            except Exception:
                pass
        self._vertex_highlights.clear()
        self.ax.figure.canvas.draw_idle()

    def _remove_perpendicular(self, point: Tuple[float, float]) -> None:
        remaining = []
        removed = False
        for (vx, vy), line_artist, label_artist in self._perpendiculars:
            if np.hypot(vx - point[0], vy - point[1]) < 1e-6:
                try:
                    line_artist.remove()
                except Exception:
                    pass
                try:
                    label_artist.remove()
                except Exception:
                    pass
                self._remove_vertex_highlight(point)
                removed = True
            else:
                remaining.append(((vx, vy), line_artist, label_artist))
        self._perpendiculars = remaining
        if removed:
            self.ax.figure.canvas.draw_idle()

    def _clear_perpendiculars(self) -> None:
        if not self._perpendiculars:
            return
        for _, line_artist, label_artist in self._perpendiculars:
            try:
                line_artist.remove()
            except Exception:
                pass
            try:
                label_artist.remove()
            except Exception:
                pass
        self._perpendiculars.clear()
        self._clear_vertex_highlights()
        self.ax.figure.canvas.draw_idle()
