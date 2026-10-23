from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shiboken6
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF

from .activity import CREATING, IDLE, REINFORCING, AgentSample

TAIL_STEPS = 7.0
MIN_TAIL = 2.5
MAX_TAIL = 14.0
DRAW_ORDER = (IDLE, REINFORCING, CREATING)


def _fill_buffer(xy: np.ndarray) -> QPolygonF:
    poly = QPolygonF()
    poly.resize(xy.shape[0])
    if xy.shape[0]:
        view = np.frombuffer(shiboken6.VoidPtr(poly.data(), xy.nbytes, True), dtype=np.float64)
        view[:] = xy.ravel()
    return poly


def _fill_list(xy: np.ndarray) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in xy.tolist()])


def _zero_copy_supported() -> bool:
    try:
        probe = np.array([[1.25, -2.5], [3.75, 4.0]], dtype=np.float64)
        poly = _fill_buffer(probe)
        return poly.size() == 2 and poly.at(1).x() == 3.75 and poly.at(1).y() == 4.0 and poly.at(0).y() == -2.5
    except Exception:
        return False


ZERO_COPY = _zero_copy_supported()


def polygon(xy: np.ndarray) -> QPolygonF:
    xy = np.ascontiguousarray(xy, dtype=np.float64).reshape(-1, 2)
    return _fill_buffer(xy) if ZERO_COPY else _fill_list(xy)


@dataclass(frozen=True)
class OverlayLayer:
    status: int
    heads: QPolygonF
    tails: QPolygonF
    count: int


@dataclass(frozen=True)
class OverlayGeometry:
    layers: tuple[OverlayLayer, ...]
    width: int
    height: int

    def count(self, status: int) -> int:
        return sum(layer.count for layer in self.layers if layer.status == status)


def build_overlay(sample: AgentSample | None) -> OverlayGeometry | None:
    if sample is None or sample.x.size == 0:
        return None
    layers = []
    finite = np.isfinite(sample.x) & np.isfinite(sample.y)
    for status in DRAW_ORDER:
        mask = (sample.status == status) & finite
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        xs = sample.x[mask].astype(np.float64)
        ys = sample.y[mask].astype(np.float64)
        heads = polygon(np.column_stack((xs, ys)))
        tails = QPolygonF()
        if status == CREATING and sample.vx is not None and sample.vy is not None:
            vx = np.nan_to_num(sample.vx[mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            vy = np.nan_to_num(sample.vy[mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            speed = np.hypot(vx, vy)
            moving = speed > 1e-6
            length = np.clip(speed * TAIL_STEPS, MIN_TAIL, MAX_TAIL)
            factor = np.where(moving, length / np.maximum(speed, 1e-6), 0.0)
            tx = xs - vx * factor
            ty = ys - vy * factor
            keep = moving & (np.abs(tx - xs) < sample.width / 2) & (np.abs(ty - ys) < sample.height / 2)
            pairs = np.empty((int(np.count_nonzero(keep)) * 2, 2), dtype=np.float64)
            pairs[0::2, 0] = tx[keep]
            pairs[0::2, 1] = ty[keep]
            pairs[1::2, 0] = xs[keep]
            pairs[1::2, 1] = ys[keep]
            tails = polygon(pairs)
        layers.append(OverlayLayer(status, heads, tails, count))
    return OverlayGeometry(tuple(layers), sample.width, sample.height)


__all__ = ["CREATING", "IDLE", "REINFORCING", "ZERO_COPY", "OverlayGeometry", "OverlayLayer", "build_overlay", "polygon"]
