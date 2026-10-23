from __future__ import annotations

import numpy as np

TAU = np.float32(2.0 * np.pi)


def cosine_palette(t: np.ndarray, palette: tuple[float, ...]) -> np.ndarray:
    p = np.asarray(palette, dtype=np.float32)
    a = p[0:3]
    b = p[3:6]
    c = np.float32(0.25) + np.float32(1.75) * p[6:9]
    d = p[9:12]
    phase = TAU * (t.astype(np.float32, copy=False)[:, None] * c + d)
    out = a + b * np.cos(phase)
    np.clip(out, 0.0, 1.0, out=out)
    return out


class PaletteLUT:
    LOW = -1.0
    HIGH = 3.0
    SIZE = 4096

    def __init__(self) -> None:
        self._key: tuple[float, ...] | None = None
        self._table = np.zeros((self.SIZE, 3), dtype=np.float32)
        self._scale = np.float32(self.SIZE / (self.HIGH - self.LOW))

    def table(self, palette: tuple[float, ...]) -> np.ndarray:
        if palette != self._key:
            t = np.linspace(self.LOW, self.HIGH, self.SIZE, endpoint=False, dtype=np.float32)
            self._table = cosine_palette(t, palette)
            self._key = palette
        return self._table

    def __call__(self, t: np.ndarray, palette: tuple[float, ...]) -> np.ndarray:
        table = self.table(palette)
        q = (t - np.float32(self.LOW)) * self._scale
        np.clip(q, 0, self.SIZE - 1, out=q)
        return table[q.astype(np.int32)]


def periodic_value_noise(h: int, w: int, cells: float, rng: np.random.Generator) -> np.ndarray:
    longest = max(h, w)
    gy = max(2, int(round(cells * h / longest)))
    gx = max(2, int(round(cells * w / longest)))
    lattice = rng.random((gy, gx), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32) * np.float32(gy / h)
    xs = np.arange(w, dtype=np.float32) * np.float32(gx / w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    ty = ys - y0
    tx = xs - x0
    sy = ty * ty * (3.0 - 2.0 * ty)
    sx = tx * tx * (3.0 - 2.0 * tx)
    y0 %= gy
    x0 %= gx
    y1 = (y0 + 1) % gy
    x1 = (x0 + 1) % gx
    top = lattice[y0][:, x0] + (lattice[y0][:, x1] - lattice[y0][:, x0]) * sx[None, :]
    bottom = lattice[y1][:, x0] + (lattice[y1][:, x1] - lattice[y1][:, x0]) * sx[None, :]
    return (top + (bottom - top) * sy[:, None]).astype(np.float32)


def fractal_noise(h: int, w: int, cells: float, octaves: int, rng: np.random.Generator) -> np.ndarray:
    total = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    norm = 0.0
    limit = max(2.0, min(h, w) / 2.0)
    for _ in range(max(1, octaves)):
        total += np.float32(amplitude) * periodic_value_noise(h, w, min(cells, limit), rng)
        norm += amplitude
        amplitude *= 0.5
        cells *= 2.0
    return total / np.float32(norm)


def curl_field(h: int, w: int, cells: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    psi = fractal_noise(h, w, cells, 4, rng)
    dpsi_dy = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) * np.float32(0.5)
    dpsi_dx = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) * np.float32(0.5)
    vx = dpsi_dy
    vy = -dpsi_dx
    magnitude = np.sqrt(vx * vx + vy * vy)
    reference = float(np.percentile(magnitude, 95))
    scale = np.float32(1.0 / reference) if reference > 1e-12 else np.float32(0.0)
    vx *= scale
    vy *= scale
    magnitude *= scale
    limit = np.minimum(np.float32(1.0), np.float32(1.5) / np.maximum(magnitude, np.float32(1e-6)))
    return (vx * limit).astype(np.float32), (vy * limit).astype(np.float32)


def blur3_wrap(grid: np.ndarray) -> np.ndarray:
    vertical = np.roll(grid, 1, 0)
    vertical += grid
    vertical += np.roll(grid, -1, 0)
    out = np.roll(vertical, 1, 1)
    out += vertical
    out += np.roll(vertical, -1, 1)
    return out
