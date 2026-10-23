from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

F32 = np.float32
LUT_SIZE = 4096
LUT_RANGE = 8.0
BYTES_PER_PIXEL = 4
OPAQUE = np.uint32(0xFF000000)


def _build_tone_lut() -> np.ndarray:
    x = np.linspace(0.0, LUT_RANGE, LUT_SIZE, dtype=np.float64)
    return (np.sqrt(1.0 - np.exp(-x)) * 255.0 + 0.5).astype(np.uint8)


TONE_LUT = _build_tone_lut()
_CHANNEL_LUTS = (
    TONE_LUT.astype(np.uint32) << np.uint32(16),
    TONE_LUT.astype(np.uint32) << np.uint32(8),
    TONE_LUT.astype(np.uint32),
)


class Renderer:
    def __init__(self, smoothing: float = 0.2) -> None:
        self.smoothing = smoothing
        self.scale: float | None = None
        self._scratch_f: np.ndarray | None = None
        self._scratch_i: np.ndarray | None = None
        self._scratch_u: np.ndarray | None = None

    def reset(self) -> None:
        self.scale = None

    def _robust_scale(self, accum: np.ndarray) -> float:
        stride = max(1, accum.shape[1] // 32768)
        sample = accum[:, ::stride].max(axis=0)
        nonzero = sample[(sample > 0) & np.isfinite(sample)]
        if nonzero.size < 8:
            return self.scale if self.scale else 1.0
        level = float(np.percentile(nonzero, 99.5))
        if not np.isfinite(level) or level <= 0:
            return self.scale if self.scale else 1.0
        if self.scale is None:
            self.scale = level
        else:
            self.scale += (level - self.scale) * self.smoothing
        return self.scale

    def _scratch(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._scratch_f is None or self._scratch_f.size != size:
            self._scratch_f = np.empty(size, dtype=F32)
            self._scratch_i = np.empty(size, dtype=np.int32)
            self._scratch_u = np.empty(size, dtype=np.uint32)
        return self._scratch_f, self._scratch_i, self._scratch_u

    def pack(self, accum: np.ndarray, exposure: float) -> np.ndarray:
        size = accum.shape[1]
        scale = self._robust_scale(accum)
        k = F32(exposure / max(scale, 1e-12) * (LUT_SIZE / LUT_RANGE))
        tmp, idx, part = self._scratch(size)
        out = np.empty(size, dtype=np.uint32)
        with np.errstate(invalid="ignore", over="ignore"):
            for c in range(3):
                np.multiply(accum[c], k, out=tmp)
                np.clip(tmp, 0, LUT_SIZE - 1, out=tmp)
                idx[...] = tmp
                if c == 0:
                    np.take(_CHANNEL_LUTS[c], idx, out=out, mode="clip")
                else:
                    np.take(_CHANNEL_LUTS[c], idx, out=part, mode="clip")
                    out |= part
        out |= OPAQUE
        return out

    def render(self, accum: np.ndarray, h: int, w: int, exposure: float, glow: float) -> np.ndarray:
        packed = self.pack(accum, exposure)
        apply_glow(packed, h, w, glow)
        return packed

    def to_float(self, accum: np.ndarray, h: int, w: int, exposure: float) -> np.ndarray:
        return unpack_rgb(self.pack(accum, exposure), h, w).astype(F32) * F32(1.0 / 255.0)


PREVIEW_PIXEL_BUDGET = 1_100_000


def preview_factor(h: int, w: int, budget: int = PREVIEW_PIXEL_BUDGET) -> int:
    factor = 1
    while (h // factor) * (w // factor) > budget and min(h, w) // (factor * 2) >= 64:
        factor *= 2
    return factor


def downsample(accum: np.ndarray, h: int, w: int, factor: int) -> tuple[np.ndarray, int, int]:
    if factor <= 1:
        return accum, h, w
    hh, ww = h // factor, w // factor
    grid = accum.reshape(3, h, w)[:, : hh * factor, : ww * factor]
    out = grid[:, 0::factor, 0::factor].copy()
    for dy in range(factor):
        for dx in range(factor):
            if dy or dx:
                out += grid[:, dy::factor, dx::factor]
    return out.reshape(3, hh * ww), hh, ww


def unpack_rgb(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    bgrx = packed.view(np.uint8).reshape(h, w, 4)
    return bgrx[..., 2::-1]


def apply_glow(packed: np.ndarray, h: int, w: int, glow: float) -> None:
    if not glow > 0.01 or h < 16 or w < 16:
        return
    surface = pygame.image.frombuffer(packed, (w, h), "BGRA")
    small_size = (max(2, w // 8), max(2, h // 8))
    small = pygame.transform.smoothscale(surface, small_size)
    small = pygame.transform.gaussian_blur(small, max(1, min(small_size) // 48 + 2))
    level = int(max(0.0, min(1.0, glow)) * 255)
    small.fill((level, level, level), special_flags=pygame.BLEND_RGB_MULT)
    halo = pygame.transform.smoothscale(small, (w, h))
    surface.blit(halo, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
