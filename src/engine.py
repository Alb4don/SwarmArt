from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from .activity import CHECK_INTERVAL, ActivityTracker
from .config import Params, structural_change
from .fields import TAU, PaletteLUT, blur3_wrap, curl_field

log = logging.getLogger("swarmart.engine")

F32 = np.float32
PAD = 64


@dataclass
class Deposit:
    index: np.ndarray
    colors: np.ndarray
    weight: np.ndarray | float
    overwrite: bool = False


class Model:
    name = "base"

    def __init__(self, params: Params, rng: np.random.Generator, h: int, w: int) -> None:
        self.p = params
        self.rng = rng
        self.h = h
        self.w = w
        self.phase = 0.0
        self.lut = PaletteLUT()
        self.agent_state = np.zeros(0, dtype=np.uint8)

    def update(self, params: Params) -> None:
        self.p = params

    def step(self) -> Deposit:
        raise NotImplementedError

    def clear(self) -> None:
        return None

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def respawn(self, fraction: float) -> int:
        raise NotImplementedError

    def inject(self, x: float, y: float) -> None:
        raise NotImplementedError

    def repair(self) -> bool:
        return False

    def velocities(self) -> tuple[np.ndarray, np.ndarray] | None:
        return None

    def _advance_phase(self) -> float:
        self.phase = (self.phase + 1.5e-4) % 1.0
        return self.phase

    def _flat_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = x.astype(np.int32)
        iy = y.astype(np.int32)
        np.clip(ix, 0, self.w - 1, out=ix)
        np.clip(iy, 0, self.h - 1, out=iy)
        return iy * np.int32(self.w) + ix

    def _pick(self, n: int, fraction: float) -> np.ndarray:
        count = int(max(1, min(n, round(n * fraction))))
        return self.rng.choice(n, size=count, replace=False) if count < n else np.arange(n)

    def _repair_arrays(self, specs: tuple[tuple[str, float], ...]) -> bool:
        repaired = False
        for attr, scale in specs:
            arr = getattr(self, attr)
            bad = ~np.isfinite(arr)
            if bad.any():
                arr[bad] = self.rng.random(int(bad.sum()), dtype=F32) * F32(scale)
                repaired = True
        return repaired


class Physarum(Model):
    name = "physarum"

    def __init__(self, params: Params, rng: np.random.Generator, h: int, w: int) -> None:
        super().__init__(params, rng, h, w)
        n = params.agents
        theta = rng.random(n, dtype=F32) * TAU
        radius = np.sqrt(rng.random(n, dtype=F32)) * F32(0.35 * min(h, w))
        self.x = ((F32(w * 0.5) + np.cos(theta) * radius) % F32(w)).astype(F32)
        self.y = ((F32(h * 0.5) + np.sin(theta) * radius) % F32(h)).astype(F32)
        self.a = ((theta + F32(np.pi)) % TAU).astype(F32)
        self.agent_state = np.zeros(n, dtype=np.uint8)
        self.trail = np.zeros((h, w), dtype=F32)
        self._wrap_x = (np.arange(-PAD, w + 2 * PAD) % w).astype(np.int32)
        self._wrap_y = ((np.arange(-PAD, h + 2 * PAD) % h) * w).astype(np.int32)

    def _sense(self, ca: np.ndarray, sa: np.ndarray, offset: float) -> np.ndarray:
        co = F32(math.cos(offset) * self.p.sensor_dist)
        so = F32(math.sin(offset) * self.p.sensor_dist)
        sx = np.take(self._wrap_x, (self.x + (ca * co - sa * so) + F32(PAD)).astype(np.int32), mode="clip")
        sy = np.take(self._wrap_y, (self.y + (sa * co + ca * so) + F32(PAD)).astype(np.int32), mode="clip")
        return np.take(self.trail.ravel(), sy + sx, mode="clip")

    def step(self) -> Deposit:
        p = self.p
        n = self.x.size
        ca = np.cos(self.a)
        sa = np.sin(self.a)
        front = self._sense(ca, sa, 0.0)
        left = self._sense(ca, sa, p.sensor_angle)
        right = self._sense(ca, sa, -p.sensor_angle)
        turn = F32(p.turn)
        rnd = self.rng.random(n, dtype=F32)
        steer = np.where(left > right, turn, -turn)
        steer = np.where((front < left) & (front < right), (rnd - F32(0.5)) * (F32(2.0) * turn), steer)
        following = (front >= left) & (front >= right)
        exploring = (front < left) & (front < right)
        steer = np.where(following, F32(0.0), steer)
        self.agent_state = np.where(following, 0, np.where(exploring, 2, 1)).astype(np.uint8)
        jitter = (self.rng.random(n, dtype=F32) - F32(0.5)) * F32(p.jitter * 0.6)
        self.a = ((self.a + steer + jitter) % TAU).astype(F32)
        speed = F32(p.speed)
        self.x = (self.x + np.cos(self.a) * speed) % F32(self.w)
        self.y = (self.y + np.sin(self.a) * speed) % F32(self.h)
        index = self._flat_index(self.x, self.y)
        np.add.at(self.trail.ravel(), index, F32(p.deposit))
        diffused = blur3_wrap(self.trail)
        diffused *= F32(p.decay / 9.0)
        np.minimum(diffused, F32(1e4), out=diffused)
        self.trail = diffused
        hue = self.a * F32(p.hue_spread / (2.0 * math.pi)) + F32(self._advance_phase())
        return Deposit(index, self.lut(hue, p.palette), 0.5 * p.deposit)

    def clear(self) -> None:
        self.trail.fill(0.0)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def velocities(self) -> tuple[np.ndarray, np.ndarray]:
        speed = F32(self.p.speed)
        return np.cos(self.a) * speed, np.sin(self.a) * speed

    def respawn(self, fraction: float) -> int:
        sel = self._pick(self.x.size, fraction)
        self.x[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.w)
        self.y[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.h)
        self.a[sel] = self.rng.random(sel.size, dtype=F32) * TAU
        return int(sel.size)

    def inject(self, x: float, y: float) -> None:
        sel = self._pick(self.x.size, 0.03)
        spread = F32(max(4.0, 0.02 * min(self.w, self.h)))
        self.x[sel] = (F32(x) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.w)
        self.y[sel] = (F32(y) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.h)
        self.a[sel] = self.rng.random(sel.size, dtype=F32) * TAU
        r = int(spread * 2)
        cx, cy = int(x), int(y)
        ys = np.arange(cy - r, cy + r + 1) % self.h
        xs = np.arange(cx - r, cx + r + 1) % self.w
        d = np.arange(-r, r + 1, dtype=F32)
        blob = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2.0 * spread * spread)).astype(F32)
        self.trail[np.ix_(ys, xs)] += blob * F32(20.0 * self.p.deposit)

    def repair(self) -> bool:
        fixed = self._repair_arrays((("x", self.w), ("y", self.h), ("a", 2 * math.pi)))
        if not np.isfinite(self.trail).all():
            np.nan_to_num(self.trail, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            fixed = True
        return fixed


class Boids(Model):
    name = "boids"

    def __init__(self, params: Params, rng: np.random.Generator, h: int, w: int) -> None:
        super().__init__(params, rng, h, w)
        n = params.agents
        self.x = rng.random(n, dtype=F32) * F32(w)
        self.y = rng.random(n, dtype=F32) * F32(h)
        angle = rng.random(n, dtype=F32) * TAU
        self.vx = np.cos(angle) * F32(params.speed)
        self.vy = np.sin(angle) * F32(params.speed)
        self.agent_state = np.full(n, 3, dtype=np.uint8)

    @staticmethod
    def _normalized(grid: np.ndarray) -> np.ndarray:
        peak = float(np.abs(grid).max())
        return grid * F32(1.0 / peak) if peak > 1e-9 else grid

    def step(self) -> Deposit:
        p = self.p
        n = self.x.size
        cell_size = int(min(64, max(6, round(p.sensor_dist * 1.5))))
        gw = max(3, self.w // cell_size)
        gh = max(3, self.h // cell_size)
        cx = np.minimum((self.x * F32(gw / self.w)).astype(np.int64), gw - 1)
        cy = np.minimum((self.y * F32(gh / self.h)).astype(np.int64), gh - 1)
        np.maximum(cx, 0, out=cx)
        np.maximum(cy, 0, out=cy)
        cell = cy * gw + cx
        size = gw * gh
        count = np.bincount(cell, minlength=size).astype(F32).reshape(gh, gw)
        sum_vx = np.bincount(cell, weights=self.vx, minlength=size).astype(F32).reshape(gh, gw)
        sum_vy = np.bincount(cell, weights=self.vy, minlength=size).astype(F32).reshape(gh, gw)
        near = blur3_wrap(count)
        mean_vx = (blur3_wrap(sum_vx) / np.maximum(near, F32(1.0))).ravel()
        mean_vy = (blur3_wrap(sum_vy) / np.maximum(near, F32(1.0))).ravel()
        far = blur3_wrap(blur3_wrap(near))
        cohesion_x = self._normalized(np.roll(far, -1, 1) - np.roll(far, 1, 1)).ravel()
        cohesion_y = self._normalized(np.roll(far, -1, 0) - np.roll(far, 1, 0)).ravel()
        crowd_x = self._normalized(np.roll(count, -1, 1) - np.roll(count, 1, 1)).ravel()
        crowd_y = self._normalized(np.roll(count, -1, 0) - np.roll(count, 1, 0)).ravel()
        noise = self.rng.standard_normal((2, n), dtype=F32) * F32(p.jitter * 0.6)
        align, cohere, separate = F32(p.alignment), F32(p.cohesion), F32(p.separation)
        ax = align * (mean_vx[cell] - self.vx) + cohere * cohesion_x[cell] - separate * crowd_x[cell] + noise[0]
        ay = align * (mean_vy[cell] - self.vy) + cohere * cohesion_y[cell] - separate * crowd_y[cell] + noise[1]
        separation_force = separate * np.hypot(crowd_x[cell], crowd_y[cell])
        cohesion_force = cohere * np.hypot(cohesion_x[cell], cohesion_y[cell])
        misalignment = np.hypot(mean_vx[cell] - self.vx, mean_vy[cell] - self.vy)
        neighbours = near.ravel()[cell]
        state = np.full(n, 3, dtype=np.uint8)
        state[cohesion_force > F32(0.2)] = 1
        state[(neighbours >= 3) & (misalignment < F32(0.5 * p.speed))] = 0
        state[separation_force > np.maximum(cohesion_force, F32(0.3))] = 2
        self.agent_state = state
        self.vx += ax * F32(0.2)
        self.vy += ay * F32(0.2)
        speed = np.sqrt(self.vx * self.vx + self.vy * self.vy)
        factor = F32(p.speed) / np.maximum(speed, F32(1e-6))
        self.vx *= factor
        self.vy *= factor
        self.x = (self.x + self.vx) % F32(self.w)
        self.y = (self.y + self.vy) % F32(self.h)
        index = self._flat_index(self.x, self.y)
        heading = np.arctan2(self.vy, self.vx)
        hue = heading * F32(p.hue_spread / (2.0 * math.pi)) + F32(self._advance_phase())
        return Deposit(index, self.lut(hue, p.palette), 0.4)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def velocities(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vx, self.vy

    def respawn(self, fraction: float) -> int:
        sel = self._pick(self.x.size, fraction)
        self.x[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.w)
        self.y[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.h)
        angle = self.rng.random(sel.size, dtype=F32) * TAU
        self.vx[sel] = np.cos(angle) * F32(self.p.speed)
        self.vy[sel] = np.sin(angle) * F32(self.p.speed)
        return int(sel.size)

    def inject(self, x: float, y: float) -> None:
        sel = self._pick(self.x.size, 0.04)
        spread = F32(max(3.0, 0.015 * min(self.w, self.h)))
        self.x[sel] = (F32(x) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.w)
        self.y[sel] = (F32(y) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.h)
        angle = self.rng.random(sel.size, dtype=F32) * TAU
        self.vx[sel] = np.cos(angle) * F32(self.p.speed)
        self.vy[sel] = np.sin(angle) * F32(self.p.speed)

    def repair(self) -> bool:
        return self._repair_arrays((("x", self.w), ("y", self.h), ("vx", 0.0), ("vy", 0.0)))


class Flow(Model):
    name = "flow"

    def __init__(self, params: Params, rng: np.random.Generator, h: int, w: int) -> None:
        super().__init__(params, rng, h, w)
        n = params.agents
        self.x = rng.random(n, dtype=F32) * F32(w)
        self.y = rng.random(n, dtype=F32) * F32(h)
        self.vx = np.zeros(n, dtype=F32)
        self.vy = np.zeros(n, dtype=F32)
        self.life = rng.integers(20, 320, n).astype(np.int32)
        self.agent_age = np.zeros(n, dtype=np.int32)
        self.agent_state = np.zeros(n, dtype=np.uint8)
        self.age = 0
        self._field_scale = -1.0
        self._build_field()

    def _build_field(self) -> None:
        cells = self.p.noise_scale * max(self.h, self.w) / 512.0
        field_rng = np.random.default_rng(self.p.seed + 101)
        self.fx, self.fy = curl_field(self.h, self.w, cells, field_rng)
        self._field_scale = self.p.noise_scale

    def update(self, params: Params) -> None:
        super().update(params)
        if params.noise_scale != self._field_scale:
            self._build_field()

    def step(self) -> Deposit:
        p = self.p
        n = self.x.size
        self.age += 1
        index = self._flat_index(self.x, self.y)
        phi = 0.35 * math.sin(self.age * 0.004)
        c, s = F32(math.cos(phi)), F32(math.sin(phi))
        fx = self.fx.ravel()[index]
        fy = self.fy.ravel()[index]
        tx = (fx * c - fy * s) * F32(p.speed)
        ty = (fx * s + fy * c) * F32(p.speed)
        k = F32(min(1.0, p.turn))
        self.vx += (tx - self.vx) * k
        self.vy += (ty - self.vy) * k
        if p.jitter > 0:
            noise = self.rng.standard_normal((2, n), dtype=F32) * F32(p.jitter * 0.15)
            self.vx += noise[0]
            self.vy += noise[1]
        self.x = (self.x + self.vx) % F32(self.w)
        self.y = (self.y + self.vy) % F32(self.h)
        self.life -= 1
        self.agent_age += 1
        dead = self.life <= 0
        if dead.any():
            count = int(dead.sum())
            self.x[dead] = self.rng.random(count, dtype=F32) * F32(self.w)
            self.y[dead] = self.rng.random(count, dtype=F32) * F32(self.h)
            self.vx[dead] = 0.0
            self.vy[dead] = 0.0
            self.life[dead] = self.rng.integers(40, 320, count).astype(np.int32)
            self.agent_age[dead] = 0
        state = np.zeros(n, dtype=np.uint8)
        state[self.life < 30] = 2
        state[self.agent_age < 20] = 1
        self.agent_state = state
        index = self._flat_index(self.x, self.y)
        velocity = np.sqrt(self.vx * self.vx + self.vy * self.vy) / F32(max(p.speed, 1e-3))
        heading = np.arctan2(self.vy, self.vx)
        hue = heading * F32(p.hue_spread / (2.0 * math.pi)) + velocity * F32(0.25) + F32(self._advance_phase())
        taper = np.minimum(self.life.astype(F32) * F32(1.0 / 30.0), F32(1.0))
        return Deposit(index, self.lut(hue, p.palette), taper * F32(0.6 * p.deposit))

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def velocities(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vx, self.vy

    def respawn(self, fraction: float) -> int:
        sel = self._pick(self.x.size, fraction)
        self.x[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.w)
        self.y[sel] = self.rng.random(sel.size, dtype=F32) * F32(self.h)
        self.vx[sel] = 0.0
        self.vy[sel] = 0.0
        self.agent_age[sel] = 0
        return int(sel.size)

    def inject(self, x: float, y: float) -> None:
        sel = self._pick(self.x.size, 0.05)
        spread = F32(max(3.0, 0.02 * min(self.w, self.h)))
        self.x[sel] = (F32(x) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.w)
        self.y[sel] = (F32(y) + self.rng.normal(0, 1, sel.size).astype(F32) * spread) % F32(self.h)
        self.vx[sel] = 0.0
        self.vy[sel] = 0.0
        self.life[sel] = self.rng.integers(120, 400, sel.size).astype(np.int32)
        self.agent_age[sel] = 0

    def repair(self) -> bool:
        return self._repair_arrays((("x", self.w), ("y", self.h), ("vx", 0.0), ("vy", 0.0)))


class Turmite(Model):
    name = "turmite"
    MAX_ANTS = 8192
    DX = np.array([1, 0, -1, 0], dtype=np.int64)
    DY = np.array([0, 1, 0, -1], dtype=np.int64)
    RULES = (
        "RL",
        "LLRR",
        "LRRL",
        "RLLR",
        "LRRRRRLLR",
        "LLRRRLRLRLLR",
        "RRLLLRLLLRRR",
        "RRLRLLRRRRRR",
        "LLRR",
        "RLR",
        "RRLL",
        "LRRRRLLLRRR",
    )

    def __init__(self, params: Params, rng: np.random.Generator, h: int, w: int) -> None:
        super().__init__(params, rng, h, w)
        rule_rng = np.random.default_rng((params.seed * 2654435761) % (2**32))
        rule = self.RULES[int(rule_rng.integers(0, len(self.RULES)))]
        self.rule = rule
        self.states = len(rule)
        self.table = np.array([1 if ch == "R" else -1 for ch in rule], dtype=np.int64)
        self.grid = np.zeros((h, w), dtype=np.uint8)
        n = min(params.agents, self.MAX_ANTS)
        colonies = int(min(n, max(1, rule_rng.integers(1, 5))))
        centers_x = rng.uniform(0.15, 0.85, colonies) * w
        centers_y = rng.uniform(0.15, 0.85, colonies) * h
        member = rng.integers(0, colonies, n)
        sigma = 0.004 * min(h, w)
        self.x = np.round(centers_x[member] + rng.normal(0, sigma, n)).astype(np.int64) % w
        self.y = np.round(centers_y[member] + rng.normal(0, sigma, n)).astype(np.int64) % h
        self.d = rng.integers(0, 4, n).astype(np.int64)
        self.agent_state = np.zeros(n, dtype=np.uint8)

    def step(self) -> Deposit:
        p = self.p
        substeps = max(1, int(round(p.speed * 16)))
        indices = []
        states = []
        headings = []
        for _ in range(substeps):
            s = self.grid[self.y, self.x]
            new_state = (s + 1) % self.states
            self.d = (self.d + self.table[s]) & 3
            self.grid[self.y, self.x] = new_state
            indices.append(self.y * self.w + self.x)
            states.append(new_state)
            headings.append(self.d)
            self.x = (self.x + self.DX[self.d]) % self.w
            self.y = (self.y + self.DY[self.d]) % self.h
        self.agent_state = (states[-1] == 0).astype(np.uint8)
        index = np.concatenate(indices)
        raw_state = np.concatenate(states)
        state = raw_state.astype(F32) * F32(1.0 / self.states)
        heading = np.concatenate(headings).astype(F32) * F32(0.25)
        hue = state * F32(0.35 + p.hue_spread) + heading * F32(p.hue_spread * 0.15) + F32(self._advance_phase())
        weight = np.where(raw_state > 0, F32(p.deposit), F32(0.0)).astype(F32)
        return Deposit(index, self.lut(hue, p.palette), weight, overwrite=True)

    def clear(self) -> None:
        self.grid.fill(0)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def velocities(self) -> tuple[np.ndarray, np.ndarray]:
        reach = F32(min(6, max(1, int(round(self.p.speed * 16)))))
        return self.DX[self.d].astype(F32) * reach, self.DY[self.d].astype(F32) * reach

    def respawn(self, fraction: float) -> int:
        sel = self._pick(self.x.size, fraction)
        self.x[sel] = self.rng.integers(0, self.w, sel.size)
        self.y[sel] = self.rng.integers(0, self.h, sel.size)
        return int(sel.size)

    def inject(self, x: float, y: float) -> None:
        sel = self._pick(self.x.size, 0.1)
        self.x[sel] = int(x) % self.w
        self.y[sel] = int(y) % self.h
        self.d[sel] = self.rng.integers(0, 4, sel.size)


MODEL_TYPES: dict[str, type[Model]] = {cls.name: cls for cls in (Physarum, Boids, Flow, Turmite)}


class HealthMonitor:
    def __init__(self, interval: int = 30, patience: int = 3, threshold: float = 0.06) -> None:
        self.interval = interval
        self.patience = patience
        self.threshold = threshold
        self.tick = 0
        self.collapse_hits = 0

    def occupancy(self, x: np.ndarray, y: np.ndarray, w: int, h: int, rng: np.random.Generator) -> float:
        n = x.size
        if n == 0:
            return 1.0
        if n > 4096:
            pick = rng.integers(0, n, 4096)
            x, y = x[pick], y[pick]
            n = 4096
        grid = 16
        cx = np.clip((np.asarray(x, dtype=np.float64) * grid / w).astype(np.int64), 0, grid - 1)
        cy = np.clip((np.asarray(y, dtype=np.float64) * grid / h).astype(np.int64), 0, grid - 1)
        occupied = np.unique(cy * grid + cx).size
        return occupied / min(grid * grid, n)

    def check(self, sim: Simulation) -> list[str]:
        self.tick += 1
        if self.tick % self.interval:
            return []
        events = []
        if not np.isfinite(sim.accum).all():
            np.nan_to_num(sim.accum, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            events.append("nonfinite_canvas")
        if sim.model.repair():
            events.append("nonfinite_agents")
        x, y = sim.model.positions()
        if x.size >= 64:
            occ = self.occupancy(x, y, sim.w, sim.h, sim.rng)
            if occ < self.threshold:
                self.collapse_hits += 1
            else:
                self.collapse_hits = 0
            if self.collapse_hits >= self.patience:
                sim.model.respawn(0.25)
                self.collapse_hits = 0
                events.append("swarm_collapse")
        for event in events:
            log.info("health recovery: %s (model=%s)", event, sim.params.model)
        return events


class Simulation:
    def __init__(self, params: Params) -> None:
        self._install(params, self._make_state(params))

    @staticmethod
    def _make_state(p: Params) -> tuple[np.random.Generator, np.ndarray, Model]:
        rng = np.random.default_rng(p.seed)
        accum = np.zeros((3, p.height * p.width), dtype=F32)
        model = MODEL_TYPES[p.model](p, rng, p.height, p.width)
        return rng, accum, model

    def _install(self, params: Params, state: tuple[np.random.Generator, np.ndarray, Model]) -> None:
        self.rng, self.accum, self.model = state
        self.params = params
        self.h, self.w = params.height, params.width
        self.size = self.h * self.w
        self.health = HealthMonitor()
        self.steps = 0
        self.activity = ActivityTracker(params.model, self.model.positions()[0].size, self.h, self.w, params.seed)

    def apply(self, params: Params) -> bool:
        if structural_change(self.params, params):
            self._install(params, self._make_state(params))
            return True
        self.model.update(params)
        self.params = params
        return False

    def reset(self) -> None:
        self._install(self.params, self._make_state(self.params))

    def clear(self) -> None:
        self.accum.fill(0.0)
        self.model.clear()
        self.activity.reset()
        self.steps = 0

    def inject(self, fx: object, fy: object) -> bool:
        if isinstance(fx, bool) or isinstance(fy, bool) or not isinstance(fx, (int, float)) or not isinstance(fy, (int, float)):
            return False
        fx, fy = float(fx), float(fy)
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return False
        fx = min(max(fx, 0.0), 1.0)
        fy = min(max(fy, 0.0), 1.0)
        self.model.inject(fx * (self.w - 1), fy * (self.h - 1))
        return True

    def _splat(self, dep: Deposit) -> None:
        weight = dep.weight[:, None] if isinstance(dep.weight, np.ndarray) else F32(dep.weight)
        values = dep.colors * weight
        for c in range(3):
            if dep.overwrite:
                self.accum[c][dep.index] = values[:, c]
            else:
                np.add.at(self.accum[c], dep.index, values[:, c])

    def step(self, n: int = 1) -> list[str]:
        events: list[str] = []
        fade = self.params.fade
        target = self.params.target_steps
        was_complete = self.activity.complete
        with np.errstate(invalid="ignore", over="ignore"):
            for _ in range(max(1, n)):
                if fade > 0:
                    self.accum *= F32(1.0 - fade)
                started = time.perf_counter()
                deposit = self.model.step()
                self.activity.observe(deposit.index, deposit.weight, self.accum, self.params.exposure, deposit.overwrite)
                self._splat(deposit)
                self.steps += 1
                events.extend(self.health.check(self))
                self.activity.elapsed += time.perf_counter() - started
                if self.steps % CHECK_INTERVAL == 0 or (target > 0 and self.steps >= target):
                    self.activity.check(self.accum, self.steps, fade, target)
                if self.activity.complete and not was_complete:
                    break
        return events

    @property
    def complete(self) -> bool:
        return self.activity.complete

    def finish(self, reason: str) -> None:
        self.activity.finish(reason)

    def telemetry(self):
        x, y = self.model.positions()
        states = self.model.agent_state
        if states.size != x.size:
            states = np.zeros(x.size, dtype=np.uint8)
        return self.activity.telemetry(self.params.model, states, x, y, self.steps, self.params.target_steps)

    def agent_sample(self):
        x, y = self.model.positions()
        velocity = self.model.velocities()
        if velocity is None:
            return self.activity.sample(x, y)
        return self.activity.sample(x, y, velocity[0], velocity[1])

    @property
    def agent_count(self) -> int:
        return int(self.model.positions()[0].size)
