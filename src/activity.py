from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

F32 = np.float32

STATE_LABELS: dict[str, tuple[str, ...]] = {
    "physarum": ("following", "steering", "exploring"),
    "boids": ("flocking", "gathering", "avoiding", "wandering"),
    "flow": ("streaming", "spawning", "fading"),
    "turmite": ("building", "erasing"),
}
STATUS_LABELS = ("drawing new", "reworking", "idle")
CREATING, REINFORCING, IDLE = 0, 1, 2

SAMPLE_SIZE = 3000
TOP_COUNT = 5
DECAY = F32(0.7)
CLASSIFY_STRIDE = 2
CREATING_THRESHOLD = F32(0.4)
FRONTIER_BRIGHTNESS = 0.35
FRONTIER_LEVEL = float(-np.log(1.0 - FRONTIER_BRIGHTNESS**2))
SCALE_REFRESH = 30
RANKING_WARMUP = 60
CHECK_INTERVAL = 20
MIN_STEPS = 200
PATIENCE = 5
NOVELTY_RATIO = 0.03
CHANGE_THRESHOLD = 0.01
AUTO_STEP_CAP = 50_000
MODEL_STEP_CAPS = {"physarum": 20_000, "boids": 8_000, "flow": 10_000, "turmite": 20_000}


@dataclass(frozen=True)
class AgentSample:
    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    status: np.ndarray
    width: int
    height: int
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None


@dataclass(frozen=True)
class Contributor:
    agent_id: int
    behaviour: str
    status: str
    created: int
    x: float
    y: float


@dataclass(frozen=True)
class Telemetry:
    model: str
    steps: int
    agents: int
    status_counts: tuple[int, int, int]
    behaviour_labels: tuple[str, ...]
    behaviour_counts: tuple[int, ...]
    coverage: float
    new_pixels: float
    progress: float
    eta_seconds: float | None
    elapsed: float
    complete: bool
    reason: str
    step_limit: int
    auto_limit: bool
    top: tuple[Contributor, ...]
    width: int
    height: int
    paused: bool = False


def _per_agent(mask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if mask.size == n:
        return mask, mask.astype(np.int64)
    if n > 0 and mask.size % n == 0:
        counts = mask.reshape(-1, n).sum(axis=0, dtype=np.int64)
        return counts > 0, counts
    agent = np.arange(mask.size) % max(n, 1)
    counts = np.bincount(agent[mask], minlength=n).astype(np.int64)
    return counts > 0, counts


class ActivityTracker:
    def __init__(self, model_name: str, n_agents: int, h: int, w: int, seed: int) -> None:
        self.model_name = model_name
        self.labels = STATE_LABELS.get(model_name, ("active",))
        self.n = max(1, int(n_agents))
        self.h = h
        self.w = w
        self.painted = np.zeros(h * w, dtype=bool)
        self.activity = np.zeros(self.n, dtype=F32)
        self.created = np.zeros(self.n, dtype=np.int64)
        self.deposited = np.zeros(self.n, dtype=bool)
        rng = np.random.default_rng((seed * 40503 + 17) % (2**32))
        k = min(self.n, SAMPLE_SIZE)
        self.sample_ids = np.sort(rng.choice(self.n, size=k, replace=False)) if k < self.n else np.arange(self.n)
        gy = np.linspace(0, h - 1, min(h, 48)).astype(np.int64)
        gx = np.linspace(0, w - 1, min(w, 48)).astype(np.int64)
        self.probe = (gy[:, None] * w + gx[None, :]).ravel()
        self.reset()

    def reset(self) -> None:
        self.painted.fill(False)
        self.activity.fill(1.0)
        self.created.fill(0)
        self.deposited.fill(False)
        self.frontier_value: float | None = None
        self.observed = 0
        self.new_ema = 0.0
        self.peak = 0.0
        self.last_new = 0
        self.change_ema: float | None = None
        self.prev_thumb: np.ndarray | None = None
        self.smooth_thumb: np.ndarray | None = None
        self.hits = 0
        self.progress = 0.0
        self.complete = False
        self.reason = ""
        self.elapsed = 0.0
        self.history: deque[tuple[float, float]] = deque(maxlen=40)

    def refresh_scale(self, accum: np.ndarray, exposure: float) -> None:
        stride = max(1, accum.shape[1] // 32768)
        sample = accum[:, ::stride].max(axis=0)
        sample = sample[(sample > 0) & np.isfinite(sample)]
        if sample.size < 8:
            self.frontier_value = None
            return
        scale = float(np.percentile(sample, 99.5))
        if np.isfinite(scale) and scale > 0 and exposure > 0:
            self.frontier_value = FRONTIER_LEVEL * scale / exposure

    def observe(
        self,
        index: np.ndarray,
        weight: np.ndarray | float,
        accum: np.ndarray | None = None,
        exposure: float = 1.0,
        overwrite: bool = False,
    ) -> None:
        if index.size == 0:
            self.deposited.fill(False)
            self.activity *= DECAY
            return
        if accum is not None and self.observed % SCALE_REFRESH == 0:
            self.refresh_scale(accum, exposure)
        classify = self.observed % CLASSIFY_STRIDE == 0
        self.observed += 1
        fresh = ~self.painted[index]
        self.painted[index] = True
        positive = weight > 0 if isinstance(weight, np.ndarray) else None
        if positive is not None:
            fresh &= positive
        new = int(np.count_nonzero(fresh))
        self.last_new = new
        self.new_ema += (new - self.new_ema) * 0.05
        self.peak = max(self.peak, self.new_ema)
        if not classify:
            return
        if overwrite:
            frontier = positive if positive is not None else np.full(index.size, bool(weight > 0))
            self.deposited.fill(True)
        else:
            if accum is None or self.frontier_value is None:
                frontier = np.ones(index.size, dtype=bool)
            else:
                brightest = np.take(accum[0], index, mode="clip")
                np.maximum(brightest, np.take(accum[1], index, mode="clip"), out=brightest)
                np.maximum(brightest, np.take(accum[2], index, mode="clip"), out=brightest)
                with np.errstate(invalid="ignore"):
                    frontier = brightest < F32(self.frontier_value)
            if positive is not None:
                frontier &= positive
                self.deposited, _ = _per_agent(positive, self.n)
            else:
                self.deposited.fill(bool(weight > 0))
        frontier_any, frontier_counts = _per_agent(frontier, self.n)
        self.created += frontier_counts * CLASSIFY_STRIDE
        self.activity *= DECAY
        self.activity += frontier_any.astype(F32) * (F32(1.0) - DECAY)

    def check(self, accum: np.ndarray, steps: int, fade: float, step_limit: int) -> None:
        thumb = accum[:, self.probe].astype(F32)
        finite = thumb[np.isfinite(thumb)]
        level = float(np.percentile(finite, 99)) if finite.size else 0.0
        if not np.isfinite(level) or level <= 0:
            level = 1.0
        thumb = np.log1p(np.nan_to_num(thumb, nan=0.0, posinf=0.0, neginf=0.0) * F32(4.0 / level))
        thumb = thumb.mean(axis=0)
        if self.smooth_thumb is None:
            self.smooth_thumb = thumb
        else:
            self.smooth_thumb = self.smooth_thumb + (thumb - self.smooth_thumb) * F32(0.25)
        if self.prev_thumb is not None:
            delta = float(np.mean(np.abs(self.smooth_thumb - self.prev_thumb))) / (float(np.mean(self.smooth_thumb)) + 1e-6)
            self.change_ema = delta if self.change_ema is None else 0.7 * self.change_ema + 0.3 * delta
        self.prev_thumb = self.smooth_thumb.copy()
        if self.complete:
            return
        ratio = self.new_ema / self.peak if self.peak > 0 else 1.0
        change = self.change_ema if self.change_ema is not None else 1.0
        min_steps = MIN_STEPS + (int(min(3.0 / fade, 1_000_000.0)) if fade > 1e-9 else 0)
        settled = steps >= min_steps and ratio < NOVELTY_RATIO and change < CHANGE_THRESHOLD
        self.hits = self.hits + 1 if settled else 0
        novelty_part = min(1.0, max(0.0, (1.0 - ratio) / (1.0 - NOVELTY_RATIO)))
        visual_part = min(1.0, CHANGE_THRESHOLD / max(change, 1e-9))
        warmup = min(1.0, steps / min_steps)
        converge = warmup * (0.3 * novelty_part + 0.7 * visual_part) * 0.95 + 0.05 * self.hits / PATIENCE
        limit = self.step_cap(step_limit)
        budget = steps / limit
        estimate = min(0.995, max(converge, budget))
        self.progress = max(self.progress, estimate)
        if self.hits >= PATIENCE:
            self._finish("converged: no new territory and the image has settled")
        elif steps >= limit:
            self._finish("step target reached" if step_limit > 0 else "automatic step limit reached (this swarm keeps moving)")
        self.history.append((self.elapsed, self.progress))

    def step_cap(self, step_limit: int) -> int:
        return step_limit if step_limit > 0 else MODEL_STEP_CAPS.get(self.model_name, AUTO_STEP_CAP)

    def finish(self, reason: str) -> None:
        if not self.complete:
            self._finish(reason)
            self.history.append((self.elapsed, self.progress))

    def _finish(self, reason: str) -> None:
        self.complete = True
        self.progress = 1.0
        self.reason = reason

    def eta(self) -> float | None:
        if self.complete:
            return 0.0
        if len(self.history) < 4:
            return None
        t0, p0 = self.history[0]
        t1, p1 = self.history[-1]
        if t1 - t0 <= 1e-6 or p1 - p0 <= 1e-6:
            return None
        return max(0.0, (1.0 - self.progress) * (t1 - t0) / (p1 - p0))

    def statuses(self, ids: np.ndarray | None = None) -> np.ndarray:
        activity = self.activity if ids is None else self.activity[ids]
        deposited = self.deposited if ids is None else self.deposited[ids]
        out = np.full(activity.size, IDLE, dtype=np.uint8)
        out[deposited] = REINFORCING
        out[activity > CREATING_THRESHOLD] = CREATING
        return out

    def sample(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray | None = None, vy: np.ndarray | None = None) -> AgentSample:
        ids = self.sample_ids[self.sample_ids < x.size]
        has_velocity = vx is not None and vy is not None and vx.size == x.size and vy.size == x.size
        return AgentSample(
            ids=ids.copy(),
            x=np.asarray(x[ids], dtype=F32).copy(),
            y=np.asarray(y[ids], dtype=F32).copy(),
            status=self.statuses(ids),
            width=self.w,
            height=self.h,
            vx=np.asarray(vx[ids], dtype=F32).copy() if has_velocity else None,
            vy=np.asarray(vy[ids], dtype=F32).copy() if has_velocity else None,
        )

    def telemetry(self, model_name: str, states: np.ndarray, x: np.ndarray, y: np.ndarray, steps: int, step_limit: int) -> Telemetry:
        status = self.statuses()
        status_counts = np.bincount(status, minlength=3)
        behaviour = np.bincount(np.minimum(states, len(self.labels) - 1), minlength=len(self.labels))
        k = min(TOP_COUNT, self.n)
        top_ids = np.argpartition(-self.created, k - 1)[:k] if self.n > k else np.arange(self.n)
        top_ids = top_ids[np.argsort(-self.created[top_ids], kind="stable")]
        top = tuple(
            Contributor(
                agent_id=int(i),
                behaviour=self.labels[min(int(states[i]), len(self.labels) - 1)],
                status=STATUS_LABELS[int(status[i])],
                created=int(self.created[i]),
                x=float(x[i]),
                y=float(y[i]),
            )
            for i in top_ids
            if self.created[i] > 0 and self.observed >= RANKING_WARMUP
        )
        return Telemetry(
            model=model_name,
            steps=steps,
            agents=self.n,
            status_counts=(int(status_counts[0]), int(status_counts[1]), int(status_counts[2])),
            behaviour_labels=self.labels,
            behaviour_counts=tuple(int(v) for v in behaviour),
            coverage=float(np.count_nonzero(self.painted)) / float(self.painted.size),
            new_pixels=float(self.new_ema),
            progress=float(self.progress),
            eta_seconds=self.eta(),
            elapsed=float(self.elapsed),
            complete=self.complete,
            reason=self.reason,
            step_limit=self.step_cap(step_limit),
            auto_limit=step_limit <= 0,
            top=top,
            width=self.w,
            height=self.h,
        )
