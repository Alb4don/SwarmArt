from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import FLOAT_LIMITS, GENES, MODELS, Params, PresetError, from_json, random_palette, sanitize, to_json, with_model

log = logging.getLogger("swarmart.curator")

ARCHIVE_LIMIT = 64
NOVELTY_NEIGHBOURS = 5
NOVELTY_WEIGHT = 0.15
GENE_JITTER = 0.12
MAX_STATE_BYTES = 256 * 1024
DESCRIPTOR_KEYS = ("entropy", "coverage", "beta", "fractal_dim", "colorfulness", "hue_variety")
DESCRIPTOR_SCALE = np.array([1.0, 1.0, 4.0, 2.0, 0.5, 1.0])


@dataclass
class ArmStats:
    plays: int = 0
    reward_sum: float = 0.0
    reward_sq: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.plays if self.plays else 0.0

    @property
    def variance(self) -> float:
        if self.plays < 2:
            return 0.25
        return max(0.0, self.reward_sq / self.plays - self.mean**2)


@dataclass
class PieceResult:
    model: str
    seed: int
    score: float
    novelty: float
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)


def descriptor(metrics: dict[str, float]) -> np.ndarray | None:
    values = []
    for key, scale in zip(DESCRIPTOR_KEYS, DESCRIPTOR_SCALE, strict=True):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return None
        values.append(min(max(float(value) / scale, 0.0), 1.5))
    return np.asarray(values, dtype=np.float64)


class Curator:
    def __init__(self, rng: np.random.Generator | None = None, state_path: Path | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.state_path = state_path
        self.arms: dict[str, ArmStats] = {model: ArmStats() for model in MODELS}
        self.archive: list[np.ndarray] = []
        self.last_model: str | None = None
        self.pieces = 0
        self.best_score = 0.0
        self.last_result: PieceResult | None = None
        if state_path is not None:
            self.load()

    def choose_model(self, avoid: str | None = None) -> str:
        avoid = avoid if avoid in MODELS else self.last_model
        unplayed = [m for m in MODELS if self.arms[m].plays == 0 and m != avoid]
        if unplayed:
            return str(self.rng.choice(unplayed))
        candidates = [m for m in MODELS if m != avoid and self.arms[m].plays > 0]
        if not candidates:
            return avoid if avoid in MODELS else MODELS[0]
        total = sum(arm.plays for arm in self.arms.values())
        log_total = math.log(max(total, 2))
        best_model = candidates[0]
        best_value = -math.inf
        for model in candidates:
            arm = self.arms[model]
            bonus_var = arm.variance + math.sqrt(2.0 * log_total / arm.plays)
            value = arm.mean + math.sqrt(log_total / arm.plays * min(0.25, bonus_var))
            value += float(self.rng.uniform(0.0, 1e-6))
            if value > best_value:
                best_value = value
                best_model = model
        return best_model

    def propose(self, base: Params, avoid: str | None = None) -> Params:
        model = self.choose_model(avoid if avoid is not None else base.model)
        params = with_model(base, model)
        data = asdict(params)
        data["seed"] = int(self.rng.integers(0, 2**31 - 1))
        data["palette"] = list(random_palette(self.rng))
        data["target_steps"] = 0
        for name in GENES.get(model, ()):
            lo, hi = FLOAT_LIMITS[name]
            normalized = (data[name] - lo) / (hi - lo)
            normalized = min(1.0, max(0.0, normalized + float(self.rng.normal(0.0, GENE_JITTER))))
            data[name] = round(lo + (hi - lo) * normalized, 6)
        return sanitize(data)

    def novelty(self, vector: np.ndarray | None) -> float:
        if vector is None:
            return 0.0
        if not self.archive:
            return 1.0
        distances = np.sort(np.linalg.norm(np.asarray(self.archive) - vector, axis=1))
        nearest = distances[: min(NOVELTY_NEIGHBOURS, distances.size)]
        return float(min(1.0, nearest.mean() / 0.5))

    def record(self, params: Params, score: float, metrics: dict[str, float]) -> PieceResult:
        score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 0.0
        score = min(1.0, max(0.0, score))
        vector = descriptor(metrics)
        novelty = self.novelty(vector)
        reward = (1.0 - NOVELTY_WEIGHT) * score + NOVELTY_WEIGHT * novelty
        model = params.model if params.model in self.arms else MODELS[0]
        arm = self.arms[model]
        arm.plays += 1
        arm.reward_sum += reward
        arm.reward_sq += reward * reward
        if vector is not None:
            self.archive.append(vector)
            del self.archive[:-ARCHIVE_LIMIT]
        self.last_model = model
        self.pieces += 1
        self.best_score = max(self.best_score, score)
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        self.last_result = PieceResult(model, params.seed, score, novelty, reward, clean)
        self.save()
        return self.last_result

    def to_state(self) -> dict:
        return {
            "format": "swarmart-curator/1",
            "arms": {m: [a.plays, a.reward_sum, a.reward_sq] for m, a in self.arms.items()},
            "archive": [v.round(5).tolist() for v in self.archive],
            "pieces": self.pieces,
            "best": self.best_score,
            "last_model": self.last_model,
        }

    def apply_state(self, state: object) -> bool:
        if not isinstance(state, dict) or state.get("format") != "swarmart-curator/1":
            return False
        arms = state.get("arms")
        if not isinstance(arms, dict):
            return False
        parsed: dict[str, ArmStats] = {model: ArmStats() for model in MODELS}
        for model, values in arms.items():
            if model not in parsed or not isinstance(values, list) or len(values) != 3:
                continue
            plays, total, squares = values
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
                continue
            plays = int(min(max(plays, 0), 10**9))
            if plays == 0:
                continue
            total = min(max(float(total), 0.0), float(plays))
            squares = min(max(float(squares), 0.0), float(plays))
            parsed[model] = ArmStats(plays, total, squares)
        archive = []
        for item in state.get("archive", [])[-ARCHIVE_LIMIT:] if isinstance(state.get("archive"), list) else []:
            if isinstance(item, list) and len(item) == len(DESCRIPTOR_KEYS):
                vector = np.asarray(
                    [v if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan for v in item], dtype=np.float64
                )
                if np.isfinite(vector).all():
                    archive.append(np.clip(vector, 0.0, 1.5))
        pieces = state.get("pieces")
        best = state.get("best")
        last = state.get("last_model")
        self.arms = parsed
        self.archive = archive
        self.pieces = int(min(max(pieces, 0), 10**9)) if isinstance(pieces, int) and not isinstance(pieces, bool) else 0
        self.best_score = (
            min(max(float(best), 0.0), 1.0)
            if isinstance(best, (int, float)) and not isinstance(best, bool) and math.isfinite(best)
            else 0.0
        )
        self.last_model = last if last in MODELS else None
        return True

    def load(self) -> None:
        path = self.state_path
        if path is None:
            return
        try:
            if not path.is_file() or path.stat().st_size > MAX_STATE_BYTES:
                return
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError, UnicodeDecodeError):
            log.warning("curator state unreadable; starting fresh")
            return
        if not self.apply_state(state):
            log.warning("curator state invalid; starting fresh")

    def save(self) -> None:
        path = self.state_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(self.to_state(), separators=(",", ":")).encode("utf-8")
            fd, tmp = tempfile.mkstemp(prefix=".curator-", suffix=".json", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError:
            log.warning("curator state could not be saved")


def refine_task(payload: str) -> str | None:
    from .evolve import evolve

    try:
        request = json.loads(payload)
        params = from_json(request["params"])
        generations = int(min(max(int(request.get("generations", 3)), 1), 12))
        size = int(min(max(int(request.get("size", 96)), 32), 192))
        steps = int(min(max(int(request.get("steps", 110)), 10), 400))
        seed = int(request.get("seed", 0)) % (2**32)
    except (KeyError, TypeError, ValueError, PresetError, RecursionError):
        return None
    result = evolve(params, generations=generations, size=size, steps=steps, rng_seed=seed)
    return json.dumps({"params": to_json(result.params), "score": result.score, "initial": result.initial_score})


def _refine_loop(tasks, results) -> None:
    while True:
        try:
            item = tasks.get()
        except (EOFError, OSError):
            return
        if item is None:
            return
        task_id, payload = item
        try:
            reply = refine_task(payload)
        except Exception:
            reply = None
        try:
            results.put((task_id, reply))
        except (EOFError, OSError):
            return


@dataclass
class Refinement:
    task_id: int
    params: Params | None
    score: float
    initial: float


class RefinementService:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._tasks = None
        self._results = None
        self._ids = itertools.count(1)
        self._pending: set[int] = set()
        self.restarts = 0

    def _ensure(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self.restarts += 1
            log.warning("refinement process restarted")
        self._tasks = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._process = self._ctx.Process(target=_refine_loop, args=(self._tasks, self._results), name="swarmart-curator", daemon=True)
        self._process.start()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def submit(self, params: Params, generations: int = 3, size: int = 96, steps: int = 110, seed: int = 0) -> int | None:
        try:
            self._ensure()
            task_id = next(self._ids)
            payload = json.dumps({"params": to_json(params), "generations": generations, "size": size, "steps": steps, "seed": seed})
            self._tasks.put((task_id, payload))
            self._pending.add(task_id)
            return task_id
        except (OSError, ValueError, RuntimeError):
            log.warning("refinement unavailable")
            return None

    @property
    def pending(self) -> int:
        return len(self._pending)

    def poll(self) -> list[Refinement]:
        out: list[Refinement] = []
        if self._results is None:
            return out
        while True:
            try:
                task_id, reply = self._results.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError, ValueError, TypeError):
                break
            if task_id in self._pending:
                self._pending.discard(task_id)
                out.append(self._decode(task_id, reply))
        if self._pending and not self.alive:
            log.warning("refinement process exited with %d pending task(s)", len(self._pending))
            out.extend(Refinement(task_id, None, 0.0, 0.0) for task_id in sorted(self._pending))
            self._pending.clear()
        return out

    @staticmethod
    def _decode(task_id: int, reply: str | None) -> Refinement:
        if not isinstance(reply, str) or len(reply) > MAX_STATE_BYTES:
            return Refinement(task_id, None, 0.0, 0.0)
        try:
            data = json.loads(reply)
            params = from_json(data["params"])
            score = float(data["score"])
            initial = float(data["initial"])
        except (KeyError, TypeError, ValueError, PresetError, RecursionError):
            return Refinement(task_id, None, 0.0, 0.0)
        if not (math.isfinite(score) and math.isfinite(initial)):
            return Refinement(task_id, None, 0.0, 0.0)
        return Refinement(task_id, params, score, initial)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        with contextlib.suppress(Exception):
            self._tasks.put_nowait(None)
        process.join(0.5)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        self._pending.clear()
        for q in (self._tasks, self._results):
            with contextlib.suppress(Exception):
                q.close()
                q.cancel_join_thread()
        self._process = None
