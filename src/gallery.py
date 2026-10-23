from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .config import Params
from .curator import Curator, RefinementService
from .evolve import aesthetic_score
from .storage import export_png

log = logging.getLogger("swarmart.gallery")

AUTOSAVE_PREFIX = "swarmart_auto_"
SCORE_EDGE = 256


@dataclass(frozen=True)
class GalleryState:
    enabled: bool
    pieces: int
    last_score: float | None
    best_score: float
    next_model: str | None
    next_refined: bool
    countdown: float | None
    folder: str
    last_file: str | None
    last_reason: str | None


def score_frame(data: bytes, width: int, height: int) -> tuple[float, dict[str, float]]:
    if width <= 0 or height <= 0 or len(data) != width * height * 4:
        return 0.0, {}
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    stride = max(1, max(width, height) // SCORE_EDGE)
    rgb = pixels[::stride, ::stride, 2::-1].astype(np.float32) * np.float32(1.0 / 255.0)
    return aesthetic_score(rgb)


def rotate_autosaves(folder: Path, keep: int) -> int:
    try:
        files = sorted(
            (p for p in folder.glob(f"{AUTOSAVE_PREFIX}*.png") if p.is_file() and not p.is_symlink()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return 0
    removed = 0
    for path in files[: max(0, len(files) - keep)]:
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed


class Gallery(QObject):
    advance = Signal(object)
    autosave_requested = Signal(str)
    state_changed = Signal(object)
    notice = Signal(str)
    _scored = Signal(object, float, object, str)

    def __init__(
        self,
        base_provider: Callable[[], Params],
        folder: Path,
        state_path: Path | None,
        hold_seconds: float = 5.0,
        refine: bool = True,
        keep_files: int = 500,
        rng: np.random.Generator | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.base_provider = base_provider
        self.folder = Path(folder)
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.keep_files = max(1, int(keep_files))
        self.rng = rng or np.random.default_rng()
        self.curator = Curator(self.rng, state_path)
        self.refiner = RefinementService() if refine else None
        self.enabled = False
        self._proposal: Params | None = None
        self._refined: Params | None = None
        self._task: int | None = None
        self._hold_deadline: float | None = None
        self._last_file: str | None = None
        self._last_reason: str | None = None
        self._pending_save: str | None = None
        self._saving = False
        self.last_started_refined = False
        self._scored.connect(self._on_scored)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._on_hold_done)
        self._tick = QTimer(self)
        self._tick.setInterval(250)
        self._tick.timeout.connect(self._on_tick)
        self._prepare_timer = QTimer(self)
        self._prepare_timer.setSingleShot(True)
        self._prepare_timer.timeout.connect(self.prepare_next)

    def state(self) -> GalleryState:
        countdown = None if self._hold_deadline is None else max(0.0, self._hold_deadline - time.monotonic())
        nxt = self._refined or self._proposal
        last = self.curator.last_result
        return GalleryState(
            enabled=self.enabled,
            pieces=self.curator.pieces,
            last_score=None if last is None else last.score,
            best_score=self.curator.best_score,
            next_model=None if nxt is None else nxt.model,
            next_refined=self._refined is not None,
            countdown=countdown,
            folder=str(self.folder),
            last_file=self._last_file,
            last_reason=self._last_reason,
        )

    def _publish(self) -> None:
        self.state_changed.emit(self.state())

    def set_enabled(self, enabled: bool, prepare_delay_ms: int = 0) -> None:
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if enabled:
            self._tick.start()
            if self._proposal is None:
                self._prepare_timer.start(max(0, int(prepare_delay_ms)))
        else:
            self._hold_timer.stop()
            self._prepare_timer.stop()
            self._hold_deadline = None
            self._tick.stop()
        self._publish()

    def prepare_next(self) -> None:
        if not self.enabled:
            return
        try:
            base = self.base_provider()
            self._proposal = self.curator.propose(base)
        except Exception:
            log.exception("curator proposal failed")
            self._proposal = None
            return
        self._refined = None
        self._task = None
        if self.refiner is not None and self.refiner.pending == 0:
            self._task = self.refiner.submit(self._proposal, generations=3, size=96, steps=110, seed=int(self.rng.integers(0, 2**31 - 1)))
        self._publish()

    def piece_completed(self, reason: str) -> None:
        if not self.enabled or self._hold_deadline is not None:
            return
        self._last_reason = reason
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self.base_provider()
        path = self.folder / f"{AUTOSAVE_PREFIX}{stamp}_{base.model}_{base.seed}.png"
        self._pending_save = str(path)
        self.autosave_requested.emit(str(path))
        self._hold_deadline = time.monotonic() + self.hold_seconds
        self._hold_timer.start(int(self.hold_seconds * 1000))
        self._publish()

    def is_autosave(self, path: str) -> bool:
        return path == self._pending_save

    def handle_snapshot(self, frame, path: str) -> None:
        self._pending_save = None
        if frame is None:
            self.notice.emit("Could not render the finished piece; continuing with the next one.")
            return
        self._saving = True

        def job() -> None:
            saved = ""
            score, metrics = 0.0, {}
            try:
                score, metrics = score_frame(frame.data, frame.width, frame.height)
            except Exception:
                log.exception("scoring failed")
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                saved = str(export_png(path, frame.data, frame.width, frame.height, frame.params))
                rotate_autosaves(self.folder, self.keep_files)
            except OSError:
                log.warning("autosave failed")
            except Exception:
                log.exception("autosave failed")
            self._scored.emit(frame.params, float(score), metrics, saved)

        threading.Thread(target=job, name="swarmart-autosave", daemon=True).start()

    def _on_scored(self, params: Params, score: float, metrics: dict, saved: str) -> None:
        self._saving = False
        result = self.curator.record(params, score, metrics if isinstance(metrics, dict) else {})
        if saved:
            self._last_file = saved
            self.notice.emit(f"Saved {Path(saved).name} · score {result.score:.2f}")
        else:
            self.notice.emit("The finished piece could not be saved; check the gallery folder.")
        self._publish()

    def _on_tick(self) -> None:
        if self.refiner is not None:
            for item in self.refiner.poll():
                if self._task is not None and item.task_id == self._task:
                    self._task = None
                    if item.params is not None and item.score > item.initial:
                        base = self.base_provider()
                        self._refined = replace(item.params, width=base.width, height=base.height)
        self._publish()

    def _on_hold_done(self) -> None:
        self._hold_deadline = None
        if not self.enabled:
            return
        self.start_next()

    def start_next(self) -> None:
        self._hold_timer.stop()
        self._hold_deadline = None
        self.last_started_refined = self._refined is not None
        params = self._refined or self._proposal
        if params is None:
            try:
                params = self.curator.propose(self.base_provider())
            except Exception:
                log.exception("curator proposal failed")
                return
        base = self.base_provider()
        params = replace(params, width=base.width, height=base.height, target_steps=0)
        self._proposal = None
        self._refined = None
        self._task = None
        self.advance.emit(params)
        self._prepare_timer.start(1500)
        self._publish()

    def shutdown(self) -> None:
        self.enabled = False
        for timer in (self._hold_timer, self._tick, self._prepare_timer):
            timer.stop()
        if self.refiner is not None:
            self.refiner.close()
