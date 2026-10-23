from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .activity import AgentSample
from .config import Params
from .engine import Simulation
from .evolve import Cancelled, evolve
from .overlay import OverlayGeometry, build_overlay
from .render import Renderer, downsample, preview_factor

log = logging.getLogger("swarmart.workers")

EVENT_TEXT = {
    "nonfinite_canvas": "repaired invalid canvas values",
    "nonfinite_agents": "repaired invalid agent state",
    "swarm_collapse": "swarm collapsed, re-seeded 25% of agents",
}


@dataclass
class FramePacket:
    image: QImage
    data: bytes
    width: int
    height: int
    params: Params
    steps: int
    agents: AgentSample | None = None
    overlay: OverlayGeometry | None = None


def _render_job(
    renderer: Renderer,
    snapshot: np.ndarray,
    params: Params,
    steps: int,
    reset: bool,
    full: bool,
    agents: AgentSample | None = None,
) -> FramePacket:
    if reset:
        renderer.reset()
    factor = 1 if full else preview_factor(params.height, params.width)
    accum, h, w = downsample(snapshot, params.height, params.width, factor)
    packed = renderer.render(accum, h, w, params.exposure, params.glow)
    data = packed.tobytes()
    image = QImage(data, w, h, w * 4, QImage.Format.Format_RGB32).copy()
    overlay = None
    if agents is not None:
        try:
            overlay = build_overlay(agents)
        except Exception:
            log.exception("overlay build failed")
    return FramePacket(image, data, w, h, params, steps, agents, overlay)


class SimulationWorker(QThread):
    frame_ready = Signal(object)
    stats_ready = Signal(dict)
    notice = Signal(str)
    params_changed = Signal(object)
    snapshot_ready = Signal(object, str)
    telemetry_ready = Signal(object)
    completed = Signal(object)

    FRAME_INTERVAL = 1.0 / 60.0

    def __init__(self, params: Params) -> None:
        super().__init__()
        self._initial = params
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._ack = threading.Event()
        self._ack.set()
        self.paused = False
        self.auto_stop = True
        self.piece_limit = 0.0
        self.piece_seconds = 0.0

    def post(self, *command: object) -> None:
        self._commands.put(command)

    def ack(self) -> None:
        self._ack.set()

    def stop(self) -> None:
        self.post("stop")
        self.requestInterruption()

    def run(self) -> None:
        try:
            self._loop()
        except Exception:
            log.exception("simulation thread failed")
            self.notice.emit("The simulation stopped unexpectedly. Restart the app to continue.")

    def _build(self, params: Params, fallback: Simulation | None) -> Simulation:
        try:
            if fallback is None:
                return Simulation(params)
            fallback.apply(params)
            return fallback
        except MemoryError:
            log.warning("not enough memory for %dx%d with %d agents", params.width, params.height, params.agents)
            self.notice.emit("Not enough memory for that size. Reverted to the previous settings.")
            if fallback is not None:
                return fallback
            raise

    def _loop(self) -> None:
        sim = self._build(self._initial, None)
        renderer = Renderer()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swarmart-render")
        pending: Future | None = None
        snapshots: list[tuple[Future, str]] = []
        self._snapshot_requests: list[str] = []
        reset_renderer = True
        dirty = True
        last_emit = 0.0
        failures: deque[float] = deque(maxlen=5)
        window_start = time.monotonic()
        frames = 0
        steps = 0
        announced = False
        next_telemetry = 0.0
        last_tick = time.monotonic()
        try:
            while not self.isInterruptionRequested():
                stop, changed, reset_flag, sim = self._drain(sim)
                if stop:
                    break
                while self._snapshot_requests:
                    target = self._snapshot_requests.pop(0)
                    job = pool.submit(_render_job, Renderer(), sim.accum.copy(), sim.params, sim.steps, False, True)
                    snapshots.append((job, target))
                for job, target in [item for item in snapshots if item[0].done()]:
                    snapshots.remove((job, target))
                    try:
                        self.snapshot_ready.emit(job.result(), target)
                    except Exception:
                        log.exception("snapshot render failed")
                        self.snapshot_ready.emit(None, target)
                dirty = dirty or changed
                reset_renderer = reset_renderer or reset_flag
                if reset_flag:
                    self.piece_seconds = 0.0
                now = time.monotonic()
                if not self.paused:
                    self.piece_seconds += min(now - last_tick, 0.5)
                last_tick = now
                if not self.paused:
                    try:
                        events = sim.step(sim.params.steps_per_frame)
                        if self.piece_limit > 0 and self.piece_seconds >= self.piece_limit and not sim.complete:
                            sim.finish("display time limit reached")
                        steps += sim.params.steps_per_frame
                        dirty = True
                        for event in events:
                            self.notice.emit("Auto-recovered: " + EVENT_TEXT.get(event, event))
                    except MemoryError:
                        log.warning("memory exhausted during step")
                        self.paused = True
                        self.notice.emit("Out of memory. Paused; reduce resolution or agent count.")
                    except Exception:
                        log.exception("simulation step failed")
                        failures.append(now)
                        if len(failures) == failures.maxlen and now - failures[0] < 10.0:
                            self.paused = True
                            self.notice.emit("Repeated simulation errors. Paused; try Reset or different settings.")
                        else:
                            sim.reset()
                            reset_renderer = True
                            self.notice.emit("Recovered from an internal error; the canvas was reset.")
                if sim.complete:
                    if not announced:
                        announced = True
                        if self.auto_stop and not self.paused:
                            self.paused = True
                        self._emit_telemetry(sim, self.completed)
                        next_telemetry = 0.0
                        dirty = True
                else:
                    announced = False
                if now >= next_telemetry:
                    self._emit_telemetry(sim, self.telemetry_ready)
                    next_telemetry = now + (0.5 if self.paused else 0.2)
                if pending is not None and pending.done():
                    try:
                        packet = pending.result()
                        self._ack.clear()
                        self.frame_ready.emit(packet)
                        frames += 1
                    except Exception:
                        log.exception("render failed")
                    pending = None
                if pending is None and dirty and self._ack.is_set() and now - last_emit >= self.FRAME_INTERVAL:
                    pending = pool.submit(
                        _render_job, renderer, sim.accum.copy(), sim.params, sim.steps, reset_renderer, False, sim.agent_sample()
                    )
                    reset_renderer = False
                    dirty = False
                    last_emit = now
                if now - window_start >= 0.5:
                    span = now - window_start
                    self.stats_ready.emit(
                        {
                            "fps": frames / span,
                            "sps": steps / span,
                            "agents": sim.agent_count,
                            "size": (sim.w, sim.h),
                            "paused": self.paused,
                        }
                    )
                    window_start, frames, steps = now, 0, 0
                if self.paused or pending is not None or snapshots:
                    time.sleep(0.004)
                else:
                    time.sleep(0)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _emit_telemetry(self, sim: Simulation, signal) -> None:
        try:
            signal.emit(replace(sim.telemetry(), paused=self.paused))
        except Exception:
            log.exception("telemetry failed")

    def _drain(self, sim: Simulation) -> tuple[bool, bool, bool, Simulation]:
        changed = False
        reset = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return False, changed, reset, sim
            kind = command[0]
            if kind == "stop":
                return True, changed, reset, sim
            if kind == "params" and isinstance(command[1], Params):
                requested = command[1]
                before = (sim.params.width, sim.params.height, sim.params.model, sim.params.seed)
                sim = self._build(requested, sim)
                if sim.params != requested:
                    self.params_changed.emit(sim.params)
                after = (sim.params.width, sim.params.height, sim.params.model, sim.params.seed)
                reset = reset or before != after
                changed = True
            elif kind == "reset":
                sim.reset()
                reset = changed = True
            elif kind == "clear":
                sim.clear()
                reset = changed = True
            elif kind == "pause":
                self.paused = bool(command[1])
                changed = True
            elif kind == "piece_limit" and len(command) == 2 and isinstance(command[1], (int, float)):
                value = float(command[1])
                self.piece_limit = value if value == value and 0.0 < value < 1e7 else 0.0
            elif kind == "auto_stop" and len(command) == 2:
                self.auto_stop = bool(command[1])
            elif kind == "snapshot" and len(command) == 2 and isinstance(command[1], str):
                self._snapshot_requests.append(command[1])
            elif kind == "inject":
                if len(command) == 3 and sim.inject(command[1], command[2]):
                    changed = True


class EvolveWorker(QThread):
    progress = Signal(int, int, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, params: Params, generations: int) -> None:
        super().__init__()
        self.params = params
        self.generations = generations
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            result = evolve(
                self.params,
                generations=self.generations,
                progress=lambda g, t, s: self.progress.emit(g, t, s),
                should_stop=self._stop.is_set,
            )
            self.finished_ok.emit(result)
        except Cancelled:
            self.failed.emit("Evolution cancelled. Your settings are unchanged.")
        except Exception:
            log.exception("evolution failed")
            self.failed.emit("Evolution could not finish. Your settings are unchanged.")
