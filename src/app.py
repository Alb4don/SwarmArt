from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QStandardPaths, Qt, QTimer, QUrl, Signal, qInstallMessageHandler
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .activity import CREATING, IDLE, REINFORCING, Contributor, Telemetry
from .activity_panel import STATUS_COLORS, ActivityPanel
from .config import (
    FLOAT_LIMITS,
    INT_LIMITS,
    MODELS,
    RESOLUTIONS,
    Params,
    PresetError,
    random_palette,
    update,
    with_model,
)
from .gallery import Gallery, GalleryState
from .logsafe import install_exception_hooks, setup_logging
from .overlay import OverlayGeometry
from .storage import export_png, load_preset, save_preset
from .workers import EvolveWorker, FramePacket, SimulationWorker

log = logging.getLogger("swarmart.app")

ALL_MODELS = frozenset(MODELS)

SLIDERS: tuple[tuple[str, str, frozenset[str], str], ...] = (
    ("speed", "Speed", ALL_MODELS, "behaviour"),
    ("sensor_angle", "Sensor angle", frozenset({"physarum"}), "behaviour"),
    ("sensor_dist", "Sensing distance", frozenset({"physarum", "boids"}), "behaviour"),
    ("turn", "Steering", frozenset({"physarum", "flow"}), "behaviour"),
    ("decay", "Trail persistence", frozenset({"physarum"}), "behaviour"),
    ("deposit", "Deposit", frozenset({"physarum", "flow", "turmite"}), "behaviour"),
    ("noise_scale", "Field scale", frozenset({"flow"}), "behaviour"),
    ("cohesion", "Cohesion", frozenset({"boids"}), "behaviour"),
    ("alignment", "Alignment", frozenset({"boids"}), "behaviour"),
    ("separation", "Separation", frozenset({"boids"}), "behaviour"),
    ("jitter", "Jitter", frozenset({"physarum", "boids", "flow"}), "behaviour"),
    ("hue_spread", "Hue spread", ALL_MODELS, "look"),
    ("exposure", "Exposure", ALL_MODELS, "look"),
    ("glow", "Glow", ALL_MODELS, "look"),
    ("fade", "Canvas fade", ALL_MODELS, "look"),
)

MODEL_LABELS = {
    "physarum": "Physarum (slime mould)",
    "boids": "Boids (flocking)",
    "flow": "Flow field (curl noise)",
    "turmite": "Turmites (stigmergic ants)",
}

SLIDER_STEPS = 1000


def _qt_message_handler(mode, context, message) -> None:
    logging.getLogger("swarmart.qt").debug("%s", message)


class Bridge(QObject):
    message = Signal(str, bool)


class Canvas(QWidget):
    injected = Signal(float, float)

    HEAD_STYLE = {
        IDLE: (95, 0.9),
        REINFORCING: (190, 1.2),
        CREATING: (255, 1.7),
    }
    TAIL_ALPHA = {REINFORCING: 120, CREATING: 210}

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._overlay: OverlayGeometry | None = None
        self._top: tuple[Contributor, ...] = ()
        self._top_size = (1, 1)
        self.show_agents = True
        self._last_inject = 0.0
        self.setMinimumSize(320, 240)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Click or drag to drop a stimulus the agents react to")

    def set_image(self, image: QImage, overlay: OverlayGeometry | None = None) -> None:
        self._image = image
        self._overlay = overlay
        self.update()

    def set_top(self, top: tuple[Contributor, ...], width: int, height: int) -> None:
        self._top = top
        self._top_size = (max(1, width), max(1, height))

    def set_show_agents(self, visible: bool) -> None:
        self.show_agents = bool(visible)
        self.update()

    def _paint_agents(self, painter: QPainter, target: QRectF) -> None:
        overlay = self._overlay
        if overlay is not None and overlay.layers:
            painter.save()
            painter.translate(target.x(), target.y())
            painter.scale(target.width() / max(1, overlay.width), target.height() / max(1, overlay.height))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for layer in overlay.layers:
                if layer.tails.size():
                    color = QColor(STATUS_COLORS[layer.status])
                    color.setAlpha(self.TAIL_ALPHA.get(layer.status, 120))
                    pen = QPen(color)
                    pen.setWidthF(1.2)
                    pen.setCosmetic(True)
                    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                    painter.setPen(pen)
                    painter.drawLines(layer.tails)
            for layer in overlay.layers:
                alpha, radius = self.HEAD_STYLE[layer.status]
                if layer.status == CREATING:
                    outline = QPen(QColor(6, 6, 10, 150))
                    outline.setWidthF(radius * 2 + 1.5)
                    outline.setCosmetic(True)
                    outline.setCapStyle(Qt.PenCapStyle.SquareCap)
                    painter.setPen(outline)
                    painter.drawPoints(layer.heads)
                color = QColor(STATUS_COLORS[layer.status])
                color.setAlpha(alpha)
                pen = QPen(color)
                pen.setWidthF(radius * 2)
                pen.setCosmetic(True)
                pen.setCapStyle(Qt.PenCapStyle.SquareCap)
                painter.setPen(pen)
                painter.drawPoints(layer.heads)
            painter.restore()
        ox, oy = target.x(), target.y()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tw, th = self._top_size
        font = QFont(painter.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.85))
        painter.setFont(font)
        for rank, c in enumerate(self._top):
            point = QPointF(ox + c.x * target.width() / tw, oy + c.y * target.height() / th)
            ring = QPen(QColor(255, 255, 255, 230))
            ring.setWidthF(1.6)
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, 7.0, 7.0)
            label = f"{rank + 1}· #{c.agent_id}"
            box = painter.fontMetrics().boundingRect(label)
            bg = QRectF(point.x() + 9, point.y() - box.height() / 2 - 2, box.width() + 8, box.height() + 4)
            painter.fillRect(bg, QColor(10, 10, 14, 190))
            painter.setPen(QColor(240, 240, 245))
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, label)
        self._paint_legend(painter)

    def _paint_legend(self, painter: QPainter) -> None:
        overlay = self._overlay
        items = tuple(
            (f"{name} {overlay.count(status):,}" if overlay is not None else name, status)
            for name, status in (("drawing new", CREATING), ("reworking", REINFORCING), ("idle", IDLE))
        )
        metrics = painter.fontMetrics()
        width = sum(metrics.horizontalAdvance(text) + 26 for text, _ in items) + 8
        rect = QRectF(10, self.height() - metrics.height() - 18, width, metrics.height() + 10)
        painter.fillRect(rect, QColor(10, 10, 14, 180))
        x = rect.x() + 8
        for text, status in items:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(STATUS_COLORS[status])
            painter.drawEllipse(QPointF(x + 5, rect.center().y()), 4.5, 4.5)
            painter.setPen(QColor(220, 220, 228))
            painter.drawText(QPointF(x + 14, rect.center().y() + metrics.ascent() / 2 - 1), text)
            x += metrics.horizontalAdvance(text) + 26

    def _target(self) -> QRectF:
        if self._image is None or self._image.isNull():
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        tw, th = iw * scale, ih * scale
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 8, 12))
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            target = self._target()
            painter.drawImage(target, self._image)
            if self.show_agents:
                self._paint_agents(painter, target)
        painter.end()

    def _emit(self, pos: QPointF) -> None:
        now = time.monotonic()
        if now - self._last_inject < 0.03:
            return
        target = self._target()
        if target.width() <= 0 or target.height() <= 0 or not target.contains(pos):
            return
        self._last_inject = now
        self.injected.emit((pos.x() - target.x()) / target.width(), (pos.y() - target.y()) / target.height())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit(event.position())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit(event.position())


class ParamSlider(QWidget):
    changed = Signal()

    def __init__(self, name: str, label: str) -> None:
        super().__init__()
        self.name = name
        self.lo, self.hi = FLOAT_LIMITS[name]
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(44)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_change)
        self.title = label

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / SLIDER_STEPS

    def set_value(self, value: float) -> None:
        position = int(round((value - self.lo) / (self.hi - self.lo) * SLIDER_STEPS))
        self.slider.blockSignals(True)
        self.slider.setValue(max(0, min(SLIDER_STEPS, position)))
        self.slider.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        digits = 3 if self.hi - self.lo < 1.5 else 2
        self.value_label.setText(f"{self.value():.{digits}f}")

    def _on_change(self) -> None:
        self._refresh()
        self.changed.emit()


class MainWindow(QMainWindow):
    def __init__(
        self,
        params: Params | None = None,
        autonomous: bool = True,
        gallery_dir: Path | None = None,
        state_path: Path | None = None,
        hold_seconds: float = 5.0,
        refine: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"SwarmArt {__version__} — emergent agent painting")
        self.params = params or Params()
        self.rng = np.random.default_rng()
        self.evolver: EvolveWorker | None = None
        self.was_paused_before_evolve = False
        self.bridge = Bridge()
        self.bridge.message.connect(self._show_message)
        self._exporting = False

        self.canvas = Canvas()
        self.canvas.injected.connect(lambda fx, fy: self.worker.post("inject", fx, fy))

        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(40)
        self._push_timer.timeout.connect(self._push_params)

        panel = self._build_panel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(scroll)
        self.setCentralWidget(central)

        self.status_fps = QLabel()
        self.status_info = QLabel()
        self.statusBar().addPermanentWidget(self.status_info)
        self.statusBar().addPermanentWidget(self.status_fps)

        self.activity_panel = ActivityPanel()
        self.activity_panel.overlay_toggled.connect(self._set_overlay)
        self.activity_panel.auto_stop_toggled.connect(lambda on: self.worker.post("auto_stop", bool(on)))
        self.activity_panel.step_target_changed.connect(self._on_step_target)
        self.activity_panel.export_requested.connect(self.export_image)
        self.activity_panel.continue_requested.connect(self._continue_run)
        self.activity_panel.new_run_requested.connect(self._new_run)
        activity_scroll = QScrollArea()
        activity_scroll.setWidgetResizable(True)
        activity_scroll.setWidget(self.activity_panel)
        activity_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        activity_scroll.setMinimumWidth(300)
        self.activity_dock = QDockWidget("Agent activity", self)
        self.activity_dock.setObjectName("agent_activity")
        self.activity_dock.setWidget(activity_scroll)
        self.activity_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.activity_dock)

        self._build_actions()
        self._build_menu()
        self._sync_controls()

        self.worker = SimulationWorker(self.params)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.stats_ready.connect(self._on_stats)
        self.worker.notice.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.worker.params_changed.connect(self._on_params_reverted)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.telemetry_ready.connect(self._on_telemetry)
        self.worker.completed.connect(self._on_completed)
        self.worker.start()
        self._last_telemetry: Telemetry | None = None

        self.gallery = Gallery(
            self._read_controls,
            gallery_dir or self._default_gallery_dir(),
            state_path if state_path is not None else self._default_state_path(),
            hold_seconds=hold_seconds,
            refine=refine,
            parent=self,
        )
        self.gallery.advance.connect(self._gallery_advance)
        self.gallery.autosave_requested.connect(lambda path: self.worker.post("snapshot", path))
        self.gallery.state_changed.connect(self._on_gallery_state)
        self.gallery.notice.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.activity_panel.gallery_toggled.connect(self._set_gallery)
        self.activity_panel.piece_limit_changed.connect(self._set_piece_limit)
        self.activity_panel.open_gallery_requested.connect(self._open_gallery)
        self._set_gallery(autonomous, prepare_delay_ms=2500)
        self.resize(1480, 900)

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)

        swarm_box = QGroupBox("Swarm")
        swarm = QFormLayout(swarm_box)
        self.model_combo = QComboBox()
        for name in MODELS:
            self.model_combo.addItem(MODEL_LABELS[name], name)
        self.model_combo.currentIndexChanged.connect(self._on_model)
        swarm.addRow("Model", self.model_combo)

        self.agents_spin = QSpinBox()
        self.agents_spin.setRange(*INT_LIMITS["agents"])
        self.agents_spin.setSingleStep(1000)
        self.agents_spin.setKeyboardTracking(False)
        self.agents_spin.valueChanged.connect(self._schedule_push)
        swarm.addRow("Agents", self.agents_spin)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(*INT_LIMITS["seed"])
        self.seed_spin.setKeyboardTracking(False)
        self.seed_spin.valueChanged.connect(self._schedule_push)
        new_seed = QPushButton("New")
        new_seed.setToolTip("Random seed (N)")
        new_seed.clicked.connect(self.new_seed)
        seed_layout.addWidget(self.seed_spin, 1)
        seed_layout.addWidget(new_seed)
        swarm.addRow("Seed", seed_row)

        self.res_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self.res_combo.addItem(f"{w} × {h}", (w, h))
        self.res_combo.currentIndexChanged.connect(self._schedule_push)
        swarm.addRow("Canvas", self.res_combo)

        self.spf_spin = QSpinBox()
        self.spf_spin.setRange(*INT_LIMITS["steps_per_frame"])
        self.spf_spin.valueChanged.connect(self._schedule_push)
        swarm.addRow("Steps / frame", self.spf_spin)
        root.addWidget(swarm_box)

        self.sliders: dict[str, ParamSlider] = {}
        self.slider_rows: dict[str, tuple[QFormLayout, ParamSlider]] = {}
        behaviour_box = QGroupBox("Local rules")
        behaviour = QFormLayout(behaviour_box)
        look_box = QGroupBox("Look")
        look = QFormLayout(look_box)
        for name, label, _, group in SLIDERS:
            slider = ParamSlider(name, label)
            slider.changed.connect(self._schedule_push)
            form = behaviour if group == "behaviour" else look
            form.addRow(label, slider)
            self.sliders[name] = slider
            self.slider_rows[name] = (form, slider)
        palette_button = QPushButton("Random palette")
        palette_button.setToolTip("Random cosine palette (P)")
        palette_button.clicked.connect(self.random_palette)
        look.addRow(palette_button)
        root.addWidget(behaviour_box)
        root.addWidget(look_box)

        ai_box = QGroupBox("AI evolver")
        ai = QFormLayout(ai_box)
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(2, 40)
        self.gen_spin.setValue(8)
        ai.addRow("Generations", self.gen_spin)
        self.evolve_button = QPushButton("Evolve rules")
        self.evolve_button.setToolTip(
            "CMA-ES searches the local-rule space, scoring previews on entropy, "
            "1/f spectrum, fractal dimension, coverage and colourfulness (Ctrl+E)"
        )
        self.evolve_button.clicked.connect(self.toggle_evolve)
        ai.addRow(self.evolve_button)
        self.evolve_progress = QProgressBar()
        self.evolve_progress.setRange(0, 1)
        self.evolve_progress.setValue(0)
        ai.addRow(self.evolve_progress)
        self.score_label = QLabel("Score: —")
        ai.addRow(self.score_label)
        root.addWidget(ai_box)

        canvas_box = QGroupBox("Canvas")
        canvas_layout = QVBoxLayout(canvas_box)
        row1 = QHBoxLayout()
        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_pause)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.worker.post("reset"))
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.worker.post("clear"))
        for button in (self.play_button, reset_button, clear_button):
            row1.addWidget(button)
        canvas_layout.addLayout(row1)
        row2 = QHBoxLayout()
        export_button = QPushButton("Export PNG")
        export_button.clicked.connect(self.export_image)
        save_button = QPushButton("Save preset")
        save_button.clicked.connect(self.save_preset)
        load_button = QPushButton("Load…")
        load_button.setToolTip("Load a .json preset or a PNG exported by SwarmArt")
        load_button.clicked.connect(self.load_preset)
        for button in (export_button, save_button, load_button):
            row2.addWidget(button)
        canvas_layout.addLayout(row2)
        self.auto_palette = QCheckBox("New palette on new seed")
        canvas_layout.addWidget(self.auto_palette)
        root.addWidget(canvas_box)
        root.addStretch(1)
        return panel

    def _build_actions(self) -> None:
        bindings = (
            ("Play/Pause", QKeySequence(Qt.Key.Key_Space), self.toggle_pause),
            ("Reset", QKeySequence(Qt.Key.Key_R), lambda: self.worker.post("reset")),
            ("Clear", QKeySequence(Qt.Key.Key_C), lambda: self.worker.post("clear")),
            ("New seed", QKeySequence(Qt.Key.Key_N), self.new_seed),
            ("Random palette", QKeySequence(Qt.Key.Key_P), self.random_palette),
            ("Export", QKeySequence.StandardKey.Save, self.export_image),
            ("Load", QKeySequence.StandardKey.Open, self.load_preset),
            ("Evolve", QKeySequence("Ctrl+E"), self.toggle_evolve),
            ("Fullscreen", QKeySequence(Qt.Key.Key_F11), self.toggle_fullscreen),
        )
        for text, shortcut, slot in bindings:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    def _build_menu(self) -> None:
        view = self.menuBar().addMenu("&View")
        panel_action = self.activity_dock.toggleViewAction()
        panel_action.setText("Agent activity panel")
        panel_action.setShortcut(QKeySequence(Qt.Key.Key_F2))
        view.addAction(panel_action)
        self.overlay_action = QAction("Show agents on canvas", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.setShortcut(QKeySequence(Qt.Key.Key_A))
        self.overlay_action.toggled.connect(self._set_overlay)
        view.addAction(self.overlay_action)

    def _set_overlay(self, visible: bool) -> None:
        visible = bool(visible)
        for widget in (self.overlay_action, self.activity_panel.overlay_check):
            widget.blockSignals(True)
            widget.setChecked(visible)
            widget.blockSignals(False)
        self.canvas.set_show_agents(visible)

    def _on_step_target(self, value: int) -> None:
        self._push_timer.stop()
        self.params = update(self._read_controls(), target_steps=int(value))
        self.worker.post("params", self.params)

    def _on_telemetry(self, telemetry: Telemetry) -> None:
        self._last_telemetry = telemetry
        self.activity_panel.update_telemetry(telemetry)
        self.canvas.set_top(telemetry.top, telemetry.width, telemetry.height)

    def _default_gallery_dir(self) -> Path:
        return Path(self._pictures_dir()) / "SwarmArt Gallery"

    @staticmethod
    def _default_state_path() -> Path | None:
        location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        return Path(location) / "curator.json" if location else None

    def _set_gallery(self, enabled: bool, prepare_delay_ms: int = 0) -> None:
        enabled = bool(enabled)
        self.gallery.set_enabled(enabled, prepare_delay_ms)
        self.worker.post("piece_limit", float(self.activity_panel.piece_limit_spin.value()) if enabled else 0.0)
        if enabled:
            self.activity_panel.auto_stop_check.setChecked(True)
            self.worker.post("auto_stop", True)
            last = self._last_telemetry
            if last is not None and last.complete:
                self.gallery.piece_completed(last.reason)
        self.activity_panel.update_gallery(self.gallery.state())

    def _set_piece_limit(self, seconds: int) -> None:
        if self.gallery.enabled:
            self.worker.post("piece_limit", float(seconds))

    def _open_gallery(self) -> None:
        folder = self.gallery.folder
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.statusBar().showMessage("The gallery folder could not be created.", 6000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_gallery_state(self, state: GalleryState) -> None:
        self.activity_panel.update_gallery(state)

    def _gallery_advance(self, params: Params) -> None:
        self._apply_params(params)
        if self.worker.paused:
            self.toggle_pause()
        self.statusBar().showMessage(f"New piece started: {params.model} · seed {params.seed}", 6000)

    def _on_completed(self, telemetry: Telemetry) -> None:
        self._last_telemetry = telemetry
        self.gallery.piece_completed(telemetry.reason)
        self._on_telemetry(telemetry)
        self.play_button.setText("Play" if self.worker.paused else "Pause")
        self.statusBar().showMessage(
            f"Artwork complete after {telemetry.steps:,} steps — {telemetry.reason}. "
            f"{100 * telemetry.coverage:.0f}% of the canvas painted.",
            12000,
        )
        QApplication.alert(self)

    def _continue_run(self) -> None:
        if self.worker.paused:
            self.toggle_pause()

    def _new_run(self) -> None:
        if self.gallery.enabled:
            self.gallery.start_next()
            return
        self.new_seed()
        if self.worker.paused:
            self.toggle_pause()

    def _sync_controls(self) -> None:
        p = self.params
        self.activity_panel.set_step_target(p.target_steps)
        widgets = [self.model_combo, self.agents_spin, self.seed_spin, self.res_combo, self.spf_spin]
        for widget in widgets:
            widget.blockSignals(True)
        self.model_combo.setCurrentIndex(MODELS.index(p.model))
        self.agents_spin.setValue(p.agents)
        self.seed_spin.setValue(p.seed)
        res_index = next((i for i, r in enumerate(RESOLUTIONS) if r == (p.width, p.height)), -1)
        if res_index < 0:
            self.res_combo.addItem(f"{p.width} × {p.height}", (p.width, p.height))
            res_index = self.res_combo.count() - 1
        self.res_combo.setCurrentIndex(res_index)
        self.spf_spin.setValue(p.steps_per_frame)
        for widget in widgets:
            widget.blockSignals(False)
        for name, slider in self.sliders.items():
            slider.set_value(getattr(p, name))
        self._update_visibility()

    def _update_visibility(self) -> None:
        model = self.params.model
        for name, _, models, _ in SLIDERS:
            form, slider = self.slider_rows[name]
            visible = model in models
            form.setRowVisible(slider, visible)

    def _read_controls(self) -> Params:
        size = self.res_combo.currentData() or (self.params.width, self.params.height)
        changes = {name: slider.value() for name, slider in self.sliders.items()}
        changes.update(
            agents=self.agents_spin.value(),
            seed=self.seed_spin.value(),
            width=int(size[0]),
            height=int(size[1]),
            steps_per_frame=self.spf_spin.value(),
        )
        return update(self.params, **changes)

    def _schedule_push(self) -> None:
        self._push_timer.start()

    def _push_params(self) -> None:
        self.params = self._read_controls()
        self.worker.post("params", self.params)

    def _apply_params(self, params: Params) -> None:
        self._push_timer.stop()
        self.params = params
        self._sync_controls()
        self.worker.post("params", params)

    def _on_params_reverted(self, params: Params) -> None:
        self.params = params
        self._sync_controls()

    def _on_model(self) -> None:
        model = self.model_combo.currentData()
        if model in MODELS and model != self.params.model:
            self._apply_params(with_model(self._read_controls(), model))

    def _on_frame(self, packet: FramePacket) -> None:
        self.canvas.set_image(packet.image, packet.overlay)
        self.worker.ack()

    def _on_stats(self, stats: dict) -> None:
        w, h = stats["size"]
        state = "paused" if stats["paused"] else f"{stats['sps']:.0f} steps/s"
        self.status_fps.setText(f"{stats['fps']:.0f} fps · {state}")
        self.status_info.setText(f"{stats['agents']:,} agents · {w}×{h}")
        self.play_button.setText("Play" if stats["paused"] else "Pause")

    def _show_message(self, text: str, is_error: bool) -> None:
        if is_error:
            QMessageBox.warning(self, "SwarmArt", text)
        else:
            self.statusBar().showMessage(text, 6000)

    def toggle_pause(self) -> None:
        paused = not self.worker.paused
        self.worker.paused = paused
        self.worker.post("pause", paused)
        self.play_button.setText("Play" if paused else "Pause")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def new_seed(self) -> None:
        changes = {"seed": int(self.rng.integers(0, INT_LIMITS["seed"][1]))}
        if self.auto_palette.isChecked():
            changes["palette"] = random_palette(self.rng)
        self._apply_params(update(self._read_controls(), **changes))

    def random_palette(self) -> None:
        self._apply_params(update(self._read_controls(), palette=random_palette(self.rng)))

    def _pictures_dir(self) -> str:
        location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        return location or str(Path.home())

    def export_image(self) -> None:
        if self._exporting:
            self.statusBar().showMessage("An export is already running.", 4000)
            return
        p = self.params
        default = str(Path(self._pictures_dir()) / f"swarmart_{p.model}_{p.seed}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", default, "PNG image (*.png)")
        if not path:
            return
        self._exporting = True
        self.statusBar().showMessage("Rendering full-resolution image…")
        self.worker.post("snapshot", path)

    def _on_snapshot(self, frame: FramePacket | None, path: str) -> None:
        if self.gallery.is_autosave(path):
            self.gallery.handle_snapshot(frame, path)
            return
        if frame is None:
            self._exporting = False
            self._show_message("Could not render the full-resolution image.", True)
            return

        def job() -> None:
            try:
                target = export_png(path, frame.data, frame.width, frame.height, frame.params)
                self.bridge.message.emit(f"Exported {target.name} ({frame.width}×{frame.height})", False)
            except OSError:
                log.warning("export failed")
                self.bridge.message.emit("Could not save the image. Check the folder permissions and free space.", True)
            except Exception:
                log.exception("export failed")
                self.bridge.message.emit("Could not save the image.", True)
            finally:
                self._exporting = False

        threading.Thread(target=job, name="swarmart-export", daemon=True).start()

    def save_preset(self) -> None:
        default = str(Path(self._pictures_dir()) / f"swarmart_{self.params.model}_{self.params.seed}.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", default, "SwarmArt preset (*.json)")
        if not path:
            return
        try:
            target = save_preset(path, self._read_controls())
            self.statusBar().showMessage(f"Saved {target.name}", 4000)
        except OSError:
            log.warning("preset save failed")
            QMessageBox.warning(self, "SwarmArt", "Could not save the preset. Check the folder permissions.")

    def load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", self._pictures_dir(), "SwarmArt presets (*.json *.png);;All files (*)")
        if not path:
            return
        try:
            params = load_preset(path)
        except PresetError:
            QMessageBox.warning(self, "SwarmArt", "That file is not a valid SwarmArt preset or exported image.")
            return
        self._apply_params(params)
        self.statusBar().showMessage("Preset loaded.", 4000)

    def toggle_evolve(self) -> None:
        if self.evolver is not None:
            self.evolver.cancel()
            self.evolve_button.setEnabled(False)
            self.evolve_button.setText("Cancelling…")
            return
        base = self._read_controls()
        self.evolver = EvolveWorker(base, self.gen_spin.value())
        self.evolver.progress.connect(self._on_evolve_progress)
        self.evolver.finished_ok.connect(self._on_evolve_done)
        self.evolver.failed.connect(self._on_evolve_failed)
        self.evolver.finished.connect(self._on_evolve_thread_finished)
        self.evolve_progress.setRange(0, self.gen_spin.value())
        self.evolve_progress.setValue(0)
        self.evolve_button.setText("Cancel")
        self.was_paused_before_evolve = self.worker.paused
        if not self.worker.paused:
            self.toggle_pause()
        self.statusBar().showMessage("Evolving local rules in the background…")
        self.evolver.start()

    def _on_evolve_progress(self, generation: int, total: int, score: float) -> None:
        self.evolve_progress.setRange(0, total)
        self.evolve_progress.setValue(generation)
        self.score_label.setText(f"Best score: {score:.3f}")

    def _on_evolve_done(self, result) -> None:
        gain = result.score - result.initial_score
        self.score_label.setText(f"Score: {result.initial_score:.3f} → {result.score:.3f}")
        if gain > 1e-9:
            self._apply_params(result.params)
            self.worker.post("clear")
            self.statusBar().showMessage(f"Evolved rules applied (+{gain:.3f}, validated on held-out seeds).", 8000)
        else:
            self.statusBar().showMessage("No validated improvement found; settings kept.", 8000)

    def _on_evolve_failed(self, text: str) -> None:
        self.statusBar().showMessage(text, 6000)

    def _on_evolve_thread_finished(self) -> None:
        worker = self.evolver
        self.evolver = None
        if worker is not None:
            worker.deleteLater()
        self.evolve_button.setEnabled(True)
        self.evolve_button.setText("Evolve rules")
        if not self.was_paused_before_evolve and self.worker.paused:
            self.toggle_pause()

    def closeEvent(self, event) -> None:
        self.gallery.shutdown()
        if self.evolver is not None:
            self.evolver.cancel()
            self.evolver.wait(5000)
        self.worker.stop()
        if not self.worker.wait(5000):
            log.warning("simulation thread did not stop in time")
        event.accept()


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    base = QColor(24, 24, 30)
    panel = QColor(32, 32, 40)
    text = QColor(225, 225, 232)
    accent = QColor(120, 140, 255)
    palette.setColor(QPalette.ColorRole.Window, panel)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, panel)
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(44, 44, 54))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(10, 10, 14))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(110, 110, 120))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(110, 110, 120))
    app.setPalette(palette)


def run_gui(params: Params | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SwarmArt")
    app.setOrganizationName("SwarmArt")
    qInstallMessageHandler(_qt_message_handler)

    def on_error() -> None:
        if QApplication.instance() is not None and threading.current_thread() is threading.main_thread():
            QMessageBox.critical(None, "SwarmArt", "An unexpected error occurred. Details were written to the log file.")

    install_exception_hooks(on_error)
    apply_dark_theme(app)
    window = MainWindow(params)
    window.show()

    def request_close(signum, frame) -> None:
        QTimer.singleShot(0, window.close)

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, request_close)
    pulse = QTimer()
    pulse.setInterval(250)
    pulse.timeout.connect(lambda: None)
    pulse.start()
    code = app.exec()
    pulse.stop()
    return code


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    from .cli import parse_args, run_headless

    multiprocessing.freeze_support()
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("starting SwarmArt %s on %s", __version__, sys.platform)
    if args.headless:
        return run_headless(args)
    params = None
    if args.preset:
        try:
            params = load_preset(args.preset)
        except PresetError:
            print("The preset file is not valid; starting with defaults.", file=sys.stderr)
    return run_gui(params)
