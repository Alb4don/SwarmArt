from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .activity import CREATING, IDLE, REINFORCING, STATE_LABELS, Telemetry

STATUS_COLORS = {
    CREATING: QColor(255, 190, 70),
    REINFORCING: QColor(110, 165, 255),
    IDLE: QColor(115, 115, 128),
}
STATUS_INDEX = {"drawing new": CREATING, "reworking": REINFORCING, "idle": IDLE}
STATUS_TEXT = {CREATING: "Drawing new strokes", REINFORCING: "Reworking existing strokes", IDLE: "Idle"}
BEHAVIOUR_COLORS = (QColor(94, 234, 212), QColor(244, 114, 182), QColor(167, 139, 250), QColor(250, 204, 21))
COMPLETE_COLOR = QColor(96, 214, 140)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _swatch(color: QColor) -> QLabel:
    label = QLabel()
    label.setFixedSize(10, 10)
    label.setStyleSheet(f"background:{color.name()}; border-radius:5px;")
    return label


def _meter(color: QColor) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 1000)
    bar.setTextVisible(False)
    bar.setFixedHeight(6)
    bar.setStyleSheet(
        "QProgressBar{background:rgba(255,255,255,18); border:none; border-radius:3px;}"
        f"QProgressBar::chunk{{background:{color.name()}; border-radius:3px;}}"
    )
    return bar


class Sparkline(QWidget):
    def __init__(self, series: tuple[tuple[str, QColor], ...], capacity: int = 240) -> None:
        super().__init__()
        self.series = series
        self.values = [deque(maxlen=capacity) for _ in series]
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def clear(self) -> None:
        for values in self.values:
            values.clear()
        self.update()

    def push(self, *points: float) -> None:
        for values, point in zip(self.values, points, strict=False):
            values.append(min(1.0, max(0.0, float(point))))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QColor(255, 255, 255, 10))
        grid = QPen(QColor(255, 255, 255, 25))
        painter.setPen(grid)
        for frac in (0.25, 0.5, 0.75):
            y = rect.bottom() - frac * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        for (_, color), values in zip(self.series, self.values, strict=False):
            if len(values) < 2:
                continue
            capacity = values.maxlen or len(values)
            step = rect.width() / max(1, capacity - 1)
            offset = rect.right() - step * (len(values) - 1)
            path = QPainterPath()
            for i, value in enumerate(values):
                point = QPointF(offset + i * step, rect.bottom() - value * rect.height())
                if i == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            pen = QPen(color)
            pen.setWidthF(1.8)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.end()


class ActivityPanel(QWidget):
    overlay_toggled = Signal(bool)
    auto_stop_toggled = Signal(bool)
    step_target_changed = Signal(int)
    gallery_toggled = Signal(bool)
    piece_limit_changed = Signal(int)
    open_gallery_requested = Signal()
    export_requested = Signal()
    continue_requested = Signal()
    new_run_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._model = ""
        self._last_steps = -1
        root = QVBoxLayout(self)
        root.setSpacing(10)

        header = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        self.elapsed_label = QLabel("0:00")
        self.elapsed_label.setStyleSheet("color:#9a9aa6;")
        header.addWidget(self.status_label, 1)
        header.addWidget(self.elapsed_label)
        root.addLayout(header)

        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color:#9a9aa6;")
        root.addWidget(self.reason_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p% to completion")
        root.addWidget(self.progress)
        meta = QHBoxLayout()
        self.step_label = QLabel()
        self.eta_label = QLabel()
        self.eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        meta.addWidget(self.step_label, 1)
        meta.addWidget(self.eta_label)
        root.addLayout(meta)
        actions = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_requested)
        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_requested)
        self.new_run_button = QPushButton("New run")
        self.new_run_button.clicked.connect(self.new_run_requested)
        for button in (self.export_button, self.continue_button, self.new_run_button):
            actions.addWidget(button)
        root.addLayout(actions)

        root.addWidget(self._divider())
        gallery_title = QLabel("Autonomous gallery")
        gallery_title.setStyleSheet("font-weight:600;")
        root.addWidget(gallery_title)
        self.gallery_check = QCheckBox("Paint, save and start new pieces on its own")
        self.gallery_check.setChecked(True)
        self.gallery_check.toggled.connect(self.gallery_toggled)
        root.addWidget(self.gallery_check)
        self.gallery_label = QLabel()
        self.gallery_label.setWordWrap(False)
        self.gallery_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.gallery_label)
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Max time per piece"))
        self.piece_limit_spin = QSpinBox()
        self.piece_limit_spin.setRange(0, 3600)
        self.piece_limit_spin.setSingleStep(30)
        self.piece_limit_spin.setSuffix(" s")
        self.piece_limit_spin.setSpecialValueText("No limit")
        self.piece_limit_spin.setValue(180)
        self.piece_limit_spin.setKeyboardTracking(False)
        self.piece_limit_spin.valueChanged.connect(self.piece_limit_changed)
        limit_row.addWidget(self.piece_limit_spin, 1)
        root.addLayout(limit_row)
        self.open_gallery_button = QPushButton("Open gallery folder")
        self.open_gallery_button.clicked.connect(self.open_gallery_requested)
        root.addWidget(self.open_gallery_button)

        root.addWidget(self._divider())
        self.agents_title = QLabel("Agents")
        self.agents_title.setStyleSheet("font-weight:600;")
        root.addWidget(self.agents_title)
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(8)
        status_grid.setVerticalSpacing(4)
        self.status_rows: dict[int, tuple[QLabel, QProgressBar]] = {}
        for row, status in enumerate((CREATING, REINFORCING, IDLE)):
            color = STATUS_COLORS[status]
            count = QLabel("0")
            count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bar = _meter(color)
            status_grid.addWidget(_swatch(color), row * 2, 0)
            status_grid.addWidget(QLabel(STATUS_TEXT[status]), row * 2, 1)
            status_grid.addWidget(count, row * 2, 2)
            status_grid.addWidget(bar, row * 2 + 1, 1, 1, 2)
            self.status_rows[status] = (count, bar)
        status_grid.setColumnStretch(1, 1)
        root.addLayout(status_grid)

        self.behaviour_title = QLabel("Behaviour")
        self.behaviour_title.setStyleSheet("font-weight:600;")
        root.addWidget(self.behaviour_title)
        self.behaviour_grid = QGridLayout()
        self.behaviour_grid.setHorizontalSpacing(8)
        self.behaviour_grid.setVerticalSpacing(4)
        self.behaviour_grid.setColumnStretch(1, 1)
        self.behaviour_rows: list[tuple[QLabel, QLabel, QProgressBar]] = []
        holder = QWidget()
        holder.setLayout(self.behaviour_grid)
        root.addWidget(holder)

        root.addWidget(self._divider())
        self.coverage_label = QLabel()
        root.addWidget(self.coverage_label)
        self.sparkline = Sparkline((("drawing new", STATUS_COLORS[CREATING]), ("painted", COMPLETE_COLOR)))
        root.addWidget(self.sparkline)
        legend = QLabel(
            f"<span style='color:{STATUS_COLORS[CREATING].name()}'>━</span> agents drawing new &nbsp; "
            f"<span style='color:{COMPLETE_COLOR.name()}'>━</span> canvas painted"
        )
        legend.setStyleSheet("color:#9a9aa6; font-size:11px;")
        root.addWidget(legend)

        root.addWidget(self._divider())
        top_title = QLabel("Top contributors")
        top_title.setStyleSheet("font-weight:600;")
        root.addWidget(top_title)
        self.top_table = QTableWidget(0, 4)
        self.top_table.setHorizontalHeaderLabels(["Agent", "Doing", "Strokes", "Where"])
        self.top_table.verticalHeader().setVisible(False)
        self.top_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.top_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.top_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.top_table.horizontalHeader()
        for col, mode in enumerate(
            (
                QHeaderView.ResizeMode.ResizeToContents,
                QHeaderView.ResizeMode.Stretch,
                QHeaderView.ResizeMode.ResizeToContents,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        ):
            header.setSectionResizeMode(col, mode)
        self.top_table.verticalHeader().setDefaultSectionSize(22)
        self.top_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.top_table.setFixedHeight(header.sizeHint().height() + 5 * 22 + 4)
        root.addWidget(self.top_table)
        self.top_hint = QLabel("Ranking appears once agents have had time to compete (about 60 steps).")
        self.top_hint.setWordWrap(True)
        self.top_hint.setStyleSheet("color:#9a9aa6; font-size:11px;")
        root.addWidget(self.top_hint)

        root.addWidget(self._divider())
        self.overlay_check = QCheckBox("Show agents on canvas (A)")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.overlay_toggled)
        root.addWidget(self.overlay_check)
        self.auto_stop_check = QCheckBox("Pause when the artwork is complete")
        self.auto_stop_check.setChecked(True)
        self.auto_stop_check.toggled.connect(self.auto_stop_toggled)
        root.addWidget(self.auto_stop_check)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Step target"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 1_000_000)
        self.target_spin.setSingleStep(500)
        self.target_spin.setSpecialValueText("Auto (until converged)")
        self.target_spin.setKeyboardTracking(False)
        self.target_spin.valueChanged.connect(self.step_target_changed)
        target_row.addWidget(self.target_spin, 1)
        root.addLayout(target_row)

        root.addStretch(1)
        self._set_status("Starting…", QColor(154, 154, 166))
        self.continue_button.setEnabled(False)

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:rgba(255,255,255,30);")
        return line

    def _set_status(self, text: str, color: QColor) -> None:
        self.status_label.setText(f"<span style='color:{color.name()}; font-size:16px;'>●</span>&nbsp;<b>{text}</b>")

    def update_gallery(self, state) -> None:
        self.gallery_check.blockSignals(True)
        self.gallery_check.setChecked(state.enabled)
        self.gallery_check.blockSignals(False)
        if not state.enabled:
            self.gallery_label.setText("<span style='color:#9a9aa6'>Off. Runs stop when complete.</span>")
            return
        lines = [f"Pieces finished: <b>{state.pieces:,}</b>"]
        if state.last_score is not None:
            lines.append(f"Last score {state.last_score:.2f} · best {state.best_score:.2f}")
        if state.countdown is not None:
            lines.append(f"<b>Next piece in {state.countdown:.0f} s</b>")
        if state.next_model:
            tuned = "AI-tuned" if state.next_refined else "tuning rules…"
            lines.append(f"Up next: {state.next_model} · {tuned}")
        else:
            lines.append("Choosing the next piece…")
        self.gallery_label.setText("<br>".join(lines))
        self.open_gallery_button.setToolTip(state.folder)

    def set_step_target(self, value: int) -> None:
        self.target_spin.blockSignals(True)
        self.target_spin.setValue(int(value))
        self.target_spin.blockSignals(False)

    def _ensure_behaviours(self, model: str, labels: tuple[str, ...]) -> None:
        if model == self._model and len(self.behaviour_rows) == len(labels):
            return
        self._model = model
        while self.behaviour_grid.count():
            item = self.behaviour_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.behaviour_rows = []
        for i, label in enumerate(labels):
            color = BEHAVIOUR_COLORS[i % len(BEHAVIOUR_COLORS)]
            name = QLabel(label.capitalize())
            count = QLabel("0")
            count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bar = _meter(color)
            self.behaviour_grid.addWidget(_swatch(color), i * 2, 0)
            self.behaviour_grid.addWidget(name, i * 2, 1)
            self.behaviour_grid.addWidget(count, i * 2, 2)
            self.behaviour_grid.addWidget(bar, i * 2 + 1, 1, 1, 2)
            self.behaviour_rows.append((name, count, bar))
        self.behaviour_title.setText(f"Behaviour · {model}")
        self.sparkline.clear()

    def update_telemetry(self, t: Telemetry) -> None:
        self._ensure_behaviours(t.model, t.behaviour_labels or STATE_LABELS.get(t.model, ()))
        total = max(1, t.agents)
        creating = t.status_counts[CREATING]
        if t.complete and t.paused:
            self._set_status("Complete", COMPLETE_COLOR)
        elif t.complete:
            self._set_status("Continuing past completion", QColor(94, 234, 212))
        elif t.paused:
            self._set_status("Paused", QColor(154, 154, 166))
        else:
            painting = t.status_counts[CREATING] + t.status_counts[REINFORCING]
            self._set_status(f"Working · {painting:,} agents painting", STATUS_COLORS[CREATING])
        self.reason_label.setText(f"Finished: {t.reason}." if t.complete else "")
        self.reason_label.setVisible(t.complete)
        self.elapsed_label.setText(format_duration(t.elapsed))
        self.progress.setValue(int(round(t.progress * 1000)))
        self.progress.setFormat("Complete" if t.complete else "%p% to completion")
        limit_text = f"auto ≤ {t.step_limit:,}" if t.auto_limit else f"{t.step_limit:,}"
        self.step_label.setText(f"Step {t.steps:,} / {limit_text}")
        if t.complete:
            self.eta_label.setText("done")
        elif t.paused:
            self.eta_label.setText("paused")
        else:
            self.eta_label.setText(f"ETA {format_duration(t.eta_seconds)}" if t.eta_seconds is not None else "estimating…")
        self.agents_title.setText(f"Agents · {t.agents:,}")
        for status, (count, bar) in self.status_rows.items():
            value = t.status_counts[status]
            count.setText(f"{value:,} · {100.0 * value / total:.0f}%")
            bar.setValue(int(1000 * value / total))
        for (_, count, bar), value in zip(self.behaviour_rows, t.behaviour_counts, strict=False):
            count.setText(f"{value:,}")
            bar.setValue(int(1000 * value / total))
        self.coverage_label.setText(f"Canvas painted <b>{100 * t.coverage:.1f}%</b> · {t.new_pixels:,.0f} new px/step")
        if t.steps != self._last_steps:
            if t.steps < self._last_steps:
                self.sparkline.clear()
            self.sparkline.push(creating / total, t.coverage)
            self._last_steps = t.steps
        self.top_table.setRowCount(len(t.top))
        self.top_hint.setVisible(not t.top)
        for row, c in enumerate(t.top):
            values = (f"#{c.agent_id}", c.behaviour, f"{c.created:,}", f"{c.x:.0f}, {c.y:.0f}")
            color = STATUS_COLORS.get(STATUS_INDEX.get(c.status, IDLE), STATUS_COLORS[IDLE])
            for col, text in enumerate(values):
                item = self.top_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem()
                    self.top_table.setItem(row, col, item)
                item.setText(text)
                item.setToolTip(f"Agent #{c.agent_id} is {c.status} while {c.behaviour}; it has drawn {c.created:,} new strokes")
                if col == 0:
                    item.setForeground(color)
        self.continue_button.setEnabled(t.complete and t.paused)
