import time

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from swarmart.app import MainWindow, apply_dark_theme
from swarmart.config import MODELS, Params, update


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_dark_theme(instance)
    return instance


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def window(app):
    w = MainWindow(update(Params(), width=512, height=512, agents=8000), refine=False)
    w.show()
    yield w
    w.close()
    assert not w.worker.isRunning()


def test_frames_flow_and_stats(app, window):
    frames = []
    window.worker.frame_ready.connect(lambda p: frames.append(p))
    pump(app, 2.5)
    assert len(frames) >= 3
    assert frames[-1].image.width() == 512
    assert "agents" in window.status_info.text()


def test_gui_monkey_fuzz(app, window):
    rng = np.random.default_rng(7)
    sliders = list(window.sliders.values())
    for _ in range(160):
        action = int(rng.integers(0, 9))
        if action == 0:
            s = sliders[int(rng.integers(0, len(sliders)))]
            s.slider.setValue(int(rng.integers(0, 1001)))
        elif action == 1:
            window.model_combo.setCurrentIndex(int(rng.integers(0, len(MODELS))))
        elif action == 2:
            window.res_combo.setCurrentIndex(int(rng.integers(0, 2)))
        elif action == 3:
            window.agents_spin.setValue(int(rng.integers(1, 20000)))
        elif action == 4:
            window.toggle_pause()
        elif action == 5:
            window.worker.post(["reset", "clear"][int(rng.integers(0, 2))])
        elif action == 6:
            target = window.canvas.rect()
            QTest.mouseClick(
                window.canvas,
                Qt.MouseButton.LeftButton,
                pos=target.center() + QPointF(rng.integers(-50, 50), rng.integers(-50, 50)).toPoint(),
            )
        elif action == 7:
            window.new_seed()
        else:
            window.random_palette()
        pump(app, 0.01)
    if window.worker.paused:
        window.toggle_pause()
    frames = []
    window.worker.frame_ready.connect(lambda p: frames.append(p))
    pump(app, 2.5)
    assert window.worker.isRunning()
    assert frames, "simulation must keep producing frames after fuzzing"
    assert frames[-1].params.model == window.params.model


def test_model_switch_shows_relevant_controls(app, window):
    window.model_combo.setCurrentIndex(MODELS.index("boids"))
    pump(app, 0.1)
    form, slider = window.slider_rows["cohesion"]
    assert form.isRowVisible(slider)
    form, slider = window.slider_rows["decay"]
    assert not form.isRowVisible(slider)


def test_evolve_runs_and_can_be_cancelled(app, window, monkeypatch):
    window.gen_spin.setValue(2)
    window.toggle_evolve()
    assert window.evolver is not None
    pump(app, 0.3)
    window.toggle_evolve()
    deadline = time.monotonic() + 30
    while window.evolver is not None and time.monotonic() < deadline:
        pump(app, 0.05)
    assert window.evolver is None
    assert window.evolve_button.text() == "Evolve rules"


def test_export_full_resolution(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "art.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "PNG image (*.png)"))
    window = MainWindow(update(Params(), model="flow", width=1920, height=1080, agents=20000), refine=False)
    window.show()
    try:
        frames = []
        window.worker.frame_ready.connect(lambda p: frames.append(p))
        pump(app, 1.5)
        assert frames and frames[-1].width == 960 and frames[-1].height == 540
        window.export_image()
        deadline = time.monotonic() + 20
        while (window._exporting or not target.exists()) and time.monotonic() < deadline:
            pump(app, 0.05)
        from PIL import Image

        with Image.open(target) as im:
            assert im.size == (1920, 1080)
        from swarmart.storage import load_preset

        assert load_preset(target).model == "flow"
    finally:
        window.close()


def test_run_reaches_completion_and_can_continue(app):
    window = MainWindow(
        update(Params(), model="flow", width=128, height=128, agents=1500, target_steps=200), autonomous=False, refine=False
    )
    window.show()
    try:
        completed = []
        telemetry = []
        window.worker.completed.connect(lambda t: completed.append(t))
        window.worker.telemetry_ready.connect(lambda t: telemetry.append(t))
        deadline = time.monotonic() + 30
        while not completed and time.monotonic() < deadline:
            pump(app, 0.05)
        assert completed, "run must report completion"
        result = completed[0]
        assert result.complete and result.steps == 200 and result.reason == "step target reached"
        assert telemetry and telemetry[0].steps < 200
        pump(app, 0.6)
        assert window.worker.paused
        assert "Complete" in window.activity_panel.status_label.text()
        assert window.activity_panel.continue_button.isEnabled()
        assert window.play_button.text() == "Play"
        window.activity_panel.continue_button.click()
        pump(app, 1.0)
        assert not window.worker.paused
        assert "Continuing" in window.activity_panel.status_label.text()
    finally:
        window.close()


def test_overlay_toggle_and_auto_stop_off(app):
    window = MainWindow(update(Params(), model="turmite", width=128, height=128, target_steps=40), autonomous=False, refine=False)
    window.show()
    try:
        window.activity_panel.auto_stop_check.setChecked(False)
        completed = []
        window.worker.completed.connect(lambda t: completed.append(t))
        deadline = time.monotonic() + 20
        while not completed and time.monotonic() < deadline:
            pump(app, 0.05)
        assert completed
        pump(app, 0.3)
        assert not window.worker.paused
        assert window.canvas.show_agents
        window.overlay_action.trigger()
        assert not window.canvas.show_agents and not window.activity_panel.overlay_check.isChecked()
        window.activity_panel.overlay_check.setChecked(True)
        assert window.canvas.show_agents and window.overlay_action.isChecked()
        frames = []
        window.worker.frame_ready.connect(lambda p: frames.append(p))
        pump(app, 0.8)
        assert frames and frames[-1].agents is not None and frames[-1].agents.x.size > 0
        window.canvas.grab()
    finally:
        window.close()


def test_step_target_spin_updates_params(app, window):
    window.activity_panel.target_spin.setValue(5000)
    pump(app, 0.2)
    assert window.params.target_steps == 5000


def test_canvas_actually_shows_the_artwork(app):
    window = MainWindow(update(Params(), model="flow", width=256, height=256, agents=4000), refine=False)
    window.show()
    try:
        pump(app, 2.0)
        for overlay in (False, True):
            window.canvas.set_show_agents(overlay)
            grabbed = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
            data = np.frombuffer(grabbed.constBits(), np.uint8, grabbed.sizeInBytes()).reshape(grabbed.height(), -1, 4)
            assert data[..., :3].mean() > 20, "canvas must render the image, not a transparent frame"
    finally:
        window.close()


def test_agents_visibly_drawing_right_after_launch(app):
    from swarmart.activity import CREATING

    window = MainWindow(refine=False)
    window.show()
    try:
        frames = []
        telemetry = []
        window.worker.frame_ready.connect(lambda p: frames.append(p))
        window.worker.telemetry_ready.connect(lambda t: telemetry.append(t))
        start = time.monotonic()
        while not frames and time.monotonic() - start < 5:
            pump(app, 0.02)
        assert frames, "first frame must arrive quickly"
        assert time.monotonic() - start < 3.0
        first = frames[0]
        assert first.overlay is not None
        assert first.overlay.count(CREATING) >= 1000
        creating_layer = next(layer for layer in first.overlay.layers if layer.status == CREATING)
        assert creating_layer.tails.size() > 0
        pump(app, 0.5)
        assert telemetry and telemetry[-1].status_counts[CREATING] > 0
        assert "agents painting" in window.activity_panel.status_label.text()
    finally:
        window.close()


def test_autonomous_gallery_runs_unattended(app, tmp_path):
    from swarmart.gallery import AUTOSAVE_PREFIX

    folder = tmp_path / "gallery"
    window = MainWindow(
        update(Params(), model="flow", width=192, height=192, agents=3000),
        gallery_dir=folder,
        state_path=tmp_path / "curator.json",
        hold_seconds=0.3,
        refine=False,
    )
    window.activity_panel.piece_limit_spin.setValue(2)
    window.show()
    try:
        started = []
        window.gallery.advance.connect(lambda p: started.append(p.model))
        deadline = time.monotonic() + 60
        while window.gallery.curator.pieces < 3 and time.monotonic() < deadline:
            pump(app, 0.05)
        assert window.gallery.curator.pieces >= 3
        pump(app, 1.0)
        assert window.worker.isRunning()
        assert len(started) >= 2
        sequence = ["flow", *started]
        assert all(a != b for a, b in zip(sequence, sequence[1:], strict=False))
        files = sorted(folder.glob(f"{AUTOSAVE_PREFIX}*.png"))
        assert len(files) >= 3
        from swarmart.storage import load_preset

        assert load_preset(files[0]).model in MODELS
        assert (tmp_path / "curator.json").is_file()
        assert "Pieces finished" in window.activity_panel.gallery_label.text()
    finally:
        window.close()
    assert not window.worker.isRunning()


def test_gallery_toggle_off_stops_advancing(app, tmp_path):
    window = MainWindow(
        update(Params(), model="flow", width=128, height=128, agents=1500),
        gallery_dir=tmp_path / "g",
        state_path=tmp_path / "c.json",
        hold_seconds=0.3,
        refine=False,
    )
    window.activity_panel.piece_limit_spin.setValue(1)
    window.show()
    try:
        window.activity_panel.gallery_check.setChecked(False)
        assert not window.gallery.enabled
        pump(app, 3.0)
        assert window.gallery.curator.pieces == 0
        assert "Off" in window.activity_panel.gallery_label.text()
        window.activity_panel.gallery_check.setChecked(True)
        deadline = time.monotonic() + 30
        while window.gallery.curator.pieces < 1 and time.monotonic() < deadline:
            pump(app, 0.05)
        assert window.gallery.curator.pieces >= 1
    finally:
        window.close()
