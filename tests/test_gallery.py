import os
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from swarmart.config import MODELS, Params, update, with_model
from swarmart.engine import Simulation
from swarmart.gallery import AUTOSAVE_PREFIX, Gallery, rotate_autosaves, score_frame
from swarmart.render import Renderer
from swarmart.workers import _render_job


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def make_frame(params: Params, steps: int = 40):
    sim = Simulation(params)
    sim.step(steps)
    return _render_job(Renderer(), sim.accum.copy(), sim.params, sim.steps, True, True)


def test_rotation_only_touches_own_files(tmp_path):
    for i in range(8):
        path = tmp_path / f"{AUTOSAVE_PREFIX}{i:02d}.png"
        path.write_bytes(b"x")
        os.utime(path, (1000 + i, 1000 + i))
    keepers = [tmp_path / "holiday.png", tmp_path / "swarmart_manual.png", tmp_path / f"{AUTOSAVE_PREFIX}notes.txt"]
    for path in keepers:
        path.write_bytes(b"y")
    target = tmp_path / "outside.png"
    target.write_bytes(b"z")
    link = tmp_path / f"{AUTOSAVE_PREFIX}00link.png"
    try:
        link.symlink_to(target)
    except OSError:
        link = None
    assert rotate_autosaves(tmp_path, 3) == 5
    remaining = sorted(p.name for p in tmp_path.glob(f"{AUTOSAVE_PREFIX}*.png") if not p.is_symlink())
    assert remaining == [f"{AUTOSAVE_PREFIX}05.png", f"{AUTOSAVE_PREFIX}06.png", f"{AUTOSAVE_PREFIX}07.png"]
    assert all(p.exists() for p in keepers) and target.exists()
    assert rotate_autosaves(tmp_path / "missing", 3) == 0


def test_score_frame_valid_and_malformed():
    frame = make_frame(update(with_model(Params(), "flow"), width=160, height=120, agents=3000))
    score, metrics = score_frame(frame.data, frame.width, frame.height)
    assert 0.0 < score <= 1.0 and "hue_variety" in metrics
    assert score_frame(b"\x00" * 10, 4, 4) == (0.0, {})
    assert score_frame(b"", 0, 0) == (0.0, {})
    black = np.full(64 * 64, 0xFF000000, dtype=np.uint32).tobytes()
    assert score_frame(black, 64, 64)[0] == 0.0


def test_gallery_cycle_without_window(app, tmp_path):
    current = {"params": update(with_model(Params(), "physarum"), width=128, height=128, agents=2000)}
    gallery = Gallery(lambda: current["params"], tmp_path / "gal", tmp_path / "state.json", hold_seconds=0.2, refine=False)
    requested, advanced, notices = [], [], []
    gallery.autosave_requested.connect(requested.append)
    gallery.advance.connect(advanced.append)
    gallery.notice.connect(notices.append)
    try:
        gallery.set_enabled(True)
        pump(app, 0.2)
        assert gallery.state().next_model not in (None, "physarum")
        gallery.piece_completed("converged")
        gallery.piece_completed("converged")
        assert len(requested) == 1 and requested[0].startswith(str(tmp_path / "gal"))
        assert gallery.is_autosave(requested[0])
        gallery.handle_snapshot(make_frame(current["params"]), requested[0])
        deadline = time.monotonic() + 10
        while (not advanced or gallery.curator.pieces < 1) and time.monotonic() < deadline:
            pump(app, 0.05)
        assert gallery.curator.pieces == 1
        assert advanced and advanced[0].model != "physarum" and advanced[0].model in MODELS
        assert (advanced[0].width, advanced[0].height) == (128, 128)
        assert len(list((tmp_path / "gal").glob(f"{AUTOSAVE_PREFIX}*.png"))) == 1
        assert any("Saved" in n for n in notices)
        current["params"] = advanced[0]
        gallery.piece_completed("time")
        gallery.handle_snapshot(None, requested[-1])
        assert any("could not render" in n.lower() for n in notices)
        deadline = time.monotonic() + 5
        while len(advanced) < 2 and time.monotonic() < deadline:
            pump(app, 0.05)
        assert len(advanced) == 2 and advanced[1].model != advanced[0].model
    finally:
        gallery.shutdown()


def test_gallery_save_failure_is_reported_not_raised(app, tmp_path):
    blocker = tmp_path / "file_not_folder"
    blocker.write_bytes(b"x")
    params = update(with_model(Params(), "flow"), width=96, height=96, agents=1000)
    gallery = Gallery(lambda: params, blocker, None, hold_seconds=0.1, refine=False)
    notices = []
    gallery.notice.connect(notices.append)
    try:
        gallery.set_enabled(True)
        gallery.piece_completed("done")
        gallery.handle_snapshot(make_frame(params, 10), str(blocker / "x.png"))
        deadline = time.monotonic() + 10
        while gallery.curator.pieces < 1 and time.monotonic() < deadline:
            pump(app, 0.05)
        assert gallery.curator.pieces == 1
        assert any("could not be saved" in n for n in notices)
    finally:
        gallery.shutdown()


def test_disabled_gallery_ignores_completion(app, tmp_path):
    params = Params()
    gallery = Gallery(lambda: params, tmp_path, None, refine=False)
    requested = []
    gallery.autosave_requested.connect(requested.append)
    gallery.piece_completed("done")
    assert requested == [] and not gallery.is_autosave("anything")
    gallery.shutdown()


def test_refinement_backlog_never_exceeds_one(app, tmp_path):
    class SlowRefiner:
        def __init__(self):
            self.pending = 0
            self.submitted = 0

        def submit(self, *args, **kwargs):
            self.pending += 1
            self.submitted += 1
            return self.submitted

        def poll(self):
            return []

        def close(self):
            pass

    params = update(with_model(Params(), "flow"), width=96, height=96, agents=500)
    gallery = Gallery(lambda: params, tmp_path, None, hold_seconds=0.0, refine=False)
    gallery.refiner = SlowRefiner()
    try:
        gallery.set_enabled(True)
        for _ in range(12):
            gallery.prepare_next()
            gallery.start_next()
        assert gallery.refiner.submitted == 1
        assert gallery.state().next_model is None or gallery.state().next_model in MODELS
    finally:
        gallery.shutdown()
