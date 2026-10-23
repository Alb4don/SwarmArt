import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from swarmart.config import FLOAT_LIMITS, MODELS, Params, sanitize, update, with_model
from swarmart.engine import HealthMonitor, Simulation
from swarmart.fields import PaletteLUT, cosine_palette, curl_field

floats = {name: st.floats(lo, hi, allow_nan=False) for name, (lo, hi) in FLOAT_LIMITS.items()}


@st.composite
def small_params(draw):
    data = {name: draw(strategy) for name, strategy in floats.items()}
    data.update(
        model=draw(st.sampled_from(MODELS)),
        width=draw(st.integers(64, 160)),
        height=draw(st.integers(64, 160)),
        agents=draw(st.integers(1, 3000)),
        seed=draw(st.integers(0, 2**31 - 1)),
        steps_per_frame=draw(st.integers(1, 3)),
        palette=draw(st.lists(st.floats(0, 1), min_size=12, max_size=12)),
    )
    return sanitize(data)


def check_state(sim: Simulation) -> None:
    assert sim.accum.shape == (3, sim.h * sim.w)
    assert sim.accum.dtype == np.float32
    assert np.isfinite(sim.accum).all()
    assert (sim.accum >= 0).all()
    x, y = sim.model.positions()
    assert np.isfinite(np.asarray(x, dtype=np.float64)).all()
    assert (np.asarray(x) >= 0).all() and (np.asarray(x) <= sim.w).all()
    assert (np.asarray(y) >= 0).all() and (np.asarray(y) <= sim.h).all()


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(small_params(), st.integers(1, 25))
def test_fuzz_all_models_stay_valid(params, steps):
    sim = Simulation(params)
    sim.step(steps)
    check_state(sim)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(small_params(), small_params())
def test_fuzz_live_parameter_changes(first, second):
    sim = Simulation(first)
    sim.step(3)
    rebuilt = sim.apply(second)
    assert sim.params == second
    assert rebuilt == any(getattr(first, k) != getattr(second, k) for k in ("model", "width", "height", "agents", "seed"))
    sim.step(3)
    check_state(sim)


@settings(max_examples=80, deadline=None)
@given(st.sampled_from(MODELS), st.floats(allow_nan=True, allow_infinity=True), st.floats(allow_nan=True, allow_infinity=True))
def test_fuzz_inject_coordinates(model, fx, fy):
    sim = Simulation(update(with_model(Params(), model), width=96, height=80, agents=500))
    accepted = sim.inject(fx, fy)
    assert accepted == (math.isfinite(fx) and math.isfinite(fy))
    sim.step(2)
    check_state(sim)


@pytest.mark.parametrize("model", MODELS)
def test_determinism(model):
    p = update(with_model(Params(), model), width=128, height=96, agents=800, seed=1234)
    a, b = Simulation(p), Simulation(p)
    a.step(30)
    b.step(30)
    assert np.array_equal(a.accum, b.accum)


@pytest.mark.parametrize("model", MODELS)
def test_models_actually_draw(model):
    p = update(with_model(Params(), model), width=128, height=128, seed=5)
    sim = Simulation(p)
    sim.step(40)
    assert (sim.accum.max(axis=0) > 0).mean() > 0.002


@pytest.mark.parametrize("model", MODELS)
def test_reset_and_clear(model):
    sim = Simulation(update(with_model(Params(), model), width=96, height=96, agents=400))
    sim.step(5)
    sim.clear()
    assert not sim.accum.any()
    sim.step(2)
    sim.reset()
    assert sim.steps == 0 and not sim.accum.any()


@pytest.mark.parametrize("model", ["physarum", "boids", "flow"])
def test_nonfinite_state_is_repaired(model):
    sim = Simulation(update(with_model(Params(), model), width=96, height=96, agents=400))
    sim.model.x[:10] = np.nan
    sim.accum[0, :5] = np.inf
    sim.health.tick = sim.health.interval - 1
    events = sim.step(1)
    assert "nonfinite_canvas" in events or "nonfinite_agents" in events
    sim.health.tick = sim.health.interval - 1
    sim.step(1)
    check_state(sim)


def test_collapse_detection_is_debounced():
    sim = Simulation(update(with_model(Params(), "boids"), width=128, height=128, agents=2000, seed=3))
    monitor = HealthMonitor(interval=1, patience=3)
    sim.model.x[:] = 10.0
    sim.model.y[:] = 10.0
    assert monitor.check(sim) == []
    assert monitor.check(sim) == []
    events = monitor.check(sim)
    assert events == ["swarm_collapse"]
    assert monitor.occupancy(sim.model.x, sim.model.y, sim.w, sim.h, sim.rng) > 0.06


def test_healthy_swarm_never_flagged():
    sim = Simulation(update(with_model(Params(), "flow"), width=128, height=128, agents=3000))
    monitor = HealthMonitor(interval=1, patience=1)
    for _ in range(20):
        sim.model.step()
        assert monitor.check(sim) == []


def test_palette_lut_matches_analytic():
    rng = np.random.default_rng(0)
    palette = tuple(float(v) for v in rng.random(12))
    t = rng.uniform(-0.9, 2.9, 5000).astype(np.float32)
    lut = PaletteLUT()
    approx = lut(t, palette)
    exact = cosine_palette(t, palette)
    assert np.abs(approx - exact).max() < 0.02


def test_curl_field_is_divergence_free_and_bounded():
    vx, vy = curl_field(128, 128, 4.0, np.random.default_rng(1))
    div = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) + (np.roll(vy, -1, 0) - np.roll(vy, 1, 0))
    assert np.abs(div).mean() < 0.05 * np.abs(vx).mean()
    assert np.hypot(vx, vy).max() <= 1.5 + 1e-5


def test_turmite_rules_are_structured():
    for seed in range(40):
        sim = Simulation(update(with_model(Params(), "turmite"), width=64, height=64, seed=seed))
        assert set(sim.model.rule) <= {"L", "R"} and "L" in sim.model.rule and "R" in sim.model.rule
