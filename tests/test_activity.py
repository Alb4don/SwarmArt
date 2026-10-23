import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from swarmart.activity import (
    CREATING,
    IDLE,
    MODEL_STEP_CAPS,
    REINFORCING,
    STATE_LABELS,
    ActivityTracker,
)
from swarmart.config import MODELS, Params, update, with_model
from swarmart.engine import Simulation


def run_until_complete(sim: Simulation, limit: int) -> list[float]:
    progress = []
    while not sim.complete and sim.steps < limit:
        sim.step(20)
        progress.append(sim.activity.progress)
    return progress


def bright_canvas(n_pixels: int, bright: list[int]) -> np.ndarray:
    accum = np.zeros((3, n_pixels), dtype=np.float32)
    accum[:, :] = 0.001
    accum[:, bright] = 100.0
    return accum


def test_new_agents_start_as_drawing_new():
    tracker = ActivityTracker("physarum", 50, 8, 8, seed=1)
    assert (tracker.statuses() == CREATING).all()


def test_frontier_versus_reworking_classification():
    accum = bright_canvas(16, [0, 1, 2, 3])
    tracker = ActivityTracker("physarum", 2, 4, 4, seed=1)
    for _ in range(6):
        tracker.observe(np.array([10, 1]), 1.0, accum, 1.6)
    assert tracker.statuses().tolist() == [CREATING, REINFORCING]
    assert tracker.created[0] > 0 and tracker.created[1] == 0
    assert tracker.deposited.all()


def test_agent_moving_from_dark_to_bright_changes_status():
    accum = bright_canvas(16, [5])
    tracker = ActivityTracker("flow", 1, 4, 4, seed=1)
    for _ in range(4):
        tracker.observe(np.array([9]), 1.0, accum, 1.6)
    assert tracker.statuses()[0] == CREATING
    for _ in range(10):
        tracker.observe(np.array([5]), 1.0, accum, 1.6)
    assert tracker.statuses()[0] == REINFORCING


def test_zero_weight_agents_are_idle():
    accum = bright_canvas(16, [15])
    tracker = ActivityTracker("flow", 2, 4, 4, seed=1)
    weight = np.array([0.5, 0.0], dtype=np.float32)
    for _ in range(10):
        tracker.observe(np.array([3, 4]), weight, accum, 1.6)
    assert tracker.statuses().tolist() == [CREATING, IDLE]


def test_turmite_building_draws_and_erasing_reworks():
    tracker = ActivityTracker("turmite", 2, 4, 4, seed=1)
    index = np.array([0, 1, 2, 1, 3, 1])
    weight = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(6):
        tracker.observe(index, weight, None, 1.6, overwrite=True)
    assert tracker.deposited.tolist() == [True, True]
    assert tracker.statuses().tolist() == [CREATING, REINFORCING]
    assert tracker.created[0] > 0 and tracker.created[1] == 0


def test_startup_shows_agents_drawing_in_real_time():
    sim = Simulation(Params())
    sim.step(2)
    first = sim.telemetry()
    assert first.status_counts[CREATING] == first.agents
    fractions = []
    while sim.steps < 400:
        sim.step(20)
        if sim.steps >= 100:
            t = sim.telemetry()
            assert t.status_counts[IDLE] == 0
            fractions.append(t.status_counts[CREATING] / t.agents)
    assert min(fractions) > 0.05, fractions
    sample = sim.agent_sample()
    assert (sample.status == CREATING).sum() > 100
    assert sample.vx is not None and np.isfinite(sample.vx).all()


def test_statuses_partition_population():
    sim = Simulation(update(with_model(Params(), "flow"), width=96, height=96, agents=2000))
    sim.step(30)
    t = sim.telemetry()
    assert sum(t.status_counts) == t.agents == 2000
    assert sum(t.behaviour_counts) == 2000
    assert t.behaviour_labels == STATE_LABELS["flow"]
    assert 0.0 < t.coverage <= 1.0


@pytest.mark.parametrize("model", ["physarum", "flow"])
def test_convergent_models_complete_with_monotone_progress(model):
    sim = Simulation(update(with_model(Params(), model), width=160, height=160, agents=4000, seed=3))
    progress = run_until_complete(sim, 12_000)
    assert sim.complete
    assert "converged" in sim.telemetry().reason
    assert all(b >= a for a, b in zip(progress, progress[1:], strict=False))
    assert sim.activity.progress == 1.0
    assert sim.steps >= 200


def test_step_target_completes_endless_swarm():
    sim = Simulation(update(with_model(Params(), "boids"), width=128, height=128, agents=1500, target_steps=300))
    run_until_complete(sim, 2000)
    t = sim.telemetry()
    assert t.complete and t.steps == 300
    assert t.reason == "step target reached"
    assert t.step_limit == 300 and not t.auto_limit


def test_auto_limit_reported():
    sim = Simulation(update(with_model(Params(), "turmite"), width=64, height=64))
    t = sim.telemetry()
    assert t.auto_limit and t.step_limit == MODEL_STEP_CAPS["turmite"]


def test_no_premature_completion_with_fade():
    sim = Simulation(update(with_model(Params(), "boids"), width=128, height=128, agents=3000, fade=0.01))
    sim.step(400)
    assert not sim.complete


def test_clear_and_reset_restart_completion():
    sim = Simulation(update(with_model(Params(), "flow"), width=96, height=96, agents=1000, target_steps=100))
    run_until_complete(sim, 500)
    assert sim.complete
    sim.clear()
    assert not sim.complete and sim.steps == 0 and sim.telemetry().coverage == 0.0
    run_until_complete(sim, 500)
    assert sim.complete
    sim.reset()
    assert not sim.complete


def test_top_contributors_sorted_and_valid():
    sim = Simulation(update(with_model(Params(), "physarum"), width=128, height=128, agents=3000))
    sim.step(60)
    t = sim.telemetry()
    created = [c.created for c in t.top]
    assert created == sorted(created, reverse=True) and len(t.top) <= 5
    for c in t.top:
        assert 0 <= c.agent_id < 3000 and 0 <= c.x <= 128 and 0 <= c.y <= 128
        assert sim.activity.created[c.agent_id] == c.created


def test_agent_sample_is_stable_and_bounded():
    sim = Simulation(update(with_model(Params(), "boids"), width=200, height=100, agents=8000))
    first = sim.agent_sample()
    sim.step(10)
    second = sim.agent_sample()
    assert np.array_equal(first.ids, second.ids)
    assert second.ids.size == 3000
    assert (second.x >= 0).all() and (second.x <= 200).all() and (second.y <= 100).all()
    assert set(np.unique(second.status).tolist()) <= {CREATING, REINFORCING, IDLE}


def test_nonfinite_canvas_does_not_break_completion_check():
    sim = Simulation(update(with_model(Params(), "flow"), width=64, height=64, agents=500))
    sim.step(20)
    sim.accum[:, :50] = np.nan
    sim.accum[0, 60:80] = np.inf
    sim.step(40)
    assert 0.0 <= sim.activity.progress <= 1.0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sampled_from(MODELS),
    st.integers(64, 140),
    st.integers(64, 140),
    st.integers(1, 3000),
    st.integers(0, 2**31 - 1),
    st.integers(0, 400),
    st.integers(1, 120),
)
def test_fuzz_telemetry_invariants(model, w, h, agents, seed, target, steps):
    sim = Simulation(update(with_model(Params(), model), width=w, height=h, agents=agents, seed=seed, target_steps=target))
    sim.step(steps)
    t = sim.telemetry()
    assert sum(t.status_counts) == t.agents
    assert sum(t.behaviour_counts) == t.agents
    assert 0.0 <= t.progress <= 1.0 and 0.0 <= t.coverage <= 1.0
    assert t.complete == (t.progress == 1.0)
    if target and steps >= target:
        assert t.complete
    sample = sim.agent_sample()
    assert sample.x.size == sample.status.size == sample.ids.size


@pytest.mark.parametrize("fade", [5e-324, 1e-300, 1e-12, 1e-9, 1e-7, 0.2])
def test_tiny_canvas_fade_values_are_safe(fade):
    sim = Simulation(update(with_model(Params(), "physarum"), width=64, height=64, agents=50, fade=fade))
    sim.step(45)
    assert 0.0 <= sim.telemetry().progress <= 1.0
