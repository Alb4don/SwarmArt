import json
import math
import os
import time

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from swarmart.config import FLOAT_LIMITS, MODELS, Params, update, with_model
from swarmart.curator import Curator, RefinementService, descriptor, refine_task

GOOD_METRICS = {"entropy": 0.7, "coverage": 0.5, "beta": 2.2, "fractal_dim": 1.5, "colorfulness": 0.2, "hue_variety": 0.4}


def assert_valid(p: Params) -> None:
    assert p.model in MODELS
    for name, (lo, hi) in FLOAT_LIMITS.items():
        assert lo <= getattr(p, name) <= hi
    assert all(0.0 <= v <= 1.0 for v in p.palette)


def test_explores_every_model_before_exploiting():
    curator = Curator(np.random.default_rng(0))
    current = "physarum"
    seen = []
    for _ in range(len(MODELS)):
        proposal = curator.propose(with_model(Params(), current))
        assert proposal.model != current
        seen.append(proposal.model)
        curator.record(proposal, 0.5, GOOD_METRICS)
        current = proposal.model
    assert set(seen) | {"physarum"} == set(MODELS)


def test_bandit_learns_best_model_and_never_repeats():
    rng = np.random.default_rng(1)
    curator = Curator(np.random.default_rng(2))
    quality = {"physarum": 0.85, "boids": 0.35, "flow": 0.55, "turmite": 0.3}
    current = "boids"
    picks = []
    for _ in range(160):
        proposal = curator.propose(with_model(Params(), current))
        assert proposal.model != current
        picks.append(proposal.model)
        score = float(np.clip(quality[proposal.model] + rng.normal(0, 0.05), 0, 1))
        curator.record(proposal, score, GOOD_METRICS)
        current = proposal.model
    counts = {m: picks.count(m) for m in MODELS}
    assert counts["physarum"] == max(counts.values())
    assert counts["physarum"] >= 60
    assert counts["turmite"] < counts["flow"]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(MODELS), st.integers(0, 2**31 - 1), st.sampled_from([(512, 512), (1920, 1080), (64, 64)]))
def test_proposals_are_valid_and_keep_canvas(model, seed, size):
    base = update(with_model(Params(), model), width=size[0], height=size[1])
    curator = Curator(np.random.default_rng(seed))
    p = curator.propose(base)
    assert_valid(p)
    assert (p.width, p.height) == size
    assert p.model != model
    assert p.target_steps == 0


def test_novelty_rewards_unusual_pieces():
    curator = Curator(np.random.default_rng(3))
    assert curator.novelty(descriptor(GOOD_METRICS)) == 1.0
    for _ in range(6):
        curator.record(with_model(Params(), "flow"), 0.6, GOOD_METRICS)
    assert curator.novelty(descriptor(GOOD_METRICS)) < 0.05
    unusual = dict(GOOD_METRICS, entropy=0.1, coverage=0.95, hue_variety=0.0)
    assert curator.novelty(descriptor(unusual)) > 0.5


@pytest.mark.parametrize(
    "score,metrics",
    [(float("nan"), GOOD_METRICS), (float("inf"), {}), (-5.0, {"entropy": "x"}), (7.0, {"beta": float("nan")}), (0.4, None or {})],
)
def test_record_tolerates_garbage(score, metrics):
    curator = Curator(np.random.default_rng(4))
    result = curator.record(Params(), score, metrics)
    assert 0.0 <= result.score <= 1.0 and math.isfinite(result.reward)
    assert curator.arms["physarum"].plays == 1


def test_state_roundtrip(tmp_path):
    path = tmp_path / "curator.json"
    a = Curator(np.random.default_rng(5), path)
    for model in MODELS:
        a.record(with_model(Params(), model), 0.5 + 0.1 * MODELS.index(model), GOOD_METRICS)
    b = Curator(np.random.default_rng(6), path)
    assert b.pieces == a.pieces == 4
    assert {m: b.arms[m].plays for m in MODELS} == {m: 1 for m in MODELS}
    assert abs(b.arms["turmite"].mean - a.arms["turmite"].mean) < 1e-9
    assert len(b.archive) == 4
    assert not list(tmp_path.glob(".curator-*"))


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True, allow_infinity=True) | st.text(max_size=20),
    lambda c: st.lists(c, max_size=5) | st.dictionaries(st.text(max_size=10), c, max_size=5),
    max_leaves=30,
)


@settings(max_examples=200, deadline=None)
@given(json_values)
def test_hostile_state_values(state):
    curator = Curator(np.random.default_rng(7))
    curator.apply_state(state)
    curator.apply_state({"format": "swarmart-curator/1", "arms": state, "archive": state, "pieces": state, "best": state})
    for arm in curator.arms.values():
        assert arm.plays >= 0 and math.isfinite(arm.mean) and 0.0 <= arm.mean <= 1.0
    assert curator.choose_model() in MODELS
    assert 0.0 <= curator.best_score <= 1.0


@settings(max_examples=60, deadline=None)
@given(st.binary(max_size=600))
def test_hostile_state_files(tmp_path_factory, blob):
    path = tmp_path_factory.mktemp("st") / "curator.json"
    path.write_bytes(blob)
    curator = Curator(np.random.default_rng(8), path)
    assert curator.choose_model() in MODELS


def test_oversized_and_deeply_nested_state(tmp_path):
    path = tmp_path / "curator.json"
    path.write_text("[" * 200_000 + "]" * 200_000)
    assert Curator(np.random.default_rng(9), path).pieces == 0
    path.write_text(json.dumps({"format": "swarmart-curator/1", "arms": {"flow": [10**12, 1e300, -4]}}))
    curator = Curator(np.random.default_rng(9), path)
    assert curator.arms["flow"].plays == 10**9 and curator.arms["flow"].mean <= 1.0


@pytest.mark.parametrize(
    "payload",
    ["", "{}", "[]", "not json", json.dumps({"params": "{}"}), json.dumps({"params": "[" * 5000}), json.dumps({"params": 5})],
)
def test_refine_task_rejects_garbage(payload):
    assert refine_task(payload) is None


def test_refine_task_improves_or_keeps_params():
    from swarmart.config import to_json

    payload = json.dumps({"params": to_json(update(with_model(Params(), "flow"), agents=4000)), "generations": 1, "size": 48, "steps": 30})
    reply = json.loads(refine_task(payload))
    assert reply["score"] >= reply["initial"]


def _wait(service: RefinementService, limit: float = 120.0):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        results = service.poll()
        if results:
            return results
        time.sleep(0.1)
    return []


@pytest.mark.skipif(os.environ.get("SWARMART_SKIP_PROCESS_TESTS") == "1", reason="process tests disabled")
def test_refinement_process_lifecycle_and_crash_recovery():
    service = RefinementService()
    try:
        params = update(with_model(Params(), "flow"), agents=3000)
        task = service.submit(params, generations=1, size=48, steps=30, seed=1)
        results = _wait(service)
        assert [r.task_id for r in results] == [task]
        assert results[0].params is not None and results[0].params.model == "flow"
        doomed = service.submit(params, generations=8, size=128, steps=200, seed=2)
        time.sleep(0.5)
        service._process.terminate()
        service._process.join(5)
        lost = _wait(service, 10)
        assert [(r.task_id, r.params) for r in lost] == [(doomed, None)]
        again = service.submit(params, generations=1, size=48, steps=30, seed=3)
        results = _wait(service)
        assert results and results[0].task_id == again and results[0].params is not None
        assert service.restarts == 1
    finally:
        service.close()
    assert not service.alive
