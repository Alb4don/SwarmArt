import contextlib
import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarmart.config import (
    FLOAT_LIMITS,
    INT_LIMITS,
    MAX_PRESET_BYTES,
    MODELS,
    Params,
    PresetError,
    from_json,
    sanitize,
    to_json,
    update,
    with_model,
)

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True, allow_infinity=True) | st.text(max_size=40),
    lambda children: st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=12), children, max_size=6),
    max_leaves=40,
)

field_names = list(FLOAT_LIMITS) + list(INT_LIMITS) + ["model", "palette"]


def assert_valid(p: Params) -> None:
    assert p.model in MODELS
    for name, (lo, hi) in FLOAT_LIMITS.items():
        value = getattr(p, name)
        assert isinstance(value, float) and math.isfinite(value) and lo <= value <= hi, name
    for name, (lo, hi) in INT_LIMITS.items():
        value = getattr(p, name)
        assert isinstance(value, int) and lo <= value <= hi, name
    assert len(p.palette) == 12 and all(0.0 <= v <= 1.0 for v in p.palette)


@settings(max_examples=400, deadline=None)
@given(st.dictionaries(st.sampled_from(field_names) | st.text(max_size=8), json_values, max_size=30))
def test_sanitize_never_produces_invalid_params(data):
    assert_valid(sanitize(data))


@settings(max_examples=200, deadline=None)
@given(json_values)
def test_sanitize_accepts_any_type(data):
    assert_valid(sanitize(data))


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=400))
def test_from_json_rejects_or_validates_arbitrary_text(text):
    try:
        p = from_json(text)
    except PresetError:
        return
    assert_valid(p)


@settings(max_examples=200, deadline=None)
@given(st.binary(max_size=400))
def test_from_json_handles_arbitrary_bytes(blob):
    with contextlib.suppress(PresetError):
        assert_valid(from_json(blob))


@pytest.mark.parametrize(
    "payload",
    [
        "[" * 50_000 + "]" * 50_000,
        '{"a":' * 30_000 + "1" + "}" * 30_000,
        "x" * (MAX_PRESET_BYTES + 1),
        b"\xff\xfe\x00",
        '{"speed": NaN, "decay": Infinity}',
        "[]",
        "{}",
        "null",
        42,
        None,
    ],
)
def test_from_json_hostile_payloads(payload):
    if payload == '{"speed": NaN, "decay": Infinity}':
        p = from_json(payload)
        assert p.speed == Params().speed and p.decay == Params().decay
        return
    with pytest.raises(PresetError):
        from_json(payload)


def test_roundtrip_and_clamping():
    p = update(Params(), speed=99, decay=-5, seed=-3, agents=10**9, model="boids", palette=[2] * 12)
    assert p.speed == FLOAT_LIMITS["speed"][1]
    assert p.decay == FLOAT_LIMITS["decay"][0]
    assert p.seed == 0 and p.agents == INT_LIMITS["agents"][1]
    assert p.palette == (1.0,) * 12
    assert from_json(to_json(p)) == p
    assert json.loads(to_json(p))["format"] == "swarmart/1"


def test_bool_and_bad_strings_are_not_numbers():
    p = sanitize({"speed": True, "agents": "1e400", "seed": "  12 ", "decay": "0x10"})
    assert p.speed == Params().speed
    assert p.agents == Params().agents
    assert p.seed == 12
    assert p.decay == Params().decay


@pytest.mark.parametrize("model", MODELS)
def test_with_model_defaults_valid(model):
    assert_valid(with_model(Params(), model))
    assert with_model(Params(), model).model == model
    assert with_model(Params(), "nope") == Params()
