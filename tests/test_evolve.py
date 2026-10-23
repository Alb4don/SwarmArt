import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarmart.config import GENES, MODELS, PALETTE_GENES, Params, update, with_model
from swarmart.evolve import (
    CMAES,
    Cancelled,
    aesthetic_score,
    box_counting_dimension,
    decode,
    encode,
    evolve,
    render_preview,
    spectral_slope,
)


def test_cmaes_minimises_rotated_ellipsoid():
    rng = np.random.default_rng(0)
    n = 8
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    scales = np.logspace(0, 2, n)
    target = np.full(n, 0.3)

    def f(x):
        z = q @ (x - target)
        return -float(np.sum(scales * z * z))

    es = CMAES(np.full(n, 0.8), 0.3, rng)
    for _ in range(250):
        xs = es.ask()
        es.tell(xs, np.array([f(x) for x in xs]))
    assert np.linalg.norm(es.mean - target) < 1e-3
    assert np.allclose(es.C, es.C.T)
    assert np.all(np.linalg.eigvalsh(es.C) > 0)


def test_cmaes_respects_box_and_survives_nan():
    rng = np.random.default_rng(1)
    es = CMAES(np.full(5, 0.5), 2.0, rng)
    for _ in range(20):
        xs = es.ask()
        assert xs.min() >= 0.0 and xs.max() <= 1.0
        scores = rng.random(len(xs))
        scores[0] = np.nan
        es.tell(xs, scores)
    assert np.isfinite(es.C).all() and 1e-4 <= es.sigma <= 1.0


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(MODELS), st.lists(st.floats(-5, 5) | st.just(float("nan")), min_size=20, max_size=20))
def test_decode_always_valid(model, genome):
    base = with_model(Params(), model)
    n = len(GENES[model]) + len(PALETTE_GENES)
    p = decode(base, np.asarray(genome[:n]))
    assert p.model == model
    assert decode(base, encode(p)) == p


def test_encode_decode_identity():
    for model in MODELS:
        p = with_model(Params(), model)
        assert decode(p, encode(p)) == p


def test_spectral_slope_recovers_known_exponent():
    rng = np.random.default_rng(3)
    n = 128
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1
    for beta in (1.0, 2.0, 3.0):
        spectrum = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * radius ** (-beta / 2)
        image = np.real(np.fft.ifft2(spectrum))
        assert abs(spectral_slope(image) - beta) < 0.3


def test_box_counting_dimension_reference_shapes():
    line = np.zeros((128, 128))
    line[:, 64] = 1.0
    filled = np.random.default_rng(0).random((128, 128))
    assert abs(box_counting_dimension(line) - 1.0) < 0.2
    assert box_counting_dimension(filled) > 1.7


def test_aesthetic_score_filters_degenerate_images():
    assert aesthetic_score(np.zeros((64, 64, 3)))[0] == 0.0
    assert aesthetic_score(np.ones((64, 64, 3)))[0] == 0.0
    nan_image = np.zeros((64, 64, 3))
    nan_image[0, 0, 0] = np.nan
    assert aesthetic_score(nan_image)[0] == 0.0
    assert aesthetic_score(np.zeros((64, 64)))[0] == 0.0
    noise = np.random.default_rng(0).random((64, 64, 3))
    structured = render_preview(update(with_model(Params(), "flow"), seed=4), 96, 120)
    s_noise, _ = aesthetic_score(noise)
    s_struct, metrics = aesthetic_score(structured)
    assert 0.0 <= s_noise <= 1.0 and 0.0 <= s_struct <= 1.0
    assert s_struct > s_noise
    assert set(metrics) == {"entropy", "coverage", "beta", "fractal_dim", "colorfulness", "hue_variety"}


def test_hue_variety_penalises_monochrome():
    from swarmart.evolve import hue_variety

    rng = np.random.default_rng(2)
    mono = np.zeros((64, 64, 3))
    mono[..., 1] = rng.random((64, 64))
    rainbow = rng.random((64, 64, 3))
    assert hue_variety(mono) < 0.05
    assert hue_variety(rainbow) > 0.5
    assert hue_variety(np.zeros((8, 8, 3))) == 0.0


def test_evolve_never_returns_worse_than_validated_baseline():
    base = update(with_model(Params(), "flow"), seed=21, agents=6000)
    progress = []
    result = evolve(base, generations=2, size=64, steps=50, progress=lambda g, t, s: progress.append((g, t, s)))
    assert [g for g, _, _ in progress] == [1, 2]
    assert result.score >= result.initial_score
    assert result.params.model == "flow"
    assert result.evaluations > 0


def test_evolve_cancellation():
    with pytest.raises(Cancelled):
        evolve(with_model(Params(), "boids"), generations=3, size=48, steps=10, should_stop=lambda: True)
