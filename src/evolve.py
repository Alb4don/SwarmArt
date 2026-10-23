from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from .config import FLOAT_LIMITS, GENES, PALETTE_GENES, Params, sanitize
from .engine import Simulation
from .render import Renderer

log = logging.getLogger("swarmart.evolve")

EPS = 1e-4
WEIGHTS = {"entropy": 0.18, "coverage": 0.18, "spectrum": 0.18, "fractal": 0.22, "color": 0.12, "variety": 0.12}


class Cancelled(Exception):
    pass


def gene_names(model: str) -> tuple[str, ...]:
    return GENES.get(model, ())


def encode(params: Params) -> np.ndarray:
    values = []
    for name in gene_names(params.model):
        lo, hi = FLOAT_LIMITS[name]
        values.append((getattr(params, name) - lo) / (hi - lo))
    values.extend(params.palette[i] for i in PALETTE_GENES)
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def decode(base: Params, genome: np.ndarray) -> Params:
    genome = np.clip(np.nan_to_num(np.asarray(genome, dtype=np.float64), nan=0.5), 0.0, 1.0)
    names = gene_names(base.model)
    data = asdict(base)
    for i, name in enumerate(names):
        lo, hi = FLOAT_LIMITS[name]
        data[name] = round(lo + (hi - lo) * float(genome[i]), 6)
    palette = list(base.palette)
    for j, index in enumerate(PALETTE_GENES):
        palette[index] = round(float(genome[len(names) + j]), 6)
    data["palette"] = palette
    return sanitize(data)


def preview_params(params: Params, size: int) -> Params:
    area_ratio = (size * size) / float(params.width * params.height)
    data = asdict(params)
    data.update(width=size, height=size, agents=max(64, int(round(params.agents * area_ratio))))
    if params.model == "turmite":
        data["agents"] = max(8, int(round(min(params.agents, 8192) * area_ratio)))
    return sanitize(data)


def render_preview(params: Params, size: int = 128, steps: int = 160) -> np.ndarray:
    small = preview_params(params, size)
    sim = Simulation(small)
    sim.step(steps)
    return Renderer().to_float(sim.accum, sim.h, sim.w, small.exposure)


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def spectral_slope(lum: np.ndarray) -> float:
    h, w = lum.shape
    spectrum = np.abs(np.fft.rfft2(lum - lum.mean())) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    mask = (radius > 2.0 / min(h, w)) & (radius < 0.45) & (spectrum > 0)
    if mask.sum() < 16:
        return 0.0
    bins = np.logspace(np.log10(radius[mask].min()), np.log10(radius[mask].max()), 20)
    which = np.digitize(radius[mask], bins)
    xs, ys = [], []
    for b in np.unique(which):
        sel = which == b
        if sel.sum() >= 2:
            xs.append(np.log(radius[mask][sel].mean()))
            ys.append(np.log(spectrum[mask][sel].mean()))
    if len(xs) < 4:
        return 0.0
    slope = np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0]
    return float(-slope) if np.isfinite(slope) else 0.0


def box_counting_dimension(lum: np.ndarray) -> float:
    gy, gx = np.gradient(lum)
    magnitude = np.hypot(gx, gy)
    threshold = magnitude.mean() + magnitude.std()
    edges = magnitude > threshold
    side = 1 << int(math.floor(math.log2(max(4, min(edges.shape)))))
    edges = edges[:side, :side]
    if edges.sum() < 10:
        return 0.0
    sizes, counts = [], []
    box = 2
    while box <= side // 4:
        view = edges.reshape(side // box, box, side // box, box).any(axis=(1, 3))
        occupied = int(view.sum())
        if occupied > 0:
            sizes.append(box)
            counts.append(occupied)
        box *= 2
    if len(sizes) < 3:
        return 0.0
    slope = np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    return float(-slope) if np.isfinite(slope) else 0.0


def colorfulness(img: np.ndarray) -> float:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def hue_variety(img: np.ndarray) -> float:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    chroma = np.hypot(rg, yb)
    total = float(chroma.sum())
    if total < 1e-6:
        return 0.0
    resultant = math.hypot(float(rg.sum()), float(yb.sum())) / total
    return float(min(1.0, max(0.0, 1.0 - resultant)))


def aesthetic_score(img: np.ndarray) -> tuple[float, dict[str, float]]:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3 or not np.isfinite(img).all():
        return 0.0, {}
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    if lum.std() < 0.02:
        return 0.0, {"std": float(lum.std())}
    hist, _ = np.histogram(lum, bins=64, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum() / 6.0)
    coverage = float((lum > 0.08).mean())
    beta = spectral_slope(lum)
    dimension = box_counting_dimension(lum)
    chroma = colorfulness(img)
    variety = hue_variety(img)
    metrics = {
        "entropy": entropy,
        "coverage": coverage,
        "beta": beta,
        "fractal_dim": dimension,
        "colorfulness": chroma,
        "hue_variety": variety,
    }
    parts = {
        "entropy": min(1.0, entropy / 0.75),
        "coverage": _gauss(coverage, 0.5, 0.22),
        "spectrum": _gauss(beta, 2.2, 0.6),
        "fractal": _gauss(dimension, 1.55, 0.25),
        "color": min(1.0, chroma / 0.25),
        "variety": min(1.0, variety / 0.45),
    }
    log_score = sum(WEIGHTS[k] * math.log(max(v, EPS)) for k, v in parts.items())
    return float(math.exp(log_score)), metrics


class CMAES:
    def __init__(self, x0: np.ndarray, sigma: float, rng: np.random.Generator, population: int | None = None) -> None:
        n = int(x0.size)
        self.n = n
        self.rng = rng
        self.mean = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma)
        self.lam = population or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        raw = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw / raw.sum()
        self.mueff = 1.0 / float(np.sum(self.weights**2))
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.inv_sqrt_c = np.eye(n)
        self.evaluations = 0

    @staticmethod
    def reflect(x: np.ndarray) -> np.ndarray:
        y = np.mod(x, 2.0)
        return np.where(y > 1.0, 2.0 - y, y)

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ (self.B * self.D).T
        return self.reflect(self.mean + self.sigma * y)

    def tell(self, xs: np.ndarray, fitness: np.ndarray) -> None:
        fitness = np.nan_to_num(np.asarray(fitness, dtype=np.float64), nan=-np.inf)
        order = np.argsort(-fitness)
        selected = xs[order[: self.mu]]
        old = self.mean
        self.mean = self.weights @ selected
        self.evaluations += self.lam
        y_w = (self.mean - old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.inv_sqrt_c @ y_w)
        norm_ps = float(np.linalg.norm(self.ps))
        denom = math.sqrt(1 - (1 - self.cs) ** (2 * self.evaluations / self.lam))
        hsig = 1.0 if norm_ps / max(denom, 1e-12) / self.chi_n < 1.4 + 2 / (self.n + 1) else 0.0
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        steps = (selected - old) / self.sigma
        rank_mu = (steps.T * self.weights) @ steps
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1))
        self.sigma = float(min(max(self.sigma, 1e-4), 1.0))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        if not np.isfinite(self.C).all():
            self.C = np.eye(self.n)
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigenvalues, 1e-20))
        self.inv_sqrt_c = (self.B / self.D) @ self.B.T


@dataclass
class EvolutionResult:
    params: Params
    score: float
    initial_score: float
    evaluations: int


def fitness(params: Params, seeds: tuple[int, ...], size: int, steps: int) -> float:
    scores = []
    for seed in seeds:
        candidate = sanitize({**asdict(params), "seed": seed})
        score, _ = aesthetic_score(render_preview(candidate, size, steps))
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def evolve(
    base: Params,
    generations: int = 8,
    size: int = 112,
    steps: int = 140,
    progress: Callable[[int, int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    rng_seed: int | None = None,
) -> EvolutionResult:
    generations = max(1, min(int(generations), 60))
    rng = np.random.default_rng(rng_seed if rng_seed is not None else base.seed + 977)
    seeds = (base.seed, (base.seed + 1) % (2**31 - 1))
    holdout = ((base.seed + 2) % (2**31 - 1), (base.seed + 3) % (2**31 - 1))
    x0 = encode(base)
    initial = fitness(base, seeds, size, steps)
    elites: list[tuple[float, np.ndarray]] = [(initial, x0)]
    es = CMAES(x0, 0.25, rng)

    def check_stop() -> None:
        if should_stop is not None and should_stop():
            raise Cancelled()

    for generation in range(generations):
        candidates = es.ask()
        scores = np.empty(len(candidates))
        for i, x in enumerate(candidates):
            check_stop()
            scores[i] = fitness(decode(base, x), seeds, size, steps)
        es.tell(candidates, scores)
        for i in np.argsort(-scores)[:2]:
            elites.append((float(scores[i]), candidates[i].copy()))
        elites = sorted(elites, key=lambda item: -item[0])[:4]
        log.info("evolve generation %d/%d best=%.4f sigma=%.4f", generation + 1, generations, elites[0][0], es.sigma)
        if progress is not None:
            progress(generation + 1, generations, elites[0][0])
    validated: list[tuple[float, np.ndarray]] = []
    for score, x in elites:
        check_stop()
        confirm = fitness(decode(base, x), holdout, size, steps)
        validated.append((0.5 * (score + confirm), x))
    initial_validated = next((s for s, x in validated if x is x0), None)
    if initial_validated is None:
        initial_validated = 0.5 * (initial + fitness(base, holdout, size, steps))
    best_score, best_x = max(validated, key=lambda item: item[0])
    if best_score <= initial_validated:
        best_score, best_x = initial_validated, x0
    return EvolutionResult(decode(base, best_x), float(best_score), float(initial_validated), es.evaluations)
