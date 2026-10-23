from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

MODELS = ("physarum", "boids", "flow", "turmite")
RESOLUTIONS = ((512, 512), (768, 768), (1024, 1024), (1280, 720), (1920, 1080))
MAX_PRESET_BYTES = 64 * 1024
PALETTE_LEN = 12

FLOAT_LIMITS: dict[str, tuple[float, float]] = {
    "speed": (0.2, 4.0),
    "sensor_angle": (0.05, 1.5),
    "sensor_dist": (1.0, 40.0),
    "turn": (0.01, 1.2),
    "decay": (0.5, 0.995),
    "deposit": (0.05, 5.0),
    "noise_scale": (0.5, 12.0),
    "cohesion": (0.0, 2.0),
    "alignment": (0.0, 1.0),
    "separation": (0.0, 3.0),
    "jitter": (0.0, 1.0),
    "hue_spread": (0.0, 1.0),
    "exposure": (0.2, 6.0),
    "glow": (0.0, 1.0),
    "fade": (0.0, 0.2),
}

INT_LIMITS: dict[str, tuple[int, int]] = {
    "width": (64, 3840),
    "height": (64, 2160),
    "agents": (1, 400_000),
    "seed": (0, 2**31 - 1),
    "steps_per_frame": (1, 16),
    "target_steps": (0, 1_000_000),
}

DEFAULT_PALETTE = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.43, 0.43, 0.43, 0.0, 0.33, 0.67)


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Params:
    model: str = "physarum"
    width: int = 768
    height: int = 768
    agents: int = 80_000
    seed: int = 7
    steps_per_frame: int = 2
    target_steps: int = 0
    speed: float = 1.2
    sensor_angle: float = 0.55
    sensor_dist: float = 9.0
    turn: float = 0.4
    decay: float = 0.9
    deposit: float = 1.0
    noise_scale: float = 3.0
    cohesion: float = 0.6
    alignment: float = 0.5
    separation: float = 1.0
    jitter: float = 0.15
    hue_spread: float = 0.5
    exposure: float = 1.6
    glow: float = 0.35
    fade: float = 0.0
    palette: tuple[float, ...] = DEFAULT_PALETTE


MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "physarum": {"agents": 80_000, "steps_per_frame": 2, "speed": 1.2, "deposit": 1.0, "fade": 0.0},
    "boids": {"agents": 30_000, "steps_per_frame": 2, "speed": 1.4, "sensor_dist": 12.0, "fade": 0.01},
    "flow": {"agents": 20_000, "steps_per_frame": 3, "speed": 1.5, "turn": 0.25, "deposit": 1.0, "fade": 0.0},
    "turmite": {"agents": 4, "steps_per_frame": 8, "speed": 3.0, "deposit": 1.0, "fade": 0.0, "hue_spread": 0.6},
}

GENES: dict[str, tuple[str, ...]] = {
    "physarum": ("speed", "sensor_angle", "sensor_dist", "turn", "decay", "deposit", "jitter", "hue_spread"),
    "boids": ("speed", "sensor_dist", "cohesion", "alignment", "separation", "jitter", "hue_spread"),
    "flow": ("speed", "noise_scale", "turn", "deposit", "jitter", "hue_spread"),
    "turmite": ("speed", "hue_spread", "fade"),
}

PALETTE_GENES = tuple(range(3, PALETTE_LEN))

_FIELD_NAMES = tuple(f.name for f in fields(Params))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and 0 < len(value) <= 32:
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def sanitize(data: Any) -> Params:
    base = Params()
    if isinstance(data, Params):
        data = asdict(data)
    if not isinstance(data, dict):
        return base
    values: dict[str, Any] = {}
    model = data.get("model")
    values["model"] = model if isinstance(model, str) and model in MODELS else base.model
    for name, (lo, hi) in FLOAT_LIMITS.items():
        number = _as_number(data.get(name))
        values[name] = _clamp(number, lo, hi) if number is not None else getattr(base, name)
    for name, (lo, hi) in INT_LIMITS.items():
        number = _as_number(data.get(name))
        if number is None or abs(number) > 1e15:
            values[name] = getattr(base, name)
        else:
            values[name] = int(_clamp(round(number), lo, hi))
    values["palette"] = _sanitize_palette(data.get("palette"))
    return Params(**values)


def _sanitize_palette(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != PALETTE_LEN:
        return DEFAULT_PALETTE
    out = []
    for item in raw:
        number = _as_number(item)
        if number is None:
            return DEFAULT_PALETTE
        out.append(_clamp(number, 0.0, 1.0))
    return tuple(out)


def with_model(params: Params, model: str) -> Params:
    if model not in MODELS:
        return params
    merged = asdict(params)
    merged.update(MODEL_DEFAULTS[model])
    merged["model"] = model
    return sanitize(merged)


def update(params: Params, **changes: Any) -> Params:
    merged = asdict(params)
    merged.update({k: v for k, v in changes.items() if k in _FIELD_NAMES})
    return sanitize(merged)


def structural_change(old: Params, new: Params) -> bool:
    return any(getattr(old, name) != getattr(new, name) for name in ("model", "width", "height", "agents", "seed"))


def to_json(params: Params) -> str:
    payload = asdict(params)
    payload["palette"] = list(params.palette)
    payload["format"] = "swarmart/1"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def from_json(text: Any) -> Params:
    if isinstance(text, bytes):
        if len(text) > MAX_PRESET_BYTES:
            raise PresetError("preset too large")
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PresetError("preset is not valid UTF-8") from exc
    if not isinstance(text, str):
        raise PresetError("preset must be text")
    if len(text) > MAX_PRESET_BYTES:
        raise PresetError("preset too large")
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise PresetError("preset is not valid JSON") from exc
    if not isinstance(data, dict):
        raise PresetError("preset must be a JSON object")
    if not any(name in data for name in _FIELD_NAMES):
        raise PresetError("preset has no known fields")
    return sanitize(data)


def random_palette(rng: Any) -> tuple[float, ...]:
    a = [float(x) for x in rng.uniform(0.35, 0.65, 3)]
    b = [float(x) for x in rng.uniform(0.3, 0.55, 3)]
    c = [float(x) for x in rng.uniform(0.1, 0.7, 3)]
    d = [float(x) for x in rng.uniform(0.0, 1.0, 3)]
    return _sanitize_palette(a + b + c + d)


__all__ = [
    "GENES",
    "MAX_PRESET_BYTES",
    "MODELS",
    "MODEL_DEFAULTS",
    "PALETTE_GENES",
    "Params",
    "PresetError",
    "RESOLUTIONS",
    "from_json",
    "random_palette",
    "sanitize",
    "structural_change",
    "to_json",
    "update",
    "with_model",
]
