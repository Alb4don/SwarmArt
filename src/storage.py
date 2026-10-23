from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from .config import MAX_PRESET_BYTES, Params, PresetError, from_json, to_json

MAX_IMAGE_FILE_BYTES = 256 * 1024 * 1024
META_KEY = "swarmart"


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    directory = path.parent if str(path.parent) else Path(".")
    fd, tmp = tempfile.mkstemp(prefix=".swarmart-", suffix=path.suffix, dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def ensure_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    return path if path.suffix.lower() == suffix else path.with_name(path.name + suffix)


def export_png(path: str | Path, bgrx: bytes, width: int, height: int, params: Params) -> Path:
    if width <= 0 or height <= 0 or len(bgrx) != width * height * 4:
        raise ValueError("frame size mismatch")
    target = ensure_suffix(path, ".png")
    image = Image.frombytes("RGB", (width, height), bgrx, "raw", "BGRX")
    info = PngInfo()
    info.add_text(META_KEY, to_json(params))
    info.add_text("Software", "SwarmArt")
    _atomic_write(target, lambda handle: image.save(handle, format="PNG", pnginfo=info, compress_level=6))
    return target


def save_preset(path: str | Path, params: Params) -> Path:
    target = ensure_suffix(path, ".json")
    data = to_json(params).encode("utf-8")
    _atomic_write(target, lambda handle: handle.write(data))
    return target


def load_preset(path: str | Path) -> Params:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PresetError("file not accessible") from exc
    if path.suffix.lower() == ".png":
        return _load_png_params(path, size)
    if size > MAX_PRESET_BYTES:
        raise PresetError("preset too large")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PresetError("file not readable") from exc
    return from_json(data)


def _load_png_params(path: Path, size: int) -> Params:
    if size > MAX_IMAGE_FILE_BYTES:
        raise PresetError("image too large")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise PresetError("not a PNG image")
            text = image.info.get(META_KEY)
    except PresetError:
        raise
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise PresetError("image could not be read") from exc
    if not isinstance(text, str):
        raise PresetError("image has no SwarmArt settings")
    return from_json(text)
