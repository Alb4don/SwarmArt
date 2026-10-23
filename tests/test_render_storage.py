import contextlib
import struct
import zlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from swarmart.config import Params, PresetError, to_json, update
from swarmart.engine import Simulation
from swarmart.render import Renderer, apply_glow, downsample, preview_factor, unpack_rgb
from swarmart.storage import ensure_suffix, export_png, load_preset, save_preset


def test_tonemap_handles_empty_and_extreme_buffers():
    r = Renderer()
    zero = np.zeros((3, 64 * 32), np.float32)
    assert unpack_rgb(r.pack(zero, 1.0), 32, 64).max() == 0
    hot = np.full((3, 64 * 32), 3e38, np.float32)
    packed = r.pack(hot, 6.0)
    assert unpack_rgb(packed, 32, 64).min() >= 0
    bad = np.zeros((3, 64 * 32), np.float32)
    bad[0, :100] = np.nan
    bad[1, 100:200] = np.inf
    bad[2, 200:300] = -np.inf
    assert r.pack(bad, 1.6).shape == (64 * 32,)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 90), st.integers(1, 90), st.floats(0, 1), st.floats(0.2, 6))
def test_render_any_size(h, w, glow, exposure):
    rng = np.random.default_rng(h * 100 + w)
    accum = rng.random((3, h * w), dtype=np.float32) * 10
    out = Renderer().render(accum, h, w, exposure, glow)
    assert out.dtype == np.uint32 and out.size == h * w


def test_unpack_channel_order():
    accum = np.zeros((3, 4), np.float32)
    accum[0, 0] = 1
    accum[1, 1] = 1
    accum[2, 2] = 1
    rgb = unpack_rgb(Renderer().pack(accum, 4.0), 1, 4)[0]
    assert rgb[0, 0] > 200 and rgb[0, 1] == 0 and rgb[0, 2] == 0
    assert rgb[1, 1] > 200 and rgb[1, 0] == 0
    assert rgb[2, 2] > 200 and rgb[2, 0] == 0
    assert (rgb[3] == 0).all()


def test_glow_brightens_neighbourhood():
    packed = np.zeros(64 * 64, np.uint32)
    packed.reshape(64, 64)[28:36, 28:36] = 0x00FFFFFF
    before = unpack_rgb(packed.copy(), 64, 64).astype(int).sum()
    apply_glow(packed, 64, 64, 1.0)
    assert unpack_rgb(packed, 64, 64).astype(int).sum() > before


def test_export_and_reload_png(tmp_path):
    p = update(Params(), model="flow", seed=99, width=96, height=64, agents=500)
    sim = Simulation(p)
    sim.step(10)
    packed = Renderer().render(sim.accum, sim.h, sim.w, p.exposure, p.glow)
    target = export_png(tmp_path / "art", packed.tobytes(), sim.w, sim.h, p)
    assert target.suffix == ".png"
    with Image.open(target) as im:
        assert im.size == (96, 64) and im.mode == "RGB"
        assert np.array_equal(np.asarray(im), unpack_rgb(packed, 64, 96))
    assert load_preset(target) == p
    assert not list(tmp_path.glob(".swarmart-*"))


def test_export_rejects_size_mismatch(tmp_path):
    with pytest.raises(ValueError):
        export_png(tmp_path / "x.png", b"\x00" * 10, 4, 4, Params())
    assert not list(tmp_path.iterdir())


def test_preset_roundtrip(tmp_path):
    p = update(Params(), model="turmite", seed=5)
    path = save_preset(tmp_path / "p", p)
    assert path.name == "p.json"
    assert load_preset(path) == p


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png(width: int, height: int, extra: bytes = b"") -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = zlib.compress(b"".join(b"\x00" + b"\x00" * width * 3 for _ in range(min(height, 4))))
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + extra + _png_chunk(b"IDAT", raw) + _png_chunk(b"IEND", b"")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\x89PNG\r\n\x1a\n",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
        _png(4, 4),
        _png(4, 4, _png_chunk(b"tEXt", b"swarmart\x00not json")),
        _png(4, 4, _png_chunk(b"tEXt", b"swarmart\x00" + b"[" * 60000)),
        _png(4, 4, _png_chunk(b"tEXt", b"swarmart\x00{}")),
        _png(100000, 100000, _png_chunk(b"tEXt", b"swarmart\x00{bad")),
        b"GIF89a" + b"\x00" * 100,
    ],
)
def test_hostile_png_files(tmp_path, content):
    path = tmp_path / "evil.png"
    path.write_bytes(content)
    with pytest.raises(PresetError):
        load_preset(path)


def test_decompression_bomb_is_refused_even_with_valid_metadata(tmp_path):
    path = tmp_path / "bomb.png"
    meta = to_json(update(Params(), seed=42)).encode()
    path.write_bytes(_png(60000, 60000, _png_chunk(b"tEXt", b"swarmart\x00" + meta)))
    with pytest.raises(PresetError):
        load_preset(path)


def test_large_but_legal_png_metadata_loads(tmp_path):
    path = tmp_path / "large.png"
    meta = to_json(update(Params(), seed=42)).encode()
    path.write_bytes(_png(8000, 8000, _png_chunk(b"tEXt", b"swarmart\x00" + meta)))
    assert load_preset(path).seed == 42


@settings(max_examples=150, deadline=None)
@given(st.binary(max_size=2048))
def test_fuzz_png_loader(tmp_path_factory, blob):
    path = tmp_path_factory.mktemp("fz") / "f.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + blob)
    with contextlib.suppress(PresetError):
        load_preset(path)


@settings(max_examples=150, deadline=None)
@given(st.binary(max_size=2048))
def test_fuzz_json_loader(tmp_path_factory, blob):
    path = tmp_path_factory.mktemp("fj") / "f.json"
    path.write_bytes(blob)
    with contextlib.suppress(PresetError):
        load_preset(path)


def test_missing_and_oversized_files(tmp_path):
    with pytest.raises(PresetError):
        load_preset(tmp_path / "missing.json")
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 200_000)
    with pytest.raises(PresetError):
        load_preset(big)


def test_png_metadata_written_by_other_tool(tmp_path):
    image = Image.new("RGB", (8, 8))
    info = PngInfo()
    info.add_text("swarmart", '{"model": "boids", "seed": 3, "speed": 999}')
    path = tmp_path / "other.png"
    image.save(path, pnginfo=info)
    p = load_preset(path)
    assert p.model == "boids" and p.seed == 3 and p.speed == 4.0


def test_ensure_suffix():
    assert ensure_suffix("a.PNG", ".png").name == "a.PNG"
    assert ensure_suffix("a.tar", ".png").name == "a.tar.png"


@settings(max_examples=60, deadline=None)
@given(st.integers(64, 300), st.integers(64, 300), st.sampled_from([1, 2, 4]))
def test_downsample_preserves_energy(h, w, factor):
    accum = np.random.default_rng(h + w).random((3, h * w), dtype=np.float32)
    out, hh, ww = downsample(accum, h, w, factor)
    assert out.shape == (3, hh * ww) and hh == h // factor and ww == w // factor
    cropped = accum.reshape(3, h, w)[:, : hh * factor, : ww * factor]
    assert np.allclose(out.sum(), cropped.sum(), rtol=1e-4)


def test_preview_factor():
    assert preview_factor(768, 768) == 1
    assert preview_factor(1080, 1920) == 2
    assert preview_factor(2160, 3840) == 4
    assert preview_factor(64, 4000) == 1


def test_packed_pixels_are_opaque_for_qt_rgb32():
    accum = np.random.default_rng(1).random((3, 32 * 32), dtype=np.float32)
    packed = Renderer().render(accum, 32, 32, 1.6, 0.5)
    assert ((packed >> np.uint32(24)) == 0xFF).all()
    zero = Renderer().pack(np.zeros((3, 16), np.float32), 1.0)
    assert (zero == np.uint32(0xFF000000)).all()
