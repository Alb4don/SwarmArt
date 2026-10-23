from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__
from .config import MODELS, Params, PresetError, sanitize
from .engine import Simulation
from .render import Renderer
from .storage import export_png, load_preset

log = logging.getLogger("swarmart.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swarmart", description="Emergent art from autonomous agents.")
    parser.add_argument("--version", action="version", version=f"SwarmArt {__version__}")
    parser.add_argument("--preset", help="JSON preset or SwarmArt PNG to start from")
    parser.add_argument("--headless", metavar="OUT.png", help="render without a window and save a PNG")
    parser.add_argument("--steps", type=int, default=600, help="simulation steps for headless rendering")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--size", metavar="WxH", help="canvas size, e.g. 1920x1080")
    parser.add_argument("--agents", type=int)
    parser.add_argument("--evolve", type=int, default=0, metavar="GENERATIONS", help="evolve rules before rendering")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_size(text: str) -> tuple[int, int] | None:
    parts = text.lower().replace("×", "x").split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def resolve_params(args: argparse.Namespace) -> Params:
    params = load_preset(args.preset) if args.preset else Params()
    data = dict(params.__dict__)
    if args.model:
        from .config import with_model

        data = dict(with_model(params, args.model).__dict__)
    if args.seed is not None:
        data["seed"] = args.seed
    if args.agents is not None:
        data["agents"] = args.agents
    if args.size:
        size = _parse_size(args.size)
        if size is None:
            raise PresetError("size must look like 1920x1080")
        data["width"], data["height"] = size
    return sanitize(data)


def run_headless(args: argparse.Namespace) -> int:
    try:
        params = resolve_params(args)
        if args.evolve > 0:
            from .evolve import evolve

            def report(g: int, t: int, s: float) -> None:
                print(f"generation {g}/{t}: best score {s:.3f}", flush=True)

            result = evolve(params, generations=args.evolve, progress=report)
            print(f"validated score {result.initial_score:.3f} -> {result.score:.3f}")
            params = result.params
        steps = max(1, min(int(args.steps), 1_000_000))
        sim = Simulation(params)
        start = time.perf_counter()
        sim.step(steps)
        renderer = Renderer()
        packed = renderer.render(sim.accum, sim.h, sim.w, params.exposure, params.glow)
        target = export_png(args.headless, packed.tobytes(), sim.w, sim.h, params)
        print(f"saved {target} ({steps} steps in {time.perf_counter() - start:.1f}s)")
        return 0
    except PresetError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2
    except OSError:
        log.warning("headless output failed")
        print("Could not write the output file.", file=sys.stderr)
        return 3
    except MemoryError:
        print("Not enough memory for that canvas size or agent count.", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        return 130
    except Exception:
        log.exception("headless render failed")
        print("Rendering failed. Details were written to the log file.", file=sys.stderr)
        return 1
