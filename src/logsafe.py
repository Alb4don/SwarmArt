from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "SwarmArt"
MAX_MESSAGE = 4000

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_TOKEN = re.compile(r"\b(?=[A-Za-z0-9+/_-]*\d)(?=[A-Za-z0-9+/_-]*[A-Za-z])[A-Za-z0-9+/_-]{32,}={0,2}")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SECRET_KV = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)\b\s*[:=]\s*\S+")


def _home_variants() -> list[str]:
    variants = set()
    for candidate in (str(Path.home()), os.environ.get("USERPROFILE", ""), os.environ.get("HOME", "")):
        if candidate and len(candidate) > 1:
            variants.add(candidate)
            variants.add(candidate.replace("\\", "/"))
            variants.add(candidate.replace("/", "\\"))
    return sorted(variants, key=len, reverse=True)


_HOMES = _home_variants()


def redact(text: object) -> str:
    value = str(text)
    for home in _HOMES:
        value = value.replace(home, "~")
    value = _SECRET_KV.sub(lambda m: f"{m.group(1)}=<redacted>", value)
    value = _EMAIL.sub("<email>", value)
    value = _TOKEN.sub("<redacted>", value)
    value = value.replace("\r", " ").replace("\n", " ")
    value = _CONTROL.sub("?", value)
    if len(value) > MAX_MESSAGE:
        value = value[:MAX_MESSAGE] + "...[truncated]"
    return value


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        return True


class SafeFormatter(logging.Formatter):
    def formatException(self, ei) -> str:
        exc_type, exc, tb = ei
        frames = traceback.extract_tb(tb) if tb else []
        lines = [f"{exc_type.__name__ if exc_type else 'Exception'}: {redact(exc)[:500]}"]
        for frame in frames[-12:]:
            lines.append(f"  at {Path(frame.filename).name}:{frame.lineno} in {frame.name}")
        return "\n".join(lines)

    def formatStack(self, stack_info: str) -> str:
        return ""


def log_directory() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "swarmart" / "logs"


def setup_logging(level: int = logging.INFO, directory: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("swarmart")
    if getattr(logger, "_swarmart_ready", False):
        return logger
    logger.setLevel(level)
    logger.propagate = False
    handler: logging.Handler
    try:
        target = directory or log_directory()
        target.mkdir(parents=True, exist_ok=True)
        path = target / "swarmart.log"
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8", delay=False)
        if os.name == "posix":
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)
    except OSError:
        handler = logging.NullHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(SafeFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger._swarmart_ready = True
    return logger


def install_exception_hooks(on_error=None) -> None:
    logger = logging.getLogger("swarmart")

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        if on_error is not None:
            try:
                on_error()
            except Exception:
                logger.error("error callback failed")

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        logger.critical("unhandled thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
