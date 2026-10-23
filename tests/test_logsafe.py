import logging
from pathlib import Path

from swarmart.logsafe import SafeFormatter, redact, setup_logging


def test_redaction():
    home = str(Path.home())
    text = redact(f"open {home}/secret/file.png by jhon@example.com token=abc123 key ABCDEFGHIJKLMNOP1234567890abcdefgh\nINJECTED")
    assert home not in text
    assert "example.com" not in text
    assert "abc123" not in text
    assert "ABCDEFGHIJKLMNOP1234567890abcdefgh" not in text
    assert "\n" not in text
    assert "~" in text


def test_redaction_truncates_and_strips_controls():
    text = redact("a\x00b\x1b[31m" + "x" * 10000)
    assert "\x00" not in text and "\x1b" not in text
    assert len(text) < 4100


def test_formatter_hides_source_and_locals():
    formatter = SafeFormatter("%(message)s")
    secret = "hunter2-password"
    try:
        local_secret = secret
        raise ValueError(f"failed at {Path.home()} {len(local_secret)}")
    except ValueError:
        import sys

        rendered = formatter.formatException(sys.exc_info())
    assert "ValueError" in rendered
    assert "local_secret" not in rendered and secret not in rendered
    assert str(Path.home()) not in rendered
    assert "test_logsafe.py" in rendered


def test_log_file_is_redacted(tmp_path):
    logger = logging.getLogger("swarmart")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if hasattr(logger, "_swarmart_ready"):
        del logger._swarmart_ready
    setup_logging(directory=tmp_path)
    logging.getLogger("swarmart.test").warning("user %s saved to %s", "a@b.io", str(Path.home() / "x.png"))
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "swarmart.log").read_text(encoding="utf-8")
    assert "a@b.io" not in content and str(Path.home()) not in content
    assert "<email>" in content
