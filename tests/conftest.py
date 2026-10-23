import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
_ISOLATED_HOME = tempfile.mkdtemp(prefix="swarmart-test-home-")
os.environ["HOME"] = _ISOLATED_HOME
os.environ["USERPROFILE"] = _ISOLATED_HOME
os.environ["XDG_STATE_HOME"] = os.path.join(_ISOLATED_HOME, ".local", "state")
os.environ["XDG_DATA_HOME"] = os.path.join(_ISOLATED_HOME, ".local", "share")
os.environ["XDG_CONFIG_HOME"] = os.path.join(_ISOLATED_HOME, ".config")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
