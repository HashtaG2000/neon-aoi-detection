from __future__ import annotations

import pathlib
import sys


SRC_DIR = pathlib.Path(__file__).resolve().parent
APP_DIR = SRC_DIR.parent
PROJECT_ROOT = APP_DIR.parent
RECORDINGS_DIR = PROJECT_ROOT / "Recordings"
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
IMAGES_DIR = PROJECT_ROOT / "images"
CONFIG_DIR = APP_DIR / "config"
RUNTIME_DIR = APP_DIR / "runtime"
TOOLS_DIR = APP_DIR / "tools"
SCRIPTS_DIR = APP_DIR / "scripts"
APP_ICON = APP_DIR / "assets" / "neon-player-icon.png"
VENDOR_DIR = APP_DIR / "vendor"
MARKER_MAPPER_SRC = VENDOR_DIR / "pl-marker-mapper-main" / "src"
NEON_RECORDING_SRC = VENDOR_DIR / "pl-neon-recording-main" / "src"


def ensure_vendor_paths() -> None:
    for path in (MARKER_MAPPER_SRC, NEON_RECORDING_SRC):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
