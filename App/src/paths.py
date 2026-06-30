from __future__ import annotations

import json
import pathlib
import sys


SRC_DIR = pathlib.Path(__file__).resolve().parent
APP_DIR = SRC_DIR.parent
PROJECT_ROOT = APP_DIR.parent
RECORDINGS_DIR = PROJECT_ROOT / "Recordings"
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
IMAGES_DIR = PROJECT_ROOT / "images"
CONFIG_DIR = APP_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
AOIS_CONFIG_FILE = CONFIG_DIR / "aois.json"
RUNTIME_DIR = APP_DIR / "runtime"
TOOLS_DIR = APP_DIR / "tools"
SCRIPTS_DIR = APP_DIR / "scripts"
APP_ICON = SRC_DIR / "assets" / "neon-player.ico"
APP_ICON_SVG = SRC_DIR / "assets" / "neon-player.svg"
FONTS_DIR = SRC_DIR / "assets" / "fonts"
VENDOR_DIR = APP_DIR / "vendor"
MARKER_MAPPER_SRC = VENDOR_DIR / "pl-marker-mapper-main" / "src"
NEON_RECORDING_SRC = VENDOR_DIR / "pl-neon-recording-main" / "src"

_SKIP_REC_PARTS = frozenset({"aoi_results", "raw", "aoi_master", "aoi_comparison"})


def ensure_vendor_paths() -> None:
    for path in (MARKER_MAPPER_SRC, NEON_RECORDING_SRC):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _folder_has_recordings(folder: pathlib.Path) -> bool:
    if (folder / "info.json").exists():
        return True
    return any((p / "info.json").exists() for p in folder.iterdir() if p.is_dir())


def list_source_folders() -> list[pathlib.Path]:
    """Source folders for the Studio/Dashboard recording picker."""
    seen: set[pathlib.Path] = set()
    folders: list[pathlib.Path] = []

    for raw in load_settings().get("source_folders", []):
        path = pathlib.Path(raw)
        if path.is_dir() and _folder_has_recordings(path):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                folders.append(path)

    if RECORDINGS_DIR.is_dir():
        for child in sorted(RECORDINGS_DIR.iterdir()):
            if child.is_dir() and _folder_has_recordings(child):
                resolved = child.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    folders.append(child)

    return folders


def find_recording_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Return all Neon recording folders under *root* (recursive)."""
    root = pathlib.Path(root)
    if not root.is_dir():
        return []
    found: list[pathlib.Path] = []
    for info in sorted(root.rglob("info.json")):
        rec = info.parent
        if any(part in _SKIP_REC_PARTS for part in rec.parts):
            continue
        found.append(rec)
    return sorted(set(found))


def resolve_app_icon() -> pathlib.Path | None:
    if APP_ICON.exists():
        return APP_ICON
    if APP_ICON_SVG.exists():
        return APP_ICON_SVG
    return None
