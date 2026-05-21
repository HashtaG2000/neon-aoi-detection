"""
Automatic Recording Watcher
============================
Watches this folder every 30 seconds for new Neon recording folders
(named YYYY-MM-DD-HH-MM-SS) and automatically runs the full AOI analysis
on any that haven't been processed yet.

The main app starts this watcher automatically. It can still be run manually
for diagnostics with: python App/src/watcher.py

Leave the window open. Every time you export a new recording here,
it will be picked up and processed automatically — no action needed.

Press Ctrl+C to stop watching.

Logs are written to: App/runtime/watcher.log
"""

from __future__ import annotations

import logging
import atexit
import os
import pathlib
import re
import sys
import time

from paths import RECORDINGS_DIR, RUNTIME_DIR, ensure_vendor_paths

ensure_vendor_paths()

# ── Logging: console + file ───────────────────────────────────────────────────

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUNTIME_DIR / "watcher.log"
LOCK_FILE = RUNTIME_DIR / "watcher.lock"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_watcher_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            existing_pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
            if _pid_is_running(existing_pid):
                log.info("Watcher already running with PID %s. Exiting.", existing_pid)
                return False
        except Exception:
            pass

    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")

    def cleanup_lock() -> None:
        try:
            if LOCK_FILE.exists() and LOCK_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_FILE.unlink()
        except Exception:
            pass

    atexit.register(cleanup_lock)
    return True

# ── Config ────────────────────────────────────────────────────────────────────

POLL_INTERVAL_S   = 30    # how often to scan for new folders (seconds)
SETTLE_WAIT_S     = 60    # seconds to wait after detecting a new folder before
                           # processing (gives time for file copying to finish)
RECORDING_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_recording_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and RECORDING_PATTERN.match(p.name)
    )


def is_processed(rec_dir: pathlib.Path) -> bool:
    raw_dir = rec_dir / "aoi_results" / "raw"
    analysis_csv = raw_dir / "analysis.csv"
    validation_video = raw_dir / "validation_video.mp4"
    if (raw_dir / ".processing").exists():
        return True
    return (
        analysis_csv.exists()
        and analysis_csv.stat().st_size > 0
        and validation_video.exists()
        and validation_video.stat().st_size > 0
    )


def is_recording_complete(rec_dir: pathlib.Path) -> bool:
    """
    Check that the recording folder has the minimum files needed.
    This avoids processing a folder while the device is still copying files.
    """
    required = ["calibration.bin", "info.json", "gaze.dtype"]
    return all((rec_dir / f).exists() for f in required)


def folder_is_stable(rec_dir: pathlib.Path, wait_s: int = SETTLE_WAIT_S) -> bool:
    """
    Wait up to wait_s seconds, checking every 5s whether the folder
    size has stopped changing. Returns True when stable or timeout reached.
    """
    log.info("  Waiting %ds for file copy to finish: %s", wait_s, rec_dir.name)
    prev_size = -1
    stable_count = 0
    elapsed = 0
    while elapsed < wait_s:
        try:
            size = sum(f.stat().st_size for f in rec_dir.rglob("*") if f.is_file())
        except Exception:
            size = 0
        if size == prev_size:
            stable_count += 1
            if stable_count >= 2:   # stable for 10 seconds → good to go
                log.info("  Folder stable. Starting analysis.")
                return True
        else:
            stable_count = 0
        prev_size = size
        time.sleep(5)
        elapsed += 5
    log.info("  Timeout reached — proceeding with analysis anyway.")
    return True


# ── Main watch loop ───────────────────────────────────────────────────────────

def watch() -> None:
    root = RECORDINGS_DIR

    log.info("=" * 62)
    log.info("  AOI Watcher started")
    log.info("  Watching : %s", root)
    log.info("  Poll interval : %ds", POLL_INTERVAL_S)
    log.info("  Log file : %s", LOG_FILE)
    log.info("  Press Ctrl+C to stop.")
    log.info("=" * 62)

    # Process any existing unprocessed folders on startup
    existing = [d for d in find_recording_dirs(root) if not is_processed(d)]
    if existing:
        log.info("Found %d unprocessed recording(s) on startup:", len(existing))
        for d in existing:
            log.info("  %s", d.name)
        _process_dirs(existing)
    else:
        log.info("All existing recordings already processed. Watching for new ones...")

    seen: set[str] = {d.name for d in find_recording_dirs(root)}

    while True:
        try:
            time.sleep(POLL_INTERVAL_S)
            current = {d.name: d for d in find_recording_dirs(root)}
            new_names = set(current.keys()) - seen

            for name in sorted(new_names):
                rec_dir = current[name]
                seen.add(name)
                log.info("NEW recording detected: %s", name)

                if not is_recording_complete(rec_dir):
                    log.warning("  Folder incomplete — will retry next cycle.")
                    seen.discard(name)   # retry next scan
                    continue

                folder_is_stable(rec_dir)

                if not is_processed(rec_dir):
                    _process_dirs([rec_dir])
                else:
                    log.info("  Already processed — skipping.")

        except KeyboardInterrupt:
            log.info("")
            log.info("Watcher stopped by user.")
            break
        except Exception:
            log.exception("Unexpected error in watcher loop — continuing.")


def _process_dirs(dirs: list[pathlib.Path]) -> None:
    """Import the analyzer and run it on the given list of directories."""
    import analyzer

    for rec_dir in dirs:
        log.info("Processing: %s", rec_dir.name)
        try:
            analyzer.analyze_recording(rec_dir)
            log.info("Completed: %s", rec_dir.name)
        except Exception:
            log.exception("FAILED: %s", rec_dir.name)


if __name__ == "__main__":
    if acquire_watcher_lock():
        watch()
