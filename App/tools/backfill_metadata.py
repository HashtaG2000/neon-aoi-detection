"""Backfill auxiliary sensor metadata into already-analysed recordings.

Adds pupillometry, eyelid aperture, saccade, blink, device-worn and head-motion
columns to each recording's analysis.csv, and writes recording_metadata.json
(provenance + session summary) alongside it.

Existing columns are never modified -- this is purely additive, so recordings do
not need to be re-analysed.

Usage:
    python App/tools/backfill_metadata.py                 # all conditions
    python App/tools/backfill_metadata.py Gamified        # one condition
    python App/tools/backfill_metadata.py --force         # redo already-done
"""
from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "App" / "src"))

import metadata  # noqa: E402


def main(argv: list[str]) -> int:
    force = "--force" in argv
    names = [a for a in argv if not a.startswith("--")]
    conditions = names or ["Gamified", "NonGamified"]

    recordings: list[pathlib.Path] = []
    for cond in conditions:
        cdir = ROOT / "Recordings" / cond
        if not cdir.exists():
            print(f"  skip: {cdir} does not exist")
            continue
        for d in sorted(cdir.iterdir()):
            if d.is_dir() and d.name != "aoi_master":
                recordings.append(d)

    print(f"Backfilling metadata for {len(recordings)} recording(s) "
          f"(force={force})\n")
    ok = skipped = failed = 0
    t0 = time.time()
    for i, rec in enumerate(recordings, 1):
        try:
            res = metadata.augment_recording(rec, force=force)
            status = res.get("status", "?")
            if status == "ok":
                ok += 1
                print(f"[{i:2d}/{len(recordings)}] {rec.name:8s} ok "
                      f"({res.get('rows', '?')} rows, "
                      f"+{len(res.get('columns_added', []))} cols)", flush=True)
            else:
                skipped += 1
                print(f"[{i:2d}/{len(recordings)}] {rec.name:8s} {status}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i:2d}/{len(recordings)}] {rec.name:8s} FAILED: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min  |  "
          f"ok={ok}  skipped={skipped}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
