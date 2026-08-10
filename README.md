# AOI Studio — Gamification & Visual Attention Eye-Tracking Pipeline

Offline analysis pipeline and desktop application for a study investigating
how gamification influences visual attention during assembly tasks,
measured with a Pupil Neon head-mounted eye tracker.

Participants perform the same assembly task under two conditions —
**Gamified** and **Non-Gamified** — while wearing the eye tracker. This
repository detects the physical work-surfaces (board, component boxes,
control deck, screen) from printed fiducial markers, maps gaze onto them
frame by frame, and produces per-recording, per-condition, and
between-condition statistical outputs.

## What's here

```
App/
  src/          Runtime modules — app.py (PySide6 desktop GUI), analyzer.py
                (per-frame AOI inference engine), rigid_surface.py (scene-wide
                rigid-body reconstruction), reporting.py (aggregate exports +
                condition-comparison statistics), masks.py, paths.py
  tools/        Diagnostic/maintenance scripts
  config/       AOI marker configuration, screen-zone definitions, settings
  vendor/       Vendored Pupil Labs recording-reader and marker-mapper libraries
  START_APP.bat Application entry point
PROJECT_DOCUMENTATION.md   Full project background, physical AOI setup,
                           development history, CSV output schema
CHANGES.md                 Chronological log of pipeline changes
App/APP_STRUCTURE.md       Module-by-module structure overview
```

`Recordings/`, `Test recordings/`, and the venv are not tracked here (see
`.gitignore`) — they contain raw participant data / are environment-specific.

## How it works, in brief

1. **Marker-based scene reconstruction** — AprilTag markers on each surface
   are detected per frame; a one-time calibration pass builds a single
   rigid-body 3D model of the whole desk (including triangulating the
   rarely-visible screen markers), so surfaces stay trackable even when
   their own markers are briefly out of view.
2. **Gaze-to-AOI mapping** — every frame's gaze point is projected into each
   visible surface's normalised coordinates and attributed to the AOI it
   falls on (or `NoAOI`).
3. **Manual review** — a desktop GUI lets a researcher trim dead time,
   correct misclassified spans, segment the task repetitions, and log
   manual assembly errors.
4. **Structured export** — per-recording, per-condition-aggregate, and
   between-condition comparison datasets (dwell %, fixation count/duration,
   gaze entropy, AOI transition rate/matrix, within-Screen spatial
   statistics), with Mann-Whitney U tests and rank-biserial effect sizes.

For the full pipeline description see `PROJECT_DOCUMENTATION.md`.

## Running it

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install -r App\requirements-qt.txt
App\START_APP.bat
```

The app has three tabs: **Studio** (load/analyse/review one recording),
**Dashboard** (per-recording or per-condition-aggregate metrics), and
**Comparison** (Gamified vs Non-Gamified statistics).

## Requirements

Python 3, PySide6, OpenCV, NumPy, pandas, Plotly, SciPy, `pupil_apriltags`.
See `App/requirements-qt.txt` for the full list.
