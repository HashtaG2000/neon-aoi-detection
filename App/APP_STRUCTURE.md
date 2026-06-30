# App Folder Structure

Use `START_APP.bat` as the only user-facing entry point.

```text
App/
  START_APP.bat
  requirements-qt.txt

  config/       JSON config files (aois.json, settings.json, aoi_masks.json)
  src/          runtime Python modules
  vendor/       vendored Pupil Labs libraries
```

## Runtime Modules

- `src/app.py`       — AOI Studio desktop app (PySide6). Three tabs: Studio, Dashboard, Comparison.
- `src/analyzer.py`  — Per-frame AOI inference engine. Writes analysis.csv, fixation_summary.csv, data_quality.json.
- `src/masks.py`     — Surface-space AOI mask config loader (aoi_masks.json).
- `src/paths.py`     — Central path registry and vendor sys.path injection.
- `src/reporting.py` — Master CSV/workbook exports and NonGamified vs Gamified condition comparison.

## Per-Recording Output Files

All outputs land in `<recording>/aoi_results/raw/` after analysis:

| File | Description |
|---|---|
| `analysis.csv` | Per-frame gaze, AOI label, fixation, transition data |
| `fixation_summary.csv` | One row per fixation: duration, dominant AOI, centroid XY |
| `data_quality.json` | Valid gaze %, fixation count, recording duration |
| `progress.json` | Live progress during analysis (polled by app) |
| `tasks.json` | Task repetition start/end frame annotations (saved by app) |

After Export in the app, a final `analysis.csv` is copied to `<recording>/aoi_results/analysis.csv`.

## Research Output Folders

| Folder | Contents |
|---|---|
| `<condition_dir>/aoi_master/` | Master workbook + CSVs from Export Workbook (per condition) |
| `<recordings_root>/aoi_comparison/` | Condition comparison PNGs from Save PNGs in Comparison tab |
