# Neon AOI Analysis Pipeline — Project Documentation

**Study:** THOWL Gamification Learning Research  
**Hardware:** Pupil Neon Eye Tracker (200 Hz gaze, 30 fps scene camera @ 1600×1200 px)  
**Scale:** 280+ participants across three difficulty levels  
**Last updated:** 2026-05-07

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [Physical Setup and AOI Definitions](#2-physical-setup-and-aoi-definitions)
3. [Repository Structure (Current)](#3-repository-structure-current)
4. [Vendor Libraries](#4-vendor-libraries)
5. [Development History](#5-development-history)
6. [Current File Reference](#6-current-file-reference)
7. [CSV Output Schema](#7-csv-output-schema)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Open Issues and Remaining Work](#9-open-issues-and-remaining-work)

---

## 1. Project Background

Participants wear a Pupil Neon eye tracker while completing a gamified learning task at a physical desk. Pupil Cloud (Pupil Labs' hosted service) was ruled out — the pipeline must run fully offline and automated.

Study difficulty levels and session durations:

| Difficulty | Max Duration |
|---|---|
| Easy | ~15 minutes |
| Medium | ~30 minutes |
| Hard | ~40 minutes |

### Required outputs per recording

- **`analysis.csv`** — per-frame gaze data: timestamp, gaze XY, fixation info, which AOI was hit, transitions, dwell time, visit counts
- **`validation_video.mp4`** — annotated scene video with AOI outlines, gaze dot, fixation circles, scanpath, and current AOI banner
- **Review GUI** — lets the researcher manually correct any misdetections before final export

---

## 2. Physical Setup and AOI Definitions

AprilTag markers (family `tag36h11`) are printed and physically attached to objects on the desk. Six primary AOIs are defined:

| AOI | Tag IDs | Marker Count |
|---|---|---|
| Board | 0, 1, 2, 3 | 4 |
| Left_Box | 8, 9, 10, 11 | 4 |
| Stream_Deck | 6, 7 | 2 |
| Middle_Box | 16, 17, 18, 19 | 4 |
| Right_Box | 20, 21, 22, 23 | 4 |
| Screen | 12, 13, 14, 15 | 4 |

Three sub-AOIs are tracked inside the Screen (game UI elements):

| Sub-AOI | Normalized Region (u_min, u_max, v_min, v_max) |
|---|---|
| Points_Bar | (0.10, 0.35, 0.00, 0.16) — top-left |
| Progress_Bar | (0.70, 0.95, 0.00, 0.16) — top-right |
| Avatar | (0.00, 0.22, 0.78, 1.00) — bottom-left |

---

## 3. Repository Structure (Current)

After the restructuring pass described in Phase 5, the layout is:

```
Pupil-labs/
├── App/
│   ├── config/
│   │   ├── aoi_masks.example.json
│   │   └── settings.json
│   ├── maintenance/
│   │   ├── bootstrap/
│   │   │   └── get-pip.py
│   │   └── patches/
│   │       └── (old one-off patch scripts)
│   ├── runtime/
│   │   ├── watcher.log
│   │   └── watcher.lock
│   ├── scripts/
│   │   ├── START_QT_APP.bat
│   │   └── START_WATCHER.bat
│   ├── src/
│   │   ├── analyze_aois.py         ← core inference engine
│   │   ├── aoi_masks.py            ← optional surface-space mask support
│   │   ├── app_paths.py            ← central path registry + vendor injection
│   │   ├── qt_app.py               ← PySide6 review GUI
│   │   ├── setup_aois.py           ← standalone precheck/validation tool
│   │   └── watch_and_analyze.py    ← headless batch watcher
│   ├── tools/
│   │   ├── check_images.py
│   │   ├── check_tags.py
│   │   └── generate_aoi_masks.py
│   └── vendor/
│       ├── pl-marker-mapper-main/  ← vendored Pupil Labs surface + AOI library
│       └── pl-neon-recording-main/ ← vendored Pupil Labs recording reader
├── Recordings/                     ← live participant recordings (git-ignored)
├── Test recordings/                ← test recordings (git-ignored)
├── images/                         ← AprilTag reference images
└── .gitignore
```

### Entry points

- `App/scripts/START_QT_APP.bat` — launches the review GUI
- `App/scripts/START_WATCHER.bat` — launches the headless batch watcher

### Config and runtime file locations

| File | Location |
|---|---|
| App settings | `App/config/settings.json` |
| Optional AOI masks | `App/config/aoi_masks.json` |
| Watcher log / lock | `App/runtime/` |
| Task Segmentation | `<recording_dir>/aoi_results/tasks.json` |

### Source rules

- Runtime app modules → `App/src/`
- Developer/diagnostic helpers → `App/tools/`
- One-off patch scripts and bootstrap files → `App/maintenance/`
- Do not place logs, locks, or temporary patch files in `App/src/`

---

## 4. Vendor Libraries

Both libraries live under `App/vendor/` and are injected into `sys.path` at startup via `app_paths.ensure_vendor_paths()`. They are not pip-installed.

- **`pl-marker-mapper`** — builds a 3D physical surface model from AprilTag detections; provides `AOI`, `Surface`, `Camera`, `perspective_transform`
- **`pl-neon-recording`** — reads Neon `.bin`/`.time`/`.dtype` sensor files and exposes gaze, scene video, fixations, and calibration as Python objects

### Vendor modification

`App/vendor/pl-marker-mapper-main/src/pupil_labs/marker_mapper/aoi.py` was patched in Phase 1:

```python
# Before
if len(aoi_detections) < 2:
    return False

# After
if len(aoi_detections) < 1:
    return False
```

A single visible marker is now sufficient to bootstrap the 3D surface model inside the library. Note: `analyze_aois.py` still applies an additional outer guard (`== len(aoi.marker_ids)`) that requires all markers for the initial surface lock-in — the vendor patch only affects code paths that call `aoi.initialize()` directly.

---

## 5. Development History

### Git Commit Log

```
6724a41  Initial Commit
17e24ef  Restructure repo, add .gitignore, fix 1-marker AOI initialisation  [Claude]
597f5ae  Fix 90% NoAOI: lower quad-AOI marker threshold from 3 to 1         [Claude]
e2e3b3d  Initial clean commit — AOI gaze analysis pipeline  [orphan push, GitHub remote]
```

`e2e3b3d` exists only on the remote (GitHub) from an earlier orphan-branch force-push to remove Git LFS references. Local master has the full history.

---

### Phase 1 — Initial Commit (`6724a41`) · *User*

Project skeleton created by the user. Contents:

- Vendor libraries bundled under `App/vendor/`
- `check_tags.py` and `check_images.py` as quick diagnostic utilities
- No `.gitignore`; venv and recordings tracked
- No automated analysis or GUI yet

---

### Phase 2 — Claude Session 1 (`17e24ef`) · *Claude*

**Problem:** The pipeline existed but had two blockers:
1. `AOI.initialize()` in the vendor library required ≥ 2 markers, so 2-marker AOIs (e.g., Stream_Deck) could never initialize with anything less.
2. No `.gitignore` existed; a previous push attempt failed because Git LFS had tracked `.mp4` test fixtures from the vendor test suite.

**Changes:**

#### Vendor patch — `aoi.py`
Changed initialization threshold from `< 2` to `< 1` (see Section 4 above).

#### `.gitignore` created
Excludes `Recordings/`, `Test recordings/`, `aoi_results/`, Neon sensor formats (`*.mp4`, `*.bin`, `*.raw`, `*.dtype`, `*.time`, `*.time_aux`, `*.zip`, `*.proto`), Python build artifacts, runtime files, editor/OS files, and `.claude/`.

#### Git LFS fix
An orphan branch was created from only the 163 source code files and force-pushed as the new `master`:

```bash
git checkout --orphan clean-start
git add .gitignore App/ images/
git commit -m "Initial clean commit — AOI gaze analysis pipeline"
git push origin clean-start:master --force
```

#### Root-level SIFT scene re-localizer
A `SceneReLocalizer` class was built at the project root using SIFT + FLANN + RANSAC for boxes that fail AprilTag detection. This is **not** used by the Qt App — it targets test recordings at the project root only.

Key parameters:
- `SIFT_SCALE = 0.25` (400×300 for speed)
- `MATCH_INTERVAL = 3` (re-run SIFT every 3rd frame)
- `CARRY_FORWARD_FRAMES = 30` (reuse last known position up to 30 frames)

---

### Phase 3 — Gemini / Codex Rework · *Gemini + Codex*

Between Claude Session 1 and Session 2, `App/analyze_aois.py` was completely rewritten. No separate commit messages — the changes arrived as a unified new state.

**Problem being solved:** The previous 2D per-frame hit-testing approach required ≥ 3 of 4 tags to be perfectly visible in every single frame. Any head movement, occlusion, or motion blur caused immediate `NoAOI`. This resulted in a ~90% data dropout rate.

#### `App/analyze_aois.py` — complete architectural rewrite

| Aspect | Old version | New version |
|---|---|---|
| AOI detection | Custom geometry on tag centre pixels | `AOI.initialize()` + `AOI.localize()` with 3D surface model |
| Camera model | None (raw pixels) | Full `Camera` with distortion matrix from calibration |
| Polygon source | Convex hull / parallelogram of centres | `get_expanded_surface_boundary()` via `perspective_transform` |
| Edge handling | None | 10% polygon dilation outward |
| Sub-AOIs | Bilinear interpolation in pixel space | `get_sub_aoi_polygon()` via `perspective_transform` |
| Detector | Single detector | Dual detector: `quad_decimate=1.0` (primary) + `2.0` fallback for motion blur |
| Initialization guard | `len(found) < 3` → return None | `len(visible_tags) == len(aoi.marker_ids)` — requires ALL markers once |

**The 1-Tag Rule:** Once a surface is locked into memory from a full clean frame, the system requires only 1 visible tag to reconstruct the entire bounding box per frame, even if the participant turns their head and other tags leave view.

**Multi-pass motion blur detection:** If the primary high-resolution pass fails to find enough tags (typically due to motion blur), detection is immediately re-run on a downsampled (`quad_decimate=2.0`) frame. This improves detection of fast, blurred tags.

**Polygon dilation (10% padding):** `get_expanded_surface_boundary()` mathematically expands the 3D surface outward by 10% before projecting to 2D pixels, catching edge-gaze that was landing outside the strict mathematical polygon.

**Zero hallucinations:** By relying strictly on `pl-marker-mapper` AprilTag homographies instead of background feature-tracking (SIFT), the floating-box hallucinations from earlier versions remain eliminated. If zero tags are visible, the bounding box gracefully disappears.

New functions:
- `make_camera(recording)` — builds `Camera` from recording calibration data
- `get_expanded_surface_boundary(s2i, camera, scale=1.10)` — dilated 2D boundary from 3D surface
- `get_sub_aoi_polygon(s2i, camera, u_min, u_max, v_min, v_max)` — sub-AOI via perspective transform

#### Initialization bug ("Tiny Box" fix)

After the 3D update a severe bug was found — detection fell to ~1%. The cause: the system was eagerly initializing the surface on the first frame it saw *any* tag. If that frame only caught a single corner tag, the entire surface was defined as the size of that single 5×5 cm tag for the rest of the recording.

**Fix:** The initialization loop was rewritten in both `analyze_aois.py` and `setup_aois.py` to wait until a frame has a clear view of **all** expected tags for the object. That full-visibility frame locks in the true physical dimensions. After that, the 1-Tag Rule takes over.

#### `App/app_paths.py` — `TEST_RECORDINGS_DIR` added

```python
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
```

#### `App/qt_app.py` — custom source folder picker

- New `QComboBox` to switch between `"Recordings"` and `"Test recordings"` folders
- `_get_current_source_dir()` helper returns the active path
- `_refresh_recordings()` uses the selected source
- Cascading update: changing the parent source folder automatically refreshes the recording dropdown

#### `App/setup_aois.py` — standalone terminal precheck tool (new file)

Samples 8 evenly-spaced frames, opens OpenCV windows for visual inspection, reports detection rates. Controls: `SPACE/ENTER` = next frame, `C` = confirm, `Q` = skip. AOI is considered OK if ≥ 50% of sampled frames detect it. Intended as a pre-flight check before committing to 15–40 minute processing runs.

#### State preservation (autosave and draft saving)

- The app now continuously autosaves all manual annotations to `review_state.json` in the background (600 ms debounce after any edit)
- On next load of the same recording the app restores to the exact frame the user was on
- A dedicated **Save Draft** button was added alongside the final export button

---

### Phase 4 — Claude Session 2 (`597f5ae`) · *Claude*

**Problem:** User reported ~90% NoAOI in Qt App results. Investigation revealed `App/analyze_aois.py` had been rewritten since Session 1 (the SIFT work from Phase 2 was on a different file). Two targeted fixes were applied.

#### `App/analyze_aois.py` — progress reporting interval

```python
# Before
if frame_idx % 100 == 0:

# After
if frame_idx % 30 == 0:
```

At 30 fps, `% 100` updates every ~3.3 seconds. The Qt App polls `progress.json` every 1 second. Changing to `% 30` aligns updates to ~1-second intervals for smooth progress reporting over 15–40 minute runs.

#### `App/qt_app.py` — precheck dialog integrated into GUI flow

`setup_aois.py` had useful pre-validation logic but ran as a separate terminal tool that nobody was using before processing. The equivalent was embedded directly into the Qt App's analysis flow.

New components:

- `WorkerSignals.precheck_ready = Signal(dict)` — carries detection stats from background thread to main thread
- `NeonAoiQtApp._precheck_event` (threading.Event) + `_precheck_proceed` (bool) — synchronisation: background thread emits the signal and waits; main thread shows the dialog, records the decision, sets the event
- `_run_precheck(rec_dir)` — runs in the background thread; loads recording, constructs detector, samples 8 frames, runs CLAHE enhancement + AprilTag detection, counts frames with ≥ 2 markers per AOI, returns `{"counts": {aoi_name: int}, "total": 8}`
- `_on_precheck_ready(stats)` — runs on main thread; shows a `QDialog` with a 3-column table (AOI / Frames detected / Status). Status is green "OK" if ≥ 50% frames detected, red "LOW" otherwise. If any AOI is LOW, shows a warning with actionable advice. Two buttons: **Proceed with Analysis** and **Cancel**

Trigger flow:
```
User clicks "Run / Load Review"
  └─ needs_analysis?
       ├─ No  → load existing results immediately
       └─ Yes → _run_precheck()  (~2–3 seconds)
                  └─ emit precheck_ready
                       └─ show QDialog
                            ├─ Cancel → "Analysis cancelled."
                            └─ Proceed → analyze_aois.analyze_recording()
```

---

### Phase 5 — AOI Robustness + Restructuring Pass · *Claude / Codex*

**Problem:** Even with the 3D surface mapper, a ~90% NoAOI rate was still observed in edge cases. Root cause: the analyzer waited until **all** markers for an AOI appeared in the same frame before the 3D surface path could activate. During the waiting period, valid partial-marker frames were still written as `NoAOI`.

#### 1. Partial-marker AOI fallback — `App/src/analyze_aois.py`

A 2D fallback path now runs before any frame is finally written as `NoAOI`.

Behavior:
- The precise Pupil Labs 3D marker-mapper path is still preferred
- Full marker visibility still initializes the accurate 3D surface model
- If the 3D surface is not initialized or not localized, the analyzer builds a partial 2D AOI polygon from visible marker detections

Fallback rules by marker count:

| AOI type | Visible markers | Fallback action |
|---|---|---|
| 4-marker AOI | 3 markers | Completed quadrilateral |
| 4-marker AOI | 2 markers | Convex hull from marker corners |
| 4-marker AOI | 0–1 markers | No classification |
| 2-marker AOI | Both markers | Fallback classification |
| 2-marker AOI | 1 marker | No classification |

Fallback polygons are slightly expanded with `FALLBACK_POLYGON_SCALE = 1.08` to reduce edge misses.

New/updated helper functions:
- `normalize_quad_corners`
- `_scale_polygon`
- `get_fallback_aoi_polygon`
- `_surface_xy_from_image_quad`
- `_project_norm_points_to_image_quad`
- `_project_region_outline_2d`

New CSV metadata columns: `primary_marker_count`, `primary_surface_initialized`

New `aoi_hit_source` values: `fallback_2d_2tag`, `fallback_2d_3tag`, `bbox_fallback_2d`, `polygon_fallback_2d`, `mask_fallback_2d`

The reviewer can now distinguish high-confidence 3D surface hits from partial-marker fallback hits.

#### 2. Dynamic AOI category handling — `App/src/qt_app.py`

The UI previously relied on a fixed `AOI_NAMES` list. After loading any analysis CSV it now discovers AOIs dynamically.

Added:
- `_discover_aoi_names`
- `_sync_aoi_controls`
- Dynamic `self.aoi_names`
- Dynamic timeline AOI rows

AOIs are discovered from `*_hit` columns in `analysis.csv` and observed `primary_aoi` labels. Backend helper fields are explicitly excluded from categories: `any_aoi`, `any_aoi_hit`, `final_any_aoi_hit`.

#### 3. All-AOI short gap fill — `App/src/qt_app.py`

The old gap fill only closed short `None` gaps between `Board` segments.

Now it fills short `None` gaps for any AOI when both sides of the gap agree:

- `Screen → None → Screen` becomes `Screen`
- `Right_Box → None → Right_Box` becomes `Right_Box`
- `Board → None → Board` still becomes `Board`

Source is still marked as `auto_gap_fill`.

#### 4. Precheck fallback visibility — `App/src/setup_aois.py`

`setup_aois.py` now imports `get_fallback_aoi_polygon` from the analyzer. The sampled validation preview can draw fallback AOI polygons marked with an asterisk (e.g., `Screen*`), making precheck less misleading when a surface is partially visible but not yet fully initialized by the 3D path.

#### 5. Dependency fix — `App/requirements-qt.txt`

Added `plotly`. The UI already imported Plotly for the dashboard but it was missing from the requirements file.

#### 6. App folder restructuring

The `App` folder was reorganized so runtime code, scripts, config, and maintenance artifacts are no longer co-located. All path references in `app_paths.py`, `qt_app.py`, `analyze_aois.py`, `watch_and_analyze.py`, and launcher scripts were updated accordingly. See [Section 3](#3-repository-structure-current) for the current layout.

---

### Phase 6 — Analytics Dashboard & Task Segmentation · *Antigravity*

**Problem:** The user requested an advanced learning curve analysis feature and a dark-mode graphical dashboard embedded directly into the application. Additionally, a robust "Force Restart" mechanism was needed to handle edge cases where the `watch_and_analyze.py` lock file (`.processing`) became orphaned due to a crash.

#### 1. Task Segmentation UI (`App/src/app.py`)
A new "Task Segmentation" control block was added to the Review Studio:
- Allows the researcher to define bounds (Start Frame -> End Frame) for up to 10 Tasks.
- Includes `Set Start`, `Set End`, and `Clear` buttons.
- Task boundaries are visually painted as bright blue strips directly on the `TimelineWidget`.
- Saves state into `tasks.json` inside the respective recording's folder.

#### 2. Learning Curve Dashboard (`App/src/app.py`)
The Plotly Analytics Dashboard was expanded from a single pane to a dual-chart layout (`make_subplots`):
- **Chart 1:** Total Dwell Time (Bar Chart) per AOI over the session.
- **Chart 2:** Learning Curve (Line Graph) plotting Task Number vs. Completion Duration in seconds. It dynamically reads from `tasks.json`.
- The entire dashboard was themed with Plotly's `plotly_dark` template to match the new Dark Glassmorphism Qt styling.

#### 3. Orphaned Lock Handling (`App/src/app.py`)
If `.processing` is detected but no background worker is active, the disabled "Run / Load" button now dynamically transforms into a "Force Restart" button. Clicking it prompts the user to safely delete the orphaned lock file and restart analysis, preventing the app from becoming permanently soft-locked.

---

## 6. Current File Reference

### `App/src/analyze_aois.py` — core inference engine

Per-frame processing loop:
1. Load recording + camera calibration
2. CLAHE contrast enhancement on grayscale frame
3. AprilTag detection — primary (`quad_decimate=1.0`), fallback (`quad_decimate=2.0`) on motion-blur frames
4. For each AOI: attempt full initialization (all markers required once), then localize (1 marker sufficient)
5. Project 3D surface boundary to 2D pixel polygon, dilate 10%
6. Point-in-polygon test for gaze hit
7. If 3D path fails: run 2D partial-marker fallback
8. Screen sub-AOIs via `perspective_transform` on normalized surface coordinates
9. Write CSV row + validation video frame
10. Write `progress.json` every 30 frames

Outputs to `<recording>/aoi_results/raw/`:
- `analysis.csv`
- `validation_video.mp4`
- `progress.json` (live progress)
- `.processing` sentinel (deleted on completion)

### `App/src/qt_app.py` — Neon AOI Review Studio (PySide6 GUI)

Features:
- Source picker: Recordings / Test recordings / custom folder (browse anywhere)
- Recording dropdown + manual folder chooser
- **Re-run detection** checkbox — clears `aoi_results/raw/` and re-runs analysis
- Precheck dialog before any new analysis run
- Live progress bar polling `progress.json` every 1 second
- Video player: play, pause, step frame, jump ±5 seconds
- **Task Segmentation UI**: Map Start/End timestamps for up to 10 analytical Tasks, saved to `tasks.json`.
- **Analytics Dashboard**: Embedded PySide6-WebEngine Plotly dashboard showing Total Dwell Time and a dynamic Learning Curve graph.
- **Force Restart / Orphan Lock Recovery**: Automatically detects crashed background analyses and allows 1-click safe recovery.
- Colour-coded timeline (all AOIs as horizontal strips, dynamically discovered)
- Segment table (filterable by AOI category) with per-segment Play button
- Manual range correction: select segment → adjust start/end spinboxes → Apply Range
- Timeline drag-to-select editing
- **Auto Fill Gaps**: fills short `None` gaps for any AOI where both sides agree
- Undo (up to 50 levels, Ctrl+Z)
- Autosave to `review_state.json` 600 ms after any edit; auto-restored on next load
- **Save Draft** button for explicit mid-session saves
- **Save / Export Final**: writes `aoi_results/analysis.csv` (with `raw_primary_aoi`, `final_primary_aoi`, `edit_source` columns) and re-renders `validation_video.mp4` with corrected labels

### `App/src/setup_aois.py` — standalone terminal precheck

Samples 8 evenly-spaced frames from a recording, opens OpenCV windows, reports detection rates per AOI. Draws fallback polygons (marked with `*`) when a surface is partially visible. Can be run independently as a CLI tool before committing to a full analysis run.

### `App/src/aoi_masks.py` — optional surface-space mask support

Provides optional custom AOI mask overlays defined in `App/config/aoi_masks.json`. Not required for standard operation.

### `App/src/watch_and_analyze.py` — headless batch watcher

Polls `Recordings/` every 30 seconds for new `YYYY-MM-DD-HH-MM-SS` folders. On detecting a new folder:
1. Checks required files exist (`calibration.bin`, `info.json`, `gaze.dtype`)
2. Waits until folder size is stable (file copy finished — up to 60 seconds)
3. Runs `analyze_aois.analyze_recording()` automatically
4. Uses a `.lock` file to prevent multiple instances

Logs to `App/runtime/watcher.log`.

### `App/src/app_paths.py` — central path registry

```python
PROJECT_ROOT        = ...
RECORDINGS_DIR      = PROJECT_ROOT / "Recordings"
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
VENDOR_DIR          = PROJECT_ROOT / "App" / "vendor"
MARKER_MAPPER_SRC   = VENDOR_DIR / "pl-marker-mapper-main" / "src"
NEON_RECORDING_SRC  = VENDOR_DIR / "pl-neon-recording-main" / "src"
```

`ensure_vendor_paths()` injects vendor libraries at the front of `sys.path` so they override any pip-installed versions.

### `App/tools/`

| File | Purpose |
|---|---|
| `check_tags.py` | Print detected tag IDs from a set of images |
| `check_images.py` | Print image dimensions |
| `generate_aoi_masks.py` | Helper for generating `aoi_masks.json` config |

Tools in `App/tools/` add `App/src` to `sys.path` before importing app modules.

---

## 7. CSV Output Schema

### Raw output — `aoi_results/raw/analysis.csv`

| Column | Description |
|---|---|
| `timestamp_ns` | Gaze timestamp in nanoseconds |
| `time_s` | Time since recording start (seconds) |
| `frame_idx` | Scene camera frame index |
| `gaze_x_px`, `gaze_y_px` | Gaze position in scene camera pixels |
| `is_fixation` | Whether this frame falls within a fixation |
| `fixation_id` | Fixation index (blank if saccade/blink) |
| `fixation_dur_ms` | Fixation duration in milliseconds |
| `fixation_gaze_x`, `fixation_gaze_y` | Mean gaze position of the fixation |
| `any_aoi_hit` | True if gaze is inside any AOI |
| `primary_aoi` | Name of the AOI hit (empty = NoAOI) |
| `gaze_on_aoi_x`, `gaze_on_aoi_y` | Reserved (currently empty) |
| `{aoi}_hit` | Boolean per AOI — true if gaze is inside that AOI |
| `aoi_transition` | `"PrevAOI→NewAOI"` string when AOI changes |
| `time_in_current_aoi_ms` | Continuous time spent in current AOI |
| `{aoi}_visits` | Cumulative visit count for that AOI |
| `markers_detected` | Semicolon-separated tag IDs detected this frame |
| `primary_marker_count` | Number of markers visible for the primary AOI this frame |
| `primary_surface_initialized` | Whether the 3D surface model was initialized at this frame |
| `aoi_hit_source` | Detection path used: `3d_surface`, `fallback_2d_2tag`, `fallback_2d_3tag`, `bbox_fallback_2d`, `polygon_fallback_2d`, `mask_fallback_2d` |

### Final output — `aoi_results/analysis.csv` (after Export in Qt App)

Adds three columns to the raw CSV:

| Column | Description |
|---|---|
| `raw_primary_aoi` | Original inference result (before manual edits) |
| `final_primary_aoi` | Post-review result (may differ if researcher corrected it) |
| `edit_source` | `"raw"`, `"manual"`, or `"auto_gap_fill"` |

---

## 8. Running the Pipeline

### Option A — Qt App (recommended for participant-by-participant review)

```bash
App/scripts/START_QT_APP.bat
```

or directly:

```bash
cd App/
venv/Scripts/python.exe src/qt_app.py
```

1. Select source folder (Recordings / Test recordings / custom)
2. Pick a recording from the dropdown
3. Click **Run / Load Review**
4. Review the precheck dialog → Proceed (or Cancel if marker visibility is critically low)
5. Wait for analysis (progress bar shows %)
6. Review timeline, correct any misdetections
7. Click **Save / Export Final**

### Option B — Batch watcher (recommended for processing all recordings overnight)

```bash
App/scripts/START_WATCHER.bat
```

Processes all unanalysed recordings in `Recordings/` automatically. Logs to `App/runtime/watcher.log`.

### Option C — Command line (single recording)

```bash
cd App/
venv/Scripts/python.exe src/analyze_aois.py
venv/Scripts/python.exe src/analyze_aois.py --force   # re-run even if results exist
```

### Option D — Precheck only (terminal)

```bash
cd App/
venv/Scripts/python.exe src/setup_aois.py
```

Visual marker detection check. Run this before committing to a full analysis if you suspect marker visibility problems.

### Environment setup (if venv is missing or broken)

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r App\requirements-qt.txt
```

Dependencies include `PySide6`, `opencv-python`, `numpy`, `pandas`, `plotly`, and the vendored Pupil Labs libraries.

---

## 9. Open Issues and Remaining Work

### Critical — Broken venv

The current tracked venv points to a missing interpreter:
```
C:\Users\varma\AppData\Local\Programs\Python\Python311\python.exe
```

The venv must be rebuilt from a valid Python 3.11 install before any runtime testing can be done reliably. The venv directory should also be untracked from git (it is currently dirty in git status).

### Re-run detection on existing recordings

Old `analysis.csv` files do not contain the new fallback labels (`aoi_hit_source` fallback values) or the new metadata columns (`primary_marker_count`, `primary_surface_initialized`). Re-run detection on any recordings that need these.

### Before/after NoAOI rate comparison

Recommended metrics to validate the Phase 5 fallback improvements:
- Percentage of frames where `primary_aoi` is empty / `NoAOI` (before vs. after)
- Percentage of frames where `aoi_hit_source` contains `fallback_2d`
- Manual correction count after review

### Fallback aggressiveness tuning

Current fallback polygon expansion:
```python
FALLBACK_POLYGON_SCALE = 1.08
```
Increase slightly if edge gazes still become `None`. Decrease if fallback creates false positives. Evaluate against a few recordings with known ground truth before changing.

### Box AOI initialization still requires all 4 markers simultaneously

In `App/src/analyze_aois.py`:

```python
if len(visible_tags) == len(aoi.marker_ids):
    aoi.initialize(detections, camera)
```

For `Left_Box`, `Middle_Box`, `Right_Box` this means all 4 markers must appear in the same frame at least once to bootstrap the 3D surface model. If a participant's body or hand persistently occludes markers, initialization never happens and that AOI will show 100% NoAOI for the entire recording.

The precheck dialog will surface this (0/8 frames detected for that AOI) before wasting processing time.

**Potential fix (not yet applied):** Lower the initialization guard from `== len(aoi.marker_ids)` to `>= 2`, accepting a slightly less accurate initial surface model in exchange for guaranteeing initialization on difficult recordings.

### Project hygiene

- Many local one-off patch scripts remain in `App/maintenance/`
- The venv is tracked and dirty in git
- Clean both before any serious version control or handoff work
