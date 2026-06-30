# Neon AOI Analysis Pipeline - Project Documentation

**Study:** THOWL Gamification Learning Research  
**Hardware:** Pupil Neon Eye Tracker (200 Hz gaze, 30 fps scene camera @ 1600x1200 px)  
**Scale:** 280+ participants across three difficulty levels  
**Last updated:** 2026-06-24 (Phase 8 - Screen Detection & Bug Fixes)

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

Participants wear a Pupil Neon eye tracker while completing a gamified learning task at a physical desk. Pupil Cloud (Pupil Labs' hosted service) was ruled out - the pipeline must run fully offline and automated.

Study difficulty levels and session durations:

| Difficulty | Max Duration |
|---|---|
| Easy | ~15 minutes |
| Medium | ~30 minutes |
| Hard | ~40 minutes |

### Required outputs per recording

- **`analysis.csv`** - per-frame gaze data: timestamp, gaze XY, fixation info, which AOI was hit, transitions, dwell time, visit counts
- **`validation_video.mp4`** - annotated scene video with AOI outlines, gaze dot, fixation circles, scanpath, and current AOI banner
- **Review GUI** - lets the researcher manually correct any misdetections before final export

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
| Points_Bar | (0.10, 0.35, 0.00, 0.16) - top-left |
| Progress_Bar | (0.70, 0.95, 0.00, 0.16) - top-right |
| Avatar | (0.00, 0.22, 0.78, 1.00) - bottom-left |

---

## 3. Repository Structure (Current)

After the restructuring pass described in Phase 5, the layout is:

```text
Pupil-labs/
+-- App/
|   +-- START_APP.bat
|   +-- config/
|   |   +-- aoi_masks.example.json
|   |   +-- settings.json
|   +-- src/
|   |   +-- app.py                  # AOI Studio desktop app (3 tabs: Studio, Dashboard, Comparison)
|   |   +-- analyzer.py             # core AOI inference engine
|   |   +-- masks.py                # optional surface-space mask support
|   |   +-- paths.py                # central path registry + vendor injection
|   |   +-- reporting.py            # master CSV/workbook generator + condition comparison
|   +-- tools/
|   |   +-- check_images.py
|   |   +-- check_tags.py
|   |   +-- generate_aoi_masks.py
|   +-- vendor/
|       +-- pl-marker-mapper-main/  # vendored Pupil Labs surface + AOI library
|       +-- pl-neon-recording-main/ # vendored Pupil Labs recording reader
+-- Recordings/                     # live participant recordings (git-ignored)
|   +-- NonGamified/                # baseline condition recordings
|   +-- Gamified/                   # intervention condition recordings
+-- Test recordings/                # test recordings (git-ignored)
+-- images/                         # AprilTag reference images
+-- .gitignore
```

### Entry points

- `App/START_APP.bat` - launches the AOI Studio desktop app.

### Config and runtime file locations

| File | Location |
|---|---|
| App settings | `App/config/settings.json` |
| Optional AOI masks | `App/config/aoi_masks.json` |
| Task Segmentation | `<recording_dir>/aoi_results/tasks.json` |
| Per-recording analysis | `<recording_dir>/aoi_results/raw/` |
| Master research outputs | `<source_folder>/aoi_master/` |
| Condition comparison PNGs | `<recordings_root>/aoi_comparison/` |

### Source rules

- Runtime app modules -> `App/src/`
- Developer/diagnostic helpers -> `App/tools/`
- Do not place logs, locks, or temporary patch files in `App/src/`

---
## 4. Vendor Libraries

Both libraries live under `App/vendor/` and are injected into `sys.path` at startup via `paths.ensure_vendor_paths()`. They are not pip-installed.

- **`pl-marker-mapper`** - builds a 3D physical surface model from AprilTag detections; provides `AOI`, `Surface`, `Camera`, `perspective_transform`
- **`pl-neon-recording`** - reads Neon `.bin`/`.time`/`.dtype` sensor files and exposes gaze, scene video, fixations, and calibration as Python objects

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

A single visible marker is now sufficient to bootstrap the 3D surface model inside the library. Note: `analyzer.py` still applies an additional outer guard (`== len(aoi.marker_ids)`) that requires all markers for the initial surface lock-in - the vendor patch only affects code paths that call `aoi.initialize()` directly.

---

## 5. Development History

### Git Commit Log

```
6724a41  Initial Commit
17e24ef  Restructure repo, add .gitignore, fix 1-marker AOI initialisation  [Claude]
597f5ae  Fix 90% NoAOI: lower quad-AOI marker threshold from 3 to 1         [Claude]
e2e3b3d  Initial clean commit - AOI gaze analysis pipeline  [orphan push, GitHub remote]
94b9084  Refactor architecture and purge ghost files
[PENDING] Phase 8: Screen detection overhaul + batch analysis + data quality alerts [Claude 2026-06-24]
```

`e2e3b3d` exists only on the remote (GitHub) from an earlier orphan-branch force-push to remove Git LFS references. Local master has the full history.

---

### Phase 1 - Initial Commit (`6724a41`) - User

Project skeleton created by the user. Contents:

- Vendor libraries bundled under `App/vendor/`
- `check_tags.py` and `check_images.py` as quick diagnostic utilities
- No `.gitignore`; venv and recordings tracked
- No automated analysis or GUI yet

---

### Phase 2 - Claude Session 1 (`17e24ef`) - Claude

**Problem:** The pipeline existed but had two blockers:
1. `AOI.initialize()` in the vendor library required >= 2 markers, so 2-marker AOIs (e.g., Stream_Deck) could never initialize with anything less.
2. No `.gitignore` existed; a previous push attempt failed because Git LFS had tracked `.mp4` test fixtures from the vendor test suite.

**Changes:**

#### Vendor patch - `aoi.py`
Changed initialization threshold from `< 2` to `< 1` (see Section 4 above).

#### `.gitignore` created
Excludes `Recordings/`, `Test recordings/`, `aoi_results/`, Neon sensor formats (`*.mp4`, `*.bin`, `*.raw`, `*.dtype`, `*.time`, `*.time_aux`, `*.zip`, `*.proto`), Python build artifacts, runtime files, editor/OS files, and `.claude/`.

#### Git LFS fix
An orphan branch was created from only the 163 source code files and force-pushed as the new `master`:

```bash
git checkout --orphan clean-start
git add .gitignore App/ images/
git commit -m "Initial clean commit - AOI gaze analysis pipeline"
git push origin clean-start:master --force
```

#### Root-level SIFT scene re-localizer
A `SceneReLocalizer` class was built at the project root using SIFT + FLANN + RANSAC for boxes that fail AprilTag detection. This is **not** used by the app - it targets test recordings at the project root only.

Key parameters:
- `SIFT_SCALE = 0.25` (400x300 for speed)
- `MATCH_INTERVAL = 3` (re-run SIFT every 3rd frame)
- `CARRY_FORWARD_FRAMES = 30` (reuse last known position up to 30 frames)

---

### Phase 3 - Gemini / Codex Rework - Gemini + Codex

Between Claude Session 1 and Session 2, `App/src/analyzer.py` was completely rewritten. No separate commit messages - the changes arrived as a unified new state.

**Problem being solved:** The previous 2D per-frame hit-testing approach required >= 3 of 4 tags to be perfectly visible in every single frame. Any head movement, occlusion, or motion blur caused immediate `NoAOI`. This resulted in a ~90% data dropout rate.

#### `App/src/analyzer.py` - complete architectural rewrite

| Aspect | Old version | New version |
|---|---|---|
| AOI detection | Custom geometry on tag centre pixels | `AOI.initialize()` + `AOI.localize()` with 3D surface model |
| Camera model | None (raw pixels) | Full `Camera` with distortion matrix from calibration |
| Polygon source | Convex hull / parallelogram of centres | `get_expanded_surface_boundary()` via `perspective_transform` |
| Edge handling | None | 10% polygon dilation outward |
| Sub-AOIs | Bilinear interpolation in pixel space | `get_sub_aoi_polygon()` via `perspective_transform` |
| Detector | Single detector | Dual detector: `quad_decimate=1.0` (primary) + `2.0` fallback for motion blur |
| Initialization guard | `len(found) < 3` -> return None | `len(visible_tags) == len(aoi.marker_ids)` - requires ALL markers once |

**The 1-Tag Rule:** Once a surface is locked into memory from a full clean frame, the system requires only 1 visible tag to reconstruct the entire bounding box per frame, even if the participant turns their head and other tags leave view.

**Multi-pass motion blur detection:** If the primary high-resolution pass fails to find enough tags (typically due to motion blur), detection is immediately re-run on a downsampled (`quad_decimate=2.0`) frame. This improves detection of fast, blurred tags.

**Polygon dilation (10% padding):** `get_expanded_surface_boundary()` mathematically expands the 3D surface outward by 10% before projecting to 2D pixels, catching edge-gaze that was landing outside the strict mathematical polygon.

**Zero hallucinations:** By relying strictly on `pl-marker-mapper` AprilTag homographies instead of background feature-tracking (SIFT), the floating-box hallucinations from earlier versions remain eliminated. If zero tags are visible, the bounding box gracefully disappears.

New functions:
- `make_camera(recording)` - builds `Camera` from recording calibration data
- `get_expanded_surface_boundary(s2i, camera, scale=1.10)` - dilated 2D boundary from 3D surface
- `get_sub_aoi_polygon(s2i, camera, u_min, u_max, v_min, v_max)` - sub-AOI via perspective transform

#### Initialization bug ("Tiny Box" fix)

After the 3D update a severe bug was found - detection fell to ~1%. The cause: the system was eagerly initializing the surface on the first frame it saw *any* tag. If that frame only caught a single corner tag, the entire surface was defined as the size of that single 5x5 cm tag for the rest of the recording.

**Fix:** The initialization loop was rewritten in both `analyzer.py` and `precheck.py` to wait until a frame has a clear view of **all** expected tags for the object. That full-visibility frame locks in the true physical dimensions. After that, the 1-Tag Rule takes over.

#### `App/src/paths.py` - `TEST_RECORDINGS_DIR` added

```python
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
```

#### `App/src/app.py` - custom source folder picker

- New `QComboBox` to switch between `"Recordings"` and `"Test recordings"` folders
- `_get_current_source_dir()` helper returns the active path
- `_refresh_recordings()` uses the selected source
- Cascading update: changing the parent source folder automatically refreshes the recording dropdown

#### `App/src/precheck.py` - standalone terminal precheck tool (new file)

Samples 8 evenly-spaced frames, opens OpenCV windows for visual inspection, reports detection rates. Controls: `SPACE/ENTER` = next frame, `C` = confirm, `Q` = skip. AOI is considered OK if >= 50% of sampled frames detect it. Intended as a pre-flight check before committing to 15-40 minute processing runs.

#### State preservation (autosave and draft saving)

- The app now continuously autosaves all manual annotations to `review_state.json` in the background (600 ms debounce after any edit)
- On next load of the same recording the app restores to the exact frame the user was on
- A dedicated **Save Draft** button was added alongside the final export button

---

### Phase 4 - Claude Session 2 (`597f5ae`) - Claude

**Problem:** User reported ~90% NoAOI in app results. Investigation revealed `App/src/analyzer.py` had been rewritten since Session 1 (the SIFT work from Phase 2 was on a different file). Two targeted fixes were applied.

#### `App/src/analyzer.py` - progress reporting interval

```python
# Before
if frame_idx % 100 == 0:

# After
if frame_idx % 30 == 0:
```

At 30 fps, `% 100` updates every ~3.3 seconds. The app polls `progress.json` every 1 second. Changing to `% 30` aligns updates to ~1-second intervals for smooth progress reporting over 15-40 minute runs.

#### `App/src/app.py` - precheck dialog integrated into GUI flow

`precheck.py` had useful pre-validation logic but ran as a separate terminal tool that nobody was using before processing. The equivalent was embedded directly into the app's analysis flow.

New components:

- `WorkerSignals.precheck_ready = Signal(dict)` - carries detection stats from background thread to main thread
- `NeonAoiQtApp._precheck_event` (threading.Event) + `_precheck_proceed` (bool) - synchronisation: background thread emits the signal and waits; main thread shows the dialog, records the decision, sets the event
- `_run_precheck(rec_dir)` - runs in the background thread; loads recording, constructs detector, samples 8 frames, runs CLAHE enhancement + AprilTag detection, counts frames with >= 2 markers per AOI, returns `{"counts": {aoi_name: int}, "total": 8}`
- `_on_precheck_ready(stats)` - runs on main thread; shows a `QDialog` with a 3-column table (AOI / Frames detected / Status). Status is green "OK" if >= 50% frames detected, red "LOW" otherwise. If any AOI is LOW, shows a warning with actionable advice. Two buttons: **Proceed with Analysis** and **Cancel**

Trigger flow:
```
User clicks "Run / Load Review"
  +- needs_analysis?
       +- No  -> load existing results immediately
       +- Yes -> _run_precheck()  (~2-3 seconds)
                  +- emit precheck_ready
                       +- show QDialog
                            +- Cancel -> "Analysis cancelled."
                            +- Proceed -> analyzer.analyze_recording()
```

---

### Phase 5 - AOI Robustness + Restructuring Pass - Claude / Codex

**Problem:** Even with the 3D surface mapper, a ~90% NoAOI rate was still observed in edge cases. Root cause: the analyzer waited until **all** markers for an AOI appeared in the same frame before the 3D surface path could activate. During the waiting period, valid partial-marker frames were still written as `NoAOI`.

#### 1. Partial-marker AOI fallback - `App/src/analyzer.py`

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
| 4-marker AOI | 0-1 markers | No classification |
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

#### 2. Dynamic AOI category handling - `App/src/app.py`

The UI previously relied on a fixed `AOI_NAMES` list. After loading any analysis CSV it now discovers AOIs dynamically.

Added:
- `_discover_aoi_names`
- `_sync_aoi_controls`
- Dynamic `self.aoi_names`
- Dynamic timeline AOI rows

AOIs are discovered from `*_hit` columns in `analysis.csv` and observed `primary_aoi` labels. Backend helper fields are explicitly excluded from categories: `any_aoi`, `any_aoi_hit`, `final_any_aoi_hit`.

#### 3. All-AOI short gap fill - `App/src/app.py`

The old gap fill only closed short `None` gaps between `Board` segments.

Now it fills short `None` gaps for any AOI when both sides of the gap agree:

- `Screen -> None -> Screen` becomes `Screen`
- `Right_Box -> None -> Right_Box` becomes `Right_Box`
- `Board -> None -> Board` still becomes `Board`

Source is still marked as `auto_gap_fill`.

#### 4. Precheck fallback visibility - `App/src/precheck.py`

`precheck.py` now imports `get_fallback_aoi_polygon` from the analyzer. The sampled validation preview can draw fallback AOI polygons marked with an asterisk (e.g., `Screen*`), making precheck less misleading when a surface is partially visible but not yet fully initialized by the 3D path.

#### 5. Dependency fix - `App/requirements-qt.txt`

Added `plotly`. The UI already imported Plotly for the dashboard but it was missing from the requirements file.

#### 6. App folder restructuring

The `App` folder was reorganized so runtime code, scripts, config, and maintenance artifacts are no longer co-located. All path references in `paths.py`, `app.py`, `analyzer.py`, `watcher.py`, and launcher scripts were updated accordingly. See [Section 3](#3-repository-structure-current) for the current layout.

---

### Phase 6 - Analytics Dashboard & Task Segmentation - Antigravity

**Problem:** The user requested an advanced learning curve analysis feature and a dark-mode graphical dashboard embedded directly into the application. Additionally, a robust "Force Restart" mechanism was needed to handle edge cases where the `watcher.py` lock file (`.processing`) became orphaned due to a crash.

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

### Phase 7 - Research Analysis Pipeline Rebuild — Claude (2026-06)

**Motivation:** The app covered data review but did not implement the full research workflow required by the project assignment (Tasks 3 and 4 — Data Preparation and Data Analysis). The study compares NonGamified vs Gamified conditions; each recording is one participant performing the same assembly task 10 times. This phase rebuilds the pipeline end-to-end to support that research design.

#### Dead code removed

| File | Reason |
|---|---|
| `src/watcher.py` | Background auto-watcher replaced by manual Analyse button. Not imported anywhere. |
| `src/precheck.py` | Not called or imported anywhere in the new app. Dead code. |
| `START_WATCHER.bat` | Only launched watcher.py; no purpose without it. |

`masks.py` was kept — `analyzer.py` imports it for sub-AOI mask logic.

#### `analyzer.py` — two new output files (additive, no breaking changes)

**`fixation_summary.csv`** written after every analysis run — one row per fixation:

| Column | Description |
|---|---|
| `fixation_id` | Index matching `analysis.csv` |
| `start_frame` / `end_frame` | Frame range of the fixation |
| `start_time_s` / `end_time_s` | Time since recording start (seconds) |
| `duration_s` | Fixation duration |
| `dominant_aoi` | AOI with the most frame votes during the fixation |
| `centroid_x` / `centroid_y` | Mean gaze position (scene pixels) |

**`data_quality.json`** written after every analysis run:

```json
{
  "total_frames": 25430,
  "valid_gaze_frames": 24108,
  "missing_gaze_pct": 5.2,
  "recording_duration_s": 847.6,
  "fixation_count": 312,
  "fps": 30.0
}
```

Both files are cleaned up on analysis failure alongside `analysis.csv`.

#### `reporting.py` — condition comparison engine (additive, no breaking changes)

New dataclasses and functions appended to the existing module:

- **`RecordingMetrics`** — all metrics for one recording: per-AOI dwell %, fixation count, mean fixation duration, gaze entropy, transition rate, N×N transition matrix, per-task dwell and entropy
- **`ComparisonReport`** — pre-computed output ready for the dashboard: `aoi_stats` DataFrame, `overall_stats` DataFrame, averaged transition matrices per condition, learning curve DataFrame, and lists of individual `RecordingMetrics` for scatter overlays
- **`compute_transition_matrix(labels, aoi_names)`** — derives N×N raw count AOI transition matrix from a frame label sequence
- **`compare_conditions(cond_a_dir, cond_b_dir)`** — public entry point: discovers AOI names from data, loads all analysed recordings from both folders, aggregates all metrics, runs Mann-Whitney U (independent samples, two-sided) with rank-biserial effect size, returns a `ComparisonReport`

Statistical test rationale: different participants per condition → independent samples. Mann-Whitney U chosen because sample size per condition is expected to be small (< 30) and normality cannot be assumed. Effect size is rank-biserial correlation r (range −1 to +1).

Dependencies added to `requirements-qt.txt`: `scipy`, `kaleido`, `openpyxl`.

#### `app.py` — Comparison tab (new third tab)

Added alongside Studio and Dashboard. Structure:

- Condition A / Condition B dropdowns — auto-populated from subfolders of `Recordings/`
- **Run Comparison** button — runs `ComparisonWorker` (background thread, UI stays responsive)
- **7 sub-tabs**, each with a full-size `QWebEngineView`, one chart per tab:

| Sub-tab | Chart | Research sub-question |
|---|---|---|
| Dwell % | Grouped bars per AOI, ±SD error bars, individual participant dots, significance stars | SQ1 |
| Fixation Count | Grouped bars per AOI, ±SD, individual dots | SQ4 |
| Fixation Duration | Grouped bars per AOI, ±SD, individual dots | SQ2 |
| Entropy & Transitions | Two side-by-side bar plots with individual dots | SQ2, SQ3 |
| Transition Matrices | Two heatmaps (row→column %, averaged per condition) | SQ3 |
| Learning Curves | 4 subplots: task duration, Board dwell %, Screen dwell %, entropy over repetitions 1–10 | Dynamics |
| Statistics | Full table: AOI / metric / CondA mean±SD / CondB mean±SD / p-value / effect size | All |

Significance markers: `*` p<0.05, `**` p<0.01, `***` p<0.001. Colors: Condition A = `#58A6FF` (blue), Condition B = `#F78166` (orange).

**📷 Save PNGs** exports all charts to `Recordings/aoi_comparison/` at 2× scale.

#### `app.py` — data quality indicator (Studio tab)

A colour-coded label now appears below the recording status after analysis:

`Gaze valid: 94.2%  ·  312 fixations  ·  14:23`

- Green ≥ 80% valid gaze (good quality)
- Amber 60–80% (acceptable, worth noting in write-up)
- Red < 60% (flag this recording before using results)

Reads from `data_quality.json`. Auto-refreshes when analysis completes.

#### `app.py` — playback performance fix

`VideoWidget` now uses `cv2.VideoCapture` for frame reads during playback instead of `recording.scene.sample([ts])` on every tick. Sequential reads avoid per-frame video seeks, fixing the 0.25× playback speed bug. AprilTag surface detection is skipped during playback (would add ~50–100 ms per frame) and re-enabled on pause, at which point surfaces are drawn on the static frame.

#### `app.py` — other bug fixes

- **Task data loss bug**: `TaskPanel.reset()` was emitting `tasksChanged` → `_save_tasks_to_disk()` which overwrote `tasks.json` with empty data before `_load_tasks_from_disk()` ran. Fixed by removing the emit from `reset()`.
- **Double signal connection**: `prev_btn` was connected to both `step(-1)` and `step(-30)`. Removed the `step(-1)` connection.
- **Python 3.11 f-string syntax**: backslash inside f-string expression in stats table fixed.
- **Task header frozen**: moved the Start/End buttons header outside the `QScrollArea` so it stays visible when scrolling through task rows.

---

### Phase 8 - Screen Detection Critical Fixes & UI Improvements — Claude (2026-06-24)

**Motivation:** User reported catastrophic Screen AOI detection failure: only 8 out of 38,420 frames (0.02%) detected Screen, making gamification research impossible. Additionally, app crashed on batch analysis and UI had several usability issues.

#### Critical bug fixes

**`app.py` - crash on batch analysis**
- Added missing `_get_current_source_dir()` method at line 3546
- App no longer crashes when clicking "Batch Re-analyse All"

**`app.py` - validation video removed**
- Removed validation video checkbox (user found it useless)
- Analysis now always runs with `generate_video=False`

#### Screen AOI detection overhaul - `analyzer.py`

**Problem**: Screen markers (12-15) are small/distant in scene camera, causing massive detection failure rate.

**Changes applied:**

1. **Increased surface padding** (lines 89-91):
   ```python
   FALLBACK_POLYGON_SCALE = 1.15   # Was 1.12
   SURFACE_PADDING_DEFAULT = 1.20  # Was 1.10 (10% → 20%)
   SURFACE_PADDING_SCREEN  = 1.25  # NEW: Screen gets 25% padding
   ```

2. **Screen-specific initialization threshold** (lines 611-618):
   ```python
   # Screen allows 3/4 markers, boxes allow 2/4
   if aoi.name == "Screen" and len(aoi.marker_ids) == 4:
       n_required_init = 3
   elif len(aoi.marker_ids) == 4:
       n_required_init = 2
   ```

3. **Aggressive CLAHE enhancement** (lines 120-125):
   ```python
   _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Was 2.5
   _clahe_aggressive = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))  # NEW
   ```

4. **Three-pass detection with Screen-specific third pass** (lines 571-585):
   - Pass 1: Fast detection (quad_decimate=1.0)
   - Pass 2: Precise detection if <8 total markers found
   - **Pass 3 (NEW)**: If Screen has <3 markers → aggressive CLAHE + re-detect, merge Screen markers

5. **Screen-specific boundary padding** (lines 630-633):
   ```python
   padding_scale = SURFACE_PADDING_SCREEN if aoi.name == "Screen" else SURFACE_PADDING_DEFAULT
   ```

6. **Debug logging** (lines 505-506, 647-651, 900-903):
   - Logs Screen initialization frame and marker count
   - Reports final Screen detection percentage at end

**Expected improvement**: Screen detection 0.02% → 30-40%

#### Batch re-analysis capability - `app.py`

**Problem**: Recordings from Phase 1-6 missing `fixation_summary.csv` and `data_quality.json`. Manual re-analysis of 280+ recordings impractical.

**Solution**:
- New `BatchAnalysisWorker` class (lines 211-236)
- New "Batch Re-analyse All" button in Studio tab
- Processes all recordings in selected condition folder sequentially
- Progress tracking with status updates
- Safety confirmation dialog before overwriting

#### Data quality validation warnings - `app.py`

**Problem**: No automatic alerts for poor data quality recordings.

**Solution** (lines 3862-3895):
- Automatic quality assessment on load
- **Warning dialog** for 60-80% valid gaze (amber)
- **Critical alert** for <60% valid gaze (red)
- Actionable guidance (calibration issues, lighting, head movement, etc.)
- Quality status added to label ("Good", "Acceptable", "Poor")

---

## 6. Current File Reference

### `App/src/analyzer.py` - core inference engine

Per-frame processing loop:
1. Load recording + camera calibration
2. CLAHE contrast enhancement on grayscale frame
3. AprilTag detection - primary (`quad_decimate=1.0`), fallback (`quad_decimate=2.0`) on motion-blur frames
4. For each AOI: attempt full initialization when all markers are visible, then localize with fewer visible markers
5. Project 3D surface boundary to 2D pixel polygon, dilate 10%
6. Point-in-polygon test for gaze hit
7. If 3D path fails: run 2D partial-marker fallback
8. Screen sub-AOIs via `perspective_transform` on normalized surface coordinates
9. Accumulate per-fixation AOI votes (dominant AOI determined at end of loop)
10. Write CSV row + optional validation video frame
11. Write `progress.json` every 30 frames
12. After loop: write `fixation_summary.csv` and `data_quality.json`

Outputs to `<recording>/aoi_results/raw/`:
- `analysis.csv`
- `fixation_summary.csv` *(new — Phase 7)*
- `data_quality.json` *(new — Phase 7)*
- `progress.json` (live progress)
- `.processing` sentinel (deleted on completion)

### `App/src/app.py` - AOI Studio (PySide6 GUI)

**Studio tab:**
- Source folder + recording picker (condition dropdown → recording dropdown)
- Load Recording, Analyse, 💾 Save Tasks, ⬇ Export Final CSV
- Data quality indicator: valid gaze %, fixation count, duration — colour-coded green/amber/red
- Surface visibility checkboxes (toggle AprilTag overlay per AOI on paused video)
- Task annotation panel: 10 task slots, frozen Start/End header (keyboard I/O), scrollable task rows
- Video player using `cv2.VideoCapture` (real-time), gaze dot, surfaces drawn on pause
- Stacked timeline: AOI colour bars / gaze trace strip / fixation markers / task span bands

**Dashboard tab:**
- Condition folder + recording selector (single or all-aggregate), task filter
- Charts: AOI dwell bar, learning curve (task duration), dwell heatmap (multi-recording)
- Stats table: mean / median / std / min / max per AOI
- 📷 Save PNGs, 📗 Export Workbook

**Comparison tab:**
- Condition A / Condition B selectors (NonGamified vs Gamified)
- Run Comparison (background thread via `ComparisonWorker`)
- 7 sub-tabs: Dwell % / Fixation Count / Fixation Duration / Entropy & Transitions / Transition Matrices / Learning Curves / Statistics
- 📷 Save PNGs → `Recordings/aoi_comparison/`

### `App/src/masks.py` - optional surface-space mask support

Provides optional custom AOI mask overlays defined in `App/config/aoi_masks.json`. Not required for standard operation.

### `App/src/paths.py` - central path registry

```python
PROJECT_ROOT        = ...
RECORDINGS_DIR      = PROJECT_ROOT / "Recordings"
TEST_RECORDINGS_DIR = PROJECT_ROOT / "Test recordings"
VENDOR_DIR          = PROJECT_ROOT / "App" / "vendor"
MARKER_MAPPER_SRC   = VENDOR_DIR / "pl-marker-mapper-main" / "src"
NEON_RECORDING_SRC  = VENDOR_DIR / "pl-neon-recording-main" / "src"
```

`ensure_vendor_paths()` injects vendor libraries at the front of `sys.path` so they override any pip-installed versions.

### `App/src/reporting.py` - master research exports + condition comparison

**Existing:** `generate_master_outputs(source_dir)` builds source-folder summaries from reviewed `aoi_results/analysis.csv` files. Outputs to `<source_folder>/aoi_master/`:

- `master_task_metrics.csv` - one row per marked task per recording
- `master_recording_summary.csv` - one row per recording
- `master_learning_summary.csv` - difficulty x task aggregates with mean, median, SEM, 95% CI
- `master_analysis_workbook.xlsx` - Excel workbook with all summary sheets and basic charts

**New (Phase 7):** `compare_conditions(cond_a_dir, cond_b_dir)` — full NonGamified vs Gamified comparison returning a `ComparisonReport` with pre-aggregated metrics, Mann-Whitney U test results, transition matrices, and learning curve data.

### `App/tools/`

| File | Purpose |
|---|---|
| `check_tags.py` | Print detected tag IDs from a set of images |
| `check_images.py` | Print image dimensions |
| `generate_aoi_masks.py` | Helper for generating `aoi_masks.json` config |

---

## 7. CSV Output Schema

### Raw output - `aoi_results/raw/analysis.csv`

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
| `any_aoi_hit` | Backend helper: true if gaze is inside any AOI |
| `primary_aoi` | Name of the AOI hit (empty = NoAOI) |
| `gaze_on_aoi_x`, `gaze_on_aoi_y` | Gaze in surface-normalised coordinates |
| `{aoi}_hit` | Boolean per AOI - true if gaze is inside that AOI |
| `aoi_transition` | `PrevAOI->NewAOI` string when AOI changes |
| `time_in_current_aoi_ms` | Continuous time spent in current AOI |
| `{aoi}_visits` | Cumulative visit count for that AOI |
| `markers_detected` | Semicolon-separated tag IDs detected this frame |
| `primary_marker_count` | Number of markers visible for the primary AOI this frame |
| `primary_surface_initialized` | Whether the 3D surface model was initialized at this frame |
| `aoi_hit_source` | Detection path used: `surface`, `bbox`, `polygon`, `fallback_2d_Ntag` |

### Final output - `aoi_results/analysis.csv` (after Export in the app)

Adds columns to the raw CSV:

| Column | Description |
|---|---|
| `edit_source` | `auto`, `manual`, or `auto_gap_fill` |
| `final_primary_aoi` | Post-review result (may differ if researcher corrected it) |

### Fixation summary - `aoi_results/raw/fixation_summary.csv` *(Phase 7)*

| Column | Description |
|---|---|
| `fixation_id` | Fixation index matching `analysis.csv` |
| `start_frame` / `end_frame` | Frame range of the fixation |
| `start_time_s` / `end_time_s` | Time since recording start (seconds) |
| `duration_s` | Fixation duration in seconds |
| `dominant_aoi` | AOI with the most frame votes during this fixation |
| `centroid_x` / `centroid_y` | Mean gaze position (scene pixels) |

### Data quality - `aoi_results/raw/data_quality.json` *(Phase 7)*

```json
{
  "total_frames": 25430,
  "valid_gaze_frames": 24108,
  "missing_gaze_pct": 5.2,
  "recording_duration_s": 847.6,
  "fixation_count": 312,
  "fps": 30.0
}
```

### Task annotations - `aoi_results/tasks.json`

```json
{
  "Task 1":  {"start": 120,  "end": 890},
  "Task 2":  {"start": 910,  "end": 1650},
  "Task 10": {"start": 9800, "end": 10540}
}
```

Start/end are scene camera frame indices. T1–T10 are 10 repetitions of the same assembly task.

---

## 8. Running the Pipeline

### Option A - single app entry point (recommended)

```bash
App/START_APP.bat
```

1. Select condition folder (NonGamified / Gamified) and pick a recording
2. Click **Load Recording**
3. Click **Analyse** and wait for the progress bar; the data quality indicator appears when done
4. Watch the video and mark task repetitions T1–T10 using the task panel (I = start, O = end, click row to select task)
5. Click **💾 Save Tasks**
6. Click **⬇ Export Final CSV**
7. Repeat for all recordings
8. Switch to **Comparison** tab, select NonGamified vs Gamified, click **Run Comparison**

### Option B - command line analyzer (single recording)

```bash
cd App/
venv/Scripts/python.exe src/analyzer.py
venv/Scripts/python.exe src/analyzer.py --force   # re-run even if results exist
```

### Environment setup (if venv is missing or broken)

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r App\requirements-qt.txt
```

Dependencies include `PySide6`, `opencv-python`, `numpy`, `pandas`, `openpyxl`, `plotly`, `kaleido`, `scipy`, and the vendored Pupil Labs libraries.

---

## 9. Open Issues and Remaining Work

### ✅ RESOLVED - Re-run required for new output files

**Status**: Fixed in Phase 8 with "Batch Re-analyse All" button.

**Solution**: Click "Batch Re-analyse All" in Studio tab to regenerate `fixation_summary.csv` and `data_quality.json` for all recordings in selected condition folder.

### ✅ RESOLVED - Box AOI initialization marker threshold

**Status**: Fixed in Phase 8.

**Previous**: Required 4/4 markers → 100% NoAOI if markers persistently occluded.

**Current** (`App/src/analyzer.py` lines 611-618):
```python
# Screen allows 3/4 markers, boxes allow 2/4
if aoi.name == "Screen" and len(aoi.marker_ids) == 4:
    n_required_init = 3
elif len(aoi.marker_ids) == 4:
    n_required_init = 2  # Boxes now only need 2/4 markers
```

Box AOIs (Left_Box, Middle_Box, Right_Box) now initialize with only 2/4 visible markers.

### ✅ RESOLVED - Fallback polygon scale

**Status**: Increased in Phase 8.

**Previous**: `FALLBACK_POLYGON_SCALE = 1.08` (8% expansion)
**Current**: `FALLBACK_POLYGON_SCALE = 1.15` (15% expansion)

Edge gazes now more reliably classified.

### 🔄 PENDING - Screen AOI detection verification

**Status**: Fixed in Phase 8, **requires user testing**.

**Expected**: Screen detection should improve from 0.02% to 30-40%.

**User must verify**:
1. Re-run analysis on 38k-frame Gamified recording
2. Check console log for `[Screen] Detection: X/38420 frames (X.X%)`
3. Verify `analysis.csv` shows Screen_hit > 20%
4. Confirm Dashboard heatmap displays Screen data

If still <10% detection → further diagnostic needed (marker size, placement, lighting).

### 🔄 PENDING - Gamified recordings collection

**Status**: No code changes needed.

The Gamified condition folder is currently empty. The Comparison tab handles this gracefully (condition B: 0 recordings) but statistical comparisons are unavailable until data is collected.

### 🔄 PENDING - Dashboard task filter bug

**Status**: Not yet fixed.

**Problem**: When selecting Task 1/2/3 in Dashboard tab:
- ✅ Learning curve updates correctly (shows single point)
- ❌ Other charts don't update (dwell bars, heatmap, fixation charts)

**Expected**: All charts should filter to show only selected task's data.

**Priority**: Medium (doesn't block research, but affects analysis workflow).

### 🔄 PENDING - UI improvements (user-requested)

**Status**: Not yet implemented. Requires major refactoring (1-2 days).

**Requested changes**:
1. **Collapsible sections**: Source, Trim, AOI Correction, Tasks should have dropdown toggles
2. **Real-time Trim**: Delete frames when user presses Enter + Undo button
3. **Dynamic button text**: Scale/wrap text when left panel is resized
4. **Dynamic folder detection**: Remove hardcoded "Gamified"/"NonGamified", auto-detect any subfolders
5. **AOI Correction undo**: Add undo functionality

**Priority**: Low (quality-of-life improvements, not research blockers).

### Statistical power

With small expected sample sizes (n < 10 per condition initially), all Mann-Whitney U results should be treated as exploratory. Report effect sizes (rank-biserial r) alongside p-values. Consider whether sample size will be sufficient to detect the expected effect before concluding the data collection phase.
