"""
Phase 2 — Full AOI Analysis + Annotated Video
==============================================
Processes every recording folder (YYYY-MM-DD-HH-MM-SS) and writes to
<recording>/aoi_results/:

  analysis.csv          — comprehensive per-frame data + summary + transition matrix
  validation_video.mp4  — scene video with gaze, fixations, scanpath, AOI boxes

Geometry Strategy (Updated):
  - Uses Pupil Labs 3D Marker Mapper (pl-marker-mapper).
  - Initializes a 3D physical surface model once the full marker set is visible.
  - Uses partial-marker 2D fallback polygons before full initialization is ready.
  - Applies 10% polygon dilation to capture gaze on the edges of surfaces.
  - Supports optional surface-space AOI masks from App/config/aoi_masks.json.
  - Performs hit-testing using undistorted 3D -> 2D pixel projections.
"""

from __future__ import annotations

import collections
import csv
import json
import logging
import os
import pathlib
import re
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pupil_apriltags

# `frame.gray` uses the fast Y-plane path (range 16-235); the library warns that the
# full-range alternative is 4x slower — we deliberately keep the fast path. Silence it.
warnings.filterwarnings("ignore", message=".*yuv420p.*")

# Tag detection is the bottleneck (~40 ms/frame). Speed comes from a small thread
# pool (pupil_apriltags releases the GIL, so N detectors run concurrently across
# cores, ~2.5x). DETECT_STRIDE=1 detects EVERY frame (no accuracy loss); raise it to
# skip frames and reuse the last detections if you want more speed for less accuracy.
DETECT_STRIDE = 1
_N_DET = max(1, (os.cpu_count() or 4) // 4)
_DETECT_CHUNK = 300

from paths import CONFIG_DIR, RECORDINGS_DIR, ensure_vendor_paths

ensure_vendor_paths()

import pupil_labs.neon_recording as nr
from pupil_labs.camera import Camera, perspective_transform
from pupil_labs.marker_mapper.aoi import AOI
import pupil_labs.marker_mapper.surface as surface

from masks import AoiMaskConfig, AoiRegion, load_aoi_mask_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AOI_CONFIG: dict[str, list[int]] = {
    "Board":       [0, 1, 2, 3],
    "Left_Box":    [8, 9, 10, 11],
    "Stream_Deck": [6, 7],
    "Middle_Box":  [16, 17, 18, 19],
    "Right_Box":   [20, 21, 22, 23],
    "Screen":      [12, 13, 14, 15],
}

# (u_min, u_max, v_min, v_max) inside the normalized Screen polygon.
# u: left -> right, v: top -> bottom.
# Points_Bar / Progress_Bar / Avatar were removed — they are not the current
# gamification elements. Real screen task/gamification zones live in
# App/config/screen_zones.json and are wired in as part of the surface redesign.
SUB_AOIS_PROPORTIONS: dict[str, tuple[float, float, float, float]] = {}

AOI_COLORS: dict[str, tuple[int, int, int]] = {
    "Board":       (  0, 200, 255),   # orange
    "Left_Box":    (  0, 255,   0),   # green
    "Stream_Deck": (255,   0, 255),   # magenta
    "Middle_Box":  (  0, 165, 255),   # amber
    "Right_Box":   (255, 255,   0),   # cyan
    "Screen":      (255,  80,  80),   # blue
}
DEFAULT_COLOR = (180, 180, 180)

GENERATE_VIDEO       = True
VIDEO_SCALE          = 0.5    # 0.5 = 800×600 for Neon's 1600×1200
SCANPATH_HISTORY     = 5
FALLBACK_POLYGON_SCALE = 1.08

# ═══════════════════════════════════════════════════════════════════════════════

RECORDING_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def find_recording_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and RECORDING_PATTERN.match(p.name))


def make_detector(decimate: float = 1.0) -> pupil_apriltags.Detector:
    return pupil_apriltags.Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.5,
        debug=0,
    )


def make_screen_detector() -> pupil_apriltags.Detector:
    """Aggressive detector for the calibration pass. Combined with 2x upscaling it
    recovers the small, grazing-angle SCREEN corner tags (12/13/14/15) that the
    standard 1x detector misses — measured on 96ARP to go from 1 screen tag to 3,
    which is what lets the screen triangulate. Slower, so only used in calibration."""
    return pupil_apriltags.Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=1.5,
        debug=0,
    )


_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
def enhance_frame(gray: np.ndarray, clahe=None) -> np.ndarray:
    """Normalise the frame so AprilTags decode reliably in all lighting/blur.
    The rig and lighting are physically constant across recordings, so this is
    tuned once: CLAHE evens out brightness/contrast, then a mild unsharp mask
    counters motion blur. Applied identically in calibration and the main pass.
    Pass a per-thread `clahe` when calling from worker threads (CLAHE is stateful)."""
    eq = (clahe or _clahe).apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 1.0)
    return cv2.addWeighted(eq, 1.5, blur, -0.5, 0)


_tls = threading.local()
def _thread_detector():
    """Per-thread detector + CLAHE. AprilTag/CLAHE objects are stateful and NOT
    thread-safe, so every worker thread must have its own (sharing one causes
    native crashes under concurrency)."""
    if not hasattr(_tls, "det"):
        _tls.det = make_detector(1.0)
        _tls.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return _tls.det, _tls.clahe


def _detect_gray(gray):
    det, clahe = _thread_detector()
    return det.detect(enhance_frame(gray, clahe))


def _calib_detect(gray):
    """High-recall detection for the CALIBRATION pass only. Runs two passes and
    unions them so it never detects fewer tags than the main pass, but adds the
    rare screen tags:
      1. standard 1x enhanced (solid for the well-seen Board/Box/Deck tags),
      2. 2x-upscaled RAW + aggressive detector (recovers the tiny screen tags —
         CLAHE/unsharp actually hurts those, so this pass uses raw gray).
    2x corners are preferred where a tag appears in both (higher precision).
    Returns {tag_id: 4x2 corners} in original-image pixels."""
    det, clahe = _thread_detector()
    if not hasattr(_tls, "sdet"):
        _tls.sdet = make_screen_detector()
    out = {int(d.tag_id): d.corners.astype(np.float64)
           for d in det.detect(enhance_frame(gray, clahe))}
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for d in _tls.sdet.detect(up):
        out[int(d.tag_id)] = d.corners.astype(np.float64) / 2.0
    return out


def _stream_detections(frames, gaze_samps):
    """Yield (frame_idx, frame, gaze, detections) for the main pass.

    Detects every DETECT_STRIDE-th frame (reusing the last detections in between,
    since the rig is static) and runs those detections concurrently on a thread
    pool — each worker thread uses its OWN detector. Gaze is delivered per frame."""
    pool = ThreadPoolExecutor(max_workers=_N_DET)
    last_det: list = []
    buf: list = []

    def flush():
        nonlocal last_det
        det_idx = [k for k in range(len(buf)) if buf[k][0] % DETECT_STRIDE == 0]
        detmap = dict(zip(det_idx, pool.map(_detect_gray, [buf[k][1].gray for k in det_idx])))
        for k in range(len(buf)):
            if k in detmap:
                last_det = detmap[k]
            yield buf[k][0], buf[k][1], buf[k][2], last_det

    try:
        for fi, (frame, gaze) in enumerate(zip(frames, gaze_samps, strict=False)):
            buf.append((fi, frame, gaze))
            if len(buf) >= _DETECT_CHUNK:
                yield from flush()
                buf = []
        if buf:
            yield from flush()
    finally:
        pool.shutdown(wait=True)


# ── Geometry & Hit-Test Helpers ───────────────────────────────────────────────

def make_camera(recording: nr.NeonRecording) -> Camera:
    cal = recording.calibration
    return Camera(1600, 1200,
                  cal.scene_camera_matrix,
                  cal.scene_distortion_coefficients)


def sample_gaze(recording: nr.NeonRecording, scene_ts):
    """Gaze aligned to each scene timestamp.

    The library's ``recording.gaze.sample()`` uses ``pandas.merge_asof``, which
    raises "right keys must be sorted" when a recording's gaze timestamps are not
    monotonic (a data glitch present in some recordings). Fall back to sorting the
    gaze stream and taking the nearest sample per scene frame so analysis still runs.
    """
    gd = np.asarray(recording.gaze.data)
    gts = gd["time"]
    if len(gts) > 1 and np.any(np.diff(gts) < 0):
        log.warning("  Gaze timestamps not monotonic — using sorted nearest-sample fallback.")
        gd = gd[np.argsort(gts, kind="stable")]
        gts = gd["time"]
        st = np.asarray(scene_ts)
        idx = np.clip(np.searchsorted(gts, st), 0, len(gts) - 1)
        left = np.clip(idx - 1, 0, len(gts) - 1)
        choose_left = np.abs(gts[left] - st) <= np.abs(gts[idx] - st)
        return gd[np.where(choose_left, left, idx)].view(np.recarray)
    return recording.gaze.sample(scene_ts)

def get_expanded_surface_boundary(s2i: np.ndarray, camera: Camera, scale: float = 1.10, n: int = 10) -> np.ndarray:
    """Gets the 2D pixel boundary of the surface, expanded outward by 'scale' to capture edge-gaze."""
    norm_boundary = surface.normalized_boundary_points(n)
    centroid = np.array([0.5, 0.5])
    expanded_norm = centroid + (norm_boundary - centroid) * scale
    undist = perspective_transform(expanded_norm, s2i)
    return camera.distort_points(undist).astype(np.float32)

def get_sub_aoi_polygon(s2i: np.ndarray, camera: Camera, u_min: float, u_max: float, v_min: float, v_max: float) -> np.ndarray:
    norm_pts = np.array([
        [u_min, v_min],
        [u_max, v_min],
        [u_max, v_max],
        [u_min, v_max]
    ], dtype=np.float32)
    undist_pts = perspective_transform(norm_pts, s2i)
    return camera.distort_points(undist_pts).astype(np.float32)

def _contains_gaze_polygon(corners_px: np.ndarray, gx: float, gy: float) -> bool:
    pts = corners_px.reshape(-1, 1, 2).astype(np.float32)
    return cv2.pointPolygonTest(pts, (float(gx), float(gy)), False) >= 0


def normalize_quad_corners(poly: np.ndarray) -> np.ndarray:
    """Return quad corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        return pts
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmax(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmin(diffs)],
        ],
        dtype=np.float32,
    )


def _scale_polygon(poly: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or scale == 1.0:
        return pts
    center = pts.mean(axis=0)
    return center + (pts - center) * scale



class OpticalFlowTracker:
    """Tracks a 2D polygon using Lucas-Kanade optical flow."""
    def __init__(self, max_points: int = 100, quality: float = 0.05, min_dist: float = 5.0):
        self.max_points = max_points
        self.quality = quality
        self.min_dist = min_dist
        self.prev_gray = None
        self.tracked_pts = None
        self.prev_poly = None
        import cv2
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
    def reset(self):
        self.prev_gray = None
        self.tracked_pts = None
        self.prev_poly = None
        
    def initialize(self, gray, poly):
        import numpy as np
        import cv2
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_points, qualityLevel=self.quality, minDistance=self.min_dist, mask=mask)
        if pts is not None and len(pts) >= 4:
            self.prev_gray = gray.copy()
            self.tracked_pts = pts
            self.prev_poly = poly.copy()
            return True
        return False
        
    def track(self, gray):
        import numpy as np
        import cv2
        if self.prev_gray is None or self.tracked_pts is None or len(self.tracked_pts) < 4: return None
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.tracked_pts, None, **self.lk_params)
        good_new = new_pts[status == 1]
        good_old = self.tracked_pts[status == 1]
        if len(good_new) < 4:
            self.reset(); return None
        H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
        if H is None:
            self.reset(); return None
        poly_reshaped = self.prev_poly.reshape(-1, 1, 2).astype(np.float32)
        new_poly = cv2.perspectiveTransform(poly_reshaped, H).reshape(-1, 2)
        self.prev_gray = gray.copy()
        self.tracked_pts = good_new.reshape(-1, 1, 2)
        self.prev_poly = new_poly
        return new_poly


def get_fallback_aoi_polygon(detections: list, aoi_ids: list[int]) -> np.ndarray | None:
    """Build a 2D polygon from visible markers when 3D surface mapping is not ready."""
    found_dets = [d for d in detections if d.tag_id in aoi_ids]
    found = {d.tag_id: np.asarray(d.center, dtype=np.float32) for d in found_dets}
    n_found = len(found)
    if n_found == 0:
        return None

    if len(aoi_ids) == 4:
        if n_found >= 3:
            pts = np.array(list(found.values()), dtype=np.float32).reshape(-1, 2)
            if n_found == 3:
                pairs = [(0, 1), (0, 2), (1, 2)]
                i, j = max(pairs, key=lambda pair: float(np.linalg.norm(pts[pair[0]] - pts[pair[1]])))
                k = ({0, 1, 2} - {i, j}).pop()
                pts = np.vstack([pts, pts[i] + pts[j] - pts[k]])
            return _scale_polygon(normalize_quad_corners(pts), FALLBACK_POLYGON_SCALE)

        if n_found == 2:
            all_corners = []
            for det in found_dets:
                all_corners.extend(det.corners)
            hull = cv2.convexHull(np.array(all_corners, dtype=np.float32)).reshape(-1, 2)
            return _scale_polygon(hull, FALLBACK_POLYGON_SCALE)

    if len(aoi_ids) == 2 and n_found >= 2:
        all_corners = []
        for det in found_dets:
            all_corners.extend(det.corners)
        hull = cv2.convexHull(np.array(all_corners, dtype=np.float32)).reshape(-1, 2)
        return _scale_polygon(hull, FALLBACK_POLYGON_SCALE)

    return None


def _surface_xy_from_image_quad(poly: np.ndarray, gx: float, gy: float) -> np.ndarray | None:
    pts = normalize_quad_corners(poly)
    if len(pts) != 4 or not (np.isfinite(gx) and np.isfinite(gy)):
        return None
    transform = cv2.getPerspectiveTransform(pts.astype(np.float32), surface.normalized_corners())
    mapped = cv2.perspectiveTransform(np.array([[[gx, gy]]], dtype=np.float32), transform)
    return mapped.reshape(2).astype(np.float64)


def _project_norm_points_to_image_quad(norm_points: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    pts = normalize_quad_corners(poly)
    if len(pts) != 4:
        return None
    transform = cv2.getPerspectiveTransform(surface.normalized_corners(), pts.astype(np.float32))
    projected = cv2.perspectiveTransform(norm_points.reshape(-1, 1, 2).astype(np.float32), transform)
    return projected.reshape(-1, 2).astype(np.float32)


def _project_region_outline(region: AoiRegion, s2i: np.ndarray, camera: Camera) -> np.ndarray | None:
    norm_outline = region.normalized_outline()
    if norm_outline is None or len(norm_outline) < 3:
        return None
    undist = perspective_transform(norm_outline, s2i)
    return camera.distort_points(undist).astype(np.float32)


def _project_region_outline_2d(region: AoiRegion, surface_poly: np.ndarray) -> np.ndarray | None:
    norm_outline = region.normalized_outline()
    if norm_outline is None or len(norm_outline) < 3:
        return None
    return _project_norm_points_to_image_quad(norm_outline, surface_poly)


def _surface_gaze(
    aoi: AOI,
    gx: float,
    gy: float,
    camera: Camera,
    img2surface: np.ndarray,
) -> np.ndarray | None:
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return None
    try:
        return aoi.map_gaze(np.array([gx, gy], dtype=np.float32), camera, img2surface)
    except Exception:
        return None


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_polygon_box(img, corners_px: np.ndarray, name: str, color: tuple, scale: float = 1.0) -> None:
    pts = (corners_px * scale).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    corners_2d = corners_px.reshape(-1, 2)
    cx = int(corners_2d[:, 0].mean() * scale)
    cy = int(corners_2d[:, 1].mean() * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(name, font, 0.55, 2)
    cv2.rectangle(img, (cx - 4, cy - th - 6), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
    cv2.putText(img, name, (cx, cy), font, 0.55, color, 2, cv2.LINE_AA)

def _draw_scanpath(img, points: list, scale: float = 1.0):
    if len(points) < 2: return
    n = len(points)
    for i in range(1, n):
        alpha = i / (n - 1)
        brightness = int(80 + 175 * alpha)
        color = (brightness, brightness, brightness)
        thickness = max(1, int(2 * alpha))
        p1 = (int(points[i - 1][0] * scale), int(points[i - 1][1] * scale))
        p2 = (int(points[i][0] * scale),     int(points[i][1] * scale))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def _draw_fixation_circle(img, fx, fy, dur_ms, scale: float = 1.0):
    radius = max(12, int(np.sqrt(max(dur_ms, 1)) * 1.8))
    overlay = img.copy()
    cv2.circle(overlay, (int(fx * scale), int(fy * scale)), radius, (255, 220, 50), -1)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
    cv2.circle(img, (int(fx * scale), int(fy * scale)), radius, (255, 220, 50), 2, cv2.LINE_AA)

def _draw_gaze_dot(img, gx, gy, scale: float = 1.0):
    if not (np.isfinite(gx) and np.isfinite(gy)): return
    pt = (int(gx * scale), int(gy * scale))
    cv2.circle(img, pt, 6, (0, 0, 200), -1, cv2.LINE_AA)
    cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

def _draw_banner(img, text, color, h, w):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)

def _draw_frame_info(img, frame_idx, time_s):
    label = f"Frame {frame_idx:05d}   {time_s:7.2f} s"
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

# ── Fixation loader ───────────────────────────────────────────────────────────

def load_fixations(recording: nr.NeonRecording):
    try:
        fd      = recording.fixations.data
        starts  = np.asarray(fd["start_time"],  dtype=np.int64)
        stops   = np.asarray(fd["stop_time"],   dtype=np.int64)
        mean_x  = np.asarray(fd["mean_gaze_x"], dtype=np.float64)
        mean_y  = np.asarray(fd["mean_gaze_y"], dtype=np.float64)
        log.info("  Loaded %d fixations", len(starts))
        return starts, stops, mean_x, mean_y
    except Exception as exc:
        log.warning("  Could not load fixations: %s", exc)
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([]), np.array([])


def write_summaries(out_dir: pathlib.Path, csv_path: pathlib.Path, rec_name: str, fps: float) -> None:
    """Regenerate the per-recording summary files the dashboard reads:
    fixation_summary.csv (one row per fixation), data_quality.json and
    preprocessing_report.json. Rebuilt from analysis.csv after the main pass.
    (These were lost when the analyzer was refactored, so the Fixation and
    Data-Quality dashboard tabs read nothing.)"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return
    lbl = df["primary_aoi"].fillna("NoAOI").astype(str).replace("", "NoAOI")
    dur = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if n > 1 else 0.0
    n_fix = int(df["fixation_id"].dropna().nunique())
    valid = int(df["gaze_x_px"].notna().sum())

    # ── fixation_summary.csv (one row per fixation) ──
    cols = ["fixation_id", "start_frame", "end_frame", "start_time_s",
            "end_time_s", "duration_s", "dominant_aoi", "centroid_x", "centroid_y"]
    rows = []
    for fid, g in df.dropna(subset=["fixation_id"]).groupby("fixation_id"):
        aois = g["primary_aoi"].fillna("NoAOI").astype(str).replace("", "NoAOI")
        modes = aois.mode()
        dur_s = (float(g["fixation_dur_ms"].iloc[0]) / 1000.0
                 if "fixation_dur_ms" in g and pd.notna(g["fixation_dur_ms"].iloc[0])
                 else float(g["time_s"].max() - g["time_s"].min()))
        rows.append({
            "fixation_id": int(fid),
            "start_frame": int(g["frame_idx"].min()),
            "end_frame": int(g["frame_idx"].max()),
            "start_time_s": round(float(g["time_s"].min()), 4),
            "end_time_s": round(float(g["time_s"].max()), 4),
            "duration_s": round(dur_s, 4),
            "dominant_aoi": modes.iloc[0] if len(modes) else "NoAOI",
            "centroid_x": round(float(g["gaze_x_px"].mean()), 2),
            "centroid_y": round(float(g["gaze_y_px"].mean()), 2),
        })
    fdf = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    fdf.to_csv(out_dir / "fixation_summary.csv", index=False, encoding="utf-8")

    # ── data_quality.json ──
    (out_dir / "data_quality.json").write_text(json.dumps({
        "recording": rec_name,
        "total_frames": n,
        "valid_gaze_frames": valid,
        "missing_gaze_pct": round(100.0 * (n - valid) / n, 2),
        "recording_duration_s": round(dur, 2),
        "fixation_count": n_fix,
        "fps": round(fps, 1),
    }, indent=2), encoding="utf-8")

    # ── preprocessing_report.json (dwell / coverage / detection method) ──
    aoi_names = [c[:-4] for c in df.columns if c.endswith("_hit")]
    dwell = {a: int((lbl == a).sum()) for a in aoi_names}
    detected = int((lbl != "NoAOI").sum())
    # Detection-method split for the Data Quality breakdown: DIRECT = surface formed
    # from >=2 of its own tags; PROJECTED = from the rigid body.
    src = (df["aoi_hit_source"].fillna("").astype(str)
           if "aoi_hit_source" in df.columns else pd.Series([""] * n))
    direct = int((src == "direct").sum())
    projected = int((src == "projected").sum())
    (out_dir / "preprocessing_report.json").write_text(json.dumps({
        "recording": rec_name,
        "total_frames": n,
        "aoi_detected_frames": detected,
        "no_aoi_frames": n - detected,
        "no_aoi_pct": round(100.0 * (n - detected) / n, 2),
        "surface_3d_frames": direct,
        "surface_3d_pct": round(100.0 * direct / n, 2),
        "fallback_2d_frames": projected,
        "fallback_2d_pct": round(100.0 * projected / n, 2),
        "fixation_count": n_fix,
        "fps": round(fps, 1),
        "recording_duration_s": round(dur, 2),
        "aoi_dwell_frames": dwell,
        "aoi_dwell_pct": {a: round(100.0 * v / n, 2) for a, v in dwell.items()},
    }, indent=2), encoding="utf-8")

# ── Core analysis ─────────────────────────────────────────────────────────────

def analyze_recording(
    recording_dir: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    *,
    clear_output: bool = True,
    generate_video: bool = False,
    fast_mode: bool = False,
    trim_range: tuple[int, int] | None = None,
    trim_padding_s: float = 0.5,
) -> None:
    log.info("=" * 62)
    log.info("Recording : %s", recording_dir.name)
    log.info("=" * 62)

    recording = nr.load(str(recording_dir))
    camera    = make_camera(recording)
    
    aois = [AOI(name, ids) for name, ids in AOI_CONFIG.items()]
    try:
        mask_config: AoiMaskConfig = load_aoi_mask_config(CONFIG_DIR, SUB_AOIS_PROPORTIONS)
    except Exception as exc:
        log.warning("  Could not load aoi_masks.json (%s). Falling back to built-in Screen boxes.", exc)
        mask_config = AoiMaskConfig([
            AoiRegion(
                name=name,
                surface="Screen",
                kind="bbox",
                priority=10,
                bounds=(float(u1), float(u2), float(v1), float(v2)),
            )
            for name, (u1, u2, v1, v2) in SUB_AOIS_PROPORTIONS.items()
        ])

    # Base AOIs + optional surface-space mask AOIs.
    aoi_names = list(dict.fromkeys(list(AOI_CONFIG.keys()) + mask_config.names))
    if mask_config.has_custom_config:
        log.info("  AOI masks: %s", mask_config.config_path)
    elif mask_config.names:
        log.info("  AOI masks: using built-in Screen bbox defaults (%s)", ", ".join(mask_config.names))

    scene_ts   = recording.scene.time
    frames     = recording.scene.sample(scene_ts)
    gaze_samps = sample_gaze(recording, scene_ts)

    global_start_idx = 0
    if trim_range is not None:
        fps_est = max(1.0, (len(scene_ts) - 1) / ((scene_ts[-1] - scene_ts[0]) / 1e9)) if len(scene_ts) > 1 else 30.0
        pad_frames = int(round(trim_padding_s * fps_est))
        s_idx = max(0, trim_range[0] - pad_frames)
        e_idx = min(len(scene_ts), trim_range[1] + pad_frames + 1)
        scene_ts = scene_ts[s_idx:e_idx]
        frames = list(frames)[s_idx:e_idx]
        gaze_samps = list(gaze_samps)[s_idx:e_idx]
        global_start_idx = s_idx
        log.info("  Trimmed analysis to frames %d-%d (padded)", s_idx, e_idx - 1)

    total = len(scene_ts)

    fix_starts, fix_stops, fix_mean_x, fix_mean_y = load_fixations(recording)
    n_fixations = len(fix_starts)

    fps = max(1.0, (total - 1) / ((scene_ts[-1] - scene_ts[0]) / 1e9)) if total > 1 else 30.0
    frame_dur_ms = 1000.0 / fps

    # ── Scene-wide rigid-body calibration (pre-pass) ──────────────────────────
    # Detect tags on a subsample and build ONE rigid model of the whole rig, so
    # every surface (incl. the rarely-seen Screen, placed by triangulation) can
    # be localised from any visible tag during the main pass.
    import rigid_surface
    _cal = recording.calibration
    camera_K = np.array(_cal.scene_camera_matrix, dtype=np.float64)
    camera_D = np.array(_cal.scene_distortion_coefficients, dtype=np.float64).reshape(-1)
    # Dense calibration sample (~2000 frames) + high-recall detection (2x upscale)
    # so the rarely-decoded screen tags are caught often enough to triangulate the
    # screen. Proven on 96ARP: screen goes NO->yes. This only runs once (calibration);
    # the main pass keeps the fast 1x detection, so the long pass is not slowed.
    calib_stride = max(1, total // 2000)
    calib_positions = list(range(0, total, calib_stride))
    calib_ts = np.array([int(scene_ts[i]) for i in calib_positions])
    log.info("  Calibrating scene rigid body from %d sampled frames (high-recall)...",
             len(calib_positions))
    _cgrays = [cf.gray for cf in recording.scene.sample(calib_ts)]
    with ThreadPoolExecutor(max_workers=_N_DET) as _cpool:
        calib_dets = list(_cpool.map(_calib_detect, _cgrays))
    del _cgrays
    scene_model = rigid_surface.calibrate_scene(calib_dets, AOI_CONFIG, camera_K, camera_D, log=log)

    def _quad_area(q):
        return 0.5 * float(np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[1])))
    # Smaller surfaces win over the large Board behind them when quads overlap.
    surface_priority = {
        s: r for r, (s, _a) in enumerate(sorted(
            ((s, _quad_area(q)) for s, q in scene_model.surface_quad_world.items()),
            key=lambda kv: kv[1]))
    }

    # Fresh frame iterator for the main pass (calibration consumed its own).
    if trim_range is None:
        frames = recording.scene.sample(scene_ts)

    log.info("  Frames: %d  |  FPS: %.1f  |  Duration: %.1f s  |  Fixations: %d", total, fps, total / fps, n_fixations)

    out_dir = output_dir or recording_dir / "aoi_results" / "raw"
    if clear_output and out_dir.exists():
        try:
            shutil.rmtree(out_dir)
            log.info("  Cleared old output folder: %s", out_dir)
        except PermissionError:
            log.warning("  Could not delete old output folder — files may be open. Overwriting in place.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the rig so AOI corrections can re-project gaze onto a new surface
    # (updating the heatmap) without re-running the whole calibration.
    try:
        rigid_surface.save_scene_model(scene_model, out_dir / "scene_model.pkl")
    except Exception as _e:
        log.warning("  Could not save scene model: %s", _e)

    csv_path   = out_dir / "analysis.csv"
    video_path = out_dir / "validation_video.mp4"
    processing_path = out_dir / ".processing"
    progress_path = out_dir / "progress.json"

    def write_progress(percent: float, frame_idx: int, status: str) -> None:
        payload = {
            "recording": recording_dir.name,
            "percent": round(max(0.0, min(100.0, percent)), 1),
            "frame_idx": int(frame_idx),
            "total_frames": int(total),
            "status": status,
        }
        progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    processing_path.write_text("processing", encoding="utf-8")
    write_progress(0.0, 0, "starting")

    vid_w = int((recording.scene.width  or 1600) * VIDEO_SCALE)
    vid_h = int((recording.scene.height or 1200) * VIDEO_SCALE)
    video_writer = None
    if generate_video:
        fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (vid_w, vid_h))
        log.info("  Video: %d\u00d7%d @ %.1f fps -> %s", vid_w, vid_h, fps, video_path.name)

    dwell_frames    = {n: 0    for n in aoi_names}
    visit_count     = {n: 0    for n in aoi_names}
    fix_count_aoi   = {n: 0    for n in aoi_names}
    fix_dur_sum_aoi = {n: 0.0  for n in aoi_names}
    first_visit_s   = {n: None for n in aoi_names}
    last_visit_s    = {n: None for n in aoi_names}
    sum_gaze_x_aoi  = {n: 0.0  for n in aoi_names}
    sum_gaze_y_aoi  = {n: 0.0  for n in aoi_names}

    all_labels = aoi_names + ["NoAOI"]
    transitions: dict[str, dict[str, int]] = {src: {dst: 0 for dst in all_labels} for src in all_labels}

    prev_aoi        = ""
    aoi_entry_frame = 0
    cum_visits      = {n: 0 for n in aoi_names}
    time_in_aoi_ms  = 0.0

    fix_ptr     = 0
    last_fix_id = -1
    scanpath: collections.deque = collections.deque(maxlen=SCANPATH_HISTORY)
    fix_counted_for_aoi: set    = set()

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "timestamp_ns", "time_s", "frame_idx", "gaze_x_px", "gaze_y_px",
                "is_fixation", "fixation_id", "fixation_dur_ms", "fixation_gaze_x", "fixation_gaze_y",
                "any_aoi_hit", "primary_aoi", "gaze_on_aoi_x", "gaze_on_aoi_y",
                "primary_surface", "aoi_hit_source", "primary_marker_count", "primary_surface_initialized",
            ] + [f"{n}_hit" for n in aoi_names] + ["aoi_transition", "time_in_current_aoi_ms"]
              + [f"{n}_visits" for n in aoi_names] + ["markers_detected"])

            for frame_idx, frame, gaze, detections in _stream_detections(frames, gaze_samps):
                global_frame = global_start_idx + frame_idx
                if frame_idx % 30 == 0:
                    percent = frame_idx / total * 100 if total else 0.0
                    print(f"[{percent:5.1f}%] frame {global_frame} (local {frame_idx}/{total})")
                    write_progress(percent, global_frame, "processing")

                ts     = int(gaze.time)
                time_s = (ts - int(scene_ts[0])) / 1e9
                gx     = float(gaze.point_x)
                gy     = float(gaze.point_y)

                while fix_ptr < n_fixations and fix_stops[fix_ptr] < ts:
                    fix_ptr += 1

                is_fix  = False
                fix_id  = -1
                fix_dur = 0.0
                fix_cx  = float("nan")
                fix_cy  = float("nan")

                if fix_ptr < n_fixations and fix_starts[fix_ptr] <= ts:
                    is_fix  = True
                    fix_id  = fix_ptr
                    fix_dur = (fix_stops[fix_ptr] - fix_starts[fix_ptr]) / 1e6
                    fix_cx  = float(fix_mean_x[fix_ptr])
                    fix_cy  = float(fix_mean_y[fix_ptr])
                    if fix_id != last_fix_id:
                        scanpath.append((fix_cx, fix_cy))
                        last_fix_id = fix_id

                # `detections` is provided by _stream_detections (strided + threaded).
                detected_ids  = [d.tag_id for d in detections]

                row_hits        = {n: False for n in aoi_names}
                active_polygons = {}
                primary_aoi     = ""
                primary_surface = ""
                hit_source      = ""
                gaze_on_aoi_x   = ""
                gaze_on_aoi_y   = ""
                primary_marker_count = ""
                primary_surface_initialized = ""

                # 1. Evaluate AOIs. Each surface forms its quad DIRECTLY from its own
                # visible tags when present (accurate — this is how the screen makes a
                # proper rectangle when you look at it); otherwise it falls back to the
                # rigid-body projection from a well-constrained whole-rig camera pose.
                loc = scene_model.localize(detections)
                good_pose = loc is not None and loc[2] <= rigid_surface.CAMERA_MAX_REPROJ_PX
                cam_rvec = loc[0] if good_pose else None
                cam_tvec = loc[1] if good_pose else None
                if True:
                    best_rank = None
                    for aoi_name, aoi_ids in AOI_CONFIG.items():
                        quad = scene_model.surface_quad(aoi_name, detections, cam_rvec, cam_tvec)
                        if quad is None:
                            continue
                        active_polygons[aoi_name] = quad
                        uv = scene_model.gaze_to_surface(
                            aoi_name, gx, gy, cam_rvec, cam_tvec, image_quad=quad)
                        if uv is None:
                            continue
                        u, v = uv
                        # inside the surface (5% edge margin catches border gaze)
                        if -0.05 <= u <= 1.05 and -0.05 <= v <= 1.05:
                            row_hits[aoi_name] = True
                            rank = surface_priority.get(aoi_name, 99)
                            if best_rank is None or rank < best_rank:
                                best_rank = rank
                                primary_aoi = aoi_name
                                primary_surface = aoi_name
                                own_tags = sum(1 for d in detections if d.tag_id in aoi_ids)
                                # DIRECT = surface formed from >=2 of its own tags (matches
                                # surface_quad's gate); PROJECTED = from the rigid body.
                                hit_source = "direct" if own_tags >= 2 else "projected"
                                gaze_on_aoi_x = f"{min(max(u, 0.0), 1.0):.5f}"
                                gaze_on_aoi_y = f"{min(max(v, 0.0), 1.0):.5f}"
                                primary_marker_count = str(own_tags)
                                primary_surface_initialized = "True"

                # 4. Handle Logging
                if primary_aoi:
                    dwell_frames[primary_aoi]   += 1
                    sum_gaze_x_aoi[primary_aoi] += gx
                    sum_gaze_y_aoi[primary_aoi] += gy
                    if first_visit_s[primary_aoi] is None:
                        first_visit_s[primary_aoi] = time_s
                    last_visit_s[primary_aoi] = time_s
                    
                    # Also count the parent surface when a configured mask/sub-AOI wins.
                    if primary_surface and primary_surface != primary_aoi and primary_surface in aoi_names:
                        row_hits[primary_surface] = True
                        dwell_frames[primary_surface] += 1
                        sum_gaze_x_aoi[primary_surface] += gx
                        sum_gaze_y_aoi[primary_surface] += gy
                        
                    if is_fix:
                        key = (fix_id, primary_aoi)
                        if key not in fix_counted_for_aoi:
                            fix_count_aoi[primary_aoi]   += 1
                            fix_dur_sum_aoi[primary_aoi] += fix_dur
                            fix_counted_for_aoi.add(key)

                cur_label  = primary_aoi if primary_aoi else "NoAOI"
                prev_label = prev_aoi    if prev_aoi    else "NoAOI"
                transition_str = ""

                if cur_label != prev_label:
                    transition_str = f"{prev_label}->{cur_label}"
                    transitions[prev_label][cur_label] += 1
                    if primary_aoi:
                        cum_visits[primary_aoi] += 1
                        visit_count[primary_aoi] += 1
                    aoi_entry_frame = frame_idx
                    time_in_aoi_ms  = 0.0
                else:
                    time_in_aoi_ms = (frame_idx - aoi_entry_frame) * frame_dur_ms
                prev_aoi = primary_aoi

                writer.writerow([
                    ts, f"{time_s:.4f}", global_frame, f"{gx:.2f}", f"{gy:.2f}",
                    is_fix, fix_id if fix_id >= 0 else "", f"{fix_dur:.1f}" if is_fix else "",
                    f"{fix_cx:.2f}" if is_fix else "", f"{fix_cy:.2f}" if is_fix else "",
                    bool(primary_aoi), primary_aoi, gaze_on_aoi_x, gaze_on_aoi_y,
                    primary_surface, hit_source, primary_marker_count, primary_surface_initialized,
                ] + [row_hits[n] for n in aoi_names] + [transition_str, f"{time_in_aoi_ms:.1f}"]
                  + [cum_visits.get(n, 0) for n in aoi_names]
                  + [";".join(str(i) for i in sorted(detected_ids))])

                if video_writer is not None:
                    # Downscale immediately to save memory and processing
                    img = cv2.resize(frame.bgr, (vid_w, vid_h))
                    
                    _draw_scanpath(img, list(scanpath), scale=VIDEO_SCALE)

                    for name, poly in active_polygons.items():
                        color = AOI_COLORS.get(name, DEFAULT_COLOR)
                        _draw_polygon_box(img, poly, name, color, scale=VIDEO_SCALE)

                    if is_fix:
                        _draw_fixation_circle(img, fix_cx, fix_cy, fix_dur, scale=VIDEO_SCALE)
                    _draw_gaze_dot(img, gx, gy, scale=VIDEO_SCALE)

                    if primary_aoi:
                        _draw_banner(img, f"LOOKING AT: {primary_aoi.replace('_', ' ')}", 
                                   AOI_COLORS.get(primary_aoi, DEFAULT_COLOR), vid_h, vid_w)

                    _draw_frame_info(img, global_frame, time_s)
                    video_writer.write(img)

        if video_writer is not None:
            video_writer.release()
            video_writer = None
        write_progress(100.0, total, "complete")
        try:
            write_summaries(out_dir, csv_path, recording_dir.name, fps)
        except Exception as exc:
            log.warning("  Could not write summary files: %s", exc)
        if processing_path.exists():
            processing_path.unlink()
        log.info("  Done. Results saved in %s", out_dir)
    except Exception:
        if video_writer is not None:
            video_writer.release()
        write_progress(0.0, 0, "failed")
        for partial_path in (csv_path, video_path):
            try:
                if partial_path.exists():
                    partial_path.unlink()
            except OSError:
                log.warning("  Could not remove partial output: %s", partial_path)
        if processing_path.exists():
            processing_path.unlink()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 2D Hit-Test AOI Analysis")
    parser.add_argument("--force", action="store_true", help="Reprocess previously completed directories")
    args = parser.parse_args()
    
    root = RECORDINGS_DIR
    recordings = find_recording_dirs(root)
    
    pending = []
    for r in recordings:
        if args.force or not (r / "aoi_results" / "raw" / "analysis.csv").exists():
            pending.append(r)
            
    log.info("Analysing %d recording(s)", len(pending))
    for p in pending:
        analyze_recording(p)
