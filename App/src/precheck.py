"""
Phase 1 — AOI Setup & Validation
=================================
Run this before a full analyzer pass.

Samples 8 frames evenly across each recording, detects AprilTag markers,
draws the AOI bounding boxes, and shows them to you one by one so you can
confirm the detection is correct before investing time in the full analysis.

Controls (OpenCV window):
  SPACE or ENTER  →  next sample frame
  Q               →  skip this recording entirely
  C               →  mark as confirmed, stop previewing, move on

At the end of each recording you will be asked:
  "Proceed with full analysis? [y/n]"
  If y, analyzer.py is run automatically for that recording.
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys

import cv2
import numpy as np
import pupil_apriltags

from paths import RECORDINGS_DIR, ensure_vendor_paths

ensure_vendor_paths()

import pupil_labs.neon_recording as nr
from pupil_labs.camera import Camera
from pupil_labs.marker_mapper.aoi import AOI
from pupil_labs.marker_mapper.utils import get_surface_boundary
from analyzer import get_fallback_aoi_polygon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Mirror the config from analyzer.py ────────────────────────────────────

AOI_CONFIG: dict[str, list[int]] = {
    "Board":       [0, 1, 2, 3],
    "Left_Box":    [8, 9, 10, 11],
    "Stream_Deck": [6, 7],
    "Middle_Box":  [16, 17, 18, 19],
    "Right_Box":   [20, 21, 22, 23],
    "Screen":      [12, 13, 14, 15],
}

AOI_COLORS: dict[str, tuple[int, int, int]] = {
    "Board":       (  0, 200, 255),
    "Left_Box":    (  0, 255,   0),
    "Stream_Deck": (255,   0, 255),
    "Middle_Box":  (  0, 165, 255),
    "Right_Box":   (255, 255,   0),
    "Screen":      (255,  80,  80),
}
DEFAULT_COLOR  = (180, 180, 180)
N_SAMPLES      = 8      # frames to sample per recording
DISPLAY_WIDTH  = 1100   # pixels wide for display window

RECORDING_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def find_recording_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and RECORDING_PATTERN.match(p.name))


def make_camera(recording: nr.NeonRecording) -> Camera:
    cal = recording.calibration
    return Camera(1600, 1200,
                  cal.scene_camera_matrix,
                  cal.scene_distortion_coefficients)


def make_detector() -> pupil_apriltags.Detector:
    """High-accuracy detector — no decimation, no blur."""
    return pupil_apriltags.Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.5,
        debug=0,
    )


_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def enhance_frame(gray: np.ndarray) -> np.ndarray:
    """Adaptive contrast enhancement for better detection of angled/shadowed markers."""
    return _clahe.apply(gray)


def _draw_aoi_box(img, surface2image, camera, name, color):
    try:
        boundary = get_surface_boundary(surface2image, distorted=True, camera=camera)
        pts = boundary.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
        cx = int(boundary[:, 0].mean())
        cy = int(boundary[:, 1].mean())
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(name, font, 0.65, 2)
        cv2.rectangle(img, (cx - 5, cy - th - 7), (cx + tw + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(img, name, (cx, cy), font, 0.65, color, 2, cv2.LINE_AA)
    except Exception:
        pass


def _draw_aoi_polygon(img, polygon, name, color):
    try:
        pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        corners_2d = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        cx = int(corners_2d[:, 0].mean())
        cy = int(corners_2d[:, 1].mean())
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(name, font, 0.55, 2)
        cv2.rectangle(img, (cx - 5, cy - th - 7), (cx + tw + 5, cy + 5), (0, 0, 0), -1)
        cv2.putText(img, name, (cx, cy), font, 0.55, color, 2, cv2.LINE_AA)
    except Exception:
        pass


def validate_recording(
    rec_dir: pathlib.Path,
    recording: nr.NeonRecording,
    camera: Camera,
    detector: pupil_apriltags.Detector,
) -> bool:
    """
    Show N sample frames. Returns True if user confirms, False to skip.
    """
    scene_ts = recording.scene.time
    total    = len(scene_ts)

    # Pick N evenly-spaced timestamps across the recording
    sample_ts = np.linspace(scene_ts[0], scene_ts[-1], N_SAMPLES, dtype=np.int64)
    sampled   = recording.scene.sample(sample_ts)

    # Fresh AOIs for detection testing
    aois = [AOI(name, ids) for name, ids in AOI_CONFIG.items()]

    # Per-AOI detection counts across sampled frames
    det_counts: dict[str, int] = {name: 0 for name in AOI_CONFIG}

    confirmed  = False
    win_title  = f"AOI Validation — {rec_dir.name}"

    print(f"\n{'='*70}")
    print(f"  Recording : {rec_dir.name}  |  {total} frames")
    print(f"  Previewing {N_SAMPLES} sample frames.")
    print(f"  Controls: SPACE/ENTER = next frame | C = confirm | Q = skip recording")
    print(f"{'='*70}")

    for sample_num, frame in enumerate(sampled):
        detections  = detector.detect(enhance_frame(frame.gray))
        detected_ids = {d.tag_id for d in detections}

        # Initialize AOIs from this frame if possible
        for aoi in aois:
            if not aoi.is_initialized:
                visible_tags = [d for d in detections if d.tag_id in aoi.marker_ids]
                if len(visible_tags) == len(aoi.marker_ids):
                    aoi.initialize(detections, camera)

        # Track which AOIs have ≥2 markers detected
        for aoi in aois:
            if aoi.is_initialized or get_fallback_aoi_polygon(detections, aoi.marker_ids) is not None:
                det_counts[aoi.name] += 1

        # ── Draw ─────────────────────────────────────────────────────────────
        img = frame.bgr.copy()

        # Draw AOI boundary boxes for detected AOIs
        for aoi in aois:
            loc = aoi.localize(detections, camera)
            if loc is not None:
                _, s2i = loc
                color = AOI_COLORS.get(aoi.name, DEFAULT_COLOR)
                _draw_aoi_box(img, s2i, camera, aoi.name, color)
            else:
                fallback_poly = get_fallback_aoi_polygon(detections, aoi.marker_ids)
                if fallback_poly is not None:
                    color = AOI_COLORS.get(aoi.name, DEFAULT_COLOR)
                    _draw_aoi_polygon(img, fallback_poly, f"{aoi.name}*", color)

        # Draw every detected tag ID at its centre (yellow number)
        for d in detections:
            cx, cy = int(d.center[0]), int(d.center[1])
            tid_str = str(d.tag_id)
            cv2.putText(img, tid_str, (cx - 9, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, tid_str, (cx - 9, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 230), 2, cv2.LINE_AA)

        # Frame info banner
        time_s  = (scene_ts[min(sample_num * total // N_SAMPLES, total - 1)]
                   - scene_ts[0]) / 1e9
        tags_str = " ".join(str(i) for i in sorted(detected_ids)) or "none"
        info = (f"Sample {sample_num + 1}/{N_SAMPLES}  |  t={time_s:.1f}s  "
                f"|  Tags: [{tags_str}]")
        cv2.putText(img, info, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, info, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        hint = "SPACE/ENTER=next  |  C=confirm  |  Q=skip recording"
        h_img = img.shape[0]
        cv2.putText(img, hint, (10, h_img - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hint, (10, h_img - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Resize for display
        h, w = img.shape[:2]
        disp_h = int(h * DISPLAY_WIDTH / w)
        img_disp = cv2.resize(img, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(win_title, img_disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in (ord('q'), ord('Q')):
            print(f"  >> Skipped by user.")
            return False
        if key in (ord('c'), ord('C')):
            confirmed = True
            print(f"  >> Confirmed at sample {sample_num + 1}.")
            break
        # SPACE, ENTER, or any other key → next frame

    # ── Detection statistics ──────────────────────────────────────────────────
    print(f"\n  Detection summary ({N_SAMPLES} sampled frames):")
    print(f"  {'AOI':<16} {'Detected':>10}   {'Rate':>7}   Status")
    print(f"  {'-'*52}")
    all_ok = True
    for name in AOI_CONFIG:
        cnt  = det_counts[name]
        rate = 100.0 * cnt / N_SAMPLES
        status = "OK" if rate >= 50 else "LOW — may miss data"
        if rate < 50:
            all_ok = False
        print(f"  {name:<16} {cnt:>4}/{N_SAMPLES:<4}   {rate:>5.0f}%   {status}")

    if not all_ok:
        print("\n  WARNING: Some AOIs have low detection rates.")
        print("  This could be due to: marker occlusion, steep angle,")
        print("  small physical marker size, or lighting conditions.")

    if confirmed:
        return True

    print(f"\n  Proceed with full analysis for this recording? [y/n]: ", end="", flush=True)
    resp = input().strip().lower()
    return resp in ("y", "yes", "")


def main() -> None:
    # OpenCV/av init workaround
    cv2.imshow("init", np.zeros(1))
    cv2.destroyAllWindows()

    root           = RECORDINGS_DIR
    recording_dirs = find_recording_dirs(root)

    if not recording_dirs:
        log.error("No recording folders found in %s", root)
        sys.exit(1)

    log.info("Found %d recording(s):", len(recording_dirs))
    for r in recording_dirs:
        log.info("  %s", r.name)

    confirmed_dirs: list[pathlib.Path] = []

    for rec_dir in recording_dirs:
        try:
            recording = nr.load(str(rec_dir))
            camera    = make_camera(recording)
            detector  = make_detector()
            ok        = validate_recording(rec_dir, recording, camera, detector)
            if ok:
                confirmed_dirs.append(rec_dir)
                log.info("  Confirmed: %s", rec_dir.name)
            else:
                log.info("  Skipped:   %s", rec_dir.name)
        except Exception:
            log.exception("Error validating %s", rec_dir.name)

    print(f"\n{'='*70}")
    print(f"  Validation complete.")
    print(f"  Confirmed: {len(confirmed_dirs)} / {len(recording_dirs)} recording(s)")
    print(f"{'='*70}\n")

    if not confirmed_dirs:
        print("  Nothing to analyse. Exiting.")
        return

    print("  Run full analysis on confirmed recordings? [y/n]: ", end="", flush=True)
    if input().strip().lower() not in ("y", "yes", ""):
        print("  To run manually:  python analyzer.py")
        return

    # Import and run the analysis
    import analyzer
    for rec_dir in confirmed_dirs:
        analyzer.analyze_recording(rec_dir)


if __name__ == "__main__":
    main()
