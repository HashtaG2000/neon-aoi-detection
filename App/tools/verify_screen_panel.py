"""Verify the display-panel rectangle inside the Screen AOI surface.

The Screen AOI is defined by the four corner AprilTags, whose outer edges span
SURFACE_DIMS_CM["Screen"] and are then widened by SURFACE_EXPAND. The monitor's
actual lit panel is a smaller rectangle inside that surface, so any sub-AOI
defined against the on-screen layout must be expressed in panel coordinates and
mapped into surface coordinates before it can be tested against gaze.

This tool draws, on real scene frames:
  yellow  the Screen AOI surface quad (what the analyser uses today)
  green   the predicted display-panel rectangle
  cyan    the panel's centre cross-hairs

If the green rectangle lands on the monitor's lit edges, the transform is
correct and sub-AOI work can proceed on top of it.

Usage:
    python App/tools/verify_screen_panel.py [RECORDING] [--n 6] [--out DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "App" / "src"))

import pupil_labs.neon_recording as nr  # noqa: E402
import rigid_surface  # noqa: E402
from analyzer import make_screen_detector  # noqa: E402

# Physical active area of the iiyama ProLite T2755MSC (27 in, 16:9, Full HD),
# read off the bezel label in Setup/screen_mid-task_gamified.jpeg.
PANEL_W_CM = 59.77
PANEL_H_CM = 33.62


def panel_rect_in_surface() -> tuple[float, float, float, float]:
    """Display panel as (x0, y0, x1, y1) in Screen-surface normalised coords."""
    L, H = rigid_surface.SURFACE_DIMS_CM["Screen"]
    sw = L * rigid_surface.SURFACE_EXPAND
    sh = H * rigid_surface.SURFACE_EXPAND
    return (0.5 - (PANEL_W_CM / 2) / sw, 0.5 - (PANEL_H_CM / 2) / sh,
            0.5 + (PANEL_W_CM / 2) / sw, 0.5 + (PANEL_H_CM / 2) / sh)


def _homography_from_quad(quad: np.ndarray) -> np.ndarray:
    """Map normalised [0,1]^2 onto an image quad given as TL, TR, BR, BL."""
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, quad.astype(np.float32))


def _project(hmg: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), hmg)
    return p.reshape(-1, 2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default="8WMAP")
    ap.add_argument("--n", type=int, default=6, help="frames to render")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rec_dir = None
    for cond in ("Gamified", "NonGamified"):
        cand = ROOT / "Recordings" / cond / args.recording
        if cand.exists():
            rec_dir = cand
            break
    if rec_dir is None:
        print(f"recording {args.recording} not found")
        return 1

    model_path = rec_dir / "aoi_results" / "raw" / "scene_model.pkl"
    if not model_path.exists():
        print(f"no cached scene model at {model_path}")
        return 1
    model = rigid_surface.load_scene_model(model_path)

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "Setup" / "panel_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = nr.load(str(rec_dir))
    det = make_screen_detector()
    screen_ids = set(rigid_surface.SCREEN_CORNER_IDS.values())

    x0, y0, x1, y1 = panel_rect_in_surface()
    print(f"panel rect in surface coords: x {x0:.4f}..{x1:.4f}  y {y0:.4f}..{y1:.4f}")

    ts = rec.scene.time
    # The screen's own tags decode only rarely, which is precisely why the
    # rigid-body model exists. Reproduce the pipeline's real path: localise the
    # camera from every placed tag, then project the Screen surface. Keep the
    # frames where the screen is largest and fully in view, and prefer a low
    # camera reprojection error so the quad can be trusted.
    h_img = int(rec.scene.height or 1200)
    w_img = int(rec.scene.width or 1600)
    step = max(1, len(ts) // 900)
    cand: list[tuple[float, int, float, int, np.ndarray, object]] = []
    for i, f in enumerate(rec.scene.sample(ts[::step])):
        dets = det.detect(f.gray)
        if not dets:
            continue
        loc = model.localize(dets)
        if loc is None:
            continue
        rvec, tvec, reproj, ntags = loc
        if reproj > rigid_surface.CAMERA_MAX_REPROJ_PX:
            continue
        n_screen = sum(1 for d in dets if d.tag_id in screen_ids)
        quad = model.surface_quad("Screen", dets, rvec, tvec)
        if quad is None:
            continue
        m = 40  # require the whole quad comfortably inside the frame
        if not (quad[:, 0].min() > m and quad[:, 1].min() > m
                and quad[:, 0].max() < w_img - m and quad[:, 1].max() < h_img - m):
            continue
        area = cv2.contourArea(quad.astype(np.float32))
        if area < 0.04 * w_img * h_img:
            continue
        cand.append((area, i, reproj, n_screen, quad, f))
    if not cand:
        print("no frame had the Screen surface fully in view")
        return 1

    cand.sort(key=lambda r: -r[0])
    print(f"{len(cand)} candidate frames; largest screen area "
          f"{100 * cand[0][0] / (w_img * h_img):.1f}% of frame, "
          f"camera reproj {cand[0][2]:.1f}px, {cand[0][3]} screen tags decoded")

    written = 0
    for area, idx, reproj, ntags, quad, frame in cand[: args.n]:
        img = frame.bgr.copy()
        hmg = _homography_from_quad(quad)

        cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 255), 3)
        panel = _project(hmg, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        cv2.polylines(img, [panel.astype(np.int32)], True, (0, 220, 0), 3)

        mid = _project(hmg, np.array([[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1],
                                      [x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]]))
        cv2.line(img, tuple(mid[0].astype(int)), tuple(mid[1].astype(int)), (255, 255, 0), 1)
        cv2.line(img, tuple(mid[2].astype(int)), tuple(mid[3].astype(int)), (255, 255, 0), 1)

        cv2.putText(img, f"{args.recording} frame~{idx * step}  "
                    f"{ntags} screen tags decoded  cam reproj {reproj:.1f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "yellow = Screen AOI surface   green = predicted display panel",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        p = out_dir / f"panel_check_{args.recording}_{written:02d}.png"
        cv2.imwrite(str(p), img)
        print(f"  wrote {p}")
        written += 1

    print(f"\n{written} overlay image(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
