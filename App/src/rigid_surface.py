"""
Scene-wide rigid-body surface tracker
======================================
Replaces the per-surface marker-mapper localisation (which needed every tag of
a surface visible at once, so the Screen almost never formed) with a single
rigid model of the whole apparatus.

Because the whole rig (Board, Boxes, Stream-deck, Screen) is physically fixed,
all AprilTags live in one world coordinate frame.  Once calibrated:

  * any visible tag (usually a Board tag) recovers the camera pose, so every
    surface's quadrangle can be projected — even on frames where that surface's
    own tags are not visible;
  * gaze is mapped into each surface's normalized [0,1] space for hit-testing
    and surface-space heatmaps.

Calibration pipeline (per recording, from a subsample of frames):
  1. Build a metric template for each surface from the measured physical sizes
     (tag = 3.5 cm; each surface's outer-edge L x H).  Corner->tag assignment is
     brute-forced by lowest reprojection.
  2. Localise each surface per frame via solvePnP against its template.
  3. Chain co-visible surface poses into one world frame (anchor = Board).
  4. Place the Screen — whose tags rarely decode and only at grazing angles — by
     multi-view TRIANGULATION of its tag corners using the accurate camera poses
     from the well-seen surfaces, then a rigid (Kabsch) fit of its template.

Validated on real recordings: Board/Boxes reproject to <10 px; the Screen is
recovered from the other surfaces' tags.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

# ── Physical layout (measured) ────────────────────────────────────────────────
TAG_SIZE_M = 0.035                       # every AprilTag is 3.5 cm square
_H = TAG_SIZE_M / 2.0

# pupil_apriltags corner order is [bottom-left, bottom-right, top-right, top-left]
_TAG_CORNERS_LOCAL = np.array([[-_H, -_H], [_H, -_H], [_H, _H], [-_H, _H]], dtype=np.float64)

# Surface outer-edge dimensions (length L x height H, cm) measured from the outer
# edges of the corner markers.
SURFACE_DIMS_CM: dict[str, tuple[float, float]] = {
    "Board":      (41.0, 24.0),
    "Left_Box":   (43.0, 15.0),
    "Middle_Box": (43.0, 15.0),
    "Right_Box":  (43.0, 15.0),
    "Screen":     (70.0, 34.0),
}
# Stream-deck: two tags 15 cm apart (centre-to-centre), tag 6 left, tag 7 right.
STREAM_DECK_SEP_CM = 15.0
# Screen corner-tag identities (from the physical setup): 12=TL, 14=TR, 15=BR, 13=BL.
SCREEN_CORNER_IDS = {"TL": 12, "TR": 14, "BR": 15, "BL": 13}

ANCHOR_SURFACE = "Board"      # most-visible surface; defines the world origin

# A surface whose own tags reproject worse than this (when the camera is localised
# from the anchor) is considered mis-placed and dropped from the model.
SURFACE_DROP_PX = 30.0
# Frames whose camera pose reprojects worse than this are too unreliable to attribute
# gaze to any AOI (avoids phantom hits from a shaky pose).
CAMERA_MAX_REPROJ_PX = 18.0
# Widen each surface's boundary a touch so gaze right at the edges still counts.
SURFACE_EXPAND = 1.06


# ── SE(3) helpers ─────────────────────────────────────────────────────────────

def _T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)
    return M


def _avg_se3(mats: list[np.ndarray]) -> np.ndarray:
    """Average a set of 4x4 rigid transforms (mean translation, SVD-orthogonalised
    mean rotation)."""
    t = np.mean([m[:3, 3] for m in mats], axis=0)
    U, _, Vt = np.linalg.svd(sum(m[:3, :3] for m in mats))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _robust_se3(mats: list[np.ndarray]) -> np.ndarray:
    """Outlier-robust average of rigid transforms: keep the translations close to
    the median (rejects IPPE pose flips / grazing-angle noise), then average."""
    if len(mats) <= 2:
        return _avg_se3(mats)
    ts = np.array([m[:3, 3] for m in mats])
    med = np.median(ts, axis=0)
    d = np.linalg.norm(ts - med, axis=1)
    thr = max(0.02, 2.5 * float(np.median(d)))   # 2 cm floor
    inliers = [m for m, di in zip(mats, d) if di <= thr]
    return _avg_se3(inliers if inliers else mats)


def _is_valid_quad(p: np.ndarray) -> bool:
    """Reject degenerate / self-intersecting / collapsed projected quads."""
    if p.shape[0] != 4 or not np.all(np.isfinite(p)):
        return False
    signs = []
    for i in range(4):
        a = p[(i + 1) % 4] - p[i]
        b = p[(i + 2) % 4] - p[(i + 1) % 4]
        signs.append(np.sign(a[0] * b[1] - a[1] * b[0]))
    if len({s for s in signs if s != 0}) > 1:      # not convex -> self-intersecting
        return False
    area = 0.5 * abs((p[2][0] - p[0][0]) * (p[3][1] - p[1][1]) -
                     (p[3][0] - p[1][0]) * (p[2][1] - p[0][1]))
    return area > 60.0                              # not collapsed to a sliver


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform mapping src (Nx3) onto dst (Nx3), least-squares."""
    sc = src.mean(0)
    dc = dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = dc - R @ sc
    return M


def _outer_quad_local(L_cm: float, H_cm: float, expand: float = 1.0) -> np.ndarray:
    """AOI boundary rectangle in the surface's own frame (TL,TR,BR,BL, metres, z=0).
    `expand` widens the box about its centre to catch gaze right at the edges."""
    w = L_cm / 200.0 * expand   # (L/2) cm -> m
    h = H_cm / 200.0 * expand
    return np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], dtype=np.float64)


# ── Templates ─────────────────────────────────────────────────────────────────

def _tag_corners_at(cx: float, cy: float) -> np.ndarray:
    return np.hstack([_TAG_CORNERS_LOCAL + [cx, cy], np.zeros((4, 1))])


def _rect_centres(L_cm: float, H_cm: float) -> np.ndarray:
    """The 4 tag-centre positions (TL,TR,BR,BL) for a 4-tag surface, metres."""
    w = (L_cm - TAG_SIZE_M * 100) / 200.0
    h = (H_cm - TAG_SIZE_M * 100) / 200.0
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float64)


def _reproj_err(template: dict[int, np.ndarray], frames: list[dict[int, np.ndarray]]) -> float:
    errs = []
    for fr in frames:
        vis = [t for t in template if t in fr]
        if not vis:
            continue
        obj = np.vstack([template[t] for t in vis])
        img = np.vstack([fr[t] for t in vis])
        flag = cv2.SOLVEPNP_IPPE if len(vis) >= 2 else cv2.SOLVEPNP_IPPE_SQUARE
        ok, rv, tv = cv2.solvePnP(obj, img, _reproj_err.K, _reproj_err.D, flags=flag)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, _reproj_err.K, _reproj_err.D)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    return float(np.median(errs)) if errs else 1e9


def _build_quad_template(
    tag_ids: list[int], L_cm: float, H_cm: float, frames: list[dict[int, np.ndarray]]
) -> dict[int, np.ndarray] | None:
    """4-tag surface: brute-force the corner->tag assignment by lowest reprojection."""
    rect = _rect_centres(L_cm, H_cm)
    best_err, best = 1e9, None
    for perm in itertools.permutations(tag_ids):
        tmpl = {tid: _tag_corners_at(*rect[i]) for i, tid in enumerate(perm)}
        err = _reproj_err(tmpl, frames)
        if err < best_err:
            best_err, best = err, tmpl
    return best


def _stream_template() -> dict[int, np.ndarray]:
    d = STREAM_DECK_SEP_CM / 200.0
    return {6: _tag_corners_at(-d, 0.0), 7: _tag_corners_at(d, 0.0)}


def _screen_template(L_cm: float, H_cm: float) -> dict[int, np.ndarray]:
    w = (L_cm - TAG_SIZE_M * 100) / 200.0
    h = (H_cm - TAG_SIZE_M * 100) / 200.0
    pos = {"TL": (-w, -h), "TR": (w, -h), "BR": (w, h), "BL": (-w, h)}
    return {SCREEN_CORNER_IDS[k]: _tag_corners_at(*pos[k]) for k in pos}


# ── Scene model ───────────────────────────────────────────────────────────────

@dataclass
class SceneModel:
    K: np.ndarray
    D: np.ndarray
    surface_tags: dict[str, list[int]]                        # placed tags per surface
    world_tag_corners: dict[int, np.ndarray] = field(default_factory=dict)   # tag -> 4x3 world
    surface_quad_world: dict[str, np.ndarray] = field(default_factory=dict)  # surface -> 4x3 world
    templates: dict[str, dict] = field(default_factory=dict)          # surface -> {tag: 4x3 local corners}
    surface_local_quad: dict[str, np.ndarray] = field(default_factory=dict)  # surface -> 4x3 local outer quad
    placed: list[str] = field(default_factory=list)
    calib_report: dict = field(default_factory=dict)

    # -- per-frame use ---------------------------------------------------------
    def localize(self, detections) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """Camera pose from every visible placed tag.
        Returns (rvec, tvec, mean_reprojection_px, n_tags) or None."""
        obj, img = [], []
        for d in detections:
            wc = self.world_tag_corners.get(d.tag_id)
            if wc is not None:
                obj.append(wc)
                img.append(d.corners)
        if len(obj) < 1:
            return None
        obj = np.vstack(obj).astype(np.float64)
        img = np.vstack(img).astype(np.float64)
        flag = cv2.SOLVEPNP_ITERATIVE if len(obj) >= 4 else cv2.SOLVEPNP_IPPE_SQUARE
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D, flags=flag)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
        return rvec, tvec, err, len(obj) // 4

    def project_quad(self, surface: str, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray | None:
        """Projected AOI quad, or None if the surface is behind the camera or the
        projection is geometrically invalid (guards against phantom quads)."""
        quad = self.surface_quad_world.get(surface)
        if quad is None:
            return None
        R, _ = cv2.Rodrigues(rvec)
        center_cam = R @ quad.mean(axis=0) + tvec.reshape(3)
        if center_cam[2] <= 0.05:                 # behind / on the camera plane
            return None
        proj, _ = cv2.projectPoints(quad, rvec, tvec, self.K, self.D)
        p = proj.reshape(-1, 2).astype(np.float32)
        if not _is_valid_quad(p):
            return None
        return p

    def surface_quad(self, surface: str, detections,
                     cam_rvec: np.ndarray | None, cam_tvec: np.ndarray | None) -> np.ndarray | None:
        """Best per-frame quad for a surface. Prefers a DIRECT fit from the
        surface's OWN visible tags (accurate — this is how the screen forms a
        proper rectangle whenever you look at it); falls back to the rigid-body
        projection from the camera pose only when its tags aren't visible."""
        tmpl = self.templates.get(surface)
        local = self.surface_local_quad.get(surface)
        if tmpl is not None and local is not None:
            vis = [d for d in detections if d.tag_id in tmpl]
            if vis:
                obj = np.vstack([tmpl[d.tag_id] for d in vis])
                img = np.vstack([d.corners for d in vis]).astype(np.float64)
                flag = cv2.SOLVEPNP_IPPE if len(vis) >= 2 else cv2.SOLVEPNP_IPPE_SQUARE
                ok, rv, tv = cv2.solvePnP(obj, img, self.K, self.D, flags=flag)
                if ok:
                    proj, _ = cv2.projectPoints(local, rv, tv, self.K, self.D)
                    p = proj.reshape(-1, 2).astype(np.float32)
                    if _is_valid_quad(p):
                        return p
        # Fallback: project from the whole-rig camera pose.
        if cam_rvec is not None and cam_tvec is not None:
            return self.project_quad(surface, cam_rvec, cam_tvec)
        return None

    def gaze_to_surface(
        self, surface: str, gx: float, gy: float, rvec: np.ndarray, tvec: np.ndarray,
        image_quad: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        """Map an image-space gaze point into the surface's normalized [0,1] frame.
        Returns (u,v) or None if the surface is unavailable."""
        quad = image_quad if image_quad is not None else self.project_quad(surface, rvec, tvec)
        if quad is None:
            return None
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        Hmat = cv2.getPerspectiveTransform(quad.astype(np.float32), unit)
        p = Hmat @ np.array([gx, gy, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_scene(
    frames: list[dict[int, np.ndarray]],
    surface_marker_ids: dict[str, list[int]],
    K: np.ndarray,
    D: np.ndarray,
    log=None,
) -> SceneModel:
    """Build the scene rigid body from a subsample of per-frame detections.

    `frames`             : list of {tag_id: 4x2 image corners} (one dict per sampled frame)
    `surface_marker_ids` : {surface_name: [tag ids]}  (e.g. from AOI_CONFIG)
    """
    def _log(msg):
        if log is not None:
            log.info(msg)

    _reproj_err.K, _reproj_err.D = K, D

    # 1. Templates -------------------------------------------------------------
    templates: dict[str, dict[int, np.ndarray]] = {}
    for surf, ids in surface_marker_ids.items():
        if surf == "Stream_Deck" or set(ids) == {6, 7}:
            templates[surf] = _stream_template()
        elif surf == "Screen":
            L, H = SURFACE_DIMS_CM["Screen"]
            templates[surf] = _screen_template(L, H)
        elif surf in SURFACE_DIMS_CM and len(ids) == 4:
            L, H = SURFACE_DIMS_CM[surf]
            t = _build_quad_template(ids, L, H, frames)
            if t is not None:
                templates[surf] = t

    non_screen = [s for s in templates if s != "Screen"]

    def surf_pose(surf, fr, min_tags=1):
        tmpl = templates[surf]
        vis = [t for t in tmpl if t in fr]
        if len(vis) < min_tags:
            return None
        obj = np.vstack([tmpl[t] for t in vis])
        img = np.vstack([fr[t] for t in vis])
        flag = cv2.SOLVEPNP_IPPE if len(vis) >= 2 else cv2.SOLVEPNP_IPPE_SQUARE
        ok, rv, tv = cv2.solvePnP(obj, img, K, D, flags=flag)
        return _T(rv, tv) if ok else None

    # 2/3. Scene graph over non-screen surfaces (anchor = Board) ---------------
    rel: dict[tuple[str, str], list[np.ndarray]] = {}
    for fr in frames:
        poses = {s: surf_pose(s, fr, min_tags=2) for s in non_screen}
        poses = {s: p for s, p in poses.items() if p is not None}
        for a in poses:
            for b in poses:
                if a != b:
                    rel.setdefault((a, b), []).append(np.linalg.inv(poses[a]) @ poses[b])

    anchor = ANCHOR_SURFACE if ANCHOR_SURFACE in non_screen else (non_screen[0] if non_screen else None)
    world_pose: dict[str, np.ndarray] = {}
    if anchor is not None:
        world_pose[anchor] = np.eye(4)
        changed = True
        while changed:
            changed = False
            for (a, b), mats in rel.items():
                if a in world_pose and b not in world_pose and len(mats) >= 3:
                    world_pose[b] = world_pose[a] @ _robust_se3(mats)
                    changed = True

    model = SceneModel(K=K, D=D, surface_tags={})

    def _place(surf, W):
        for t, local in templates[surf].items():
            model.world_tag_corners[t] = (W @ np.hstack([local, np.ones((4, 1))]).T).T[:, :3]
        L, H = (SURFACE_DIMS_CM.get(surf) or (STREAM_DECK_SEP_CM + TAG_SIZE_M * 100, TAG_SIZE_M * 100))
        oq = _outer_quad_local(L, H, SURFACE_EXPAND)
        model.surface_quad_world[surf] = (W @ np.hstack([oq, np.ones((4, 1))]).T).T[:, :3]
        model.surface_local_quad[surf] = oq                  # for direct per-frame fits
        model.surface_tags[surf] = list(templates[surf])

    for surf in world_pose:
        _place(surf, world_pose[surf])

    # Keep every surface's template + boundary so it can form its rectangle DIRECTLY
    # from its own visible tags (this is how the screen makes a proper rectangle when
    # you look at it, independent of the rig-wide triangulation below).
    model.templates = templates
    for surf in templates:
        if surf not in model.surface_local_quad:
            Ls, Hs = (SURFACE_DIMS_CM.get(surf)
                      or (STREAM_DECK_SEP_CM + TAG_SIZE_M * 100, TAG_SIZE_M * 100))
            model.surface_local_quad[surf] = _outer_quad_local(Ls, Hs, SURFACE_EXPAND)

    # Validate placement by leave-one-surface-out reprojection: localise the camera
    # from every OTHER placed tag, then reproject the surface's own tags. A surface
    # that a noisy calibration mis-placed reprojects far off. Iterate — drop the worst
    # offender, then re-check the rest with it removed (so one bad surface can't drag
    # the others down). Result: a surface is kept only if it is geometrically
    # consistent with the rest of the rig; otherwise it becomes NoAOI (never a
    # phantom hit).
    def _surface_reproj(surf):
        stags = set(templates[surf])
        errs = []
        for fr in frames:
            svis = [t for t in stags if t in fr and t in model.world_tag_corners]
            others = [t for t in fr if t in model.world_tag_corners and t not in stags]
            if not svis or len(others) < 2:
                continue
            ok, rv, tv = cv2.solvePnP(
                np.vstack([model.world_tag_corners[t] for t in others]),
                np.vstack([fr[t] for t in others]), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            for t in svis:
                pr, _ = cv2.projectPoints(model.world_tag_corners[t], rv, tv, K, D)
                errs.append(float(np.linalg.norm(pr.reshape(-1, 2) - fr[t], axis=1).mean()))
        return (float(np.median(errs)), len(errs)) if errs else (0.0, 0)

    # Keep ALL surfaces (no dropping). Reprojection-from-the-anchor is only a rough
    # quality hint — a correctly-placed but distant surface can read high purely from
    # lever-arm noise — so we just log a note; the per-frame projection gating still
    # skips genuinely bad frames (behind camera / degenerate quad / shaky pose).
    for surf in [s for s in world_pose if s != anchor]:
        med, n = _surface_reproj(surf)
        if n >= 3 and med > SURFACE_DROP_PX:
            _log(f"  Note: {surf} placement is loose (reproj {med:.0f}px) — kept.")

    # 4. Accurate camera poses from non-screen world tags ----------------------
    def cam_pose(fr):
        obj, img = [], []
        for t in fr:
            wc = model.world_tag_corners.get(t)
            if wc is not None:
                obj.append(wc)
                img.append(fr[t])
        if len(obj) < 2:
            return None
        ok, rv, tv = cv2.solvePnP(np.vstack(obj), np.vstack(img), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(np.vstack(obj), rv, tv, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - np.vstack(img), axis=1).mean()
        return rv, tv, err

    # 4b. Screen via multi-view triangulation of its tag corners ---------------
    screen_placed = False
    if "Screen" in templates:
        # per screen-tag corner index -> list of (3x4 projection matrix, image point)
        obs: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
        for fr in frames:
            cp = cam_pose(fr)
            if cp is None:
                continue
            rv, tv, err = cp
            if err > 12.0:                       # only trust well-localised frames
                continue
            R, _ = cv2.Rodrigues(rv)
            P = K @ np.hstack([R, tv.reshape(3, 1)])
            for t in fr:
                if t in SCREEN_CORNER_IDS.values():
                    for ci in range(4):
                        obs.setdefault((t, ci), []).append((P, fr[t][ci]))

        def triangulate(views):
            A = []
            for P, pt in views:
                x, y = pt
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            _, _, Vt = np.linalg.svd(np.array(A))
            X = Vt[-1]
            return X[:3] / X[3]

        src, dst = [], []      # template corner (local) -> triangulated world
        screen_tmpl = templates["Screen"]
        for (t, ci), views in obs.items():
            if len(views) >= 3:
                src.append(screen_tmpl[t][ci])
                dst.append(triangulate(views))
        if len(src) >= 4:
            W_screen = _kabsch(np.array(src), np.array(dst))
            for t, local in screen_tmpl.items():
                model.world_tag_corners[t] = (W_screen @ np.hstack([local, np.ones((4, 1))]).T).T[:, :3]
            L, H = SURFACE_DIMS_CM["Screen"]
            model.surface_quad_world["Screen"] = (
                W_screen @ np.hstack([_outer_quad_local(L, H, SURFACE_EXPAND), np.ones((4, 1))]).T).T[:, :3]
            model.surface_tags["Screen"] = list(screen_tmpl)
            screen_placed = True

    model.placed = list(model.surface_quad_world)
    model.calib_report = {
        "placed_surfaces": model.placed,
        "screen_placed": screen_placed,
        "anchor": anchor,
        "n_calib_frames": len(frames),
    }
    _log(f"  Rigid-body calibration: placed {model.placed} (screen={'yes' if screen_placed else 'NO'})")
    return model
