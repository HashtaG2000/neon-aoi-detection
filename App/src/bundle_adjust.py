"""Joint refinement of the scene rigid body.

The initial scene graph is built by chaining pairwise relative poses out from
the anchor surface, and the Screen is then placed by triangulation that depends
on those already-drifted camera poses. Both steps accumulate error, and because
the Screen sits far from the anchor, a small angular error there becomes a large
positional one -- enough to displace its AOI quad by hundreds of pixels.

This module refines every surface pose and every calibration-frame camera pose
simultaneously, minimising total marker reprojection error. Each surface stays
rigid (its markers keep their measured layout), so only six parameters per
surface are estimated; the anchor is held fixed to fix the gauge.
"""
from __future__ import annotations

import numpy as np

# Levenberg-Marquardt is only worth running when SciPy is present; callers fall
# back to the unrefined model otherwise.
try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


def _rodrigues_batch(rvecs: np.ndarray) -> np.ndarray:
    """(N,3) rotation vectors -> (N,3,3) rotation matrices."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    small = theta[:, 0] < 1e-12
    axis = np.where(theta > 1e-12, rvecs / np.where(theta == 0, 1, theta), 0.0)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(x)
    Kx = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=1).reshape(-1, 3, 3)
    th = theta[:, 0][:, None, None]
    eye = np.broadcast_to(np.eye(3), (len(rvecs), 3, 3))
    R = eye + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    R[small] = np.eye(3)
    return R


def _project(pts_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Pinhole + Brown-Conrady projection, vectorised over all points."""
    z = np.where(np.abs(pts_cam[:, 2]) < 1e-9, 1e-9, pts_cam[:, 2])
    x, y = pts_cam[:, 0] / z, pts_cam[:, 1] / z
    d = np.zeros(8)
    d[: min(8, len(np.ravel(D)))] = np.ravel(D)[: min(8, len(np.ravel(D)))]
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)


def _pose_to_params(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    # log map of SO(3)
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-8:
        rv = np.zeros(3)
    else:
        rv = th / (2 * np.sin(th)) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.concatenate([rv, t])


def _params_to_pose(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rodrigues_batch(p[None, :3])[0]
    T[:3, 3] = p[3:]
    return T


def refine_scene(
    *,
    templates: dict[str, dict[int, np.ndarray]],
    surface_world_pose: dict[str, np.ndarray],
    frames: list[dict[int, np.ndarray]],
    K: np.ndarray,
    D: np.ndarray,
    anchor: str,
    max_frames: int = 400,
    log=None,
) -> tuple[dict[str, np.ndarray], dict] | None:
    """Jointly refine surface poses and camera poses.

    Returns (refined_surface_world_pose, report), or None if refinement could
    not run. The anchor surface is held at its incoming pose.
    """
    def _log(m):
        if log is not None:
            log.info(m)

    if not HAVE_SCIPY:
        _log("  Bundle adjustment: SciPy unavailable, keeping initial scene")
        return None

    surfs = [s for s in surface_world_pose if s in templates]
    if anchor not in surfs or len(surfs) < 2:
        return None
    # Anchor first so its parameters can be sliced off and held fixed.
    surfs = [anchor] + [s for s in surfs if s != anchor]
    s_index = {s: i for i, s in enumerate(surfs)}
    tag_surface = {t: s for s in surfs for t in templates[s]}

    # Pick calibration frames. Screen observations are the scarcest and the most
    # valuable (they are what pins the Screen down), so take every frame that
    # sees a Screen tag before filling up with the richest remaining frames.
    screen_tags = set(templates.get("Screen", {}))
    with_screen, without = [], []
    for fr in frames:
        vis = [t for t in fr if t in tag_surface]
        if len(vis) < 2:
            continue
        (with_screen if any(t in screen_tags for t in vis) else without).append(fr)
    without.sort(key=lambda fr: -sum(1 for t in fr if t in tag_surface))
    chosen = with_screen + without[: max(0, max_frames - len(with_screen))]
    if len(chosen) < 8:
        _log("  Bundle adjustment: too few usable calibration frames")
        return None

    # Initial camera poses from the incoming scene, plus the observation table.
    import cv2
    cam_params: list[np.ndarray] = []
    obs_local, obs_uv, obs_s, obs_c = [], [], [], []
    kept = 0
    for fr in chosen:
        obj, img, loc, si = [], [], [], []
        for t, corners in fr.items():
            s = tag_surface.get(t)
            if s is None:
                continue
            W = surface_world_pose[s]
            world = (W @ np.hstack([templates[s][t], np.ones((4, 1))]).T).T[:, :3]
            obj.append(world)
            img.append(corners)
            loc.append(templates[s][t])
            si.append(np.full(4, s_index[s]))
        if len(obj) < 2:
            continue
        ok, rv, tv = cv2.solvePnP(np.vstack(obj).astype(np.float64),
                                  np.vstack(img).astype(np.float64), K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        ci = kept
        kept += 1
        cam_params.append(np.concatenate([rv.ravel(), tv.ravel()]))
        obs_local.append(np.vstack(loc))
        obs_uv.append(np.vstack(img))
        obs_s.append(np.concatenate(si))
        obs_c.append(np.full(len(np.vstack(loc)), ci))
    if kept < 8:
        _log("  Bundle adjustment: too few localisable frames")
        return None

    obs_local = np.vstack(obs_local).astype(np.float64)
    obs_uv = np.vstack(obs_uv).astype(np.float64)
    obs_s = np.concatenate(obs_s).astype(int)
    obs_c = np.concatenate(obs_c).astype(int)
    n_surf, n_cam, n_obs = len(surfs), kept, len(obs_local)

    surf_params = np.array([_pose_to_params(surface_world_pose[s]) for s in surfs])
    anchor_fixed = surf_params[0].copy()
    x0 = np.concatenate([surf_params[1:].ravel(), np.array(cam_params).ravel()])
    n_free_surf = n_surf - 1

    def unpack(x):
        sp = np.vstack([anchor_fixed, x[: n_free_surf * 6].reshape(n_free_surf, 6)])
        cp = x[n_free_surf * 6:].reshape(n_cam, 6)
        return sp, cp

    def residuals(x):
        sp, cp = unpack(x)
        Rs, ts = _rodrigues_batch(sp[:, :3]), sp[:, 3:]
        Rc, tc = _rodrigues_batch(cp[:, :3]), cp[:, 3:]
        world = np.einsum("nij,nj->ni", Rs[obs_s], obs_local) + ts[obs_s]
        cam = np.einsum("nij,nj->ni", Rc[obs_c], world) + tc[obs_c]
        return (_project(cam, K, D) - obs_uv).ravel()

    sparsity = lil_matrix((n_obs * 2, len(x0)), dtype=int)
    rows = np.arange(n_obs)
    for k in range(6):
        free = obs_s > 0
        sparsity[2 * rows[free], (obs_s[free] - 1) * 6 + k] = 1
        sparsity[2 * rows[free] + 1, (obs_s[free] - 1) * 6 + k] = 1
        sparsity[2 * rows, n_free_surf * 6 + obs_c * 6 + k] = 1
        sparsity[2 * rows + 1, n_free_surf * 6 + obs_c * 6 + k] = 1

    def _rms(r):
        return float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2, axis=1).mean()))

    before = _rms(residuals(x0))
    # Two stages. A robust loss whose scale is far below the current residuals
    # puts every observation in the outlier regime at once, where the gradient
    # is nearly flat and the optimiser cannot move -- so start with a scale set
    # from the data to pull the geometry into the right basin, then tighten it
    # to reject genuinely bad detections.
    x = x0
    for f_scale in (max(20.0, before), 5.0):
        res = least_squares(residuals, x, jac_sparsity=sparsity, method="trf",
                            loss="soft_l1", f_scale=f_scale, max_nfev=80, verbose=0)
        x = res.x
    after = _rms(residuals(x))

    # Per-surface residuals make it obvious which surface still does not fit.
    def _per_surface(r):
        e = np.linalg.norm(r.reshape(-1, 2), axis=1)
        return {s: round(float(np.median(e[obs_s == i])), 2)
                for i, s in enumerate(surfs) if np.any(obs_s == i)}

    med_before = _per_surface(residuals(x0))
    med_after = _per_surface(residuals(x))

    sp, _ = unpack(x)
    refined = {s: _params_to_pose(sp[i]) for i, s in enumerate(surfs)}

    shifts = {}
    for s in surfs:
        d = np.linalg.norm(refined[s][:3, 3] - surface_world_pose[s][:3, 3]) * 100
        shifts[s] = round(float(d), 2)

    report = {
        "frames_used": n_cam,
        "observations": int(n_obs),
        "surfaces": surfs,
        "rms_px_before": round(before, 2),
        "rms_px_after": round(after, 2),
        "median_px_before": med_before,
        "median_px_after": med_after,
        "surface_shift_cm": shifts,
    }
    _log(f"  Bundle adjustment: {n_cam} frames, {n_obs} marker corners, "
         f"RMS {before:.1f} -> {after:.1f} px")
    _log(f"    per-surface median px: {med_before} -> {med_after}")
    _log(f"    surface shifts (cm): {shifts}")
    return refined, report


def refine_tags(
    *,
    world_tag_corners: dict[int, np.ndarray],
    fixed_tags: set[int],
    frames: list[dict[int, np.ndarray]],
    K: np.ndarray,
    D: np.ndarray,
    tag_local: np.ndarray,
    max_frames: int = 400,
    log=None,
) -> tuple[dict[int, np.ndarray], dict] | None:
    """Refine each marker's world pose individually.

    The surface templates assume markers sit exactly on the corners of a
    measured rectangle. In reality they are taped on by hand, so centimetre-level
    departures remain that no rigid surface pose can absorb, and they show up as
    a residual floor. Here every marker gets its own six parameters, keeping only
    its printed size fixed, so the rig geometry is determined by the imagery
    rather than by the measuring tape. The anchor's markers are held fixed to
    keep the world frame and scale from drifting.
    """
    def _log(m):
        if log is not None:
            log.info(m)
    if not HAVE_SCIPY:
        return None
    import cv2

    tags = sorted(world_tag_corners)
    free = [t for t in tags if t not in fixed_tags]
    if not free:
        return None
    t_index = {t: i for i, t in enumerate(tags)}

    pose0 = {}
    for t in tags:
        pose0[t] = _pose_to_params(_kabsch_local(tag_local, world_tag_corners[t]))

    usable = []
    for fr in frames:
        if sum(1 for t in fr if t in t_index) >= 2:
            usable.append(fr)
    usable.sort(key=lambda fr: -sum(1 for t in fr if t in t_index))
    usable = usable[:max_frames]
    if len(usable) < 8:
        return None

    cam_params, obs_local, obs_uv, obs_t, obs_c = [], [], [], [], []
    kept = 0
    for fr in usable:
        obj = [world_tag_corners[t] for t in fr if t in t_index]
        img = [fr[t] for t in fr if t in t_index]
        ids = [t for t in fr if t in t_index]
        if len(obj) < 2:
            continue
        ok, rv, tv = cv2.solvePnP(np.vstack(obj).astype(np.float64),
                                  np.vstack(img).astype(np.float64), K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        for t, c in zip(ids, img):
            obs_local.append(tag_local)
            obs_uv.append(c)
            obs_t.append(np.full(4, t_index[t]))
            obs_c.append(np.full(4, kept))
        cam_params.append(np.concatenate([rv.ravel(), tv.ravel()]))
        kept += 1
    if kept < 8:
        return None

    obs_local = np.vstack(obs_local).astype(np.float64)
    obs_uv = np.vstack(obs_uv).astype(np.float64)
    obs_t = np.concatenate(obs_t).astype(int)
    obs_c = np.concatenate(obs_c).astype(int)
    n_obs, n_cam = len(obs_local), kept

    tag_params = np.array([pose0[t] for t in tags])
    free_idx = np.array([t_index[t] for t in free])
    is_free = np.zeros(len(tags), dtype=bool)
    is_free[free_idx] = True
    slot = -np.ones(len(tags), dtype=int)
    slot[free_idx] = np.arange(len(free))

    x0 = np.concatenate([tag_params[free_idx].ravel(), np.array(cam_params).ravel()])
    nft = len(free)

    def unpack(x):
        tp = tag_params.copy()
        tp[free_idx] = x[: nft * 6].reshape(nft, 6)
        return tp, x[nft * 6:].reshape(n_cam, 6)

    def residuals(x):
        tp, cp = unpack(x)
        Rt, tt = _rodrigues_batch(tp[:, :3]), tp[:, 3:]
        Rc, tc = _rodrigues_batch(cp[:, :3]), cp[:, 3:]
        world = np.einsum("nij,nj->ni", Rt[obs_t], obs_local) + tt[obs_t]
        cam = np.einsum("nij,nj->ni", Rc[obs_c], world) + tc[obs_c]
        return (_project(cam, K, D) - obs_uv).ravel()

    sparsity = lil_matrix((n_obs * 2, len(x0)), dtype=int)
    rows = np.arange(n_obs)
    fmask = is_free[obs_t]
    for k in range(6):
        sparsity[2 * rows[fmask], slot[obs_t[fmask]] * 6 + k] = 1
        sparsity[2 * rows[fmask] + 1, slot[obs_t[fmask]] * 6 + k] = 1
        sparsity[2 * rows, nft * 6 + obs_c * 6 + k] = 1
        sparsity[2 * rows + 1, nft * 6 + obs_c * 6 + k] = 1

    def _rms(r):
        return float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2, axis=1).mean()))

    before = _rms(residuals(x0))
    x = x0
    for f_scale in (max(10.0, before), 3.0):
        res = least_squares(residuals, x, jac_sparsity=sparsity, method="trf",
                            loss="soft_l1", f_scale=f_scale, max_nfev=80, verbose=0)
        x = res.x
    after = _rms(residuals(x))

    tp, _ = unpack(x)
    out = {}
    for t in tags:
        T = _params_to_pose(tp[t_index[t]])
        out[t] = (T @ np.hstack([tag_local, np.ones((4, 1))]).T).T[:, :3]

    shifts = {int(t): round(float(np.linalg.norm(
        out[t].mean(axis=0) - world_tag_corners[t].mean(axis=0)) * 100), 2) for t in tags}
    report = {"frames_used": n_cam, "observations": int(n_obs),
              "free_tags": len(free), "rms_px_before": round(before, 2),
              "rms_px_after": round(after, 2),
              "max_tag_shift_cm": max(shifts.values()) if shifts else 0.0}
    _log(f"  Per-marker refinement: {len(free)} markers, {n_cam} frames, "
         f"RMS {before:.1f} -> {after:.1f} px "
         f"(largest marker move {report['max_tag_shift_cm']:.1f} cm)")
    return out, report


def _kabsch_local(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform taking src (Nx3) onto dst (Nx3)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dc - R @ sc
    return T
