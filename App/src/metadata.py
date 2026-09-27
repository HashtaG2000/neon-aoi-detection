"""Recording metadata and auxiliary sensor-stream extraction.

The analysis pipeline historically consumed only gaze, fixations and the scene
video. A Neon recording folder also ships pupillometry, eyelid state, saccades,
blinks, head motion (IMU), a device-worn flag, events and a provenance block in
info.json -- all useful for interpreting attention data (pupil diameter as a
cognitive-load proxy, head motion as an explanation for gaze landing on no
surface, worn/blink flags for data-quality gating).

Everything here is ADDITIVE and runs as a post-process: per-frame values are
aligned to the frames already present in analysis.csv (matched on timestamp_ns),
so no recording has to be re-analysed to gain these columns.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import pupil_labs.neon_recording as nr

# Columns this module appends to analysis.csv.
PER_FRAME_COLUMNS = [
    "pupil_left_mm", "pupil_right_mm", "pupil_mean_mm",
    "eyelid_aperture_left_mm", "eyelid_aperture_right_mm",
    "is_blink", "is_saccade", "saccade_amplitude_deg", "saccade_peak_velocity",
    "worn", "head_angular_velocity_dps", "head_acceleration_g",
]


def _nearest_index(src_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Index of the temporally closest src sample for every target timestamp."""
    if len(src_times) == 0:
        return np.array([], dtype=int)
    order = np.argsort(src_times, kind="stable")
    st = src_times[order]
    idx = np.clip(np.searchsorted(st, target_times), 0, len(st) - 1)
    left = np.clip(idx - 1, 0, len(st) - 1)
    choose_left = np.abs(st[left] - target_times) <= np.abs(st[idx] - target_times)
    return order[np.where(choose_left, left, idx)]


def _in_interval(starts: np.ndarray, stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Boolean mask: is each timestamp inside any [start, stop] interval."""
    mask = np.zeros(len(t), dtype=bool)
    if len(starts) == 0:
        return mask
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], stops[order]
    pos = np.searchsorted(s, t, side="right") - 1
    valid = pos >= 0
    mask[valid] = t[valid] <= e[pos[valid]]
    return mask


def _interval_values(starts, stops, t, *value_arrays):
    """For each timestamp inside an interval, pull that interval's values."""
    out = [np.full(len(t), np.nan) for _ in value_arrays]
    if len(starts) == 0:
        return out
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], stops[order]
    vals = [np.asarray(v)[order] for v in value_arrays]
    pos = np.searchsorted(s, t, side="right") - 1
    valid = pos >= 0
    valid[valid] = t[valid] <= e[pos[valid]]
    for k, v in enumerate(vals):
        out[k][valid] = v[pos[valid]]
    return out


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def per_frame_metadata(recording, timestamps_ns: np.ndarray) -> pd.DataFrame:
    """Auxiliary sensor values sampled onto the given frame timestamps."""
    t = np.asarray(timestamps_ns, dtype=np.int64)
    out = pd.DataFrame(index=range(len(t)))

    # -- pupillometry (200 Hz) ------------------------------------------------
    pup = _safe(lambda: np.asarray(recording.pupil.data))
    if pup is not None and len(pup):
        i = _nearest_index(pup["time"], t)
        left = pup["diameter_left"][i].astype(float)
        right = pup["diameter_right"][i].astype(float)
        out["pupil_left_mm"] = np.round(left, 4)
        out["pupil_right_mm"] = np.round(right, 4)
        out["pupil_mean_mm"] = np.round(np.nanmean(np.vstack([left, right]), axis=0), 4)

    # -- eyelid aperture (200 Hz) --------------------------------------------
    lid = _safe(lambda: np.asarray(recording.eyelid.data))
    if lid is not None and len(lid):
        i = _nearest_index(lid["time"], t)
        out["eyelid_aperture_left_mm"] = np.round(lid["aperture_left"][i].astype(float), 4)
        out["eyelid_aperture_right_mm"] = np.round(lid["aperture_right"][i].astype(float), 4)

    # -- blinks / saccades (event intervals) ---------------------------------
    bl = _safe(lambda: np.asarray(recording.blinks.data))
    if bl is not None and len(bl):
        out["is_blink"] = _in_interval(bl["start_time"], bl["stop_time"], t)

    sac = _safe(lambda: np.asarray(recording.saccades.data))
    if sac is not None and len(sac):
        out["is_saccade"] = _in_interval(sac["start_time"], sac["stop_time"], t)
        amp, vel = _interval_values(
            sac["start_time"], sac["stop_time"], t,
            sac["amplitude_angle"], sac["max_velocity"])
        out["saccade_amplitude_deg"] = np.round(amp, 4)
        out["saccade_peak_velocity"] = np.round(vel, 4)

    # -- device worn flag (200 Hz) -------------------------------------------
    worn = _safe(lambda: np.asarray(recording.worn.data))
    if worn is not None and len(worn):
        i = _nearest_index(worn["time"], t)
        out["worn"] = worn["worn"][i].astype(bool)

    # -- head motion (IMU, ~110 Hz) ------------------------------------------
    imu = _safe(lambda: np.asarray(recording.imu.data))
    if imu is not None and len(imu):
        i = _nearest_index(imu["time"], t)
        gyro = np.vstack([imu["angular_velocity_x"][i], imu["angular_velocity_y"][i],
                          imu["angular_velocity_z"][i]]).astype(float)
        acc = np.vstack([imu["acceleration_x"][i], imu["acceleration_y"][i],
                         imu["acceleration_z"][i]]).astype(float)
        out["head_angular_velocity_dps"] = np.round(np.linalg.norm(gyro, axis=0), 4)
        out["head_acceleration_g"] = np.round(np.linalg.norm(acc, axis=0), 4)

    return out


def recording_metadata(recording, rec_name: str) -> dict:
    """Provenance and session-level summary statistics for one recording.

    The wearer display NAME is deliberately omitted (only the opaque UUID is
    kept) so these files stay safe to share alongside anonymised results.
    """
    info = _safe(lambda: dict(recording.info), {}) or {}
    dur_ns = _safe(lambda: int(recording.duration), 0) or 0
    dur_s = dur_ns / 1e9 if dur_ns else 0.0

    meta = {
        "recording": rec_name,
        "provenance": {
            "recording_id": info.get("recording_id"),
            "wearer_id": info.get("wearer_id"),
            "device_serial": str(_safe(lambda: recording.device_serial, "")),
            "start_time_ns": info.get("start_time"),
            "duration_s": round(dur_s, 3),
            "data_format_version": info.get("data_format_version"),
            "app_version": info.get("app_version"),
            "pipeline_version": info.get("pipeline_version"),
            "firmware_version": info.get("firmware_version"),
            "gaze_frequency_hz": info.get("gaze_frequency"),
            "gaze_mode": info.get("gaze_mode"),
            "gaze_offset": info.get("gaze_offset"),
            "wearer_ied_mm": info.get("wearer_ied"),
            "frame_name": info.get("frame_name"),
            "android_device_model": info.get("android_device_model"),
        },
        "streams": {},
        "summary": {},
    }

    def _count(name):
        a = _safe(lambda: np.asarray(getattr(recording, name).data))
        return int(len(a)) if a is not None else 0

    for s in ["gaze", "fixations", "saccades", "blinks", "pupil", "eyelid",
              "eyeball", "imu", "worn", "events", "scene"]:
        meta["streams"][s] = _count(s)

    s = meta["summary"]
    bl = _safe(lambda: np.asarray(recording.blinks.data))
    if bl is not None and len(bl):
        d_ms = (bl["stop_time"] - bl["start_time"]) / 1e6
        s["blink_count"] = int(len(bl))
        s["blink_rate_per_min"] = round(len(bl) / (dur_s / 60), 3) if dur_s else None
        s["blink_duration_ms_mean"] = round(float(np.mean(d_ms)), 2)

    sac = _safe(lambda: np.asarray(recording.saccades.data))
    if sac is not None and len(sac):
        s["saccade_count"] = int(len(sac))
        s["saccade_rate_per_min"] = round(len(sac) / (dur_s / 60), 3) if dur_s else None
        s["saccade_amplitude_deg_mean"] = round(float(np.nanmean(sac["amplitude_angle"])), 3)
        s["saccade_peak_velocity_mean"] = round(float(np.nanmean(sac["max_velocity"])), 3)

    pup = _safe(lambda: np.asarray(recording.pupil.data))
    if pup is not None and len(pup):
        both = np.vstack([pup["diameter_left"], pup["diameter_right"]]).astype(float)
        m = np.nanmean(both, axis=0)
        s["pupil_diameter_mm_mean"] = round(float(np.nanmean(m)), 4)
        s["pupil_diameter_mm_sd"] = round(float(np.nanstd(m)), 4)

    worn = _safe(lambda: np.asarray(recording.worn.data))
    if worn is not None and len(worn):
        s["worn_pct"] = round(100.0 * float(np.mean(worn["worn"].astype(bool))), 2)

    imu = _safe(lambda: np.asarray(recording.imu.data))
    if imu is not None and len(imu):
        g = np.linalg.norm(np.vstack([imu["angular_velocity_x"], imu["angular_velocity_y"],
                                      imu["angular_velocity_z"]]).astype(float), axis=0)
        s["head_angular_velocity_dps_mean"] = round(float(np.nanmean(g)), 3)
        s["head_angular_velocity_dps_p95"] = round(float(np.nanpercentile(g, 95)), 3)

    ev = _safe(lambda: np.asarray(recording.events.data))
    if ev is not None and len(ev):
        meta["events"] = [{"time_ns": int(r["time"]), "name": str(r["event"])} for r in ev]

    return meta


def augment_recording(rec_dir: pathlib.Path, *, force: bool = False,
                      recording=None, log=None,
                      csv_path: pathlib.Path | None = None) -> dict:
    """Append auxiliary per-frame columns to a recording analysis.csv and write
    recording_metadata.json next to it. Idempotent unless force=True.

    Pass csv_path explicitly when the analysis was written to a custom output
    directory, otherwise the recording's default aoi_results copy is used.
    """
    rec_dir = pathlib.Path(rec_dir)
    if csv_path is not None:
        csv_path = pathlib.Path(csv_path)
        if not (csv_path.exists() and csv_path.stat().st_size > 0):
            return {"recording": rec_dir.name, "status": "no analysis.csv"}
    else:
        for cand in [rec_dir / "aoi_results" / "analysis.csv",
                     rec_dir / "aoi_results" / "raw" / "analysis.csv"]:
            if cand.exists() and cand.stat().st_size > 0:
                csv_path = cand
                break
    if csv_path is None:
        return {"recording": rec_dir.name, "status": "no analysis.csv"}

    df = pd.read_csv(csv_path)
    already = [c for c in PER_FRAME_COLUMNS if c in df.columns]
    if already and not force:
        return {"recording": rec_dir.name, "status": "already augmented"}

    if recording is None:
        recording = nr.load(str(rec_dir))

    meta = recording_metadata(recording, rec_dir.name)
    (csv_path.parent / "recording_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    if "timestamp_ns" not in df.columns:
        return {"recording": rec_dir.name,
                "status": "metadata only (no timestamp_ns)", "metadata": True}

    t = pd.to_numeric(df["timestamp_ns"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    aux = per_frame_metadata(recording, t)
    df = df.drop(columns=[c for c in PER_FRAME_COLUMNS if c in df.columns], errors="ignore")
    for col in aux.columns:
        df[col] = aux[col].to_numpy()

    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    if log:
        log.info("  Metadata: +%d per-frame columns, recording_metadata.json written",
                 len(aux.columns))
    return {"recording": rec_dir.name, "status": "ok",
            "columns_added": list(aux.columns), "rows": len(df)}
