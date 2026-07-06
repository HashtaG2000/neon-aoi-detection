"""
AOI Studio — Neon Player-inspired review tool
==============================================
Scene video + live AprilTag surface overlays + gaze dot.
Two tabs: Studio (video player + task annotation) and Dashboard (Plotly charts).
No validation video, no gamification toggle, no watcher.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pupil_apriltags
from PySide6.QtCore import (
    QObject, QRect, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontDatabase, QIcon, QImage, QKeySequence,
    QPainter, QPen, QPixmap,
)

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QVBoxLayout, QWidget,
)
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from paths import (
    AOIS_CONFIG_FILE, CONFIG_DIR, FONTS_DIR, RECORDINGS_DIR, SRC_DIR,
    ensure_vendor_paths, list_source_folders, resolve_app_icon,
)

ensure_vendor_paths()

# Custom Plotly theme matching the app's dark UI (replaces the stock "plotly_dark").
pio.templates["aoi_studio"] = pio.templates["plotly_dark"]
pio.templates["aoi_studio"].layout.update(
    paper_bgcolor="#0e0f11",
    plot_bgcolor="#101113",
    font=dict(family="Hanken Grotesk, IBM Plex Mono, sans-serif", color="#cdd0d6", size=12),
    colorway=["#d4a24a", "#6fae7d", "#c07ba8", "#d68a55", "#5ea9b3", "#7d86c9", "#6e8fd6"],
    title=dict(font=dict(color="#e7e8ea", size=15)),
    xaxis=dict(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.08)"),
    legend=dict(font=dict(color="#cdd0d6")),
)

import pupil_labs.neon_recording as nr
import reporting


def _svg_icon(name: str) -> QIcon:
    """Load an SVG from the assets folder as a QIcon."""
    return QIcon(str(SRC_DIR / "assets" / name))

APP_TITLE = "AOI Studio"

_COND_COLORS      = ["#6e8fd6", "#6b6e74"]           # A = accent, B = neutral grey
_COND_FILL_COLORS = ["rgba(110,143,214,0.15)", "rgba(107,110,116,0.15)"]

# ─── AOI palette ─────────────────────────────────────────────────────────────

def _load_aoi_config() -> tuple[dict[str, list[int]], dict[str, tuple]]:
    data = json.loads(AOIS_CONFIG_FILE.read_text(encoding="utf-8"))
    aois   = {k: list(v) for k, v in data["aois"].items()}
    colors = {k: tuple(v) for k, v in data["colors"].items()}
    return aois, colors

AOI_CONFIG, _CV_COLORS = _load_aoi_config()

# Qt colors (for timeline + UI)
AOI_COLORS_QT: dict[str, QColor] = {
    name: QColor(r, g, b)
    for name, (b, g, r) in _CV_COLORS.items()   # aois.json stores BGR
}
# cv2 colors (for drawing on frame)
AOI_COLORS_CV: dict[str, tuple] = _CV_COLORS     # already BGR

NONE_LABEL = "NoAOI"
AOI_NAMES  = list(AOI_CONFIG.keys())
EDITABLE_AOIS = AOI_NAMES + [NONE_LABEL]
GAP_FILL_MAX_FRAMES = 15

# ─── Theme tokens (from the Claude Design "AOI Studio v2" handoff) ────────────

class Theme:
    BG_BASE        = "#0e0f11"
    BG_PANEL       = "#121315"   # sidebars
    BG_FIELD       = "#131416"   # filled buttons / inputs
    BG_CARD        = "#101113"   # tables / chart containers
    BORDER         = "rgba(255,255,255,.10)"
    BORDER_SUBTLE  = "rgba(255,255,255,.07)"
    BORDER_FAINT   = "rgba(255,255,255,.04)"
    TEXT           = "#e7e8ea"
    TEXT_BRIGHT    = "#f1f2f4"
    TEXT_SECONDARY = "#cdd0d6"
    TEXT_MUTED     = "#9b9ea4"
    TEXT_DIM       = "#83868c"
    TEXT_FAINT     = "#6a6d73"
    TEXT_VFAINT    = "#62656b"
    ACCENT         = "#6e8fd6"
    SUCCESS        = "#6fae7d"
    WARNING        = "#d4a24a"
    DANGER         = "#cf6b6b"
    RADIUS         = 8
    FONT_UI        = "Hanken Grotesk"
    FONT_MONO      = "IBM Plex Mono"


def _load_app_fonts() -> None:
    """Register the bundled Hanken Grotesk / IBM Plex Mono weights with Qt."""
    if not FONTS_DIR.exists():
        return
    for ttf in sorted(FONTS_DIR.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(ttf))

# ─── Detection helpers ────────────────────────────────────────────────────────

# Bug 10 fix: CLAHE removed from live detection. The analyzer uses raw frames
# (no CLAHE) so the live overlay must match to avoid showing different results.
# The old _clahe / _enhance function caused the live view to detect surfaces
# that the analyzer did not (or vice versa).

def _enhance(gray: np.ndarray) -> np.ndarray:
    return gray  # identity — no pre-processing, same as analyzer

def _make_detector() -> pupil_apriltags.Detector:
    # Full-resolution detection (quad_decimate=1.0) to match the analyzer's
    # high-precision detector. Downscaling (2.0) dropped small/far tags — most
    # notably the screen tags — so the live overlay disagreed with the analysis.
    return pupil_apriltags.Detector(
        families="tag36h11", nthreads=4,
        quad_decimate=1.0, quad_sigma=0.0,
        refine_edges=1, decode_sharpening=0.5,
    )

def _surface_polygon(detections: list, marker_ids: list[int]) -> Optional[np.ndarray]:
    """Convex hull of all visible marker corners for this surface."""
    visible = [d for d in detections if d.tag_id in marker_ids]
    if not visible:
        return None
    corners = []
    for d in visible:
        corners.extend(d.corners)
    hull = cv2.convexHull(np.array(corners, dtype=np.float32))
    return hull.reshape(-1, 2)

def _bgr_to_pixmap(bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    h, w = bgr.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 0.99:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h2, w2 = rgb.shape[:2]
    qi = QImage(rgb.data, w2, h2, 3 * w2, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qi)

# ─── Worker signals ───────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    progress = Signal(int, int)   # (current_frame, total_frames)
    status   = Signal(str)
    finished = Signal()
    failed   = Signal(str)


class AnalysisWorker(threading.Thread):
    def __init__(
        self,
        rec_dir: pathlib.Path,
        trim: Optional[dict] = None,
        *,
        generate_video: bool = False,
        generation: int = 0,
    ) -> None:
        super().__init__(daemon=True)
        self.rec_dir = rec_dir
        self.trim = trim or {}
        self.generate_video = generate_video
        self.generation = generation
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            import analyzer
            self.signals.status.emit("Analyzing…")
            raw_dir = self.rec_dir / "aoi_results" / "raw"
            start_frame = self.trim.get("start_frame")
            end_frame   = self.trim.get("end_frame")
            trim_range  = (start_frame, end_frame) if start_frame is not None and end_frame is not None else None
            padding_s   = float(self.trim.get("padding_s", 0.5))
            analyzer.analyze_recording(
                self.rec_dir, raw_dir, generate_video=self.generate_video,
                trim_range=trim_range, trim_padding_s=padding_s,
            )
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchAnalysisWorker(threading.Thread):
    def __init__(self, source_dir: pathlib.Path, generation: int = 0) -> None:
        super().__init__(daemon=True)
        self.source_dir = source_dir
        self.generation = generation
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            import analyzer
            recordings = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])
            total = len(recordings)
            for idx, rec_dir in enumerate(recordings, 1):
                self.signals.status.emit(f"Analysing {idx}/{total}: {rec_dir.name}")
                raw_dir = rec_dir / "aoi_results" / "raw"
                # Force re-analysis by removing lock file
                lock = raw_dir / ".processing"
                lock.unlink(missing_ok=True)
                try:
                    analyzer.analyze_recording(rec_dir, raw_dir, generate_video=False)
                except Exception as e:
                    self.signals.status.emit(f"Failed {rec_dir.name}: {e}")
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class WatcherWorker(threading.Thread):
    """TEMPORARY: sweeps every recording under Recordings/ (both conditions) and
    analyses the ones without an analysis.csv yet, in the background."""
    def __init__(self, generation: int = 0) -> None:
        super().__init__(daemon=True)
        self.generation = generation
        self.signals = WorkerSignals()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            import analyzer
            conds = sorted(p for p in RECORDINGS_DIR.iterdir() if p.is_dir()) if RECORDINGS_DIR.exists() else []
            pending = []
            for cond in conds:
                for rec in sorted(p for p in cond.iterdir() if p.is_dir()):
                    if not (rec / "info.json").exists():
                        continue
                    done = ((rec / "aoi_results" / "raw" / "analysis.csv").exists()
                            or (rec / "aoi_results" / "analysis.csv").exists())
                    if not done:
                        pending.append(rec)
            total = len(pending)
            if total == 0:
                self.signals.status.emit("Watcher: everything is already analysed.")
                self.signals.finished.emit()
                return
            for i, rec in enumerate(pending, 1):
                if self._stop.is_set():
                    self.signals.status.emit("Watcher stopped.")
                    break
                self.signals.status.emit(f"Watcher {i}/{total}: {rec.name}")
                raw = rec / "aoi_results" / "raw"
                (raw / ".processing").unlink(missing_ok=True)
                try:
                    analyzer.analyze_recording(rec, raw, generate_video=False)
                except Exception as e:
                    self.signals.status.emit(f"Watcher failed {rec.name}: {e}")
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class ComparisonWorker(threading.Thread):
    def __init__(self, cond_a: pathlib.Path, cond_b: pathlib.Path) -> None:
        super().__init__(daemon=True)
        self.cond_a  = cond_a
        self.cond_b  = cond_b
        self.signals = WorkerSignals()
        self.result  = None   # ComparisonReport set on success

    def run(self) -> None:
        try:
            self.result = reporting.compare_conditions(self.cond_a, self.cond_b)
            self.signals.finished.emit()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ─── Video widget ─────────────────────────────────────────────────────────────

class VideoWidget(QLabel):
    """
    Center piece of the Studio tab.
    Renders the scene video frame with:
      - Live AprilTag surface-boundary polygons
      - Gaze dot (from recording.gaze)
    """
    frameChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background: #14151a; border: 1px solid rgba(255,255,255,.06); "
            f"border-radius: {Theme.RADIUS}px; color: {Theme.TEXT_VFAINT}; "
            f"font-family: '{Theme.FONT_MONO}', monospace;"
        )
        self.setText("scene camera + gaze overlay")

        self._recording: Optional[nr.NeonRecording] = None
        self._scene_ts: Optional[np.ndarray] = None
        self._n_frames: int = 0
        self._frame_idx: int = 0
        self._detector = _make_detector()
        self._show_surfaces: bool = True
        self._show_gaze: bool = True
        self._playing: bool = False
        self._cap: Optional[cv2.VideoCapture] = None   # fast sequential reader

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, recording: nr.NeonRecording, rec_dir: pathlib.Path) -> None:
        self._recording = recording
        self._scene_ts  = recording.scene.time
        self._n_frames  = len(self._scene_ts)
        self._frame_idx = 0
        self._open_cap(rec_dir)
        self.render()

    def _open_cap(self, rec_dir: pathlib.Path) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        scene_candidates = sorted(rec_dir.glob("*Scene Camera*.mp4"))
        other_candidates = sorted(p for p in rec_dir.glob("*.mp4") if p not in scene_candidates)
        for p in scene_candidates + other_candidates:
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                self._cap = cap
                return

    def unload(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._recording = None
        self._scene_ts  = None
        self._n_frames  = 0
        self._frame_idx = 0
        self.clear()
        self.setText("Load a recording to begin.")

    def seek(self, idx: int) -> None:
        if self._n_frames == 0:
            return
        self._frame_idx = max(0, min(idx, self._n_frames - 1))
        # Force cap to correct position on explicit seeks
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
        self.render()
        self.frameChanged.emit(self._frame_idx)

    def play_step(self) -> bool:
        """Advance one frame during playback. Returns False if at end."""
        if self._frame_idx >= self._n_frames - 1:
            return False
        self._frame_idx += 1
        self.render()
        self.frameChanged.emit(self._frame_idx)
        return True

    def step(self, delta: int) -> None:
        self.seek(self._frame_idx + delta)

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def set_playing(self, playing: bool) -> None:
        self._playing = playing

    # ── Rendering ───────────────────────────────────────────────────────────

    def render(self) -> None:
        if self._n_frames == 0:
            return
        try:
            bgr = self._read_frame(self._frame_idx)
            if bgr is None:
                return
            ts = self._scene_ts[self._frame_idx]
            if self._show_surfaces:
                self._draw_surfaces(bgr)
            if not self._playing:
                self._draw_fixation_scanpath(bgr, ts)
            self._draw_gaze(bgr, ts)
            self._draw_hud(bgr)
            pixmap = _bgr_to_pixmap(bgr, self.width() - 4, self.height() - 4)
            self.setPixmap(pixmap)
        except Exception as exc:
            self.setText(f"Render error: {exc}")

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Fast sequential read during playback; seek only when needed."""
        if self._cap is not None:
            if self._playing:
                # During playback: just read the next frame sequentially — never seek
                ret, frame = self._cap.read()
                if ret:
                    return frame
            else:
                cap_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cap_pos != idx:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self._cap.read()
                if ret:
                    return frame
        # Fallback to NeonRecording (slower)
        if self._recording is not None:
            ts    = self._scene_ts[idx]
            frame = self._recording.scene.sample([ts], method="backward")[0]
            return np.array(frame.bgr, copy=True)
        return None

    def _draw_surfaces(self, bgr: np.ndarray) -> None:
        if not self._show_surfaces or self._playing:
            return
        gray       = _enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        detections = self._detector.detect(gray)
        for name, ids in AOI_CONFIG.items():
            poly = _surface_polygon(detections, ids)
            if poly is None:
                continue
            color = AOI_COLORS_CV.get(name, (180, 180, 180))
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(name, font, 0.50, 1)
            cv2.rectangle(bgr, (cx - 3, cy - th - 5), (cx + tw + 3, cy + 3), (0, 0, 0), -1)
            cv2.putText(bgr, name, (cx, cy), font, 0.50, color, 1, cv2.LINE_AA)

    def _draw_gaze(self, bgr: np.ndarray, ts_ns: int) -> None:
        if not self._show_gaze:
            return
        try:
            g  = self._recording.gaze.sample([ts_ns], method="nearest")[0]
            gx, gy = float(g.point_x), float(g.point_y)
            if not (np.isfinite(gx) and np.isfinite(gy)):
                return
            px, py = int(gx), int(gy)
            cv2.circle(bgr, (px, py), 14, (0, 0, 200), -1, cv2.LINE_AA)
            cv2.circle(bgr, (px, py), 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(bgr, (px, py),  4, (255, 255, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_fixation_scanpath(self, bgr: np.ndarray, ts_ns: int) -> None:
        """Draw scanpath (connected circles) and active fixation circle.
        Ported from Neon Player's ScanpathViz + FixationCircleViz.
        Only called when paused — not during playback.
        """
        if self._recording is None:
            return
        try:
            fd      = self._recording.fixations.data
            starts  = np.asarray(fd["start_time"],  dtype=np.int64)
            stops   = np.asarray(fd["stop_time"],   dtype=np.int64)
            mean_x  = np.asarray(fd["mean_gaze_x"], dtype=np.float64)
            mean_y  = np.asarray(fd["mean_gaze_y"], dtype=np.float64)
        except Exception:
            return

        # Last 7 completed fixations + current active fixation
        history_ns  = 30_000_000_000  # 30s history window
        past_mask   = (stops  <= ts_ns) & (stops  >= ts_ns - history_ns)
        active_mask = (starts <= ts_ns) & (stops  >  ts_ns)
        past_idx    = np.where(past_mask)[0]
        if len(past_idx) > 7:
            past_idx = past_idx[-7:]
        active_idx  = np.where(active_mask)[0]
        render_idx  = np.sort(np.concatenate([past_idx, active_idx]))

        if len(render_idx) == 0:
            return

        prev_pt = None
        for idx in render_idx:
            cx  = int(mean_x[idx])
            cy  = int(mean_y[idx])
            dur_ms  = (int(stops[idx]) - int(starts[idx])) / 1e6
            radius  = min(80, max(10, int(10 * dur_ms / 100.0)))
            is_active = len(active_idx) > 0 and idx == active_idx[0]

            # Scanpath line between successive fixations
            if prev_pt is not None:
                cv2.line(bgr, prev_pt, (cx, cy), (90, 104, 110), 2, cv2.LINE_AA)
            prev_pt = (cx, cy)

            # Circle: filled amber if active, outlined grey if past
            if is_active:
                cv2.circle(bgr, (cx, cy), radius, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(bgr, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(bgr, (cx, cy), radius, (90, 104, 110), 2, cv2.LINE_AA)

            # Fixation ID label (white, small)
            label = str(int(idx) + 1)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(bgr, label, (cx - tw // 2, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        ts_s = (self._scene_ts[self._frame_idx] - self._scene_ts[0]) / 1e9 if self._n_frames else 0
        label = f"{self._frame_idx:05d}  {ts_s:6.2f}s"
        cv2.putText(bgr, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Bug 9 fix: debounce renders during drag-resize.  Without this,
        # every pixel of window movement triggers a full render (including
        # AprilTag detection), making the UI feel sluggish.
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(100)  # ms
            self._resize_timer.timeout.connect(self.render)
        self._resize_timer.start()


# ─── Playback controls ────────────────────────────────────────────────────────

class PlaybackBar(QWidget):
    seeked = Signal(int)   # absolute frame index

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(40)
        self._n_frames = 0
        self._fps      = 30.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        # Step back 30 frames (~1 s)
        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(_svg_icon("arrow_back.svg"))
        self.prev_btn.setIconSize(QSize(16, 16))
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.setObjectName("iconButton")
        self.prev_btn.setToolTip("Step back 1 second")

        self.play_btn = QPushButton()
        self.play_btn.setIcon(_svg_icon("playbutton.svg"))
        self.play_btn.setIconSize(QSize(18, 18))
        self.play_btn.setFixedWidth(40)
        self.play_btn.setObjectName("primaryButton")

        # Step forward 30 frames (~1 s)
        self.next_btn = QPushButton()
        self.next_btn.setIcon(_svg_icon("arrow_forward.svg"))
        self.next_btn.setIconSize(QSize(16, 16))
        self.next_btn.setFixedWidth(36)
        self.next_btn.setObjectName("iconButton")
        self.next_btn.setToolTip("Step forward 1 second")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)

        self.time_label = QLabel("0:00:00 / 0:00:00")
        self.time_label.setFixedWidth(120)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 13px; font-family: '{Theme.FONT_MONO}', monospace;")

        layout.addWidget(self.prev_btn)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.time_label)

        self.slider.valueChanged.connect(self.seeked)

    def configure(self, n_frames: int, fps: float) -> None:
        self._n_frames = n_frames
        self._fps = max(fps, 1.0)
        self.slider.setRange(0, max(0, n_frames - 1))
        self._update_time(0)

    def set_frame(self, idx: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._update_time(idx)

    def set_playing(self, playing: bool) -> None:
        self.play_btn.setIcon(_svg_icon("pause.svg" if playing else "playbutton.svg"))

    def _update_time(self, idx: int) -> None:
        def fmt(s: float) -> str:
            h = int(s) // 3600
            m = (int(s) % 3600) // 60
            sec = int(s) % 60
            return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
        cur = idx / self._fps
        tot = self._n_frames / self._fps
        self.time_label.setText(f"{fmt(cur)} / {fmt(tot)}")


# ─── Combined timeline (AOI + gaze + fixations + tasks) ──────────────────────

class AOITimeline(QWidget):
    """
    Stacked timeline rows (Neon Player style):
      Row 1 – AOI label colour bar (coloured segments per AOI)
      Row 2 – Gaze presence strip  (white = gaze detected, dark = no gaze/blink)
      Row 3 – Fixation markers     (bright dots where fixations occur)
      Row 4 – Task spans           (coloured bands per task)
    """
    frameClicked = Signal(int)

    _ROW_AOI      = 18
    _ROW_GAZE     = 10
    _ROW_FIX      = 10
    _ROW_TASKS    = 20
    _PAD          = 0   # no gap between AOI / Gaze / Fix / Tasks lanes
    _LEFT_MARGIN  = 44   # pixels reserved on the left for row labels

    def __init__(self) -> None:
        super().__init__()
        h = (self._ROW_AOI + self._ROW_GAZE + self._ROW_FIX +
             self._ROW_TASKS + self._PAD * 3)
        self.setMinimumHeight(h)
        self.resize(self.width(), h)
        self.setMouseTracking(True)

        self._n_frames:    int        = 0
        self._labels:      list[str]  = []
        self._gaze_x:      np.ndarray = np.array([])
        self._gaze_y:      np.ndarray = np.array([])
        self._fix_frames:  list[int]  = []
        self._frame_idx:   int        = 0
        self._tasks:       dict       = {}

        # Zoom/pan: the visible frame window. (0, 0) means "not set yet" —
        # _visible_range() falls back to the full recording.
        self._view_start:      int   = 0
        self._view_end:        int   = 0
        self._drag_active:     bool  = False
        self._drag_start_x:    float = 0.0
        self._drag_start_view: tuple[int, int] = (0, 0)

        # Static-content cache — rebuilt only when data changes, not on every tick
        self._cache:       Optional[QPixmap] = None
        self._cache_dirty: bool              = True

    # ── Public API ──────────────────────────────────────────────────────────

    def set_recording_length(self, n: int) -> None:
        self._n_frames    = n
        self._view_start  = 0
        self._view_end    = n
        self._cache_dirty = True
        self.update()

    def set_analysis(self, labels: list[str],
                     gaze_x: np.ndarray, gaze_y: np.ndarray,
                     fix_frames: list[int]) -> None:
        self._labels      = labels
        self._gaze_x      = gaze_x
        self._gaze_y      = gaze_y
        self._fix_frames  = fix_frames
        self._cache_dirty = True
        self.update()

    def set_tasks(self, tasks: dict) -> None:
        self._tasks       = tasks
        self._cache_dirty = True
        self.update()

    def set_frame(self, idx: int) -> None:
        self._frame_idx = idx
        # Auto-follow: if zoomed in and the playhead leaves the visible
        # window (e.g. during playback), slide the window to keep it in view.
        vs, ve = self._visible_range()
        width = ve - vs
        if width < self._n_frames and not (vs <= idx <= ve):
            new_start = max(0, idx) if idx < vs else min(self._n_frames - width, idx - width)
            self._view_start = int(new_start)
            self._view_end   = int(new_start + width)
            self._cache_dirty = True
        self.update()   # fast: blit cache + draw playhead only

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._cache_dirty = True
        self.update()

    # ── Zoom / pan ───────────────────────────────────────────────────────────

    def _visible_range(self) -> tuple[int, int]:
        if self._view_end <= self._view_start:
            return 0, max(1, self._n_frames)
        return self._view_start, self._view_end

    def wheelEvent(self, event) -> None:
        if self._n_frames <= 0:
            return
        lx = self._LEFT_MARGIN
        tw = max(1, self.width() - lx)
        vs, ve = self._visible_range()
        view_w = ve - vs
        ratio = max(0.0, min(1.0, (event.position().x() - lx) / tw))
        frame_at_cursor = vs + ratio * view_w

        factor = 1.25
        new_width = view_w / factor if event.angleDelta().y() > 0 else view_w * factor
        min_width = min(60, self._n_frames)   # don't zoom in past ~2s
        new_width = max(min_width, min(self._n_frames, new_width))

        new_start = max(0, min(self._n_frames - new_width, frame_at_cursor - ratio * new_width))
        self._view_start = int(round(new_start))
        self._view_end   = int(round(new_start + new_width))
        self._cache_dirty = True
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self._view_start = 0
        self._view_end   = self._n_frames
        self._cache_dirty = True
        self.update()

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        w = self.width()
        h = self.height()

        # Rebuild the static cache if data changed or widget was resized
        if self._cache_dirty or self._cache is None:
            self._cache = self._build_cache(w, h)
            self._cache_dirty = False

        painter = QPainter(self)
        # Blit static cache (fast GPU copy — does NOT iterate over frames)
        painter.drawPixmap(0, 0, self._cache)

        # Draw moving playhead only inside the data area (right of label margin),
        # and only when it falls within the currently zoomed/panned view.
        lx = self._LEFT_MARGIN
        tw = max(1, w - lx)
        vs, ve = self._visible_range()
        view_w = max(1, ve - vs)
        if vs <= self._frame_idx <= ve:
            cx = lx + int((self._frame_idx - vs) / view_w * tw)
            painter.setPen(QPen(QColor(Theme.DANGER), 1))
            painter.drawLine(cx, 0, cx, h)

    def _build_cache(self, w: int, h: int) -> QPixmap:
        """Render all static rows into a QPixmap (called once per data change)."""
        px = QPixmap(max(w, 1), max(h, 1))
        px.fill(QColor(Theme.BG_BASE))
        if w <= 0 or h <= 0:
            return px

        painter = QPainter(px)
        painter.setRenderHint(QPainter.Antialiasing)

        lx    = self._LEFT_MARGIN          # left edge of data area
        tw    = max(1, w - lx)             # width of data area
        vs, ve = self._visible_range()
        view_w = max(1, ve - vs)

        def fx(frame_i: float) -> int:
            """Map a frame index to an x pixel through the current zoom/pan window."""
            return lx + int((frame_i - vs) / view_w * tw)

        p       = self._PAD

        # Row heights scale with the widget's actual height (resizable via the
        # Studio splitter) while keeping the original AOI:Gaze:Fix:Tasks ratio.
        weights  = (self._ROW_AOI, self._ROW_GAZE, self._ROW_FIX, self._ROW_TASKS)
        avail    = max(4 * 6, h - p * 3)
        total_w  = sum(weights)
        row_aoi, row_gaze, row_fix = (max(6, int(avail * wt / total_w)) for wt in weights[:3])
        row_task = max(8, avail - row_aoi - row_gaze - row_fix)

        y0_aoi  = 0
        y0_gaze = y0_aoi  + row_aoi  + p
        y0_fix  = y0_gaze + row_gaze + p
        y0_task = y0_fix  + row_fix  + p

        # ── Row labels (left margin) ─────────────────────────────────────────
        label_font = QFont(Theme.FONT_UI, 7)
        label_color = QColor("#73767c")
        painter.setFont(label_font)
        painter.setPen(label_color)
        lm = lx - 3  # right-align text just before the data area
        for text, y0, row_h in [
            ("AOI",   y0_aoi,  row_aoi),
            ("Gaze",  y0_gaze, row_gaze),
            ("Fix",   y0_fix,  row_fix),
            ("Tasks", y0_task, row_task),
        ]:
            painter.drawText(QRect(0, y0, lm, row_h), Qt.AlignVCenter | Qt.AlignRight, text)

        # Thin vertical separator between labels and data
        painter.setPen(QColor(255, 255, 255, 18))
        painter.drawLine(lx - 1, 0, lx - 1, h)

        if not self._labels:
            painter.setFont(QFont(Theme.FONT_UI, 7))
            painter.setPen(QColor(Theme.TEXT_VFAINT))
            painter.drawText(lx + 4, y0_aoi + 13, "Run Analysis to see AOI timeline")

        # Row 1: AOI colour bars — run-length encoded to draw one rect per
        # contiguous segment instead of one per frame (Bug 6 fix).
        if self._labels:
            n_labels = len(self._labels)
            i = 0
            while i < n_labels:
                lbl = self._labels[i]
                color = AOI_COLORS_QT.get(lbl)
                if color is None:
                    i += 1
                    continue
                # Find the end of this contiguous run
                j = i + 1
                while j < n_labels and self._labels[j] == lbl:
                    j += 1
                x1 = fx(i)
                x2 = max(x1 + 1, fx(j))
                painter.fillRect(x1, y0_aoi, x2 - x1, row_aoi, color)
                i = j

        # Row 2: Gaze presence strip — batch-computed with numpy instead of
        # a per-point Python loop (Bug 5 fix).  Pre-computes all pixel
        # coordinates, then draws only the visible range.
        painter.fillRect(lx, y0_gaze, tw, row_gaze, QColor("#0d0e10"))
        if len(self._gaze_x):
            scene_h = 1200.0
            gaze_color = QColor(Theme.ACCENT)
            # Only process the visible frame range (zoom-aware)
            lo = max(0, vs)
            hi = min(len(self._gaze_x), ve + 1)
            if hi > lo:
                gx_slice = self._gaze_x[lo:hi]
                gy_slice = self._gaze_y[lo:hi]
                valid = np.isfinite(gx_slice) & np.isfinite(gy_slice)
                if np.any(valid):
                    indices = np.where(valid)[0]
                    frame_indices = indices + lo
                    # Vectorised pixel-coordinate computation
                    x_coords = lx + ((frame_indices - vs) * tw / view_w).astype(int)
                    norm_y = np.clip(gy_slice[indices] / scene_h, 0.0, 1.0)
                    y_coords = (y0_gaze + norm_y * row_gaze).astype(int)
                    for xi, yi in zip(x_coords, y_coords):
                        painter.fillRect(int(xi), int(yi), 1, 1, gaze_color)

        # Row 3: Fixation markers
        painter.fillRect(lx, y0_fix, tw, row_fix, QColor("#0d0e10"))
        fix_color = QColor("#d68a55")
        for fi in self._fix_frames:
            x = fx(fi)
            painter.fillRect(x, y0_fix, 2, row_fix, fix_color)

        # Row 4: Task spans
        painter.fillRect(lx, y0_task, tw, row_task, QColor(Theme.BG_BASE))
        accent = QColor(Theme.ACCENT)
        for i, (name, tdata) in enumerate(self._tasks.items()):
            s, e = tdata.get("start"), tdata.get("end")
            if s is None or e is None or e <= s:
                continue
            color = QColor(accent.red(), accent.green(), accent.blue(), 90 if i % 2 == 0 else 60)
            x1 = fx(s)
            x2 = max(x1 + 2, fx(e))
            painter.fillRect(x1, y0_task, x2 - x1, row_task, color)
            painter.setPen(QColor(Theme.TEXT_SECONDARY))
            painter.setFont(QFont(Theme.FONT_MONO, 7))
            painter.drawText(QRect(x1 + 2, y0_task, x2 - x1 - 2, row_task),
                             Qt.AlignVCenter | Qt.AlignLeft, name.replace("Task ", "T"))

        # Row dividers
        painter.setPen(QColor(255, 255, 255, 18))
        for y in (y0_gaze - 1, y0_fix - 1, y0_task - 1):
            painter.drawLine(lx, y, w, y)

        painter.end()
        return px

    def mousePressEvent(self, event) -> None:
        self._drag_start_x    = event.position().x()
        self._drag_start_view = self._visible_range()
        self._drag_active     = False

    def mouseMoveEvent(self, event) -> None:
        if not (event.buttons() & Qt.LeftButton):
            return
        dx = event.position().x() - self._drag_start_x
        if not self._drag_active and abs(dx) > 4:
            self._drag_active = True
        if not self._drag_active:
            return
        lx = self._LEFT_MARGIN
        tw = max(1, self.width() - lx)
        vs0, ve0 = self._drag_start_view
        view_w = ve0 - vs0
        new_start = max(0, min(self._n_frames - view_w, vs0 - dx / tw * view_w))
        self._view_start = int(round(new_start))
        self._view_end   = int(round(new_start + view_w))
        self._cache_dirty = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._drag_active:
            lx     = self._LEFT_MARGIN
            tw     = max(1, self.width() - lx)
            vs, ve = self._visible_range()
            view_w = max(1, ve - vs)
            click_x = max(0.0, event.position().x() - lx)
            idx     = int(vs + click_x / tw * view_w)
            self.frameClicked.emit(max(0, min(idx, self._n_frames - 1)))
        self._drag_active = False


# ─── Left panel widgets ───────────────────────────────────────────────────────

class TrimPanel(QWidget):
    """Optional [in, out] frame range to exclude dead time (setup/calibration) from analysis."""
    trimChanged = Signal(object, object)  # (start_frame | None, end_frame | None)
    cropDataRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._current_frame = 0
        self._undo_stack: list[tuple] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._in_btn = QPushButton("In")
        self._out_btn = QPushButton("Out")
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("iconButton")
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setObjectName("iconButton")
        self._in_btn.setToolTip("Mark trim start at the current frame")
        self._out_btn.setToolTip("Mark trim end at the current frame")
        self._clear_btn.setToolTip("Remove trim — analyse the full recording")
        self._undo_btn.setToolTip("Undo the last trim change")
        btn_row.addWidget(self._in_btn)
        btn_row.addWidget(self._out_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._undo_btn)
        layout.addLayout(btn_row)

        self._range_lbl = QLabel("Full recording (no trim)")
        self._range_lbl.setWordWrap(True)
        self._range_lbl.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-size: 12px;")
        layout.addWidget(self._range_lbl)
        
        self._crop_btn = QPushButton("✂ Crop")
        self._crop_btn.setToolTip("Instantly remove all data outside the trimmed range from analysis.csv without re-running the slow analysis")
        self._crop_btn.setObjectName("secondaryButton")
        layout.addWidget(self._crop_btn)

        pad_row = QHBoxLayout()
        pad_row.setSpacing(6)
        pad_lbl = QLabel("Padding (s):")
        pad_lbl.setStyleSheet(f"font-size: 13px; color: {Theme.TEXT_MUTED};")
        self._pad_spin = QDoubleSpinBox()
        self._pad_spin.setRange(0.0, 5.0)
        self._pad_spin.setSingleStep(0.1)
        self._pad_spin.setValue(0.5)
        self._pad_spin.setToolTip("Extra seconds kept on each side of the trimmed range, so a fixation right at the cut isn't sliced off")
        pad_row.addWidget(pad_lbl)
        pad_row.addWidget(self._pad_spin, 1)
        layout.addLayout(pad_row)

        self._in_btn.clicked.connect(self._set_in)
        self._out_btn.clicked.connect(self._set_out)
        self._clear_btn.clicked.connect(self._clear)
        self._undo_btn.clicked.connect(self._undo)
        self._pad_spin.valueChanged.connect(lambda _: self._emit_changed())

    def set_current_frame(self, idx: int) -> None:
        self._current_frame = idx

    def _push_undo(self) -> None:
        self._undo_stack.append((self._start, self._end))

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self._start, self._end = self._undo_stack.pop()
        self._refresh()

    def _set_in(self) -> None:
        self._push_undo()
        self._start = self._current_frame
        if self._end is not None and self._end < self._start:
            self._end = None
        self._refresh()

    def _set_out(self) -> None:
        self._push_undo()
        self._end = self._current_frame
        if self._start is not None and self._start > self._end:
            self._start = None
        self._refresh()

    def _clear(self) -> None:
        self._push_undo()
        self._start = None
        self._end = None
        self._refresh()

    def _refresh(self) -> None:
        if self._start is None and self._end is None:
            self._range_lbl.setText("Full recording (no trim)")
            self._range_lbl.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-size: 12px;")
        elif self._start is not None and self._end is not None:
            self._range_lbl.setText(
                f"✓ Active — frame {self._start} → {self._end}  "
                f"(+{self._pad_spin.value():.1f}s padding)\nWill apply on next Analyse"
            )
            self._range_lbl.setStyleSheet(f"color: {Theme.SUCCESS}; font-size: 12px; font-weight: 600;")
        else:
            s = self._start if self._start is not None else "start"
            e = self._end if self._end is not None else "end"
            self._range_lbl.setText(f"Trim: frame {s} → {e} — set the other bound to activate")
            self._range_lbl.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-size: 12px;")
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.trimChanged.emit(self._start, self._end)

    def get_trim(self) -> dict:
        return {"start_frame": self._start, "end_frame": self._end, "padding_s": self._pad_spin.value()}

    def load_trim(self, data: dict) -> None:
        self._start = data.get("start_frame")
        self._end = data.get("end_frame")
        if "padding_s" in data:
            self._pad_spin.setValue(float(data["padding_s"]))
        self._refresh()

    def reset(self) -> None:
        self._start = None
        self._end = None
        self._pad_spin.setValue(0.5)
        self._refresh()


class CorrectionPanel(QWidget):
    """Manual AOI relabelling for frames or a marked range."""
    correctionApplied = Signal(bool)  # True = apply marked range, False = current frame only
    correctionSaved = Signal()
    undoRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._current_frame = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._aoi_combo = QComboBox()
        for name in EDITABLE_AOIS:
            self._aoi_combo.addItem(name.replace("_", " "), userData=name)
        layout.addWidget(self._aoi_combo)

        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(6)
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)
        
        self._frame_btn = QPushButton("Frame")
        self._in_btn = QPushButton("In")
        self._out_btn = QPushButton("Out")
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("primaryButton")
        self._save_btn = QPushButton("Save")
        self._undo_btn = QPushButton("Undo")

        self._frame_btn.setToolTip("Apply the selected AOI to the current frame")
        self._in_btn.setToolTip("Mark range start at the current frame")
        self._out_btn.setToolTip("Mark range end at the current frame")
        self._apply_btn.setToolTip("Apply the selected AOI to every frame in the marked range")
        self._save_btn.setToolTip("Save corrections back to analysis.csv (Updates dashboard)")
        self._undo_btn.setToolTip("Undo the last correction")

        btn_row1.addWidget(self._frame_btn)
        btn_row1.addWidget(self._in_btn)
        btn_row1.addWidget(self._out_btn)

        btn_row2.addWidget(self._apply_btn)
        btn_row2.addWidget(self._undo_btn)
        btn_row2.addWidget(self._save_btn)

        layout.addLayout(btn_row1)
        layout.addLayout(btn_row2)

        self._range_lbl = QLabel("No correction range")
        self._range_lbl.setWordWrap(True)
        self._range_lbl.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-size: 12px;")
        layout.addWidget(self._range_lbl)

        self._frame_btn.clicked.connect(self._apply_frame)
        self._in_btn.clicked.connect(self._set_in)
        self._out_btn.clicked.connect(self._set_out)
        self._apply_btn.clicked.connect(self._apply_range)
        self._save_btn.clicked.connect(self.correctionSaved.emit)
        self._undo_btn.clicked.connect(self.undoRequested.emit)

    def set_current_frame(self, idx: int) -> None:
        self._current_frame = idx

    def reset(self) -> None:
        self._start = None
        self._end = None
        self._refresh()

    def selected_aoi(self) -> str:
        data = self._aoi_combo.currentData()
        return str(data) if data else NONE_LABEL

    def _apply_frame(self) -> None:
        self.correctionApplied.emit(False)

    def _set_in(self) -> None:
        self._start = self._current_frame
        if self._end is not None and self._end < self._start:
            self._end = None
        self._refresh()

    def _set_out(self) -> None:
        self._end = self._current_frame
        if self._start is not None and self._end < self._start:
            self._start, self._end = self._end, self._start
        self._refresh()

    def _apply_range(self) -> None:
        if self._start is None or self._end is None:
            return
        self.correctionApplied.emit(True)

    def _refresh(self) -> None:
        if self._start is None and self._end is None:
            self._range_lbl.setText("No correction range")
        elif self._start is not None and self._end is not None:
            self._range_lbl.setText(f"Range: {self._start} → {self._end}")
        elif self._start is not None:
            self._range_lbl.setText(f"In: {self._start}  (now set Out)")
        else:
            self._range_lbl.setText(f"Out: {self._end}  (now set In)")

    def active_range(self) -> tuple[Optional[int], Optional[int]]:
        return self._start, self._end


# The design uses one accent dot for every task row — no per-task hue.
_TASK_COLORS = [Theme.ACCENT] * 10


class TaskPanel(QWidget):
    """
    10 tasks (T1–T10). Click a row to select it, then use S/E buttons
    (or keyboard I/O) to mark start/end at the current video frame.
    """
    tasksChanged = Signal(dict)
    N_TASKS = 10

    def __init__(self) -> None:
        super().__init__()
        self._current_frame: int = 0
        self._fps: float = 30.0
        self._selected: Optional[str] = None  # no task selected until user clicks a row
        self._tasks: dict[str, dict] = {
            f"Task {i}": {"start": None, "end": None} for i in range(1, self.N_TASKS + 1)
        }
        self._rows: dict[str, dict] = {}

        # ── Frozen header (lives OUTSIDE the scroll area — set up in _build_ui) ──
        self.header_widget = QWidget()
        self.header_widget.setObjectName("taskHeader")
        header_l = QHBoxLayout(self.header_widget)
        header_l.setContentsMargins(6, 4, 6, 4)
        header_l.setSpacing(4)

        self._active_dot = QLabel("")
        self._active_dot.setFixedWidth(12)

        self._active_lbl = QLabel("select a task")
        self._active_lbl.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-style: italic; font-size: 12px;")
        self._active_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._active_lbl.setMinimumWidth(0)

        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedSize(48, 24)
        self._start_btn.setObjectName("taskHeaderBtn")
        self._start_btn.setEnabled(False)
        self._start_btn.setToolTip("Set start of selected task at current frame  (shortcut: I)")

        self._end_btn = QPushButton("End")
        self._end_btn.setFixedSize(40, 24)
        self._end_btn.setObjectName("taskHeaderBtn")
        self._end_btn.setEnabled(False)
        self._end_btn.setToolTip("Set end of selected task at current frame  (shortcut: O)")

        header_l.addWidget(self._active_dot)
        header_l.addWidget(self._active_lbl, 1)
        header_l.addWidget(self._start_btn)
        header_l.addWidget(self._end_btn)

        # ── Scrollable rows only ──────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Task rows
        for i, name in enumerate(self._tasks):
            color = _TASK_COLORS[i % len(_TASK_COLORS)]
            row_w = QWidget()
            row_w.setObjectName("taskRow")
            row_w.setCursor(Qt.PointingHandCursor)
            row_l = QHBoxLayout(row_w)
            # Extra right margin clears the task_scroll vertical scrollbar so the
            # trash button doesn't end up partially hidden underneath it.
            row_l.setContentsMargins(6, 4, 20, 4)
            row_l.setSpacing(6)

            dot = QLabel("●")
            dot.setFixedWidth(14)
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")

            name_lbl = QLabel(name)
            name_lbl.setFixedWidth(48)
            name_lbl.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")

            range_lbl = QLabel("─ not set ─")
            range_lbl.setStyleSheet(f"color: {Theme.TEXT_VFAINT}; font-size: 11px; font-family: '{Theme.FONT_MONO}', monospace;")

            dur_lbl = QLabel("")
            dur_lbl.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px; font-family: '{Theme.FONT_MONO}', monospace;")
            dur_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            clr_btn = QPushButton()
            clr_btn.setIcon(_svg_icon("trash.svg"))
            clr_btn.setIconSize(QSize(13, 13))
            clr_btn.setFixedSize(22, 22)
            clr_btn.setObjectName("iconButton")
            clr_btn.setToolTip(f"Clear {name}")
            clr_btn.clicked.connect(lambda _, n=name: self._clear(n))

            row_l.addWidget(dot)
            row_l.addWidget(name_lbl)
            row_l.addWidget(range_lbl, 1)
            row_l.addWidget(dur_lbl)
            row_l.addWidget(clr_btn)
            outer.addWidget(row_w)

            self._rows[name] = {
                "widget": row_w, "color": color,
                "range_lbl": range_lbl, "dur_lbl": dur_lbl,
            }
            self._tasks[name]["_row_color"] = color

            # Click row to select
            row_w.mousePressEvent = lambda ev, n=name: self._select(n)

        self._start_btn.clicked.connect(self._on_start)
        self._end_btn.clicked.connect(self._on_end)
        # No task pre-selected — user must click a row first

    # ── Public API ──────────────────────────────────────────────────────────

    def set_current_frame(self, idx: int) -> None:
        self._current_frame = idx

    def set_fps(self, fps: float) -> None:
        self._fps = max(1.0, float(fps))

    def reset(self) -> None:
        for name in self._tasks:
            self._tasks[name]["start"] = None
            self._tasks[name]["end"]   = None
            self._refresh_row(name)
        # Do NOT emit tasksChanged here — caller controls the save

    def get_tasks(self) -> dict[str, dict]:
        return {k: {"start": v["start"], "end": v["end"]}
                for k, v in self._tasks.items()}

    def load_tasks(self, tasks: dict) -> None:
        for name, data in tasks.items():
            if name in self._tasks:
                self._tasks[name]["start"] = data.get("start")
                self._tasks[name]["end"]   = data.get("end")
                self._refresh_row(name)
        self.tasksChanged.emit(self.get_tasks())

    # ── Internal ────────────────────────────────────────────────────────────

    def _select(self, name: str) -> None:
        self._selected = name
        self._active_dot.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 11px;")
        self._active_dot.setText("●")
        self._active_lbl.setText(name)
        self._active_lbl.setStyleSheet(f"color: {Theme.TEXT_BRIGHT}; font-weight: 600; font-size: 12px; font-style: normal;")
        self._start_btn.setEnabled(True)
        self._end_btn.setEnabled(True)
        for n, row in self._rows.items():
            row["widget"].setStyleSheet(
                f"QWidget#taskRow {{ background: {'rgba(255,255,255,.05)' if n == name else 'transparent'}; "
                f"border-radius: {Theme.RADIUS}px; }}"
            )

    def _on_start(self) -> None:
        if self._selected is None:
            return
        self._tasks[self._selected]["start"] = self._current_frame
        self._refresh_row(self._selected)
        self.tasksChanged.emit(self.get_tasks())

    def _on_end(self) -> None:
        if self._selected is None:
            return
        self._tasks[self._selected]["end"] = self._current_frame
        self._refresh_row(self._selected)
        self.tasksChanged.emit(self.get_tasks())

    def _clear(self, name: str) -> None:
        self._tasks[name]["start"] = None
        self._tasks[name]["end"]   = None
        self._refresh_row(name)
        self.tasksChanged.emit(self.get_tasks())

    def _refresh_row(self, name: str) -> None:
        row  = self._rows[name]
        s, e = self._tasks[name]["start"], self._tasks[name]["end"]
        mono = f"font-family: '{Theme.FONT_MONO}', monospace;"
        if s is not None and e is not None:
            dur_s = abs(e - s) / self._fps
            row["range_lbl"].setText(f"{s} → {e}")
            row["range_lbl"].setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px; font-weight: 500; {mono}")
            row["dur_lbl"].setText(f"{int(dur_s//60)}:{int(dur_s%60):02d}")
        elif s is not None:
            row["range_lbl"].setText(f"{s} → ?")
            row["range_lbl"].setStyleSheet(f"color: {Theme.TEXT_DIM}; font-size: 11px; {mono}")
            row["dur_lbl"].setText("")
        else:
            row["range_lbl"].setText("─ not set ─")
            row["range_lbl"].setStyleSheet(f"color: {Theme.TEXT_VFAINT}; font-size: 11px; {mono}")
            row["dur_lbl"].setText("")


# ─── Dashboard tab ────────────────────────────────────────────────────────────

class DashboardWidget(QWidget):
    """
    Dashboard with:
      • Recording picker: individual recording or all (aggregate)
      • Chart 1: AOI dwell % bar chart  (mean ± std when multiple recordings)
      • Chart 2: Learning curve — task duration vs task number
      • Chart 3: Per-recording heatmap
      • Stats table in HTML
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # ── Left sidebar: all controls, stacked vertically ──────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("leftPanel")
        sidebar.setFixedWidth(224)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(16, 18, 16, 18)
        side_l.setSpacing(10)

        self._diff_combo = QComboBox()
        self._diff_combo.setToolTip("Difficulty folder (Easy / Medium / Hard)")

        self._rec_combo = QComboBox()
        self._rec_combo.setToolTip("Individual recording or All (aggregate mean)")

        self._task_combo = QComboBox()
        self._task_combo.addItems(["All Tasks"] + [f"Task {i}" for i in range(1, 11)])

        self._load_btn     = QPushButton("  Load  ")
        self._gen_btn      = QPushButton("  Rebuild  ")
        self._png_btn      = QPushButton("📷  PNGs")
        self._export_btn   = QPushButton("📗  Workbook")

        self._load_btn.setObjectName("primaryButton")
        self._load_btn.setToolTip("Load saved analysis for the selected recording")
        self._gen_btn.setToolTip("Force-rebuild all charts (same as Load but always regenerates)")
        self._png_btn.setToolTip("Save every chart as a PNG image")
        self._export_btn.setToolTip("Export all charts and tables to an Excel workbook")

        side_l.addWidget(QLabel("Difficulty:"))
        side_l.addWidget(self._diff_combo)
        side_l.addWidget(QLabel("Recording:"))
        side_l.addWidget(self._rec_combo)
        side_l.addWidget(QLabel("Task:"))
        side_l.addWidget(self._task_combo)

        _div1 = QFrame(); _div1.setFrameShape(QFrame.HLine); _div1.setObjectName("divider")
        side_l.addWidget(_div1)

        side_l.addWidget(self._load_btn)
        side_l.addWidget(self._gen_btn)
        side_l.addWidget(self._png_btn)
        side_l.addWidget(self._export_btn)
        side_l.addStretch(1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 12px;")
        side_l.addWidget(self._status)

        layout.addWidget(sidebar)

        # ── Right: chart area fills all remaining space ─────────────────────────
        chart_area = QVBoxLayout()
        chart_area.setContentsMargins(0, 0, 0, 0)

        # ── Stacked: aggregate view (page 0) vs per-recording view (page 1) ──
        self._stack = QStackedWidget()

        # Page 0 — aggregate / multi-recording 6-tab view (mirrors individual)
        agg_container = QWidget()
        agg_vl = QVBoxLayout(agg_container)
        agg_vl.setContentsMargins(0, 0, 0, 0)
        self._agg_tabs = QTabWidget()
        self._agg_tab_views: dict[str, QWebEngineView] = {}
        for slug, label in _REC_TAB_DEFS:
            view = QWebEngineView()
            view.setHtml(_CHART_PLACEHOLDER)
            self._agg_tabs.addTab(view, f"  {label}  ")
            self._agg_tab_views[slug] = view
        agg_vl.addWidget(self._agg_tabs)
        self._stack.addWidget(agg_container)

        # Page 1 — individual recording: 6 sub-tabs
        rec_container = QWidget()
        rec_vl = QVBoxLayout(rec_container)
        rec_vl.setContentsMargins(0, 0, 0, 0)
        self._rec_tabs = QTabWidget()
        self._rec_tab_views: dict[str, QWebEngineView] = {}
        for slug, label in _REC_TAB_DEFS:
            view = QWebEngineView()
            view.setHtml(_CHART_PLACEHOLDER)
            self._rec_tabs.addTab(view, f"  {label}  ")
            self._rec_tab_views[slug] = view
        rec_vl.addWidget(self._rec_tabs)
        self._stack.addWidget(rec_container)

        chart_area.addWidget(self._stack, 1)
        layout.addLayout(chart_area, 1)

        self._diff_combo.currentIndexChanged.connect(self._refresh_rec_combo)
        self._load_btn.clicked.connect(self._generate)
        self._gen_btn.clicked.connect(self._generate)
        self._png_btn.clicked.connect(self._export_png)
        self._export_btn.clicked.connect(self._export_workbook)
        self._populate_diff_combo()

    # ── Populate dropdowns ────────────────────────────────────────────────────

    def _populate_diff_combo(self) -> None:
        self._diff_combo.blockSignals(True)
        self._diff_combo.clear()
        if RECORDINGS_DIR.exists():
            for d in sorted(RECORDINGS_DIR.iterdir()):
                if d.is_dir():
                    self._diff_combo.addItem(d.name, userData=d)
        self._diff_combo.blockSignals(False)
        self._refresh_rec_combo()

    def _refresh_rec_combo(self) -> None:
        self._rec_combo.clear()
        src: Optional[pathlib.Path] = self._diff_combo.currentData()
        if src is None or not src.exists():
            return
        self._rec_combo.addItem("All recordings (aggregate)", userData=None)
        for d in sorted(src.iterdir()):
            if d.is_dir() and (d / "info.json").exists():
                self._rec_combo.addItem(d.name, userData=d)

    def _current_source(self) -> Optional[pathlib.Path]:
        return self._diff_combo.currentData()

    # ── Collect CSVs ─────────────────────────────────────────────────────────

    def _collect_csvs(self) -> tuple[Optional[pathlib.Path], Optional[pathlib.Path], list[tuple[str, pathlib.Path]]]:
        """Returns (src_dir, rec_filter, csvs_list)."""
        src: Optional[pathlib.Path] = self._current_source()
        if src is None:
            return None, None, []
        rec_filter: Optional[pathlib.Path] = self._rec_combo.currentData()
        csvs: list[tuple[str, pathlib.Path]] = []
        for rec_dir in sorted(src.iterdir()):
            if not rec_dir.is_dir():
                continue
            if rec_filter is not None and rec_dir != rec_filter:
                continue
            for cand in [
                rec_dir / "aoi_results" / "analysis.csv",
                rec_dir / "aoi_results" / "raw" / "analysis.csv",
            ]:
                if cand.exists() and cand.stat().st_size > 0:
                    csvs.append((rec_dir.name, cand))
                    break
        return src, rec_filter, csvs

    # ── Generate ──────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-read from disk and rebuild the current view (called after a Studio
        edit/crop so the dashboard stays in sync). Silent no-op if not yet set up."""
        try:
            if self._diff_combo.count() > 0:
                self._generate()
        except Exception:
            pass

    def _generate(self) -> None:
        src, rec_filter, csvs = self._collect_csvs()
        if src is None:
            return
        if not csvs:
            self._status.setText("No analysed recordings found. Run Analysis first.")
            return
        self._status.setText("Building charts…")
        QApplication.processEvents()
        task_filter = self._task_combo.currentText()

        if rec_filter is not None:
            # ── Single recording → individual 6-tab view ──────────────────────
            self._stack.setCurrentIndex(1)
            self._generate_single(rec_filter, csvs[0][1], task_filter)
            self._status.setText(f"{rec_filter.name}")
        else:
            # ── All recordings → aggregate 6-tab view ─────────────────────────
            self._stack.setCurrentIndex(0)
            self._generate_aggregate(csvs, task_filter)
            self._status.setText(f"{len(csvs)} recording(s) from {src.name}.")

    # ── Individual recording view ─────────────────────────────────────────────

    @staticmethod
    def _fig_to_html(fig) -> str:
        chart_html = fig.to_html(
            full_html=False, include_plotlyjs="cdn",
            config={"responsive": True}, default_height="100%",
        )
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>html,body{height:100%;background:#0e0f11;margin:0;padding:8px;"
            "box-sizing:border-box}.plotly-graph-div{height:100% !important;"
            "width:100% !important}</style></head>"
            f"<body>{chart_html}</body></html>"
        )

    @staticmethod
    def _no_data_html(msg: str) -> str:
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>body{background:#0e0f11;color:#62656b;display:flex;align-items:center;"
            "justify-content:center;height:90vh;font-family:'Hanken Grotesk',sans-serif;"
            "font-size:14px;text-align:center;padding:20px}</style>"
            f"</head><body>{msg}</body></html>"
        )

    def _generate_single(self, rec_dir: pathlib.Path, csv_path: pathlib.Path,
                         task_filter: str) -> None:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            for v in self._rec_tab_views.values():
                v.setHtml(self._no_data_html(f"Could not read analysis.csv:<br>{exc}"))
            return

        lbl_col = "final_primary_aoi" if "final_primary_aoi" in df.columns else "primary_aoi"
        df["_aoi"] = df[lbl_col].fillna(NONE_LABEL).astype(str)

        # Estimate fps from the full recording's time_s (stable regardless of task).
        fps = 30.0
        if "time_s" in df.columns and len(df) > 1:
            times = pd.to_numeric(df["time_s"], errors="coerce").dropna()
            dur = float(times.iloc[-1] - times.iloc[0]) if len(times) > 1 else 0.0
            if dur > 0:
                fps = max(1.0, (len(times) - 1) / dur)

        # Task filter drives EVERY tab: slice the per-frame data to the selected
        # task's [start,end] frame range so dwell/fixation/heatmap/transition all
        # reflect just that task (was previously applied to the Learning tab only).
        bounds = self._load_task_bounds(rec_dir, task_filter)
        dft = self._slice_task(df, bounds)
        total = len(dft)

        self._pop_dq_tab(rec_dir, dft, total, fps)
        self._pop_dwell_tab(dft, total, fps)
        self._pop_fixation_tab(rec_dir, bounds)
        self._pop_heatmap_tab(dft)
        self._pop_transition_tab(dft)
        self._pop_learning_tab(rec_dir, fps, task_filter)

    @staticmethod
    def _load_task_bounds(rec_dir: pathlib.Path, task_filter: str) -> tuple | None:
        """(start_frame, end_frame) for the selected task, or None for All Tasks."""
        if task_filter == "All Tasks":
            return None
        for p in [rec_dir / "aoi_results" / "tasks.json",
                  rec_dir / "aoi_results" / "raw" / "tasks.json"]:
            if p.exists():
                try:
                    td = json.loads(p.read_text(encoding="utf-8")).get(task_filter)
                except Exception:
                    return None
                if isinstance(td, dict):
                    s, e = td.get("start"), td.get("end")
                    if s is not None and e is not None and int(e) > int(s):
                        return int(s), int(e)
                return None
        return None

    @staticmethod
    def _slice_task(df: pd.DataFrame, bounds: tuple | None) -> pd.DataFrame:
        if bounds is None or "frame_idx" not in df.columns:
            return df
        fi = pd.to_numeric(df["frame_idx"], errors="coerce")
        return df[(fi >= bounds[0]) & (fi <= bounds[1])].copy()

    # ── Tab 1: Data Quality ───────────────────────────────────────────────────

    def _pop_dq_tab(self, rec_dir: pathlib.Path, df: pd.DataFrame,
                    total: int, fps: float) -> None:
        view = self._rec_tab_views["dq"]

        dq, pr = {}, {}
        for fname, target in [("data_quality.json", "dq"),
                               ("preprocessing_report.json", "pr")]:
            for base in [rec_dir / "aoi_results" / "raw" / fname,
                         rec_dir / "aoi_results" / fname]:
                if base.exists():
                    try:
                        val = json.loads(base.read_text(encoding="utf-8"))
                        if target == "dq":
                            dq = val
                        else:
                            pr = val
                    except Exception:
                        pass
                    break

        valid_pct  = 100.0 - float(dq.get("missing_gaze_pct", 0.0))
        fix_n      = int(dq.get("fixation_count", 0))
        dur_s      = float(dq.get("recording_duration_s", total / fps))
        fps_val    = float(dq.get("fps", fps))
        dur_str    = f"{int(dur_s)//60}:{int(dur_s)%60:02d}"
        no_aoi_pct = float((df["_aoi"] == NONE_LABEL).sum() / total * 100) if total > 0 else 0.0
        cov_pct    = 100.0 - no_aoi_pct

        def _col(v, good, warn):
            return "#6fae7d" if v >= good else "#d4a24a" if v >= warn else "#cf6b6b"

        # AOI dwell table rows
        aoi_rows = ""
        for aoi in AOI_NAMES:
            cnt  = int((df["_aoi"] == aoi).sum())
            if cnt == 0:
                continue
            pct  = cnt / total * 100
            dwell_s = cnt / fps
            color = AOI_COLORS_QT.get(aoi, QColor("#888")).name()
            aoi_rows += (
                f"<tr><td style='color:{color}'><b>{aoi}</b></td>"
                f"<td>{cnt}</td><td>{pct:.1f}%</td><td>{dwell_s:.1f}s</td></tr>"
            )

        # Detection method breakdown (from preprocessing_report.json)
        det_section = ""
        if pr:
            det_section = f"""
<div class='section'><h3>Detection Method Breakdown</h3>
<table><tr><th>Method</th><th>Frames</th><th>%</th></tr>
<tr><td>3D Surface Mapper</td><td>{pr.get('surface_3d_frames','—')}</td>
    <td>{pr.get('surface_3d_pct','—')}%</td></tr>
<tr><td>2D Partial-marker Fallback</td><td>{pr.get('fallback_2d_frames','—')}</td>
    <td>{pr.get('fallback_2d_pct','—')}%</td></tr>
<tr><td>No Detection (NoAOI)</td><td>{pr.get('no_aoi_frames','—')}</td>
    <td>{pr.get('no_aoi_pct','—')}%</td></tr>
</table>
<p style='color:#5a5d63;font-size:12px;margin-top:10px;line-height:1.7'>
3D Surface Mapper = highest accuracy (all markers visible).<br>
2D Fallback = partial marker visibility, still classified.<br>
NoAOI = insufficient markers or gaze outside defined areas.
</p></div>"""

        html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body{{background:#0e0f11;color:#e7e8ea;font-family:'Hanken Grotesk',sans-serif;margin:0;padding:20px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}
.card{{background:#131416;border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:18px}}
.val{{font-size:28px;font-weight:500;margin:8px 0 4px;font-family:'IBM Plex Mono',monospace}}
.lbl{{font-size:12px;color:#83868c}}
.section{{margin-top:24px}}
.section h3{{color:#6a6d73;font-size:11px;letter-spacing:1.1px;text-transform:uppercase;font-weight:500;
  border-bottom:1px solid rgba(255,255,255,.07);padding-bottom:8px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{padding:9px 14px 9px 0}}
th{{color:#62656b;font-size:11px;letter-spacing:.05em;border-bottom:1px solid rgba(255,255,255,.06)}}
td{{border-bottom:1px solid rgba(255,255,255,.04)}}
</style></head><body>
<h2 style='margin:0 0 14px;font-size:15px;font-family:"IBM Plex Mono",monospace;font-weight:400'>{rec_dir.name}</h2>
<div class='grid'>
  <div class='card'><div class='lbl'>Valid Gaze</div>
    <div class='val' style='color:{_col(valid_pct,80,60)}'>{valid_pct:.1f}%</div>
    <div class='lbl'>{int(dq.get("valid_gaze_frames",0))} / {total} frames</div></div>
  <div class='card'><div class='lbl'>Fixations Detected</div>
    <div class='val' style='color:#6e8fd6'>{fix_n}</div>
    <div class='lbl'>from Neon SDK</div></div>
  <div class='card'><div class='lbl'>Duration</div>
    <div class='val'>{dur_str}</div>
    <div class='lbl'>{fps_val:.0f} fps · {total} frames</div></div>
  <div class='card'><div class='lbl'>AOI Coverage</div>
    <div class='val' style='color:{_col(cov_pct,80,60)}'>{cov_pct:.1f}%</div>
    <div class='lbl'>frames within an AOI</div></div>
  <div class='card'><div class='lbl'>NoAOI</div>
    <div class='val' style='color:{"#cf6b6b" if no_aoi_pct>30 else "#666"}'>{no_aoi_pct:.1f}%</div>
    <div class='lbl'>outside all AOI boundaries</div></div>
  <div class='card'><div class='lbl'>Missing Gaze</div>
    <div class='val' style='color:{"#cf6b6b" if (100-valid_pct)>20 else "#666"}'>{100-valid_pct:.1f}%</div>
    <div class='lbl'>blinks / data loss</div></div>
</div>
<div class='section'><h3>AOI Frame Distribution</h3>
<table><tr><th>AOI</th><th>Frames</th><th>%</th><th>Dwell (s)</th></tr>
{aoi_rows}</table></div>
{det_section}
</body></html>"""
        view.setHtml(html)

    # ── Tab 2: Dwell Time ─────────────────────────────────────────────────────

    def _pop_dwell_tab(self, df: pd.DataFrame, total: int, fps: float) -> None:
        view = self._rec_tab_views["dwell"]
        rows = []
        for aoi in AOI_NAMES:
            cnt = int((df["_aoi"] == aoi).sum())
            rows.append({"aoi": aoi, "dwell_pct": cnt / total * 100 if total else 0.0,
                         "dwell_s": cnt / fps})
        ddf   = pd.DataFrame(rows)
        colors = [AOI_COLORS_QT.get(a, QColor("#6e8fd6")).name() for a in ddf["aoi"]]
        fig = go.Figure(go.Bar(
            x=ddf["aoi"], y=ddf["dwell_pct"],
            marker_color=colors,
            customdata=ddf["dwell_s"].round(1).values,
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>%{customdata}s<extra></extra>",
        ))
        fig.update_layout(title="Gaze Dwell Time per AOI",
                          xaxis_title="AOI", yaxis_title="Dwell (%)",
                          template="aoi_studio", autosize=True,
                          margin=dict(t=50, b=40))
        view.setHtml(self._fig_to_html(fig))

    # ── Tab 3: Fixation Metrics ───────────────────────────────────────────────

    def _pop_fixation_tab(self, rec_dir: pathlib.Path, bounds: tuple | None = None) -> None:
        view = self._rec_tab_views["fixations"]
        fix_path = None
        for p in [rec_dir / "aoi_results" / "raw" / "fixation_summary.csv",
                  rec_dir / "aoi_results" / "fixation_summary.csv"]:
            if p.exists() and p.stat().st_size > 0:
                fix_path = p
                break
        if fix_path is None:
            view.setHtml(self._no_data_html(
                "Re-run Analysis to generate fixation_summary.csv.<br>"
                "Recordings analysed before Phase 7 need to be re-analysed."))
            return
        try:
            fdf = pd.read_csv(fix_path)
        except Exception as exc:
            view.setHtml(self._no_data_html(str(exc)))
            return

        # Task filter: keep fixations whose start falls inside the task range.
        if bounds is not None and "start_frame" in fdf.columns:
            sf = pd.to_numeric(fdf["start_frame"], errors="coerce")
            fdf = fdf[(sf >= bounds[0]) & (sf <= bounds[1])]
            if fdf.empty:
                view.setHtml(self._no_data_html("No fixations in the selected task range."))
                return

        rows = []
        for aoi in AOI_NAMES:
            aoi_rows = fdf[fdf["dominant_aoi"] == aoi]
            rows.append({"aoi": aoi,
                         "count": len(aoi_rows),
                         "mean_dur": float(aoi_rows["duration_s"].mean()) if len(aoi_rows) else 0.0,
                         "total_dur": float(aoi_rows["duration_s"].sum()) if len(aoi_rows) else 0.0})
        adf    = pd.DataFrame(rows)
        colors = [AOI_COLORS_QT.get(a, QColor("#6e8fd6")).name() for a in adf["aoi"]]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Fixation Count per AOI",
                                            "Mean Fixation Duration (s)"))
        fig.add_trace(go.Bar(x=adf["aoi"], y=adf["count"], marker_color=colors,
                             hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=adf["aoi"], y=adf["mean_dur"].round(3),
                             marker_color=colors,
                             hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f}s<extra></extra>"),
                      row=1, col=2)
        fig.update_layout(template="aoi_studio", autosize=True,
                          showlegend=False, margin=dict(t=50, b=40))
        view.setHtml(self._fig_to_html(fig))

    # ── Tab 4: AOI Heatmaps ───────────────────────────────────────────────────

    def _pop_heatmap_tab(self, df: pd.DataFrame) -> None:
        view = self._rec_tab_views["heatmaps"]
        if "gaze_on_aoi_x" not in df.columns or "gaze_on_aoi_y" not in df.columns:
            view.setHtml(self._no_data_html(
                "No surface coordinate data in analysis.csv.<br>"
                "Heatmaps require 3D surface detection during analysis."))
            return

        from scipy.ndimage import gaussian_filter as _gf

        aois_data = []
        for aoi in AOI_NAMES:
            sub  = df[df["_aoi"] == aoi]
            # Attention heatmap: keep fixation frames only when available. Longer
            # fixations span more frames, so this is inherently dwell-weighted, and
            # saccade frames (gaze in flight) are dropped as noise.
            if "is_fixation" in sub.columns:
                fix = sub[sub["is_fixation"].astype(str).str.lower() == "true"]
                if len(fix) >= 10:
                    sub = fix
            x    = pd.to_numeric(sub["gaze_on_aoi_x"], errors="coerce").dropna().values
            y    = pd.to_numeric(sub["gaze_on_aoi_y"], errors="coerce").dropna().values
            mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
            if mask.sum() >= 10:
                aois_data.append((aoi, x[mask], y[mask]))

        if not aois_data:
            view.setHtml(self._no_data_html(
                "Not enough surface coordinate data for heatmaps.<br>"
                "Re-run Analysis to regenerate with current pipeline."))
            return

        n    = len(aois_data)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig  = make_subplots(rows=rows, cols=cols,
                             subplot_titles=[a[0] for a in aois_data])

        for i, (aoi_name, x, y) in enumerate(aois_data):
            r, c = i // cols + 1, i % cols + 1
            color = AOI_COLORS_QT.get(aoi_name, QColor("#6e8fd6"))
            rv, gv, bv = color.red(), color.green(), color.blue()
            H, xe, ye = np.histogram2d(x, y, bins=25, range=[[0, 1], [0, 1]])
            Hs = _gf(H.T, sigma=1.5)
            Hn = Hs / Hs.max() if Hs.max() > 0 else Hs
            colorscale = [[0.0, "rgba(0,0,0,0)"],
                          [0.3, f"rgba({rv},{gv},{bv},0.3)"],
                          [1.0, f"rgba({rv},{gv},{bv},1.0)"]]
            fig.add_trace(go.Heatmap(
                z=Hn,
                x=xe[:-1] + (xe[1] - xe[0]) / 2,
                y=ye[:-1] + (ye[1] - ye[0]) / 2,
                colorscale=colorscale, showscale=False,
                hovertemplate=f"X:%{{x:.2f}} Y:%{{y:.2f}}<br>Density:%{{z:.2f}}<extra>{aoi_name}</extra>",
            ), row=r, col=c)

        fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False)
        fig.update_layout(
            title="Spatial Gaze Distribution within Each AOI Surface (0–1 normalised)",
            template="aoi_studio",
            height=max(360, 340 * rows),
            margin=dict(t=60, b=20))
        view.setHtml(self._fig_to_html(fig))

    # ── Tab 5: Transition Matrix ──────────────────────────────────────────────

    def _pop_transition_tab(self, df: pd.DataFrame) -> None:
        view = self._rec_tab_views["transitions"]
        labels = df["_aoi"].values
        matrix = pd.DataFrame(0, index=AOI_NAMES, columns=AOI_NAMES, dtype=int)
        for i in range(len(labels) - 1):
            src, dst = str(labels[i]), str(labels[i + 1])
            if src != dst and src in matrix.index and dst in matrix.columns:
                matrix.loc[src, dst] += 1
        row_sums = matrix.sum(axis=1)
        normed   = (matrix.div(row_sums.replace(0, np.nan), axis=0)
                    .fillna(0) * 100).round(1)

        if normed.values.sum() == 0:
            view.setHtml(self._no_data_html("No AOI transitions detected in this recording."))
            return

        fig = go.Figure(go.Heatmap(
            z=normed.values.tolist(),
            x=list(normed.columns),
            y=list(normed.index),
            colorscale="Blues",
            text=normed.values.round(1).tolist(),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="From: %{y}<br>To: %{x}<br>%{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title="AOI Transition Probabilities — row→column (% of outgoing transitions)",
            template="aoi_studio",
            height=max(360, 50 * len(AOI_NAMES) + 160),
            xaxis_title="To AOI", yaxis_title="From AOI",
            margin=dict(t=60, b=60, l=110, r=40))
        view.setHtml(self._fig_to_html(fig))

    # ── Tab 6: Learning Curve ─────────────────────────────────────────────────

    def _pop_learning_tab(self, rec_dir: pathlib.Path, fps: float,
                          task_filter: str) -> None:
        view = self._rec_tab_views["learning"]
        task_path = None
        for p in [rec_dir / "aoi_results" / "tasks.json",
                  rec_dir / "aoi_results" / "raw" / "tasks.json"]:
            if p.exists():
                task_path = p
                break
        if task_path is None:
            view.setHtml(self._no_data_html(
                "No tasks annotated yet.<br>"
                "Mark task repetitions T1–T10 in the Studio tab."))
            return
        try:
            tasks = json.loads(task_path.read_text(encoding="utf-8"))
        except Exception as exc:
            view.setHtml(self._no_data_html(str(exc)))
            return

        analysis_df = None
        for csv_p in [
            rec_dir / "aoi_results" / "analysis.csv",
            rec_dir / "aoi_results" / "raw" / "analysis.csv",
        ]:
            if csv_p.exists() and csv_p.stat().st_size > 0:
                try:
                    analysis_df = pd.read_csv(csv_p)
                    break
                except Exception:
                    pass

        rows = []
        for t_name, t_data in tasks.items():
            s, e = t_data.get("start"), t_data.get("end")
            if s is None or e is None or e <= s:
                continue
            digits = "".join(c for c in t_name if c.isdigit())
            if analysis_df is not None:
                dur_s = reporting._duration_between_frame_range(
                    analysis_df, int(s), int(e), fps
                )
            else:
                dur_s = (int(e) - int(s)) / fps
            rows.append({"task": t_name,
                         "task_idx": int(digits) if digits else 0,
                         "dur_s": dur_s})
        if not rows:
            view.setHtml(self._no_data_html("No complete task annotations found."))
            return

        tdf = pd.DataFrame(rows).sort_values("task_idx")
        if task_filter != "All Tasks":
            tdf = tdf[tdf["task"] == task_filter]
        if tdf.empty:
            view.setHtml(self._no_data_html(f"No data for filter: {task_filter}"))
            return

        # Errors render as a second panel DIRECTLY under the learning curve, sharing
        # the Task 1-10 x-axis so repetitions line up vertically.
        series = _error_series(_load_errors_json(rec_dir))
        has_err = self._has_errors(series)
        if has_err:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                                row_heights=[0.55, 0.45],
                                subplot_titles=("Learning Curve — Task Duration",
                                                "Assembly Errors per Task"))
        else:
            fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(
            x=tdf["task_idx"], y=tdf["dur_s"], mode="lines+markers",
            line=dict(width=2, color="#6e8fd6"), marker=dict(size=8, color="#6e8fd6"),
            customdata=tdf["task"].values, showlegend=False,
            hovertemplate="<b>%{customdata}</b><br>Duration: %{y:.1f}s<extra></extra>",
        ), row=1, col=1)
        if len(tdf) > 2:
            z = np.polyfit(tdf["task_idx"].values, tdf["dur_s"].values, 1)
            xr = np.linspace(tdf["task_idx"].min(), tdf["task_idx"].max(), 50)
            fig.add_trace(go.Scatter(x=xr, y=np.poly1d(z)(xr), mode="lines",
                line=dict(width=1.5, dash="dash", color="#333"),
                showlegend=False, hoverinfo="skip"), row=1, col=1)
        mean_dur = tdf["dur_s"].mean()
        fig.add_hline(y=mean_dur, line_dash="dot", line_color="#555",
                      annotation_text=f"Mean {mean_dur:.1f}s",
                      annotation_position="bottom right", row=1, col=1)
        fig.update_yaxes(title_text="Duration (s)", row=1, col=1)
        layout = dict(template="aoi_studio", autosize=True, margin=dict(t=60, b=40))
        if has_err:
            buttons = self._add_error_traces(fig, series, row=2, n_prefix=len(fig.data))
            fig.update_xaxes(title_text="Task", dtick=1, row=2, col=1)
            fig.update_yaxes(title_text="Error count", row=2, col=1)
            layout["updatemenus"] = [dict(type="buttons", direction="right", x=0.0, y=1.10,
                                          xanchor="left", active=2, buttons=buttons)]
            layout["height"] = 640
        else:
            fig.update_xaxes(title_text="Repetition", dtick=1, row=1, col=1)
            fig.update_layout(title="Learning Curve — Task Duration per Repetition")
        fig.update_layout(**layout)
        view.setHtml(self._fig_to_html(fig))

    # ── Assembly-error traces (rendered UNDER the learning curve) ─────────────

    @staticmethod
    def _has_errors(series: dict) -> bool:
        return bool(series) and not all(sum(v.values()) == 0 for v in series.values())

    def _agg_error_series(self, csvs: list[tuple[str, pathlib.Path]]) -> dict:
        """Mean per-task error series across the given recordings."""
        acc: dict[int, dict[str, list]] = {}
        for _rec, csv_path in csvs:
            series = _error_series(_load_errors_json(self._rec_dir_from_csv(csv_path)))
            for t, v in series.items():
                d = acc.setdefault(t, {"comp_type": [], "comp_pos": [], "cable_type": [], "cable_pos": []})
                for k in d:
                    d[k].append(v[k])
        return {t: {k: (sum(vals) / len(vals) if vals else 0.0) for k, vals in d.items()}
                for t, d in acc.items()}

    def _add_error_traces(self, fig, series: dict, row: int, n_prefix: int) -> list:
        """Add the 6 error traces to `fig` at `row`; return the Components/Cables/
        All-Both toggle buttons (learning traces at indices <n_prefix stay visible)."""
        tasks = sorted(series.keys())
        ct = [series[t]["comp_type"] for t in tasks]
        cp = [series[t]["comp_pos"] for t in tasks]
        kt = [series[t]["cable_type"] for t in tasks]
        kp = [series[t]["cable_pos"] for t in tasks]
        comp_tot = [ct[i] + cp[i] for i in range(len(tasks))]
        cable_tot = [kt[i] + kp[i] for i in range(len(tasks))]
        specs = [("Component · Type", ct, "#d4a24a"), ("Component · Position", cp, "#6fae7d"),
                 ("Cable · Type", kt, "#c07ba8"), ("Cable · Position", kp, "#6e8fd6"),
                 ("Components (total)", comp_tot, "#d4a24a"), ("Cables (total)", cable_tot, "#6e8fd6")]
        for name, y, color in specs:
            fig.add_trace(go.Scatter(x=tasks, y=y, mode="lines+markers", name=name,
                                     line=dict(color=color), showlegend=True), row=row, col=1)
        views = {"Components": [True, True, False, False, False, False],
                 "Cables":     [False, False, True, True, False, False],
                 "All / Both": [False, False, False, False, True, True]}
        for i in range(6):
            fig.data[n_prefix + i].visible = views["All / Both"][i]
        return [dict(label=k, method="restyle", args=[{"visible": [True] * n_prefix + v}])
                for k, v in views.items()]

    # ── Aggregate (multi-recording) tab population ────────────────────────────

    @staticmethod
    def _rec_dir_from_csv(csv_path: pathlib.Path) -> pathlib.Path:
        """Derive the recording folder from its analysis.csv path."""
        return csv_path.parent.parent.parent if csv_path.parent.name == "raw" \
               else csv_path.parent.parent

    def _agg_load(self, csv_path: pathlib.Path, task_filter: str):
        """Read one recording's analysis, add _aoi, and slice to that recording's
        own task range so aggregate tabs aggregate the SAME task across recordings.
        Returns (sliced_df, bounds)."""
        df = pd.read_csv(csv_path)
        lbl_col = "final_primary_aoi" if "final_primary_aoi" in df.columns else "primary_aoi"
        df["_aoi"] = df[lbl_col].fillna(NONE_LABEL).astype(str)
        bounds = self._load_task_bounds(self._rec_dir_from_csv(csv_path), task_filter)
        return self._slice_task(df, bounds), bounds

    def _generate_aggregate(self, csvs: list[tuple[str, pathlib.Path]],
                            task_filter: str) -> None:
        self._pop_agg_dq_tab(csvs)
        QApplication.processEvents()
        self._pop_agg_dwell_tab(csvs, task_filter)
        QApplication.processEvents()
        self._pop_agg_fixation_tab(csvs, task_filter)
        QApplication.processEvents()
        self._pop_agg_heatmap_tab(csvs, task_filter)
        QApplication.processEvents()
        self._pop_agg_transition_tab(csvs, task_filter)
        QApplication.processEvents()
        self._pop_agg_learning_tab(csvs, task_filter)

    # ── Aggregate Tab 1: Data Quality ─────────────────────────────────────────

    def _pop_agg_dq_tab(self, csvs: list[tuple[str, pathlib.Path]]) -> None:
        view = self._agg_tab_views["dq"]
        n = len(csvs)
        valid_pcts, fix_counts, dur_ss, cov_pcts = [], [], [], []
        aoi_frames_all: dict[str, list[float]] = {a: [] for a in AOI_NAMES}

        for _rec_name, csv_path in csvs:
            rec_dir = self._rec_dir_from_csv(csv_path)
            dq: dict = {}
            for dq_path in [rec_dir / "aoi_results" / "raw" / "data_quality.json",
                            rec_dir / "aoi_results" / "data_quality.json"]:
                if dq_path.exists():
                    try:
                        dq = json.loads(dq_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                    break
            valid_pcts.append(100.0 - float(dq.get("missing_gaze_pct", 0.0)))
            fix_counts.append(float(dq.get("fixation_count", 0)))
            dur_ss.append(float(dq.get("recording_duration_s", 0.0)))
            try:
                df = pd.read_csv(csv_path)
                lbl_col = "final_primary_aoi" if "final_primary_aoi" in df.columns else "primary_aoi"
                df["_aoi"] = df[lbl_col].fillna(NONE_LABEL).astype(str)
                total = max(1, len(df))
                cov_pcts.append(100.0 - (df["_aoi"] == NONE_LABEL).sum() / total * 100)
                for aoi in AOI_NAMES:
                    aoi_frames_all[aoi].append((df["_aoi"] == aoi).sum() / total * 100)
            except Exception:
                cov_pcts.append(0.0)

        def _ms(vals: list) -> tuple[float, float]:
            if not vals:
                return 0.0, 0.0
            arr = np.array(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        def _col(v: float, good: float, warn: float) -> str:
            return "#6fae7d" if v >= good else "#d4a24a" if v >= warn else "#cf6b6b"

        def _pm(std: float) -> str:
            return f" <span style='color:#444'>±{std:.1f}</span>" if std > 0 else ""

        mv, sv = _ms(valid_pcts)
        mf, sf = _ms(fix_counts)
        md, sd = _ms(dur_ss)
        mc, sc = _ms(cov_pcts)
        mm = 100.0 - mv
        missing_color = "#cf6b6b" if mm > 20 else "#62656b"

        aoi_rows = ""
        for aoi in AOI_NAMES:
            vals = aoi_frames_all[aoi]
            if not vals or all(v == 0 for v in vals):
                continue
            m, s = _ms(vals)
            color = AOI_COLORS_QT.get(aoi, QColor("#888")).name()
            aoi_rows += (
                f"<tr><td style='color:{color}'><b>{aoi}</b></td>"
                f"<td>{m:.1f}%{_pm(s)}</td></tr>"
            )

        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>"
            "body{background:#0e0f11;color:#e7e8ea;font-family:'Hanken Grotesk',sans-serif;margin:0;padding:20px}"
            ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}"
            ".card{background:#131416;border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:18px}"
            ".val{font-size:28px;font-weight:500;margin:8px 0 4px;font-family:'IBM Plex Mono',monospace}"
            ".lbl{font-size:12px;color:#83868c}"
            ".section{margin-top:24px}"
            ".section h3{color:#6a6d73;font-size:11px;letter-spacing:1.1px;text-transform:uppercase;font-weight:500;"
            "border-bottom:1px solid rgba(255,255,255,.07);padding-bottom:8px;margin-bottom:10px}"
            "table{width:100%;border-collapse:collapse;font-size:13px}"
            "td,th{padding:9px 14px 9px 0}"
            "th{color:#62656b;font-size:11px;letter-spacing:.05em;border-bottom:1px solid rgba(255,255,255,.06)}"
            "td{border-bottom:1px solid rgba(255,255,255,.04)}"
            "</style></head><body>"
            f"<h2 style='margin:0 0 4px;font-size:15px'>Aggregate — {n} recording(s)</h2>"
            f"<p style='color:#83868c;font-size:12px;margin:0 0 14px'>Mean ± SD across all recordings</p>"
            "<div class='grid'>"
            f"<div class='card'><div class='lbl'>Valid Gaze</div>"
            f"<div class='val' style='color:{_col(mv,80,60)}'>{mv:.1f}%{_pm(sv)}</div>"
            f"<div class='lbl'>mean across {n} recordings</div></div>"
            f"<div class='card'><div class='lbl'>Fixations Detected</div>"
            f"<div class='val' style='color:#6e8fd6'>{mf:.0f}{_pm(sf)}</div>"
            f"<div class='lbl'>mean per recording</div></div>"
            f"<div class='card'><div class='lbl'>Duration</div>"
            f"<div class='val'>{int(md)//60}:{int(md)%60:02d}</div>"
            f"<div class='lbl'>mean · ±{sd:.0f}s SD</div></div>"
            f"<div class='card'><div class='lbl'>AOI Coverage</div>"
            f"<div class='val' style='color:{_col(mc,80,60)}'>{mc:.1f}%{_pm(sc)}</div>"
            f"<div class='lbl'>frames within an AOI</div></div>"
            f"<div class='card'><div class='lbl'>Missing Gaze</div>"
            f"<div class='val' style='color:{missing_color}'>{mm:.1f}%</div>"
            f"<div class='lbl'>blinks / data loss</div></div>"
            f"<div class='card'><div class='lbl'>Recordings</div>"
            f"<div class='val'>{n}</div><div class='lbl'>analysed</div></div>"
            "</div>"
            "<div class='section'><h3>Mean AOI Dwell (% of frames)</h3>"
            "<table><tr><th>AOI</th><th>Mean % ± SD</th></tr>"
            f"{aoi_rows}</table></div>"
            "</body></html>"
        )
        view.setHtml(html)

    # ── Aggregate Tab 2: Dwell Time ───────────────────────────────────────────

    def _pop_agg_dwell_tab(self, csvs: list[tuple[str, pathlib.Path]], task_filter: str = "All Tasks"):
        view = self._agg_tab_views["dwell"]
        dwell_rows: list[dict] = []
        fps = 30.0
        for rec_name, csv_path in csvs:
            try:
                df, _b = self._agg_load(csv_path, task_filter)
                total = max(1, len(df))
                if total <= 1:
                    continue
                for aoi in AOI_NAMES:
                    cnt = int((df["_aoi"] == aoi).sum())
                    dwell_rows.append({"recording": rec_name, "AOI": aoi,
                                       "dwell_pct": cnt / total * 100, "dwell_s": cnt / fps})
            except Exception:
                pass
        if not dwell_rows:
            view.setHtml(self._no_data_html("No dwell data available."))
            return None
        ddf = pd.DataFrame(dwell_rows)
        agg = ddf.groupby("AOI")["dwell_pct"].agg(mean="mean", std="std", median="median").reset_index()
        agg["std"] = agg["std"].fillna(0)
        n_recs = ddf["recording"].nunique()
        colors = [AOI_COLORS_QT.get(a, QColor("#6e8fd6")).name() for a in agg["AOI"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agg["AOI"], y=agg["mean"],
            error_y=dict(type="data", array=agg["std"].tolist(), visible=(n_recs > 1)),
            marker_color=colors,
            customdata=np.stack([agg["median"], agg["std"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b><br>Mean: %{y:.1f}%<br>"
                "Median: %{customdata[0]:.1f}%<br>SD: %{customdata[1]:.1f}%<extra></extra>"
            ),
        ))
        for rec, grp in ddf.groupby("recording"):
            fig.add_trace(go.Scatter(
                x=grp["AOI"], y=grp["dwell_pct"],
                mode="markers",
                marker=dict(color="#FFFFFF", size=5, opacity=0.35),
                showlegend=False,
                hovertemplate=f"<b>%{{x}}</b><br>{rec}: %{{y:.1f}}%<extra></extra>",
            ))
        fig.update_layout(
            title=f"Gaze Dwell Time per AOI — {n_recs} recording(s)",
            xaxis_title="AOI", yaxis_title="Mean Dwell (%)",
            template="aoi_studio", autosize=True, margin=dict(t=50, b=40),
        )
        view.setHtml(self._fig_to_html(fig))
        return fig

    # ── Aggregate Tab 3: Fixation Metrics ─────────────────────────────────────

    def _pop_agg_fixation_tab(self, csvs: list[tuple[str, pathlib.Path]], task_filter: str = "All Tasks"):
        view = self._agg_tab_views["fixations"]
        rows: list[dict] = []
        for rec_name, csv_path in csvs:
            rec_dir = self._rec_dir_from_csv(csv_path)
            bounds = self._load_task_bounds(rec_dir, task_filter)
            for fix_path in [rec_dir / "aoi_results" / "raw" / "fixation_summary.csv",
                             rec_dir / "aoi_results" / "fixation_summary.csv"]:
                if fix_path.exists() and fix_path.stat().st_size > 0:
                    try:
                        fdf = pd.read_csv(fix_path)
                        if bounds is not None and "start_frame" in fdf.columns:
                            sf = pd.to_numeric(fdf["start_frame"], errors="coerce")
                            fdf = fdf[(sf >= bounds[0]) & (sf <= bounds[1])]
                        for aoi in AOI_NAMES:
                            aoi_rows = fdf[fdf["dominant_aoi"] == aoi]
                            rows.append({
                                "recording": rec_name, "aoi": aoi,
                                "count": len(aoi_rows),
                                "mean_dur": float(aoi_rows["duration_s"].mean())
                                            if len(aoi_rows) else 0.0,
                            })
                    except Exception:
                        pass
                    break
        if not rows:
            view.setHtml(self._no_data_html(
                "No fixation_summary.csv found.<br>"
                "Re-run Analysis on each recording to generate fixation data."))
            return None
        fdf2 = pd.DataFrame(rows)
        agg_c = (fdf2.groupby("aoi")["count"]
                 .agg(mean="mean", std="std").reset_index()
                 .set_index("aoi").reindex(AOI_NAMES).reset_index().fillna(0))
        agg_d = (fdf2.groupby("aoi")["mean_dur"]
                 .agg(mean="mean", std="std").reset_index()
                 .set_index("aoi").reindex(AOI_NAMES).reset_index().fillna(0))
        colors = [AOI_COLORS_QT.get(a, QColor("#6e8fd6")).name() for a in agg_c["aoi"]]
        n_recs = fdf2["recording"].nunique()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Mean Fixation Count per AOI",
                                            "Mean Fixation Duration (s)"))
        fig.add_trace(go.Bar(
            x=agg_c["aoi"], y=agg_c["mean"],
            error_y=dict(type="data", array=agg_c["std"].tolist(), visible=(n_recs > 1)),
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Mean count: %{y:.1f}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Bar(
            x=agg_d["aoi"], y=agg_d["mean"].round(3),
            error_y=dict(type="data", array=agg_d["std"].tolist(), visible=(n_recs > 1)),
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Mean dur: %{y:.3f}s<extra></extra>"),
            row=1, col=2)
        fig.update_layout(template="aoi_studio", autosize=True,
                          showlegend=False, margin=dict(t=50, b=40))
        view.setHtml(self._fig_to_html(fig))
        return fig

    # ── Aggregate Tab 4: AOI Heatmaps ─────────────────────────────────────────

    def _pop_agg_heatmap_tab(self, csvs: list[tuple[str, pathlib.Path]], task_filter: str = "All Tasks"):
        view = self._agg_tab_views["heatmaps"]
        aoi_gaze: dict[str, tuple[list, list]] = {a: ([], []) for a in AOI_NAMES}
        for _rec_name, csv_path in csvs:
            try:
                df, _b = self._agg_load(csv_path, task_filter)
                if "gaze_on_aoi_x" not in df.columns or "gaze_on_aoi_y" not in df.columns:
                    continue
                for aoi in AOI_NAMES:
                    sub = df[df["_aoi"] == aoi]
                    if "is_fixation" in sub.columns:
                        fix = sub[sub["is_fixation"].astype(str).str.lower() == "true"]
                        if len(fix) >= 10:
                            sub = fix
                    x = pd.to_numeric(sub["gaze_on_aoi_x"], errors="coerce").dropna().values
                    y = pd.to_numeric(sub["gaze_on_aoi_y"], errors="coerce").dropna().values
                    mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
                    aoi_gaze[aoi][0].extend(x[mask].tolist())
                    aoi_gaze[aoi][1].extend(y[mask].tolist())
            except Exception:
                pass
        aois_data = [(aoi, np.array(xs), np.array(ys))
                     for aoi, (xs, ys) in aoi_gaze.items() if len(xs) >= 10]
        if not aois_data:
            view.setHtml(self._no_data_html(
                "No surface coordinate data available.<br>"
                "Heatmaps require 3D surface detection (gaze_on_aoi_x/y columns)."))
            return None
        from scipy.ndimage import gaussian_filter as _gf
        n = len(aois_data)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[a[0] for a in aois_data])
        for i, (aoi_name, x, y) in enumerate(aois_data):
            r, c = i // cols + 1, i % cols + 1
            color = AOI_COLORS_QT.get(aoi_name, QColor("#6e8fd6"))
            rv, gv, bv = color.red(), color.green(), color.blue()
            H, xe, ye = np.histogram2d(x, y, bins=25, range=[[0, 1], [0, 1]])
            Hs = _gf(H.T, sigma=1.5)
            Hn = Hs / Hs.max() if Hs.max() > 0 else Hs
            cs = [[0.0, "rgba(0,0,0,0)"],
                  [0.3, f"rgba({rv},{gv},{bv},0.3)"],
                  [1.0, f"rgba({rv},{gv},{bv},1.0)"]]
            fig.add_trace(go.Heatmap(
                z=Hn, x=xe[:-1] + (xe[1] - xe[0]) / 2, y=ye[:-1] + (ye[1] - ye[0]) / 2,
                colorscale=cs, showscale=False,
                hovertemplate=f"X:%{{x:.2f}} Y:%{{y:.2f}}<br>Density:%{{z:.2f}}<extra>{aoi_name}</extra>",
            ), row=r, col=c)
        fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False)
        fig.update_layout(
            title=f"Pooled Spatial Gaze Distribution — {len(csvs)} recording(s)",
            template="aoi_studio",
            height=max(360, 340 * rows),
            margin=dict(t=60, b=20))
        view.setHtml(self._fig_to_html(fig))
        return fig

    # ── Aggregate Tab 5: Transition Matrix ────────────────────────────────────

    def _pop_agg_transition_tab(self, csvs: list[tuple[str, pathlib.Path]], task_filter: str = "All Tasks"):
        view = self._agg_tab_views["transitions"]
        matrices: list[pd.DataFrame] = []
        for _rec_name, csv_path in csvs:
            try:
                df, _b = self._agg_load(csv_path, task_filter)
                labels = df["_aoi"]
                matrix = pd.DataFrame(0, index=AOI_NAMES, columns=AOI_NAMES, dtype=int)
                arr = labels.values
                for i in range(len(arr) - 1):
                    src, dst = str(arr[i]), str(arr[i + 1])
                    if src != dst and src in matrix.index and dst in matrix.columns:
                        matrix.loc[src, dst] += 1
                row_sums = matrix.sum(axis=1)
                normed = (matrix.div(row_sums.replace(0, np.nan), axis=0)
                          .fillna(0) * 100).round(1)
                matrices.append(normed)
            except Exception:
                pass
        if not matrices:
            view.setHtml(self._no_data_html("No transition data available."))
            return None
        avg = matrices[0].copy().astype(float)
        for m in matrices[1:]:
            avg = avg.add(m.astype(float), fill_value=0.0)
        avg = (avg / len(matrices)).round(1)
        if avg.values.sum() == 0:
            view.setHtml(self._no_data_html("No AOI transitions found across recordings."))
            return None
        fig = go.Figure(go.Heatmap(
            z=avg.values.tolist(),
            x=list(avg.columns),
            y=list(avg.index),
            colorscale="Blues",
            text=avg.values.round(1).tolist(),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="From: %{y}<br>To: %{x}<br>%{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=f"Mean AOI Transition Probabilities — {len(csvs)} recording(s)",
            template="aoi_studio",
            height=max(360, 50 * len(AOI_NAMES) + 160),
            xaxis_title="To AOI", yaxis_title="From AOI",
            margin=dict(t=60, b=60, l=110, r=40))
        view.setHtml(self._fig_to_html(fig))
        return fig

    # ── Aggregate Tab 6: Learning Curve ───────────────────────────────────────

    def _pop_agg_learning_tab(self, csvs: list[tuple[str, pathlib.Path]],
                              task_filter: str):
        view = self._agg_tab_views["learning"]
        task_rows: list[dict] = []
        fps = 30.0
        for rec_name, csv_path in csvs:
            rec_dir = self._rec_dir_from_csv(csv_path)
            task_path = None
            for p in [rec_dir / "aoi_results" / "tasks.json",
                      rec_dir / "aoi_results" / "raw" / "tasks.json"]:
                if p.exists():
                    task_path = p
                    break
            if task_path is None:
                continue
            try:
                tasks = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                df_head = pd.read_csv(csv_path, usecols=["time_s"])
                times = pd.to_numeric(df_head["time_s"], errors="coerce").dropna()
                if len(times) > 1:
                    dur = float(times.iloc[-1] - times.iloc[0])
                    if dur > 0:
                        fps = max(1.0, (len(times) - 1) / dur)
            except Exception:
                pass
            for t_name, t_data in tasks.items():
                s, e = t_data.get("start"), t_data.get("end")
                if s is None or e is None or e <= s:
                    continue
                digits = "".join(c for c in t_name if c.isdigit())
                task_rows.append({
                    "recording": rec_name, "task": t_name,
                    "task_idx": int(digits) if digits else 0,
                    "dur_s": (e - s) / fps,
                })
        if not task_rows:
            view.setHtml(self._no_data_html(
                "No task annotations found.<br>"
                "Annotate tasks T1–T10 in the Studio tab and click Save in the Tasks section."))
            return None
        tdf = pd.DataFrame(task_rows)
        if task_filter != "All Tasks":
            tdf = tdf[tdf["task"] == task_filter]
        if tdf.empty:
            view.setHtml(self._no_data_html(f"No data for filter: {task_filter}"))
            return None
        n_recs = tdf["recording"].nunique()
        agg = (tdf.groupby("task_idx")["dur_s"]
               .agg(mean="mean", std="std", median="median", n="count")
               .reset_index().sort_values("task_idx"))
        agg["std"] = agg["std"].fillna(0)
        agg_err = self._agg_error_series(csvs)
        has_err = self._has_errors(agg_err)
        if has_err:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                                row_heights=[0.55, 0.45],
                                subplot_titles=(f"Learning Curve — {n_recs} recording(s)",
                                                "Mean Assembly Errors per Task"))
        else:
            fig = make_subplots(rows=1, cols=1)
        if n_recs > 1:
            x_l = agg["task_idx"].tolist()
            fig.add_trace(go.Scatter(
                x=x_l + x_l[::-1],
                y=(agg["mean"] + agg["std"]).tolist() +
                  (agg["mean"] - agg["std"]).tolist()[::-1],
                fill="toself", fillcolor="rgba(88,166,255,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            for rn, rg in tdf.groupby("recording"):
                rg = rg.sort_values("task_idx")
                fig.add_trace(go.Scatter(
                    x=rg["task_idx"], y=rg["dur_s"], mode="lines+markers",
                    line=dict(width=1, color="#3a3d43"), marker=dict(size=4),
                    name=rn, showlegend=False,
                ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=agg["task_idx"], y=agg["mean"], mode="lines+markers",
            line=dict(width=3, color="#6e8fd6"), marker=dict(size=8, color="#6e8fd6"),
            name="Mean", showlegend=False,
            customdata=np.stack([agg["median"], agg["std"], agg["n"]], axis=1),
            hovertemplate=("Task %{x}<br>Mean: %{y:.1f}s<br>Median: %{customdata[0]:.1f}s<br>"
                           "SD: %{customdata[1]:.1f}s<br>n=%{customdata[2]}<extra></extra>"),
        ), row=1, col=1)
        fig.update_yaxes(title_text="Duration (s)", row=1, col=1)
        layout = dict(template="aoi_studio", autosize=True, margin=dict(t=60, b=40))
        if has_err:
            buttons = self._add_error_traces(fig, agg_err, row=2, n_prefix=len(fig.data))
            fig.update_xaxes(title_text="Task", dtick=1, row=2, col=1)
            fig.update_yaxes(title_text="Mean error count", row=2, col=1)
            layout["updatemenus"] = [dict(type="buttons", direction="right", x=0.0, y=1.10,
                                          xanchor="left", active=2, buttons=buttons)]
            layout["height"] = 640
        else:
            fig.update_xaxes(title_text="Task Number", dtick=1, row=1, col=1)
            fig.update_layout(showlegend=False,
                title=f"Learning Curve — Task Duration over Repetitions — {n_recs} recording(s)")
        fig.update_layout(**layout)
        view.setHtml(self._fig_to_html(fig))
        return fig

    # ── Build figures ─────────────────────────────────────────────────────────

    def _build_figures(
        self,
        csvs: list[tuple[str, pathlib.Path]],
        folder: str,
        task_filter: str,
    ) -> tuple[list[tuple[str, "go.Figure"]], str]:
        """Returns ([(slug, fig), ...], stats_html)."""
        dfs: list[pd.DataFrame] = []
        for rec_name, p in csvs:
            try:
                df = pd.read_csv(p)
                df["recording"] = rec_name
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            return [], ""

        combined = pd.concat(dfs, ignore_index=True)
        lbl_col  = "final_primary_aoi" if "final_primary_aoi" in combined.columns else "primary_aoi"
        combined["aoi"] = combined[lbl_col].fillna(NONE_LABEL).astype(str)
        fps = 30.0

        # Dwell stats per recording
        dwell_rows: list[dict] = []
        for rec_name, grp in combined.groupby("recording"):
            total = max(1, len(grp))
            for aoi in AOI_NAMES:
                cnt = int((grp["aoi"] == aoi).sum())
                dwell_rows.append({
                    "recording": rec_name, "AOI": aoi,
                    "dwell_s": cnt / fps, "dwell_pct": cnt / total * 100,
                })
        dwell_df = pd.DataFrame(dwell_rows)

        result: list[tuple[str, go.Figure]] = []

        # Chart 1: AOI dwell bar
        if not dwell_df.empty:
            agg = dwell_df.groupby("AOI")["dwell_pct"].agg(["mean", "std", "median"]).reset_index()
            agg["std"] = agg["std"].fillna(0)
            n_recs = dwell_df["recording"].nunique()
            colors_bar = [AOI_COLORS_QT.get(a, QColor("#6b6e74")).name() for a in agg["AOI"]]
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=agg["AOI"], y=agg["mean"],
                error_y=dict(type="data", array=agg["std"].tolist(), visible=(n_recs > 1)),
                marker_color=colors_bar,
                name="Mean dwell %",
                customdata=np.stack([agg["median"], agg["std"]], axis=1),
                hovertemplate=(
                    "<b>%{x}</b><br>Mean: %{y:.1f}%<br>"
                    "Median: %{customdata[0]:.1f}%<br>"
                    "Std: %{customdata[1]:.1f}%<extra></extra>"
                ),
            ))
            title_suffix = f" — {folder}" + (f" · {n_recs} recordings" if n_recs > 1 else "")
            fig1.update_layout(
                title=f"AOI Dwell Time{title_suffix}",
                xaxis_title="AOI", yaxis_title="Mean Dwell (%)",
                template="aoi_studio", height=380, margin=dict(t=50, b=40),
            )
            result.append(("01_aoi_dwell", fig1))

        # Chart 2: Learning curve
        task_rows: list[dict] = []
        for rec_name, p in csvs:
            task_path = p.parent.parent / "tasks.json"
            if not task_path.exists():
                task_path = p.parent / "tasks.json"
            if not task_path.exists():
                continue
            try:
                tasks = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for t_name, t_data in tasks.items():
                s, e = t_data.get("start"), t_data.get("end")
                if s is None or e is None or e <= s:
                    continue
                digits = "".join(c for c in t_name if c.isdigit())
                task_rows.append({
                    "recording": rec_name, "task": t_name,
                    "task_idx": int(digits) if digits else 0,
                    "dur_s": (e - s) / fps,
                })

        if task_rows:
            tdf = pd.DataFrame(task_rows)
            if task_filter != "All Tasks":
                tdf = tdf[tdf["task"] == task_filter]
            if not tdf.empty:
                agg_t = tdf.groupby("task_idx")["dur_s"].agg(
                    mean="mean", std="std", median="median", n="count"
                ).reset_index().sort_values("task_idx")
                agg_t["std"] = agg_t["std"].fillna(0)
                fig2 = go.Figure()
                n_recs2 = tdf["recording"].nunique()
                if n_recs2 > 1:
                    fig2.add_trace(go.Scatter(
                        x=pd.concat([agg_t["task_idx"], agg_t["task_idx"][::-1]]),
                        y=pd.concat([agg_t["mean"] + agg_t["std"],
                                     (agg_t["mean"] - agg_t["std"])[::-1]]),
                        fill="toself", fillcolor="rgba(88,166,255,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    for rn, rg in tdf.groupby("recording"):
                        rg = rg.sort_values("task_idx")
                        fig2.add_trace(go.Scatter(
                            x=rg["task_idx"], y=rg["dur_s"],
                            mode="lines+markers",
                            line=dict(width=1, color="#3a3d43"),
                            marker=dict(size=4),
                            name=rn, showlegend=False,
                        ))
                fig2.add_trace(go.Scatter(
                    x=agg_t["task_idx"], y=agg_t["mean"],
                    mode="lines+markers",
                    line=dict(width=3, color="#6e8fd6"),
                    marker=dict(size=8, color="#6e8fd6"),
                    name="Mean",
                    customdata=np.stack([agg_t["median"], agg_t["std"], agg_t["n"]], axis=1),
                    hovertemplate=(
                        "Task %{x}<br>Mean: %{y:.1f}s<br>"
                        "Median: %{customdata[0]:.1f}s<br>"
                        "Std: %{customdata[1]:.1f}s<br>"
                        "N=%{customdata[2]}<extra></extra>"
                    ),
                ))
                fig2.update_layout(
                    title="Learning Curve — Task Duration over Tasks",
                    xaxis=dict(title="Task Number", dtick=1),
                    yaxis_title="Duration (s)",
                    template="aoi_studio", height=360, margin=dict(t=50, b=40),
                )
                result.append(("02_learning_curve", fig2))

        # Chart 3: Per-recording heatmap (>1 recording only)
        if not dwell_df.empty and dwell_df["recording"].nunique() > 1:
            pivot = dwell_df.pivot(index="recording", columns="AOI",
                                   values="dwell_pct").fillna(0)
            fig3 = px.imshow(
                pivot, aspect="auto", color_continuous_scale="Blues",
                title="Dwell % per Recording (heatmap)",
                labels=dict(x="AOI", y="Recording", color="Dwell %"),
            )
            fig3.update_layout(
                template="aoi_studio",
                height=max(220, 36 * len(pivot) + 100),
                margin=dict(t=50, b=40),
            )
            result.append(("03_dwell_heatmap", fig3))

        # Stats table HTML
        stats_html = ""
        if not dwell_df.empty:
            agg_tbl = dwell_df.groupby("AOI")["dwell_pct"].agg(
                ["mean", "median", "std", "min", "max"]
            ).round(1).reset_index()
            _fallback = QColor("#aaa")
            rows = "".join(
                "<tr>"
                f"<td style='color:{AOI_COLORS_QT.get(r.AOI, _fallback).name()}'>"
                f"<b>{r.AOI}</b></td>"
                f"<td>{r.mean}%</td><td>{r.median}%</td>"
                f"<td>{r.std}%</td><td>{r.min}%</td><td>{r.max}%</td></tr>"
                for r in agg_tbl.itertuples()
            )
            stats_html = (
                "<h4 style='color:#9b9ea4;margin:16px 0 8px'>AOI Dwell Statistics</h4>"
                "<table style='border-collapse:collapse;font-size:12px;width:100%'>"
                "<tr style='border-bottom:1px solid rgba(255,255,255,.07);color:#9b9ea4'>"
                "<th style='text-align:left;padding:4px 12px 4px 0'>AOI</th>"
                "<th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>"
                + rows + "</table>"
            )

        return result, stats_html

    # ── Render HTML from figures ──────────────────────────────────────────────

    def _render_html(self, figs: list[tuple[str, "go.Figure"]], stats_html: str) -> str:
        if not figs and not stats_html:
            return "<body style='background:#0e0f11;color:#e7e8ea;padding:20px'>No readable data.</body>"
        parts: list[str] = []
        first = True
        for _slug, fig in figs:
            parts.append(fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if first else False,
            ))
            first = False
        body = "<hr style='border-color:rgba(255,255,255,.07);margin:16px 0'>".join(parts) + stats_html
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>body{background:#0e0f11;color:#e7e8ea;"
            "font-family:'Hanken Grotesk',sans-serif;margin:0;padding:12px}"
            "td,th{padding:4px 12px 4px 0}</style>"
            f"</head><body>{body}</body></html>"
        )

    # ── Export PNGs ───────────────────────────────────────────────────────────

    def _export_png(self) -> None:
        if self._stack.currentIndex() == 1:
            self._export_png_single()
        else:
            self._export_png_aggregate()

    def _export_png_aggregate(self) -> None:
        src, rec_filter, csvs = self._collect_csvs()
        if src is None or not csvs:
            self._status.setText("Load charts first.")
            return
        task_filter = self._task_combo.currentText()
        self._status.setText("Rendering PNGs…")
        QApplication.processEvents()
        # Rebuild all Plotly-based aggregate figures for export
        named_figs: list[tuple[str, object]] = []
        for slug, method, args in [
            ("01_agg_dwell",       self._pop_agg_dwell_tab,      (csvs,)),
            ("02_agg_fixations",   self._pop_agg_fixation_tab,   (csvs,)),
            ("03_agg_heatmaps",    self._pop_agg_heatmap_tab,    (csvs,)),
            ("04_agg_transitions", self._pop_agg_transition_tab, (csvs,)),
            ("05_agg_learning",    self._pop_agg_learning_tab,   (csvs, task_filter)),
        ]:
            try:
                fig = method(*args)
                if fig is not None:
                    named_figs.append((slug, fig))
            except Exception:
                pass
        if not named_figs:
            self._status.setText("No charts to export.")
            return
        out_dir = (src / reporting.SUMMARY_DIR_NAME) if rec_filter is None \
                  else (rec_filter / "aoi_results" / "charts")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[pathlib.Path] = []
        for slug, fig in named_figs:
            path = out_dir / f"{slug}.png"
            try:
                fig.write_image(str(path), width=1400, height=fig.layout.height or 400, scale=2)
                saved.append(path)
            except Exception as exc:
                self._status.setText(f"PNG render failed: {exc}")
                return
        self._status.setText(f"Saved {len(saved)} PNG(s) → {out_dir}")
        QMessageBox.information(self, APP_TITLE,
            f"Saved {len(saved)} chart(s) to:\n{out_dir}\n\n"
            + "\n".join(p.name for p in saved))

    def _export_png_single(self) -> None:
        _, rec_filter, csvs = self._collect_csvs()
        if rec_filter is None or not csvs:
            self._status.setText("Select a single recording first.")
            return
        task_filter = self._task_combo.currentText()
        self._status.setText("Rendering PNGs…")
        QApplication.processEvents()

        try:
            df = pd.read_csv(csvs[0][1])
        except Exception as exc:
            self._status.setText(f"Could not read CSV: {exc}")
            return

        lbl_col = "final_primary_aoi" if "final_primary_aoi" in df.columns else "primary_aoi"
        df["_aoi"] = df[lbl_col].fillna(NONE_LABEL).astype(str)
        total = len(df)
        fps = 30.0

        out_dir = rec_filter / "aoi_results" / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[pathlib.Path] = []

        # Build and save each chart
        def _save(slug, fig):
            p = out_dir / f"{slug}.png"
            try:
                fig.write_image(str(p), width=1400, height=fig.layout.height or 400, scale=2)
                saved.append(p)
            except Exception as exc:
                self._status.setText(f"PNG failed for {slug}: {exc}")

        # Dwell chart
        rows = [{"aoi": aoi, "dwell_pct": int((df["_aoi"] == aoi).sum()) / total * 100}
                for aoi in AOI_NAMES]
        ddf   = pd.DataFrame(rows)
        colors = [AOI_COLORS_QT.get(a, QColor("#6e8fd6")).name() for a in ddf["aoi"]]
        fig_d = go.Figure(go.Bar(x=ddf["aoi"], y=ddf["dwell_pct"], marker_color=colors))
        fig_d.update_layout(title="Dwell Time per AOI", template="aoi_studio", height=400)
        _save("01_dwell", fig_d)

        # Transition matrix
        labels = df["_aoi"].values
        matrix = pd.DataFrame(0, index=AOI_NAMES, columns=AOI_NAMES, dtype=int)
        for i in range(len(labels) - 1):
            src, dst = str(labels[i]), str(labels[i + 1])
            if src != dst and src in matrix.index and dst in matrix.columns:
                matrix.loc[src, dst] += 1
        row_sums = matrix.sum(axis=1)
        normed = (matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0) * 100).round(1)
        if normed.values.sum() > 0:
            fig_t = go.Figure(go.Heatmap(z=normed.values.tolist(),
                                         x=list(normed.columns), y=list(normed.index),
                                         colorscale="Blues"))
            fig_t.update_layout(title="Transition Matrix", template="aoi_studio", height=400)
            _save("02_transitions", fig_t)

        # Heatmaps (if data exists)
        if "gaze_on_aoi_x" in df.columns and "gaze_on_aoi_y" in df.columns:
            from scipy.ndimage import gaussian_filter as _gf
            aois_data = []
            for aoi in AOI_NAMES:
                sub = df[df["_aoi"] == aoi]
                x = pd.to_numeric(sub["gaze_on_aoi_x"], errors="coerce").dropna().values
                y = pd.to_numeric(sub["gaze_on_aoi_y"], errors="coerce").dropna().values
                m = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
                if m.sum() >= 10:
                    aois_data.append((aoi, x[m], y[m]))
            if aois_data:
                n = len(aois_data)
                cols = min(3, n)
                rows_n = (n + cols - 1) // cols
                fig_h = make_subplots(rows=rows_n, cols=cols,
                                      subplot_titles=[a[0] for a in aois_data])
                for i, (aoi_name, x, y) in enumerate(aois_data):
                    r, c = i // cols + 1, i % cols + 1
                    color = AOI_COLORS_QT.get(aoi_name, QColor("#6e8fd6"))
                    rv, gv, bv = color.red(), color.green(), color.blue()
                    H, xe, ye = np.histogram2d(x, y, bins=25, range=[[0, 1], [0, 1]])
                    Hn = _gf(H.T, sigma=1.5)
                    if Hn.max() > 0:
                        Hn /= Hn.max()
                    cs = [[0.0, "rgba(0,0,0,0)"],
                          [0.3, f"rgba({rv},{gv},{bv},0.3)"],
                          [1.0, f"rgba({rv},{gv},{bv},1.0)"]]
                    fig_h.add_trace(go.Heatmap(
                        z=Hn, x=xe[:-1], y=ye[:-1], colorscale=cs, showscale=False
                    ), row=r, col=c)
                fig_h.update_layout(title="AOI Heatmaps", template="aoi_studio",
                                    height=max(360, 340 * rows_n))
                _save("03_heatmaps", fig_h)

        self._status.setText(f"Saved {len(saved)} PNG(s) → {out_dir}")
        QMessageBox.information(self, APP_TITLE,
            f"Saved {len(saved)} chart(s) to:\n{out_dir}\n\n"
            + "\n".join(p.name for p in saved))

    # ── Export workbook ───────────────────────────────────────────────────────

    def _export_workbook(self) -> None:
        src = self._current_source()
        if src is None:
            return
        try:
            out = reporting.generate_master_outputs(src)
            QMessageBox.information(
                self, APP_TITLE,
                f"Workbook saved:\n{out.workbook_path}\n\n"
                f"Recordings: {out.recording_count}  Tasks: {out.task_count}",
            )
        except Exception as exc:
            QMessageBox.warning(self, APP_TITLE, f"Export failed:\n{exc}")


# ─── Comparison tab ───────────────────────────────────────────────────────────

_REC_TAB_DEFS = [
    ("dq",           "Data Quality"),
    ("dwell",        "Dwell Time"),
    ("fixations",    "Fixation Metrics"),
    ("heatmaps",     "AOI Heatmaps"),
    ("transitions",  "Transition Matrix"),
    ("learning",     "Learning Curve"),
]

# Roll-up of the 5 manual error fields into the graph's Type / Position axes.
#   Type     = wrong identity  (component Type / cable Colour)
#   Position = wrong placement (component Number+Orientation / cable Hole)
def _error_series(errs: dict) -> dict:
    """errs = {task_name: {field: count}} -> per task_idx sums for each series."""
    out: dict[int, dict[str, float]] = {}
    for tname, v in (errs or {}).items():
        digits = "".join(c for c in str(tname) if c.isdigit())
        if not digits:
            continue
        ti = int(digits)
        g = out.setdefault(ti, {"comp_type": 0, "comp_pos": 0, "cable_type": 0, "cable_pos": 0})
        g["comp_type"]  += float(v.get("comp_type", 0) or 0)
        g["comp_pos"]   += float(v.get("comp_number", 0) or 0) + float(v.get("comp_orientation", 0) or 0)
        g["cable_type"] += float(v.get("cable_colour", 0) or 0)
        g["cable_pos"]  += float(v.get("cable_position", 0) or 0)
    return out


def _load_errors_json(rec_dir: pathlib.Path) -> dict:
    for p in [rec_dir / "aoi_results" / "errors.json",
              rec_dir / "aoi_results" / "raw" / "errors.json"]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}

_TAB_DEFS = [
    ("01_dwell_pct",          "Dwell %"),
    ("02_fixation_count",     "Fixation Count"),
    ("03_fixation_duration",  "Fixation Duration"),
    ("04_entropy_transition", "Entropy & Transitions"),
    ("05_transition_matrices","Transition Matrices"),
    ("06_learning_curves",    "Learning Curves"),
    ("__stats__",             "Statistics"),
]

_CHART_PLACEHOLDER = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<style>body{background:#0e0f11;color:#3a3d43;font-family:'Hanken Grotesk',sans-serif;"
    "display:flex;align-items:center;justify-content:center;"
    "height:95vh;margin:0;font-size:14px}</style>"
    "</head><body>Run comparison to populate this chart.</body></html>"
)

class ComparisonWidget(QWidget):
    """
    Condition Comparison tab (NonGamified vs Gamified).
    Runs reporting.compare_conditions() in a background thread and renders
    7 Plotly charts + a statistical summary table.
    """

    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[ComparisonWorker] = None
        self._last_report = None
        self._build_ui()
        self._populate_combos()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # ── Left sidebar: all controls, stacked vertically ──────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("leftPanel")
        sidebar.setFixedWidth(224)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(16, 18, 16, 18)
        side_l.setSpacing(10)

        self._cond_a_combo = QComboBox()
        self._cond_a_combo.setToolTip("Condition A (e.g. NonGamified)")

        self._cond_b_combo = QComboBox()
        self._cond_b_combo.setToolTip("Condition B (e.g. Gamified)")

        self._run_btn = QPushButton("Compare")
        self._run_btn.setObjectName("primaryButton")
        self._run_btn.setToolTip("Run the statistical comparison between condition A and B")

        self._png_btn = QPushButton("📷  PNGs")
        self._png_btn.setToolTip("Save every comparison chart as a PNG image")

        side_l.addWidget(QLabel("Condition A:"))
        side_l.addWidget(self._cond_a_combo)
        side_l.addWidget(QLabel("Condition B:"))
        side_l.addWidget(self._cond_b_combo)

        _div1 = QFrame(); _div1.setFrameShape(QFrame.HLine); _div1.setObjectName("divider")
        side_l.addWidget(_div1)

        side_l.addWidget(self._run_btn)
        side_l.addWidget(self._png_btn)
        side_l.addStretch(1)

        self._status = QLabel("Select two conditions and click Compare.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {Theme.TEXT_FAINT}; font-size: 12px;")
        side_l.addWidget(self._status)

        layout.addWidget(sidebar)

        # ── Right: chart area fills all remaining space ─────────────────────────
        # Sub-tabs — one per chart
        self._chart_tabs = QTabWidget()
        self._tab_views: dict[str, QWebEngineView] = {}
        for slug, label in _TAB_DEFS:
            view = QWebEngineView()
            view.setHtml(_CHART_PLACEHOLDER)
            self._chart_tabs.addTab(view, f"  {label}  ")
            self._tab_views[slug] = view

        layout.addWidget(self._chart_tabs, 1)

        self._run_btn.clicked.connect(self._run)
        self._png_btn.clicked.connect(self._export_png)

    def _populate_combos(self) -> None:
        self._cond_a_combo.clear()
        self._cond_b_combo.clear()
        dirs = list_source_folders()
        for d in dirs:
            self._cond_a_combo.addItem(d.name, userData=d)
            self._cond_b_combo.addItem(d.name, userData=d)
        if len(dirs) >= 2:
            self._cond_b_combo.setCurrentIndex(1)

    # ── Run ──────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        cond_a: Optional[pathlib.Path] = self._cond_a_combo.currentData()
        cond_b: Optional[pathlib.Path] = self._cond_b_combo.currentData()
        if cond_a is None or cond_b is None:
            return
        if cond_a == cond_b:
            self._status.setText("Select two different condition folders.")
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._run_btn.setEnabled(False)
        self._status.setText("Loading recordings and computing statistics…")
        QApplication.processEvents()
        worker = ComparisonWorker(cond_a, cond_b)
        worker.signals.finished.connect(self._on_done)
        worker.signals.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_done(self) -> None:
        self._run_btn.setEnabled(True)
        report = self._worker.result
        self._last_report = report
        figs = self._build_figures(report)
        fig_dict = {slug: fig for slug, fig in figs}
        for slug, view in self._tab_views.items():
            if slug == "__stats__":
                view.setHtml(self._stats_page_html(report))
            elif slug in fig_dict:
                view.setHtml(self._fig_to_html(fig_dict[slug]))
        scipy_note = "scipy ✓" if reporting._SCIPY_OK else "⚠ scipy missing — p-values unavailable"
        self._status.setText(
            f"{report.condition_a}: {report.n_a} recording(s)   ·   "
            f"{report.condition_b}: {report.n_b} recording(s)   ·   {scipy_note}"
        )

    def _on_failed(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")
        QMessageBox.warning(self, APP_TITLE, f"Comparison failed:\n{msg}")

    # ── Figure building ───────────────────────────────────────────────────────

    @staticmethod
    def _sig_marker(p) -> str:
        try:
            pf = float(p)
        except (TypeError, ValueError):
            return ""
        if math.isnan(pf):
            return ""
        if pf < 0.001: return "***"
        if pf < 0.01:  return "**"
        if pf < 0.05:  return "*"
        return ""

    def _build_figures(self, report) -> list[tuple[str, object]]:
        result: list[tuple[str, object]] = []
        cond_a, cond_b = report.condition_a, report.condition_b
        col_a,  col_b  = _COND_COLORS
        fill_a, fill_b = _COND_FILL_COLORS

        # ── Grouped-bar helper (charts 1-3) ──────────────────────────────────
        def _grouped_bar(metric_label: str, y_title: str, slug: str,
                         rec_attr: str, title: str) -> None:
            df = report.aoi_stats[report.aoi_stats["metric"] == metric_label].copy()
            if df.empty:
                return
            # Skip if all zeros (fixation data not yet available)
            if df[f"{cond_a}_mean"].sum() + df[f"{cond_b}_mean"].sum() == 0:
                return
            fig = go.Figure()
            for cond, col, recs in [
                (cond_a, col_a, report.recordings_a),
                (cond_b, col_b, report.recordings_b),
            ]:
                fig.add_trace(go.Bar(
                    x=df["aoi"], y=df[f"{cond}_mean"],
                    error_y=dict(type="data", array=df[f"{cond}_std"].tolist(), visible=True),
                    name=cond, marker_color=col,
                    hovertemplate=f"<b>%{{x}}</b><br>Mean: %{{y:.3f}}<extra>{cond}</extra>",
                ))
                # Individual recording dots
                for aoi in df["aoi"]:
                    pts = [getattr(r, rec_attr).get(aoi, 0.0) for r in recs]
                    if pts:
                        fig.add_trace(go.Scatter(
                            x=[aoi] * len(pts), y=pts,
                            mode="markers",
                            marker=dict(color=col, size=6, opacity=0.55),
                            showlegend=False, hoverinfo="skip",
                        ))
            # Significance markers
            for _, row in df.iterrows():
                marker = self._sig_marker(row["p_value"])
                if marker:
                    max_y = max(
                        float(row[f"{cond_a}_mean"]) + float(row[f"{cond_a}_std"]),
                        float(row[f"{cond_b}_mean"]) + float(row[f"{cond_b}_std"]),
                    )
                    fig.add_annotation(
                        x=row["aoi"], y=max_y * 1.08, text=marker,
                        showarrow=False, font=dict(size=13, color="white"),
                        xref="x", yref="y",
                    )
            fig.update_layout(
                title=f"{title}   (* p<0.05  ** p<0.01  *** p<0.001)",
                barmode="group", template="aoi_studio", autosize=True,
                xaxis_title="AOI", yaxis_title=y_title,
                legend=dict(orientation="h", y=1.08),
                margin=dict(t=65, b=40),
            )
            result.append((slug, fig))

        _grouped_bar("Dwell %",                   "Mean Dwell (%)",              "01_dwell_pct",         "dwell_pct",      "AOI Gaze Dwell Time")
        _grouped_bar("Fixation Count",             "Mean Fixation Count",         "02_fixation_count",    "fixation_count", "Fixation Count per AOI")
        _grouped_bar("Mean Fixation Duration (s)", "Mean Fixation Duration (s)",  "03_fixation_duration", "mean_fix_dur",   "Mean Fixation Duration per AOI")

        # ── Chart 4: Gaze Entropy + Transition Rate ───────────────────────────
        if not report.overall_stats.empty:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Gaze Entropy (bits)", "Transition Rate (per sec)"),
            )
            overall_metrics = [
                ("Gaze Entropy (bits)",        "entropy",         1),
                ("Transition Rate (per sec)",   "transition_rate", 2),
            ]
            first_legend = True
            for label, attr, col_n in overall_metrics:
                ov_rows = report.overall_stats[report.overall_stats["metric"] == label]
                if ov_rows.empty:
                    continue
                ov = ov_rows.iloc[0]
                for cond, col, recs in [
                    (cond_a, col_a, report.recordings_a),
                    (cond_b, col_b, report.recordings_b),
                ]:
                    show_leg = first_legend
                    fig.add_trace(go.Bar(
                        x=[cond], y=[ov[f"{cond}_mean"]],
                        error_y=dict(type="data", array=[ov[f"{cond}_std"]], visible=True),
                        name=cond, marker_color=col, showlegend=show_leg,
                    ), row=1, col=col_n)
                    pts = [getattr(r, attr) for r in recs]
                    if pts:
                        fig.add_trace(go.Scatter(
                            x=[cond] * len(pts), y=pts,
                            mode="markers",
                            marker=dict(color=col, size=8, opacity=0.6),
                            showlegend=False, hoverinfo="skip",
                        ), row=1, col=col_n)
                first_legend = False
            fig.update_layout(
                barmode="group", template="aoi_studio", autosize=True,
                margin=dict(t=60, b=40),
                legend=dict(orientation="h", y=1.12),
            )
            result.append(("04_entropy_transition", fig))

        # ── Chart 5: Transition matrices ──────────────────────────────────────
        tm_a, tm_b = report.trans_matrix_a, report.trans_matrix_b
        if not tm_a.empty and not tm_b.empty:
            zmax = max(float(tm_a.values.max()), float(tm_b.values.max()), 1.0)
            n_aoi = len(tm_a)
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    f"Transitions — {cond_a}",
                    f"Transitions — {cond_b}",
                ),
            )
            for col_n, tm in enumerate([tm_a, tm_b], start=1):
                fig.add_trace(go.Heatmap(
                    z=tm.values.tolist(),
                    x=list(tm.columns),
                    y=list(tm.index),
                    colorscale="Blues", zmin=0, zmax=zmax,
                    showscale=(col_n == 2),
                    hovertemplate="From: %{y}<br>To: %{x}<br>%{z:.1f}%<extra></extra>",
                ), row=1, col=col_n)
            fig.update_layout(
                title="AOI Transition Probabilities — row→column (% of outgoing transitions)",
                template="aoi_studio",
                height=max(320, 42 * n_aoi + 130),
                margin=dict(t=65, b=50, l=90, r=60),
            )
            result.append(("05_transition_matrices", fig))

        # ── Chart 6: Learning curves ──────────────────────────────────────────
        ldf = report.learning_df
        if not ldf.empty:
            metric_order = ["duration_s", "board_dwell_pct", "screen_dwell_pct", "entropy"]
            metrics_present = [m for m in metric_order if m in ldf["metric_key"].values]
            if metrics_present:
                titles = {
                    "duration_s":       "Task Duration (s)",
                    "board_dwell_pct":  "Board Dwell %",
                    "screen_dwell_pct": "Screen Dwell %",
                    "entropy":          "Gaze Entropy (bits)",
                }
                n_m = len(metrics_present)
                rows_n = (n_m + 1) // 2
                fig = make_subplots(
                    rows=rows_n, cols=2,
                    subplot_titles=[titles[m] for m in metrics_present],
                )
                shown_legend: set = set()
                for idx, mkey in enumerate(metrics_present):
                    r_pos = idx // 2 + 1
                    c_pos = idx % 2 + 1
                    mdf = ldf[ldf["metric_key"] == mkey]
                    for cond, col, fill in [
                        (cond_a, col_a, fill_a),
                        (cond_b, col_b, fill_b),
                    ]:
                        cdf = mdf[mdf["condition"] == cond].sort_values("task_idx")
                        if cdf.empty:
                            continue
                        show_leg = cond not in shown_legend
                        if show_leg:
                            shown_legend.add(cond)
                        x   = cdf["task_idx"].tolist()
                        y   = cdf["mean"].tolist()
                        std = cdf["std"].tolist()
                        # Shaded ±1 SD band
                        fig.add_trace(go.Scatter(
                            x=x + x[::-1],
                            y=[m + s for m, s in zip(y, std)] +
                              [m - s for m, s in zip(y[::-1], std[::-1])],
                            fill="toself", fillcolor=fill,
                            line=dict(color="rgba(0,0,0,0)"),
                            showlegend=False, hoverinfo="skip",
                        ), row=r_pos, col=c_pos)
                        # Mean line
                        fig.add_trace(go.Scatter(
                            x=x, y=y,
                            mode="lines+markers",
                            line=dict(width=2, color=col),
                            marker=dict(size=6),
                            name=cond, showlegend=show_leg,
                            customdata=cdf["n"].tolist(),
                            hovertemplate=(
                                f"Rep %{{x}}<br>Mean: %{{y:.3f}}"
                                f"<br>n=%{{customdata}}<extra>{cond}</extra>"
                            ),
                        ), row=r_pos, col=c_pos)
                fig.update_xaxes(title_text="Repetition", dtick=1)
                fig.update_layout(
                    title="Learning Curves — Metric Evolution over Task Repetitions",
                    template="aoi_studio",
                    height=300 * rows_n,
                    margin=dict(t=65, b=40),
                    legend=dict(orientation="h", y=1.04),
                )
                result.append(("06_learning_curves", fig))

        return result

    # ── Per-tab HTML helpers ──────────────────────────────────────────────────

    @staticmethod
    def _fig_to_html(fig) -> str:
        chart_html = fig.to_html(
            full_html=False, include_plotlyjs="cdn",
            config={"responsive": True}, default_height="100%",
        )
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>html,body{height:100%;background:#0e0f11;margin:0;padding:8px;"
            "box-sizing:border-box}.plotly-graph-div{height:100% !important;"
            "width:100% !important}</style></head>"
            f"<body>{chart_html}</body></html>"
        )

    def _stats_page_html(self, report) -> str:
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>body{background:#0e0f11;color:#e7e8ea;"
            "font-family:'Hanken Grotesk',sans-serif;margin:0;padding:16px}"
            "td,th{padding:5px 14px 5px 0}"
            "td{text-align:right} td:first-child,td:nth-child(2){text-align:left}"
            "th{text-align:right} th:first-child,th:nth-child(2){text-align:left}"
            "</style></head>"
            f"<body>{self._stats_table_html(report)}</body></html>"
        )

    def _stats_table_html(self, report) -> str:
        cond_a, cond_b = report.condition_a, report.condition_b

        def _fmt_p(p) -> str:
            try:
                pf = float(p)
            except (TypeError, ValueError):
                return "—"
            if math.isnan(pf):
                return "—"
            if pf < 0.001: return "<b style='color:#cf6b6b'>&lt;0.001</b>"
            if pf < 0.01:  return f"<b style='color:#d4a24a'>{pf:.3f}</b>"
            if pf < 0.05:  return f"<b style='color:#d4a24a'>{pf:.3f}</b>"
            return f"{pf:.3f}"

        def _fmt_eff(e) -> str:
            try:
                ef = float(e)
            except (TypeError, ValueError):
                return "—"
            return "—" if math.isnan(ef) else f"{ef:.3f}"

        rows_html = ""
        for df, section in [
            (report.aoi_stats,    "Per-AOI Metrics"),
            (report.overall_stats, "Overall Metrics"),
        ]:
            if df is None or df.empty:
                continue
            rows_html += (
                f"<tr><td colspan='6' style='color:#6a6d73;font-size:12px;"
                f"padding-top:16px;padding-bottom:4px;"
                f"border-bottom:1px solid rgba(255,255,255,.07)'>{section}</td></tr>"
            )
            for _, row in df.iterrows():
                aoi_cell = (
                    f"<td style='color:#9b9ea4'>{row['aoi']}</td>"
                    if "aoi" in row.index else "<td></td>"
                )
                rows_html += (
                    "<tr>"
                    + aoi_cell
                    + f"<td>{row['metric']}</td>"
                    + f"<td>{float(row[f'{cond_a}_mean']):.3f}"
                    + f" <span style='color:#62656b'>±{float(row[f'{cond_a}_std']):.3f}</span></td>"
                    + f"<td>{float(row[f'{cond_b}_mean']):.3f}"
                    + f" <span style='color:#62656b'>±{float(row[f'{cond_b}_std']):.3f}</span></td>"
                    + f"<td>{_fmt_p(row['p_value'])}</td>"
                    + f"<td>{_fmt_eff(row['effect_size'])}</td>"
                    + "</tr>"
                )

        return (
            "<h4 style='color:#9b9ea4;margin:28px 0 8px;font-weight:500'>Statistical Summary"
            " &nbsp;<span style='font-weight:400;font-size:12px;color:#6a6d73'>"
            "Mann-Whitney U · effect size = rank-biserial r · * p&lt;0.05</span></h4>"
            "<table style='border-collapse:collapse;font-size:13px;width:100%;font-family:\"IBM Plex Mono\",monospace'>"
            f"<tr style='border-bottom:1px solid rgba(255,255,255,.07);color:#62656b'>"
            "<th>AOI</th><th>Metric</th>"
            f"<th>{cond_a} (mean±SD)</th>"
            f"<th>{cond_b} (mean±SD)</th>"
            "<th>p-value</th><th>Effect size</th>"
            "</tr>"
            + rows_html
            + "</table>"
        )

    # ── Export PNGs ───────────────────────────────────────────────────────────

    def _export_png(self) -> None:
        if self._last_report is None:
            self._status.setText("Run comparison first.")
            return
        cond_a: Optional[pathlib.Path] = self._cond_a_combo.currentData()
        if cond_a is None:
            return
        out_dir = cond_a.parent / "aoi_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._status.setText("Rendering PNGs…")
        QApplication.processEvents()
        figs = self._build_figures(self._last_report)
        saved: list[pathlib.Path] = []
        for slug, fig in figs:
            path = out_dir / f"{slug}.png"
            try:
                h = fig.layout.height or 500
                fig.write_image(str(path), width=1400, height=h, scale=2)
                saved.append(path)
            except Exception as exc:
                self._status.setText(f"PNG render failed: {exc}")
                return
        self._status.setText(f"Saved {len(saved)} PNG(s) → {out_dir}")
        QMessageBox.information(
            self, APP_TITLE,
            f"Saved {len(saved)} chart(s) to:\n{out_dir}\n\n"
            + "\n".join(p.name for p in saved),
        )


# ─── Export / review helpers ──────────────────────────────────────────────────

def _gap_fill_review(
    labels: list[str],
    sources: list[str],
    max_gap: int = GAP_FILL_MAX_FRAMES,
) -> tuple[list[str], list[str]]:
    """Fill short auto-labelled gaps between matching manual corrections."""
    out_labels = list(labels)
    out_sources = list(sources)
    n = len(out_labels)
    manual_idx = [i for i in range(n) if out_sources[i] == "manual"]
    for mi in range(len(manual_idx) - 1):
        i = manual_idx[mi]
        j = manual_idx[mi + 1]
        if out_labels[i] != out_labels[j] or j - i <= 1:
            continue
        if j - i - 1 > max_gap:
            continue
        for k in range(i + 1, j):
            if out_sources[k] == "auto":
                out_labels[k] = out_labels[i]
                out_sources[k] = "auto_gap_fill"
    return out_labels, out_sources


def _export_reviewed_csv(
    raw_df: pd.DataFrame,
    edit_labels: list[str],
    edit_sources: list[str],
) -> pd.DataFrame:
    """Merge auto analysis with manual review labels for final export."""
    df = raw_df.copy()
    filled_labels, filled_sources = _gap_fill_review(edit_labels, edit_sources)
    n_video = len(filled_labels)

    if "frame_idx" in df.columns:
        fi = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(-1).astype(int)
    else:
        fi = pd.Series(np.arange(len(df)), index=df.index)

    final_aoi: list[str] = []
    edit_src: list[str] = []
    for row_i, idx in enumerate(fi):
        if 0 <= int(idx) < n_video:
            final_aoi.append(filled_labels[int(idx)])
            edit_src.append(filled_sources[int(idx)])
        else:
            primary = (
                str(df["primary_aoi"].iloc[row_i])
                if "primary_aoi" in df.columns
                else NONE_LABEL
            )
            final_aoi.append(primary)
            edit_src.append("auto")

    df["final_primary_aoi"] = final_aoi
    df["edit_source"] = edit_src
    return df


# ─── Main window ──────────────────────────────────────────────────────────────

class CollapsibleBox(QWidget):
    """A collapsible container widget."""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 6px;
                background-color: {Theme.BG_FIELD};
                border: none;
                border-radius: 4px;
                color: {Theme.TEXT_FAINT};
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 1.1px;
            }}
            QPushButton:hover {{
                background-color: {Theme.BG_CARD};
                color: {Theme.TEXT};
            }}
        """)
        # Arrow in the text shows state: ▼ Title (open) / ▶ Title (collapsed).
        # Sections start COLLAPSED — open only what you need.
        self._title = title
        self.toggle_button.setText(f"▶  {self._title}")
        self.toggle_button.clicked.connect(self.on_pressed)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 8, 0, 8)
        self.content_layout.setSpacing(10)
        self.content_area.setVisible(False)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        # Don't reserve vertical space when collapsed.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def on_pressed(self):
        # Toggle: if currently hidden, show it; if currently shown, hide it.
        is_collapsed = self.content_area.isHidden()
        self.content_area.setVisible(is_collapsed)
        indicator = "▼" if is_collapsed else "▶"
        self.toggle_button.setText(f"{indicator}  {self._title}")
        
    def addWidget(self, widget, stretch=0):
        self.content_layout.addWidget(widget, stretch)


# Manual assembly-error entry (filled by hand from the board photos, per task).
ERROR_FIELDS = [
    ("comp_number",      "Number / Gap"),
    ("comp_type",        "Type"),
    ("comp_orientation", "Orientation"),
    ("cable_position",   "Position"),
    ("cable_colour",     "Colour"),
]


class ErrorPanel(QWidget):
    """Per-task assembly error counts (Components + Cables), entered manually."""
    errorsChanged = Signal()
    N_TASKS = 10

    def __init__(self) -> None:
        super().__init__()
        self._data = {f"Task {i}": {k: 0 for k, _ in ERROR_FIELDS}
                      for i in range(1, self.N_TASKS + 1)}
        self._current = "Task 1"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Task:"))
        self._task_combo = QComboBox()
        self._task_combo.addItems([f"Task {i}" for i in range(1, self.N_TASKS + 1)])
        row.addWidget(self._task_combo, 1)
        lay.addLayout(row)

        self._spins: dict[str, QSpinBox] = {}

        def section(title: str, keys: list[tuple[str, str]]) -> None:
            t = QLabel(title); t.setObjectName("sectionTitle"); lay.addWidget(t)
            for key, label in keys:
                r = QHBoxLayout(); r.setSpacing(6)
                lbl = QLabel(label)
                lbl.setStyleSheet(f"font-size: 12px; color: {Theme.TEXT_MUTED};")
                sp = QSpinBox(); sp.setRange(0, 999); sp.setFixedWidth(64)
                sp.valueChanged.connect(self._on_value_changed)
                r.addWidget(lbl, 1); r.addWidget(sp)
                lay.addLayout(r)
                self._spins[key] = sp

        section("COMPONENTS", ERROR_FIELDS[:3])
        section("CABLES", ERROR_FIELDS[3:])

        self._task_combo.currentTextChanged.connect(self._on_task_changed)
        self._refresh_spins()

    def _on_task_changed(self, task: str) -> None:
        self._current = task
        self._refresh_spins()

    def _refresh_spins(self) -> None:
        vals = self._data.get(self._current, {})
        for k, sp in self._spins.items():
            sp.blockSignals(True)
            sp.setValue(int(vals.get(k, 0)))
            sp.blockSignals(False)

    def _on_value_changed(self, _v: int) -> None:
        d = self._data.setdefault(self._current, {k: 0 for k, _ in ERROR_FIELDS})
        for k, sp in self._spins.items():
            d[k] = sp.value()
        self.errorsChanged.emit()

    def get_errors(self) -> dict:
        return {t: dict(v) for t, v in self._data.items()}

    def load(self, data: dict) -> None:
        if isinstance(data, dict):
            for t, v in data.items():
                if t in self._data and isinstance(v, dict):
                    for k in self._data[t]:
                        try:
                            self._data[t][k] = int(v.get(k, 0) or 0)
                        except (TypeError, ValueError):
                            self._data[t][k] = 0
        self._refresh_spins()

    def reset(self) -> None:
        self._data = {f"Task {i}": {k: 0 for k, _ in ERROR_FIELDS}
                      for i in range(1, self.N_TASKS + 1)}
        self._refresh_spins()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        icon_path = resolve_app_icon()
        if icon_path and icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1440, 900)

        self._rec_dir: pathlib.Path | None = None
        self._df: pd.DataFrame | None = None
        self._video_path: pathlib.Path | None = None
        self._fps = 30.0
        self._n_frames = 0
        self._current_frame = 0
        self._edit_labels: list[str] = []
        self._edit_sources: list[str] = []
        self._undo_stack: list[tuple[list[str], list[str], int]] = []
        self._analysis_worker: Worker | None = None
        # Bumped on each analysis run so stale worker callbacks are ignored.
        self._analysis_generation = 0

        # Playback state + timer. Dropped during the refactor; _set_playing()
        # reads _play_timer/_playing/_speed, so loading a recording (which calls
        # _set_playing(False)) crashed without these.
        self._playing = False
        self._speed = 1.0
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        # Debounced auto-save of AOI corrections (writes straight into analysis.csv).
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(lambda: self._save_corrections_to_disk(silent=True))

        # Polls progress.json during analysis to fill the Analyse button.
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._poll_analysis)

        # State variables for new logic
        self._trim: dict[str, Any] = {}
        self._tasks: dict[str, Any] = {}
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Menu bar — batch/overnight analysis lives here, out of the workflow panel.
        analyse_menu = self.menuBar().addMenu("Analyse")
        self._batch_action = QAction("Analyse all in condition…", self)
        self._batch_action.setToolTip("Re-run analysis on every recording in the current condition")
        self._batch_action.triggered.connect(self._run_batch_analysis)
        analyse_menu.addAction(self._batch_action)
        # Validation video is slow to encode — off by default; toggle on to eyeball tracking.
        self._gen_video_action = QAction("Generate validation video (slower)", self)
        self._gen_video_action.setCheckable(True)
        self._gen_video_action.setChecked(False)
        self._gen_video_action.setToolTip("When on, Analyse also writes the overlay .mp4 (much slower)")
        analyse_menu.addAction(self._gen_video_action)
        analyse_menu.addSeparator()
        self._watcher: Optional[WatcherWorker] = None
        self._watcher_action = QAction("▶  Auto-analyse all pending (temporary)", self)
        self._watcher_action.setToolTip("Background sweep: analyse every un-analysed recording across BOTH conditions while you work")
        self._watcher_action.triggered.connect(self._toggle_watcher)
        analyse_menu.addAction(self._watcher_action)

        # Top-right corner (beside the tab bar): Save + Export + Help. Frees the
        # lower-left panel and keeps the primary output actions always reachable.
        help_btn = QPushButton("?")
        help_btn.setObjectName("iconButton")
        help_btn.setFixedSize(26, 26)
        help_btn.setToolTip("Keyboard shortcuts")
        help_btn.clicked.connect(self._show_shortcuts_help)

        self._save_tasks_btn = QPushButton("  Save")
        self._save_tasks_btn.setIcon(_svg_icon("export.svg"))
        self._save_tasks_btn.setIconSize(QSize(14, 14))
        self._save_tasks_btn.setToolTip("Save task start/end annotations to disk")
        self._save_tasks_btn.setEnabled(False)

        self._export_btn = QPushButton("  Export")
        self._export_btn.setObjectName("primaryButton")
        self._export_btn.setIcon(_svg_icon("export.svg"))
        self._export_btn.setIconSize(QSize(14, 14))
        self._export_btn.setToolTip("Export the corrected AOI labels and tasks to a final CSV")

        corner = QWidget()
        corner_l = QHBoxLayout(corner)
        corner_l.setContentsMargins(0, 0, 8, 0)
        corner_l.setSpacing(6)
        corner_l.addWidget(self._save_tasks_btn)
        corner_l.addWidget(self._export_btn)
        corner_l.addWidget(help_btn)
        self._tabs.setCornerWidget(corner, Qt.TopRightCorner)

        # ── Studio tab ──────────────────────────────────────────────────────
        studio = QWidget()
        studio_layout = QHBoxLayout(studio)
        studio_layout.setContentsMargins(10, 10, 10, 10)
        studio_layout.setSpacing(12)
        self._tabs.addTab(studio, "  Studio  ")

        # Left panel — minimum width only; the user drags the splitter to widen
        # it (e.g. when 5-digit frame numbers crowd the task rows).
        left = QFrame()
        left.setObjectName("leftPanel")
        left.setMinimumWidth(308)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 18, 16, 18)
        left_layout.setSpacing(12)

        # ── Source folder ──
        src_title = QLabel("SOURCE")
        src_title.setObjectName("sectionTitle")
        left_layout.addWidget(src_title)

        src_row = QHBoxLayout()
        src_row.setSpacing(4)
        self._src_combo = QComboBox()
        self._src_combo.setToolTip("Condition or difficulty folder (e.g. NonGamified)")
        browse_src = QPushButton("…")
        browse_src.setFixedWidth(26)
        browse_src.setObjectName("iconButton")
        browse_src.setToolTip("Browse for source folder")
        src_row.addWidget(self._src_combo, 1)
        src_row.addWidget(browse_src)
        left_layout.addLayout(src_row)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(4)
        self._rec_combo = QComboBox()
        self._rec_combo.setToolTip("Recording folder")
        browse_rec = QPushButton("…")
        browse_rec.setFixedWidth(26)
        browse_rec.setObjectName("iconButton")
        browse_rec.setToolTip("Browse for recording folder directly")
        refresh_btn = QPushButton()
        refresh_btn.setIcon(_svg_icon("refresh.svg"))
        refresh_btn.setIconSize(QSize(14, 14))
        refresh_btn.setFixedWidth(26)
        refresh_btn.setObjectName("iconButton")
        refresh_btn.setToolTip("Refresh list")
        rec_row.addWidget(self._rec_combo, 1)
        rec_row.addWidget(browse_rec)
        rec_row.addWidget(refresh_btn)
        left_layout.addLayout(rec_row)

        # Filter the recording list by analysis status (🟢 analysed / ⚪ pending).
        self._rec_filter_combo = QComboBox()
        self._rec_filter_combo.addItem("All recordings", "all")
        self._rec_filter_combo.addItem("● Analysed", "done")
        self._rec_filter_combo.addItem("○ Pending", "pending")
        self._rec_filter_combo.setToolTip("Show all recordings, only analysed, or only those still needing analysis")
        left_layout.addWidget(self._rec_filter_combo)

        self._load_btn = QPushButton("Load")
        self._load_btn.setObjectName("primaryButton")
        self._load_btn.setToolTip("Load the selected recording into the player")
        left_layout.addWidget(self._load_btn)

        self._analyze_btn = QPushButton("  Analyse")
        self._analyze_btn.setIcon(_svg_icon("playbutton.svg"))
        self._analyze_btn.setIconSize(QSize(14, 14))
        self._analyze_btn.setObjectName("primaryButton")
        self._analyze_btn.setEnabled(False)
        left_layout.addWidget(self._analyze_btn)


        # Progress is shown ON the Analyse button (fills as it processes), not a
        # separate bar. Kept for compatibility but not placed in the panel.
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.hide()

        # Status lives in the window's bottom status bar, not as a panel info box.
        self._status_lbl = QLabel("No recording loaded.")
        self.statusBar().addWidget(self._status_lbl, 1)
        self._quality_lbl = QLabel("")   # data-quality note (right of the status bar)
        self.statusBar().addPermanentWidget(self._quality_lbl)
        self._quality_lbl.hide()

        _div1 = QFrame(); _div1.setFrameShape(QFrame.HLine); _div1.setObjectName("divider")
        left_layout.addWidget(_div1)

        # ── Trim ──
        self._trim_box = CollapsibleBox("TRIM")
        self._trim_panel = TrimPanel()
        self._trim_box.addWidget(self._trim_panel)
        left_layout.addWidget(self._trim_box)

        # ── AOI Correction ──
        self._corr_box = CollapsibleBox("AOI CORRECTION")
        self._correction_panel = CorrectionPanel()
        self._corr_box.addWidget(self._correction_panel)
        left_layout.addWidget(self._corr_box)

        # ── Task annotation ──
        self._task_box = CollapsibleBox("TASKS")
        self._task_panel = TaskPanel()
        self._task_box.addWidget(self._task_panel.header_widget)
        
        task_scroll = QScrollArea()
        task_scroll.setWidgetResizable(True)
        task_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        task_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        task_scroll.setWidget(self._task_panel)
        task_scroll.setMinimumHeight(200)   # visible when the section is expanded
        self._task_box.addWidget(task_scroll, 1)
        left_layout.addWidget(self._task_box)

        # ── Assembly errors (manual entry from board photos) ──
        self._error_box = CollapsibleBox("ERRORS")
        self._error_panel = ErrorPanel()
        self._error_box.addWidget(self._error_panel)
        left_layout.addWidget(self._error_box)

        # Collapsible sections stack from the top; slack goes to the bottom so a
        # collapsed section doesn't leave a big gap.
        left_layout.addStretch(1)
        # Save + Export now live in the top-right corner (created above).

        # Right: video + controls, and timeline — split so the timeline's
        # height is freely user-resizable by dragging the splitter handle.
        video_pane = QWidget()
        video_pane_layout = QVBoxLayout(video_pane)
        video_pane_layout.setContentsMargins(4, 4, 4, 4)
        video_pane_layout.setSpacing(10)

        self._video = VideoWidget()
        video_pane_layout.addWidget(self._video, 1)

        self._playback = PlaybackBar()
        video_pane_layout.addWidget(self._playback)

        self._timeline = AOITimeline()

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(6)
        right_splitter.addWidget(video_pane)
        right_splitter.addWidget(self._timeline)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 0)
        right_splitter.setSizes([600, 110])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(6)
        main_splitter.addWidget(left)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([260, 1000])
        studio_layout.addWidget(main_splitter, 1)

        # ── Dashboard tab ────────────────────────────────────────────────────
        self._dashboard = DashboardWidget()
        self._tabs.addTab(self._dashboard, "  Dashboard  ")

        # ── Comparison tab ───────────────────────────────────────────────────
        self._comparison = ComparisonWidget()
        self._tabs.addTab(self._comparison, "  Comparison  ")

        # Store button refs for wiring
        self._browse_src_btn = browse_src
        self._browse_rec_btn = browse_rec
        self._refresh_btn    = refresh_btn

        # Apply the custom theme, connect every signal, and populate the source
        # dropdowns. These were split into helpers during the refactor but the
        # calls were dropped — without them the stylesheet never loads (boxy
        # default look) and no button is connected.
        self._apply_style()
        self._wire()
        self._refresh_sources()

    # ── Wire signals ─────────────────────────────────────────────────────────

    def _wire(self) -> None:
        self._src_combo.currentIndexChanged.connect(self._refresh_recordings)
        self._rec_filter_combo.currentIndexChanged.connect(self._refresh_recordings)
        self._browse_src_btn.clicked.connect(self._browse_source)
        self._browse_rec_btn.clicked.connect(self._browse_recording)
        self._refresh_btn.clicked.connect(self._refresh_recordings)
        self._load_btn.clicked.connect(self._load_selected)
        self._analyze_btn.clicked.connect(self._run_analysis)
        self._save_tasks_btn.clicked.connect(self._manual_save_tasks)
        self._export_btn.clicked.connect(self._export_final)

        self._playback.prev_btn.clicked.connect(lambda: self._video.step(-30))
        self._playback.play_btn.clicked.connect(self._toggle_play)
        self._playback.next_btn.clicked.connect(lambda: self._video.step(30))
        self._playback.seeked.connect(self._on_seek)

        self._video.frameChanged.connect(self._on_frame_changed)
        self._timeline.frameClicked.connect(self._on_seek)

        self._task_panel.tasksChanged.connect(self._on_tasks_changed)
        self._trim_panel.trimChanged.connect(self._on_trim_changed)
        self._trim_panel._crop_btn.clicked.connect(self._on_crop_data_requested)
        self._correction_panel.correctionApplied.connect(self._on_correction_applied)
        self._correction_panel.correctionSaved.connect(self._on_correction_saved)
        self._correction_panel.undoRequested.connect(self._undo_correction)
        self._error_panel.errorsChanged.connect(self._save_errors)

        # Shortcuts
        for key, fn in [
            (Qt.Key_Space, self._toggle_play),
            (Qt.Key_Left,  lambda: self._video.step(-1)),
            (Qt.Key_Right, lambda: self._video.step(1)),
            (Qt.Key_A,     lambda: self._video.step(-1)),
            (Qt.Key_D,     lambda: self._video.step(1)),
            (Qt.Key_I,     self._task_panel._on_start),
            (Qt.Key_O,     self._task_panel._on_end),
            # AOI correction — fast annotation
            (Qt.Key_F,     self._correction_panel._apply_frame),   # apply AOI to current frame
            (Qt.Key_J,     self._correction_panel._set_in),        # range In
            (Qt.Key_K,     self._correction_panel._set_out),       # range Out
            (Qt.Key_L,     self._correction_panel._apply_range),   # Apply range
        ]:
            act = QAction(self)
            act.setShortcut(key)
            act.setShortcutContext(Qt.ApplicationShortcut)
            act.triggered.connect(fn)
            self.addAction(act)

    # ── Source / recording loading ────────────────────────────────────────────

    def _refresh_sources(self) -> None:
        self._src_combo.blockSignals(True)
        self._src_combo.clear()
        for d in list_source_folders():
            self._src_combo.addItem(d.name, userData=d)
        self._src_combo.blockSignals(False)
        self._refresh_recordings()

    def _get_current_source_dir(self) -> Optional[pathlib.Path]:
        """Get the currently selected source directory."""
        return self._src_combo.currentData()

    @staticmethod
    def _is_analysed(d: pathlib.Path) -> bool:
        return ((d / "aoi_results" / "raw" / "analysis.csv").exists() or
                (d / "aoi_results" / "analysis.csv").exists())

    def _refresh_recordings(self) -> None:
        prev = self._rec_combo.currentData()
        self._rec_combo.blockSignals(True)
        self._rec_combo.clear()
        src: Optional[pathlib.Path] = self._src_combo.currentData()
        if src is not None and src.exists():
            flt = self._rec_filter_combo.currentData()
            for d in sorted(src.iterdir()):
                if not (d.is_dir() and (d / "info.json").exists()):
                    continue
                done = self._is_analysed(d)
                if flt == "done" and not done:
                    continue
                if flt == "pending" and done:
                    continue
                self._rec_combo.addItem(f"{'●' if done else '○'}  {d.name}", userData=d)
                self._rec_combo.setItemData(
                    self._rec_combo.count() - 1,
                    QColor("#6fae7d") if done else QColor("#6a6d73"),
                    Qt.ForegroundRole)
        if prev is not None:                       # keep the current selection
            for i in range(self._rec_combo.count()):
                if self._rec_combo.itemData(i) == prev:
                    self._rec_combo.setCurrentIndex(i)
                    break
        self._rec_combo.blockSignals(False)

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select source folder", str(RECORDINGS_DIR))
        if path:
            d = pathlib.Path(path)
            self._src_combo.addItem(d.name, userData=d)
            self._src_combo.setCurrentIndex(self._src_combo.count() - 1)

    def _browse_recording(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select recording folder", str(RECORDINGS_DIR))
        if path:
            d = pathlib.Path(path)
            if (d / "info.json").exists():
                self._rec_combo.addItem(d.name, userData=d)
                self._rec_combo.setCurrentIndex(self._rec_combo.count() - 1)
                self._load_recording(d)
            else:
                QMessageBox.warning(self, APP_TITLE, "Not a Neon recording (no info.json).")

    def _load_selected(self) -> None:
        rec_dir: Optional[pathlib.Path] = self._rec_combo.currentData()
        if rec_dir:
            self._load_recording(rec_dir)

    def _flush_autosave(self) -> None:
        """Write any pending corrections for the CURRENT recording immediately."""
        if getattr(self, "_autosave_timer", None) is not None and self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._save_corrections_to_disk(silent=True)

    def _load_recording(self, rec_dir: pathlib.Path) -> None:
        self._flush_autosave()   # persist edits on the OUTGOING recording first
        self._status_lbl.setText("Loading…")
        QApplication.processEvents()
        try:
            import av
            av.logging.set_level(av.logging.ERROR)
        except Exception:
            pass
        try:
            recording = nr.load(str(rec_dir))
            self._recording = recording
            self._rec_dir   = rec_dir
            n = len(recording.scene.time)
            ts = recording.scene.time
            raw_fps = max(1.0, (n - 1) / ((ts[-1] - ts[0]) / 1e9)) if n > 1 else 30.0
            self._fps = min((s for s in (24.0, 25.0, 30.0, 50.0, 60.0) if abs(raw_fps - s) < 4),
                            key=lambda s: abs(raw_fps - s), default=raw_fps)

            self._video.load(recording, rec_dir)
            self._playback.configure(n, self._fps)
            self._task_panel.set_fps(self._fps)
            self._set_playing(False)

            # Reset analysis and task state for the new recording
            self._csv_labels = []
            self._edit_labels = []
            self._edit_sources = []
            self._correction_panel.reset()
            self._timeline.set_recording_length(n)
            self._timeline.set_analysis([], np.array([]), np.array([]), [])
            self._task_panel.reset()
            self._tasks = {f"Task {i}": {"start": None, "end": None} for i in range(1, 11)}
            self._timeline.set_tasks(self._tasks)
            self._trim_panel.reset()
            self._trim = self._trim_panel.get_trim()

            # Load existing analysis / tasks / trim / quality / errors if available
            self._load_analysis_if_ready()
            self._load_tasks_from_disk()
            self._load_trim_from_disk()
            self._load_quality()
            self._load_errors_from_disk()

            self._analyze_btn.setEnabled(True)
            self._save_tasks_btn.setEnabled(True)
            dur = n / self._fps
            self._status_lbl.setText(
                f"{rec_dir.name}  ·  {n} frames  ·  "
                f"{int(dur//3600)}:{int(dur%3600//60):02d}:{int(dur%60):02d}  ·  {self._fps:.0f} fps"
            )
        except Exception as exc:
            self._status_lbl.setText(f"Load error: {exc}")
            QMessageBox.warning(self, APP_TITLE, f"Could not load recording:\n{exc}")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _init_edit_state(self, n_frames: int, df: Optional[pd.DataFrame] = None) -> None:
        self._edit_labels = list(self._csv_labels) if self._csv_labels else [NONE_LABEL] * n_frames
        if len(self._edit_labels) < n_frames:
            self._edit_labels.extend([NONE_LABEL] * (n_frames - len(self._edit_labels)))
        self._edit_sources = ["auto"] * n_frames
        if df is None:
            return
        label_col = "final_primary_aoi" if "final_primary_aoi" in df.columns else None
        if label_col is None:
            return
        if "frame_idx" in df.columns:
            fi = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(-1).astype(int).to_numpy()
            vals = df[label_col].fillna(NONE_LABEL).astype(str).to_numpy()
            srcs = (
                df["edit_source"].astype(str).to_numpy()
                if "edit_source" in df.columns
                else np.array(["auto"] * len(df))
            )
            for idx, lbl, src in zip(fi, vals, srcs):
                if 0 <= int(idx) < n_frames:
                    self._edit_labels[int(idx)] = lbl if lbl else NONE_LABEL
                    self._edit_sources[int(idx)] = src

    def _on_correction_applied(self, range_mode: bool) -> None:
        if not self._edit_labels:
            return
        aoi = self._correction_panel.selected_aoi()
        if range_mode:
            start, end = self._correction_panel.active_range()
            if start is None or end is None:
                return
            lo, hi = (start, end) if start <= end else (end, start)
            frames = range(lo, hi + 1)
        else:
            frames = [self._video.frame_idx]
        # Snapshot the affected frames so this correction can be undone one step.
        snapshot = [(f, self._edit_labels[f], self._edit_sources[f])
                    for f in frames if 0 <= f < len(self._edit_labels)]
        if snapshot:
            self._undo_stack.append(snapshot)
        for f in frames:
            if 0 <= f < len(self._edit_labels):
                self._edit_labels[f] = aoi
                self._edit_sources[f] = "manual"
        self._refresh_correction_timeline()
        self._schedule_autosave()
        self._status_lbl.setText(f"Corrected {len(frames)} frame(s) → {aoi.replace('_', ' ')}")

    def _refresh_correction_timeline(self) -> None:
        self._timeline.set_analysis(
            list(self._edit_labels),
            self._timeline._gaze_x,
            self._timeline._gaze_y,
            self._timeline._fix_frames,
        )

    def _undo_correction(self) -> None:
        if not self._undo_stack:
            self._status_lbl.setText("Nothing to undo.")
            return
        snapshot = self._undo_stack.pop()
        for f, lbl, src in snapshot:
            if 0 <= f < len(self._edit_labels):
                self._edit_labels[f] = lbl
                self._edit_sources[f] = src
        self._refresh_correction_timeline()
        self._schedule_autosave()
        self._status_lbl.setText(f"Undid correction on {len(snapshot)} frame(s).")

    # ── Correction persistence (overwrite primary_aoi in place + auto-save) ─────
    def _schedule_autosave(self) -> None:
        """Debounced auto-save so corrections are always on disk (no data loss)."""
        self._autosave_timer.start(500)

    def _save_corrections_to_disk(self, silent: bool = True) -> None:
        if not self._rec_dir or not self._edit_labels:
            return
        csv_path = self._rec_dir / "aoi_results" / "raw" / "analysis.csv"
        if not csv_path.exists():
            csv_path = self._rec_dir / "aoi_results" / "analysis.csv"
        if not csv_path.exists():
            if not silent:
                QMessageBox.warning(self, APP_TITLE, "analysis.csv not found.")
            return
        try:
            df = pd.read_csv(csv_path)
            # Rows are positioned by absolute frame_idx, so map edited frames to the
            # correct rows (robust to trimmed/cropped CSVs).
            if "frame_idx" in df.columns:
                fi = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(-1).astype(int)
                pos = {f: r for r, f in enumerate(fi.tolist())}
            else:
                pos = {i: i for i in range(len(df))}
            # Ensure string labels can be written even if the column was all-NaN (float).
            for c in ("primary_aoi", "aoi_hit_source", "gaze_on_aoi_x", "gaze_on_aoi_y"):
                if c in df.columns:
                    df[c] = df[c].astype("object")
            changes = 0
            corrected: list[tuple[int, str]] = []
            for f, (lbl, src) in enumerate(zip(self._edit_labels, self._edit_sources)):
                if src == "manual" and f in pos:
                    df.at[pos[f], "primary_aoi"] = lbl
                    if "aoi_hit_source" in df.columns:
                        df.at[pos[f], "aoi_hit_source"] = "manual"
                    corrected.append((f, lbl))
                    changes += 1
            # Single source of truth is primary_aoi — drop any legacy correction column.
            if "final_primary_aoi" in df.columns:
                df = df.drop(columns=["final_primary_aoi"])
            if "aoi_transition" in df.columns:
                df["aoi_transition"] = df["primary_aoi"].ne(df["primary_aoi"].shift()) & df["primary_aoi"].notna()
            # Re-project the corrected frames' gaze onto their NEW surface so the
            # heatmap reflects the edit (needs the persisted scene model).
            self._reproject_corrected(df, pos, corrected)
            df.to_csv(csv_path, index=False)
            # Rebuild fixation_summary/data_quality from the corrected CSV so the
            # Fixation and Data-Quality tabs update too.
            try:
                import analyzer
                analyzer.write_summaries(csv_path.parent, csv_path, self._rec_dir.name, self._fps)
            except Exception:
                pass
            self._dashboard.reload()
            if not silent:
                self._status_lbl.setText(f"Saved {changes} correction(s) to analysis.csv")
                self._correction_panel._save_btn.setText("Saved ✓")
                QTimer.singleShot(1200, lambda: self._correction_panel._save_btn.setText("Save"))
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, APP_TITLE, f"Failed to save corrections: {e}")

    def _reproject_corrected(self, df: pd.DataFrame, pos: dict, corrected: list) -> None:
        """Recompute gaze_on_aoi_x/y for corrected frames by re-detecting the frame
        and projecting the gaze onto the NEW surface (so the heatmap updates). Uses
        the scene model saved during analysis; silently no-ops for older analyses."""
        if not corrected or self._recording is None:
            return
        model_path = None
        for p in [self._rec_dir / "aoi_results" / "raw" / "scene_model.pkl",
                  self._rec_dir / "aoi_results" / "scene_model.pkl"]:
            if p.exists():
                model_path = p
                break
        if model_path is None or "gaze_on_aoi_x" not in df.columns:
            return
        try:
            import rigid_surface, analyzer
        except Exception:
            return

        class _D:
            __slots__ = ("tag_id", "corners")
            def __init__(self, t, c):
                self.tag_id = t; self.corners = c

        try:
            model = rigid_surface.load_scene_model(model_path)
            det = analyzer.make_detector(1.0)
            scene_ts = self._recording.scene.time
            for f, lbl in corrected:
                if f not in pos:
                    continue
                r = pos[f]
                if lbl == NONE_LABEL:
                    df.at[r, "gaze_on_aoi_x"] = ""
                    df.at[r, "gaze_on_aoi_y"] = ""
                    continue
                if f >= len(scene_ts):
                    continue
                gx = pd.to_numeric(pd.Series([df.at[r, "gaze_x_px"]]), errors="coerce").iloc[0]
                gy = pd.to_numeric(pd.Series([df.at[r, "gaze_y_px"]]), errors="coerce").iloc[0]
                if not (np.isfinite(gx) and np.isfinite(gy)):
                    continue
                frame = next(iter(self._recording.scene.sample(np.array([int(scene_ts[f])]))))
                dets = [_D(int(d.tag_id), d.corners.astype(np.float64))
                        for d in det.detect(analyzer.enhance_frame(frame.gray))]
                loc = model.localize(dets)
                cr, ct = (loc[0], loc[1]) if loc else (None, None)
                quad = model.surface_quad(lbl, dets, cr, ct)
                if quad is None:
                    continue
                uv = model.gaze_to_surface(lbl, float(gx), float(gy), cr, ct, image_quad=quad)
                if uv is not None:
                    df.at[r, "gaze_on_aoi_x"] = f"{min(max(uv[0], 0.0), 1.0):.5f}"
                    df.at[r, "gaze_on_aoi_y"] = f"{min(max(uv[1], 0.0), 1.0):.5f}"
        except Exception:
            pass

    def _on_correction_saved(self) -> None:
        # Save button = force an immediate (non-silent) save.
        self._save_corrections_to_disk(silent=False)

    def _run_analysis(self) -> None:
        if self._rec_dir is None:
            return
        if self._analysis_worker is not None and self._analysis_worker.is_alive():
            reply = QMessageBox.question(self, APP_TITLE,
                                         "Analysis is running. Force restart?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            lock = self._rec_dir / "aoi_results" / "raw" / ".processing"
            lock.unlink(missing_ok=True)

        self._set_analyse_progress(0)
        self._progress_timer.start(300)
        self._status_lbl.setText("Analysis running…")

        self._analysis_generation += 1
        gen = self._analysis_generation
        worker = AnalysisWorker(
            self._rec_dir,
            trim=self._trim_panel.get_trim(),
            generate_video=self._gen_video_action.isChecked(),  # off by default (fast)
            generation=gen,
        )
        worker.signals.status.connect(self._status_lbl.setText)
        worker.signals.finished.connect(lambda: self._on_analysis_done(gen))
        worker.signals.failed.connect(lambda msg: self._on_analysis_failed(msg, gen))
        self._analysis_worker = worker
        worker.start()

    def _run_batch_analysis(self) -> None:
        source_dir = self._get_current_source_dir()
        if source_dir is None:
            QMessageBox.warning(self, APP_TITLE, "No source folder selected.")
            return

        recordings = [d for d in source_dir.iterdir() if d.is_dir()]
        if not recordings:
            QMessageBox.warning(self, APP_TITLE, f"No recordings found in {source_dir.name}")
            return

        reply = QMessageBox.question(
            self, APP_TITLE,
            f"Re-analyse all {len(recordings)} recordings in '{source_dir.name}'?\n\n"
            f"This will regenerate fixation_summary.csv and data_quality.json for all recordings.\n"
            f"Existing analysis.csv files will be overwritten.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self._batch_action.setEnabled(False)
        self._status_lbl.setText("Batch analysis starting…")

        self._analysis_generation += 1
        gen = self._analysis_generation
        worker = BatchAnalysisWorker(source_dir, generation=gen)
        worker.signals.status.connect(self._status_lbl.setText)
        worker.signals.finished.connect(lambda: self._on_batch_analysis_done(gen))
        worker.signals.failed.connect(lambda msg: self._on_batch_analysis_failed(msg, gen))
        worker.start()

    def _toggle_watcher(self) -> None:
        """TEMPORARY: start/stop a background sweep that analyses every pending
        recording in both conditions while you keep working."""
        if self._watcher is not None and self._watcher.is_alive():
            self._watcher.stop()
            self._watcher_action.setText("▶  Auto-analyse all pending (temporary)")
            self._status_lbl.setText("Stopping watcher…")
            return
        self._watcher = WatcherWorker()
        self._watcher.signals.status.connect(self._status_lbl.setText)
        self._watcher.signals.finished.connect(self._on_watcher_done)
        self._watcher.signals.failed.connect(
            lambda m: self._status_lbl.setText(f"Watcher error: {m}"))
        self._watcher_action.setText("■  Stop auto-analysis")
        self._status_lbl.setText("Watcher started — analysing pending recordings in the background…")
        self._watcher.start()

    def _on_watcher_done(self) -> None:
        self._watcher_action.setText("▶  Auto-analyse all pending (temporary)")
        self._refresh_recordings()
        self._status_lbl.setText("Watcher finished — pending recordings analysed.")

    def _on_batch_analysis_done(self, generation: int) -> None:
        if generation != self._analysis_generation:
            return
        self._batch_action.setEnabled(True)
        self._refresh_recordings()   # update 🟢/⚪ status dots
        self._status_lbl.setText("Batch analysis complete. All recordings updated.")
        QMessageBox.information(self, APP_TITLE, "Batch analysis complete!\n\nAll recordings have been re-analysed with new output files.")

    def _on_batch_analysis_failed(self, msg: str, generation: int) -> None:
        if generation != self._analysis_generation:
            return
        self._batch_action.setEnabled(True)
        self._status_lbl.setText("Batch analysis failed.")
        QMessageBox.warning(self, APP_TITLE, f"Batch analysis failed:\n{msg}")

    def _set_analyse_progress(self, pct: Optional[int]) -> None:
        """Show analysis progress by filling the Analyse button; None = idle reset."""
        if pct is None:
            self._analyze_btn.setStyleSheet("")   # revert to global primaryButton style
            self._analyze_btn.setText("  Analyse")
            return
        pct = max(0, min(100, int(pct)))
        self._analyze_btn.setText(f"  Analysing… {pct}%")
        s = pct / 100.0
        e = min(s + 0.0001, 1.0)
        self._analyze_btn.setStyleSheet(
            "QPushButton {"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 #7e9bdb, stop:{s:.4f} #7e9bdb,"
            f" stop:{e:.4f} #333f5c, stop:1 #333f5c);"
            " color:#ffffff; border:1px solid #6e8fd6; border-radius:8px;"
            " padding:8px 12px; font-weight:600; }"
        )

    def _poll_analysis(self) -> None:
        if self._rec_dir is None:
            return
        for progress_path in (self._rec_dir / "aoi_results" / "raw" / "progress.json",
                              self._rec_dir / "aoi_results" / "progress.json"):
            if progress_path.exists():
                try:
                    data = json.loads(progress_path.read_text(encoding="utf-8"))
                    self._set_analyse_progress(int(data.get("percent", 0)))
                except Exception:
                    pass
                return

    def _on_analysis_done(self, generation: int) -> None:
        if generation != self._analysis_generation:
            return
        self._progress_timer.stop()
        self._set_analyse_progress(None)
        self._analyze_btn.setEnabled(True)
        self._flush_autosave()   # never let a reload clobber unsaved corrections
        self._load_analysis_if_ready()
        self._load_quality()
        self._refresh_recordings()   # update ●/○ status dots
        self._dashboard.reload()     # keep dashboard in sync with the new analysis
        self._status_lbl.setText("Analysis complete.")

    def _on_analysis_failed(self, msg: str, generation: int) -> None:
        if generation != self._analysis_generation:
            return
        self._progress_timer.stop()
        self._set_analyse_progress(None)
        self._analyze_btn.setEnabled(True)
        self._status_lbl.setText(f"Analysis failed.")
        QMessageBox.warning(self, APP_TITLE, f"Analysis failed:\n{msg}")

    def _load_analysis_if_ready(self) -> None:
        if self._rec_dir is None:
            return
        n_frames = self._video.n_frames
        for csv_path in [
            self._rec_dir / "aoi_results" / "analysis.csv",
            self._rec_dir / "aoi_results" / "raw" / "analysis.csv",
        ]:
            if csv_path.exists() and csv_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(csv_path)
                    col = "final_primary_aoi" if "final_primary_aoi" in df.columns else "primary_aoi"

                    # CSV rows are positioned by absolute frame_idx (analyzer.py writes
                    # the position in the *original* video, not the row's position in
                    # a trimmed CSV) — scatter onto a full-length array by that column
                    # rather than assuming row order starts at video frame 0. Without
                    # this, a trimmed analysis would visually appear to sit at the start
                    # of the timeline instead of at its actual trimmed position.
                    if "frame_idx" in df.columns:
                        abs_idx = pd.to_numeric(df["frame_idx"], errors="coerce").fillna(-1).astype(int).to_numpy()
                    else:
                        abs_idx = np.arange(len(df))

                    labels = np.full(n_frames, NONE_LABEL, dtype=object)
                    aoi_vals = df[col].fillna(NONE_LABEL).astype(str).to_numpy()
                    in_range = (abs_idx >= 0) & (abs_idx < n_frames)
                    labels[abs_idx[in_range]] = aoi_vals[in_range]
                    self._csv_labels = labels.tolist()

                    # Extract gaze trace for timeline
                    gaze_x = np.full(n_frames, np.nan)
                    gaze_y = np.full(n_frames, np.nan)
                    if "gaze_x_px" in df.columns and "gaze_y_px" in df.columns:
                        gx = pd.to_numeric(df["gaze_x_px"], errors="coerce").to_numpy()
                        gy = pd.to_numeric(df["gaze_y_px"], errors="coerce").to_numpy()
                        gaze_x[abs_idx[in_range]] = gx[in_range]
                        gaze_y[abs_idx[in_range]] = gy[in_range]

                    # Fixation frame indices
                    fix_frames: list[int] = []
                    if "is_fixation" in df.columns:
                        fix_mask = pd.to_numeric(df["is_fixation"], errors="coerce").fillna(0) > 0
                        # Sample every 5th fixation frame to avoid dense overdraw
                        fix_idx = abs_idx[fix_mask.to_numpy() & in_range]
                        fix_frames = sorted(int(i) for i in fix_idx)[::5]

                    self._timeline.set_recording_length(n_frames)
                    self._init_edit_state(n_frames, df)
                    display_labels = self._edit_labels if self._edit_labels else self._csv_labels
                    self._timeline.set_analysis(display_labels, gaze_x, gaze_y, fix_frames)
                    return
                except Exception:
                    pass

    def _load_quality(self) -> None:
        """Read data_quality.json and update the quality indicator label."""
        if self._rec_dir is None:
            self._quality_lbl.hide()
            return
        for path in [
            self._rec_dir / "aoi_results" / "raw" / "data_quality.json",
            self._rec_dir / "aoi_results" / "data_quality.json",
        ]:
            if path.exists():
                try:
                    dq       = json.loads(path.read_text(encoding="utf-8"))
                    valid    = 100.0 - float(dq.get("missing_gaze_pct", 0.0))
                    fix_n    = int(dq.get("fixation_count", 0))
                    dur_s    = float(dq.get("recording_duration_s", 0.0))
                    mm       = int(dur_s) // 60
                    ss       = int(dur_s) % 60

                    # Color coding and warning
                    if valid >= 80:
                        color = "#6fae7d"
                        quality_status = "Good"
                    elif valid >= 60:
                        color = "#d4a24a"
                        quality_status = "Acceptable"
                        QMessageBox.warning(
                            self, APP_TITLE,
                            f"⚠ Data Quality Warning\n\n"
                            f"Gaze validity: {valid:.1f}% (60-80% range)\n\n"
                            f"This recording has acceptable but not ideal data quality.\n"
                            f"Consider noting this in your research write-up.\n\n"
                            f"Possible causes:\n"
                            f"• Poor eye tracker calibration\n"
                            f"• Frequent looking away from scene\n"
                            f"• Lighting conditions\n"
                            f"• Excessive head movement"
                        )
                    else:
                        color = "#cf6b6b"
                        quality_status = "Poor"
                        QMessageBox.critical(
                            self, APP_TITLE,
                            f"❌ Low Data Quality Alert\n\n"
                            f"Gaze validity: {valid:.1f}% (below 60%)\n\n"
                            f"This recording has poor data quality and should be flagged.\n"
                            f"Results may not be reliable for analysis.\n\n"
                            f"Recommended actions:\n"
                            f"• Check if the recording can be excluded\n"
                            f"• Review participant instructions\n"
                            f"• Verify eye tracker calibration procedure\n"
                            f"• Consider re-recording if possible"
                        )

                    self._quality_lbl.setText(
                        f"Gaze valid: {valid:.1f}%   ·   {fix_n} fixations   ·   {mm}:{ss:02d}   ·   {quality_status}"
                    )
                    self._quality_lbl.setStyleSheet(
                        f"color: {color}; font-size: 12px; font-family: 'IBM Plex Mono', monospace; "
                        f"background: transparent; border: 1px solid rgba(255,255,255,.08); "
                        f"border-radius: {Theme.RADIUS}px; padding: 6px 8px;"
                    )
                    self._quality_lbl.show()
                    return
                except Exception:
                    pass
        self._quality_lbl.hide()

    # ── Playback ──────────────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._video.set_playing(playing)
        self._playback.set_playing(playing)
        if playing:
            interval = max(16, int(1000 / (self._fps * self._speed)))
            self._play_timer.start(interval)
        else:
            self._play_timer.stop()
            self._video.render()   # re-render with surfaces now that we're paused

    def _play_tick(self) -> None:
        if not self._video.play_step():
            self._set_playing(False)

    def _on_seek(self, idx: int) -> None:
        self._video.seek(idx)

    def _on_frame_changed(self, idx: int) -> None:
        self._playback.set_frame(idx)
        self._timeline.set_frame(idx)
        self._task_panel.set_current_frame(idx)
        self._trim_panel.set_current_frame(idx)
        self._correction_panel.set_current_frame(idx)

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def _on_tasks_changed(self, tasks: dict) -> None:
        self._tasks = tasks
        self._timeline.set_tasks(tasks)
        self._save_tasks_to_disk()

    def _refresh_timeline(self) -> None:
        self._timeline.set_tasks(self._tasks)
        self._timeline.set_frame(self._video.frame_idx)

    def _save_tasks_to_disk(self) -> None:
        if self._rec_dir is None:
            return
        out_dir = self._rec_dir / "aoi_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        task_path = out_dir / "tasks.json"
        task_path.write_text(json.dumps(self._tasks, indent=2), encoding="utf-8")

    def _manual_save_tasks(self) -> None:
        self._tasks = self._task_panel.get_tasks()
        self._save_tasks_to_disk()
        self._save_tasks_btn.setText("  Saved ✓")
        QTimer.singleShot(1800, lambda: self._save_tasks_btn.setText("  Save"))

    def _load_tasks_from_disk(self) -> None:
        if self._rec_dir is None:
            return
        for task_path in [
            self._rec_dir / "aoi_results" / "tasks.json",
            self._rec_dir / "aoi_results" / "raw" / "tasks.json",
        ]:
            if task_path.exists():
                try:
                    data = json.loads(task_path.read_text(encoding="utf-8"))
                    self._task_panel.load_tasks(data)
                    return
                except Exception:
                    pass

    # ── Trim ──────────────────────────────────────────────────────────────────

    def _on_trim_changed(self, start: object, end: object) -> None:
        self._trim = self._trim_panel.get_trim()
        if self._rec_dir is None:
            return
        out_dir = self._rec_dir / "aoi_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trim.json").write_text(json.dumps(self._trim, indent=2), encoding="utf-8")

    def _on_crop_data_requested(self) -> None:
        if self._rec_dir is None:
            QMessageBox.warning(self, APP_TITLE, "Load a recording first.")
            return
            
        trim = self._trim_panel.get_trim()
        start = trim.get("start_frame")
        end = trim.get("end_frame")
        pad_s = trim.get("padding_s", 0)
        fps = self._fps if self._fps > 0 else 30.0
        pad_frames = int(pad_s * fps)
        
        start_f = (start - pad_frames) if start is not None else 0
        end_f = (end + pad_frames) if end is not None else self._n_frames
        
        raw_csv = self._rec_dir / "aoi_results" / "raw" / "analysis.csv"
        raw_fix = self._rec_dir / "aoi_results" / "raw" / "fixation_summary.csv"
        
        if not raw_csv.exists():
            QMessageBox.warning(self, APP_TITLE, "No analysis.csv found to crop.")
            return
            
        # Ask for confirmation
        res = QMessageBox.question(
            self, APP_TITLE, 
            f"This will PERMANENTLY remove data outside frames {max(0, start_f)} - {end_f} from the raw CSVs.\n\n"
            "This lets you instantly crop the data without re-running the 40-minute analysis.\n\n"
            "Are you sure you want to crop the data?", 
            QMessageBox.Yes | QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return
            
        try:
            import pandas as pd
            df = pd.read_csv(raw_csv)
            original_len = len(df)
            df = df[(df['frame_idx'] >= start_f) & (df['frame_idx'] <= end_f)]
            df.to_csv(raw_csv, index=False)
            
            if raw_fix.exists():
                fdf = pd.read_csv(raw_fix)
                fdf = fdf[(fdf['end_frame'] >= start_f) & (fdf['start_frame'] <= end_f)]
                fdf.to_csv(raw_fix, index=False)
                
            self._dashboard.reload()   # refresh charts automatically
            QMessageBox.information(self, APP_TITLE, f"Cropped data from {original_len} to {len(df)} frames. Dashboard updated.")
        except Exception as e:
            QMessageBox.critical(self, APP_TITLE, f"Failed to crop data: {e}")

    def _save_errors(self) -> None:
        if self._rec_dir is None:
            return
        out = self._rec_dir / "aoi_results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "errors.json").write_text(
            json.dumps(self._error_panel.get_errors(), indent=2), encoding="utf-8")
        self._dashboard.reload()

    def _load_errors_from_disk(self) -> None:
        self._error_panel.reset()
        if self._rec_dir is None:
            return
        for p in [self._rec_dir / "aoi_results" / "errors.json",
                  self._rec_dir / "aoi_results" / "raw" / "errors.json"]:
            if p.exists():
                try:
                    self._error_panel.load(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass
                return

    def _load_trim_from_disk(self) -> None:
        if self._rec_dir is None:
            return
        for trim_path in [
            self._rec_dir / "aoi_results" / "trim.json",
            self._rec_dir / "aoi_results" / "raw" / "trim.json",
        ]:
            if trim_path.exists():
                try:
                    data = json.loads(trim_path.read_text(encoding="utf-8"))
                    self._trim_panel.load_trim(data)
                    return
                except Exception:
                    pass

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_final(self) -> None:
        if self._rec_dir is None:
            QMessageBox.information(self, APP_TITLE, "Load a recording first.")
            return
        raw_csv = self._rec_dir / "aoi_results" / "raw" / "analysis.csv"
        if not raw_csv.exists() or raw_csv.stat().st_size == 0:
            QMessageBox.information(self, APP_TITLE,
                                    "Run Analysis first to generate the CSV.")
            return
        if not self._edit_labels:
            QMessageBox.information(self, APP_TITLE,
                                    "Load analysis results before exporting.")
            return

        try:
            raw_df = pd.read_csv(raw_csv)
            out_dir = self._rec_dir / "aoi_results"
            out_dir.mkdir(parents=True, exist_ok=True)

            df = _export_reviewed_csv(raw_df, self._edit_labels, self._edit_sources)
            out_csv = out_dir / "analysis.csv"
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            task_path = out_dir / "tasks.json"
            task_path.write_text(json.dumps(self._tasks, indent=2), encoding="utf-8")

            manual_n = sum(1 for s in self._edit_sources if s == "manual")
            gap_n = sum(1 for s in df["edit_source"].astype(str) if s == "auto_gap_fill")
            QMessageBox.information(
                self, APP_TITLE,
                f"Exported to:\n{out_csv}\n{task_path}\n\n"
                f"Manual corrections: {manual_n} frame(s)\n"
                f"Auto gap-fill: {gap_n} frame(s)",
            )
        except Exception as exc:
            QMessageBox.warning(self, APP_TITLE, f"Export failed:\n{exc}")

    # ── Help ──────────────────────────────────────────────────────────────────

    def _show_shortcuts_help(self) -> None:
        QMessageBox.information(self, f"{APP_TITLE} — Keyboard Shortcuts", (
            "<b>Playback</b><br>"
            "Space — Play / Pause<br>"
            "Left / Right Arrow — Step 1 frame<br>"
            "Prev / Next buttons — Step 1 second (30 frames)<br>"
            "<br><b>Tasks</b><br>"
            "I — Mark start of selected task at current frame<br>"
            "O — Mark end of selected task at current frame<br>"
            "<br><b>AOI Correction</b><br>"
            "Choose an AOI, then Frame for one frame, or mark In/Out and Apply for a range<br>"
            "F — Apply AOI to current frame<br>"
            "J — Range In &nbsp;·&nbsp; K — Range Out &nbsp;·&nbsp; L — Apply range<br>"
            "(corrections auto-save to analysis.csv)<br>"
            "<br><b>Trim</b><br>"
            "In / Out buttons — Mark the analysis range at the current frame<br>"
            "Clear — Remove trim, analyse the full recording"
        ))

    # ── Stylesheet ────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        check_svg = str(CONFIG_DIR / "check.svg").replace("\\", "/")
        T = Theme
        self.setStyleSheet(f"""
QWidget {{
    color: {T.TEXT};
    font-family: '{T.FONT_UI}', 'Segoe UI', sans-serif;
    font-size: 13px;
    background: transparent;
}}
QMainWindow, QDialog {{ background: {T.BG_BASE}; }}

QTabWidget::pane {{
    border: none;
    background: {T.BG_BASE};
}}
QTabBar {{ background: {T.BG_BASE}; }}
QTabBar::tab {{
    background: {T.BG_BASE};
    color: {T.TEXT_DIM};
    padding: 14px 16px;
    margin-right: 10px;
    border-bottom: 2px solid transparent;
    font-weight: 500;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    color: {T.TEXT_BRIGHT};
    border-bottom: 2px solid {T.ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {T.TEXT_SECONDARY}; }}

QFrame#leftPanel {{
    background: {T.BG_PANEL};
    border-right: 1px solid {T.BORDER_SUBTLE};
}}
QFrame#divider {{
    color: {T.BORDER_SUBTLE};
    background: {T.BORDER_SUBTLE};
    max-height: 1px;
    margin: 2px 0;
}}
QSplitter::handle {{
    background: {T.BORDER_SUBTLE};
}}
QSplitter::handle:hover {{
    background: {T.ACCENT};
}}
QLabel#sectionTitle {{
    color: {T.TEXT_FAINT};
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.1px;
    margin-top: 4px;
}}
QLabel#statusLabel {{
    color: {T.TEXT_DIM};
    font-family: '{T.FONT_MONO}', monospace;
    font-size: 12px;
    background: transparent;
    border: 1px solid {T.BORDER_SUBTLE};
    border-radius: {T.RADIUS}px;
    padding: 8px 10px;
}}

QPushButton {{
    background: transparent;
    border: 1px solid {T.BORDER};
    border-radius: {T.RADIUS}px;
    padding: 8px 12px;
    color: {T.TEXT_SECONDARY};
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: rgba(255,255,255,.22);
    color: {T.TEXT_BRIGHT};
}}
QPushButton:pressed {{ background: rgba(255,255,255,.04); }}
QPushButton#primaryButton {{
    background: {T.ACCENT};
    border-color: {T.ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background: #7e9bdb;
    border-color: #7e9bdb;
}}
QPushButton#primaryButton:pressed {{ background: #5f7dc0; }}
QPushButton#taskHeaderBtn {{
    background: {T.BG_FIELD};
    border: 1px solid {T.BORDER_SUBTLE};
    border-radius: 6px;
    padding: 1px 6px;
    color: {T.TEXT_DIM};
    font-size: 11px;
    font-weight: 500;
}}
QPushButton#taskHeaderBtn:hover {{ border-color: rgba(255,255,255,.22); color: {T.TEXT_SECONDARY}; }}
QPushButton#taskHeaderBtn:pressed {{ background: rgba(255,255,255,.04); }}
QPushButton#taskHeaderBtn:disabled {{ background: transparent; border-color: {T.BORDER_FAINT}; color: {T.TEXT_VFAINT}; }}
QPushButton#iconButton {{
    padding: 4px 6px;
    background: {T.BG_FIELD};
    border-color: {T.BORDER_SUBTLE};
}}
QPushButton#iconButton:hover {{ border-color: rgba(255,255,255,.22); }}

QWidget#taskHeader {{
    background: {T.BG_FIELD};
    border-radius: {T.RADIUS}px;
    border: 1px solid {T.BORDER_SUBTLE};
}}

QPushButton#collapsibleHeader {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 2px 0;
    color: {T.TEXT_FAINT};
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.1px;
    text-align: left;
}}
QPushButton#collapsibleHeader:hover {{
    color: {T.TEXT_SECONDARY};
    background: transparent;
}}

QComboBox {{
    background: transparent;
    border: 1px solid {T.BORDER};
    border-radius: {T.RADIUS}px;
    padding: 6px 10px;
    color: {T.TEXT_SECONDARY};
}}
QComboBox:hover {{ border-color: rgba(255,255,255,.2); }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: none; }}
QComboBox QAbstractItemView {{
    background: {T.BG_FIELD};
    border: 1px solid {T.BORDER};
    selection-background-color: {T.ACCENT};
    color: {T.TEXT_SECONDARY};
}}

QDoubleSpinBox, QSpinBox {{
    background: transparent;
    border: 1px solid {T.BORDER};
    border-radius: {T.RADIUS}px;
    padding: 4px 8px;
    color: {T.TEXT};
    font-family: '{T.FONT_MONO}', monospace;
}}

QSlider::groove:horizontal {{
    background: #1c1f24;
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {T.TEXT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: rgba(255,255,255,.35); border-radius: 2px; }}

QProgressBar {{
    background: {T.BG_FIELD};
    border: 1px solid {T.BORDER_SUBTLE};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 10px;
    color: {T.TEXT_FAINT};
}}
QProgressBar::chunk {{
    background: {T.ACCENT};
    border-radius: 4px;
}}

QScrollBar:vertical {{
    background: {T.BG_BASE};
    width: 9px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{ background: rgba(255,255,255,.10); border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(255,255,255,.18); }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QCheckBox {{ color: {T.TEXT_SECONDARY}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border-radius: 3px;
    border: 1.5px solid {T.BORDER};
    background: transparent;
}}
QCheckBox::indicator:hover {{ border-color: rgba(255,255,255,.3); }}
QCheckBox::indicator:checked {{
    background: {T.ACCENT};
    border-color: {T.ACCENT};
    image: url({check_svg});
}}
""")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    if sys.platform == "win32":
        # Without this, Windows groups the taskbar entry under python.exe and
        # shows the interpreter's icon instead of ours, no matter what
        # setWindowIcon is called with.
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AOIStudio.App")
        except Exception:
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    _load_app_fonts()
    app.setFont(QFont(Theme.FONT_UI, 10))
    icon = resolve_app_icon()
    if icon:
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
