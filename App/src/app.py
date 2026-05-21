from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from paths import APP_ICON, CONFIG_DIR, RECORDINGS_DIR, SRC_DIR

import locale
import json

def get_system_lang() -> str:
    loc, _ = locale.getlocale()
    if loc and loc.startswith("de"):
        return "de"
    return "en"

LANG = get_system_lang()

STRINGS = {
    "en": {
        "review_queue": "Review Queue",
        "source_folder": "Source Folder (Parent)",
        "recording_folder": "Recording Folder",
        "choose": "Choose...",
        "refresh": "Refresh",
        "rerun": "Re-run detection",
        "run_load": "Run / Load Review",
        "category": "Category",
        "edit_range": "Edit frame range inside selected segment",
        "start": "Start",
        "end": "End",
        "apply": "Apply Range",
        "undo": "Undo",
        "save_draft": "Save Draft",
        "save_final": "Save / Export Final",
        "auto_fill": "Auto Fill Short Gaps",
        "max_gap": "Max gap",
        "play": "Play",
        "dashboard": "Analytics Dashboard",
        "review_studio": "Review Studio",
    },
    "de": {
        "review_queue": "Überprüfungswarteschlange",
        "source_folder": "Quellordner (Übergeordnet)",
        "recording_folder": "Aufnahmeordner",
        "choose": "Auswählen...",
        "refresh": "Aktualisieren",
        "rerun": "Erkennung neu starten",
        "run_load": "Ausführen / Laden",
        "category": "Kategorie",
        "edit_range": "Frame-Bereich bearbeiten",
        "start": "Start",
        "end": "Ende",
        "apply": "Anwenden",
        "undo": "Rückgängig",
        "save_draft": "Entwurf speichern",
        "save_final": "Speichern / Export",
        "auto_fill": "Lücken füllen",
        "max_gap": "Max Lücke",
        "play": "Abspielen",
        "dashboard": "Analyse-Dashboard",
        "review_studio": "Überprüfungsstudio",
    }
}

def tr(key: str) -> str:
    return STRINGS.get(LANG, STRINGS["en"]).get(key, key)

APP_TITLE = "Neon AOI Review Studio"
NONE_LABEL = "None"
RAW_NONE_LABEL = "NoAOI"

AOI_NAMES = [
    "Board",
    "Left_Box",
    "Middle_Box",
    "Right_Box",
    "Screen",
    "Stream_Deck",
    "Points_Bar",
    "Progress_Bar",
    "Avatar",
    NONE_LABEL,
]

AOI_COLORS = {
    "Board": QColor("#F6AD55"),
    "Left_Box": QColor("#48BB78"),
    "Middle_Box": QColor("#ED8936"),
    "Right_Box": QColor("#38B2AC"),
    "Screen": QColor("#4299E1"),
    "Stream_Deck": QColor("#D53F8C"),
    "Points_Bar": QColor("#ECC94B"),
    "Progress_Bar": QColor("#68D391"),
    "Avatar": QColor("#9F7AEA"),
    NONE_LABEL: QColor("#718096"),
}

HELPER_AOI_COLUMNS = {
    "any_aoi_hit",
    "final_any_aoi_hit",
}
HELPER_AOI_LABELS = {
    "any_aoi",
    "any_aoi_hit",
    "final_any_aoi_hit",
}


def to_ui_label(value: object) -> str:
    if pd.isna(value):
        return NONE_LABEL
    label = str(value).strip()
    if not label or label == RAW_NONE_LABEL:
        return NONE_LABEL
    return label


@dataclass(frozen=True)
class Segment:
    aoi: str
    start: int
    end: int


class WorkerSignals(QObject):
    status = Signal(str)
    loaded = Signal(pathlib.Path)
    exported = Signal(pathlib.Path)
    failed = Signal(str)
    analysis_finished = Signal()


class VideoLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(340)
        self.setStyleSheet(
            "background: #111827; border-radius: 8px; color: #E5E7EB;"
        )
        self.setText("Load a recording to review the validation video.")


class TimelineWidget(QWidget):
    frameSelected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.labels: list[str] = []
        self.current_frame = 0
        self.is_scrubbing = False
        self.aoi_names: list[str] = list(AOI_NAMES)
        self.setMinimumHeight(270)
        self.setMouseTracking(True)
        self.tasks_data = {}


    def set_aoi_names(self, aoi_names: list[str]) -> None:
        self.aoi_names = list(aoi_names) or list(AOI_NAMES)
        self.setMinimumHeight(max(270, 25 * len(self.aoi_names) + 70))
        self.update()


    def set_tasks(self, tasks_data: dict) -> None:
        self.tasks_data = tasks_data
        self.update()

    def set_data(self, labels: list[str], current_frame: int) -> None:
        self.labels = labels
        self.current_frame = current_frame
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0D1117"))

        if not self.labels:
            painter.setPen(QColor("#64748B"))
            painter.drawText(20, 32, "Timeline appears after analysis is loaded.")
            return

        label_w = 118
        top = 14
        row_h = 25
        usable_w = max(1, self.width() - label_w - 16)
        total = max(1, len(self.labels))

        for row_idx, aoi in enumerate(self.aoi_names):
            y = top + row_idx * row_h
            painter.setPen(QColor("#E1E4E8"))
            painter.drawText(10, y + 17, aoi)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#21262D"))
            painter.drawRoundedRect(QRect(label_w, y + 4, usable_w, row_h - 8), 4, 4)
            self._paint_segments(painter, aoi, label_w, y + 4, usable_w, row_h - 8, total)


        # Draw task segments below AOI rows
        task_y = top + len(self.aoi_names) * row_h + 8
        painter.setPen(QColor("#8B949E"))
        painter.drawText(10, task_y + 12, "TASKS")
        for tname, tdata in self.tasks_data.items():
            start = tdata.get("start")
            end = tdata.get("end")
            if start is not None and end is not None and end > start:
                x1 = label_w + int((start / total) * usable_w)
                x2 = label_w + max(1, int((end / total) * usable_w))
                painter.setBrush(QColor("#1F6FEB"))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(QRect(x1, task_y, x2 - x1, 14), 4, 4)
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(QRect(x1, task_y, x2 - x1, 14), Qt.AlignCenter, tname)

        cursor_x = label_w + int((self.current_frame / total) * usable_w)
        painter.setPen(QColor("#EF4444"))
        painter.drawLine(cursor_x, top, cursor_x, top + len(self.aoi_names) * row_h)

    def _paint_segments(
        self,
        painter: QPainter,
        aoi: str,
        label_w: int,
        y: int,
        usable_w: int,
        height: int,
        total: int,
    ) -> None:
        color = AOI_COLORS.get(aoi, QColor("#94A3B8"))
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        idx = 0
        while idx < total:
            if self.labels[idx] != aoi:
                idx += 1
                continue
            start = idx
            while idx < total and self.labels[idx] == aoi:
                idx += 1
            end = idx
            x1 = label_w + int((start / total) * usable_w)
            x2 = label_w + max(1, int((end / total) * usable_w))
            painter.drawRoundedRect(QRect(x1, y, x2 - x1, height), 4, 4)

    def mousePressEvent(self, event) -> None:
        if not self.labels or event.button() != Qt.LeftButton:
            return
        frame = self._frame_from_x(event.position().toPoint().x())
        if frame is None:
            return
        self.is_scrubbing = True
        self.frameSelected.emit(frame)

    def mouseMoveEvent(self, event) -> None:
        if not self.is_scrubbing:
            return
        frame = self._frame_from_x(event.position().toPoint().x())
        if frame is not None:
            self.frameSelected.emit(frame)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.is_scrubbing = False

    def _aoi_from_y(self, y: int) -> str | None:
        row = int((y - 14) // 25)
        if 0 <= row < len(self.aoi_names):
            return self.aoi_names[row]
        return None

    def _frame_from_x(self, x: int) -> int | None:
        if not self.labels:
            return None
        label_w = 118
        usable_w = max(1, self.width() - label_w - 16)
        ratio = max(0.0, min(1.0, (x - label_w) / usable_w))
        return int(round(ratio * (len(self.labels) - 1)))


class NeonAoiQtApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        if APP_ICON.exists():
            self.setWindowIcon(QIcon(str(APP_ICON)))
        self.resize(1440, 900)

        self.recording_dir: pathlib.Path | None = None
        self.df: pd.DataFrame | None = None
        self.video_path: pathlib.Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.frame_count = 0
        self.current_frame = 0
        self.aoi_names: list[str] = list(AOI_NAMES)
        self.raw_labels: list[str] = []
        self.edited_labels: list[str] = []
        self.tasks_data: dict[str, dict[str, int | None]] = {f"Task {i}": {"start": None, "end": None} for i in range(1, 11)}
        self.edit_source: list[str] = []
        self.undo_stack: list[tuple[list[str], list[str], int]] = []
        self.playing = False
        self.play_until_frame: int | None = None
        self.analysis_worker_running = False
        self.watcher_process: subprocess.Popen | None = None
        self.signals = WorkerSignals()

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._autosave_review)
        self.preanalysis_timer = QTimer(self)
        self.preanalysis_timer.timeout.connect(self._poll_preanalysis_status)

        self.signals.status.connect(self._set_status)
        self.signals.loaded.connect(self._load_review)
        self.signals.exported.connect(self._export_done)
        self.signals.failed.connect(self._show_error)
        self.signals.analysis_finished.connect(self._analysis_worker_finished)

        self._build_ui()
        self._load_settings()
        self._wire_shortcuts()
        self._refresh_recordings()
        self._apply_style()
        self.preanalysis_timer.start(1000)
        self._start_watcher()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # TAB 1: REVIEW STUDIO
        self.review_tab = QWidget()
        root = QHBoxLayout(self.review_tab)
        root.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        self.tabs.addTab(self.review_tab, tr("review_studio"))

        # TAB 2: ANALYTICS DASHBOARD
        self.dashboard_tab = QWidget()
        dash_layout = QVBoxLayout(self.dashboard_tab)
        dash_layout.setContentsMargins(20, 20, 20, 20)
        
        dash_header = QHBoxLayout()
        self.dash_title = QLabel(tr("dashboard"))
        self.dash_title.setObjectName("panelTitle")
        self.refresh_dash_btn = QPushButton("Generate Dashboard")
        self.refresh_dash_btn.setObjectName("primaryButton")
        self.refresh_dash_btn.clicked.connect(self._render_dashboard)
        
        dash_header.addWidget(self.dash_title)
        dash_header.addStretch(1)
        dash_header.addWidget(self.refresh_dash_btn)
        dash_layout.addLayout(dash_header)
        
        self.web_view = QWebEngineView()
        dash_layout.addWidget(self.web_view, 1)
        self.tabs.addTab(self.dashboard_tab, tr("dashboard"))

        left = QFrame()
        left.setObjectName("sidePanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        title = QLabel(tr("review_queue"))
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)

        source_label = QLabel(tr("source_folder"))
        source_label.setObjectName("fieldLabel")
        left_layout.addWidget(source_label)
        
        source_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.choose_source_button = QPushButton(tr("choose"))
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(self.choose_source_button)
        left_layout.addLayout(source_row)

        recording_label = QLabel(tr("recording_folder"))
        recording_label.setObjectName("fieldLabel")
        left_layout.addWidget(recording_label)
        
        recording_row = QHBoxLayout()
        self.recording_combo = QComboBox()
        self.choose_button = QPushButton(tr("choose"))
        self.refresh_button = QPushButton(tr("refresh"))
        recording_row.addWidget(self.recording_combo, 1)
        recording_row.addWidget(self.choose_button)
        recording_row.addWidget(self.refresh_button)
        left_layout.addLayout(recording_row)

        self.force_checkbox = QCheckBox(tr("rerun"))
        left_layout.addWidget(self.force_checkbox)

        self.load_button = QPushButton(tr("run_load"))
        self.load_button.setObjectName("primaryButton")
        left_layout.addWidget(self.load_button)

        self.preanalysis_progress = QProgressBar()
        self.preanalysis_progress.setRange(0, 100)
        self.preanalysis_progress.setValue(0)
        self.preanalysis_progress.setTextVisible(True)
        self.preanalysis_progress.hide()
        left_layout.addWidget(self.preanalysis_progress)

        filter_label = QLabel(tr("category"))
        filter_label.setObjectName("fieldLabel")
        left_layout.addWidget(filter_label)
        self.category_filter = QComboBox()
        self.category_filter.addItems(["All"] + AOI_NAMES)
        self.category_filter.setCurrentText(NONE_LABEL)
        left_layout.addWidget(self.category_filter)

        self.segment_table = QTableWidget(0, 4)
        self.segment_table.setHorizontalHeaderLabels(["Play", "AOI", "Start", "End"])
        self.segment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.segment_table.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self.segment_table, 1)

        range_label = QLabel(tr("edit_range"))
        range_label.setObjectName("fieldLabel")
        left_layout.addWidget(range_label)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel(tr("start")))
        self.start_minus_button = QPushButton("-")
        self.start_minus_button.setFixedWidth(28)
        self.start_minus_button.setObjectName("iconButton")
        self.edit_start_spin = QSpinBox()
        self.edit_start_spin.setRange(0, 0)
        self.edit_start_spin.setSingleStep(1)
        self.edit_start_spin.setAccelerated(True)
        self.edit_start_spin.setMinimumWidth(86)
        self.edit_start_spin.setKeyboardTracking(False)
        self.edit_start_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.start_plus_button = QPushButton("+")
        self.start_plus_button.setFixedWidth(28)
        self.start_plus_button.setObjectName("iconButton")
        range_row.addWidget(self.start_minus_button)
        range_row.addWidget(self.edit_start_spin)
        range_row.addWidget(self.start_plus_button)
        range_row.addWidget(QLabel(tr("end")))
        self.end_minus_button = QPushButton("-")
        self.end_minus_button.setFixedWidth(28)
        self.end_minus_button.setObjectName("iconButton")
        self.edit_end_spin = QSpinBox()
        self.edit_end_spin.setRange(0, 0)
        self.edit_end_spin.setSingleStep(1)
        self.edit_end_spin.setAccelerated(True)
        self.edit_end_spin.setMinimumWidth(86)
        self.edit_end_spin.setKeyboardTracking(False)
        self.edit_end_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.end_plus_button = QPushButton("+")
        self.end_plus_button.setFixedWidth(28)
        self.end_plus_button.setObjectName("iconButton")
        range_row.addWidget(self.end_minus_button)
        range_row.addWidget(self.edit_end_spin)
        range_row.addWidget(self.end_plus_button)
        left_layout.addLayout(range_row)

        edit_row = QHBoxLayout()
        self.edit_to_combo = QComboBox()
        self.edit_to_combo.addItems(AOI_NAMES)
        self.edit_to_combo.setCurrentText("Board")
        self.edit_button = QPushButton(tr("apply"))
        edit_row.addWidget(self.edit_to_combo)
        edit_row.addWidget(self.edit_button)
        left_layout.addLayout(edit_row)

        undo_save_row = QHBoxLayout()
        self.undo_button = QPushButton(tr("undo"))
        self.save_draft_button = QPushButton(tr("save_draft"))
        self.save_button = QPushButton(tr("save_final"))
        self.save_button.setObjectName("primaryButton")
        undo_save_row.addWidget(self.undo_button)
        undo_save_row.addWidget(self.save_draft_button)
        undo_save_row.addWidget(self.save_button)
        left_layout.addLayout(undo_save_row)

        task_label = QLabel("Task Segmentation")
        task_label.setObjectName("fieldLabel")
        left_layout.addWidget(task_label)

        task_row = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems([f"Task {i}" for i in range(1, 11)])
        self.task_start_button = QPushButton("Set Start")
        self.task_end_button = QPushButton("Set End")
        self.task_clear_button = QPushButton("Clear")
        task_row.addWidget(self.task_combo, 1)
        task_row.addWidget(self.task_start_button)
        task_row.addWidget(self.task_end_button)
        task_row.addWidget(self.task_clear_button)
        left_layout.addLayout(task_row)

        self.status_label = QLabel("Select a recording.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("statusLabel")
        left_layout.addWidget(self.status_label)

        right = QFrame()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 0, 0, 0)
        right_layout.setSpacing(10)

        self.video_label = VideoLabel()
        right_layout.addWidget(self.video_label, 7)

        controls = QHBoxLayout()
        self.back_5_button = QPushButton("<< 5s")
        self.prev_frame_button = QPushButton("< Frame")
        self.play_button = QPushButton(tr("play"))
        self.next_frame_button = QPushButton("Frame >")
        self.forward_5_button = QPushButton("5s >>")
        self.gap_minus_button = QPushButton("-")
        self.gap_minus_button.setFixedWidth(28)
        self.gap_minus_button.setObjectName("iconButton")
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(33, 3000)
        self.gap_spin.setValue(500)
        self.gap_spin.setSuffix(" ms")
        self.gap_spin.setSingleStep(33)
        self.gap_spin.setKeyboardTracking(False)
        self.gap_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.gap_plus_button = QPushButton("+")
        self.gap_plus_button.setFixedWidth(28)
        self.gap_plus_button.setObjectName("iconButton")
        self.auto_fill_button = QPushButton(tr("auto_fill"))
        self.time_label = QLabel("00:00.000 / 00:00.000")
        controls.addWidget(self.back_5_button)
        controls.addWidget(self.prev_frame_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_frame_button)
        controls.addWidget(self.forward_5_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel(tr("max_gap")))
        controls.addWidget(self.gap_minus_button)
        controls.addWidget(self.gap_spin)
        controls.addWidget(self.gap_plus_button)
        controls.addWidget(self.auto_fill_button)
        controls.addStretch(1)
        controls.addWidget(self.time_label)
        right_layout.addLayout(controls)

        self.timeline = TimelineWidget()
        right_layout.addWidget(self.timeline, 3)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 980])

        self.source_combo.currentIndexChanged.connect(self._refresh_recordings)
        self.choose_source_button.clicked.connect(self._choose_source)
        self.recording_combo.currentIndexChanged.connect(self._poll_preanalysis_status)
        self.choose_button.clicked.connect(self._choose_recording)
        self.refresh_button.clicked.connect(self._refresh_recordings)
        self.load_button.clicked.connect(self._run_or_load)
        self.category_filter.currentTextChanged.connect(self._refresh_segments)
        self.segment_table.itemSelectionChanged.connect(self._segment_selection_changed)
        self.edit_start_spin.valueChanged.connect(self._clamp_edit_range)
        self.edit_end_spin.valueChanged.connect(self._clamp_edit_range)
        self.start_minus_button.clicked.connect(lambda: self._nudge_spin(self.edit_start_spin, -1))
        self.start_plus_button.clicked.connect(lambda: self._nudge_spin(self.edit_start_spin, 1))
        self.end_minus_button.clicked.connect(lambda: self._nudge_spin(self.edit_end_spin, -1))
        self.end_plus_button.clicked.connect(lambda: self._nudge_spin(self.edit_end_spin, 1))
        self.gap_minus_button.clicked.connect(lambda: self._nudge_spin(self.gap_spin, -33))
        self.gap_plus_button.clicked.connect(lambda: self._nudge_spin(self.gap_spin, 33))
        self.edit_button.clicked.connect(self._edit_selected_segment)
        self.undo_button.clicked.connect(self._undo)
        self.save_draft_button.clicked.connect(self._save_draft)
        self.save_button.clicked.connect(self._export_final)
        self.auto_fill_button.clicked.connect(self._auto_fill_short_gaps)
        self.task_start_button.clicked.connect(self._set_task_start)
        self.task_end_button.clicked.connect(self._set_task_end)
        self.task_clear_button.clicked.connect(self._clear_task)
        self.back_5_button.clicked.connect(lambda: self._jump_seconds(-5))
        self.forward_5_button.clicked.connect(lambda: self._jump_seconds(5))
        self.prev_frame_button.clicked.connect(lambda: self._step_frames(-1))
        self.next_frame_button.clicked.connect(lambda: self._step_frames(1))
        self.play_button.clicked.connect(self._toggle_play)
        self.timeline.frameSelected.connect(self._show_frame)

    def _wire_shortcuts(self) -> None:
        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self._undo)
        self.addAction(undo_action)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                color: #E1E4E8;
                font-family: 'Segoe UI', Inter, Roboto, sans-serif;
                font-size: 13px;
                background-color: transparent;
            }
            QMainWindow { background: #0D1117; }
            QTabWidget::pane {
                border: 1px solid #30363D;
                background: #0D1117;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #161B22;
                color: #8B949E;
                padding: 10px 20px;
                border: 1px solid #30363D;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #0D1117;
                color: #58A6FF;
                border-bottom: 2px solid #0D1117;
            }
            QTabBar::tab:hover:!selected {
                background: #1F2428;
                color: #C9D1D9;
            }
            QFrame#sidePanel {
                background: #161B22;
                border: 1px solid #30363D;
                border-radius: 12px;
            }
            QLabel#panelTitle {
                color: #E1E4E8;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QLabel#fieldLabel {
                color: #8B949E;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 11px;
                margin-top: 8px;
            }
            QLabel#statusLabel {
                color: #58A6FF;
                background: rgba(88, 166, 255, 0.1);
                border: 1px solid rgba(88, 166, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                font-weight: 500;
            }
            QCheckBox { color: #E1E4E8; spacing: 8px; }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #30363D;
                background: #0D1117;
            }
            QCheckBox::indicator:checked {
                background: #238636;
                border: 1px solid #2EA043;
                image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'><polyline points='20 6 9 17 4 12'></polyline></svg>");
            }
            QPushButton {
                background: #21262D;
                border: 1px solid #363B42;
                border-radius: 6px;
                padding: 8px 14px;
                color: #C9D1D9;
                font-weight: 600;
            }
            QPushButton:hover { 
                background: #30363D; 
                border: 1px solid #8B949E;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background: #282E33;
            }
            QPushButton#primaryButton {
                background: #238636;
                color: #FFFFFF;
                border: 1px solid #2EA043;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover { 
                background: #2EA043; 
                border: 1px solid #3FB950;
            }
            QPushButton#iconButton, QTableWidget QPushButton {
                padding: 0px;
                font-size: 16px;
                font-weight: bold;
            }
            QComboBox, QSpinBox {
                background: #0D1117;
                color: #E1E4E8;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 7px 12px;
                min-height: 22px;
            }
            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #8B949E;
            }
            QComboBox:focus, QSpinBox:focus {
                border: 1px solid #58A6FF;
                background: #161B22;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QComboBox QAbstractItemView {
                background: #161B22;
                color: #E1E4E8;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 4px;
                selection-background-color: #1F6FEB;
                selection-color: #FFFFFF;
                outline: none;
            }
            QTableWidget {
                background: #0D1117;
                color: #E1E4E8;
                border: 1px solid #30363D;
                border-radius: 8px;
                gridline-color: #21262D;
                selection-background-color: rgba(31, 111, 235, 0.3);
                selection-color: #FFFFFF;
            }
            QTableWidget::item { color: #E1E4E8; padding: 4px; }
            QHeaderView::section {
                background: #161B22;
                color: #8B949E;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #30363D;
                font-weight: 700;
                text-transform: uppercase;
                font-size: 10px;
            }
            QProgressBar {
                background: #0D1117;
                border: 1px solid #30363D;
                border-radius: 6px;
                color: #E1E4E8;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F6FEB, stop:1 #58A6FF);
                border-radius: 5px;
            }
            """
        )

    def _get_current_source_dir(self) -> pathlib.Path:
        data = self.source_combo.currentData()
        if data:
            return pathlib.Path(data)
        return pathlib.Path.home()

    def _load_settings(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.settings_path = CONFIG_DIR / "settings.json"
        self.app_settings = {"source_folders": []}
        if self.settings_path.exists():
            try:
                self.app_settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        folders = self.app_settings.get("source_folders", [])
        for f in folders:
            p = pathlib.Path(f)
            if p.exists():
                self.source_combo.addItem(p.name, str(p))

    def _save_settings(self) -> None:
        folders = []
        for i in range(self.source_combo.count()):
            folders.append(self.source_combo.itemData(i))
        self.app_settings["source_folders"] = folders
        self.settings_path.write_text(json.dumps(self.app_settings, indent=2), encoding="utf-8")

    def _refresh_recordings(self) -> None:
        source_dir = self._get_current_source_dir()
        if not source_dir.exists():
            return
        current = self.recording_combo.currentData()
        self.recording_combo.clear()
        for path in sorted(source_dir.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                self.recording_combo.addItem(path.name, str(path))
        if current:
            idx = self.recording_combo.findData(current)
            if idx >= 0:
                self.recording_combo.setCurrentIndex(idx)
        self._poll_preanalysis_status()

    def _choose_source(self) -> None:
        start_dir = self._get_current_source_dir()
        chosen = QFileDialog.getExistingDirectory(
            self,
            tr("choose"),
            str(start_dir),
        )
        if not chosen:
            return
        path = pathlib.Path(chosen)
        idx = self.source_combo.findData(str(path))
        if idx < 0:
            self.source_combo.addItem(path.name, str(path))
            idx = self.source_combo.count() - 1
        self.source_combo.setCurrentIndex(idx)
        self._save_settings()
        self._set_status(f"Added new source folder: {path.name}.")

    def _choose_recording(self) -> None:
        source_dir = self._get_current_source_dir()
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose Neon recording folder",
            str(source_dir),
        )
        if not chosen:
            return
        path = pathlib.Path(chosen)
        idx = self.recording_combo.findData(str(path))
        if idx < 0:
            self.recording_combo.addItem(path.name, str(path))
            idx = self.recording_combo.findData(str(path))
        self.recording_combo.setCurrentIndex(idx)
        self._set_status(f"Selected {path.name}.")
        self._poll_preanalysis_status()

    def _selected_recording(self) -> pathlib.Path | None:
        data = self.recording_combo.currentData()
        if not data:
            QMessageBox.warning(self, APP_TITLE, "Select a recording first.")
            return None
        path = pathlib.Path(data)
        if not path.exists():
            QMessageBox.warning(self, APP_TITLE, f"Recording folder does not exist:\n{path}")
            return None
        return path

    def _raw_dir_for(self, rec_dir: pathlib.Path) -> pathlib.Path:
        return rec_dir / "aoi_results" / "raw"

    def _read_preanalysis_progress(self, raw_dir: pathlib.Path) -> tuple[int, str]:
        progress_path = raw_dir / "progress.json"
        if not progress_path.exists():
            return 0, "starting"
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            percent = int(round(float(payload.get("percent", 0))))
            status = str(payload.get("status", "processing"))
            return max(0, min(100, percent)), status
        except Exception:
            return 0, "processing"

    def _poll_preanalysis_status(self) -> None:
        data = self.recording_combo.currentData()
        if not data:
            return
        rec_dir = pathlib.Path(data)
        raw_dir = self._raw_dir_for(rec_dir)
        processing = (raw_dir / ".processing").exists()
        if processing or self.analysis_worker_running:
            percent, status = self._read_preanalysis_progress(raw_dir)
            self.load_button.setEnabled(True)
            self.load_button.setText("Force Restart")
            self.preanalysis_progress.show()
            self.preanalysis_progress.setValue(percent)
            self.preanalysis_progress.setFormat(f"Pre-analysis {percent}%")
            self._set_status(
                f"Pre-analysis running or stuck. "
                f"{percent}% complete. Click 'Force Restart' to clear it."
            )
            return

        self.load_button.setEnabled(True)
        self.load_button.setText(tr("run_load"))
        self.preanalysis_progress.hide()

    def _run_or_load(self) -> None:
        rec_dir = self._selected_recording()
        if rec_dir is None:
            return
        raw_dir = self._raw_dir_for(rec_dir)
        if (raw_dir / ".processing").exists():
            reply = QMessageBox.question(
                self, 
                APP_TITLE, 
                "It looks like analysis is stuck from a previous crash. Do you want to force restart it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    (raw_dir / ".processing").unlink(missing_ok=True)
                except Exception:
                    pass
                self.analysis_worker_running = False
            else:
                return

        self.load_button.setEnabled(False)
        self.load_button.setText(tr("run_load"))
        self.preanalysis_progress.show()
        self.preanalysis_progress.setValue(0)
        self.preanalysis_progress.setFormat("Pre-analysis 0%")
        self.analysis_worker_running = True
        self._set_status("Preparing analysis data...")
        thread = threading.Thread(
            target=self._prepare_recording,
            args=(rec_dir, self.force_checkbox.isChecked()),
            daemon=True,
        )
        thread.start()

    def _prepare_recording(self, rec_dir: pathlib.Path, force: bool) -> None:
        try:
            raw_dir = self._raw_dir_for(rec_dir)
            if (raw_dir / ".processing").exists():
                self.signals.status.emit("Pre-analysis is still running. Please wait.")
                return
            analysis_csv = raw_dir / "analysis.csv"
            validation_video = raw_dir / "validation_video.mp4"
            if force or not analysis_csv.exists() or not validation_video.exists():
                self.signals.status.emit("Running marker detection and video overlay...")
                import analyzer

                analyzer.analyze_recording(rec_dir, raw_dir)
            self.signals.loaded.emit(rec_dir)
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.signals.analysis_finished.emit()

    def _start_watcher(self) -> None:
        watcher_path = SRC_DIR / "watcher.py"
        if self.watcher_process is not None and self.watcher_process.poll() is None:
            return
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.watcher_process = subprocess.Popen(
                [sys.executable, str(watcher_path)],
                cwd=str(SRC_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._set_status("Background watcher started.")
        except Exception as exc:
            self.watcher_process = None
            self._set_status(f"Could not start background watcher: {type(exc).__name__}: {exc}")

    def _stop_watcher(self) -> None:
        if self.watcher_process is None or self.watcher_process.poll() is not None:
            return
        self.watcher_process.terminate()
        try:
            self.watcher_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.watcher_process.kill()
            self.watcher_process.wait(timeout=2)

    def _discover_aoi_names(self, df: pd.DataFrame) -> list[str]:
        discovered: list[str] = []
        seen: set[str] = set()

        def add_name(value: object) -> None:
            label = to_ui_label(value)
            if label in ("All",) or label in HELPER_AOI_LABELS or label in seen:
                return
            seen.add(label)
            discovered.append(label)

        for column in df.columns:
            if not column.endswith("_hit") or column in HELPER_AOI_COLUMNS:
                continue
            add_name(column[:-4])

        if "primary_aoi" in df.columns:
            for value in df["primary_aoi"].dropna().unique():
                add_name(value)

        for name in AOI_NAMES:
            add_name(name)

        if NONE_LABEL in discovered:
            discovered = [name for name in discovered if name != NONE_LABEL] + [NONE_LABEL]
        return discovered

    def _sync_aoi_controls(self) -> None:
        current_filter = self.category_filter.currentText() if hasattr(self, "category_filter") else NONE_LABEL
        current_edit = self.edit_to_combo.currentText() if hasattr(self, "edit_to_combo") else "Board"

        self.category_filter.blockSignals(True)
        self.category_filter.clear()
        self.category_filter.addItems(["All"] + self.aoi_names)
        self.category_filter.setCurrentText(current_filter if current_filter in ["All"] + self.aoi_names else NONE_LABEL)
        self.category_filter.blockSignals(False)

        self.edit_to_combo.clear()
        self.edit_to_combo.addItems(self.aoi_names)
        self.edit_to_combo.setCurrentText(current_edit if current_edit in self.aoi_names else "Board")
        self.timeline.set_aoi_names(self.aoi_names)

    def _load_review(self, rec_dir: pathlib.Path) -> None:
        raw_dir = self._raw_dir_for(rec_dir)
        if (raw_dir / ".processing").exists():
            percent, status = self._read_preanalysis_progress(raw_dir)
            self.load_button.setEnabled(False)
            self.preanalysis_progress.show()
            self.preanalysis_progress.setValue(percent)
            self.preanalysis_progress.setFormat(f"Pre-analysis {percent}%")
            self._set_status(
                f"Pre-analysis is still running. Please wait. "
                f"{percent}% complete ({status})."
            )
            return
        analysis_csv = raw_dir / "analysis.csv"
        video_path = raw_dir / "validation_video.mp4"
        if (
            not analysis_csv.exists()
            or analysis_csv.stat().st_size == 0
            or not video_path.exists()
            or video_path.stat().st_size == 0
        ):
            QMessageBox.warning(
                self,
                APP_TITLE,
                "Missing or incomplete analysis.csv / validation_video.mp4. "
                "If pre-analysis just started, wait until it finishes and try again.",
            )
            self._poll_preanalysis_status()
            return

        self._close_video()
        self.recording_dir = rec_dir
        self.df = pd.read_csv(analysis_csv)
        self.aoi_names = self._discover_aoi_names(self.df)
        self._sync_aoi_controls()
        self.raw_labels = self.df["primary_aoi"].map(to_ui_label).tolist()
        self.edited_labels = list(self.raw_labels)
        self.edit_source = ["raw"] * len(self.edited_labels)
        self.undo_stack.clear()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            self._close_video()
            QMessageBox.warning(
                self,
                APP_TITLE,
                "The analysis video is incomplete or cannot be opened yet. "
                "Wait for pre-analysis to finish, or re-run detection.",
            )
            self._poll_preanalysis_status()
            return
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(self.edited_labels)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        restored = self._restore_autosave_if_available(rec_dir)
        self.current_frame = restored if restored is not None else 0
        self._load_tasks(rec_dir)
        self._show_frame(self.current_frame)
        self._refresh_segments()
        self.analysis_worker_running = False
        self.load_button.setEnabled(True)
        self.preanalysis_progress.hide()
        self._set_status(f"Loaded {rec_dir.name}. Focus on reducing None segments.")

    def _restore_autosave_if_available(self, rec_dir: pathlib.Path) -> int | None:
        state_path = rec_dir / "aoi_results" / "review_state.json"
        if not state_path.exists():
            return None
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            labels = payload.get("edited_labels")
            sources = payload.get("edit_source")
            saved_frame = payload.get("current_frame")
            if isinstance(labels, list) and len(labels) == len(self.edited_labels):
                self.edited_labels = [to_ui_label(label) for label in labels]
            if isinstance(sources, list) and len(sources) == len(self.edit_source):
                self.edit_source = [str(source) for source in sources]
            self._set_status("Recovered previous autosaved review state.")
            if isinstance(saved_frame, int): return saved_frame
            return None
        except Exception:
            return None

    def _close_video(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def _show_frame(self, frame_idx: int, seek: bool = True) -> None:
        if self.cap is None:
            return
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        if seek or frame_idx != self.current_frame + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_frame = frame_idx
        label = self.edited_labels[frame_idx] if frame_idx < len(self.edited_labels) else NONE_LABEL
        source = self.edit_source[frame_idx] if frame_idx < len(self.edit_source) else "raw"
        self._draw_preview_overlay(frame, label, source)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)
        self.timeline.set_data(self.edited_labels, self.current_frame)
        self.time_label.setText(
            f"{self._fmt_time(frame_idx / self.fps)} / {self._fmt_time(self.frame_count / self.fps)}"
        )

    @staticmethod
    def _draw_preview_overlay(frame: np.ndarray, label: str, source: str) -> None:
        h, w = frame.shape[:2]
        color = (84, 97, 110) if label == NONE_LABEL else (25, 118, 110)
        if source == "manual":
            color = (37, 99, 235)
        elif source == "auto_gap_fill":
            color = (8, 145, 178)
        cv2.rectangle(frame, (0, h - 46), (w, h), color, -1)
        text = f"CURRENT AOI: {label.replace('_', ' ')}"
        if source != "raw":
            text += f" ({source.replace('_', ' ')})"
        cv2.putText(
            frame,
            text,
            (14, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _toggle_play(self) -> None:
        if self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_until_frame = None
        self.play_button.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.play_timer.start(max(1, int(1000 / self.fps)))
        else:
            self.play_timer.stop()

    def _play_tick(self) -> None:
        if self.current_frame >= self.frame_count - 1:
            self.playing = False
            self.play_timer.stop()
            self.play_button.setText("Play")
            return
        if self.play_until_frame is not None and self.current_frame >= self.play_until_frame:
            self.playing = False
            self.play_until_frame = None
            self.play_timer.stop()
            self.play_button.setText("Play")
            return
        self._show_frame(self.current_frame + 1, seek=False)

    def _step_frames(self, delta: int) -> None:
        self.playing = False
        self.play_until_frame = None
        self.play_timer.stop()
        self.play_button.setText("Play")
        self._show_frame(self.current_frame + delta)

    def _jump_seconds(self, seconds: float) -> None:
        self._step_frames(int(round(seconds * self.fps)))

    def _play_segment(self, start: int, end: int) -> None:
        if self.cap is None:
            return
        self.playing = True
        self.play_until_frame = max(start, end)
        self.play_button.setText("Pause")
        self._show_frame(start)
        self.play_timer.start(max(1, int(1000 / self.fps)))

    def _manual_range_edit(self, start: int, end: int, aoi: str) -> None:
        self._push_undo()
        self._assign_range(start, end, aoi, "manual")
        self._after_edit(f"Assigned {aoi} from frame {start} to {end}.")

    def _edit_selected_segment(self) -> None:
        row = self.segment_table.currentRow()
        if row < 0:
            QMessageBox.information(self, APP_TITLE, "Select a segment row first.")
            return
        start_item = self.segment_table.item(row, 2)
        end_item = self.segment_table.item(row, 3)
        if start_item is None or end_item is None:
            return
        segment_start = int(start_item.text())
        segment_end = int(end_item.text())
        start = self.edit_start_spin.value()
        end = self.edit_end_spin.value()
        if start > end:
            QMessageBox.warning(self, APP_TITLE, "Start frame must be before end frame.")
            return
        if start < segment_start or end > segment_end:
            QMessageBox.warning(
                self,
                APP_TITLE,
                f"Edit range must stay inside selected segment "
                f"({segment_start} to {segment_end}).",
            )
            return
        target = self.edit_to_combo.currentText()
        self._push_undo()
        self._assign_range(start, end, target, "manual")
        self._after_edit(f"Changed frames {start} to {end} to {target}.")

    def _segment_selection_changed(self) -> None:
        row = self.segment_table.currentRow()
        if row < 0:
            return
        start_item = self.segment_table.item(row, 2)
        end_item = self.segment_table.item(row, 3)
        aoi_item = self.segment_table.item(row, 1)
        if start_item is None or end_item is None:
            return
        start = int(start_item.text())
        end = int(end_item.text())
        max_frame = max(0, len(self.edited_labels) - 1)
        self.edit_start_spin.setRange(0, max_frame)
        self.edit_end_spin.setRange(0, max_frame)
        self.edit_start_spin.setValue(start)
        self.edit_end_spin.setValue(end)
        if aoi_item is not None and aoi_item.text() in self.aoi_names:
            self.edit_to_combo.setCurrentText(aoi_item.text())
        self._show_frame(start)

    def _clamp_edit_range(self) -> None:
        sender = self.sender()
        start = self.edit_start_spin.value()
        end = self.edit_end_spin.value()
        if sender is self.edit_start_spin and start > end:
            self.edit_end_spin.blockSignals(True)
            self.edit_end_spin.setValue(start)
            self.edit_end_spin.blockSignals(False)
        elif sender is self.edit_end_spin and end < start:
            self.edit_start_spin.blockSignals(True)
            self.edit_start_spin.setValue(end)
            self.edit_start_spin.blockSignals(False)

    def _nudge_spin(self, spin: QSpinBox, delta: int) -> None:
        next_value = max(spin.minimum(), min(spin.maximum(), spin.value() + delta))
        spin.setValue(next_value)
        spin.setFocus(Qt.OtherFocusReason)

    def _assign_range(self, start: int, end: int, aoi: str, source: str) -> None:
        if not self.edited_labels:
            return
        end = min(end, len(self.edited_labels) - 1)
        start = max(0, start)
        for idx in range(start, end + 1):
            self.edited_labels[idx] = aoi
            self.edit_source[idx] = source

    def _auto_fill_short_gaps(self) -> None:
        if not self.edited_labels:
            return
        self._push_undo()
        frame_dur_ms = 1000.0 / self.fps if self.fps else 33.333
        max_gap_frames = max(1, int(round(self.gap_spin.value() / frame_dur_ms)))
        labels = list(self.edited_labels)
        filled_frames = 0
        filled_gaps = 0
        idx = 0
        while idx < len(labels):
            if labels[idx] != NONE_LABEL:
                idx += 1
                continue
            start = idx
            while idx < len(labels) and labels[idx] == NONE_LABEL:
                idx += 1
            end = idx
            before = labels[start - 1] if start > 0 else None
            after = labels[end] if end < len(labels) else None
            if before and before == after and before != NONE_LABEL and (end - start) <= max_gap_frames:
                self._assign_range(start, end - 1, before, "auto_gap_fill")
                filled_frames += end - start
                filled_gaps += 1
        self._after_edit(f"Auto-filled {filled_frames} frame(s) across {filled_gaps} short AOI gap(s).")

    def _auto_fill_board_gaps(self) -> None:
        self._auto_fill_short_gaps()

    def _push_undo(self) -> None:
        self.undo_stack.append((list(self.edited_labels), list(self.edit_source), self.current_frame))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return
        labels, sources, frame = self.undo_stack.pop()
        self.edited_labels = labels
        self.edit_source = sources
        self.current_frame = frame
        self._after_edit("Undo applied.")

    def _after_edit(self, message: str) -> None:
        self._show_frame(self.current_frame)
        self._refresh_segments()
        self._set_status(message + " Autosave pending.")
        self.autosave_timer.start(600)

    def _refresh_segments(self) -> None:
        segments = self._segments()
        category = self.category_filter.currentText()
        if category != "All":
            segments = [segment for segment in segments if segment.aoi == category]
        self.segment_table.blockSignals(True)
        self.segment_table.clearContents()
        self.segment_table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            play_button = QPushButton(tr("play"))
            play_button.setToolTip("Play this segment from start to end")
            play_button.clicked.connect(
                lambda _checked=False, start=segment.start, end=segment.end: self._play_segment(start, end)
            )
            self.segment_table.setCellWidget(row, 0, play_button)
            for col, value in enumerate([segment.aoi, segment.start, segment.end], start=1):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QBrush(QColor("#102A43")))
                self.segment_table.setItem(row, col, item)
        self.segment_table.blockSignals(False)
        max_frame = max(0, len(self.edited_labels) - 1)
        self.edit_start_spin.setRange(0, max_frame)
        self.edit_end_spin.setRange(0, max_frame)
        self.timeline.set_data(self.edited_labels, self.current_frame)

    def _segments(self) -> list[Segment]:
        segments: list[Segment] = []
        if not self.edited_labels:
            return segments
        idx = 0
        while idx < len(self.edited_labels):
            label = self.edited_labels[idx]
            start = idx
            while idx < len(self.edited_labels) and self.edited_labels[idx] == label:
                idx += 1
            segments.append(Segment(label, start, idx - 1))
        return segments

    def _autosave_review(self) -> None:
        if self.recording_dir is None or self.df is None:
            return
        out_dir = self.recording_dir / "aoi_results"
        state_path = out_dir / "review_state.json"
        payload = {
            "edited_labels": self.edited_labels,
            "edit_source": self.edit_source,
            "current_frame": self.current_frame,
        }
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._set_status(f"Autosaved review state: {state_path.name}")

    def _save_draft(self) -> None:
        if self.recording_dir is None or self.df is None:
            QMessageBox.information(self, APP_TITLE, "Load a recording first.")
            return
        self._autosave_review()
        self._set_status("Draft explicitly saved.")
        QMessageBox.information(self, APP_TITLE, "Progress has been saved. You can safely close the app and resume from this exact frame later.")

    def _save_tasks(self) -> None:
        rec_dir = self._selected_recording()
        if not rec_dir: return
        tasks_file = self._raw_dir_for(rec_dir) / "tasks.json"
        tasks_file.write_text(json.dumps(self.tasks_data, indent=2), encoding="utf-8")

    def _load_tasks(self, rec_dir: pathlib.Path) -> None:
        tasks_file = self._raw_dir_for(rec_dir) / "tasks.json"
        if tasks_file.exists():
            try:
                self.tasks_data = json.loads(tasks_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        else:
            self.tasks_data = {f"Task {i}": {"start": None, "end": None} for i in range(1, 11)}
        self.timeline.set_tasks(self.tasks_data)

    def _set_task_start(self) -> None:
        if not self.edited_labels: return
        t = self.task_combo.currentText()
        self.tasks_data[t]["start"] = self.current_frame
        self._save_tasks()
        self.timeline.set_tasks(self.tasks_data)
        self._set_status(f"Marked {t} start at frame {self.current_frame}")

    def _set_task_end(self) -> None:
        if not self.edited_labels: return
        t = self.task_combo.currentText()
        self.tasks_data[t]["end"] = self.current_frame
        self._save_tasks()
        self.timeline.set_tasks(self.tasks_data)
        self._set_status(f"Marked {t} end at frame {self.current_frame}")

    def _clear_task(self) -> None:
        if not self.edited_labels: return
        t = self.task_combo.currentText()
        self.tasks_data[t]["start"] = None
        self.tasks_data[t]["end"] = None
        self._save_tasks()
        self.timeline.set_tasks(self.tasks_data)
        self._set_status(f"Cleared {t}")

    def _export_final(self) -> None:
        if self.recording_dir is None or self.df is None or self.video_path is None:
            QMessageBox.information(self, APP_TITLE, "Load a recording before exporting.")
            return
        self._autosave_review()
        labels = list(self.edited_labels)
        sources = list(self.edit_source)
        df = self.df.copy()
        rec_dir = self.recording_dir
        video_path = self.video_path
        self._set_status("Exporting final CSV and validation video...")
        thread = threading.Thread(
            target=self._write_final_outputs,
            args=(rec_dir, df, labels, sources, video_path),
            daemon=True,
        )
        thread.start()

    def _write_final_outputs(
        self,
        rec_dir: pathlib.Path,
        df: pd.DataFrame,
        labels: list[str],
        sources: list[str],
        video_path: pathlib.Path,
    ) -> None:
        try:
            out_dir = rec_dir / "aoi_results"
            final_csv = out_dir / "analysis.csv"
            final_video = out_dir / "validation_video.mp4"
            row_count = min(len(df), len(labels))
            df = df.iloc[:row_count].copy()
            df["raw_primary_aoi"] = df["primary_aoi"].map(to_ui_label)
            df["final_primary_aoi"] = labels[:row_count]
            df["final_any_aoi_hit"] = df["final_primary_aoi"] != NONE_LABEL
            df["edit_source"] = sources[:row_count]
            df.to_csv(final_csv, index=False)
            self._render_final_video(video_path, final_video, labels, sources)
            self.signals.exported.emit(out_dir)
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")

    def _render_final_video(
        self,
        source_video: pathlib.Path,
        output_video: pathlib.Path,
        labels: list[str],
        sources: list[str],
    ) -> None:
        cap = cv2.VideoCapture(str(source_video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.fps or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                label = labels[idx] if idx < len(labels) else NONE_LABEL
                source = sources[idx] if idx < len(sources) else "raw"
                self._draw_preview_overlay(frame, label, source)
                writer.write(frame)
                idx += 1
        finally:
            cap.release()
            writer.release()

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        rest = seconds - minutes * 60
        return f"{minutes:02d}:{rest:06.3f}"

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _export_done(self, out_dir: pathlib.Path) -> None:
        self._set_status(f"Exported final outputs to {out_dir}.")
        QMessageBox.information(self, APP_TITLE, f"Exported final outputs to:\n{out_dir}")

    def _show_error(self, message: str) -> None:
        self._set_status("Failed.")
        self.analysis_worker_running = False
        self._poll_preanalysis_status()
        QMessageBox.critical(self, APP_TITLE, message)

    def _analysis_worker_finished(self) -> None:
        self.analysis_worker_running = False
        self._poll_preanalysis_status()

    def closeEvent(self, event) -> None:
        self._close_video()
        self._stop_watcher()
        super().closeEvent(event)


    def _render_dashboard(self) -> None:
        if not self.edited_labels:
            self._set_status("No data to render dashboard.")
            return
            
        self._set_status("Generating Dashboard...")
        
        counts = {}
        for lbl in self.edited_labels:
            counts[lbl] = counts.get(lbl, 0) + 1
            
        for k in counts:
            counts[k] = counts[k] / self.fps
            
        import pandas as pd
        df_dash = pd.DataFrame({
            "AOI": list(counts.keys()),
            "Dwell Time (s)": list(counts.values())
        })
        
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Dwell Time", "Learning Curve"))
        
        # Chart 1: Dwell Time Bar
        bar = px.bar(df_dash, x="AOI", y="Dwell Time (s)", color="AOI")
        for trace in bar.data:
            fig.add_trace(trace, row=1, col=1)
            
        # Chart 2: Learning Curve
        task_names = []
        task_durations = []
        for i in range(1, 11):
            tname = f"Task {i}"
            tdata = self.tasks_data.get(tname, {})
            start = tdata.get("start")
            end = tdata.get("end")
            if start is not None and end is not None and end > start:
                task_names.append(tname)
                task_durations.append((end - start) / self.fps)
                
        if task_names:
            line = go.Scatter(x=task_names, y=task_durations, mode='lines+markers', name="Learning Curve", marker=dict(size=12, color="#58A6FF"), line=dict(width=4, color="#1F6FEB"))
            fig.add_trace(line, row=1, col=2)
            fig.update_xaxes(title_text="Task", row=1, col=2)
            fig.update_yaxes(title_text="Duration (s)", row=1, col=2)
        else:
            fig.add_annotation(text="No Task Markers set yet.", xref="paper", yref="paper", x=0.75, y=0.5, showarrow=False, font=dict(size=16, color="white"))
            
        fig.update_layout(template="plotly_dark", showlegend=False, title_text="Analytics Dashboard", title_font=dict(size=24))
        
        raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        self.web_view.setHtml(raw_html)
        self._set_status("Dashboard generated successfully.")




def main() -> int:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    if APP_ICON.exists():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    window = NeonAoiQtApp()
    window.show()
    return app.exec()



if __name__ == "__main__":
    raise SystemExit(main())
