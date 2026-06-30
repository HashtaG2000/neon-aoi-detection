# AOI Studio — System Design

Visual architecture for the Pupil Labs AOI analysis desktop app (THOWL gamification experiment).
All diagrams are **Mermaid** — they render in VS Code (Markdown preview) and on GitHub. Paste any
block into <https://mermaid.live> to export PNG/SVG.

---

## 1. Layered architecture (components & responsibilities)

```mermaid
flowchart TB
    subgraph USER["👤 Researcher"]
        BAT["START_APP.bat<br/>(single entry point)"]
    end

    subgraph UI["🖥️ Desktop UI — app.py (PySide6 / Qt)"]
        MW["MainWindow<br/>(QMainWindow)"]
        T1["Studio Tab<br/>VideoWidget · PlaybackBar<br/>AOITimeline · Trim/Correction/Task panels"]
        T2["Dashboard Tab<br/>Plotly charts via QWebEngineView"]
        T3["Comparison Tab<br/>NonGamified vs Gamified"]
        MW --> T1 & T2 & T3
    end

    subgraph WORKERS["⚙️ Background threads (QThread/Signals)"]
        AW["AnalysisWorker"]
        BW["BatchAnalysisWorker"]
        CW["ComparisonWorker"]
    end

    subgraph ENGINE["🧠 Processing engine"]
        AN["analyzer.py<br/>per-frame AOI inference"]
        FD["feature_detector.py<br/>ORB fallback matching"]
        MK["masks.py<br/>surface-space AOI masks"]
        RP["reporting.py<br/>master workbook + comparison"]
    end

    subgraph VENDOR["📦 Vendored Pupil Labs libs (vendor/)"]
        NR["pl-neon-recording<br/>scene · gaze · fixations"]
        MM["pl-marker-mapper<br/>3D surface model"]
        AT["pupil_apriltags<br/>tag36h11 detector"]
    end

    subgraph CFG["🗂️ Config — App/config/"]
        AOIS["aois.json<br/>AOI→marker IDs + colors"]
        MASKS["aoi_masks.json"]
        SET["settings.json<br/>source_folders"]
    end

    subgraph FS["💾 Recordings on disk"]
        REC["<recording>/<br/>Scene Camera.mp4 + gaze/fixation data"]
        OUT["aoi_results/raw/<br/>analysis.csv · fixation_summary.csv<br/>data_quality.json · progress.json · tasks.json"]
        MASTER["aoi_master/ · aoi_comparison/"]
    end

    BAT --> MW
    PATHS["paths.py<br/>central path registry + sys.path injection"]
    MW -. spawns .-> AW & BW & CW
    AW & BW --> AN
    CW --> RP
    AN --> FD & MK
    AN --> NR & MM & AT
    AN -->|reads| AOIS & MASKS & REC
    AN -->|writes| OUT
    RP -->|reads| OUT
    RP -->|writes| MASTER
    MW -->|reads| SET & AOIS
    T1 -->|reads frames/gaze| NR
    UI -. all imports .-> PATHS
    ENGINE -. all imports .-> PATHS
```

---

## 2. Analysis data pipeline (`analyzer.analyze_recording`)

```mermaid
flowchart LR
    A["Scene video frames<br/>(NeonRecording)"] --> B["AprilTag detect<br/>tag36h11"]
    B -->|markers visible| C["3D Marker Mapper<br/>surface model"]
    B -->|partial markers| D["2D convex-hull<br/>fallback polygon"]
    B -.->|tags fail| E["ORB feature match<br/>feature_detector.py"]
    C & D & E --> F["AOI polygons<br/>+10% dilation"]
    G["Gaze stream<br/>(per timestamp)"] --> H["Hit-test gaze<br/>in surface space"]
    F --> H
    M["aoi_masks.json"] -.-> H
    H --> I["AOI label per frame<br/>(priority resolved)"]
    I --> J["Fixation grouping<br/>+ gap-fill (≤15 frames)"]
    J --> K{{"Outputs to aoi_results/raw/"}}
    K --> K1["analysis.csv"]
    K --> K2["fixation_summary.csv"]
    K --> K3["data_quality.json"]
    K --> K4["progress.json (live)"]
```

---

## 3. User workflow across the three tabs

```mermaid
flowchart TD
    S0(["Launch START_APP.bat"]) --> S1["Pick source folder<br/>(settings.json source_folders)"]
    S1 --> S2["Select a recording"]

    subgraph STUDIO["STUDIO TAB"]
        S2 --> S3["Play scene video<br/>live surface + gaze overlay"]
        S3 --> S4["Trim dead time (Set In/Out)"]
        S4 --> S5["Run Analysis → AnalysisWorker"]
        S5 -->|progress.json polled| S6["AOI timeline fills in"]
        S6 --> S7["Manual corrections<br/>(relabel frame/range)"]
        S7 --> S8["Annotate Tasks T1–T10<br/>(tasks.json)"]
        S8 --> S9["Export → analysis.csv finalized"]
    end

    subgraph DASH["DASHBOARD TAB"]
        S9 --> D1["Plotly charts:<br/>dwell %, fixations, transitions, quality"]
    end

    subgraph COMP["COMPARISON TAB"]
        D1 --> C1["Export Workbook per condition<br/>(reporting → aoi_master/)"]
        C1 --> C2["Compare NonGamified vs Gamified<br/>ComparisonWorker"]
        C2 --> C3["Save PNGs → aoi_comparison/"]
    end
```

---

## 4. Threading & live-progress model

```mermaid
sequenceDiagram
    participant UI as MainWindow (UI thread)
    participant W as AnalysisWorker (thread)
    participant AN as analyzer.py
    participant FS as aoi_results/raw/

    UI->>W: start(rec_dir, trim)
    W->>AN: analyze_recording(...)
    loop per frame
        AN->>FS: write progress.json
    end
    UI-->>FS: QTimer polls progress.json
    AN->>FS: analysis.csv, fixation_summary.csv, data_quality.json
    AN-->>W: return
    W-->>UI: Signal finished()
    UI->>UI: reload timeline + dashboard
```

---

### How to view
- **VS Code:** open this file, `Ctrl+Shift+V` for Markdown preview (install *Markdown Preview Mermaid Support* if blocks don't render).
- **GitHub:** renders automatically on push.
- **Export image:** copy a block into <https://mermaid.live> → Actions → PNG/SVG.
