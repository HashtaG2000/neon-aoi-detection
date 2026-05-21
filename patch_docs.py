import re

with open("PROJECT_DOCUMENTATION.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add tasks.json to config/runtime locations
content = content.replace(
    "| Watcher log / lock | `App/runtime/` |",
    "| Watcher log / lock | `App/runtime/` |\n| Task Segmentation | `<recording_dir>/aoi_results/tasks.json` |"
)

# 2. Add Phase 6 development history
phase_6 = """### Phase 6 — Analytics Dashboard & Task Segmentation · *Antigravity*

**Problem:** The user requested an advanced learning curve analysis feature and a dark-mode graphical dashboard embedded directly into the application. Additionally, a robust "Force Restart" mechanism was needed to handle edge cases where the `watch_and_analyze.py` lock file (`.processing`) became orphaned due to a crash.

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

"""

content = content.replace(
    "## 6. Current File Reference",
    phase_6 + "## 6. Current File Reference"
)

# 3. Update the App UI features list
app_features = """- **Task Segmentation UI**: Map Start/End timestamps for up to 10 analytical Tasks, saved to `tasks.json`.
- **Analytics Dashboard**: Embedded PySide6-WebEngine Plotly dashboard showing Total Dwell Time and a dynamic Learning Curve graph.
- **Force Restart / Orphan Lock Recovery**: Automatically detects crashed background analyses and allows 1-click safe recovery."""

content = content.replace(
    "- Video player: play, pause, step frame, jump ±5 seconds",
    "- Video player: play, pause, step frame, jump ±5 seconds\n" + app_features
)

with open("PROJECT_DOCUMENTATION.md", "w", encoding="utf-8") as f:
    f.write(content)
