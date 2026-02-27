# Immediate Quality Fixes — Walkthrough

Three files modified, no behavior changes to the user — just safer internals.

---

## 1. Atomic Project Saves — [state.py](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/state.py)

render_diffs(file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/state.py)

`save_project()` now writes to `project.json.tmp` first, then atomically swaps via `os.replace()`. A crash mid-write can no longer corrupt the project file.

---

## 2. Narrowed Exception Catches

### [model_scanner.py](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/model_scanner.py) — 3 locations

render_diffs(file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/model_scanner.py)

All three `except Exception` blocks now catch `(json.JSONDecodeError, OSError)` and log the filename + error. Malformed model configs will show up in the terminal instead of silently vanishing.

### [ollama.py](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/ollama.py) — 6 locations

render_diffs(file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/ollama.py)

JSON extraction helpers now catch `(json.JSONDecodeError, ValueError)` instead of bare `Exception`.

---

## 3. Dead Import Removal — [plugin.py](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py)

render_diffs(file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py)

Removed **7 unused imports**: `SmoothBrainSession`, `ShotState`, `save_session`, `load_session`, `clear_session`, `build_video_params`, `session_age_minutes`. All legacy session functions superseded by the project-based persistence system.
