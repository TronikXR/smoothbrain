# Smooth Brain Plugin — Code Quality Assessment

What to fix now vs. what to set up for the future.

---

## Immediate — Fix Before Adding More Features

### 1. `plugin.py` is too large (2171 lines, 66 methods)

This is the single biggest barrier to future velocity. Every new feature touches this file, merge conflicts are guaranteed, and navigating it is painful. Today it contains:

- UI construction (4 `_build_step*` methods, ~500 lines)
- Event wiring (4 `_wire_step*` methods, ~400 lines)
- Render orchestration (`_queue_image_renders`, `_export_videos`, ~400 lines)
- State management helpers (~200 lines)
- Internal plumbing (`_build_task`, `_find_newest_output`, `_build_cli_state`)

**Recommended split:**

| New file | What moves there | ~Lines |
|----------|-----------------|--------|
| `ui_builder.py` | `_build_step1` through `_build_step4`, CSS, HTML helpers | 500 |
| `render_engine.py` | `_queue_image_renders`, `_export_videos`, `_find_newest_output`, `_run_render_tasks`, `_build_task`, `_build_cli_state` | 400 |
| `wiring.py` | `_wire_step1` through `_wire_step4`, `_wire_navigation`, approve/reject handlers | 400 |

The main `plugin.py` becomes a thin coordinator (~400 lines) that imports and composes.

### 2. Atomic project saves

This is the easiest high-impact fix. One function change in [state.py:140-148](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/state.py#L140-L148):

```python
# Write to temp, then atomically replace
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=str)
os.replace(tmp, path)  # atomic on same filesystem
```

User projects are the most important state in the plugin — losing them to a crash mid-save is unacceptable.

### 3. Narrow the `except Exception` blocks

The 4 locations in `model_scanner.py` and `ollama.py` silently swallow errors. When a model JSON is malformed or a file is locked, the user sees "no models found" with zero diagnostic info.

Fix pattern:
```python
except (json.JSONDecodeError, OSError) as e:
    print(f"[model_scanner] Skipping {fname}: {e}")
    continue
```

This costs ~10 minutes and prevents hours of future debugging.

### 4. Clean dead imports

Remove `save_session`, `load_session`, `session_age_minutes` (and check `ShotState`, `clear_session`). Takes 2 minutes, prevents confusion about which save system is "the real one."

---

## Near-Term — Set Up Before the Next Major Feature

### 5. Output attribution by task ID, not filesystem race

`_find_newest_output()` is a workaround for not having structured return data from `process_tasks_cli`. The right fix depends on what `process_tasks_cli` actually returns:

- **If it can return the output path** → use it directly, delete `_find_newest_output` entirely
- **If it can't** → use per-shot output subdirectories (e.g. `outputs/sb_shot_1/`) to isolate results

Either way, the current mtime-race approach will break the moment you add batch rendering, retries, or multi-user support.

### 6. Structured logging instead of `print()`

The plugin uses `print()` everywhere (~40 calls). When a user reports "it just didn't work," you're reading raw stdout dumps. Replace with Python `logging`:

```python
import logging
log = logging.getLogger("smooth_brain")
log.info("Shot %d → %s", shot_i+1, output_path)
log.warning("Refinement returned %d/%d items — skipping", ...)
log.error("Render failed", exc_info=True)
```

Benefits: log levels, filtering, timestamps, and Pinokio's log system can capture them properly.

### 7. Pin `httpx` version range

```
httpx>=0.27,<1.0
```

One line, prevents a surprise breakage when httpx 1.0 ships with API changes.

### 8. Add a minimal test suite

There are **zero tests** today. Before adding more features, write at least:

| Test | What it covers | Risk if missing |
|------|---------------|-----------------|
| `test_state.py` | `save_project` → `load_project` roundtrip, `copy_to_project` collision handling | Data loss |
| `test_model_scanner.py` | `_scan_folder` with malformed JSON, missing dirs | Silent model disappearance |
| `test_ollama.py` | `_extract_json` / `_extract_json_array` edge cases | Bad LLM output crashes |
| `test_story_templates.py` | `fill_template` with empty subject, `get_weighted_templates` with all-zero weights | Crash on edge input |

These are all pure-function tests — no GPU, no server needed. Can run in <1 second.

---

## Longer-Term — Architectural Improvements for Scale

### 9. Event-driven render pipeline

Currently, render loops are synchronous generators yielding Gradio updates. This works for one user, but:

- Can't parallelize (GPU-bound anyway, but the *orchestration* is locked up)
- Can't persist partial state across browser refreshes
- Adding "batch re-render rejected" or "render queue across projects" requires rewriting the loop

**Future direction**: a proper `RenderQueue` class with:
- Job submission (with UUID per job)
- Status polling (replaces `_render_cancelled` flag)
- Persistent queue state (survives tab close)
- Per-job cancellation tokens (fixes finding #6)

### 10. Separate cancellation tokens

Today's single `self._render_cancelled` flag is fine for Gradio's serialized model, but if you ever add:
- WebSocket-based progress
- Background batch mode
- API endpoint for external triggering

...you'll need per-pipeline or per-job tokens. A simple `CancellationToken` class:

```python
class CancelToken:
    def __init__(self): self._cancelled = False
    def cancel(self): self._cancelled = True
    @property
    def is_cancelled(self): return self._cancelled
```

### 11. Config file for plugin settings

Hardcoded values are scattered across files:
- `MAX_SHOTS = 12`, `MAX_CHARS = 4` (plugin.py)
- `DEFAULT_MODEL = "qwen2.5:3b"` (ollama.py)
- `TIMEOUT = 120.0` (ollama.py)
- `VRAM_THRESHOLDS` (gpu_utils.py)

A single `config.py` (or user-editable `config.json`) would let users tune behavior without modifying code.

### 12. Installer trust model

The Ollama auto-installer downloads and runs executables without verification. Before this ships to a wider audience:
- Add SHA256 verification for the Windows installer
- Consider distributing Ollama as a pinned dependency rather than runtime-downloading it
- At minimum, warn the user before auto-installing system software

---

## Priority Summary

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 Now | Atomic project saves | 10 min | Prevents data loss |
| 🔴 Now | Narrow exception catches | 30 min | Debuggability |
| 🔴 Now | Clean dead imports | 5 min | Code clarity |
| 🟡 Soon | Split plugin.py | 2-3 hrs | Maintainability, merge sanity |
| 🟡 Soon | Output by task ID | 1-2 hrs | Correctness |
| 🟡 Soon | Structured logging | 1 hr | Diagnosability |
| 🟡 Soon | Pin httpx | 1 min | Stability |
| 🟡 Soon | Basic test suite | 2-3 hrs | Regression safety |
| 🔵 Later | Render queue architecture | Days | Scalability |
| 🔵 Later | Cancel tokens | 1 hr | Concurrency safety |
| 🔵 Later | Config system | 2 hrs | Customizability |
| 🔵 Later | Installer verification | 2 hrs | Security |
