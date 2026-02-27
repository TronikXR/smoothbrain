# Code Review Findings — Accuracy Verification

Each finding was checked against the actual source code. Verdicts below.

---

## 1. High: output-file race — ✅ Accurate (with nuance)

**Claim**: `_find_newest_output()` picks the newest file by mtime, risking wrong-file attachment.

**Verified**: The function at [plugin.py:1042–1067](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py#L1042-L1067) does:
1. Glob the entire output directory for matching extensions
2. Filter by `since_ts` (recorded just before render)
3. Return `max(candidates, key=os.path.getmtime)`

The `since_ts` filter **partially mitigates** the race — it won't pick old files. But the finding is still valid: any other process writing to the same output dir *during or just after* the render window can win the `max(mtime)` race. Called at lines 1528 and 2036 exactly as cited.

**Severity nuance**: Real-world risk is low in single-user Gradio, but the finding is technically correct.

---

## 2. High: installer trust model — ✅ Accurate

**Verified** at [ollama.py:50–51](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/ollama.py#L50-L51):
- Line 50: `_OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"`
- Line 51: `_OLLAMA_LINUX_CMD = "curl -fsSL https://ollama.com/install.sh | sh"`

At line 99–100, `urllib.request.urlretrieve` downloads without checksum.  
At line 111–112, the installer runs silently with `/VERYSILENT`.  
At line 127, Linux runs `bash -c` with the pipe-to-sh command.

All line references confirmed. No checksum/signature verification anywhere.

---

## 3. Medium: non-atomic project save — ✅ Accurate

**Verified** at [state.py:140–148](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/state.py#L140-L148):

```python
path = os.path.join(project_dir, "project.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=str)
```

Direct write to `project.json` — truncates on open, no temp-file + `os.replace()` pattern. Interruption mid-write produces corrupt/truncated JSON. Line reference (145) is accurate.

---

## 4. Medium: filename collision in `copy_to_project` — ✅ Accurate

**Verified** at [state.py:209–227](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/state.py#L209-L227):

```python
if os.path.exists(dest):
    ts = int(time.time())       # second-level precision
    name, ext = os.path.splitext(basename)
    dest = os.path.join(dest_dir, f"{name}_{ts}{ext}")
```

Uses `int(time.time())` — two copies in the same second get the same suffix. No uniqueness loop, no UUID fallback. Line 217 confirmed.

---

## 5. Medium: broad `except Exception` — ✅ Accurate

**Verified locations**:

| Reference | Line | Actual Code |
|-----------|------|-------------|
| [model_scanner.py:186](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/model_scanner.py#L186) | 186 | `except Exception:` in `_resolve_model_urls` — silently sets `urls = []` |
| [model_scanner.py:234](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/model_scanner.py#L234) | 234 | `except Exception: continue` in `_scan_folder` — skips bad JSON silently |
| [model_scanner.py:325](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/model_scanner.py#L325) | 325 | `except Exception: continue` in `scan_image_models` — same pattern |
| [ollama.py:334](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/ollama.py#L334) | 334 | `except Exception:` in `_extract_json` — bare catch on JSON parse |

All four locations confirmed. All swallow exceptions silently (no logging of what failed or why).

---

## 6. Low: shared cancel flag — ✅ Accurate

**Verified**:
- [plugin.py:80](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py#L80): `self._render_cancelled = False` — single flag on instance
- [plugin.py:1470](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py#L1470): `self._render_cancelled = False` in image render loop
- [plugin.py:1998](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py#L1998): `self._render_cancelled = False` in video render loop

Same flag checked at 1480 and 2008. Both pipelines share it.

**Severity nuance**: In Gradio, the UI serializes event handlers per session, so simultaneous image+video renders are unlikely. But the finding is technically correct.

---

## 7. Low: unpinned httpx — ✅ Accurate

**Verified** at [requirements.txt:1](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/requirements.txt#L1):

```
httpx
```

Single line, no version pin. Confirmed.

---

## 8. Low: stale/unused imports — ⚠️ Partially Accurate

**Verified** at [plugin.py:26–33](file:///F:/pinokio/api/wan.git/app/plugins/smooth_brain/plugin.py#L26-L33):

```python
from .state import (
    SmoothBrainSession, ShotState,
    save_session, load_session, clear_session,
    build_video_params, is_ltx_model, vibe_to_resolution,
    duration_to_frames, session_age_minutes,
    ...
)
```

- `save_session` — imported at line 28, **not used** in plugin.py ✅
- `load_session` — imported at line 28, **not used** in plugin.py ✅
- `session_age_minutes` — imported at line 30, **not used** in plugin.py ✅

> [!NOTE]
> The cited line numbers (28, 30) are correct. These are legacy session functions superseded by `save_project`/`load_project`. Additionally, `ShotState`, `build_video_params`, and `clear_session` should be checked — `ShotState` and `clear_session` appear unused in the viewed code as well, but confirming their usage throughout the full 2171-line file would require a full grep.

---

## Summary

| # | Finding | Verdict | Notes |
|---|---------|---------|-------|
| 1 | Output-file race | ✅ Correct | `since_ts` mitigates but doesn't eliminate |
| 2 | Installer trust | ✅ Correct | All refs confirmed |
| 3 | Non-atomic save | ✅ Correct | Direct file write |
| 4 | Filename collision | ✅ Correct | `int(time.time())` once |
| 5 | Broad except | ✅ Correct | All 4 locations confirmed |
| 6 | Shared cancel flag | ✅ Correct | Low practical risk in Gradio |
| 7 | Unpinned httpx | ✅ Correct | Single unpinned dep |
| 8 | Unused imports | ⚠️ Partial | 3 of 3 cited are genuinely unused; possibly more |

**Verdict**: All findings are substantively accurate. Findings #1 and #6 note real code issues but have lower practical severity than stated due to Gradio's single-threaded event model. Finding #8 likely understates scope (more dead imports may exist).
