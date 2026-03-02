# Smooth Brain — Code Quality Tasklist

Tracked fixes and improvements from the Feb 27, 2026 code review.

---

## ✅ Completed

- [x] **Atomic project saves** — `state.py` writes to `.tmp` then `os.replace()`
- [x] **Narrow exception catches** — `model_scanner.py` (3) + `ollama.py` (6) now use specific catches with log lines
- [x] **Remove dead imports** — 7 unused legacy session imports removed from `plugin.py`

---

## 🟡 Soon — Before Next Major Feature

- [ ] **Split `plugin.py`** (2171 lines → 3-4 files)
  - [ ] Extract `ui_builder.py` — `_build_step1` through `_build_step4`, CSS, HTML helpers (~500 lines)
  - [ ] Extract `render_engine.py` — `_queue_image_renders`, `_export_videos`, `_find_newest_output`, `_run_render_tasks`, `_build_task`, `_build_cli_state` (~400 lines)
  - [ ] Extract `wiring.py` — `_wire_step1` through `_wire_step4`, `_wire_navigation`, approve/reject handlers (~400 lines)
- [ ] **Fix output attribution** — replace mtime-based `_find_newest_output()` with task-ID or per-shot output dirs
- [ ] **Structured logging** — replace ~40 `print()` calls with `logging` module
- [ ] **Filename collision fix** — `copy_to_project` uses `uuid4` or loop-until-unique instead of `int(time.time())`
- [ ] **Add test suite** — pure-function tests, no GPU needed:
  - [ ] `test_state.py` — save/load roundtrip, copy collision handling
  - [ ] `test_model_scanner.py` — malformed JSON, missing dirs
  - [ ] `test_ollama.py` — `_extract_json` / `_extract_json_array` edge cases
  - [ ] `test_story_templates.py` — empty subject, all-zero weights

---

## 🔵 Later — Architectural

- [ ] **Render queue architecture** — event-driven pipeline with job submission, status polling, persistent queue state
- [ ] **Per-job cancel tokens** — replace single `self._render_cancelled` flag with `CancelToken` per pipeline/run
- [ ] **Config system** — centralize `MAX_SHOTS`, `DEFAULT_MODEL`, `TIMEOUT`, `VRAM_THRESHOLDS` into `config.py` or `config.json`
- [ ] **Installer trust model** — SHA256 verification for Ollama Windows installer, remove `curl | sh` on Linux
