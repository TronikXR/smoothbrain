from __future__ import annotations
import os
import glob
import time
import traceback
import logging
import gradio as gr
from typing import Any, Dict, List, Optional
from .constants import (
    RESO_MAP, VIDEO_RESOLUTION,
    STATUS_PENDING, STATUS_RENDERING, STATUS_READY, STATUS_APPROVED, STATUS_REJECTED
)
from .state import (
    save_project, copy_to_project, is_ltx_model, ShotState, build_video_params
)
from .ollama import refine_single_prompt
from .model_scanner import (
    get_image_model_overrides, scan_profiles, get_image_ref_overrides
)

log = logging.getLogger("smooth_brain")

class RenderEngineMixin:
    def _build_cli_state(self):
        """Build the minimal state dict that process_tasks_cli expects."""
        return {
            "gen": {
                "queue": [], "in_progress": False,
                "file_list": [], "file_settings_list": [],
                "audio_file_list": [], "audio_file_settings_list": [],
                "selected": 0, "audio_selected": 0,
                "prompt_no": 0, "prompts_max": 0,
                "repeat_no": 0, "total_generation": 1,
                "window_no": 0, "total_windows": 0,
                "progress_status": "", "process_status": "process:main",
            },
            "loras": [],
        }

    def _get_fastest_profile(self, model_id: str) -> dict | None:
        """Auto-select the fastest speed profile for a video model.
        Matches TronikSlate's ranking: i2v preference → latest date → lowest steps.
        Returns the profile dict {name, params} or None.
        """
        import re
        profiles = scan_profiles(model_id)
        if not profiles:
            return None
        is_i2v = "i2v" in model_id.lower() or "ti2v" in model_id.lower()

        def _rank(p):
            name = p["name"].lower()
            # 1) Prefer i2v profiles for i2v models
            i2v_score = 0 if (is_i2v and "i2v" in name) else 1
            # 2) Prefer latest date (e.g. "v2025-10-14")
            date_match = re.search(r"v?(\d{4}[-/]\d{2}[-/]\d{2})", name)
            date_str = date_match.group(1).replace("/", "-") if date_match else "0000-00-00"
            # 3) Prefer lowest steps (e.g. "4 steps")
            steps_match = re.search(r"(\d+)\s*steps?", name)
            steps = int(steps_match.group(1)) if steps_match else 999
            return (i2v_score, date_str, steps)  # sort: 0 < 1 for i2v, reverse for date, ascending for steps

        # Sort: i2v_score ascending, date DESCENDING (negate via reverse string comparison), steps ascending
        ranked = sorted(profiles, key=lambda p: (
            _rank(p)[0],
            "".join(chr(255 - ord(c)) for c in _rank(p)[1]),  # reverse date sort
            _rank(p)[2],
        ))
        return ranked[0] if ranked else None

    def _build_task(self, prompt, model_type, extra_params=None, task_id=None):
        """Build a single render task dict for process_tasks_cli."""
        try:
            defaults = self.get_default_settings(model_type)
        except Exception:
            defaults = {}
        # Start from model defaults only — do NOT inherit primary_settings
        # from main UI (which may have wrong step count, resolution, etc.)
        base = dict(defaults)
        # Force-zero all ref-mode params.
        base["video_prompt_type"] = ""
        base["image_prompt_type"] = ""
        base["audio_prompt_type"] = ""
        base["image_start"] = None
        base["image_end"] = None
        base["image_refs"] = None
        base["video_source"] = None
        base["video_guide"] = None
        base["image_guide"] = None
        base["audio_guide"] = None
        base["audio_guide2"] = None
        base["audio_source"] = None
        base["custom_guide"] = None
        base["video_mask"] = None
        base["image_mask"] = None
        # Apply Smooth Brain speed-lora overrides for image models
        sb_overrides = get_image_model_overrides(model_type)
        if sb_overrides:
            base.update(sb_overrides)
        if extra_params:
            base.update(extra_params)
        base["prompt"] = prompt
        base["model_type"] = model_type
        base["base_model_type"] = model_type
        base.setdefault("mode", "")
        # Debug: log key params for validation troubleshooting
        log.debug("  [_build_task] model=%s vpt=%r ipt=%r apt=%r refs=%r istart=%s",
                  model_type, base.get('video_prompt_type'), base.get('image_prompt_type'),
                  base.get('audio_prompt_type'), base.get('image_refs'),
                  base.get('image_start') is not None)
        return {"id": task_id or int(time.time() * 1000), "params": base, "plugin_data": {}}

    def _find_newest_output(self, extensions=None, since_ts_ns=None, output_type="image"):
        """Scan the wan2gp outputs directory for the newest matching file."""
        cfg = self.server_config if hasattr(self, 'server_config') and self.server_config else {}
        if output_type == "image":
            out_dir = cfg.get("image_save_path", cfg.get("save_path", "outputs"))
        else:
            out_dir = cfg.get("save_path", "outputs")
        if not os.path.isabs(out_dir):
            # Resolve relative to wan2gp app directory
            wgp_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out_dir = os.path.join(wgp_dir, out_dir)
        if not os.path.isdir(out_dir):
            return None
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".webp"] if output_type == "image" else [".mp4"]
        candidates = []
        for ext in extensions:
            candidates.extend(glob.glob(os.path.join(out_dir, f"*{ext}")))
        if not candidates:
            return None
        # Filter by recent timestamp (nanoseconds) if specified
        if since_ts_ns:
            candidates = [f for f in candidates if os.stat(f).st_mtime_ns >= since_ts_ns]
        if not candidates:
            return None
        return max(candidates, key=lambda f: os.stat(f).st_mtime_ns)

    def _run_render_tasks(self, tasks):
        """Run a list of render tasks in-process using process_tasks_cli.
        Returns True on success. Uses the already-loaded model (no subprocess)."""
        cli_state = self._build_cli_state()
        try:
            # Build the queue list from our tasks
            queue = []
            for task in tasks:
                params = task.get("params", {})
                queue.append({
                    "id": task.get("id", int(time.time() * 1000)),
                    "prompt": params.get("prompt", ""),
                    "params": params,
                    "plugin_data": task.get("plugin_data", {}),
                })
            return self.process_tasks_cli(queue, cli_state)
        except Exception as e:
            log.error("Render failed: %s", e, exc_info=True)
            return False

    def _queue_image_renders(self, wan2gp_state, sb_state, resolution, auto_mode):
        """Generator: render shots one at a time and yield progress + images."""
        sb_state = dict(sb_state)
        shots = [dict(s) for s in sb_state.get("shots", [])]
        sb_state["shots"] = shots
        image_model = sb_state.get("image_model", "")
        vibe = sb_state.get("vibe", "cinematic")
        shot_count = sb_state.get("shot_count", 6)
        is_auto = auto_mode == "Auto"
        n_panels = len(self.sb_storyboard_panels)

        def _yield_state(status_html: str, sb_state: dict, changed_shot: int = None, stop_btn_visible: bool = None, preserve_state: bool = False):
            """Build yield tuple for the generator, updating relevant outputs.
            preserve_state=True → emit gr.update() for state slot so user approvals aren't overwritten.
            """
            shots = sb_state.get("shots", [])
            shot_count = sb_state.get("shot_count", 0)
            n_panels = len(self.sb_storyboard_panels)

            # Progress bar
            approved = sum(1 for s in shots[:shot_count]
                           if s.get("status") == STATUS_APPROVED)
            progress = self._progress_bar_html(approved, shot_count)
            all_have_images = shot_count > 0 and all(
                shots[i].get("ref_image_path") for i in range(shot_count) if i < len(shots)
            )
            next_btn = gr.update(interactive=all_have_images)

            # Badges: always update all shots so none get stuck on a stale status
            badges = []
            for i in range(n_panels):
                if i < len(shots):
                    status = shots[i].get("status", STATUS_PENDING)
                    badges.append(gr.update(value=self._shot_badge_html(i, status)))
                else:
                    badges.append(gr.update())

            # Buttons: always update all shots
            buttons = []
            for i in range(n_panels):
                if i < len(shots):
                    status = shots[i].get("status", STATUS_PENDING)
                    show = status in (STATUS_READY, STATUS_APPROVED, STATUS_REJECTED)
                    buttons.extend([gr.update(visible=show), gr.update(visible=show)])
                else:
                    buttons.extend([gr.update(), gr.update()])

            # Images: only push a new value when the changed shot has one (avoid flicker)
            img_updates = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    path = shots[i].get("ref_image_path") if i < len(shots) else None
                    img_updates.append(gr.update(value=path) if path else gr.update())
                else:
                    img_updates.append(gr.update())

            gallery = self._refresh_gallery(sb_state, "images")

            # Stop render button visibility
            stop_btn = gr.update(visible=stop_btn_visible) if stop_btn_visible is not None else gr.update()

            # State: use gr.update() (no-op) on final yield to preserve user approvals
            state_out = gr.update() if preserve_state else sb_state

            return [
                status_html,
                state_out,
                progress, next_btn, gr.update(),
                *badges, *buttons, *img_updates,
                gallery,
                stop_btn,
            ]

        if not shots:
            yield _yield_state("<span style='color:red'>No shots to render</span>", sb_state, stop_btn_visible=False)
            return
        if not image_model:
            yield _yield_state("<span style='color:orange'>⚠️ No image model. Set one in Step 1.</span>", sb_state)
            return

        # Identify shots to render
        res_str = RESO_MAP.get(vibe, {}).get(resolution, "832x480")
        to_render = []
        for i, s in enumerate(shots[:shot_count]):
            status = s.get("status", STATUS_PENDING)
            if status in (STATUS_PENDING, STATUS_REJECTED):
                raw_prompt = s.get("image_prompt") or s.get("beat") or ""
                if raw_prompt:
                    to_render.append(i)

        if not to_render:
            yield _yield_state("<span style='color:orange'>No pending shots to generate.</span>", sb_state)
            return

        # Get character reference image
        char_images = sb_state.get("character_images", [])
        char_ref = next((c for c in char_images if c and os.path.exists(c)), None)

        sb_state["resolution"] = resolution
        rendered = 0
        self._render_cancelled = False

        # Yield initial progress — show stop button
        yield _yield_state(
            f"<span style='color:var(--primary-400)'>⏳ Rendering 0/{len(to_render)} shots...</span>",
            sb_state, stop_btn_visible=True,
        )

        for task_idx, shot_i in enumerate(to_render):
            # Check for cancellation
            if self._render_cancelled:
                yield _yield_state(
                    f"<span style='color:orange'>🛑 Render stopped. {rendered}/{len(to_render)} shots completed.</span>",
                    sb_state, stop_btn_visible=False,
                )
                return
            s = shots[shot_i]
            try:
                raw_prompt = s.get("image_prompt") or s.get("beat") or ""
                # Prepend character description from vision scan for consistency
                char_desc = sb_state.get("character_vision_description", "")
                if char_desc:
                    raw_prompt = f"{char_desc}. {raw_prompt}"
                prompt = refine_single_prompt(raw_prompt, image_model, purpose="image")
                log.info("Shot %d image → %s", shot_i+1, prompt[:120])

                extra = {
                    "resolution": res_str,
                    "image_mode": 1,
                    "seed": s.get("seed", -1),
                }
                if char_ref:
                    model_lower = image_model.lower()
                    is_wan_family = any(model_lower.startswith(p) for p in
                        ("wan", "i2v", "ti2v", "t2v", "ltx", "hunyuan", "hy_", "k5_"))
                    if is_wan_family:
                        extra["image_start"] = char_ref
                        extra["image_prompt_type"] = "S"
                    else:
                        extra["image_refs"] = [char_ref]
                        ref_overrides = get_image_ref_overrides(image_model)
                        extra.update(ref_overrides)

                task = self._build_task(prompt, image_model, extra)
                s["status"] = STATUS_RENDERING

                # Yield rendering status
                yield _yield_state(
                    f"<span style='color:var(--primary-400)'>"
                    f"🎨 <b>Rendering shot {shot_i+1}</b> ({task_idx+1}/{len(to_render)})..."
                    f"<br><small>Check terminal for live progress.</small></span>",
                    sb_state, changed_shot=shot_i,
                )

                before_ts_ns = time.time_ns()
                success = self._run_render_tasks([task])

                if success:
                    output_path = self._find_newest_output(since_ts_ns=before_ts_ns, output_type="image")
                    if output_path:
                        project_dir = sb_state.get("project_dir", "")
                        if project_dir:
                            output_path = copy_to_project(output_path, project_dir, "images")
                        s["ref_image_path"] = output_path
                        s["status"] = STATUS_APPROVED if is_auto else STATUS_READY
                        rendered += 1
                        log.info("  Shot %d → %s", shot_i+1, output_path)
                    else:
                        s["status"] = STATUS_PENDING
                else:
                    s["status"] = STATUS_PENDING

            except Exception as e:
                log.error("Shot %d failed: %s", shot_i+1, e, exc_info=True)
                s["status"] = STATUS_PENDING

            # Auto-save after each shot
            save_project(sb_state)

            # Yield completed shot — only update this shot's badge/buttons/image
            yield _yield_state(
                f"<span style='color:var(--primary-400)'>✅ {rendered}/{len(to_render)} shots done...</span>",
                sb_state, changed_shot=shot_i,
            )

        # Final status (no shot changed — just update text)
        if rendered > 0:
            status = f"<span style='color:var(--primary-500)'>✅ {rendered}/{len(to_render)} shot image(s) generated!</span>"
        else:
            status = "<span style='color:red'>❌ No images generated. Check terminal for errors.</span>"
        yield _yield_state(status, sb_state, stop_btn_visible=False, preserve_state=True)

    def _export_videos(self, wan2gp_state, sb_state, shot_duration):
        """Generator: render video shots one at a time with per-card progress."""
        sb_state = dict(sb_state)
        shots = [dict(s) for s in sb_state.get("shots", [])]
        sb_state["shots"] = shots
        video_model = sb_state.get("video_model", "")
        vibe = sb_state.get("vibe", "cinematic")
        shot_count = sb_state.get("shot_count", 6)
        resolution = sb_state.get("resolution", "480p")
        n_panels = len(self.sb_video_panels)

        # Auto-select fastest speed profile for this video model
        profile_params = {}
        try:
            profile = self._get_fastest_profile(video_model)
            if profile:
                profile_params = profile.get("params", {})
                log.info("Auto-selected speed profile: %s", profile['name'])
            else:
                log.info("No speed profiles found for %s, using defaults", video_model)
        except Exception as e:
            log.error("Profile scan failed: %s", e, exc_info=True)

        # Video status constants (reuse image ones)
        V_PENDING = STATUS_PENDING
        V_RENDERING = STATUS_RENDERING
        V_READY = STATUS_READY
        V_APPROVED = STATUS_APPROVED
        V_REJECTED = STATUS_REJECTED

        def _yield_state(status_html, sb_state, changed_shot=None, stop_btn_visible=None):
            """Build output — only update specific shot's card."""
            approved = sum(1 for s in shots[:shot_count]
                           if s.get("video_status") == V_APPROVED)
            progress = self._progress_bar_html(approved, shot_count)

            badges = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    vs = shots[i].get("video_status", V_PENDING) if i < len(shots) else V_PENDING
                    badges.append(gr.update(value=self._shot_badge_html(i, vs)))
                else:
                    badges.append(gr.update())

            buttons = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    vs = shots[i].get("video_status", V_PENDING) if i < len(shots) else V_PENDING
                    show = (vs == V_READY)
                    buttons.extend([gr.update(visible=show), gr.update(visible=show)])
                else:
                    buttons.extend([gr.update(), gr.update()])

            vid_updates = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    path = shots[i].get("video_path") if i < len(shots) else None
                    vid_updates.append(gr.update(value=path) if path else gr.update())
                else:
                    vid_updates.append(gr.update())

            prompt_updates = [gr.update()] * n_panels

            # Stop button visibility
            stop_top = gr.update(visible=stop_btn_visible) if stop_btn_visible is not None else gr.update()
            stop_bot = gr.update(visible=stop_btn_visible) if stop_btn_visible is not None else gr.update()

            return [
                status_html,
                sb_state, progress,
                *badges, *buttons, *vid_updates, *prompt_updates,
                stop_top, stop_bot,
            ]

        if not shots:
            yield _yield_state("<span style='color:red'>No shots found.</span>", sb_state)
            return
        if not video_model:
            yield _yield_state("<span style='color:red'>No video model selected.</span>", sb_state)
            return

        # Identify shots to render (pending or rejected)
        to_render = []
        errors = []
        for i, s in enumerate(shots[:shot_count]):
            vs = s.get("video_status", V_PENDING)
            if vs in (V_PENDING, V_REJECTED):
                raw_prompt = s.get("video_prompt") or s.get("beat") or ""
                if raw_prompt:
                    prompt = refine_single_prompt(raw_prompt, video_model, purpose="video")
                    try:
                        shot_obj = ShotState(
                            beat=s.get("beat", ""),
                            image_prompt=s.get("image_prompt", ""),
                            video_prompt=prompt,
                            ref_image_path=s.get("ref_image_path"),
                            seed=s.get("seed", -1),
                        )
                        try:
                            defaults = self.get_default_settings(video_model)
                        except Exception:
                            defaults = {}
                        params = build_video_params(shot_obj, video_model, shot_duration, vibe, defaults)
                        res_str = VIDEO_RESOLUTION.get(vibe, {}).get(resolution, "832x480")
                        params["resolution"] = res_str
                        # Apply speed profile params (accelerator lora, step count, etc.)
                        if profile_params:
                            params.update(profile_params)
                        ref_path = s.get("ref_image_path")
                        if ref_path and os.path.exists(ref_path):
                            params["image_start"] = ref_path
                        to_render.append((i, prompt, params))
                    except Exception as e:
                        errors.append(f"Shot {i+1}: {e}")

        if not to_render:
            err_msg = "; ".join(errors) if errors else "No pending shots"
            yield _yield_state(f"<span style='color:orange'>{err_msg}</span>", sb_state)
            return

        rendered = 0
        self._render_cancelled = False

        # Yield initial
        yield _yield_state(
            f"<span style='color:var(--primary-400)'>⏳ Rendering 0/{len(to_render)} videos...</span>",
            sb_state, stop_btn_visible=True,
        )

        for task_idx, (shot_i, prompt, params) in enumerate(to_render):
            # Check for cancellation
            if self._render_cancelled:
                yield _yield_state(
                    f"<span style='color:orange'>🛑 Render stopped. {rendered}/{len(to_render)} videos completed.</span>",
                    sb_state, stop_btn_visible=False,
                )
                return
            s = shots[shot_i]
            s["video_status"] = V_RENDERING
            # Store the refined prompt for display
            s["video_prompt_used"] = prompt
            log.info("Shot %d video → %s", shot_i+1, prompt[:120])

            yield _yield_state(
                f"<span style='color:var(--primary-400)'>"
                f"🎬 <b>Rendering shot {shot_i+1}</b> ({task_idx+1}/{len(to_render)})..."
                f"<br><small>Check terminal for live progress.</small></span>",
                sb_state, changed_shot=shot_i,
            )

            task = self._build_task(prompt, video_model, params)
            before_ts_ns = time.time_ns()
            try:
                success = self._run_render_tasks([task])
            except Exception as e:
                log.error("Video render failed: %s", e, exc_info=True)
                success = False

            if success:
                output_path = self._find_newest_output(since_ts_ns=before_ts_ns, output_type="video")
                if output_path:
                    # Copy to project folder
                    project_dir = sb_state.get("project_dir", "")
                    if project_dir:
                        output_path = copy_to_project(output_path, project_dir, "videos")
                    s["video_path"] = output_path
                    s["video_status"] = V_APPROVED  # Auto-approve for now
                    rendered += 1
                    log.info("  Shot %d → %s", shot_i+1, output_path)
                else:
                    s["video_status"] = V_PENDING
            else:
                s["video_status"] = V_PENDING

            # Auto-save after each shot
            save_project(sb_state)

            yield _yield_state(
                f"<span style='color:var(--primary-400)'>✅ {rendered}/{len(to_render)} videos done...</span>",
                sb_state, changed_shot=shot_i,
            )

        if rendered > 0:
            status = f"<span style='color:var(--primary-500)'>✅ {rendered}/{len(to_render)} video(s) rendered!</span>"
        else:
            status = "<span style='color:red'>❌ No videos generated. Check terminal for errors.</span>"
        yield _yield_state(status, sb_state, stop_btn_visible=False)
