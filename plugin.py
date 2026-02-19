# Smooth Brain — Main Plugin UI
# A 4-step wizard tab for one-click short film generation inside Wan2GP.
#
# Step 1: Story Setup      → concept + genre + model + shot count
# Step 2: Characters       → reference image uploads
# Step 3: Storyboard       → image gen per shot (queued via Wan2GP)
# Step 4: Video Export     → GPU-aware duration slider + queue video renders

from __future__ import annotations
import copy
import glob
import importlib
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

from .ollama import pack as ollama_pack, get_status as ollama_status, is_online, refine_single_prompt
from .state import (
    SmoothBrainSession, ShotState,
    save_session, load_session, clear_session,
    build_video_params, is_ltx_model, vibe_to_resolution,
    duration_to_frames, session_age_minutes,
    create_project_dir, save_project, load_project,
    list_recent_projects, copy_to_project, scan_project_gallery,
)
from .story_templates import ALL_GENRES
from .model_scanner import (
    scan_video_models, scan_image_models,
    get_best_video_model, get_best_image_model, scan_profiles,
)
from .gpu_utils import (
    get_gpu_info, smart_duration_limits,
    RESOLUTION_TIERS, VRAM_THRESHOLDS,
    get_safe_resolution_tier, resolution_tier_warning,
)

PLUGIN_ID = "SmoothBrain"
PLUGIN_NAME = "🧠 Smooth Brain"

MAX_CHARS = 4
MAX_SHOTS = 12

# Default: action=50, all others=0 (matches TronikSlate)
DEFAULT_WEIGHTS = {g: (50 if g == "action" else 0) for g in ALL_GENRES}

# Resolution map: vibe × tier → WxH (shared for images AND videos)
RESO_MAP: Dict[str, Dict[str, str]] = {
    "cinematic": {"480p": "832x480", "540p": "960x544", "720p": "1280x720", "1080p": "1920x1080"},
    "vertical":  {"480p": "480x832", "540p": "544x960", "720p": "720x1280", "1080p": "1080x1920"},
    "square":    {"480p": "512x512", "540p": "768x768", "720p": "1024x1024", "1080p": "1080x1080"},
}
# Backwards compat alias
VIDEO_RESOLUTION = RESO_MAP

# Shot status constants
STATUS_PENDING   = "pending"
STATUS_RENDERING = "rendering"
STATUS_READY     = "ready"
STATUS_APPROVED  = "approved"
STATUS_REJECTED  = "rejected"


class SmoothBrainPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self._session: Optional[SmoothBrainSession] = None
        # Cached at startup to avoid hitting filesystem on every event
        self._video_models: List[dict] = []
        self._image_models: List[dict] = []
        self._gpu_info: dict = {"name": "Unknown", "vram_mb": 0}

    # ── Plugin registration ──────────────────────────────────────────────────

    def setup_ui(self):
        self.request_global("get_current_model_settings")
        self.request_global("set_model_settings")
        self.request_global("get_default_settings")
        self.request_global("get_model_def")
        self.request_global("server_config")
        # Headless render globals — call generate_video in-process
        self.request_global("process_tasks_cli")
        self.request_global("primary_settings")
        self.request_global("get_gen_info")
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")
        self.request_component("image_start")
        self.request_component("image_prompt_type_radio")

        self.add_tab(
            tab_id=PLUGIN_ID,
            label=PLUGIN_NAME,
            component_constructor=self.create_ui,
        )

    # ── CSS ──────────────────────────────────────────────────────────────────

    CSS = """
    #sb-wizard { display: flex; flex-direction: column; gap: 8px; }
    #sb-step-header { font-size: 1.2em; font-weight: 700; color: var(--primary-500); margin-bottom: 4px; }
    #sb-shot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
    #sb-shot-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px; background: var(--background-fill-secondary); }
    .sb-genre-label { font-size: 11px; color: var(--body-text-color-subdued); }
    #sb-status-bar { font-size: 12px; font-style: italic; color: var(--body-text-color-subdued); }
    #sb-genre-zero-warn { color: orange; font-size: 12px; margin-top: 4px; }
    """

    # ── Main UI constructor ──────────────────────────────────────────────────

    def create_ui(self):
        # Scan models + GPU at build time (fast filesystem ops)
        self._video_models_simple = scan_video_models(simple=True)
        self._video_models_adv = scan_video_models(simple=False)
        self._video_models = self._video_models_simple  # default to Simple
        self._image_models = scan_image_models()
        self._gpu_info = get_gpu_info()

        video_choices = [m["name"] for m in self._video_models]
        video_values = [m["id"] for m in self._video_models]
        best_video = get_best_video_model(self._video_models) if self._video_models else ""
        best_video_name = next((m["name"] for m in self._video_models if m["id"] == best_video), "")

        image_choices = [m["name"] for m in self._image_models]
        image_values = [m["id"] for m in self._image_models]
        best_image = get_best_image_model(self._image_models) if self._image_models else ""
        best_image_name = next((m["name"] for m in self._image_models if m["id"] == best_image), "")

        with gr.Blocks(css=self.CSS) as blocks:
            # ── Top status bar ─────────────────────────────────────────────
            with gr.Row():
                gr.HTML(f"<b>{PLUGIN_NAME}</b> &nbsp;|&nbsp; One-click short film generation", elem_id="sb-step-header")
                self.sb_ollama_badge = gr.HTML(self._get_ollama_badge())

            # ── Shared state ───────────────────────────────────────────────
            self.sb_state = gr.State(self._default_state())

            # ── Step panels ────────────────────────────────────────────────
            with gr.Column(elem_id="sb-wizard") as self.step1_panel:
                self._build_step1(video_choices, video_values, best_video_name,
                                   image_choices, image_values, best_image_name)

            with gr.Column(visible=False) as self.step2_panel:
                self._build_step2()

            with gr.Column(visible=False) as self.step3_panel:
                self._build_step3()

            with gr.Column(visible=False) as self.step4_panel:
                self._build_step4()

            # ── Navigation ─────────────────────────────────────────────────
            with gr.Row():
                self.sb_back_btn = gr.Button("← Back", visible=False, scale=0)
                self.sb_step_label = gr.Markdown("**Step 1 of 4** — Story Setup")

            # ── Wiring ─────────────────────────────────────────────────────
            self._wire_navigation()
            self._wire_step1()
            self._wire_step2()
            self._wire_step3()
            self._wire_step4()

        return blocks

    # ── Step 1: Story Setup ──────────────────────────────────────────────────

    def _build_step1(self, video_choices, video_values, best_video_name,
                      image_choices, image_values, best_image_name):
        gr.Markdown("### 📝 Step 1 — Story Setup")
        gr.Markdown("Describe your concept and choose a model. The AI will generate cinematic shot beats.")

        # ── Recent Projects ───────────────────────────────────────────
        projects = list_recent_projects()
        project_items = [f"📂 {p['concept'][:40]} (Step {p['step']}, {p['age_str']})" for p in projects]
        project_items.append("📁 Import from folder...")
        self._recent_project_data = projects  # stash for the resume handler

        with gr.Group():
            gr.Markdown("**📂 Recent Projects**")
            with gr.Row():
                self.sb_recent_projects = gr.Dropdown(
                    choices=project_items if projects else ["No recent projects"],
                    value=None,
                    label="Resume a project",
                    interactive=bool(projects),
                    scale=3,
                )
                self.sb_resume_btn = gr.Button("▶ Resume", variant="primary", scale=1, interactive=bool(projects))
            self.sb_import_path = gr.Textbox(
                label="Project folder path",
                placeholder="Paste full path to a project folder...",
                visible=False,
            )

        with gr.Row():
            self.sb_concept = gr.Textbox(
                label="Concept / Logline",
                placeholder="A lone astronaut discovers a hidden message on a distant moon...",
                lines=2,
            )

        with gr.Row():
            self.sb_shot_count = gr.Radio(
                choices=[3, 6, 10],
                value=6,
                label="Number of Shots",
            )
            self.sb_vibe = gr.Radio(
                choices=["cinematic", "vertical", "square"],
                value="cinematic",
                label="Vibe / Aspect Ratio",
            )

        # ── Model pickers ──────────────────────────────────────────────────
        with gr.Row():
            self.sb_model_mode = gr.Radio(
                choices=["Simple", "Advanced"],
                value="Simple",
                label="Model Mode",
                scale=0,
            )
        self.sb_model_mode_info = gr.HTML(
            "<small style='color:var(--body-text-color-subdued)'>Simple: curated I2V models. Advanced: all models + finetunes.</small>"
        )

        with gr.Row():
            self.sb_video_model = gr.Dropdown(
                label="🎥 Video Model",
                choices=list(zip(video_choices, video_values)) if video_choices else [],
                value=next((m["id"] for m in self._video_models if m["name"] == best_video_name), None) if video_choices else None,
                interactive=True,
                info="" if video_choices else "⚠️ No I2V models found in Wan2GP defaults/",
            )
            self.sb_image_model = gr.Dropdown(
                label="🖼️ Image Model (for storyboard)",
                choices=list(zip(image_choices, image_values)) if image_choices else [],
                value=next((m["id"] for m in self._image_models if m["name"] == best_image_name), None) if image_choices else None,
                interactive=True,
                info="" if image_choices else "⚠️ No installed image models found",
            )

        # ── Genre sliders ──────────────────────────────────────────────────
        gr.Markdown("#### 🎭 Genre Mix")
        gr.Markdown("<small>Sliders are weighted proportionally — move one or more to influence the genre blend. All equal = equal mix.</small>")
        self.sb_genre_sliders: Dict[str, gr.Slider] = {}
        self.sb_genre_pcts: Dict[str, gr.HTML] = {}
        for genre in ALL_GENRES:
            with gr.Row():
                self.sb_genre_sliders[genre] = gr.Slider(
                    minimum=0, maximum=100, value=DEFAULT_WEIGHTS[genre], step=10,
                    label=genre.capitalize(), scale=8,
                    elem_classes=["sb-genre-label"],
                )
                with gr.Column(scale=1, min_width=40):
                    self.sb_genre_pcts[genre] = gr.HTML(
                        self._genre_pct_html(DEFAULT_WEIGHTS[genre], sum(DEFAULT_WEIGHTS.values())),
                    )

        self.sb_genre_zero_warn = gr.HTML(
            "" if sum(DEFAULT_WEIGHTS.values()) > 0 else "<span id='sb-genre-zero-warn'>⚠ Move at least one slider</span>"
        )

        # ── Roll button ────────────────────────────────────────────────────
        with gr.Row():
            self.sb_roll_btn = gr.Button("🎲 Roll — Generate Shots", variant="primary")
            self.sb_roll_status = gr.HTML("")

        # ── Shot cards ─────────────────────────────────────────────────────
        gr.Markdown("#### Generated Shots (edit freely)")
        self.sb_shot_groups: List[gr.Group] = []
        self.sb_shot_beats: List[gr.Textbox] = []
        self.sb_shot_labels: List[gr.Textbox] = []
        with gr.Column(elem_id="sb-shot-grid"):
            for i in range(MAX_SHOTS):
                with gr.Group(visible=(i < 6), elem_id=f"sb-shot-card-{i}") as grp:
                    with gr.Row():
                        label_box = gr.Textbox(
                            label=f"Shot {i+1} label",
                            value=f"Shot {i+1}",
                            interactive=True,
                            scale=1,
                            container=False,
                        )
                    beat_box = gr.Textbox(
                        label=f"Shot {i+1}",
                        value="",
                        lines=3,
                        placeholder="Shot beat will appear here after Roll...",
                    )
                    self.sb_shot_groups.append(grp)
                    self.sb_shot_beats.append(beat_box)
                    self.sb_shot_labels.append(label_box)

        with gr.Row():
            self.sb_step1_next = gr.Button("Next: Characters →", variant="primary")

    # ── Step 2: Characters ───────────────────────────────────────────────────

    def _build_step2(self):
        gr.Markdown("### 👤 Step 2 — Characters")
        gr.Markdown("Who's in your story? Upload a reference image or generate one with AI.")

        # ── Upload / Generate tabs ──
        with gr.Tabs():
            with gr.Tab("📷 I Have an Image"):
                gr.Markdown(
                    "<small>Drag in a single image of your character(s). "
                    "If there are multiple characters, put them all in one image.</small>"
                )
                self.sb_char_upload = gr.Image(
                    label="Upload Character Image",
                    type="filepath",
                    interactive=True,
                    height=200,
                )

            with gr.Tab("🎨 Generate One"):
                self.sb_char_description = gr.Textbox(
                    label="Character Description",
                    placeholder="A grizzled astronaut with a scar across his left cheek, wearing a retro-futuristic suit...",
                    lines=3,
                )
                with gr.Row():
                    self.sb_char_gen_btn = gr.Button(
                        "🎨 Generate Character Image", variant="primary", scale=2,
                    )
                    with gr.Column(scale=3):
                        self.sb_char_gen_status = gr.HTML("")

        # ── Character image result (visible from either tab) ──
        self.sb_char_image = gr.Image(
            label="Character Reference",
            type="filepath",
            interactive=False,
            height=250,
        )

        # ── Story reminder ──
        self.sb_story_reminder = gr.HTML("")

        # ── Resolution picker ──
        gr.Markdown("#### 📐 Image Resolution")
        with gr.Row():
            self.sb_char_resolution = gr.Radio(
                choices=RESOLUTION_TIERS,
                value=get_safe_resolution_tier(self._gpu_info.get("vram_mb", 0)),
                label="Resolution Tier",
            )
        self.sb_char_reso_warn = gr.HTML("")

        # ── Asset Pool ──
        self.sb_asset_pool = gr.Gallery(
            label="📁 All Generated Characters",
            columns=6,
            height=120,
            interactive=False,
        )

        # ── Skip / Continue buttons ──
        with gr.Row():
            self.sb_skip_char_btn = gr.Button(
                "⚡ Skip Character — Generate Anyway", variant="secondary",
            )
            self.sb_step2_next = gr.Button("Next: Storyboard →", variant="primary")

    # ── Step 3: Storyboard ───────────────────────────────────────────────────

    def _build_step3(self):
        gr.Markdown("### 🖼 Step 3 — Image Storyboard")
        gr.Markdown(
            "👍 **Approve** shots you like · 👎 **Reject** to re-generate them · All must be approved to continue."
        )

        # ── Controls row: Manual/Auto toggle + Resolution picker ──
        with gr.Row():
            self.sb_auto_mode = gr.Radio(
                choices=["Manual", "Auto"],
                value="Manual",
                label="Approval Mode",
                scale=1,
            )
            self.sb_resolution = gr.Radio(
                choices=RESOLUTION_TIERS,
                value=get_safe_resolution_tier(self._gpu_info.get("vram_mb", 0)),
                label="📐 Resolution",
                scale=1,
            )
        self.sb_resolution_warn = gr.HTML("")

        # ── Progress bar ──
        self.sb_progress_html = gr.HTML(self._progress_bar_html(0, 0))

        # ── Generate button + status ──
        with gr.Row():
            self.sb_gen_images_btn = gr.Button(
                "🖼 Generate Images", variant="primary", scale=2,
            )
            with gr.Column(scale=3):
                self.sb_gen_images_status = gr.HTML("")

        # ── Shot card grid ──
        self.sb_storyboard_panels: List[Dict[str, Any]] = []
        with gr.Column(elem_id="sb-shot-grid"):
            for i in range(MAX_SHOTS):
                with gr.Group(visible=False, elem_id=f"sb-sb-card-{i}") as grp:
                    with gr.Row():
                        # Shot number + status badge
                        shot_badge = gr.HTML(
                            f"<span style='font-weight:700'>Shot {i+1}</span> "
                            f"<span style='opacity:0.5;font-size:11px'>⏳ pending</span>"
                        )
                    beat_md = gr.Markdown("")
                    img = gr.Image(
                        label=f"Shot {i+1} Frame",
                        type="filepath",
                        interactive=True,
                        height=180,
                    )
                    char_assign = gr.Dropdown(
                        label="Character Reference",
                        choices=["None"],
                        value="None",
                        interactive=True,
                    )
                    with gr.Row():
                        approve_btn = gr.Button("👍", size="sm", visible=False)
                        reject_btn = gr.Button("👎", size="sm", visible=False)

                    self.sb_storyboard_panels.append({
                        "group": grp,
                        "badge": shot_badge,
                        "beat_md": beat_md,
                        "img": img,
                        "char_assign": char_assign,
                        "approve_btn": approve_btn,
                        "reject_btn": reject_btn,
                    })

        # ── Project image gallery ──
        self.sb_image_gallery = gr.Gallery(
            label="📁 All Generated Images",
            columns=6,
            height=140,
            interactive=False,
        )

        # ── Bottom generate button (so user doesn't scroll) ──
        with gr.Row():
            self.sb_gen_images_btn_bottom = gr.Button(
                "🖼 Re-generate Rejected", variant="secondary", scale=2,
            )
        with gr.Row():
            self.sb_step3_next = gr.Button(
                "✅ Approve All & Continue →", variant="primary", interactive=False,
            )

    # ── Step 4: Video Export ─────────────────────────────────────────────────

    def _build_step4(self):
        gr.Markdown("### 🎬 Step 4 — Video Export")
        gr.Markdown(
            "👍 **Approve** videos you like · 👎 **Reject** to re-render them · All must be approved to finish."
        )

        # GPU badge
        gpu = self._gpu_info
        gpu_label = f"({gpu['name']} · {round(gpu['vram_mb']/1024)}GB)" if gpu["vram_mb"] > 0 else ""
        gr.HTML(f"<small style='color:var(--body-text-color-subdued)'>GPU: {gpu_label or 'Unknown'}</small>")

        # Simple / Advanced toggle state
        self.sb_advanced_duration = gr.State(False)

        with gr.Row():
            self.sb_duration_mode_btn = gr.Button("🔒 Simple", size="sm", scale=0)
            gr.Markdown("**⏱️ Shot Duration**")

        self.sb_shot_duration = gr.Slider(
            label="Duration (seconds)",
            minimum=2.0,
            maximum=10.0,
            value=5.0,
            step=0.5,
        )
        self.sb_duration_hint = gr.HTML("")
        self.sb_export_summary = gr.HTML("")

        # ── Progress bar ──
        self.sb_vid_progress_html = gr.HTML(self._progress_bar_html(0, 0))

        # ── Generate button + status ──
        with gr.Row():
            self.sb_export_btn = gr.Button("🎬 Export All Videos", variant="primary", scale=2)
            with gr.Column(scale=3):
                self.sb_export_status = gr.HTML("")

        # ── Video shot card grid ──
        self.sb_video_panels: List[Dict[str, Any]] = []
        with gr.Column(elem_id="sb-video-grid"):
            for i in range(MAX_SHOTS):
                with gr.Group(visible=False, elem_id=f"sb-vid-card-{i}") as grp:
                    with gr.Row():
                        vid_badge = gr.HTML(
                            f"<span style='font-weight:700'>Shot {i+1}</span> "
                            f"<span style='opacity:0.5;font-size:11px'>⏳ pending</span>"
                        )
                    vid_prompt_md = gr.Markdown("", elem_id=f"sb-vid-prompt-{i}")
                    vid = gr.Video(
                        label=f"Shot {i+1} Video",
                        height=220,
                        interactive=False,
                    )
                    with gr.Row():
                        vid_approve_btn = gr.Button("👍", size="sm", visible=False)
                        vid_reject_btn = gr.Button("👎", size="sm", visible=False)

                    self.sb_video_panels.append({
                        "group": grp,
                        "badge": vid_badge,
                        "prompt_md": vid_prompt_md,
                        "video": vid,
                        "approve_btn": vid_approve_btn,
                        "reject_btn": vid_reject_btn,
                    })

        # ── Project video gallery ──
        self.sb_video_gallery = gr.Gallery(
            label="📁 All Generated Videos",
            columns=4,
            height=160,
            interactive=False,
        )

        # ── Bottom re-generate button ──
        with gr.Row():
            self.sb_export_btn_bottom = gr.Button(
                "🎬 Re-render Rejected", variant="secondary", scale=2,
            )

        # Remove old progress markdown — replaced by cards
        self.sb_progress_md = gr.Markdown("", visible=False)

        with gr.Row():
            self.sb_new_project_btn = gr.Button("🗑 New Project", variant="stop", scale=0)
            self.sb_reload_btn = gr.Button("🔄 Reload Logic", variant="secondary", scale=0)
            self.sb_reload_status = gr.HTML("")

    # ── Navigation wiring ─────────────────────────────────────────────────────

    def _step_visibility(self, step: int):
        return [
            gr.update(visible=(step == 1)),
            gr.update(visible=(step == 2)),
            gr.update(visible=(step == 3)),
            gr.update(visible=(step == 4)),
            gr.update(visible=(step > 1)),
            f"**Step {step} of 4** — {['', 'Story Setup', 'Characters', 'Storyboard', 'Video Export'][step]}",
        ]

    def _wire_navigation(self):
        step_outputs = [
            self.step1_panel, self.step2_panel,
            self.step3_panel, self.step4_panel,
            self.sb_back_btn, self.sb_step_label,
        ]
        # Step 1→2: create project folder + prefill character description
        self.sb_step1_next.click(
            fn=self._enter_step2,
            inputs=[self.sb_concept, self.sb_state],
            outputs=[*step_outputs, self.sb_char_description, self.sb_state, self.sb_asset_pool],
        )
        # sb_step2_next is wired in _wire_step2 (combines save + navigate)
        self.sb_step3_next.click(
            fn=self._enter_step4,
            inputs=[self.sb_state],
            outputs=[*step_outputs, *self._all_vid_panel_outputs(), self.sb_video_gallery],
        )
        self.sb_back_btn.click(fn=self._go_back, inputs=[self.sb_state], outputs=step_outputs)

    def _go_back(self, state_dict: dict):
        current = state_dict.get("current_step", 2)
        new_step = max(1, current - 1)
        state_dict["current_step"] = new_step
        return self._step_visibility(new_step)

    # ── Step 1 wiring ────────────────────────────────────────────────────────

    def _wire_step1(self):
        genre_inputs = list(self.sb_genre_sliders.values())
        genre_pct_outputs = list(self.sb_genre_pcts.values())

        # Simple/Advanced model mode toggle
        self.sb_model_mode.change(
            fn=self._toggle_model_mode,
            inputs=[self.sb_model_mode],
            outputs=[self.sb_video_model, self.sb_model_mode_info],
        )

        # Roll
        self.sb_roll_btn.click(
            fn=self._do_roll,
            inputs=[
                self.sb_concept,
                self.sb_shot_count,
                self.sb_video_model,
                self.sb_image_model,
                *genre_inputs,
            ],
            outputs=[
                self.sb_roll_status,
                *self.sb_shot_beats,
                *self.sb_shot_labels,
                self.sb_state,
            ],
        )

        # Shot count → update GROUP visibility
        self.sb_shot_count.change(
            fn=self._update_shot_visibility,
            inputs=[self.sb_shot_count],
            outputs=self.sb_shot_groups,
        )

        # Genre slider live pct update
        for genre in ALL_GENRES:
            self.sb_genre_sliders[genre].change(
                fn=self._update_genre_pcts,
                inputs=genre_inputs,
                outputs=[*genre_pct_outputs, self.sb_genre_zero_warn],
            )

        # ── Recent Projects: show/hide import path ──
        self.sb_recent_projects.change(
            fn=self._on_project_dropdown_change,
            inputs=[self.sb_recent_projects],
            outputs=[self.sb_import_path],
        )

        # ── Resume button ──
        step_outputs = [
            self.step1_panel, self.step2_panel,
            self.step3_panel, self.step4_panel,
            self.sb_back_btn, self.sb_step_label,
        ]
        self.sb_resume_btn.click(
            fn=self._resume_project,
            inputs=[self.sb_recent_projects, self.sb_import_path],
            outputs=[
                self.sb_state,
                self.sb_concept,
                self.sb_shot_count,
                self.sb_vibe,
                self.sb_roll_status,
                *step_outputs,
            ],
        )

    def _update_shot_visibility(self, shot_count):
        n = int(shot_count)
        return [gr.update(visible=(i < n)) for i in range(MAX_SHOTS)]

    def _genre_pct_html(self, value: float, total: float) -> str:
        pct = round((value / total) * 100) if total > 0 else 0
        return f"<span style='font-size:11px;color:var(--body-text-color-subdued)'>{pct}%</span>"

    def _update_genre_pcts(self, *slider_values):
        total = sum(slider_values) or 1
        pct_htmls = [self._genre_pct_html(v, total) for v in slider_values]
        zero_warn = (
            "<span style='color:orange;font-size:12px'>⚠ Move at least one slider to generate stories</span>"
            if sum(slider_values) == 0 else ""
        )
        return [*pct_htmls, zero_warn]

    def _toggle_model_mode(self, mode):
        """Switch between Simple and Advanced video model lists."""
        is_simple = mode == "Simple"
        models = self._video_models_simple if is_simple else self._video_models_adv
        self._video_models = models
        choices = [(m["name"], m["id"]) for m in models]
        best = get_best_video_model(models) if models else ""
        hint = (
            "<small style='color:var(--body-text-color-subdued)'>Simple: curated I2V models only.</small>"
            if is_simple else
            f"<small style='color:var(--body-text-color-subdued)'>Advanced: {len(models)} models including finetunes.</small>"
        )
        return gr.update(choices=choices, value=best), hint

    def _enter_step2(self, concept, sb_state):
        """Step 1→2: create project folder if not yet created, save state."""
        sb_state = dict(sb_state)
        if not sb_state.get("project_dir"):
            sb_state["project_dir"] = create_project_dir(concept)
        sb_state["concept"] = concept
        sb_state["current_step"] = 2
        save_project(sb_state)
        char_gallery = self._refresh_gallery(sb_state, "characters")
        return [*self._step_visibility(2), concept, sb_state, char_gallery]

    def _on_project_dropdown_change(self, selection):
        """Show import path textbox when 'Import from folder...' is selected."""
        if selection and "Import from folder" in selection:
            return gr.update(visible=True)
        return gr.update(visible=False)

    def _resume_project(self, dropdown_selection, import_path):
        """Load a project and jump to its saved step."""
        project_dir = None

        # Check if importing from folder
        if dropdown_selection and "Import from folder" in dropdown_selection:
            if import_path and os.path.isdir(import_path):
                project_dir = import_path
            else:
                gr.Warning("Invalid folder path")
                return [gr.update()] * 11  # sb_state + concept + shot_count + vibe + status + 6 step_outputs
        else:
            # Find the matching project from dropdown
            if dropdown_selection and hasattr(self, '_recent_project_data'):
                for i, p in enumerate(self._recent_project_data):
                    label = f"📂 {p['concept'][:40]} (Step {p['step']}, {p['age_str']})"
                    if label == dropdown_selection:
                        project_dir = p["path"]
                        break

        if not project_dir:
            gr.Warning("No project selected")
            return [gr.update()] * 11

        data = load_project(project_dir)
        if not data:
            gr.Warning("Could not load project")
            return [gr.update()] * 11

        step = data.get("current_step", 1)
        step_vis = list(self._step_visibility(step))
        gr.Info(f"📂 Resumed: {data.get('concept', 'Untitled')[:40]} at Step {step}")

        return [
            data,                                    # sb_state
            data.get("concept", ""),                 # sb_concept
            data.get("shot_count", 6),               # sb_shot_count
            data.get("vibe", "cinematic"),            # sb_vibe
            f"<span style='color:var(--primary-500)'>📂 Project loaded</span>",  # roll_status
            *step_vis,                               # step panels + back btn + label
        ]

    def _do_roll(self, concept, shot_count, video_model, image_model, *genre_sliders):
        shot_count = int(shot_count)
        weights = {g: int(v) for g, v in zip(ALL_GENRES, genre_sliders)}

        try:
            shots = ollama_pack(
                concept=concept,
                shot_count=shot_count,
                genre_weights=weights,
                image_model=image_model or "",
                video_model=video_model or "",
            )
        except Exception as e:
            shots = []
            status_html = f"<span style='color:red'>Error: {e}</span>"
            # Pad and return
            while len(shots) < MAX_SHOTS:
                shots.append({"prompt": "", "shot_label": f"Shot {len(shots)+1}", "imagePrompt": "", "videoPrompt": ""})
            beat_updates = [gr.update(value="", visible=(i < shot_count)) for i in range(MAX_SHOTS)]
            label_updates = [gr.update(value=f"Shot {i+1}", visible=(i < shot_count)) for i in range(MAX_SHOTS)]
            return [status_html, *beat_updates, *label_updates, self._default_state()]

        while len(shots) < MAX_SHOTS:
            shots.append({"prompt": "", "shot_label": f"Shot {len(shots)+1}", "imagePrompt": "", "videoPrompt": ""})

        beat_updates = []
        label_updates = []
        for i in range(MAX_SHOTS):
            s = shots[i]
            beat_updates.append(gr.update(value=s.get("prompt", ""), visible=(i < shot_count)))
            label_updates.append(gr.update(value=s.get("shot_label", f"Shot {i+1}"), visible=(i < shot_count)))

        state_dict = self._default_state()
        state_dict.update({
            "concept": concept,
            "shot_count": shot_count,
            "genre_weights": weights,
            "video_model": video_model or "",
            "image_model": image_model or "",
            "shots": [
                {
                    "beat": s.get("prompt", ""),
                    "image_prompt": s.get("imagePrompt", ""),
                    "video_prompt": s.get("videoPrompt", ""),
                    "ref_image_path": None,
                    "seed": -1,
                    "status": STATUS_PENDING,
                }
                for s in shots[:MAX_SHOTS]
            ],
        })

        source = "Ollama" if is_online() else "Template fallback"
        status_html = (
            f"<span style='color:var(--primary-500)'>✅ {shot_count} shots generated "
            f"<span style='opacity:0.6'>({source})</span></span>"
        )
        save_project(state_dict)
        return [status_html, *beat_updates, *label_updates, state_dict]

    # ── Step 2 wiring ────────────────────────────────────────────────────────

    def _wire_step2(self):
        # Resolution warning
        self.sb_char_resolution.change(
            fn=self._update_resolution_warn,
            inputs=[self.sb_char_resolution],
            outputs=[self.sb_char_reso_warn],
        )

        # Upload → sync to result display
        self.sb_char_upload.change(
            fn=lambda img: img,
            inputs=[self.sb_char_upload],
            outputs=[self.sb_char_image],
        )

        # Generate character image (generator for live progress)
        self.sb_char_gen_btn.click(
            fn=self._generate_character,
            inputs=[self.state, self.sb_state, self.sb_char_description, self.sb_char_resolution],
            outputs=[self.sb_char_gen_status, self.sb_char_image, self.sb_asset_pool],
        )

        # Skip button → next step with no character
        step_outputs = [
            self.step1_panel, self.step2_panel,
            self.step3_panel, self.step4_panel,
            self.sb_back_btn, self.sb_step_label,
        ]
        self.sb_skip_char_btn.click(
            fn=lambda: self._step_visibility(3),
            inputs=[],
            outputs=step_outputs,
        )

        # Next button → save character + navigate to step 3
        step_outputs = [
            self.step1_panel, self.step2_panel,
            self.step3_panel, self.step4_panel,
            self.sb_back_btn, self.sb_step_label,
        ]
        self.sb_step2_next.click(
            fn=self._save_characters_and_advance,
            inputs=[self.sb_state, self.sb_char_image],
            outputs=[self.sb_state, *self._storyboard_panel_outputs(), *step_outputs],
        )

    # ── Headless render helpers ────────────────────────────────────────────

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

    def _build_task(self, prompt, model_type, extra_params=None, task_id=None):
        """Build a single render task dict for process_tasks_cli."""
        try:
            defaults = self.get_default_settings(model_type)
        except Exception:
            defaults = {}
        base = self.primary_settings.copy() if hasattr(self, 'primary_settings') and self.primary_settings else {}
        base.update(defaults)
        if extra_params:
            base.update(extra_params)
        base["prompt"] = prompt
        base["model_type"] = model_type
        base["base_model_type"] = model_type
        base.setdefault("mode", "")
        return {"id": task_id or int(time.time() * 1000), "params": base, "plugin_data": {}}

    def _find_newest_output(self, extensions=None, since_ts=None, output_type="image"):
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
        # Filter by recent timestamp if specified
        if since_ts:
            candidates = [f for f in candidates if os.path.getmtime(f) >= since_ts]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

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
            traceback.print_exc()
            return False

    # ── Step 2: Character Generation ─────────────────────────────────────

    def _generate_character(self, wan2gp_state, sb_state, description, resolution):
        """Generate a character image in-process with live progress."""
        image_model = sb_state.get("image_model", "")
        vibe = sb_state.get("vibe", "cinematic")
        if not description.strip():
            yield "<span style='color:orange'>Enter a character description first.</span>", gr.update(), gr.update()
            return
        if not image_model:
            yield "<span style='color:orange'>⚠️ No image model. Set one in Step 1.</span>", gr.update(), gr.update()
            return
        try:
            # Phase 1: Refine prompt
            yield (
                "<span style='color:var(--primary-400)'>"
                "⏳ <b>Refining prompt</b> using model guide...</span>",
                gr.update(),
                gr.update(),
            )
            refined = refine_single_prompt(description.strip(), image_model, purpose="image")
            print(f"[SmoothBrain] Character gen → model={image_model}")
            print(f"  Prompt: {refined[:200]}")

            res_str = RESO_MAP.get(vibe, {}).get(resolution, "832x480")
            extra = {
                "resolution": res_str,
                "image_mode": 1,  # PNG output, not video
                "seed": -1,
            }
            task = self._build_task(refined, image_model, extra)
            before_ts = time.time()

            # Phase 2: Render
            yield (
                "<span style='color:var(--primary-400)'>"
                "🎨 <b>Generating image</b> — this may take a minute or two..."
                "<br><small>Check terminal for live progress.</small></span>",
                gr.update(),
                gr.update(),
            )
            success = self._run_render_tasks([task])

            if success:
                output_path = self._find_newest_output(since_ts=before_ts, output_type="image")
                if output_path:
                    # Copy to project folder
                    project_dir = sb_state.get("project_dir", "")
                    if project_dir:
                        copy_to_project(output_path, project_dir, "characters")
                    char_gallery = self._refresh_gallery(sb_state, "characters")
                    yield (
                        "<span style='color:var(--primary-500)'>✅ Character image generated!</span>",
                        output_path,
                        char_gallery,
                    )
                    return
                yield (
                    "<span style='color:orange'>⚠️ Render finished but output file not found in outputs/.</span>",
                    gr.update(),
                    gr.update(),
                )
                return
            yield (
                "<span style='color:red'>❌ Render failed. Check terminal for details.</span>",
                gr.update(),
                gr.update(),
            )
        except Exception as e:
            traceback.print_exc()
            yield f"<span style='color:red'>Error: {e}</span>", gr.update(), gr.update()

    def _save_characters_and_advance(self, state_dict, char_image):
        """Save character data, prefill concept, then navigate to Step 3."""
        state_dict = dict(state_dict)
        # Single character image (matches TronikSlate's single-image approach)
        state_dict["character_images"] = [char_image] if char_image else []
        state_dict["character_names"] = ["Character"] if char_image else []
        # Prefill character description from Round 1 concept (TronikSlate parity)
        if not state_dict.get("character_description"):
            state_dict["character_description"] = state_dict.get("concept", "")
        shot_count = state_dict.get("shot_count", 6)
        shots = state_dict.get("shots", [])
        char_names = state_dict.get("character_names", [])
        char_choices = ["None"] + [n for n in char_names if n]
        panel_updates = self._make_storyboard_updates(shots, shot_count, char_choices)
        step_updates = list(self._step_visibility(3))
        return [state_dict, *panel_updates, *step_updates]

    # ── Step 3 wiring ────────────────────────────────────────────────────────

    def _wire_step3(self):
        # Resolution picker → VRAM warning
        self.sb_resolution.change(
            fn=self._update_resolution_warn,
            inputs=[self.sb_resolution],
            outputs=[self.sb_resolution_warn],
        )

        # Generate Images → queue renders for all pending/rejected shots
        gen_outputs = [
            self.sb_gen_images_status,
            self.sb_progress_html,
            self.sb_step3_next,
            self.refresh_form_trigger,
            *self._all_badge_outputs(),
            *self._all_button_outputs(),
            *self._all_image_outputs(),
            self.sb_image_gallery,
        ]
        self.sb_gen_images_btn.click(
            fn=self._queue_image_renders,
            inputs=[self.state, self.sb_state, self.sb_resolution, self.sb_auto_mode],
            outputs=gen_outputs,
        )
        # Bottom button — same handler
        self.sb_gen_images_btn_bottom.click(
            fn=self._queue_image_renders,
            inputs=[self.state, self.sb_state, self.sb_resolution, self.sb_auto_mode],
            outputs=gen_outputs,
        )

        # Per-shot approve/reject buttons
        for i, panel in enumerate(self.sb_storyboard_panels):
            panel["approve_btn"].click(
                fn=lambda state, idx=i: self._approve_shot(state, idx),
                inputs=[self.sb_state],
                outputs=[
                    self.sb_state,
                    self.sb_progress_html,
                    self.sb_step3_next,
                    *self._all_badge_outputs(),
                    *self._all_button_outputs(),
                ],
            )
            panel["reject_btn"].click(
                fn=lambda state, idx=i: self._reject_shot(state, idx),
                inputs=[self.sb_state],
                outputs=[
                    self.sb_state,
                    self.sb_progress_html,
                    self.sb_step3_next,
                    *self._all_badge_outputs(),
                    *self._all_button_outputs(),
                ],
            )

    # ── Step 3 helpers ────────────────────────────────────────────────────────

    def _all_badge_outputs(self):
        return [p["badge"] for p in self.sb_storyboard_panels]

    def _all_button_outputs(self):
        out = []
        for p in self.sb_storyboard_panels:
            out.extend([p["approve_btn"], p["reject_btn"]])
        return out

    def _all_image_outputs(self):
        return [p["img"] for p in self.sb_storyboard_panels]

    def _progress_bar_html(self, approved: int, total: int) -> str:
        pct = round((approved / total) * 100) if total > 0 else 0
        return (
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='flex:1;height:8px;background:var(--border-color-primary);border-radius:4px;overflow:hidden'>"
            f"<div style='width:{pct}%;height:100%;background:var(--primary-500);border-radius:4px;"
            f"transition:width 0.3s'></div></div>"
            f"<span style='font-size:12px;white-space:nowrap'>{approved}/{total} approved</span></div>"
        )

    def _shot_badge_html(self, index: int, status: str) -> str:
        icons = {
            STATUS_PENDING:   "⏳",
            STATUS_RENDERING: "🎨",
            STATUS_READY:     "🖼",
            STATUS_APPROVED:  "✅",
            STATUS_REJECTED:  "🔄",
        }
        colors = {
            STATUS_PENDING:   "opacity:0.5",
            STATUS_RENDERING: "color:var(--primary-500)",
            STATUS_READY:     "color:cyan",
            STATUS_APPROVED:  "color:#22c55e",
            STATUS_REJECTED:  "color:#ef4444",
        }
        icon = icons.get(status, "⏳")
        style = colors.get(status, "")
        return (
            f"<span style='font-weight:700'>Shot {index+1}</span> "
            f"<span style='{style};font-size:11px'>{icon} {status}</span>"
        )

    def _build_status_updates(self, sb_state: dict):
        """Build badge + button updates for all MAX_SHOTS panels."""
        shots = sb_state.get("shots", [])
        shot_count = sb_state.get("shot_count", 6)
        approved = 0

        badge_updates = []
        button_updates = []
        for i in range(MAX_SHOTS):
            if i < shot_count and i < len(shots):
                status = shots[i].get("status", STATUS_PENDING)
                if status == STATUS_APPROVED:
                    approved += 1
                badge_updates.append(gr.update(value=self._shot_badge_html(i, status)))
                show_buttons = (status == STATUS_READY)
                button_updates.extend([
                    gr.update(visible=show_buttons),
                    gr.update(visible=show_buttons),
                ])
            else:
                badge_updates.append(gr.update())
                button_updates.extend([gr.update(), gr.update()])

        all_approved = approved == shot_count and shot_count > 0
        progress = self._progress_bar_html(approved, shot_count)
        next_btn = gr.update(interactive=all_approved)

        return progress, next_btn, badge_updates, button_updates

    def _update_resolution_warn(self, tier):
        warn = resolution_tier_warning(tier, self._gpu_info.get("vram_mb", 0))
        if warn:
            return f"<span style='color:orange;font-size:12px'>{warn}</span>"
        return ""

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

        def _yield_state(status_html, sb_state, changed_shot=None):
            """Build output list — only update the shot that just changed."""
            # Progress bar always updates
            approved = sum(1 for s in shots[:shot_count]
                           if s.get("status") == STATUS_APPROVED)
            progress = self._progress_bar_html(approved, shot_count)
            all_approved = approved == shot_count and shot_count > 0
            next_btn = gr.update(interactive=all_approved)

            # Badges: only update the changed shot
            badges = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    status = shots[i].get("status", STATUS_PENDING) if i < len(shots) else STATUS_PENDING
                    badges.append(gr.update(value=self._shot_badge_html(i, status)))
                else:
                    badges.append(gr.update())

            # Buttons: only update the changed shot
            buttons = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    status = shots[i].get("status", STATUS_PENDING) if i < len(shots) else STATUS_PENDING
                    show = (status == STATUS_READY)
                    buttons.extend([gr.update(visible=show), gr.update(visible=show)])
                else:
                    buttons.extend([gr.update(), gr.update()])

            # Images: only update the changed shot
            img_updates = []
            for i in range(n_panels):
                if changed_shot is not None and i == changed_shot:
                    path = shots[i].get("ref_image_path") if i < len(shots) else None
                    img_updates.append(gr.update(value=path) if path else gr.update())
                else:
                    img_updates.append(gr.update())

            gallery = self._refresh_gallery(sb_state, "images")

            return [
                status_html,
                progress, next_btn, gr.update(),
                *badges, *buttons, *img_updates,
                gallery,
            ]

        if not shots:
            yield _yield_state("<span style='color:red'>No shots to render</span>", sb_state)
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

        # Yield initial progress (no specific shot changed)
        yield _yield_state(
            f"<span style='color:var(--primary-400)'>⏳ Rendering 0/{len(to_render)} shots...</span>",
            sb_state,
        )

        for task_idx, shot_i in enumerate(to_render):
            s = shots[shot_i]
            raw_prompt = s.get("image_prompt") or s.get("beat") or ""
            prompt = refine_single_prompt(raw_prompt, image_model, purpose="image")
            print(f"[SmoothBrain] Shot {shot_i+1} image → {prompt[:120]}")

            extra = {
                "resolution": res_str,
                "image_mode": 1,
                "seed": s.get("seed", -1),
            }
            if char_ref:
                extra["image_start"] = char_ref

            task = self._build_task(prompt, image_model, extra)
            s["status"] = STATUS_RENDERING

            # Yield rendering status — update only this shot's badge
            yield _yield_state(
                f"<span style='color:var(--primary-400)'>"
                f"🎨 <b>Rendering shot {shot_i+1}</b> ({task_idx+1}/{len(to_render)})..."
                f"<br><small>Check terminal for live progress.</small></span>",
                sb_state, changed_shot=shot_i,
            )

            before_ts = time.time()
            try:
                success = self._run_render_tasks([task])
            except Exception as e:
                traceback.print_exc()
                success = False

            if success:
                output_path = self._find_newest_output(since_ts=before_ts, output_type="image")
                if output_path:
                    # Copy to project folder
                    project_dir = sb_state.get("project_dir", "")
                    if project_dir:
                        output_path = copy_to_project(output_path, project_dir, "images")
                    s["ref_image_path"] = output_path
                    s["status"] = STATUS_APPROVED if is_auto else STATUS_READY
                    rendered += 1
                    print(f"  Shot {shot_i+1} → {output_path}")
                else:
                    s["status"] = STATUS_PENDING
            else:
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
        yield _yield_state(status, sb_state)

    def _approve_shot(self, sb_state, shot_index):
        sb_state = dict(sb_state)
        shots = list(sb_state.get("shots", []))
        if shot_index < len(shots):
            shots[shot_index] = dict(shots[shot_index])
            shots[shot_index]["status"] = STATUS_APPROVED
        sb_state["shots"] = shots
        save_project(sb_state)
        progress, next_btn, badges, buttons = self._build_status_updates(sb_state)
        return [sb_state, progress, next_btn, *badges, *buttons]

    def _reject_shot(self, sb_state, shot_index):
        import random
        sb_state = dict(sb_state)
        shots = list(sb_state.get("shots", []))
        if shot_index < len(shots):
            shots[shot_index] = dict(shots[shot_index])
            shots[shot_index]["status"] = STATUS_REJECTED
            shots[shot_index]["seed"] = random.randint(0, 999999)
        sb_state["shots"] = shots
        save_project(sb_state)
        progress, next_btn, badges, buttons = self._build_status_updates(sb_state)
        return [sb_state, progress, next_btn, *badges, *buttons]

    def _storyboard_panel_outputs(self):
        out = []
        for p in self.sb_storyboard_panels:
            out.extend([p["group"], p["badge"], p["beat_md"], p["char_assign"]])
        return out

    def _make_storyboard_updates(self, shots, shot_count, char_choices):
        # Default char ref to first real character if available
        default_char = char_choices[1] if len(char_choices) > 1 else "None"
        updates = []
        for i, panel in enumerate(self.sb_storyboard_panels):
            visible = i < shot_count
            beat = shots[i]["beat"] if i < len(shots) else ""
            status = shots[i].get("status", STATUS_PENDING) if i < len(shots) else STATUS_PENDING
            updates.extend([
                gr.update(visible=visible),
                gr.update(value=self._shot_badge_html(i, status)),
                gr.update(value=f"*{beat}*" if beat else ""),
                gr.update(choices=char_choices, value=default_char),
            ])
        return updates

    # ── Step 4 wiring ────────────────────────────────────────────────────────

    def _wire_step4(self):
        # Duration hint updates
        self.sb_shot_duration.change(
            fn=self._update_duration_hint,
            inputs=[self.sb_shot_duration, self.sb_state],
            outputs=[self.sb_duration_hint],
        )

        # Simple / Advanced toggle
        self.sb_duration_mode_btn.click(
            fn=self._toggle_duration_mode,
            inputs=[self.sb_advanced_duration, self.sb_state],
            outputs=[self.sb_advanced_duration, self.sb_duration_mode_btn, self.sb_shot_duration],
        )

        # Export — renders in-process with per-shot progress
        export_outputs = [
            self.sb_export_status,
            self.sb_state,
            self.sb_vid_progress_html,
            *self._all_vid_badge_outputs(),
            *self._all_vid_button_outputs(),
            *self._all_vid_video_outputs(),
            *self._all_vid_prompt_outputs(),
        ]
        self.sb_export_btn.click(
            fn=self._export_videos,
            inputs=[self.state, self.sb_state, self.sb_shot_duration],
            outputs=export_outputs,
        )
        # Bottom button — same handler
        self.sb_export_btn_bottom.click(
            fn=self._export_videos,
            inputs=[self.state, self.sb_state, self.sb_shot_duration],
            outputs=export_outputs,
        )

        # Per-shot video approve/reject buttons
        for i, panel in enumerate(self.sb_video_panels):
            panel["approve_btn"].click(
                fn=lambda state, idx=i: self._approve_video_shot(state, idx),
                inputs=[self.sb_state],
                outputs=[
                    self.sb_state,
                    self.sb_vid_progress_html,
                    *self._all_vid_badge_outputs(),
                    *self._all_vid_button_outputs(),
                ],
            )
            panel["reject_btn"].click(
                fn=lambda state, idx=i: self._reject_video_shot(state, idx),
                inputs=[self.sb_state],
                outputs=[
                    self.sb_state,
                    self.sb_vid_progress_html,
                    *self._all_vid_badge_outputs(),
                    *self._all_vid_button_outputs(),
                ],
            )

        # New project
        self.sb_new_project_btn.click(
            fn=self._new_project,
            inputs=[],
            outputs=[self.sb_state, self.step1_panel, self.step2_panel,
                     self.step3_panel, self.step4_panel,
                     self.sb_back_btn, self.sb_step_label],
        )

        # Dev reload
        self.sb_reload_btn.click(
            fn=self._reload_modules,
            inputs=[],
            outputs=[self.sb_reload_status],
        )

    def _toggle_duration_mode(self, advanced: bool, sb_state: dict):
        """Toggle Simple ↔ Advanced, updating slider max based on GPU + model."""
        new_advanced = not advanced
        video_model = sb_state.get("video_model", "")
        ltx = is_ltx_model(video_model)
        limits = smart_duration_limits(self._gpu_info["vram_mb"], ltx)
        new_max = limits["hard_max"] if new_advanced else limits["recommended"]
        label = "🔓 Advanced" if new_advanced else "🔒 Simple"
        return new_advanced, gr.update(value=label), gr.update(maximum=new_max)

    def _update_duration_hint(self, duration, sb_state):
        video_model = sb_state.get("video_model", "")
        ltx = is_ltx_model(video_model)
        frames = duration_to_frames(duration, fps=24, is_ltx=ltx)
        actual_s = round(frames / 24, 2)
        limits = smart_duration_limits(self._gpu_info["vram_mb"], ltx)
        note = f" → {frames} frames @ 24fps = {actual_s}s" if ltx else f" → {frames} frames"
        gpu_note = (
            f" &nbsp;<span style='opacity:0.5'>({self._gpu_info['name']} · "
            f"rec. ≤{limits['recommended']}s)</span>"
            if self._gpu_info["vram_mb"] > 0 else ""
        )
        return f"<small>{duration}s{note}{gpu_note}</small>"

    # ── Step 4 helpers ────────────────────────────────────────────────────────

    def _refresh_gallery(self, sb_state, subfolder, extensions=None):
        """Return a gr.update for a gallery component with files from project subfolder."""
        project_dir = sb_state.get("project_dir", "") if isinstance(sb_state, dict) else ""
        if not project_dir:
            return gr.update(value=[])
        files = scan_project_gallery(project_dir, subfolder, extensions)
        return gr.update(value=files)

    def _enter_step4(self, sb_state):
        """Navigate to Step 4 and populate video cards."""
        sb_state = dict(sb_state)
        sb_state["current_step"] = 4
        save_project(sb_state)
        step_updates = list(self._step_visibility(4))
        panel_updates = self._make_video_panel_updates(sb_state)
        vid_gallery = self._refresh_gallery(sb_state, "videos", [".mp4"])
        return [*step_updates, *panel_updates, vid_gallery]

    def _all_vid_panel_outputs(self):
        """All outputs needed to populate video cards (group + badge + prompt)."""
        out = []
        for p in self.sb_video_panels:
            out.extend([p["group"], p["badge"], p["prompt_md"]])
        return out

    def _all_vid_badge_outputs(self):
        return [p["badge"] for p in self.sb_video_panels]

    def _all_vid_button_outputs(self):
        out = []
        for p in self.sb_video_panels:
            out.extend([p["approve_btn"], p["reject_btn"]])
        return out

    def _all_vid_video_outputs(self):
        return [p["video"] for p in self.sb_video_panels]

    def _all_vid_prompt_outputs(self):
        return [p["prompt_md"] for p in self.sb_video_panels]

    def _make_video_panel_updates(self, sb_state):
        """Build initial updates for video cards (visibility, prompts)."""
        shots = sb_state.get("shots", [])
        shot_count = sb_state.get("shot_count", 6)
        updates = []
        for i, panel in enumerate(self.sb_video_panels):
            visible = i < shot_count
            if i < len(shots):
                beat = shots[i].get("beat", "")
                vprompt = shots[i].get("video_prompt", "")
                prompt_text = f"*{vprompt[:200]}*" if vprompt else f"*{beat}*" if beat else ""
                vs = shots[i].get("video_status", STATUS_PENDING)
            else:
                prompt_text = ""
                vs = STATUS_PENDING
            updates.extend([
                gr.update(visible=visible),
                gr.update(value=self._shot_badge_html(i, vs)),
                gr.update(value=prompt_text),
            ])
        return updates

    def _approve_video_shot(self, sb_state, shot_index):
        sb_state = dict(sb_state)
        shots = list(sb_state.get("shots", []))
        if shot_index < len(shots):
            shots[shot_index] = dict(shots[shot_index])
            shots[shot_index]["video_status"] = STATUS_APPROVED
        sb_state["shots"] = shots
        save_project(sb_state)
        shot_count = sb_state.get("shot_count", 6)
        approved = sum(1 for s in shots[:shot_count]
                       if s.get("video_status") == STATUS_APPROVED)
        progress = self._progress_bar_html(approved, shot_count)
        badges = []
        for i in range(len(self.sb_video_panels)):
            if i == shot_index:
                vs = shots[i].get("video_status", STATUS_PENDING) if i < len(shots) else STATUS_PENDING
                badges.append(gr.update(value=self._shot_badge_html(i, vs)))
            else:
                badges.append(gr.update())
        buttons = []
        for i in range(len(self.sb_video_panels)):
            if i == shot_index:
                buttons.extend([gr.update(visible=False), gr.update(visible=False)])
            else:
                buttons.extend([gr.update(), gr.update()])
        return [sb_state, progress, *badges, *buttons]

    def _reject_video_shot(self, sb_state, shot_index):
        sb_state = dict(sb_state)
        shots = list(sb_state.get("shots", []))
        if shot_index < len(shots):
            shots[shot_index] = dict(shots[shot_index])
            shots[shot_index]["video_status"] = STATUS_REJECTED
        sb_state["shots"] = shots
        save_project(sb_state)
        shot_count = sb_state.get("shot_count", 6)
        approved = sum(1 for s in shots[:shot_count]
                       if s.get("video_status") == STATUS_APPROVED)
        progress = self._progress_bar_html(approved, shot_count)
        badges = []
        for i in range(len(self.sb_video_panels)):
            if i == shot_index:
                vs = shots[i].get("video_status", STATUS_PENDING) if i < len(shots) else STATUS_PENDING
                badges.append(gr.update(value=self._shot_badge_html(i, vs)))
            else:
                badges.append(gr.update())
        buttons = []
        for i in range(len(self.sb_video_panels)):
            if i == shot_index:
                buttons.extend([gr.update(visible=False), gr.update(visible=False)])
            else:
                buttons.extend([gr.update(), gr.update()])
        return [sb_state, progress, *badges, *buttons]

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

        # Video status constants (reuse image ones)
        V_PENDING = STATUS_PENDING
        V_RENDERING = STATUS_RENDERING
        V_READY = STATUS_READY
        V_APPROVED = STATUS_APPROVED
        V_REJECTED = STATUS_REJECTED

        def _yield_state(status_html, sb_state, changed_shot=None):
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

            prompt_updates = [gr.update()] * n_panels  # Don't touch prompts mid-render

            return [
                status_html,
                sb_state, progress,
                *badges, *buttons, *vid_updates, *prompt_updates,
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

        # Yield initial
        yield _yield_state(
            f"<span style='color:var(--primary-400)'>⏳ Rendering 0/{len(to_render)} videos...</span>",
            sb_state,
        )

        for task_idx, (shot_i, prompt, params) in enumerate(to_render):
            s = shots[shot_i]
            s["video_status"] = V_RENDERING
            # Store the refined prompt for display
            s["video_prompt_used"] = prompt
            print(f"[SmoothBrain] Shot {shot_i+1} video → {prompt[:120]}")

            yield _yield_state(
                f"<span style='color:var(--primary-400)'>"
                f"🎬 <b>Rendering shot {shot_i+1}</b> ({task_idx+1}/{len(to_render)})..."
                f"<br><small>Check terminal for live progress.</small></span>",
                sb_state, changed_shot=shot_i,
            )

            task = self._build_task(prompt, video_model, params)
            before_ts = time.time()
            try:
                success = self._run_render_tasks([task])
            except Exception as e:
                traceback.print_exc()
                success = False

            if success:
                output_path = self._find_newest_output(since_ts=before_ts, output_type="video")
                if output_path:
                    # Copy to project folder
                    project_dir = sb_state.get("project_dir", "")
                    if project_dir:
                        output_path = copy_to_project(output_path, project_dir, "videos")
                    s["video_path"] = output_path
                    s["video_status"] = V_APPROVED  # Auto-approve for now
                    rendered += 1
                    print(f"  Shot {shot_i+1} → {output_path}")
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
        yield _yield_state(status, sb_state)

    def _new_project(self):
        clear_session()
        state = self._default_state()
        return [state, *self._step_visibility(1)]

    def _reload_modules(self):
        """Hot-reload all Python logic modules (not UI layout)."""
        reloaded = []
        errors = []
        # Modules to reload in dependency order
        module_names = [
            "plugins.smooth_brain.prompt_guides",
            "plugins.smooth_brain.story_templates",
            "plugins.smooth_brain.gpu_utils",
            "plugins.smooth_brain.model_scanner",
            "plugins.smooth_brain.state",
            "plugins.smooth_brain.ollama",
        ]
        import sys
        for mod_name in module_names:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded.append(mod_name.split(".")[-1])
                except Exception as e:
                    errors.append(f"{mod_name}: {e}")

        # Re-import updated symbols into this module's namespace
        try:
            from . import ollama as _ollama_mod
            from . import state as _state_mod
            from . import story_templates as _st_mod
            from . import model_scanner as _ms_mod
            from . import gpu_utils as _gpu_mod
            # Update module-level refs used by plugin
            import plugins.smooth_brain.plugin as _self_mod
            _self_mod.ollama_pack = _ollama_mod.pack
            _self_mod.ollama_status = _ollama_mod.get_status
            _self_mod.is_online = _ollama_mod.is_online
            _self_mod.refine_single_prompt = _ollama_mod.refine_single_prompt
        except Exception as e:
            errors.append(f"re-import: {e}")

        ts = time.strftime("%H:%M:%S")
        if errors:
            return f"<span style='color:orange'>⚠️ [{ts}] Reloaded {', '.join(reloaded)} | Errors: {'; '.join(errors)}</span>"
        return f"<span style='color:var(--primary-500)'>🔄 [{ts}] Reloaded: {', '.join(reloaded)}</span>"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _default_state(self) -> dict:
        return {
            "concept": "",
            "shot_count": 6,
            "genre_weights": dict(DEFAULT_WEIGHTS),
            "vibe": "cinematic",
            "video_model": "",
            "image_model": "",
            "profile": "",
            "resolution": "480p",
            "character_names": ["Character 1", "Character 2", "Character 3", "Character 4"],
            "character_images": [None, None, None, None],
            "shots": [],
            "shot_duration": 5.0,
            "current_step": 1,
            "project_dir": "",
        }

    def _get_ollama_badge(self) -> str:
        status = ollama_status()
        if status.get("online") and status.get("model_ready"):
            model = status.get("active_model", "")
            return (
                f"<span style='color:var(--primary-500);font-size:12px'>"
                f"🟢 Ollama: {model}</span>"
            )
        elif status.get("online"):
            return "<span style='color:orange;font-size:12px'>🟡 Ollama: no model</span>"
        else:
            return (
                "<span style='color:gray;font-size:12px'>"
                "⚫ Ollama offline — using story templates</span>"
            )
