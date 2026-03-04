from __future__ import annotations
import os
import gradio as gr
from typing import Any, Dict, List, Optional
from .constants import (
    PLUGIN_NAME, ALL_GENRES, DEFAULT_WEIGHTS, MAX_SHOTS,
    RESOLUTION_TIERS, STATUS_PENDING
)
from .state import list_recent_projects
from .model_scanner import (
    scan_video_models, scan_image_models,
    get_best_video_model, get_best_image_model,
)
from .gpu_utils import get_gpu_info, get_safe_resolution_tier

class UIBuilderMixin:
    CSS = """
    #sb-wizard { display: flex; flex-direction: column; gap: 8px; }
    #sb-step-header { font-size: 1.2em; font-weight: 700; color: var(--primary-500); margin-bottom: 4px; }
    #sb-shot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
    #sb-shot-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px; background: var(--background-fill-secondary); }
    .sb-genre-label { font-size: 11px; color: var(--body-text-color-subdued); }
    #sb-status-bar { font-size: 12px; font-style: italic; color: var(--body-text-color-subdued); }
    #sb-genre-zero-warn { color: orange; font-size: 12px; margin-top: 4px; }
    """

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
                    self.sb_char_stop_btn = gr.Button(
                        "🛑 Stop", variant="stop", scale=0, visible=False,
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

    def _build_step3(self):
        gr.Markdown("### 🖼 Step 3 — Image Storyboard")
        gr.Markdown(
            "👍 **Approve** shots you like · 👎 **Reject** to re-generate them · All must be approved to continue."
        )

        # ── LTX skip button (visible only when LTX model selected) ──
        self.sb_skip_storyboard_btn = gr.Button(
            "⚡ Skip Storyboard — Use T2V Mode",
            variant="secondary",
            visible=False,
        )
        self.sb_ltx_skip_info = gr.HTML("", visible=False)

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
                    prompt_box = gr.Textbox(
                        label="Image Prompt",
                        placeholder="Edit the image prompt here...",
                        lines=2,
                        interactive=True,
                        visible=False,
                    )
                    img = gr.Image(
                        label=f"Shot {i+1} Frame",
                        type="filepath",
                        interactive=True,
                        height=400,
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
                        "prompt_box": prompt_box,
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
            self.sb_stop_render_btn = gr.Button(
                "🛑 Stop Render", variant="stop", scale=1, visible=False,
            )
        with gr.Row():
            self.sb_step3_next = gr.Button(
                "✅ Approve All & Continue →", variant="primary", interactive=False,
            )

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
            self.sb_export_btn = gr.Button("🎥 Process Videos", variant="primary", scale=2)
            self.sb_stop_video_btn_top = gr.Button(
                "🛑 Stop", variant="stop", scale=0, visible=False,
            )
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
                "🎥 Re-process Rejected", variant="secondary", scale=2,
            )
            self.sb_stop_video_btn = gr.Button(
                "🛑 Stop Render", variant="stop", scale=1, visible=False,
            )

        # Remove old progress markdown — replaced by cards
        self.sb_progress_md = gr.Markdown("", visible=False)

        with gr.Row():
            self.sb_new_project_btn = gr.Button("🗑 New Project", variant="stop", scale=0)
            self.sb_reload_btn = gr.Button("🔄 Reload Logic", variant="secondary", scale=0)
            self.sb_reload_status = gr.HTML("")
