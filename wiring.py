from __future__ import annotations
import os
import gradio as gr
from typing import Any, Dict, List, Optional
from .constants import (
    ALL_GENRES, MAX_SHOTS,
    STATUS_PENDING, STATUS_READY, STATUS_APPROVED, STATUS_REJECTED
)

class WiringMixin:
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
                self.sb_char_image,
                self.sb_progress_html,
                self.sb_step3_next,
                *self._storyboard_panel_outputs(),
                *self._all_vid_panel_outputs(),
                self.sb_video_gallery,
            ],
        )

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
            outputs=[self.sb_char_gen_status, self.sb_char_stop_btn, self.sb_char_image, self.sb_asset_pool],
        )
        # Stop character render button
        self.sb_char_stop_btn.click(
            fn=self._stop_render,
            inputs=[],
            outputs=[self.sb_char_stop_btn],
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
            outputs=[
                self.sb_state,
                *self._storyboard_panel_outputs(),
                *step_outputs,
                self.sb_skip_storyboard_btn,
                self.sb_ltx_skip_info,
            ],
        )

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
            self.sb_state,
            self.sb_progress_html,
            self.sb_step3_next,
            self.refresh_form_trigger,
            *self._all_badge_outputs(),
            *self._all_button_outputs(),
            *self._all_image_outputs(),
            self.sb_image_gallery,
            self.sb_stop_render_btn,
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
        # Stop render button
        self.sb_stop_render_btn.click(
            fn=self._stop_render,
            inputs=[],
            outputs=[self.sb_stop_render_btn],
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

        # Per-shot prompt_box → save edited prompt to state
        for i, panel in enumerate(self.sb_storyboard_panels):
            panel["prompt_box"].change(
                fn=lambda state, text, idx=i: self._update_shot_prompt(state, idx, text),
                inputs=[self.sb_state, panel["prompt_box"]],
                outputs=[self.sb_state],
            )
        # LTX skip storyboard button
        step_outputs = [
            self.step1_panel, self.step2_panel,
            self.step3_panel, self.step4_panel,
            self.sb_back_btn, self.sb_step_label,
        ]
        self.sb_skip_storyboard_btn.click(
            fn=self._enter_step4,
            inputs=[self.sb_state],
            outputs=[*step_outputs, *self._all_vid_panel_outputs(), self.sb_video_gallery],
        )

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
            self.sb_stop_video_btn_top,
            self.sb_stop_video_btn,
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

        # Stop video render buttons (top + bottom)
        self.sb_stop_video_btn.click(
            fn=self._stop_render,
            inputs=[],
            outputs=[self.sb_stop_video_btn],
        )
        self.sb_stop_video_btn_top.click(
            fn=self._stop_render,
            inputs=[],
            outputs=[self.sb_stop_video_btn_top],
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
