# Smooth Brain — Session State
# Holds wizard state across Gradio steps using gr.State + optional JSON autosave.

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

AUTOSAVE_PATH = os.path.join(os.path.dirname(__file__), ".smooth_brain_session.json")


@dataclass
class ShotState:
    beat: str                          # Original story beat text
    image_prompt: str = ""             # AI-refined image prompt
    video_prompt: str = ""             # AI-refined video prompt
    ref_image_path: Optional[str] = None  # Character reference image path
    seed: int = -1                     # -1 = random


@dataclass
class SmoothBrainSession:
    # Step 1
    concept: str = ""
    shot_count: int = 6
    genre_weights: Dict[str, int] = field(default_factory=lambda: {"action": 50})
    vibe: str = "cinematic"            # "cinematic" | "vertical" | "square"
    video_model: str = ""
    image_model: str = ""
    profile: str = ""

    # Step 2
    character_names: List[str] = field(default_factory=list)
    character_images: List[Optional[str]] = field(default_factory=list)

    # Step 3
    shots: List[ShotState] = field(default_factory=list)

    # Step 4
    shot_duration: float = 5.0         # seconds per shot
    current_step: int = 1
    saved_at: float = field(default_factory=time.time)


# ── Vibe → resolution helpers ──────────────────────────────────────────────────

VIBE_RESOLUTION: Dict[str, str] = {
    "cinematic": "832x480",
    "vertical":  "480x832",
    "square":    "624x624",
}

def vibe_to_resolution(vibe: str) -> str:
    return VIBE_RESOLUTION.get(vibe, "832x480")


# ── Frame helpers ─────────────────────────────────────────────────────────────

def snap_to_8n1(frames: int) -> int:
    """Snap frame count to nearest 8n+1 (LTX-2 requirement)."""
    if frames <= 17:
        return 17
    n = round((frames - 1) / 8)
    return max(17, n * 8 + 1)


def duration_to_frames(seconds: float, fps: int = 24, is_ltx: bool = False) -> int:
    raw = max(1, round(seconds * fps))
    return snap_to_8n1(raw) if is_ltx else raw


def is_ltx_model(model_id: str) -> bool:
    return model_id.lower().startswith("ltx")


# ── JSON autosave ─────────────────────────────────────────────────────────────

def save_session(session: SmoothBrainSession) -> None:
    try:
        data = asdict(session)
        data["saved_at"] = time.time()
        with open(AUTOSAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[smooth_brain] autosave failed: {e}")


def load_session() -> Optional[SmoothBrainSession]:
    try:
        if not os.path.exists(AUTOSAVE_PATH):
            return None
        with open(AUTOSAVE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = SmoothBrainSession(**{
            k: v for k, v in data.items()
            if k in SmoothBrainSession.__dataclass_fields__
        })
        # Re-hydrate shots
        session.shots = [ShotState(**s) for s in data.get("shots", [])]
        return session
    except Exception as e:
        print(f"[smooth_brain] session load failed: {e}")
        return None


def clear_session() -> None:
    try:
        if os.path.exists(AUTOSAVE_PATH):
            os.remove(AUTOSAVE_PATH)
    except Exception:
        pass


def session_age_minutes(session: SmoothBrainSession) -> int:
    return max(0, int((time.time() - session.saved_at) / 60))


# ── Render param builder ──────────────────────────────────────────────────────

def build_video_params(
    shot: ShotState,
    model_id: str,
    shot_duration: float,
    vibe: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the params dict to pass to set_model_settings for one shot."""
    fps = 24
    frames = duration_to_frames(shot_duration, fps, is_ltx=is_ltx_model(model_id))
    resolution = vibe_to_resolution(vibe)

    params = {
        **defaults,
        "model_type": model_id,
        "base_model_type": model_id,
        "prompt": shot.video_prompt or shot.beat,
        "video_length": frames,
        "resolution": resolution,
        "seed": shot.seed if shot.seed != -1 else -1,
    }
    # Attach reference image if present
    if shot.ref_image_path and os.path.exists(shot.ref_image_path):
        params["image_start"] = shot.ref_image_path

    return params
