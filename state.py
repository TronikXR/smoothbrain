# Smooth Brain — Session State & Project Persistence
# Holds wizard state + per-project folder management with JSON autosave.

from __future__ import annotations
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple

# Legacy single-file autosave (kept for backwards compat / migration)
AUTOSAVE_PATH = os.path.join(os.path.dirname(__file__), ".smooth_brain_session.json")

# Default base directory for project folders (relative to wan2gp root)
_WGP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROJECTS_BASE = os.path.join(_WGP_ROOT, "outputs", "smooth_brain")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShotState:
    beat: str                              # Original story beat text
    image_prompt: str = ""                 # AI-refined image prompt
    video_prompt: str = ""                 # AI-refined video prompt
    ref_image_path: Optional[str] = None   # Character reference image path
    seed: int = -1                         # -1 = random
    status: str = "pending"                # pending | rendering | ready | approved | rejected
    video_status: str = "pending"          # same values, for video phase


@dataclass
class SmoothBrainSession:
    # Step 1
    concept: str = ""
    shot_count: int = 6
    genre_weights: Dict[str, int] = field(default_factory=lambda: {"action": 50})
    vibe: str = "cinematic"
    video_model: str = ""
    image_model: str = ""
    profile: str = ""

    # Step 2
    character_names: List[str] = field(default_factory=list)
    character_images: List[Optional[str]] = field(default_factory=list)

    # Step 3
    shots: List[ShotState] = field(default_factory=list)

    # Step 4
    shot_duration: float = 5.0
    current_step: int = 1
    saved_at: float = field(default_factory=time.time)

    # Project
    project_dir: str = ""                  # Absolute path to project folder


# ── Vibe → resolution helpers ────────────────────────────────────────────────

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


# ── Project directory management ──────────────────────────────────────────────

def slugify_concept(concept: str, max_len: int = 40) -> str:
    """Turn a concept string into a filesystem-safe slug."""
    slug = concept.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)      # strip non-alnum
    slug = re.sub(r"[\s_]+", "-", slug)             # spaces/underscores → hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")      # collapse hyphens
    slug = slug[:max_len].rstrip("-")
    return slug or "untitled"


def create_project_dir(concept: str, base_dir: str = "") -> str:
    """Create a project folder and return its absolute path.

    Structure:
        <base_dir>/<slug>/
            project.json
            characters/
            images/
            videos/
    """
    if not base_dir:
        base_dir = DEFAULT_PROJECTS_BASE
    slug = slugify_concept(concept)
    project_dir = os.path.join(base_dir, slug)

    # If folder already exists, add a numeric suffix
    if os.path.exists(project_dir):
        i = 2
        while os.path.exists(f"{project_dir}-{i}"):
            i += 1
        project_dir = f"{project_dir}-{i}"

    os.makedirs(os.path.join(project_dir, "characters"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "videos"), exist_ok=True)
    print(f"[smooth_brain] Created project dir: {project_dir}")
    return project_dir


def save_project(sb_state: dict, project_dir: str = "") -> None:
    """Save sb_state dict to project.json inside the project folder."""
    if not project_dir:
        project_dir = sb_state.get("project_dir", "")
    if not project_dir:
        print("[smooth_brain] save_project: no project_dir, skipping")
        return
    try:
        os.makedirs(project_dir, exist_ok=True)
        data = dict(sb_state)
        data["saved_at"] = time.time()
        path = os.path.join(project_dir, "project.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)  # atomic on same filesystem
    except Exception as e:
        print(f"[smooth_brain] save_project failed: {e}")
        # Clean up temp file if it exists
        try:
            tmp = os.path.join(project_dir, "project.json.tmp")
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def load_project(project_dir: str) -> Optional[dict]:
    """Load a project from its folder. Returns sb_state dict or None."""
    path = os.path.join(project_dir, "project.json")
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["project_dir"] = project_dir
        return data
    except Exception as e:
        print(f"[smooth_brain] load_project failed: {e}")
        return None


def list_recent_projects(base_dir: str = "", max_results: int = 10) -> List[Dict[str, Any]]:
    """Scan for all project folders, return sorted by updated_at desc.

    Returns list of dicts: {name, path, concept, step, age_str, saved_at}
    """
    if not base_dir:
        base_dir = DEFAULT_PROJECTS_BASE
    if not os.path.isdir(base_dir):
        return []

    projects = []
    for name in os.listdir(base_dir):
        pdir = os.path.join(base_dir, name)
        pjson = os.path.join(pdir, "project.json")
        if os.path.isdir(pdir) and os.path.exists(pjson):
            try:
                with open(pjson, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved_at = data.get("saved_at", 0)
                concept = data.get("concept", name)
                step = data.get("current_step", 1)
                age_secs = time.time() - saved_at
                if age_secs < 3600:
                    age_str = f"{int(age_secs/60)}m ago"
                elif age_secs < 86400:
                    age_str = f"{int(age_secs/3600)}h ago"
                else:
                    age_str = f"{int(age_secs/86400)}d ago"
                projects.append({
                    "name": name,
                    "path": pdir,
                    "concept": concept[:60],
                    "step": step,
                    "age_str": age_str,
                    "saved_at": saved_at,
                })
            except Exception:
                pass

    projects.sort(key=lambda p: p["saved_at"], reverse=True)
    return projects[:max_results]


def copy_to_project(src_path: str, project_dir: str, subfolder: str) -> str:
    """Copy a file into the project's subfolder. Returns the destination path."""
    if not project_dir or not src_path or not os.path.exists(src_path):
        return src_path or ""
    dest_dir = os.path.join(project_dir, subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    basename = os.path.basename(src_path)
    dest = os.path.join(dest_dir, basename)
    # Avoid overwriting — add timestamp prefix
    if os.path.exists(dest):
        ts = int(time.time())
        name, ext = os.path.splitext(basename)
        dest = os.path.join(dest_dir, f"{name}_{ts}{ext}")
    try:
        shutil.copy2(src_path, dest)
        return dest
    except Exception as e:
        print(f"[smooth_brain] copy_to_project failed: {e}")
        return src_path


def scan_project_gallery(project_dir: str, subfolder: str, extensions: List[str] = None) -> List[str]:
    """Return all files in a project subfolder, newest first."""
    if not project_dir:
        return []
    folder = os.path.join(project_dir, subfolder)
    if not os.path.isdir(folder):
        return []
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".webp", ".mp4"]
    files = []
    for f in os.listdir(folder):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(folder, f))
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


# ── Legacy JSON autosave (kept for migration) ────────────────────────────────

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


# ── Render param builder ─────────────────────────────────────────────────────

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
    if shot.ref_image_path and os.path.exists(shot.ref_image_path):
        params["image_start"] = shot.ref_image_path
        params["image_prompt_type"] = "S"

    return params
