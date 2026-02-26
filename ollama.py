# Smooth Brain — Ollama Integration
# Port of TronikSlate/app/server/routes/ollama.ts
# Handles model detection, prompt packing, prompt refinement, and auto-setup.

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .prompt_guides import format_guide_for_system_prompt
from .story_templates import (
    TEMPLATES, get_weighted_templates, fill_template, ALL_GENRES
)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
TIMEOUT = 120.0  # seconds

PREFERRED_MODELS = [
    "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:1.5b", "qwen2.5:14b",
    "llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
    "gemma2:2b", "gemma2:9b", "gemma3:4b",
    "phi3:3.8b", "mistral:7b",
]

_cached_model: Optional[str] = None

# ── Auto-setup state ─────────────────────────────────────────────────────────
# Status values: "" (idle), "checking", "downloading", "installing",
#                "starting", "pulling", "ready", "failed:<reason>"
_setup_status: str = ""
_setup_lock = threading.Lock()
_setup_done = threading.Event()
_ollama_process: Optional[subprocess.Popen] = None

# Official download URLs
_OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"
_OLLAMA_LINUX_CMD = "curl -fsSL https://ollama.com/install.sh | sh"


def setup_status() -> str:
    """Return current auto-setup status string."""
    return _setup_status


def _set_status(s: str):
    global _setup_status
    _setup_status = s
    print(f"[smooth_brain/ollama] setup: {s}")


def _find_ollama() -> Optional[str]:
    """Find ollama executable on the system."""
    found = shutil.which("ollama")
    if found:
        return found
    if sys.platform == "win32":
        for base in [
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("PROGRAMFILES", ""),
        ]:
            if not base:
                continue
            for sub in [
                os.path.join("Programs", "Ollama", "ollama.exe"),
                os.path.join("Ollama", "ollama.exe"),
            ]:
                p = os.path.join(base, sub)
                if os.path.isfile(p):
                    return p
    else:
        for p in ["/usr/local/bin/ollama", "/usr/bin/ollama"]:
            if os.path.isfile(p):
                return p
    return None


def _download_ollama_windows() -> Optional[str]:
    """Download Ollama installer for Windows. Returns path to .exe or None."""
    tmp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "smooth_brain_ollama")
    os.makedirs(tmp_dir, exist_ok=True)
    installer_path = os.path.join(tmp_dir, "OllamaSetup.exe")
    if os.path.isfile(installer_path) and os.path.getsize(installer_path) > 1_000_000:
        return installer_path
    try:
        print(f"[smooth_brain/ollama] Downloading Ollama from {_OLLAMA_WINDOWS_URL}...")
        urllib.request.urlretrieve(_OLLAMA_WINDOWS_URL, installer_path)
        if os.path.isfile(installer_path) and os.path.getsize(installer_path) > 1_000_000:
            return installer_path
    except Exception as e:
        print(f"[smooth_brain/ollama] Download failed: {e}")
    return None


def _install_ollama_windows(installer_path: str) -> bool:
    """Run Ollama installer silently. Returns True on success."""
    try:
        subprocess.run(
            [installer_path, "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
            timeout=300,
            capture_output=True,
        )
        time.sleep(3)
        return _find_ollama() is not None
    except Exception as e:
        print(f"[smooth_brain/ollama] Install failed: {e}")
        return False


def _install_ollama_linux() -> bool:
    """Install Ollama on Linux via official script."""
    try:
        subprocess.run(
            ["bash", "-c", _OLLAMA_LINUX_CMD],
            timeout=300,
            capture_output=True,
        )
        time.sleep(2)
        return _find_ollama() is not None
    except Exception as e:
        print(f"[smooth_brain/ollama] Linux install failed: {e}")
        return False


def _start_ollama_server(ollama_path: str) -> bool:
    """Start 'ollama serve' in background. Returns True if server comes online."""
    global _ollama_process
    if is_online():
        return True
    try:
        if sys.platform == "win32":
            _ollama_process = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )
        else:
            _ollama_process = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for _ in range(60):
            time.sleep(0.5)
            if is_online():
                return True
        print("[smooth_brain/ollama] Server started but not responding after 30s")
    except Exception as e:
        print(f"[smooth_brain/ollama] Failed to start server: {e}")
    return False


def _pull_model(ollama_path: str, model: str = DEFAULT_MODEL) -> bool:
    """Pull a model using 'ollama pull'. Returns True on success."""
    try:
        print(f"[smooth_brain/ollama] Pulling model {model}...")
        result = subprocess.run(
            [ollama_path, "pull", model],
            timeout=600,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"[smooth_brain/ollama] Model {model} pulled successfully")
            return True
        print(f"[smooth_brain/ollama] Pull failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"[smooth_brain/ollama] Pull timed out for {model}")
    except Exception as e:
        print(f"[smooth_brain/ollama] Pull error: {e}")
    return False


def _has_any_model() -> bool:
    """Check if Ollama has at least one model installed."""
    return detect_model() is not None


def ensure_ollama() -> dict:
    """Ensure Ollama is installed, running, and has a model.
    Designed to be called from a background thread.
    Returns status dict {online, model_ready, status}.
    """
    with _setup_lock:
        # Already online with a model? Done.
        if is_online() and _has_any_model():
            _set_status("ready")
            _setup_done.set()
            return {"online": True, "model_ready": True, "status": "ready"}

        # Step 1: Find or install Ollama
        _set_status("checking")
        ollama_path = _find_ollama()

        if not ollama_path:
            _set_status("downloading")
            if sys.platform == "win32":
                installer = _download_ollama_windows()
                if not installer:
                    _set_status("failed:download")
                    _setup_done.set()
                    return {"online": False, "model_ready": False, "status": "failed:download"}
                _set_status("installing")
                if not _install_ollama_windows(installer):
                    _set_status("failed:install")
                    _setup_done.set()
                    return {"online": False, "model_ready": False, "status": "failed:install"}
            else:
                _set_status("installing")
                if not _install_ollama_linux():
                    _set_status("failed:install")
                    _setup_done.set()
                    return {"online": False, "model_ready": False, "status": "failed:install"}

            ollama_path = _find_ollama()
            if not ollama_path:
                _set_status("failed:notfound")
                _setup_done.set()
                return {"online": False, "model_ready": False, "status": "failed:notfound"}

        # Step 2: Start server if not running
        if not is_online():
            _set_status("starting")
            if not _start_ollama_server(ollama_path):
                _set_status("failed:start")
                _setup_done.set()
                return {"online": False, "model_ready": False, "status": "failed:start"}

        # Step 3: Pull default model if needed
        if not _has_any_model():
            _set_status("pulling")
            if not _pull_model(ollama_path, DEFAULT_MODEL):
                _set_status("failed:pull")
                _setup_done.set()
                return {"online": True, "model_ready": False, "status": "failed:pull"}

        _set_status("ready")
        _setup_done.set()
        clear_model_cache()
        return {"online": True, "model_ready": True, "status": "ready"}


def ensure_ollama_background() -> None:
    """Start Ollama auto-setup in a background thread (non-blocking)."""
    if _setup_done.is_set() and _setup_status == "ready":
        return
    if _setup_status and _setup_status not in ("", "failed:download", "failed:install",
                                                 "failed:start", "failed:pull", "failed:notfound"):
        return  # Already in progress
    t = threading.Thread(target=ensure_ollama, daemon=True, name="ollama-setup")
    t.start()


# ── HTTP client ──────────────────────────────────────────────────────────────

def _client() -> "httpx.Client":
    return httpx.Client(base_url=OLLAMA_BASE, timeout=TIMEOUT)


def is_online() -> bool:
    if not HAS_HTTPX:
        return False
    try:
        with _client() as c:
            r = c.get("/api/tags", timeout=5.0)
            return r.status_code == 200
    except Exception:
        return False


def detect_model() -> Optional[str]:
    """Scan installed Ollama models and return the best one. None if none found."""
    if not HAS_HTTPX:
        return None
    try:
        with _client() as c:
            r = c.get("/api/tags", timeout=5.0)
            if r.status_code != 200:
                return None
            data = r.json()
            models: List[str] = [m["name"] for m in data.get("models", [])]
            if not models:
                return None
            for pref in PREFERRED_MODELS:
                base = pref.split(":")[0]
                match = next(
                    (m for m in models if m == pref or m == f"{pref}:latest" or m.startswith(base)),
                    None,
                )
                if match:
                    return match
            return None
    except Exception:
        return None


def get_model_name() -> str:
    """Get the best available model, caching the result."""
    global _cached_model
    if _cached_model:
        return _cached_model
    detected = detect_model()
    if detected:
        _cached_model = detected
        print(f"[smooth_brain/ollama] detected model: {detected}")
        return detected
    return DEFAULT_MODEL


def clear_model_cache() -> None:
    global _cached_model
    _cached_model = None


# ── JSON extraction helpers ───────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return None


def _extract_json_array(text: str) -> Optional[List[Any]]:
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    m2 = re.search(r"\[[\s\S]*\]", text)
    if m2:
        try:
            result = json.loads(m2.group(0))
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return None


# ── Core generation ───────────────────────────────────────────────────────────

def _generate(model: str, system: str, prompt: str, temperature: float = 0.9, max_tokens: int = 4096) -> str:
    """Call Ollama /api/generate and return the response text."""
    with _client() as c:
        r = c.post("/api/generate", json={
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("response", "")


def describe_character_image(image_path: str) -> Optional[str]:
    """Use Ollama vision to analyze a character image and describe them in detail.

    Returns a concise character description (appearance, clothing, features) or None if
    vision is not available. The description is used to reinforce character consistency
    in shot prompts.
    """
    if not is_online() or not HAS_HTTPX:
        return None

    import base64
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

    model_name = get_model_name()
    system = (
        "You are a visual description specialist. Describe the character in this image in detail. "
        "Focus on: physical appearance, clothing, accessories, hair, posture, and any distinguishing features. "
        "Be specific and concise (under 60 words). Output ONLY the description, no commentary."
    )

    try:
        with _client() as c:
            r = c.post("/api/generate", json={
                "model": model_name,
                "prompt": "Describe this character in detail for use as a prompt reference.",
                "system": system,
                "images": [img_b64],
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 256,
                },
            }, timeout=TIMEOUT)
            r.raise_for_status()
            desc = r.json().get("response", "").strip()
            if desc and len(desc) > 10:
                print(f"  [vision] Character description: {desc[:120]}...")
                return desc
    except Exception as e:
        print(f"  [vision] Character scan failed (model may not support vision): {e}")
    return None


def _fallback_shots(concept: str, genre_weights: Dict[str, int], shot_count: int) -> List[Dict]:
    """Generate shots from templates when Ollama is offline."""
    templates = get_weighted_templates(genre_weights, count=1)
    template = templates[0] if templates else TEMPLATES[0]
    beats = fill_template(template, concept or "the hero", shot_count)
    return [{"prompt": b, "shot_label": f"Shot {i+1}", "imagePrompt": b, "videoPrompt": b}
            for i, b in enumerate(beats)]


def _refine_prompts(
    model_name: str,
    shots: List[Dict],
    image_model: str,
    video_model: str,
) -> Optional[List[Dict]]:
    """Batch-refine all shot prompts for image and video models."""
    image_guide = format_guide_for_system_prompt(image_model)
    video_guide = format_guide_for_system_prompt(video_model)

    if not image_guide and not video_guide:
        return None

    refine_instructions = ""
    if image_guide:
        refine_instructions += f"\n\nIMAGE MODEL GUIDE (for 'imagePrompt' field):\n{image_guide}"
    if video_guide:
        refine_instructions += f"\n\nVIDEO MODEL GUIDE (for 'videoPrompt' field):\n{video_guide}"

    system = (
        f"You are a prompt optimization specialist. Rewrite each prompt to follow the syntax, "
        f"keywords, and rules of each target model.\n{refine_instructions}\n\n"
        f"You MUST respond with ONLY valid JSON — an array of exactly {len(shots)} objects:\n"
        f"[{{\"imagePrompt\": \"...\", \"videoPrompt\": \"...\"}}]\n"
        f"Rules:\n"
        f"- ALWAYS include the main subject or character in every prompt — never omit who or what is in the scene.\n"
        f"- Preserve the original creative intent of each shot.\n"
        f"- Apply model-specific syntax rules from the guides above.\n"
        f"- If no image guide, copy the original prompt as imagePrompt.\n"
        f"- If no video guide, copy the original prompt as videoPrompt.\n"
        f"- Do NOT add new scene elements — only optimize the language.\n"
        f"- Keep each refined prompt under 150 words."
    )
    prompt_list = "\n".join(f'{i+1}. "{s["prompt"]}"' for i, s in enumerate(shots))
    user_prompt = f"Refine these {len(shots)} shot prompts:\n\n{prompt_list}"

    try:
        raw = _generate(model_name, system, user_prompt, temperature=0.4, max_tokens=4096)
        refined = _extract_json_array(raw)
        if refined and len(refined) == len(shots):
            return [
                {
                    **shot,
                    "imagePrompt": refined[i].get("imagePrompt", shot["prompt"]),
                    "videoPrompt": refined[i].get("videoPrompt", shot["prompt"]),
                }
                for i, shot in enumerate(shots)
            ]
        print(f"[smooth_brain/ollama] refinement returned {len(refined) if refined else 0}/{len(shots)} items — skipping")
    except Exception as e:
        print(f"[smooth_brain/ollama] refinement failed: {e}")

    return None


def refine_single_prompt(
    raw_prompt: str,
    model_id: str,
    purpose: str = "image",
) -> str:
    """Apply model-specific prompt guide to a single prompt at render time.

    Args:
        raw_prompt: The original prompt text.
        model_id: The model id (e.g. 'flux_dev', 'wan21_t2v').
        purpose: 'image' or 'video' — affects the system instruction.

    Returns the refined prompt, or raw_prompt unchanged if Ollama is offline
    or no guide exists for this model.
    """
    guide_text = format_guide_for_system_prompt(model_id)
    if not guide_text:
        print(f"  [prompt] No guide for {model_id}, using raw prompt")
        return raw_prompt

    if not is_online() or not HAS_HTTPX:
        print(f"  [prompt] Ollama offline — using raw prompt for {purpose}")
        return raw_prompt

    system = (
        f"You are a prompt optimization specialist. Rewrite the user's prompt to follow "
        f"the syntax, keywords, and rules of the target {purpose} model.\n"
        f"\n{guide_text}\n\n"
        f"Rules:\n"
        f"- ALWAYS include the main subject or character — never omit who or what is in the scene.\n"
        f"- Preserve the original creative intent completely.\n"
        f"- Apply model-specific syntax rules from the guide above.\n"
        f"- Do NOT add new scene elements — only optimize the language.\n"
        f"- Keep the refined prompt under 150 words.\n"
        f"- Respond with ONLY the refined prompt text, nothing else."
    )

    model_name = get_model_name()
    try:
        refined = _generate(
            model_name, system,
            f"Refine this {purpose} prompt:\n\n{raw_prompt}",
            temperature=0.3, max_tokens=512,
        ).strip()
        if refined and len(refined) > 10:
            print(f"  [prompt] Refined ({purpose}, {model_id}):")
            print(f"    Original: {raw_prompt[:120]}...")
            print(f"    Refined:  {refined[:120]}...")
            return refined
        print(f"  [prompt] Refinement too short, using raw prompt")
    except Exception as e:
        print(f"  [prompt] Refinement failed for {model_id}: {e}")

    return raw_prompt


def pack(
    concept: str = "",
    shot_count: int = 6,
    genre_weights: Optional[Dict[str, int]] = None,
    image_model: str = "",
    video_model: str = "",
) -> List[Dict]:
    """
    Main entry point. Returns a list of shot dicts:
      [{ prompt, shot_label, imagePrompt, videoPrompt }, ...]

    Falls back to template-based beats if Ollama is offline.
    """
    count = max(2, min(shot_count, 20))
    weights = genre_weights or {"action": 50}

    if not is_online() or not HAS_HTTPX:
        print("[smooth_brain/ollama] Ollama offline — using template fallback")
        return _fallback_shots(concept, weights, count)

    # ── Step 1: Generate story beat list ────────────────────────────────────
    if concept.strip():
        story_input = f'Subject/theme: "{concept}".\nInvent a compelling short story around this subject, then break it into shots.'
    else:
        story_input = "Invent a completely original, surprising, and visually stunning short story. Be creative and unexpected. Break it into shots."

    genres_active = {g: w for g, w in weights.items() if w > 0}
    if genres_active:
        total = sum(genres_active.values())
        genre_parts = [f"{g} ({round(w/total*100)}%)" for g, w in
                       sorted(genres_active.items(), key=lambda x: -x[1])]
        story_input += f"\nGenre mix (approximate weight): {', '.join(genre_parts)}."
        story_input += " Lean the tone and visual style toward the higher-weighted genres."

    system = (
        f"You are a professional filmmaker's storyboard assistant. Generate exactly {count} diverse, "
        f"cinematic shot prompts that tell the story visually.\n\n"
        f"Each shot should be distinct in: camera angle, action, mood, and pacing.\n"
        f"Include a mix of: establishing shots, close-ups, action shots, emotional beats, and transitions.\n\n"
        f"You MUST respond with ONLY valid JSON — an array of exactly {count} objects:\n"
        f"[{{\"prompt\": \"detailed cinematic scene description\", \"shot_label\": \"short title (3-6 words)\"}}]\n\n"
        f"RULES:\n"
        f"- Each prompt is self-contained and descriptive enough for AI video generation.\n"
        f"- Maintain visual consistency across all shots (same characters, setting, palette).\n"
        f"- Do NOT include dialogue or text overlays.\n"
        f"- Keep each prompt under 100 words.\n"
        f"- Make it visually compelling: lighting, camera movement, atmosphere."
    )

    model_name = get_model_name()
    try:
        raw = _generate(
            model_name,
            system,
            f"Generate a {count}-shot storyboard:\n\n{story_input}",
            temperature=0.9,
            max_tokens=4096,
        )
        shots = _extract_json_array(raw)
        if not shots or len(shots) == 0:
            print("[smooth_brain/ollama] pack: invalid JSON from LLM — using templates")
            return _fallback_shots(concept, weights, count)
    except Exception as e:
        print(f"[smooth_brain/ollama] pack generation failed: {e} — using templates")
        return _fallback_shots(concept, weights, count)

    # ── Step 2: Refine prompts for target models ─────────────────────────────
    if image_model or video_model:
        refined = _refine_prompts(model_name, shots, image_model, video_model)
        if refined:
            return refined

    # Add default imagePrompt / videoPrompt if refinement skipped
    return [
        {
            **s,
            "imagePrompt": s.get("prompt", ""),
            "videoPrompt": s.get("prompt", ""),
        }
        for s in shots
    ]


def get_status() -> Dict:
    """Return Ollama status dict."""
    if not HAS_HTTPX:
        return {"online": False, "model_ready": False, "error": "httpx not installed"}
    try:
        with _client() as c:
            r = c.get("/api/tags", timeout=5.0)
            if r.status_code != 200:
                return {"online": False, "model_ready": False, "error": "Ollama error"}
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            detected = detect_model()
            if detected:
                clear_model_cache()  # refresh
                global _cached_model
                _cached_model = detected
            return {
                "online": True,
                "model_ready": len(models) > 0,
                "active_model": detected or DEFAULT_MODEL,
                "models": models,
            }
    except Exception as e:
        return {"online": False, "model_ready": False, "error": str(e)}
