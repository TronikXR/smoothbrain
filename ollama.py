# Smooth Brain — Ollama Integration
# Handles model detection, prompt packing, prompt refinement, and auto-setup.
# Robust version: zero-dependency (urllib), portable binaries, checksum verification.

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
import urllib.error
import hashlib
from typing import Any, Dict, List, Optional

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
_setup_status: str = ""
_setup_lock = threading.Lock()
_setup_done = threading.Event()
_ollama_process: Optional[subprocess.Popen] = None

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_BIN_DIR = os.path.join(_PLUGIN_DIR, "bin")
_OLLAMA_RELEASES_URL = "https://github.com/ollama/ollama/releases/latest/download/"

def setup_status() -> str:
    """Return current auto-setup status string."""
    return _setup_status

def _set_status(s: str):
    global _setup_status
    _setup_status = s
    print(f"[smooth_brain/ollama] setup: {s}")

# ── HTTP Helper (Replaces httpx) ─────────────────────────────────────────────

def _http_request(method: str, path: str, data: Any = None, timeout: float = TIMEOUT) -> Optional[Any]:
    """Make HTTP requests to Ollama API using standard urllib."""
    url = f"{OLLAMA_BASE}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
            json_data = json.dumps(data).encode("utf-8")
        else:
            json_data = None

        with urllib.request.urlopen(req, data=json_data, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except urllib.error.HTTPError as e:
        # Include response text in error message for better diagnostics if it's a 500
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "unavailable"
        print(f"[smooth_brain/ollama] HTTP error: {method} {path} -> {e.code} {e.reason}: {err_body}")
    except urllib.error.URLError:
        # Expected if Ollama is not running
        pass
    except (OSError, ValueError) as e:
        print(f"[smooth_brain/ollama] Request failed: {method} {path} -> {e}")
    return None

def is_online() -> bool:
    """Check if Ollama server is responsive."""
    return _http_request("GET", "/api/tags", timeout=5.0) is not None

# ── Portable Installation ────────────────────────────────────────────────────

def _get_official_checksums() -> Dict[str, str]:
    """Fetch official sha256sum.txt from Ollama GitHub releases."""
    url = f"{_OLLAMA_RELEASES_URL}sha256sum.txt"
    try:
        with urllib.request.urlopen(url, timeout=10.0) as response:
            content = response.read().decode("utf-8")
            checksums = {}
            for line in content.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    checksums[parts[1]] = parts[0]
            return checksums
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[smooth_brain/ollama] Failed to fetch checksums: {e}")
        return {}

def _verify_sha256(filepath: str, expected_sha: str) -> bool:
    """Verify file integrity via SHA256."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_sha
    except OSError:
        return False

def _find_ollama() -> Optional[str]:
    """Find ollama executable, prioritizing local bin/."""
    local_name = "ollama.exe" if sys.platform == "win32" else "ollama"
    local_path = os.path.join(_BIN_DIR, local_name)
    if os.path.exists(local_path):
        return local_path

    found = shutil.which("ollama")
    if found:
        return found

    # Common system paths fallback
    if sys.platform != "win32":
        for p in ["/usr/local/bin/ollama", "/usr/bin/ollama"]:
            if os.path.isfile(p):
                return p
    return None

def _download_binary() -> Optional[str]:
    """Download the correct portable binary for the current platform."""
    os.makedirs(_BIN_DIR, exist_ok=True)

    if sys.platform == "win32":
        # Windows still uses the installer as there's no direct portable zip in releases
        url = "https://ollama.com/download/OllamaSetup.exe"
        filename = "OllamaSetup.exe"
        dest = os.path.join(_BIN_DIR, filename)
    elif sys.platform == "linux":
        import platform
        arch = platform.machine().lower()
        if "arm" in arch or "aarch64" in arch:
            filename = "ollama-linux-arm64"
        else:
            filename = "ollama-linux-amd64"
        url = f"{_OLLAMA_RELEASES_URL}{filename}"
        dest = os.path.join(_BIN_DIR, "ollama")
    else:
        return None

    _set_status(f"downloading {filename}")
    try:
        print(f"[smooth_brain/ollama] Downloading {url}...")
        urllib.request.urlretrieve(url, dest)

        checksums = _get_official_checksums()
        if filename in checksums:
            if _verify_sha256(dest, checksums[filename]):
                print(f"[smooth_brain/ollama] {filename} verified successfully.")
            else:
                print(f"[smooth_brain/ollama] ERROR: {filename} integrity check failed!")
                os.remove(dest)
                return None

        if sys.platform != "win32":
            os.chmod(dest, 0o755)
        return dest
    except Exception as e:
        print(f"[smooth_brain/ollama] Download failed: {e}")
        return None

def ensure_ollama() -> dict:
    """Background setup routine: find, install, start, and pull."""
    with _setup_lock:
        if is_online() and _has_any_model():
            _set_status("ready")
            _setup_done.set()
            return {"online": True, "model_ready": True, "status": "ready"}

        _set_status("checking")
        ollama_path = _find_ollama()
        if not ollama_path:
            binary = _download_binary()
            if not binary:
                _set_status("failed:download")
                _setup_done.set()
                return {"online": False, "model_ready": False, "status": "failed:download"}

            if sys.platform == "win32":
                _set_status("installing")
                try:
                    subprocess.run([binary, "/VERYSILENT", "/NORESTART"], timeout=300)
                    time.sleep(5)
                except Exception as e:
                    print(f"[smooth_brain/ollama] Installer error: {e}")
            ollama_path = _find_ollama()

        if not ollama_path:
            _set_status("failed:notfound")
            _setup_done.set()
            return {"online": False, "model_ready": False, "status": "failed:notfound"}

        if not is_online():
            _set_status("starting")
            _start_server(ollama_path)

        if is_online():
            if not _has_any_model():
                _set_status("pulling")
                _pull_model(ollama_path, DEFAULT_MODEL)
            _set_status("ready")
            _setup_done.set()
            clear_model_cache()
            return {"online": True, "model_ready": True, "status": "ready"}

        _set_status("failed:start")
        _setup_done.set()
        return {"online": False, "model_ready": False, "status": "failed:start"}

def _start_server(path: str):
    global _ollama_process
    try:
        env = os.environ.copy()
        # Ensure it doesn't try to open UI/tray if avoidable
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        if sys.platform == "win32":
            _ollama_process = subprocess.Popen(
                [path, "serve"],
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        else:
            _ollama_process = subprocess.Popen(
                [path, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        for _ in range(30):
            if is_online(): return
            time.sleep(1)
    except Exception as e:
        print(f"[smooth_brain/ollama] Failed to start server: {e}")

def _pull_model(path: str, model: str):
    try:
        print(f"[smooth_brain/ollama] Pulling {model}...")
        subprocess.run([path, "pull", model], timeout=600)
    except Exception as e:
        print(f"[smooth_brain/ollama] Pull error: {e}")

def _has_any_model() -> bool:
    return detect_model() is not None

def detect_model() -> Optional[str]:
    """Scan installed Ollama models and return the best one. None if none found."""
    data = _http_request("GET", "/api/tags")
    if not data or not isinstance(data, dict):
        return None
    try:
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
        # Final fallback: return the first model we found, sorted alphabetically
        models.sort()
        return models[0]
    except (KeyError, AttributeError, TypeError, ValueError):
        return None

def ensure_ollama_background():
    """Trigger setup in a daemon thread."""
    if _setup_status == "ready": return
    if _setup_status and "failed" not in _setup_status: return
    threading.Thread(target=ensure_ollama, daemon=True, name="sb-ollama-setup").start()

# ── Core API Implementation ──────────────────────────────────────────────────

def get_model_name() -> Optional[str]:
    """Get the best available model, caching the result. Returns None if no models."""
    global _cached_model
    if _cached_model:
        return _cached_model
    detected = detect_model()
    if detected:
        _cached_model = detected
        print(f"[smooth_brain/ollama] detected model: {detected}")
        return detected
    # If no model was detected online, we don't return DEFAULT_MODEL
    # because that model might not even be installed.
    return None

def clear_model_cache():
    global _cached_model
    _cached_model = None

def sanitize_prompt(text: str) -> str:
    """Clean up LLM output for cleaner prompts."""
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _extract_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None

def _extract_json_array(text: str) -> Optional[List[Any]]:
    try:
        res = json.loads(text.strip())
        if isinstance(res, list): return res
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            res = json.loads(m.group(1).strip())
            if isinstance(res, list): return res
        except (json.JSONDecodeError, ValueError):
            pass
    m2 = re.search(r"\[[\s\S]*\]", text)
    if m2:
        try:
            res = json.loads(m2.group(0))
            if isinstance(res, list): return res
        except (json.JSONDecodeError, ValueError):
            pass
    return None

def _generate(model: str, system: str, prompt: str, temperature: float = 0.9, max_tokens: int = 4096) -> str:
    """Call Ollama /api/generate and return the response text."""
    res = _http_request("POST", "/api/generate", {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    })
    if res and isinstance(res, dict):
        return res.get("response", "")
    return ""

def describe_character_image(image_path: str) -> Optional[str]:
    """Use Ollama vision to analyze a character image and describe them in detail.

    Returns a concise character description (appearance, clothing, features) or None if
    vision is not available. The description is used to reinforce character consistency
    in shot prompts.
    """
    online = is_online()
    if not online:
        return None

    import base64
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

    model_name = get_model_name()
    if not model_name:
        return None

    system = (
        "You are a visual description specialist. Describe the character in this image in detail. "
        "Focus on: physical appearance, clothing, accessories, hair, posture, and any distinguishing features. "
        "Be specific and concise (under 60 words). Output ONLY the description, no commentary."
    )

    try:
        res = _http_request("POST", "/api/generate", {
            "model": model_name,
            "prompt": "Describe this character in detail for use as a prompt reference.",
            "system": system,
            "images": [img_b64],
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            }
        })
        if res and isinstance(res, dict):
            desc = res.get("response", "").strip()
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

    if not is_online():
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
    if not model_name:
        return raw_prompt

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

    if not is_online():
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
    if not model_name:
        print("[smooth_brain/ollama] pack: no models available — using templates")
        return _fallback_shots(concept, weights, count)

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
    online = is_online()
    if not online:
        return {"online": False, "model_ready": False, "error": "Ollama offline"}
    try:
        data = _http_request("GET", "/api/tags", timeout=5.0)
        if not data or not isinstance(data, dict):
            return {"online": False, "model_ready": False, "error": "Ollama error"}
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
