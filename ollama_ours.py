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
        print(f"[smooth_brain/ollama] HTTP error: {method} {path} -> {e.code} {e.reason}")
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
    """Scan installed models and return the best match."""
    data = _http_request("GET", "/api/tags")
    if not data or not isinstance(data, dict): return None
    models: List[str] = [m["name"] for m in data.get("models", [])]
    if not models: return None
    for pref in PREFERRED_MODELS:
        base = pref.split(":")[0]
        match = next((m for m in models if m == pref or m == f"{pref}:latest" or m.startswith(base)), None)
        if match: return match
    return models[0]

def ensure_ollama_background():
    """Trigger setup in a daemon thread."""
    if _setup_status == "ready": return
    if _setup_status and "failed" not in _setup_status: return
    threading.Thread(target=ensure_ollama, daemon=True, name="sb-ollama-setup").start()

# ── Core API Implementation ──────────────────────────────────────────────────

def get_model_name() -> str:
    global _cached_model
    if _cached_model: return _cached_model
    detected = detect_model()
    _cached_model = detected or DEFAULT_MODEL
    return _cached_model

def clear_model_cache():
    global _cached_model
    _cached_model = None

def sanitize_prompt(text: str) -> str:
    """Clean up LLM output for cleaner prompts."""
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _extract_json_array(text: str) -> Optional[List[Any]]:
    try:
        res = json.loads(text.strip())
        if isinstance(res, list): return res
    except: pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            res = json.loads(m.group(0))
            if isinstance(res, list): return res
        except: pass
    return None

def _generate(model: str, system: str, prompt: str, temperature: float = 0.9, max_tokens: int = 4096) -> str:
    res = _http_request("POST", "/api/generate", {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    })
    if res and isinstance(res, dict):
        return res.get("response", "")
    return ""

def pack(concept: str = "", shot_count: int = 6, genre_weights: Dict[str, int] = None, image_model: str = "", video_model: str = "") -> List[Dict]:
    if not is_online():
        print("[smooth_brain/ollama] Offline — using template fallback.")
        return _fallback_shots(concept, genre_weights or {}, shot_count)

    model = get_model_name()
    system = "You are a storyboard artist. Respond with exactly {count} shots in a JSON array: [{\"prompt\": \"...\", \"shot_label\": \"...\"}]".replace("{count}", str(shot_count))
    raw = _generate(model, system, f"Subject: {concept}")
    shots = _extract_json_array(raw)

    if not shots:
        return _fallback_shots(concept, genre_weights or {}, shot_count)

    # Auto-fill image/video prompts
    for s in shots:
        s["imagePrompt"] = s.get("prompt", "")
        s["videoPrompt"] = s.get("prompt", "")
    return shots[:shot_count]

def _fallback_shots(concept: str, weights: Dict[str, int], count: int) -> List[Dict]:
    templates = get_weighted_templates(weights, count=1)
    template = templates[0] if templates else TEMPLATES[0]
    beats = fill_template(template, concept or "the hero", count)
    return [{"prompt": b, "shot_label": f"Shot {i+1}", "imagePrompt": b, "videoPrompt": b} for i, b in enumerate(beats)]

def refine_single_prompt(prompt: str, model_id: str, purpose: str = "image") -> str:
    if not is_online(): return prompt
    guide = format_guide_for_system_prompt(model_id)
    if not guide: return prompt

    system = f"Refine this {purpose} prompt for {model_id}.\nRules:\n{guide}\nRespond ONLY with the new prompt."
    refined = _generate(get_model_name(), system, prompt, temperature=0.3, max_tokens=512).strip()
    return refined if len(refined) > 5 else prompt

def describe_character_image(image_path: str) -> Optional[str]:
    if not is_online(): return None
    import base64
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        res = _http_request("POST", "/api/generate", {
            "model": get_model_name(),
            "prompt": "Describe this character's appearance in 50 words.",
            "images": [img_b64],
            "stream": False
        })
        if res and isinstance(res, dict):
            return res.get("response", "").strip()
    except Exception: pass
    return None

def get_status() -> Dict:
    online = is_online()
    model = detect_model() if online else None
    return {
        "online": online,
        "model_ready": model is not None,
        "active_model": model or DEFAULT_MODEL,
        "status": _setup_status
    }
