"""
model_scanner.py — Smooth Brain Plugin
Scans Wan2GP's defaults/ and finetunes/ folders to build video and image model lists.
Ported from TronikSlate's server/routes/models.ts.
"""

from __future__ import annotations
import json
import os
from typing import Optional

# Resolve wan2gp app root — plugin lives at app/plugins/smooth_brain/
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
WAN2GP_APP = os.path.abspath(os.path.join(_PLUGIN_DIR, "..", ".."))

# ── Architecture classification ────────────────────────────────────────────────

NON_VIDEO_PREFIXES = [
    "flux", "qwen", "z_image", "chatterbox", "qwen3_tts",
    "heartmula", "pi_flux", "ace_step",
]
I2V_ARCH_PATTERNS = ["i2v", "ti2v"]
# LTX models support I2V natively even without i2v in arch name
I2V_ARCH_FAMILIES = ["ltx2_19B", "ltxv_13B", "ltx2_distilled"]
IMAGE_ARCH_PREFIXES = ["flux", "qwen_image"]
IMAGE_MODEL_EXCLUDE = ["qwen_image_layered"]

# Priority order for auto-selecting the best video model
VIDEO_PRIORITY = [
    "ltx2_distilled",
    "i2v_2_2",
    "ti2v_2_2",
    "i2v",
    "hunyuan_i2v",
    "hunyuan_1_5_i2v",
    "k5_pro_i2v",
]

# Whitelist for Simple mode (match by id or architecture)
SIMPLE_I2V_IDS = [
    "i2v", "i2v_2_2", "ti2v_2_2",
    "hunyuan_i2v", "hunyuan_1_5_i2v",
    "ltx2_distilled", "ltx2_19B",
    "k5_pro_i2v", "k5_lite_i2v",
]


def _is_i2v(arch: str, model_id: str) -> bool:
    arch_l = arch.lower()
    id_l = model_id.lower()
    if any(p in arch_l or p in id_l for p in I2V_ARCH_PATTERNS):
        return True
    if arch in I2V_ARCH_FAMILIES or model_id in I2V_ARCH_FAMILIES:
        return True
    return False


def _scan_folder(folder: str, source: str) -> list[dict]:
    """Scan a defaults/ or finetunes/ folder for video model JSONs."""
    if not os.path.isdir(folder):
        return []
    models = []
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            mb = data.get("model", {})
            arch = mb.get("architecture", "")
            name = mb.get("name", "")
            if not arch or not name:
                continue
            # Skip non-video families
            if any(arch.startswith(p) for p in NON_VIDEO_PREFIXES):
                continue
            model_id = fname[:-5]  # strip .json
            models.append({
                "id": model_id,
                "name": name,
                "architecture": arch,
                "source": source,
                "isI2V": _is_i2v(arch, model_id),
                "description": mb.get("description", ""),
            })
        except Exception:
            continue
    return models


def scan_video_models(simple: bool = True) -> list[dict]:
    """
    Return I2V-capable video models installed in Wan2GP.
    simple=True → only well-known default I2V model IDs (matches TronikSlate Simple mode).
    simple=False → all I2V models including finetunes.
    """
    defaults_dir = os.path.join(WAN2GP_APP, "defaults")
    finetunes_dir = os.path.join(WAN2GP_APP, "finetunes")
    all_models = _scan_folder(defaults_dir, "default") + _scan_folder(finetunes_dir, "finetune")
    i2v_models = [m for m in all_models if m["isI2V"]]
    if simple:
        return [
            m for m in i2v_models
            if m["source"] == "default" and (
                m["id"] in SIMPLE_I2V_IDS or m["architecture"] in SIMPLE_I2V_IDS
            )
        ]
    return i2v_models


def get_best_video_model(models: list[dict]) -> str:
    """Return the highest-priority installed video model ID."""
    ids = {m["id"] for m in models}
    for pid in VIDEO_PRIORITY:
        if pid in ids:
            return pid
    return models[0]["id"] if models else ""


def scan_image_models() -> list[dict]:
    """
    Return installed image-generation models (flux / qwen_image arch).
    Checks ckpts/ to confirm the model file actually exists on disk.
    """
    defaults_dir = os.path.join(WAN2GP_APP, "defaults")
    finetunes_dir = os.path.join(WAN2GP_APP, "finetunes")
    ckpts_dir = os.path.join(WAN2GP_APP, "ckpts")
    installed = []

    for folder, source in [(defaults_dir, "default"), (finetunes_dir, "finetune")]:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            model_id = fname[:-5]
            if model_id in IMAGE_MODEL_EXCLUDE:
                continue
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                mb = data.get("model", {})
                arch = mb.get("architecture", "")
                name = mb.get("name", "")
                if not arch or not name:
                    continue
                if not any(arch.startswith(p) for p in IMAGE_ARCH_PREFIXES):
                    continue
                # Check if at least one model file exists in ckpts/
                urls = mb.get("URLs", [])
                file_exists = any(
                    os.path.isfile(os.path.join(ckpts_dir, url.split("/")[-1]))
                    for url in urls
                    if url
                )
                if file_exists:
                    installed.append({
                        "id": model_id,
                        "name": name,
                        "architecture": arch,
                        "source": source,
                    })
            except Exception:
                continue
    return installed


def get_best_image_model(models: list[dict]) -> str:
    """Priority: Klein → Qwen → first."""
    for m in models:
        if "klein" in m["name"].lower() or "klein" in m["id"].lower():
            return m["id"]
    for m in models:
        if "qwen" in m["name"].lower() or "qwen" in m["id"].lower():
            return m["id"]
    return models[0]["id"] if models else ""


def scan_profiles(model_id: str) -> list[dict]:
    """
    Return accelerator profiles for a model.
    Looks up the model's architecture from defaults/, then reads profiles/<arch>/*.json.
    """
    defaults_dir = os.path.join(WAN2GP_APP, "defaults")
    profiles_root = os.path.join(WAN2GP_APP, "profiles")
    if not os.path.isdir(profiles_root):
        return []

    # Look up architecture
    arch = ""
    json_path = os.path.join(defaults_dir, f"{model_id}.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            arch = data.get("model", {}).get("architecture", "")
        except Exception:
            pass

    if not arch:
        return []

    # Arch → profile dir mapping (from models.ts)
    ARCH_TO_PROFILE: dict[str, list[str]] = {
        "i2v": ["wan_i2v"], "i2v_720p": ["wan_i2v"], "i2v_nvfp4": ["wan_i2v"],
        "i2v_2_2": ["wan_2_2"], "ti2v_2_2": ["wan_2_2"], "ti2v_2_2_fastwan": ["wan_2_2"],
        "t2v": ["wan"], "t2v_sf": ["wan"], "t2v_2_2": ["wan_2_2"],
        "hunyuan_1_5_i2v": ["hunyuan_1_5"], "hunyuan_1_5_t2v": ["hunyuan_1_5"],
        "hunyuan_i2v": ["wan_i2v"],
    }
    profile_dirs = ARCH_TO_PROFILE.get(arch, [])

    # Fallback: match against existing subdirectory names
    if not profile_dirs:
        existing = [
            d for d in os.listdir(profiles_root)
            if os.path.isdir(os.path.join(profiles_root, d))
        ]
        exact = next((d for d in existing if d == arch), None)
        if exact:
            profile_dirs = [exact]
        else:
            prefix = next((d for d in existing if arch.startswith(d) or d.startswith(arch)), None)
            if prefix:
                profile_dirs = [prefix]

    profiles: list[dict] = []
    for pdir in profile_dirs:
        dir_path = os.path.join(profiles_root, pdir)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    params = json.load(f)
                profiles.append({"name": fname[:-5], "params": params})
            except Exception:
                continue

    profiles.sort(key=lambda p: p["name"])
    return profiles
