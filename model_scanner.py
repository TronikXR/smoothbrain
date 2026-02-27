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
IMAGE_ARCH_PREFIXES = ["flux", "qwen_image", "pi_flux"]
IMAGE_MODEL_EXCLUDE = ["qwen_image_layered", "flux2_dev", "flux2_dev_nvfp4"]

# ── Image model overrides for Smooth Brain ──────────────────────────────────
# Speed-lora optimized settings per image model. These override wan2gp's
# defaults when rendering storyboard images via Smooth Brain.
IMAGE_MODEL_OVERRIDES = {
    # Qwen Image Edit 20B — Lightning 4-step accelerator
    "qwen_image_edit_20B": {
        "num_inference_steps": 4,
        "guidance_scale": 1,
        "flow_shift": 5,
        "sample_solver": "default",
        "image_mode": 1,
        "lset_name": "qwen\\Lightning Qwen Edit v1.0 - 4 Steps.json",
        "activated_loras": [
            "https://huggingface.co/DeepBeepMeep/Qwen_image/resolve/main/"
            "loras_accelerators/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        ],
        "loras_multipliers": "1|",
    },
    # Qwen Image 20B — same Lightning accelerator
    "qwen_image_20B": {
        "num_inference_steps": 4,
        "guidance_scale": 1,
        "flow_shift": 5,
        "sample_solver": "default",
        "image_mode": 1,
        "lset_name": "qwen\\Lightning Qwen v1.0 - 4 Steps.json",
        "activated_loras": [
            "https://huggingface.co/DeepBeepMeep/Qwen_image/resolve/main/"
            "loras_accelerators/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        ],
        "loras_multipliers": "1|",
    },
    # Qwen Image 2512 20B (newer version) — uses 2511 Lightning lora
    "qwen_image_2512_20B": {
        "num_inference_steps": 4,
        "guidance_scale": 1,
        "flow_shift": 5,
        "sample_solver": "default",
        "image_mode": 1,
        "lset_name": "qwen\\Lightning Qwen Edit 2511 - 4 Steps.json",
        "activated_loras": [
            "https://huggingface.co/DeepBeepMeep/Qwen_image/resolve/main/"
            "loras_accelerators/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        ],
        "loras_multipliers": "1|",
    },
    # Qwen Image Edit Plus 20B (2509) — same v1.0 Lightning lora as base edit
    "qwen_image_edit_plus_20B": {
        "num_inference_steps": 4,
        "guidance_scale": 1,
        "flow_shift": 5,
        "sample_solver": "default",
        "image_mode": 1,
        "lset_name": "qwen\\Lightning Qwen Edit v1.0 - 4 Steps.json",
        "activated_loras": [
            "https://huggingface.co/DeepBeepMeep/Qwen_image/resolve/main/"
            "loras_accelerators/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        ],
        "loras_multipliers": "1|",
    },
    # pi-FLUX.2 Dev 32B — naturally 4-step, no loras needed
    "pi_flux2": {
        "num_inference_steps": 4,
        "guidance_scale": 5,
        "flow_shift": 5,
        "sample_solver": "",
        "image_mode": 1,
        "embedded_guidance_scale": 4,
    },
}

# Reference-mode overrides applied ONLY when image_refs are provided.
# These cause validation failure if set without actual image refs.
IMAGE_REF_OVERRIDES = {
    "qwen_image": {  # prefix match for all Qwen models
        "video_prompt_type": "I",
        "image_prompt_type": "",
        "remove_background_images_ref": 1,
        "image_refs_relative_size": 50,
    },
    "pi_flux": {  # prefix match for pi-FLUX models
        "video_prompt_type": "I",
        "image_prompt_type": "",
        "remove_background_images_ref": 0,
        "image_refs_relative_size": 50,
    },
    "flux": {  # prefix match for Flux/Klein models
        "video_prompt_type": "I",
        "image_prompt_type": "",
        "remove_background_images_ref": 0,
        "image_refs_relative_size": 50,
    },
}


def get_image_model_overrides(model_id: str) -> dict:
    """Return Smooth Brain speed-lora overrides for an image model, or {}.
    Falls back to prefix matching for future qwen_image_* / pi_flux* variants."""
    if model_id in IMAGE_MODEL_OVERRIDES:
        return dict(IMAGE_MODEL_OVERRIDES[model_id])
    # Fallback: any qwen_image model gets the base qwen_image_20B overrides
    if model_id.startswith("qwen_image"):
        return dict(IMAGE_MODEL_OVERRIDES.get("qwen_image_20B", {}))
    # Fallback: any pi_flux model gets the base pi_flux2 overrides
    if model_id.startswith("pi_flux"):
        return dict(IMAGE_MODEL_OVERRIDES.get("pi_flux2", {}))
    return {}


def get_image_ref_overrides(model_id: str) -> dict:
    """Return reference-image-handling overrides for a model (only when refs exist)."""
    for prefix, overrides in IMAGE_REF_OVERRIDES.items():
        if model_id.startswith(prefix):
            return dict(overrides)
    return {}


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


def _resolve_model_urls(model_data: dict, defaults_dir: str) -> list[str]:
    """Resolve model URLs, following string references to base model defs."""
    urls = model_data.get("model", {}).get("URLs", [])
    if isinstance(urls, str):
        # String reference to another model definition (e.g. "i2v")
        ref_path = os.path.join(defaults_dir, urls + ".json")
        if os.path.isfile(ref_path):
            try:
                with open(ref_path, encoding="utf-8") as f:
                    ref_data = json.load(f)
                urls = ref_data.get("model", {}).get("URLs", [])
            except (json.JSONDecodeError, OSError) as e:
                print(f"[model_scanner] Failed to resolve model ref '{urls}': {e}")
                urls = []
        else:
            urls = []
    if not isinstance(urls, list):
        urls = []
    return urls


def _is_installed(urls: list[str], ckpts_dir: str) -> bool:
    """Check if at least one model file from the URLs exists in ckpts/."""
    return any(
        os.path.isfile(os.path.join(ckpts_dir, url.split("/")[-1]))
        for url in urls
        if url
    )


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
                "_data": data,  # keep raw data for URL resolution
            })
        except (json.JSONDecodeError, OSError) as e:
            print(f"[model_scanner] Skipping {fname}: {e}")
            continue
    return models


def scan_video_models(simple: bool = True) -> list[dict]:
    """
    Return I2V-capable video models that are actually installed in Wan2GP.
    simple=True → only well-known default I2V model IDs (matches TronikSlate Simple mode).
    simple=False → all I2V models including finetunes.
    Filters to models whose ckpt files actually exist on disk.
    """
    defaults_dir = os.path.join(WAN2GP_APP, "defaults")
    finetunes_dir = os.path.join(WAN2GP_APP, "finetunes")
    ckpts_dir = os.path.join(WAN2GP_APP, "ckpts")
    all_models = _scan_folder(defaults_dir, "default") + _scan_folder(finetunes_dir, "finetune")
    i2v_models = [m for m in all_models if m["isI2V"]]

    # Filter to installed models only
    installed = []
    for m in i2v_models:
        urls = _resolve_model_urls(m.get("_data", {}), defaults_dir)
        if _is_installed(urls, ckpts_dir):
            # Remove internal _data before returning
            result = {k: v for k, v in m.items() if k != "_data"}
            installed.append(result)

    if simple:
        return [
            m for m in installed
            if m["source"] == "default" and (
                m["id"] in SIMPLE_I2V_IDS or m["architecture"] in SIMPLE_I2V_IDS
            )
        ]
    return installed



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
            except (json.JSONDecodeError, OSError) as e:
                print(f"[model_scanner] Skipping image model {fname}: {e}")
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
