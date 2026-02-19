"""
gpu_utils.py — Smooth Brain Plugin
GPU detection for Smart Duration limits.
Ported from TronikSlate's server/routes/models.ts GET /api/models/gpu.
"""

from __future__ import annotations
import subprocess


def get_gpu_info() -> dict:
    """
    Return {name: str, vram_mb: int} for the first NVIDIA GPU.
    Falls back to {name: "Unknown", vram_mb: 0} if nvidia-smi not available.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        name = parts[0] if parts else "Unknown"
        vram_mb = int(parts[1]) if len(parts) > 1 else 0
        return {"name": name, "vram_mb": vram_mb}
    except Exception:
        return {"name": "Unknown", "vram_mb": 0}


def smart_duration_limits(vram_mb: int, is_ltx: bool) -> dict:
    """
    Return {recommended: int, hard_max: int} based on VRAM and model family.
    Matches TronikSlate's getSmartDuration() in VideoExport.tsx.
    """
    if is_ltx:
        if vram_mb >= 20480: return {"recommended": 20, "hard_max": 20}
        if vram_mb >= 16384: return {"recommended": 16, "hard_max": 20}
        if vram_mb >= 12288: return {"recommended": 12, "hard_max": 20}
        if vram_mb >= 8192:  return {"recommended": 8,  "hard_max": 20}
        return {"recommended": 5, "hard_max": 20}
    # Wan / HunyuanVideo / others
    if vram_mb >= 16384: return {"recommended": 10, "hard_max": 10}
    if vram_mb >= 12288: return {"recommended": 7,  "hard_max": 10}
    if vram_mb >= 8192:  return {"recommended": 5,  "hard_max": 10}
    return {"recommended": 5, "hard_max": 10}


# ── Resolution tier helpers ───────────────────────────────────────────────────

RESOLUTION_TIERS = ["480p", "540p", "720p", "1080p"]

VRAM_THRESHOLDS: dict[str, int] = {
    "480p":  4000,
    "540p":  8000,
    "720p":  12000,
    "1080p": 20000,
}


def get_safe_resolution_tier(vram_mb: int) -> str:
    """Return the highest resolution tier that fits within the available VRAM."""
    best = "480p"
    for tier in RESOLUTION_TIERS:
        if vram_mb >= VRAM_THRESHOLDS[tier]:
            best = tier
    return best


def resolution_tier_warning(tier: str, vram_mb: int) -> str:
    """Return a sassy VRAM warning if the selected tier exceeds safe limits, or ''."""
    needed = VRAM_THRESHOLDS.get(tier, 0)
    safe = get_safe_resolution_tier(vram_mb)
    if vram_mb <= 0:
        return ""
    if RESOLUTION_TIERS.index(tier) > RESOLUTION_TIERS.index(safe):
        deficit = needed - vram_mb
        return (
            f"⚠️ {tier} wants ~{needed}MB VRAM — you have {vram_mb}MB. "
            f"That's {deficit}MB short. Your GPU might cry."
        )
    return ""
