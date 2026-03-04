from .story_templates import ALL_GENRES
from typing import Dict

PLUGIN_ID = "SmoothBrain"
PLUGIN_NAME = "🧠 Smooth Brain"

MAX_CHARS = 4
MAX_SHOTS = 12

# Default: action=50, all others=0 (matches TronikSlate)
DEFAULT_WEIGHTS = {g: (50 if g == "action" else 0) for g in ALL_GENRES}

# Resolution map: vibe × tier → WxH (shared for images AND videos)
RESO_MAP: Dict[str, Dict[str, str]] = {
    "cinematic": {"480p": "832x480", "540p": "960x544", "720p": "1280x720", "1080p": "1920x1080"},
    "vertical":  {"480p": "480x832", "540p": "544x960", "720p": "720x1280", "1080p": "1080x1920"},
    "square":    {"480p": "512x512", "540p": "768x768", "720p": "1024x1024", "1080p": "1080x1080"},
}
# Backwards compat alias
VIDEO_RESOLUTION = RESO_MAP

# Resolution tiers (moved from gpu_utils for easier access)
RESOLUTION_TIERS = ["480p", "540p", "720p", "1080p"]

# Shot status constants
STATUS_PENDING   = "pending"
STATUS_RENDERING = "rendering"
STATUS_READY     = "ready"
STATUS_APPROVED  = "approved"
STATUS_REJECTED  = "rejected"
