# 🧠 Smooth Brain — Wan2GP Plugin

AI-powered one-click short film generation for [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

## What It Does

Adds a **"🧠 Smooth Brain"** tab to Wan2GP that walks you through a 4-step wizard:

| Step | What happens |
|---|---|
| **1. Story Setup** | Enter a concept, set genre sliders, pick your models. Hit **Roll** — AI generates cinematic shot beats. |
| **2. Characters** | Upload up to 4 character reference images (optional). |
| **3. Storyboard** | Preview shots. Click **Generate All Images** to queue image renders via your loaded image model. |
| **4. Export** | Set shot duration, click **Export All Videos** — each shot queues into Wan2GP's render pipeline. |

## AI Integration

The plugin calls **Ollama** (if running at `localhost:11434`) for:
- **Story beat generation** — creates a cinematic shot list from your concept
- **Prompt refinement** — rewrites prompts for your chosen image and video models (LTX-2, Wan, HunyuanVideo)

If Ollama is offline, it silently falls back to built-in story templates (8 genres).

## Installation

In Wan2GP, go to **Plugins → Install Plugin** and enter:
```
https://github.com/hoodtronik/smooth-brain-wan2gp
```

Or clone manually into `app/plugins/smooth_brain/` and restart Wan2GP.

Install the Python dependency:
```bash
uv pip install httpx
```

## Models Supported

Any model available in your Wan2GP installation. Optimized prompting guides included for:
- Wan 2.1 / 2.2 (I2V, T2V)
- LTX-2 (with automatic `Audio:` sentence injection)
- HunyuanVideo

## LTX-2 Notes

- Frame counts are automatically snapped to the nearest `8n+1` value (LTX-2 requirement)
- If Ollama is used, video prompts automatically get an `"Audio: ..."` sentence so LTX-2 generates intentional sound instead of random music

## File Structure

```
smooth_brain/
├── __init__.py          # Package marker
├── plugin.py            # Main Gradio UI (WAN2GPPlugin subclass)
├── ollama.py            # Ollama pack/refine pipeline
├── prompt_guides.py     # Model-specific prompting guides
├── story_templates.py   # Offline fallback story templates (8 genres)
├── state.py             # Session state + render param builder
├── plugin_info.json     # Wan2GP metadata
└── requirements.txt     # httpx
```

## API — Programmatic Access

You can call the Ollama pipeline directly from Python:

```python
from plugins.smooth_brain.ollama import pack

shots = pack(
    concept="A lone astronaut finds a signal on Europa",
    shot_count=6,
    genre_weights={"scifi": 80, "thriller": 40},
    image_model="i2v",
    video_model="ltx2_distilled",
)
# shots: [{ "prompt": "...", "shot_label": "...", "imagePrompt": "...", "videoPrompt": "..." }, ...]
```

```python
from plugins.smooth_brain.prompt_guides import format_guide_for_system_prompt

guide = format_guide_for_system_prompt("ltx2_distilled")
# Returns a formatted system prompt section for LTX-2
```

## License

MIT
