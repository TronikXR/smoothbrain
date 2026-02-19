# Smooth Brain — Model Prompt Guides
# Port of TronikSlate/app/server/data/modelPromptGuides.ts
# Provides model-specific prompting guidance injected into Ollama system prompts.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ModelPromptGuide:
    model_name: str
    syntax_rules: str
    keywords: str
    negative_prompt: str
    pitfalls: str
    examples: str
    audio_guidance: Optional[str] = None


# ── WAN 2.1 / 2.2 ─────────────────────────────────────────────────────────────
WAN_GUIDE = ModelPromptGuide(
    model_name="Wan 2.1 / 2.2",
    syntax_rules=(
        "Write in descriptive prose, not keyword lists. "
        "Lead with camera motion, then subject action, then environment details. "
        "End with lighting/atmosphere. Separate clauses with commas."
    ),
    keywords=(
        "Camera: 'smooth tracking shot', 'slow dolly in', 'aerial wide', 'handheld close-up'\n"
        "Motion: 'fluid', 'dynamic', 'subtle sway', 'crisp motion'\n"
        "Quality: 'cinematic', 'photorealistic', '8K', 'film grain', 'shallow DOF'\n"
        "Lighting: 'golden hour', 'volumetric light', 'soft diffused light', 'dramatic backlighting'"
    ),
    negative_prompt="low quality, blurry, static, watermark, text overlay, deformed limbs, cartoon",
    pitfalls=(
        "Don't use keyword dumps — write sentences.\n"
        "Don't describe post-processing ('color graded') — describe the scene itself.\n"
        "Don't use 'ultra-realistic' or 'masterpiece' — they reduce quality for Wan."
    ),
    examples=(
        "BASIC: 'a woman walks in a park'\n"
        "ENHANCED: 'A smooth tracking shot follows a woman as she strolls through a sunlit park, "
        "autumn leaves drifting around her feet, golden afternoon light filtering through the trees.'"
    ),
)

# ── LTX-2 ─────────────────────────────────────────────────────────────────────
LTX2_GUIDE = ModelPromptGuide(
    model_name="LTX-2",
    syntax_rules=(
        "Use long, flowing sentences. Describe the full scene in one breath. "
        "Include: subject + action + camera motion + environment + light. "
        "End every prompt with 'Audio: [description]' — this model generates audio from text."
    ),
    keywords=(
        "Camera: 'steadycam', 'slow push in', 'arc shot', 'overhead crane'\n"
        "Motion: 'graceful', 'measured pace', 'deliberate movement'\n"
        "Environment: describe texture, depth, distance\n"
        "Light: 'dappled', 'low-key', 'high-contrast rim light', 'soft ambient'"
    ),
    negative_prompt="abrupt cuts, fast motion, pixel artifacts, unnatural movement, static",
    pitfalls=(
        "LTX-2 generates audio — always end with 'Audio: ...' or you'll get random music.\n"
        "Don't make frames too short (< 17 frames). Duration must be 8n+1 frames.\n"
        "Avoid very rapid motion — LTX-2 favors smooth, cinematic pacing."
    ),
    examples=(
        "BASIC: 'make the woman walk down the street'\n"
        "ENHANCED: 'The camera tracks the subject from behind with a smooth steadycam motion. "
        "The subject walks forward at a measured pace, moving away from the lens. Shadows lengthen "
        "across the pavement and leaves rustle in the wind. "
        "Audio: steady footsteps on pavement, distant city hum, soft ambient score.'"
    ),
    audio_guidance=(
        "LTX-2 generates a FULL SOUNDTRACK from the text prompt. "
        "Append 'Audio: [description]' as the FINAL sentence of every video prompt.\n"
        "Rules:\n"
        "- Describe DIEGETIC sounds first: footsteps, wind, crowd noise, machinery, rain, etc.\n"
        "- Then describe MUSIC MOOD: 'soft ambient score', 'tense strings', 'no music — silence'.\n"
        "- Keep audio description under 20 words.\n"
        "- If quiet/tense: 'Audio: near silence, distant traffic hum, no music.'\n"
        "Examples:\n"
        "  Action: 'Audio: rapid footsteps on concrete, car engine roar, tense orchestral strings.'\n"
        "  Nature: 'Audio: birdsong, rustling leaves, gentle stream, peaceful ambient score.'\n"
        "  Horror: 'Audio: near silence, slow heartbeat, distant creak, no music.'"
    ),
)

# ── HunyuanVideo ──────────────────────────────────────────────────────────────
HUNYUAN_GUIDE = ModelPromptGuide(
    model_name="HunyuanVideo",
    syntax_rules=(
        "Write in detailed descriptive prose. Start with subject, then action, then environment. "
        "Include lighting, mood, and camera motion. Keep under 200 words."
    ),
    keywords=(
        "Camera: 'cinematic pan', 'close-up reveal', 'wide establishing'\n"
        "Quality: 'high detail', 'photorealistic', 'professional photography'\n"
        "Lighting: 'natural light', 'studio lighting', 'cinematic mood'"
    ),
    negative_prompt="low quality, deformed, blurry, watermark, text, oversaturated",
    pitfalls=(
        "Don't over-specify — Hunyuan has strong defaults.\n"
        "Avoid conflicting lighting descriptions."
    ),
    examples=(
        "BASIC: 'a man in a suit'\n"
        "ENHANCED: 'A man in a tailored charcoal suit stands in a softly lit office, "
        "light streaming through venetian blinds casting parallel shadows across his face.'"
    ),
)

# ── FLUX.2 ────────────────────────────────────────────────────────────────────
FLUX2_GUIDE = ModelPromptGuide(
    model_name="FLUX.2",
    syntax_rules=(
        "Use NATURAL LANGUAGE flowing prose ('novelist-style'). Avoid comma-separated keyword lists. "
        "Weight syntax like (keyword:1.2) is NOT supported. Use natural language emphasis. "
        "Supports up to 32 000 tokens. Recommended: 10-30 words for concepts, 30-80 for production. "
        "WORD ORDER MATTERS — front-load the most important elements. "
        "Hierarchy: Subject → Action → Style → Context/Setting → Lighting → Atmosphere. "
        "Supports HEX color codes for exact color matching (e.g. 'color #F22E63'). "
        "This is an IMAGE generation model — do not describe temporal/video concepts."
    ),
    keywords=(
        "VISUAL STYLE (Photo): 'Modern Digital', 'shot on Sony A7IV', 'clean sharp', 'high dynamic range'\n"
        "VISUAL STYLE (Vintage): '80s vintage photo', '2000s digicam', 'Analog Film', 'Kodak Portra 400'\n"
        "VISUAL STYLE (Art): 'Isometric 3D cartoon', 'flat illustration', 'kawaii', 'art deco'\n"
        "LIGHTING: golden hour, volumetric lighting, studio softbox, rim light, chiaroscuro\n"
        "COMPOSITION: rule of thirds, centered, shallow DOF, bokeh, macro lens, wide angle, telephoto"
    ),
    negative_prompt=(
        "FLUX.2 does NOT support negative prompts. Focus strictly on describing what you WANT. "
        "Instead of 'no blur', use 'sharp focus throughout'. "
        "Instead of 'no artifacts', describe the quality you want."
    ),
    pitfalls=(
        "Do NOT use comma-separated tag lists — use descriptive sentences.\n"
        "Do NOT mix conflicting styles (e.g. 'photorealistic' AND 'watercolor').\n"
        "Do NOT neglect lighting descriptions — specifying light source is critical.\n"
        "The [klein] 9B variant can over-sharpen at too many steps — keep 4 steps.\n"
        "Do NOT use weight syntax like (keyword:1.2) — not supported."
    ),
    examples=(
        "BASIC: 'a coffee mug on a table'\n"
        "ENHANCED: 'Professional studio product shot on polished concrete. Subject: minimalist "
        "ceramic coffee mug, matte black, steam rising from hot coffee, centered in frame. Style: "
        "ultra-realistic commercial photography. Lighting: three-point softbox setup, diffused highlights.'\n\n"
        "BASIC: 'portrait of an old man'\n"
        "ENHANCED: 'A weathered fisherman in his late sixties stands at the bow of a small wooden boat, "
        "wearing a salt-stained wool sweater, hands gripping frayed rope. Golden hour sunlight filters "
        "through morning mist, documentary style, shot on 35mm film.'"
    ),
)

# ── FLUX.1 ────────────────────────────────────────────────────────────────────
FLUX1_GUIDE = ModelPromptGuide(
    model_name="Flux.1",
    syntax_rules=(
        "Use NATURAL LANGUAGE descriptive sentences. Avoid 'tag soup' keyword stuffing. "
        "Weight syntax like (keyword:1.2) is NOT supported — Flux does not use standard CFG. "
        "T5 encoder handles up to 512 tokens for detailed descriptions. "
        "Use STATIC, STRUCTURED descriptions. Framework: Subject → Action → Style → Context. "
        "Layered descriptions work well: Foreground, Middle ground, Background. "
        "Supports HEX color codes directly in prompts. "
        "Do NOT include politeness markers ('Please create an image of...'). "
        "This is a STILL IMAGE model — do not describe temporal/video concepts."
    ),
    keywords=(
        "VISUAL STYLE: photorealistic, cinematic, 3D render, anime, film grain, editorial raw portrait\n"
        "LIGHTING: golden hour, natural window light, studio lighting, rim light, volumetric, chiaroscuro\n"
        "MOTION (stills): motion blur, action shot, dynamic pose, frozen in time\n"
        "COMPOSITION: close-up, wide angle, low angle, shallow DOF, bokeh, rule of thirds, macro lens"
    ),
    negative_prompt=(
        "Flux.1 does NOT support negative prompts — it does not use standard CFG. "
        "Describe what you WANT to see. Use 'clear sky' instead of 'no clouds'."
    ),
    pitfalls=(
        "Do NOT use 'tag soup' keyword stuffing ('masterpiece, best quality, 8k').\n"
        "Do NOT use negative prompts — they are ignored.\n"
        "Do NOT use weight syntax like (keyword:1.5) — it does not function.\n"
        "AVOID 'white background' — causes blurry images. Use 'studio background' instead.\n"
        "Without texture keywords ('pores', 'grain'), results look 'plastic'.\n"
        "Resolutions should be divisible by 16."
    ),
    examples=(
        "BASIC: 'a cat walking through a garden'\n"
        "ENHANCED: 'A black cat stalking through tall grass in a lush garden, low angle shot, "
        "shallow depth of field, golden hour lighting, cinematic wildlife photography style.'\n\n"
        "BASIC: 'a woman in a city'\n"
        "ENHANCED: 'Professional portrait of a woman in her 30s wearing a business suit, standing on "
        "a busy Tokyo street at dusk, neon signs reflecting, bokeh, shot on 85mm lens, f/1.8.'"
    ),
)

# ── Registry ──────────────────────────────────────────────────────────────────
MODEL_GUIDES: Dict[str, ModelPromptGuide] = {
    # Wan 2.1 variants
    "t2v": WAN_GUIDE,
    "i2v": WAN_GUIDE,
    "t2v_1.3B": WAN_GUIDE,
    # Wan 2.2 variants
    "i2v_2_2": WAN_GUIDE,
    "ti2v_2_2": WAN_GUIDE,
    "t2v_2_2": WAN_GUIDE,
    # LTX family
    "ltx2": LTX2_GUIDE,
    "ltx2_distilled": LTX2_GUIDE,
    "ltx2_19B": LTX2_GUIDE,
    "ltxv_13B": LTX2_GUIDE,
    # HunyuanVideo family
    "hunyuan_1_5_t2v": HUNYUAN_GUIDE,
    "hunyuan_1_5_i2v": HUNYUAN_GUIDE,
    "hunyuan": HUNYUAN_GUIDE,
    "hunyuan_i2v": HUNYUAN_GUIDE,
    # FLUX.2 family
    "flux2_dev": FLUX2_GUIDE,
    "flux2_klein_4b": FLUX2_GUIDE,
    "flux2_klein_9b": FLUX2_GUIDE,
    # FLUX.1 family
    "flux": FLUX1_GUIDE,
    "flux_schnell": FLUX1_GUIDE,
    "flux_dev_kontext": FLUX1_GUIDE,
}


def get_guide(model_id: str) -> Optional[ModelPromptGuide]:
    """Return the guide for a model id, with prefix-matching fallback."""
    if not model_id:
        return None
    if model_id in MODEL_GUIDES:
        return MODEL_GUIDES[model_id]
    # Prefix fallback: ltx2_custom → ltx2
    for key, guide in MODEL_GUIDES.items():
        if model_id.startswith(key):
            return guide
    return None


def format_guide_for_system_prompt(model_id: str) -> str:
    """Return a formatted system prompt section for this model, or empty string."""
    guide = get_guide(model_id)
    if not guide:
        return ""

    audio_section = ""
    audio_reminder = ""
    if guide.audio_guidance:
        audio_section = f"\n\nAUDIO GENERATION (MANDATORY for this model):\n{guide.audio_guidance}"
        audio_reminder = (
            "\nCRITICAL: This model generates audio from the text prompt. "
            "Every videoPrompt MUST end with 'Audio: ...' describing the soundscape."
        )

    return (
        f"=== MODEL-SPECIFIC PROMPTING GUIDE ({guide.model_name}) ===\n"
        f"Follow these rules when enhancing prompts for this model:\n\n"
        f"SYNTAX:\n{guide.syntax_rules}\n\n"
        f"HIGH-IMPACT KEYWORDS (use these when appropriate):\n{guide.keywords}\n\n"
        f"NEGATIVE PROMPT GUIDANCE:\n{guide.negative_prompt}\n\n"
        f"MISTAKES TO AVOID:\n{guide.pitfalls}\n\n"
        f"EXAMPLE TRANSFORMATIONS (mimic this style):\n{guide.examples}"
        f"{audio_section}\n"
        f"=== END MODEL GUIDE ==={audio_reminder}"
    )
