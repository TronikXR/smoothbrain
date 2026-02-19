# Smooth Brain Mode — Story Templates
# Port of TronikSlate/app/src/data/story-templates.ts
# Each template defines a genre and a set of shot beat patterns.
# {subject} is replaced at runtime with the user's concept.

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Dict

Genre = str  # "action" | "comedy" | "drama" | "horror" | "scifi" | "romance" | "fantasy" | "thriller"

ALL_GENRES: List[Genre] = [
    "action", "comedy", "drama", "horror", "scifi",
    "romance", "fantasy", "thriller",
]


@dataclass
class StoryTemplate:
    genre: Genre
    beats: List[str]   # each beat is a shot description with {subject} placeholder


TEMPLATES: List[StoryTemplate] = [
    StoryTemplate(genre="action", beats=[
        "Extreme wide shot: {subject} stands on the edge of a rooftop at sunset, city sprawling below.",
        "Low-angle shot: {subject} sprints toward camera, buildings blurring behind.",
        "Close-up: sweat on {subject}'s face, jaw clenched with determination.",
        "Action cam: {subject} leaps across a gap between buildings, arms reaching.",
        "Slow motion: {subject} lands hard on gravel, skids to a stop.",
        "Over-the-shoulder: {subject} faces an enemy, tension electric in the air.",
        "Dutch angle close-up: fists clench as the confrontation begins.",
        "Wide: an epic fight sequence unfolds in the orange dusk.",
        "Tracking shot: {subject} escapes into the crowd below, disappearing.",
        "Final wide: the city skyline quiet again, {subject} gone.",
    ]),
    StoryTemplate(genre="comedy", beats=[
        "Wide: {subject} wakes up to find everything hilariously wrong — chaos at first glance.",
        "Reaction shot: {subject} stares at the mess with growing horror and disbelief.",
        "Montage: frantic quick cuts of {subject} attempting absurd fixes.",
        "Close-up: a plan forms — a telltale gleam in {subject}'s eye.",
        "Wide: the plan goes spectacularly sideways, doubling the chaos.",
        "Over-the-shoulder: {subject} catches someone else's judging stare.",
        "Close-up: a sheepish grin as {subject} shrugs it all off.",
        "Wide: unexpected help arrives — making things chaotically better.",
        "Warm medium shot: laughter fills the room, the mess forgotten.",
        "Final close-up: {subject} raises an eyebrow at the camera — victorious.",
    ]),
    StoryTemplate(genre="drama", beats=[
        "Wide: {subject} alone at a window, rain streaking the glass.",
        "Close-up: a worn photograph in {subject}'s hands — a memory.",
        "Flashback medium: {subject} in better days, laughing with someone now gone.",
        "Return to present: slow zoom on {subject}'s face, heavy with loss.",
        "A knock at the door — {subject} hesitates before answering.",
        "Two-shot: a difficult conversation unfolds, every word measured.",
        "Close-up cutaways: hands, eyes, the space between two people.",
        "Wide: {subject} makes a choice that cannot be undone.",
        "Tracking shot: {subject} walks away into uncertainty.",
        "Final: an empty chair, a single light left on.",
    ]),
    StoryTemplate(genre="horror", beats=[
        "Extreme wide: {subject} arrives at an isolated location as dusk falls.",
        "Medium: {subject} explores the space — something feels wrong.",
        "Close-up: {subject} finds an unsettling clue. Pause. Heartbeat.",
        "POV: something moves at the edge of frame — gone when we look.",
        "Overhead: {subject} is being watched. They don't know it yet.",
        "Jump cut: a sudden sound — {subject} spins, nothing there.",
        "Slow crawl: the camera inches toward a closed door.",
        "Wide: the door opens — darkness beyond.",
        "Chaos cut: rapid flashes — screaming, running, confusion.",
        "Final frame: silence. An ominous detail left for the audience.",
    ]),
    StoryTemplate(genre="scifi", beats=[
        "Establishing wide: a vast alien or futuristic cityscape — {subject} is tiny against it.",
        "Close-up: {subject} studies a holographic display, something anomalous.",
        "Cutaway: the anomaly spreading — a system failing.",
        "Medium: {subject} makes a desperate call to action.",
        "Tracking: {subject} races through gleaming corridors, alarms sounding.",
        "Wide: a massive machine or spaceship — {subject} dwarfed by scale.",
        "Close-up: {subject}'s gloved hands working controls under pressure.",
        "Reaction: the countdown — three seconds.",
        "Flash cut: the solution — beauty and destruction at once.",
        "Final wide: silence returns. Stars. {subject} floats, breathing.",
    ]),
    StoryTemplate(genre="romance", beats=[
        "Wide: two strangers in the same crowded space — {subject} notices.",
        "POV: their eyes meet for just a moment, then look away.",
        "Montage: chance encounters, each a little longer than the last.",
        "Close-up: {subject}'s hand brushes theirs. Stillness.",
        "Medium: the first real conversation — nervous, genuine.",
        "Golden hour wide: walking side by side as light fades.",
        "Close-up: a smile that says everything before words do.",
        "Wide: a moment of doubt — distance opens between them.",
        "Close-up: {subject} makes a choice — steps forward.",
        "Final wide: two silhouettes, together against the fading sky.",
    ]),
    StoryTemplate(genre="fantasy", beats=[
        "Epic wide: a mythical landscape — {subject} stands at its threshold.",
        "Close-up: an ancient artifact or mark on {subject}'s hand glows.",
        "Medium: a mysterious guide appears — a warning, a map.",
        "Montage: the journey — mountains, forests, rivers crossed.",
        "Close-up: {subject} faces their first test — fear and resolve.",
        "Wide: an impossible creature or wonder fills the frame.",
        "Slow motion: {subject} discovers a hidden power within.",
        "Wide: the final confrontation — light against shadow.",
        "Close-up: the decisive moment — sacrifice and triumph.",
        "Final wide: the world changed forever. {subject} changed too.",
    ]),
    StoryTemplate(genre="thriller", beats=[
        "Wide: {subject} follows someone through a crowded city — closing in.",
        "Close-up: a suspicious object discovered — face drains of color.",
        "Cutaway: a clock. A deadline. The stakes crystallize.",
        "Rapid cuts: piecing together clues across scattered locations.",
        "Medium: {subject} is confronted — something doesn't add up.",
        "Close-up: a lie detected in someone's eyes.",
        "Wide: the trap closes — {subject} is deeper than they thought.",
        "Over-the-shoulder: {subject} makes a dangerous call for help.",
        "Extreme close-up: a hand on a weapon. A choice. A breath.",
        "Final: silence after impact. The truth revealed at last.",
    ]),
]


def get_weighted_templates(weights: Dict[Genre, int], count: int = 1) -> List[StoryTemplate]:
    """Return `count` templates, sampled by genre weight (0 = excluded)."""
    pool: List[StoryTemplate] = []
    for t in TEMPLATES:
        w = max(0, weights.get(t.genre, 0))
        pool.extend([t] * w)
    if not pool:
        pool = TEMPLATES[:]
    random.shuffle(pool)
    return pool[:count]


def fill_template(template: StoryTemplate, subject: str, shot_count: int) -> List[str]:
    """Replace {subject} in beats and trim to shot_count."""
    s = subject.strip() or "the hero"
    return [b.replace("{subject}", s) for b in template.beats[:shot_count]]
