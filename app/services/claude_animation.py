"""Animated clips written by the local Claude Code CLI and rendered by HyperFrames.

Each clip is one scene: the logged-in ``claude`` CLI writes the styles, markup
and GSAP tweens, this module wraps them in a HyperFrames document it owns, and
``hyperframes render`` turns that into an MP4. Nothing is billed per clip; the
calls count against the Claude subscription's usage limits.

A video is planned before its scenes are written: one extra request turns the
script into a shared look and one idea per clip (``plan_scenes``), so separately
written scenes follow the narration and look like one piece.

When the voiceover's subtitle timings are known, scenes are cut to them
(``scene_timings``): each scene lasts as long as the lines spoken over it. Every
scene is reviewed before it is accepted: ``hyperframes check`` looks at sampled
frames in a browser, and the rendered clip is tested for an empty frame.

Scenes are independent once the plan exists, so the caller writes several at
once (``concurrency``). With a stock footage source configured
(``footage_source``), the planner may mark a scene that shows something
filmable; the caller then uses a stock clip there instead of an animation.

Accepted plans and scenes are kept in ``storage/animation_cache`` and reused
when the same request comes up again, so making a video a second time costs no
Claude requests and no renders. Scenes load GSAP from a local copy, so a render
needs no network. Every Claude request's token counts go to ``llm_usage``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import animation_styles, llm, llm_usage
from app.services.animation_styles import AnimationStyle
from app.utils import subtitle_reader, utils

PROVIDER_ID = "claude_animation"
# Pinned so a rebuilt image or a cold npx cache cannot change render behaviour.
HYPERFRAMES_VERSION = "0.9.13"
GSAP_VERSION = "3.15.0"
GSAP_URL = f"https://cdn.jsdelivr.net/npm/gsap@{GSAP_VERSION}/dist/gsap.min.js"
# The scene runs whatever this file contains, so a local copy is used only when
# it is byte for byte the published one.
GSAP_SHA256 = "92bb9a96476f983d212a2bc4f54c889039c1696dd4461d40a736860938570fbb"
GSAP_FILE_NAME = "gsap.min.js"
GSAP_DOWNLOAD_TIMEOUT_SECONDS = 20.0
# After a failed download the CDN address is used for this long before retrying.
GSAP_RETRY_SECONDS = 600.0
# Part of every cache key: raise it when a change here makes old entries wrong.
CACHE_FORMAT = 1
# Plans and scenes kept; the least recently used are deleted beyond this.
MAX_CACHE_ENTRIES = 300
_CACHE_EXTENSIONS = (".mp4", ".json")
FPS = 30
DEFAULT_AUTHOR_TIMEOUT_SECONDS = 300.0
DEFAULT_RENDER_TIMEOUT_SECONDS = 600.0
DEFAULT_QUALITY = "draft"
QUALITIES = ("draft", "looks", "delivery")
DEFAULT_MAX_ATTEMPTS = 2
MAX_ERROR_TEXT_LENGTH = 600
MAX_SCRIPT_CONTEXT_LENGTH = 1500
MAX_PLAN_SCRIPT_LENGTH = 6000
MAX_PLAN_SCENES = 60
MAX_LOOK_LENGTH = 1200
MAX_IDEA_LENGTH = 400
MAX_NARRATION_LENGTH = 300
MAX_FOOTAGE_QUERY_LENGTH = 80
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 6
FOOTAGE_SOURCES = ("pexels", "pixabay", "coverr")
# A scene shorter than this has no time to read; it is merged into its neighbour.
MIN_SCENE_SECONDS = 2.0
# Scene lengths are multiples of this, which is a whole number of frames at FPS.
SCENE_TIME_STEP = 0.1
# The last scene runs slightly past the voiceover so the cut never falls short.
SCENE_TAIL_SECONDS = 0.5
REVIEW_SAMPLES = 5
# Subtitles are drawn over the bottom of the frame; scene text must stay above.
CAPTION_BAND_TOP = 0.78
# Browser audits that fail a scene although HyperFrames only warns about them.
_REVIEW_BLOCKING_WARNINGS = frozenset({"sweep_static"})
# A frame is empty when fewer than this share of its pixels differ from its
# most common colour.
EMPTY_FRAME_MIN_CONTENT = 0.004

RECOVERY_REPAIR = "repair"
RECOVERY_SKIP = "skip"
RECOVERY_ABORT = "abort"

SYSTEM_PROMPT = (
    "You are a motion designer who writes HyperFrames scenes: HTML, CSS and "
    "GSAP tweens that a renderer seeks frame by frame. Follow the output format "
    "exactly and output nothing else."
)

PLAN_SYSTEM_PROMPT = (
    "You are the art director of a short animated video. You turn a narration "
    "script into a shot list for motion designers. Output only the JSON object "
    "you are asked for."
)

_XML_NAMESPACE_RE = re.compile(r"https?://www\.w3\.org/[\w./#-]*", re.IGNORECASE)
_BLOCK_RE = re.compile(r"```(css|html|js|javascript)[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# The scene runs in a real headless Chrome. It is generated text, so anything
# that loads code or talks to the network is refused before it reaches disk.
_FORBIDDEN_PATTERNS = (
    (re.compile(r"https?:|//[\w-]+(\.[\w-]+)*\.(com|net|org|io|dev|co|ai|app)\b", re.IGNORECASE),
     "a remote URL"),
    (re.compile(r"<\s*(script|iframe|link|object|embed|base|meta|img|video|audio)\b", re.IGNORECASE),
     "an external resource tag"),
    (re.compile(r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts)\b"),
     "a network call"),
    (re.compile(r"\b(eval|Function)\s*\(|\bimport\s*\("), "dynamic code loading"),
    (re.compile(r"url\(\s*['\"]?(?!data:|#)", re.IGNORECASE), "a url() that is not inline data"),
    (re.compile(r"@import\b", re.IGNORECASE), "a CSS @import"),
    (re.compile(r"\b(setTimeout|setInterval|requestAnimationFrame|Date\.now|Math\.random)\b"),
     "a timer or non-deterministic call"),
    (re.compile(r"repeat\s*:\s*-1"), "an infinite repeat"),
)


class ClaudeAnimationError(RuntimeError):
    """A clip could not be produced."""


class ClaudeAnimationUnavailableError(ClaudeAnimationError):
    """The runtime is missing the claude CLI, its login, or HyperFrames."""


class _SceneFailure(Exception):
    def __init__(self, stage: str, detail: str):
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


def _settings(settings: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return config.app if settings is None else settings


def _text(settings: Mapping[str, Any], key: str) -> str:
    return str(settings.get(key) or "").strip()


def _bounded_float(settings: Mapping[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(settings.get(key) or default)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return min(max(value, low), high)


def _resolve_executable(configured: str, default: str) -> str:
    command = configured or default
    found = shutil.which(command)
    if not found and configured and os.path.isfile(configured):
        found = configured
    return found or ""


def claude_cli_path(settings: Mapping[str, Any] | None = None) -> str:
    """Resolve the claude CLI, reusing the LLM provider's path when none is set here."""
    source = _settings(settings)
    configured = _text(source, "claude_animation_cli_path") or _text(source, "claude_code_cli_path")
    return _resolve_executable(configured, "claude")


def hyperframes_command(settings: Mapping[str, Any] | None = None) -> list[str]:
    """Return the argv prefix for HyperFrames, or an empty list when it cannot run."""
    source = _settings(settings)
    configured = _text(source, "claude_animation_hyperframes_path")
    direct = _resolve_executable(configured, "hyperframes")
    if direct:
        return [direct]
    if configured:
        return []
    npx = shutil.which("npx")
    return [npx, "--yes", f"hyperframes@{HYPERFRAMES_VERSION}"] if npx else []


def missing_requirements(settings: Mapping[str, Any] | None = None) -> list[str]:
    missing = []
    if not claude_cli_path(settings):
        missing.append("claude CLI")
    if not hyperframes_command(settings):
        missing.append("HyperFrames CLI (or Node.js with npx)")
    return missing


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return not missing_requirements(settings)


def quality(settings: Mapping[str, Any] | None = None) -> str:
    value = _text(_settings(settings), "claude_animation_quality").lower()
    return value if value in QUALITIES else DEFAULT_QUALITY


def max_attempts(settings: Mapping[str, Any] | None = None) -> int:
    return int(_bounded_float(_settings(settings), "claude_animation_max_attempts", DEFAULT_MAX_ATTEMPTS, 1, 4))


def planning_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return _flag(_settings(settings), "claude_animation_plan")


def _flag(settings: Mapping[str, Any], key: str) -> bool:
    value = settings.get(key, True)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def follow_narration_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Cut scenes to the voiceover's subtitle timings (on unless switched off)."""
    return _flag(_settings(settings), "claude_animation_follow_narration")


def review_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Review each scene's frames before accepting it (on unless switched off)."""
    return _flag(_settings(settings), "claude_animation_review")


def concurrency(settings: Mapping[str, Any] | None = None) -> int:
    """How many scenes are written and rendered at the same time."""
    return int(_bounded_float(
        _settings(settings), "claude_animation_concurrency", DEFAULT_CONCURRENCY, 1, MAX_CONCURRENCY
    ))


def footage_source(settings: Mapping[str, Any] | None = None) -> str:
    """The stock source for scenes better shown as footage, or ``""`` for animation only."""
    source = _text(_settings(settings), "claude_animation_footage_source").lower()
    return source if source in FOOTAGE_SOURCES else ""


def cache_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Reuse plans and scenes made earlier for the same request (on unless switched off)."""
    return _flag(_settings(settings), "claude_animation_cache")


def bill_usage_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Count Claude's list price as a task cost. Off by default: a subscription
    is not charged per request, so the price is recorded but not billed."""
    value = _settings(settings).get("claude_animation_bill_usage", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _cache_dir() -> str:
    return utils.storage_dir("animation_cache", create=True)


def _cache_key(kind: str, settings: Mapping[str, Any], *parts: Any) -> str:
    """Name one request. Everything that changes the result must be in ``parts``."""
    payload = json.dumps(
        [CACHE_FORMAT, kind, HYPERFRAMES_VERSION, GSAP_VERSION,
         _text(settings, "claude_animation_model"), *parts],
        ensure_ascii=False,
    )
    return f"{kind}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def _cached_file(key: str, extension: str) -> str:
    """The cache entry's path, or ``""`` when there is none."""
    if not key:
        return ""
    try:
        path = os.path.join(_cache_dir(), key + extension)
        if os.path.getsize(path) > 0:
            # Marks the entry as used, so pruning removes it last.
            os.utime(path)
            return path
    except OSError:
        pass
    return ""


_cache_lock = threading.Lock()


def _store_in_cache(key: str, extension: str, *, source_path: str = "", text: str = "") -> None:
    """Keep a file or a text under ``key``. A cache that cannot be written is only logged."""
    if not key:
        return
    try:
        directory = _cache_dir()
        path = os.path.join(directory, key + extension)
        # Written beside the entry and renamed, so a reader never sees half a file.
        partial = f"{path}.{uuid.uuid4().hex[:8]}.part"
        try:
            if source_path:
                shutil.copyfile(source_path, partial)
            else:
                Path(partial).write_text(text, encoding="utf-8")
            os.replace(partial, path)
        finally:
            if os.path.isfile(partial):
                os.remove(partial)
        with _cache_lock:
            entries = [
                entry for entry in os.scandir(directory)
                if entry.is_file() and entry.name.endswith(_CACHE_EXTENSIONS)
            ]
            entries.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
            for entry in entries[MAX_CACHE_ENTRIES:]:
                os.remove(entry.path)
    except OSError as exc:
        logger.warning(f"claude animation cache could not be written: {exc}")


_gsap_lock = threading.Lock()
_gsap_state = {"path": "", "failed_at": None}


def _is_gsap(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest() == GSAP_SHA256
    except OSError:
        return False


def _packaged_gsap_path(settings: Mapping[str, Any]) -> str:
    """The GSAP file an installed HyperFrames package carries, if it is installed."""
    command = hyperframes_command(settings)
    if not command:
        return ""
    for parent in Path(os.path.realpath(command[0])).parents:
        if parent.name == "hyperframes":
            return str(parent / "dist" / "vendor" / "npm" / f"gsap@{GSAP_VERSION}" / "dist" / "gsap.min.js")
    return ""


def gsap_library_path(settings: Mapping[str, Any] | None = None) -> str:
    """A verified local copy of GSAP, or ``""`` when none can be had.

    Looked for in the cache directory, then in the installed HyperFrames
    package; downloaded into the cache directory when neither has it.
    """
    with _gsap_lock:
        if _gsap_state["path"] and os.path.isfile(_gsap_state["path"]):
            return _gsap_state["path"]
        try:
            cached = os.path.join(_cache_dir(), f"gsap-{GSAP_VERSION}.min.js")
        except OSError:
            cached = ""
        for candidate in (cached, _packaged_gsap_path(_settings(settings))):
            if candidate and _is_gsap(candidate):
                _gsap_state["path"] = candidate
                return candidate
        failed_at = _gsap_state["failed_at"]
        if not cached or (failed_at is not None and time.monotonic() - failed_at < GSAP_RETRY_SECONDS):
            return ""
        partial = f"{cached}.{uuid.uuid4().hex[:8]}.part"
        try:
            with urllib.request.urlopen(GSAP_URL, timeout=GSAP_DOWNLOAD_TIMEOUT_SECONDS) as response:
                Path(partial).write_bytes(response.read())
            if not _is_gsap(partial):
                raise ValueError("the downloaded file is not the expected GSAP build")
            os.replace(partial, cached)
        except (OSError, ValueError) as exc:
            _gsap_state["failed_at"] = time.monotonic()
            logger.warning(f"no local GSAP copy, scenes load it from the CDN instead: {exc}")
            return ""
        finally:
            if os.path.isfile(partial):
                os.remove(partial)
        _gsap_state["path"] = cached
        return cached


def style_preset(settings: Mapping[str, Any] | None = None) -> AnimationStyle | None:
    """The configured named style, or ``None`` when none is set or its file is gone."""
    return animation_styles.get_style(_text(_settings(settings), "claude_animation_style_preset"))


@dataclass(frozen=True)
class PlannedScene:
    idea: str
    narration: str = ""
    # A stock search for a scene better shown as footage; ``idea`` still
    # describes the animation used when no footage is found.
    footage: str = ""


@dataclass(frozen=True)
class ScenePlan:
    look: str
    # One entry per scene in order; ``None`` where the planner gave no idea.
    scenes: tuple[PlannedScene | None, ...]


@dataclass(frozen=True)
class TimedScene:
    """One scene's place under the voiceover."""

    start: float
    duration: float
    narration: str


def _seconds(value: float) -> str:
    return f"{value:g}"


def scene_timings(
    cues: Sequence[tuple[float, float, str]],
    *,
    total_duration: float,
    target_duration: float,
) -> tuple[TimedScene, ...]:
    """Group subtitle cues into scenes that change when the speaker moves on.

    ``cues`` are ``(start, end, text)`` in seconds. A scene starts when its first
    cue starts and runs until the next scene's first cue, so pauses belong to the
    scene before them. A scene is closed at the cue boundary nearest to
    ``target_duration``. The first scene starts at 0 and the last one ends just
    after ``total_duration``. Returns ``()`` when the cues are unusable.
    """
    ordered = [
        (float(start), float(end), " ".join(str(text).split()))
        for start, end, text in cues
        if math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) > float(start)
    ]
    if not ordered or not target_duration > 0:
        return ()
    ordered.sort(key=lambda cue: cue[0])
    end_of_video = max(float(total_duration or 0), ordered[-1][1]) + SCENE_TAIL_SECONDS

    # Each group is [start, [texts]]; a group's end is the next group's start.
    groups: list[list[Any]] = [[0.0, []]]
    for index, (start, _end, text) in enumerate(ordered):
        group = groups[-1]
        next_start = ordered[index + 1][0] if index + 1 < len(ordered) else end_of_video
        length_without = start - group[0]
        length_with = next_start - group[0]
        # Close the scene before this cue when stopping here lands nearer the
        # target than taking the cue too.
        if (
            group[1]
            and length_without >= MIN_SCENE_SECONDS
            and length_with > target_duration
            and target_duration - length_without < length_with - target_duration
        ):
            group = [start, []]
            groups.append(group)
        group[1].append(text)
    if len(groups) > 1 and end_of_video - groups[-1][0] < MIN_SCENE_SECONDS:
        tail = groups.pop()
        groups[-1][1].extend(tail[1])

    # Round the cut points, not the lengths, so rounding never accumulates.
    cuts = [round(group[0] / SCENE_TIME_STEP) for group in groups]
    cuts.append(math.ceil(end_of_video / SCENE_TIME_STEP))
    return tuple(
        TimedScene(
            start=round(cuts[index] * SCENE_TIME_STEP, 1),
            duration=round((cuts[index + 1] - cuts[index]) * SCENE_TIME_STEP, 1),
            narration=" ".join(group[1])[:MAX_NARRATION_LENGTH],
        )
        for index, group in enumerate(groups)
    )


def _srt_seconds(value: str) -> float:
    hours, minutes, seconds = value.strip().replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_scene_timings(
    subtitle_path: str, *, total_duration: float, target_duration: float
) -> tuple[TimedScene, ...]:
    """Scene timings from an SRT file; ``()`` when it is missing or unreadable."""
    cues = []
    try:
        for _index, times, text in subtitle_reader.file_to_subtitles(subtitle_path):
            start, _arrow, end = times.partition("-->")
            cues.append((_srt_seconds(start), _srt_seconds(end), text))
    except (OSError, ValueError) as exc:
        logger.warning(f"claude animation could not read subtitle timings: {exc}")
        return ()
    return scene_timings(cues, total_duration=total_duration, target_duration=target_duration)


def _style_lines(preset: AnimationStyle | None, notes: str, *, with_example: bool) -> list[str]:
    lines = []
    if preset is not None:
        lines.append(f"Style: {preset.name}\n{preset.direction}")
        if with_example and preset.example:
            lines.append(
                "Reference scene in this style. It shows the style's vocabulary only: "
                "do not copy its composition, text or subject.\n" + preset.example
            )
    if notes:
        lines.append(f"Art direction: {notes}")
    return lines


def build_plan_prompt(
    *,
    script: str,
    scene_count: int,
    duration: int,
    width: int,
    height: int,
    subject: str = "",
    preset: AnimationStyle | None = None,
    style: str = "",
    timings: Sequence[TimedScene] = (),
    footage: bool = False,
) -> str:
    """``timings`` fixes each scene's narration and length; without it the planner
    splits the script. ``footage`` lets the planner ask for stock footage."""
    lines = [
        f"Plan a {width}x{height} animated video of exactly {scene_count} scenes, "
        + ("" if timings else f"{_seconds(duration)} seconds each, ")
        + "played in order under this narration.",
    ]
    if subject:
        lines.append(f"Video topic: {subject}")
    lines.append("Narration:\n" + script[:MAX_PLAN_SCRIPT_LENGTH])
    lines += _style_lines(preset, style, with_example=False)
    if timings:
        lines.append(
            "The narration is already cut into scenes. Each scene plays under these words:\n"
            + "\n".join(
                f"Scene {number} ({_seconds(timed.duration)} s): {timed.narration}"
                for number, timed in enumerate(timings[:scene_count], start=1)
            )
        )
        task = (
            f"Decide what each of the {scene_count} scenes shows. A short scene needs a "
            "simple idea; a long one can build in steps."
        )
        entry = '{"idea": "what is on screen and how it moves, in one or two sentences"'
    else:
        task = (
            f"Split the narration into {scene_count} consecutive parts of similar length, "
            "one per scene, and decide what each scene shows."
        )
        entry = (
            '{"narration": "the part of the narration this scene plays under", '
            '"idea": "what is on screen and how it moves, in one or two sentences"'
        )
    footage_rule = ""
    if footage:
        entry += ', "footage": "stock video search of 2-4 words, or empty"'
        footage_rule = """
- A scene may be stock video instead of animation: give "footage" a search that
  a stock library will match (a place, an object, people doing something), only
  when the narration names something concrete and filmable. Numbers, abstract
  ideas and processes stay animated, with "footage" empty. Use footage for at
  most half of the scenes, never for the first one, and still give every scene
  an animated "idea" in case no footage is found."""
    entry += "}"
    lines.append(
        f"""
{task} Scenes are drawn with HTML, CSS and inline
SVG shapes and a few large words: no photos, no footage, no realistic people.

Output one JSON object and nothing else:
{{
  "look": "at most 80 words every scene must follow: background, 3-5 colours as hex, typeface, shape language, how things move",
  "scenes": [
    {entry}
  ]
}}

Rules:
- Exactly {scene_count} scenes, in narration order.
- Each idea is concrete and visual (a chart filling, a coin stack growing, a path
  being drawn), never a restatement of the narration.
- Vary the composition from scene to scene; keep the look identical.{footage_rule}"""
    )
    return "\n".join(lines)


def parse_plan(reply: str, scene_count: int) -> ScenePlan:
    """Read the planner's JSON; raises ``_SceneFailure`` when it is unusable."""
    text = reply or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise _SceneFailure("plan", "reply has no JSON object")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise _SceneFailure("plan", f"reply is not valid JSON: {exc}") from None
    entries = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise _SceneFailure("plan", 'JSON has no "scenes" list')
    scenes: list[PlannedScene | None] = []
    for entry in entries[:scene_count]:
        idea = str(entry.get("idea") or "").strip() if isinstance(entry, dict) else ""
        # An entry without an idea keeps its place, so later scenes stay under
        # their own part of the narration.
        scenes.append(
            PlannedScene(
                idea=idea[:MAX_IDEA_LENGTH],
                narration=str(entry.get("narration") or "").strip()[:MAX_NARRATION_LENGTH],
                footage=" ".join(str(entry.get("footage") or "").split())[:MAX_FOOTAGE_QUERY_LENGTH],
            )
            if idea
            else None
        )
    if not any(scenes):
        raise _SceneFailure("plan", "the plan has no usable scenes")
    return ScenePlan(look=str(payload.get("look") or "").strip()[:MAX_LOOK_LENGTH], scenes=tuple(scenes))


def plan_scenes(
    *,
    script: str,
    scene_count: int,
    duration: int,
    video_aspect: VideoAspect,
    subject: str = "",
    timings: Sequence[TimedScene] = (),
    settings: Mapping[str, Any] | None = None,
) -> ScenePlan | None:
    """Plan the whole video in one request.

    Returns ``None`` when planning is off, there is no script, or the plan is
    unusable; the caller then falls back to one scene per keyword. A plan may
    hold fewer scenes than asked for, and ``None`` for a scene it has no idea
    for. With ``timings``, each scene's narration is the timed one.
    """
    source = _settings(settings)
    script = (script or "").strip()
    scene_count = min(int(scene_count), MAX_PLAN_SCENES)
    if not planning_enabled(source) or not script or scene_count <= 0:
        return None
    width, height = VideoAspect(video_aspect).to_resolution()
    prompt = build_plan_prompt(
        script=script, scene_count=scene_count, duration=duration,
        width=width, height=height, subject=subject,
        preset=style_preset(source), style=_text(source, "claude_animation_style"),
        timings=timings, footage=bool(footage_source(source)),
    )
    key = _cache_key("plan", source, PLAN_SYSTEM_PROMPT, prompt) if cache_enabled(source) else ""
    cached = _cached_file(key, ".json")
    try:
        if cached:
            reply = Path(cached).read_text(encoding="utf-8")
        else:
            reply = _ask_claude(prompt, source, system_prompt=PLAN_SYSTEM_PROMPT)
        plan = parse_plan(reply, scene_count)
    except (_SceneFailure, OSError) as failure:
        logger.warning(
            "claude animation plan failed, using one scene per keyword instead: "
            f"{str(failure)[:MAX_ERROR_TEXT_LENGTH]}"
        )
        return None
    if not cached:
        # Only a reply that parsed is kept, so a bad plan is asked for again.
        _store_in_cache(key, ".json", text=reply)
    if timings:
        plan = ScenePlan(
            look=plan.look,
            scenes=tuple(
                PlannedScene(idea=scene.idea, narration=timings[index].narration, footage=scene.footage)
                if scene is not None and index < len(timings)
                else scene
                for index, scene in enumerate(plan.scenes)
            ),
        )
    if not footage_source(source) and any(scene and scene.footage for scene in plan.scenes):
        # Footage was not offered, so a search the planner added anyway is dropped.
        plan = ScenePlan(
            look=plan.look,
            scenes=tuple(
                PlannedScene(idea=scene.idea, narration=scene.narration) if scene else None
                for scene in plan.scenes
            ),
        )
    logger.info(
        f"claude animation plan{' (from cache)' if cached else ''}: "
        f"{sum(scene is not None for scene in plan.scenes)}/{scene_count} "
        f"scenes, {sum(bool(scene and scene.footage) for scene in plan.scenes)} as footage, "
        f"look={plan.look!r}"
    )
    return plan


def decide_recovery(stage: str, attempt: int, attempts_allowed: int, detail: str) -> str:
    """Choose what happens after one scene attempt fails.

    ``stage`` is where it failed: ``author`` (Claude's reply was unusable),
    ``validate`` (the scene used a forbidden construct), ``check`` (HyperFrames
    lint errors), ``review`` (its frames show a defect: clipped or unreadable
    text, text under the subtitles, nothing moving, an empty frame) or
    ``render``. ``attempt`` counts from 1.

    Return ``RECOVERY_REPAIR`` to ask Claude again with ``detail`` as feedback,
    ``RECOVERY_SKIP`` to drop this scene and move to the next one, or
    ``RECOVERY_ABORT`` to fail the whole task.
    """
    # Same policy for every stage: repair until the attempt budget is spent,
    # then skip the scene. The caller replaces skipped scenes within a budget.
    if attempt < attempts_allowed:
        return RECOVERY_REPAIR
    return RECOVERY_SKIP


def build_prompt(
    *,
    scene: str,
    duration: float,
    width: int,
    height: int,
    subject: str = "",
    script: str = "",
    style: str = "",
    preset: AnimationStyle | None = None,
    look: str = "",
    narration: str = "",
    position: tuple[int, int] | None = None,
    variation: int = 0,
    feedback: str = "",
) -> str:
    lines = [
        f"Write one animated scene for a {width}x{height} video, exactly {_seconds(duration)} seconds long.",
        f"Scene idea: {scene}",
    ]
    if position:
        lines.append(f"This is scene {position[0]} of {position[1]} in the video.")
    if narration:
        lines.append(
            "The narrator says this while the scene plays (do not put it on screen "
            f"word for word): {narration}"
        )
    if subject:
        lines.append(f"Video topic: {subject}")
    if script and not narration:
        lines.append(
            "Narration of the whole video, for context only (do not put it on screen):\n"
            + script[:MAX_SCRIPT_CONTEXT_LENGTH]
        )
    if look:
        lines.append(
            "Look shared by every scene of this video. Follow it exactly so the "
            f"scenes match: {look}"
        )
    lines += _style_lines(preset, style, with_example=True)
    if variation:
        lines.append(
            f"This idea was already used {variation} time(s) in this video: "
            "use a clearly different composition and motion."
        )
    lines.append(
        f"""
The scene is mounted inside <div id="root"> ({width}x{height}px, position: relative,
overflow hidden, black background). A paused GSAP timeline already exists as `tl`
and is registered for you.

Rules:
- Every top-level element has a unique id, class="clip", data-start="0",
  data-duration="{_seconds(duration)}" and data-track-index (0, 1, 2...). Position them with
  CSS (position: absolute).
- Animate children of the .clip elements with tl.fromTo(selector, from, to, time).
  Never tween a .clip element itself, never use tl.from, and never set a CSS
  transform on an element GSAP transforms.
- Keep something moving across the whole {_seconds(duration)} seconds; finish every tween
  before {duration - 0.1:.1f}s.
- Draw with HTML, CSS and inline SVG only: no images, video, audio, remote URLs,
  extra scripts, timers, Math.random, Date or infinite repeats.
- Fonts: Inter, Montserrat, Outfit, Oswald, Poppins, Playfair Display or
  JetBrains Mono.
- On-screen text is at most a few large words. Keep everything inside the middle
  90% of the frame and keep the bottom 22% free of text: subtitles go there.
  Text must fit inside its box and contrast clearly with what is behind it.
- Fill the frame with a designed background; do not leave it black.

Output exactly three fenced blocks and nothing else:
```css
/* styles */
```
```html
<!-- the .clip elements -->
```
```js
// tl.fromTo(...) calls only
```"""
    )
    if feedback:
        lines.append(
            "Your previous attempt failed. Fix this and output all three blocks again:\n"
            + feedback[:MAX_ERROR_TEXT_LENGTH]
        )
    return "\n".join(lines)


def parse_scene(reply: str) -> dict[str, str]:
    """Split Claude's reply into its css/html/js blocks and screen them."""
    blocks: dict[str, str] = {}
    for language, body in _BLOCK_RE.findall(reply or ""):
        key = "js" if language.lower() == "javascript" else language.lower()
        blocks.setdefault(key, body.strip())
    missing = [key for key in ("css", "html", "js") if not blocks.get(key)]
    if missing:
        raise _SceneFailure("author", f"reply is missing the {', '.join(missing)} block(s)")
    for key, body in blocks.items():
        # XML namespace URIs on inline SVG are identifiers, never fetched.
        screened = _XML_NAMESPACE_RE.sub("", body)
        for pattern, label in _FORBIDDEN_PATTERNS:
            if pattern.search(screened):
                raise _SceneFailure("validate", f"the {key} block contains {label}, which is not allowed")
    if 'class="clip"' not in blocks["html"] and "class='clip'" not in blocks["html"]:
        raise _SceneFailure("validate", 'no element in the html block has class="clip"')
    return blocks


def build_document(
    blocks: Mapping[str, str], *, duration: float, width: int, height: int, gsap_src: str = GSAP_URL
) -> str:
    """``gsap_src`` is the CDN address or the name of a copy beside the document."""
    duration = _seconds(duration)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <script src="{gsap_src}"></script>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ margin: 0; width: {width}px; height: {height}px; overflow: hidden; background: #000; }}
      #root {{ position: relative; width: 100%; height: 100%; overflow: hidden; font-family: Inter, sans-serif; }}
    </style>
    <style>
{blocks["css"]}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{duration}"
         data-fps="{FPS}" data-width="{width}" data-height="{height}">
{blocks["html"]}
    </div>
    <script>
      const tl = gsap.timeline({{ paused: true }});
{blocks["js"]}
      window.__timelines["main"] = tl;
      tl.seek(0);
    </script>
  </body>
</html>
"""


def _ask_claude(prompt: str, settings: Mapping[str, Any], *, system_prompt: str = SYSTEM_PROMPT) -> str:
    cli_path = claude_cli_path(settings)
    if not cli_path:
        raise ClaudeAnimationUnavailableError(
            "claude CLI not found; install it or set claude_animation_cli_path in config.toml"
        )
    timeout = _bounded_float(
        settings, "claude_animation_timeout", DEFAULT_AUTHOR_TIMEOUT_SECONDS, 30, 3600
    )
    # Same isolation as the claude_code LLM provider: no tools, no user-level
    # CLAUDE.md, skills, hooks or MCP, and the prompt goes in on stdin.
    command = [
        cli_path, "-p", "--output-format", "json",
        "--system-prompt", system_prompt,
        "--tools", "",
        "--safe-mode",
    ]
    model = _text(settings, "claude_animation_model")
    if model:
        command += ["--model", model]
    env, _removed = llm.build_claude_code_env()
    with tempfile.TemporaryDirectory() as work_dir:
        try:
            completed = subprocess.run(
                command, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                cwd=work_dir, env=env,
            )
        except subprocess.TimeoutExpired:
            raise _SceneFailure("author", f"claude cli timed out after {timeout:.0f}s") from None
        except OSError as exc:
            raise ClaudeAnimationUnavailableError(f"claude cli could not be started: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else None
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        detail = (completed.stderr or stdout or "").strip()[:MAX_ERROR_TEXT_LENGTH]
        if "unknown option" in detail.lower():
            raise ClaudeAnimationUnavailableError(
                "the installed claude CLI is too old; upgrade to "
                f"{llm.CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail}"
            )
        raise _SceneFailure("author", f"claude cli returned no JSON (exit {completed.returncode}): {detail}")
    _record_usage(payload, settings)
    if payload.get("is_error") or completed.returncode != 0:
        reason = str(payload.get("result") or "").strip() or f"exit code {completed.returncode}"
        lowered = reason.lower()
        # Login and usage-limit failures will not fix themselves on a retry.
        if any(word in lowered for word in ("login", "logged in", "authenticat", "limit", "credit")):
            if "login" in lowered or "logged in" in lowered:
                reason += (
                    " (run `claude setup-token` on the host and pass the token to "
                    "the container as CLAUDE_CODE_OAUTH_TOKEN)"
                )
            raise ClaudeAnimationUnavailableError(f"claude cli: {reason[:MAX_ERROR_TEXT_LENGTH]}")
        raise _SceneFailure("author", f"claude cli error: {reason[:MAX_ERROR_TEXT_LENGTH]}")
    return str(payload.get("result") or "")


def _record_usage(payload: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
    """Add one Claude request's tokens and list price to the task's usage."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return

    def count(key: str) -> int:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0

    models = payload.get("modelUsage")
    model = next(iter(models), "") if isinstance(models, dict) else ""
    llm_usage.record_subscription_request(
        "animation", PROVIDER_ID, str(model or _text(settings, "claude_animation_model")),
        # Cached prompt tokens are input too; the CLI reports them separately.
        prompt_tokens=count("input_tokens") + count("cache_creation_input_tokens")
        + count("cache_read_input_tokens"),
        completion_tokens=count("output_tokens"),
        list_price_usd=payload.get("total_cost_usd"),
        billed=bill_usage_enabled(settings),
    )


def usage_totals() -> dict[str, Any] | None:
    """This task's Claude animation requests added up, or ``None`` when there were none."""
    summary = llm_usage.summarize()
    calls = [call for call in (summary or {}).get("calls", []) if call.get("provider") == PROVIDER_ID]
    if not calls:
        return None
    return {
        "requests": len(calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "list_price_usd": round(sum(call.get("list_price_usd") or 0 for call in calls), 4),
    }


def _hyperframes_env() -> dict[str, str]:
    env = dict(os.environ)
    # Renders are an implementation detail of this app: no usage telemetry and
    # no agent-skill installation on the machine that runs them.
    env.update({"HYPERFRAMES_NO_TELEMETRY": "1", "DO_NOT_TRACK": "1", "HYPERFRAMES_SKIP_SKILLS": "1"})
    return env


def _run_hyperframes(args: list[str], project_dir: str, settings: Mapping[str, Any], timeout: float):
    command = hyperframes_command(settings)
    if not command:
        raise ClaudeAnimationUnavailableError(
            "HyperFrames CLI not found; install Node.js 22+ (for npx) or set "
            "claude_animation_hyperframes_path in config.toml"
        )
    try:
        return subprocess.run(
            command + args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, cwd=project_dir, env=_hyperframes_env(),
        )
    except subprocess.TimeoutExpired:
        raise _SceneFailure(args[0], f"hyperframes {args[0]} timed out after {timeout:.0f}s") from None
    except OSError as exc:
        raise ClaudeAnimationUnavailableError(f"hyperframes could not be started: {exc}") from exc


def _lint_errors(project_dir: str, settings: Mapping[str, Any], timeout: float) -> str:
    completed = _run_hyperframes(["lint", "--json"], project_dir, settings, timeout)
    try:
        report = json.loads(completed.stdout or "")
    except json.JSONDecodeError:
        # An unreadable report is not proof of a bad scene; the strict render decides.
        return ""
    findings = report.get("findings") if isinstance(report, dict) else None
    errors = [
        f"{item.get('code')}: {item.get('message')} Fix: {item.get('fixHint') or ''}".strip()
        for item in findings or []
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    return "\n".join(errors)


def _review_errors(project_dir: str, duration: float, settings: Mapping[str, Any], timeout: float) -> str:
    """Defects ``hyperframes check`` sees in sampled frames, one per line.

    The check lays the scene out in a browser at several moments and reports
    text that overflows its box or the frame, text that sits where the subtitles
    go, text too close in colour to what is behind it, and a scene where nothing
    moves. Lint findings are left to ``_lint_errors``.
    """
    completed = _run_hyperframes(
        ["check", "--json", "--samples", str(REVIEW_SAMPLES),
         "--caption-zone", f"x0=0;y0={CAPTION_BAND_TOP};x1=1;y1=1;severity=error"],
        project_dir, settings, timeout,
    )
    try:
        report = json.loads(completed.stdout or "")
    except json.JSONDecodeError:
        # No report is not proof of a bad scene; the render still has to succeed.
        return ""
    if not isinstance(report, dict) or report.get("browserSkipped"):
        return ""
    errors = []
    for section in ("runtime", "layout", "contrast"):
        audit = report.get(section)
        for item in (audit.get("findings") if isinstance(audit, dict) else None) or []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "")
            if item.get("severity") != "error" and code not in _REVIEW_BLOCKING_WARNINGS:
                continue
            where = " ".join(
                part for part in (
                    str(item.get("selector") or ""),
                    f'"{str(item.get("text"))[:60]}"' if item.get("text") else "",
                    f"at {_seconds(float(item['time']))}s" if isinstance(item.get("time"), (int, float)) else "",
                ) if part
            )
            # The hints end by offering an attribute that silences the finding;
            # the scene has to be fixed, not excused.
            fix = str(item.get("fixHint") or "").split(", or mark intentional")[0]
            if not fix and item.get("suggestedColor"):
                fix = f"use {item['suggestedColor']} or a colour at least that far from the background"
            if code == "sweep_static":
                fix = f"keep visible motion going across the whole {_seconds(duration)} seconds"
            errors.append(
                f"{code}: {item.get('message')}" + (f" ({where})" if where else "")
                + (f" Fix: {fix}" if fix else "")
            )
    return "\n".join(dict.fromkeys(errors))


def _empty_frame_error(video_path: str, duration: float) -> str:
    """Say so when the middle of the rendered clip is one flat colour."""
    try:
        import numpy as np
        from moviepy import VideoFileClip

        clip = VideoFileClip(video_path, audio=False)
        try:
            frame = np.asarray(clip.get_frame(min(duration, float(clip.duration)) / 2))
        finally:
            clip.close()
    except Exception as exc:
        # A clip this test cannot read is judged by the later video stages.
        logger.debug(f"claude animation frame test skipped: {exc}")
        return ""
    # A coarse grid is enough to tell a drawn frame from a flat one.
    pixels = frame[::8, ::8, :3].reshape(-1, 3).astype(np.int16)
    if not len(pixels):
        return ""
    colours, counts = np.unique(pixels // 16, axis=0, return_counts=True)
    background = colours[counts.argmax()] * 16 + 8
    content = float((np.abs(pixels - background).max(axis=1) > 24).mean())
    if content >= EMPTY_FRAME_MIN_CONTENT:
        return ""
    return (
        f"empty_frame: halfway through, {content:.1%} of the frame differs from the "
        "background, so the viewer sees a blank screen. Fix: make sure the shapes and "
        "words are inside the frame, large, visible (opacity 1 by mid-scene) and not "
        "the same colour as the background."
    )


def _render(project_dir: str, output_path: str, settings: Mapping[str, Any], timeout: float) -> None:
    completed = _run_hyperframes(
        ["render", "--quality", quality(settings), "--fps", str(FPS), "--quiet",
         "--strict", "--output", output_path],
        project_dir, settings, timeout,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-MAX_ERROR_TEXT_LENGTH:]
        raise _SceneFailure("render", f"hyperframes render exited with {completed.returncode}: {detail}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise _SceneFailure("render", "hyperframes render produced no video file")


def generate_video(
    *,
    search_term: str,
    duration: float,
    video_aspect: VideoAspect,
    output_dir: str,
    subject: str = "",
    script: str = "",
    variation: int = 0,
    look: str = "",
    narration: str = "",
    position: tuple[int, int] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> MaterialInfo | None:
    """Author and render one clip. Returns ``None`` when the scene is skipped.

    ``search_term`` is the scene idea: a planned idea or a bare keyword.
    ``look``, ``narration`` and ``position`` come from ``plan_scenes``.
    ``duration`` is in seconds and is rounded to a tenth.
    """
    source = _settings(settings)
    preset = style_preset(source)
    try:
        duration = round(float(duration), 1)
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise ClaudeAnimationError("Claude animation clip duration must be a positive number")
    if duration == int(duration):
        duration = int(duration)
    review = review_enabled(source)
    width, height = VideoAspect(video_aspect).to_resolution()
    render_timeout = _bounded_float(
        source, "claude_animation_render_timeout", DEFAULT_RENDER_TIMEOUT_SECONDS, 30, 3600
    )
    attempts_allowed = max_attempts(source)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"claude-animation-{uuid.uuid4().hex[:12]}.mp4")

    def prompt_with(feedback: str) -> str:
        return build_prompt(
            scene=search_term, duration=duration, width=width, height=height,
            subject=subject, script=script,
            style=_text(source, "claude_animation_style"),
            preset=preset, look=look, narration=narration, position=position,
            variation=variation, feedback=feedback,
        )

    def clip(cached: bool) -> MaterialInfo:
        return MaterialInfo(
            provider=PROVIDER_ID,
            url=output_path,
            # MaterialInfo counts whole seconds; the exact length is kept beside it.
            duration=math.ceil(duration),
            source_info={
                "provider": PROVIDER_ID,
                "search_term": search_term,
                "seconds": duration,
                "style": preset.slug if preset else "",
                "rendition": {"width": width, "height": height},
                "cached": cached,
            },
        )

    # The first prompt names the request: the same one always asks for the same
    # scene, whatever repairs it took to get it accepted last time.
    key = (
        _cache_key("scene", source, SYSTEM_PROMPT, prompt_with(""), quality(source), FPS, review)
        if cache_enabled(source)
        else ""
    )
    cached = _cached_file(key, ".mp4")
    if cached:
        try:
            shutil.copyfile(cached, output_path)
        except OSError as exc:
            logger.warning(f"claude animation cache entry could not be used: {exc}")
        else:
            logger.info(f"claude animation scene reused from cache: term={search_term!r}")
            return clip(True)

    gsap_path = gsap_library_path(source)
    feedback = ""
    for attempt in range(1, attempts_allowed + 1):
        try:
            reply = _ask_claude(prompt_with(feedback), source)
            blocks = parse_scene(reply)
            with tempfile.TemporaryDirectory(prefix="mpt-animation-") as project_dir:
                if gsap_path:
                    shutil.copyfile(gsap_path, os.path.join(project_dir, GSAP_FILE_NAME))
                Path(project_dir, "index.html").write_text(
                    build_document(
                        blocks, duration=duration, width=width, height=height,
                        gsap_src=GSAP_FILE_NAME if gsap_path else GSAP_URL,
                    ),
                    encoding="utf-8",
                )
                lint_errors = _lint_errors(project_dir, source, render_timeout)
                if lint_errors:
                    raise _SceneFailure("check", lint_errors)
                # Reviewed before the render: the check takes seconds, a render
                # of a scene that will be rejected takes much longer.
                review_errors = _review_errors(project_dir, duration, source, render_timeout) if review else ""
                if review_errors:
                    raise _SceneFailure("review", review_errors)
                _render(project_dir, output_path, source, render_timeout)
            empty_frame = _empty_frame_error(output_path, duration) if review else ""
            if empty_frame:
                raise _SceneFailure("review", empty_frame)
        except _SceneFailure as failure:
            logger.warning(
                f"claude animation scene failed: term={search_term!r}, stage={failure.stage}, "
                f"attempt={attempt}/{attempts_allowed}, detail={failure.detail[:MAX_ERROR_TEXT_LENGTH]}"
            )
            action = decide_recovery(failure.stage, attempt, attempts_allowed, failure.detail)
            if action == RECOVERY_ABORT:
                raise ClaudeAnimationError(
                    f"Claude animation failed at {failure.stage}: "
                    f"{failure.detail[:MAX_ERROR_TEXT_LENGTH]}"
                ) from None
            # A clip that was rendered and then rejected must not be left behind.
            if os.path.isfile(output_path):
                os.remove(output_path)
            if action == RECOVERY_REPAIR and attempt < attempts_allowed:
                feedback = f"[{failure.stage}] {failure.detail}"
                continue
            return None
        _store_in_cache(key, ".mp4", source_path=output_path)
        return clip(False)
    return None
