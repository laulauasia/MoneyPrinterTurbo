"""Animated clips written by the local Claude Code CLI and rendered by HyperFrames.

Each clip is one scene: the logged-in ``claude`` CLI writes the styles, markup
and GSAP tweens, this module wraps them in a HyperFrames document it owns, and
``hyperframes render`` turns that into an MP4. Nothing is billed per clip; the
calls count against the Claude subscription's usage limits.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import llm

PROVIDER_ID = "claude_animation"
# Pinned so a rebuilt image or a cold npx cache cannot change render behaviour.
HYPERFRAMES_VERSION = "0.9.13"
GSAP_URL = "https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"
FPS = 30
DEFAULT_AUTHOR_TIMEOUT_SECONDS = 300.0
DEFAULT_RENDER_TIMEOUT_SECONDS = 600.0
DEFAULT_QUALITY = "draft"
QUALITIES = ("draft", "looks", "delivery")
DEFAULT_MAX_ATTEMPTS = 2
MAX_ERROR_TEXT_LENGTH = 600
MAX_SCRIPT_CONTEXT_LENGTH = 1500

RECOVERY_REPAIR = "repair"
RECOVERY_SKIP = "skip"
RECOVERY_ABORT = "abort"

SYSTEM_PROMPT = (
    "You are a motion designer who writes HyperFrames scenes: HTML, CSS and "
    "GSAP tweens that a renderer seeks frame by frame. Follow the output format "
    "exactly and output nothing else."
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


def decide_recovery(stage: str, attempt: int, attempts_allowed: int, detail: str) -> str:
    """Choose what happens after one scene attempt fails.

    ``stage`` is where it failed: ``author`` (Claude's reply was unusable),
    ``validate`` (the scene used a forbidden construct), ``check`` (HyperFrames
    lint errors) or ``render``. ``attempt`` counts from 1.

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
    duration: int,
    width: int,
    height: int,
    subject: str = "",
    script: str = "",
    style: str = "",
    variation: int = 0,
    feedback: str = "",
) -> str:
    lines = [
        f"Write one animated scene for a {width}x{height} video, exactly {duration} seconds long.",
        f"Scene idea: {scene}",
    ]
    if subject:
        lines.append(f"Video topic: {subject}")
    if script:
        lines.append(
            "Narration of the whole video, for context only (do not put it on screen):\n"
            + script[:MAX_SCRIPT_CONTEXT_LENGTH]
        )
    if style:
        lines.append(f"Art direction: {style}")
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
  data-duration="{duration}" and data-track-index (0, 1, 2...). Position them with
  CSS (position: absolute).
- Animate children of the .clip elements with tl.fromTo(selector, from, to, time).
  Never tween a .clip element itself, never use tl.from, and never set a CSS
  transform on an element GSAP transforms.
- Keep something moving across the whole {duration} seconds; finish every tween
  before {duration - 0.1:.1f}s.
- Draw with HTML, CSS and inline SVG only: no images, video, audio, remote URLs,
  extra scripts, timers, Math.random, Date or infinite repeats.
- Fonts: Inter, Montserrat, Outfit, Oswald, Poppins, Playfair Display or
  JetBrains Mono.
- On-screen text is at most a few large words. Keep everything inside the middle
  90% of the frame and keep the bottom 22% free of text: subtitles go there.
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


def build_document(blocks: Mapping[str, str], *, duration: int, width: int, height: int) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <script src="{GSAP_URL}"></script>
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


def _ask_claude(prompt: str, settings: Mapping[str, Any]) -> str:
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
        "--system-prompt", SYSTEM_PROMPT,
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
    duration: int,
    video_aspect: VideoAspect,
    output_dir: str,
    subject: str = "",
    script: str = "",
    variation: int = 0,
    settings: Mapping[str, Any] | None = None,
) -> MaterialInfo | None:
    """Author and render one clip. Returns ``None`` when the scene is skipped."""
    source = _settings(settings)
    duration = int(duration)
    if duration <= 0:
        raise ClaudeAnimationError("Claude animation clip duration must be a positive integer")
    width, height = VideoAspect(video_aspect).to_resolution()
    render_timeout = _bounded_float(
        source, "claude_animation_render_timeout", DEFAULT_RENDER_TIMEOUT_SECONDS, 30, 3600
    )
    attempts_allowed = max_attempts(source)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"claude-animation-{uuid.uuid4().hex[:12]}.mp4")

    feedback = ""
    for attempt in range(1, attempts_allowed + 1):
        try:
            reply = _ask_claude(
                build_prompt(
                    scene=search_term, duration=duration, width=width, height=height,
                    subject=subject, script=script,
                    style=_text(source, "claude_animation_style"),
                    variation=variation, feedback=feedback,
                ),
                source,
            )
            blocks = parse_scene(reply)
            with tempfile.TemporaryDirectory(prefix="mpt-animation-") as project_dir:
                Path(project_dir, "index.html").write_text(
                    build_document(blocks, duration=duration, width=width, height=height),
                    encoding="utf-8",
                )
                lint_errors = _lint_errors(project_dir, source, render_timeout)
                if lint_errors:
                    raise _SceneFailure("check", lint_errors)
                _render(project_dir, output_path, source, render_timeout)
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
            if action == RECOVERY_REPAIR and attempt < attempts_allowed:
                feedback = f"[{failure.stage}] {failure.detail}"
                continue
            return None
        return MaterialInfo(
            provider=PROVIDER_ID,
            url=output_path,
            duration=duration,
            source_info={
                "provider": PROVIDER_ID,
                "search_term": search_term,
                "rendition": {"width": width, "height": height},
            },
        )
    return None
