"""Named looks for Claude Code animation.

A style is one TOML file: a name, the rules of the look, and one worked example
scene. Built-in styles ship in ``resource/animation_styles``; files in
``storage/animation_styles`` add to them and replace a built-in with the same
file name, so a look can be added or changed without touching code.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass

from loguru import logger

from app.utils import utils

MAX_DIRECTION_LENGTH = 2000
MAX_EXAMPLE_LENGTH = 6000
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


@dataclass(frozen=True)
class AnimationStyle:
    slug: str
    name: str
    summary: str
    direction: str
    example: str = ""


def _directories() -> list[str]:
    return [
        utils.resource_dir("animation_styles"),
        utils.storage_dir("animation_styles"),
    ]


def _load(path: str, slug: str) -> AnimationStyle | None:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(f"animation style {slug!r} could not be read: {exc}")
        return None
    direction = str(data.get("direction") or "").strip()
    if not direction:
        logger.warning(f"animation style {slug!r} has no direction and is ignored")
        return None
    return AnimationStyle(
        slug=slug,
        name=str(data.get("name") or slug).strip()[:80],
        summary=str(data.get("summary") or "").strip()[:300],
        direction=direction[:MAX_DIRECTION_LENGTH],
        example=str(data.get("example") or "").strip()[:MAX_EXAMPLE_LENGTH],
    )


def list_styles() -> list[AnimationStyle]:
    """Every usable style, ordered by name; later directories win on a slug clash."""
    styles: dict[str, AnimationStyle] = {}
    for directory in _directories():
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for file_name in names:
            slug, extension = os.path.splitext(file_name)
            if extension != ".toml" or not _SLUG_RE.match(slug):
                continue
            style = _load(os.path.join(directory, file_name), slug)
            if style is not None:
                styles[slug] = style
    return sorted(styles.values(), key=lambda style: style.name.lower())


def get_style(slug: str) -> AnimationStyle | None:
    slug = str(slug or "").strip()
    if not slug:
        return None
    return next((style for style in list_styles() if style.slug == slug), None)
