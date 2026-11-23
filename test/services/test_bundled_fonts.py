"""Every bundled subtitle font must load and ship with its licence."""

from pathlib import Path

from PIL import ImageFont
import pytest

FONTS = Path(__file__).parents[2] / "resource" / "fonts"
# Fonts that were bundled before licence files were kept beside the fonts.
LEGACY = ("BeVietnamPro", "Charm", "MicrosoftYaHei", "STHeiti", "UTM Kabel KT")
FONT_FILES = sorted(
    path for path in FONTS.iterdir() if path.suffix in {".ttf", ".ttc"}
)


@pytest.mark.parametrize("font", FONT_FILES, ids=lambda path: path.name)
def test_bundled_font_draws_latin_text(font):
    face = ImageFont.truetype(str(font), 60)
    assert face.getmask("Viral 100%").getbbox() is not None


@pytest.mark.parametrize(
    "font",
    [path for path in FONT_FILES if not path.name.startswith(LEGACY)],
    ids=lambda path: path.name,
)
def test_added_font_ships_with_its_licence(font):
    family = font.stem.split("-")[0].lower()
    licences = [path.name for path in (FONTS / "licenses").glob(f"{family}-*.txt")]
    assert licences, f"no licence file for {font.name}"
