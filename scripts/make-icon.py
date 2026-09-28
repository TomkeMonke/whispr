"""Draws the whispr app icon (the titlebar brand mark) as a 1024 px PNG.

The tile is an indigo gradient rounded square with the white waveform glyph
from index.html. Regenerate every platform size from it with:

    python scripts/make-icon.py
    npx tauri icon src-tauri/icons/source.png
"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
SS = 4  # supersampling factor, for smooth edges after the downscale
OUT = Path(__file__).resolve().parent.parent / "src-tauri" / "icons" / "source.png"

# Gradient stops, top-left to bottom-right (same as .brand-mark in styles.css).
STOPS = [(0.0, (0x7C, 0x86, 0xEC)), (0.55, (0x5E, 0x6A, 0xD2)), (1.0, (0x4A, 0x55, 0xB8))]

# The lucide "audio-lines" glyph on its 24-unit grid: (x, y_top, y_bottom).
LINES = [(2, 10, 13), (6, 6, 17), (10, 3, 21), (14, 8, 15), (18, 5, 18), (22, 10, 13)]
STROKE = 2.4


def gradient_color(t: float) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return STOPS[-1][1]


def main() -> None:
    s = SIZE * SS
    margin = 56 * SS  # breathing room so the tile sits like other Windows icons
    radius = 216 * SS

    # Diagonal gradient, then clip it to the rounded tile.
    # Drawn small and scaled up: a smooth gradient loses nothing that way.
    small = 256
    grad = Image.new("RGB", (small, small))
    px = grad.load()
    for y in range(small):
        for x in range(small):
            px[x, y] = gradient_color((x + y) / (2 * (small - 1)))
    grad = grad.resize((s, s), Image.BILINEAR)

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (margin, margin, s - margin, s - margin), radius=radius, fill=255
    )
    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    icon.paste(grad, (0, 0), mask)

    # Waveform: the 24-unit glyph scaled to 60% of the tile.
    glyph = (s - 2 * margin) * 0.6
    unit = glyph / 24
    ox = (s - glyph) / 2
    oy = (s - glyph) / 2
    width = STROKE * unit
    draw = ImageDraw.Draw(icon)
    for x, y0, y1 in LINES:
        cx = ox + x * unit
        draw.rounded_rectangle(
            (cx - width / 2, oy + y0 * unit - width / 2, cx + width / 2, oy + y1 * unit + width / 2),
            radius=width / 2,
            fill=(255, 255, 255, 255),
        )

    icon.resize((SIZE, SIZE), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
