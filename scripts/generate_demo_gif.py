"""Render the CPU demo as a terminal-style animated GIF.

Runs examples/countdown/demo.py, captures its output and renders a
typewriter-style animation: the happy-path ALLOW in green, fault rejects
in red, quarantine in yellow.  Output: docs/site/assets/demo.gif

Requires: uv run --with pillow python scripts/generate_demo_gif.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "site" / "assets" / "demo.gif"

BG = (15, 23, 42)
FG = (226, 232, 240)
ACCENT = (138, 180, 248)
GREEN = (52, 211, 153)
RED = (248, 113, 113)
YELLOW = (251, 191, 36)
DIM = (148, 163, 184)

FONT_PATH = r"C:\Windows\Fonts\consola.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def colorize(line: str, font) -> tuple[str, tuple]:
    lower = line.lower()
    if "decision: allow" in lower:
        return line.replace("allow", "allow", 1), GREEN
    if "reject" in lower:
        return line, RED
    if "quarantine" in lower:
        return line, YELLOW
    if line.startswith("==") or line.startswith("---"):
        return line, DIM
    if "[" in line and line.strip().startswith("["):
        return line, ACCENT
    if line.startswith("GRPO-Guard demo") or line.startswith("All decisions"):
        return line, FG
    return line, FG


def main() -> int:
    font = _font(17)
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "countdown" / "demo.py")],
        capture_output=True, text=True, cwd=REPO)
    lines = [l for l in result.stdout.splitlines() if l.strip()]

    # measure the widest line
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    widths = [draw.textlength(l, font=font) for l in lines]
    W = max(int(max(widths)) + 48, 860)
    H = len(lines) * 26 + 56
    pad = 24

    frames = []
    n_frames = max(24, len(lines) * 3)
    for f in range(n_frames):
        visible = min(len(lines), f // 3 + 1)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((pad, 18), "GRPO-Guard demo — watch the fault get rejected", font=_font(20), fill=ACCENT)
        for i in range(visible):
            line, color = colorize(lines[i], font)
            d.text((pad, 50 + i * 26), line, font=font, fill=color)
        if visible < len(lines):
            d.rectangle((pad, 50 + visible * 26 - 2, pad + 14, 50 + visible * 26 + 12), fill=RED)
        frames.append(img)
    for _ in range(10):  # hold the final frame
        frames.append(frames[-1])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=130, loop=0, optimize=True)
    print(f"wrote {OUT} ({len(frames)} frames, {W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
