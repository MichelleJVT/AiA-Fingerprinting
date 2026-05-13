"""
Visual sanity check for the synthetic brushstroke dataset (Step 1A.2).

Writes one 4-column preview grid per tool type to the project root:
    preview_round_brush.png
    preview_flat_brush.png
    preview_palette_knife.png
    preview_fan_brush.png

Acceptance criteria (Step 1A.2):
  - Round vs flat: edge softness clearly differs
  - Palette knife: visibly sharper edges and obvious ridge highlights
  - Fan brush: obvious parallel striations
  - All four types are mutually distinguishable to a non-expert in under 3 seconds

If a class fails, edit its TOOLS dict in aiai/data/synthesise_strokes.py and regenerate.
"""
import random
from pathlib import Path
from PIL import Image

ROOT = Path("data/reference_strokes_v0_synthetic")

if not ROOT.exists():
    raise SystemExit(
        f"Dataset not found at {ROOT}. "
        "Run: python -m aiai.data.synthesise_strokes"
    )

for tool_dir in sorted(ROOT.iterdir()):
    if not tool_dir.is_dir():
        continue
    samples = list(tool_dir.glob("*.png"))
    if len(samples) < 4:
        print(f"  WARNING: only {len(samples)} samples for {tool_dir.name}, skipping")
        continue
    grid = Image.new("RGB", (512 * 4, 512), "white")
    for i, p in enumerate(random.sample(samples, 4)):
        grid.paste(Image.open(p).resize((512, 512)), (i * 512, 0))
    out = Path(f"preview_{tool_dir.name}.png")
    grid.save(out)
    print(f"  Written: {out}")

print("Preview grids written. Inspect each preview_*.png.")
