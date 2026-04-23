"""
Pictology pipeline — LAM van Pascal images.

Runs the full PictologyPipeline on all images in the LAM van Pascal folder
and writes per-function and per-image results to outputs/pictology_lam_pascal/.
"""

import sys, os, time, json, csv
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

# ── Import every analysis function individually ──
from src.pictology.spontaneity import (
    line_spontaneity_score, contrast_analysis, detail_density_analysis
)
from src.pictology.brushstroke import (
    extract_brushstrokes, hatching_analysis, pressure_variation
)
from src.pictology.surface_color import (
    surface_organization, color_analysis, light_dark_spatial, texture_rendering
)
from src.pictology.contour_rhythm import (
    contour_interior_relationship, rhythm_analysis, totality_score
)
from src.pictology.construction import (
    sequence_of_development, major_minor_analysis
)
from src.pictology.pipeline import PictologyPipeline, _prefix

# ── Configuration ──
DATA_DIR = Path(r"C:\Users\MichelleJacobs\OneDrive - Jongens van Techniek\JVT _ Engineering - Authentication in Art\AIA - AI Tool - Fingerprinting\Ontvangen (Confidential)\LAM van Pascal")
OUT_DIR  = ROOT / "outputs" / "pictology_lam_pascal"
MAX_SIZE = 512          # keep images manageable for speed
EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Discover images ──
image_paths = sorted(
    p for p in DATA_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
)
print(f"Found {len(image_paths)} LAM van Pascal images in {DATA_DIR}")

# ── Helper: load & resize ──
pipe = PictologyPipeline(max_size=MAX_SIZE)

def load(path):
    img_rgb = pipe.load_image(path)
    gray = pipe.to_gray(img_rgb)
    return img_rgb, gray

# ── Define the analysis steps (name → callable, input type) ──
ANALYSES = [
    ("surface_organization",          "surface",  lambda rgb, g: surface_organization(rgb)),
    ("pressure_variation",            "pressure", lambda rgb, g: pressure_variation(g)),
    ("color_analysis",                "color",    lambda rgb, g: color_analysis(rgb)),
    ("sequence_of_development",       "sequence", lambda rgb, g: sequence_of_development(g)),
    ("extract_brushstrokes",          "stroke",   lambda rgb, g: extract_brushstrokes(g)),
    ("light_dark_spatial",            "lightdark",lambda rgb, g: light_dark_spatial(rgb)),
    ("contrast_analysis",             "contrast", lambda rgb, g: contrast_analysis(rgb)),
    ("texture_rendering",             "texture",  lambda rgb, g: texture_rendering(g)),
    ("contour_interior_relationship", "contour",  lambda rgb, g: contour_interior_relationship(g)),
    ("major_minor_analysis",          "majmin",   lambda rgb, g: major_minor_analysis(rgb)),
    ("detail_density_analysis",       "detail",   lambda rgb, g: detail_density_analysis(g)),
    ("rhythm_analysis",               "rhythm",   lambda rgb, g: rhythm_analysis(g)),
    ("hatching_analysis",             "hatch",    lambda rgb, g: hatching_analysis(g)),
    ("line_spontaneity_score",        "spont",    lambda rgb, g: line_spontaneity_score(g)),
]

# ── Run ──
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-function results: { func_name: [ {filename, ...features} ] }
func_results = {name: [] for name, _, _ in ANALYSES}
# Full feature dicts for totality
full_features_per_image = {}
timings = {name: [] for name, _, _ in ANALYSES}

for idx, img_path in enumerate(image_paths):
    fname = img_path.name
    print(f"\n[{idx+1}/{len(image_paths)}] {fname}")
    img_rgb, gray = load(img_path)

    all_features = {}

    for func_name, prefix, fn in ANALYSES:
        t0 = time.perf_counter()
        try:
            raw = fn(img_rgb, gray)
        except Exception as e:
            print(f"  ✗ {func_name}: {e}")
            raw = {}
        elapsed = time.perf_counter() - t0
        timings[func_name].append(elapsed)

        # Prefix and flatten
        prefixed = _prefix(prefix, raw)
        all_features.update(prefixed)

        # Store raw (un-prefixed) for the per-function CSV
        row = {"filename": fname}
        for k, v in raw.items():
            if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                row[k] = round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            elif isinstance(v, list):
                for i, val in enumerate(v):
                    if isinstance(val, (int, float, list)):
                        if isinstance(val, list):
                            for j, vv in enumerate(val):
                                row[f"{k}_{i}_{j}"] = round(float(vv), 6) if isinstance(vv, float) else vv
                        else:
                            row[f"{k}_{i}"] = round(float(val), 6) if isinstance(val, float) else val
            elif isinstance(v, str):
                row[k] = v
        func_results[func_name].append(row)

        status = "✓" if raw else "∅"
        print(f"  {status} {func_name:<35s} {elapsed:.3f}s  ({len(raw)} keys)")

    # Totality (needs full feature dict)
    tot = totality_score(all_features)
    all_features["totality_score"] = tot
    full_features_per_image[fname] = all_features
    print(f"  ✓ {'totality_score':<35s}         → {tot:.4f}")

# ── Write per-function CSVs ──
print(f"\n{'='*60}")
print(f"Writing results to {OUT_DIR}/")
for func_name, rows in func_results.items():
    if not rows:
        continue
    cols = ["filename"] + sorted(set(k for r in rows for k in r if k != "filename"))
    out_path = OUT_DIR / f"{func_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {out_path.name:<45s} {len(rows)} rows × {len(cols)} cols")

# ── Write combined feature matrix ──
if full_features_per_image:
    all_keys = sorted(set(k for feats in full_features_per_image.values() for k in feats
                          if isinstance(feats[k], (int, float, bool, np.integer, np.floating, np.bool_))))
    cols = ["filename"] + all_keys
    out_path = OUT_DIR / "all_features.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for fname, feats in full_features_per_image.items():
            row = {"filename": fname}
            for k in all_keys:
                v = feats.get(k)
                if isinstance(v, (float, np.floating)):
                    row[k] = round(float(v), 6)
                elif v is not None:
                    row[k] = v
            w.writerow(row)
    print(f"  {'all_features.csv':<45s} {len(full_features_per_image)} rows × {len(cols)} cols")

# ── Write timing summary ──
out_path = OUT_DIR / "timings.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["function", "mean_s", "std_s", "min_s", "max_s", "total_s"])
    for func_name, times in timings.items():
        if times:
            arr = np.array(times)
            w.writerow([func_name, f"{arr.mean():.3f}", f"{arr.std():.3f}",
                        f"{arr.min():.3f}", f"{arr.max():.3f}", f"{arr.sum():.1f}"])
print(f"  {'timings.csv':<45s} {len(timings)} functions")

# ── Summary stats ──
print(f"\n{'='*60}")
total_time = sum(sum(t) for t in timings.values())
print(f"Total processing time: {total_time:.1f}s for {len(image_paths)} images")
print(f"Avg per image: {total_time/max(1,len(image_paths)):.1f}s")
print(f"\nOutput files in {OUT_DIR}/:")
for p in sorted(OUT_DIR.iterdir()):
    size = p.stat().st_size
    print(f"  {p.name:<45s} {size:>8,} bytes")
