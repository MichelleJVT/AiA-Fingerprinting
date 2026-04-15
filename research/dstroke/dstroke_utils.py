"""
DStroke research prototype — shared utilities.

Implements the pieces of Fu et al. 2021 ("Fast Accurate and Automatic Brushstroke
Extraction", ACM TOMM) that are reused across the 01–05 notebooks:

- ABR brush extraction (wraps the external `abrupng` CLI)
- Painting synthesis from a source image + brush library (Algorithm 1)
- Phong-style height-field rendering
- Trapped-ball region growing for binary T(x) edge maps
- Guided filter (He et al. 2010, linear-time) for soft-matting
- Porter-Duff stroke ordering / coloring (Algorithm 2)
- SAM 2 mask-to-edge-map helper (lazy import)
- IoU/F1 metrics with small-dilation tolerance
- Matplotlib comparison-grid builder

Nothing in this module imports torch at module-load time; the training notebooks
import torch themselves and pass in tensors only where needed.
"""

from __future__ import annotations

import io
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology, segmentation, util


# --------------------------------------------------------------------------
# 1. ABR brush extraction
# --------------------------------------------------------------------------

def extract_abr_brushes(
    abr_paths: Sequence[Path | str],
    out_dir: Path | str,
    abrupng_exe: str = "abrupng",
) -> dict[str, int]:
    """Dump every brush inside each .abr into grayscale PNGs via `abrupng`.

    Layout produced:
        out_dir/{pack_stem}/0000.png, 0001.png, ...

    Returns a dict mapping pack_stem -> count of PNGs produced.

    If a pack's directory already exists and is non-empty, it is skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for abr_path in abr_paths:
        abr_path = Path(abr_path)
        pack_dir = out_dir / abr_path.stem
        if pack_dir.exists() and any(pack_dir.iterdir()):
            counts[abr_path.stem] = sum(1 for _ in pack_dir.glob("*.png"))
            print(f"[skip] {abr_path.name}: {pack_dir} already populated "
                  f"({counts[abr_path.stem]} PNGs)")
            continue

        pack_dir.mkdir(parents=True, exist_ok=True)
        cmd = [abrupng_exe, "-o", str(pack_dir), str(abr_path)]
        print(f"[run ] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        counts[abr_path.stem] = sum(1 for _ in pack_dir.glob("*.png"))
        print(f"[done] {abr_path.name}: {counts[abr_path.stem]} PNGs")

    return counts


# --------------------------------------------------------------------------
# 2. Brush library
# --------------------------------------------------------------------------

@dataclass
class Brush:
    """One brushstroke sample: alpha map (acts as height map per §3.1)."""
    alpha: np.ndarray    # (H, W) float32 in [0, 1]
    source: str          # filename for provenance

    @property
    def height(self) -> int:
        return self.alpha.shape[0]

    @property
    def width(self) -> int:
        return self.alpha.shape[1]


def load_brush_library(
    curated_dir: Path | str,
    min_side: int = 24,
    max_side: int = 512,
) -> list[Brush]:
    """Load all grayscale PNGs under `curated_dir` as `Brush` entries.

    Each PNG is interpreted as an alpha / height map directly (background = 0,
    stroke body = 255). PNGs outside the [min_side, max_side] pixel range are
    dropped — tiny stamps and very large full-canvas brushes both make the
    synthesis loop misbehave.
    """
    curated_dir = Path(curated_dir)
    brushes: list[Brush] = []
    for png in sorted(curated_dir.rglob("*.png")):
        arr = np.array(Image.open(png).convert("L"), dtype=np.float32) / 255.0
        h, w = arr.shape
        if min(h, w) < min_side or max(h, w) > max_side:
            continue
        brushes.append(Brush(alpha=arr, source=str(png.relative_to(curated_dir))))
    if not brushes:
        raise RuntimeError(
            f"No usable PNG brushes under {curated_dir!r}. "
            "Did you curate some after running extract_abr_brushes?"
        )
    return brushes


# --------------------------------------------------------------------------
# 3. Painting synthesis (Algorithm 1)
# --------------------------------------------------------------------------

def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _stamp_brush(
    canvas: np.ndarray,
    edge_map: np.ndarray,
    height_field: np.ndarray,
    brush: Brush,
    cy: int,
    cx: int,
    color_rgb: np.ndarray,
    angle_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite `brush` at (cy, cx) on `canvas`. Returns new (canvas, edge, height)."""
    # rotate + scale the brush alpha
    pil = Image.fromarray((brush.alpha * 255).astype(np.uint8))
    new_h = max(4, int(brush.height * scale))
    new_w = max(4, int(brush.width * scale))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    pil = pil.rotate(angle_deg, expand=True, resample=Image.BILINEAR)
    alpha = np.array(pil, dtype=np.float32) / 255.0
    bh, bw = alpha.shape

    # canvas region to paste into
    y0 = cy - bh // 2
    x0 = cx - bw // 2
    y1 = y0 + bh
    x1 = x0 + bw

    cy0 = max(0, y0); cx0 = max(0, x0)
    cy1 = min(canvas.shape[0], y1); cx1 = min(canvas.shape[1], x1)
    if cy1 <= cy0 or cx1 <= cx0:
        return canvas, edge_map, height_field

    by0 = cy0 - y0; bx0 = cx0 - x0
    by1 = by0 + (cy1 - cy0); bx1 = bx0 + (cx1 - cx0)
    a = alpha[by0:by1, bx0:bx1][..., None]  # (h, w, 1)

    # alpha composite color
    canvas_patch = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = a * color_rgb[None, None, :] + (1.0 - a) * canvas_patch

    # accumulate height field
    height_field[cy0:cy1, cx0:cx1] = np.maximum(
        height_field[cy0:cy1, cx0:cx1], alpha[by0:by1, bx0:bx1]
    )

    # edge map: boundary of the stroke (dilation-of-alpha xor alpha>threshold).
    # This is paper §3.1 "boundaries are cumulated to form the edge map".
    mask = alpha[by0:by1, bx0:bx1] > 0.2
    if mask.any():
        dil = ndi.binary_dilation(mask, iterations=1)
        boundary = dil ^ mask
        # add the new boundary
        edge_map[cy0:cy1, cx0:cx1] = np.maximum(edge_map[cy0:cy1, cx0:cx1], boundary.astype(np.float32))
        # erase any old edge that fell INSIDE this new stroke
        # ("Remove overlapped edge from E" in Algorithm 1)
        edge_map[cy0:cy1, cx0:cx1][mask] = 0.0

    return canvas, edge_map, height_field


def synthesize_painting(
    source_rgb: np.ndarray,
    brushes: list[Brush],
    target_h: int = 256,
    target_w: int = 256,
    max_strokes: int = 800,
    ratio_threshold: float = 0.97,
    stroke_scale_range: tuple[float, float] = (0.3, 1.0),
    seed: int | None = None,
    phong_light: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper Algorithm 1.

    Iteratively paints strokes onto a blank canvas to approximate `source_rgb`.
    At each iteration we:
      1. Find the pixel of largest |source - canvas| difference → ROI centre.
      2. Pick a random brush; orient it roughly perpendicular to the local
         gradient of the source (edges run along tangent direction).
      3. Colour it with the mean of `source_rgb` in the ROI.
      4. Composite it, update the edge map, and accept iff the overall L1
         error between canvas and source decreases (trial-and-error per paper).

    Returns (painting_rgb, edge_map, height_field), all in [0, 1] float32.
    """
    rng = np.random.default_rng(seed)
    src = _resize_to(source_rgb, target_h, target_w)
    if src.ndim == 2:
        src = np.stack([src]*3, axis=-1)
    elif src.shape[-1] == 4:
        src = src[..., :3]
    src = np.clip(src.astype(np.float32), 0, 1)

    canvas = np.ones_like(src)  # start on white canvas
    edge_map = np.zeros(src.shape[:2], dtype=np.float32)
    height_field = np.zeros(src.shape[:2], dtype=np.float32)

    # Precompute source gradient (for perpendicular brush orientation).
    gray = color.rgb2gray(src)
    gy = ndi.sobel(gray, axis=0)
    gx = ndi.sobel(gray, axis=1)

    diff = np.abs(src - canvas).sum(axis=-1)  # (H, W)
    accepted = 0
    for step in range(max_strokes):
        # fraction of canvas that has been painted at all (non-white)
        painted_ratio = float((height_field > 0.01).mean())
        if painted_ratio >= ratio_threshold:
            break

        # ROI: pixel with biggest remaining difference
        idx = int(np.argmax(diff))
        cy, cx = divmod(idx, diff.shape[1])

        # local neighbourhood colour
        r = 8
        y0 = max(0, cy - r); y1 = min(src.shape[0], cy + r)
        x0 = max(0, cx - r); x1 = min(src.shape[1], cx + r)
        color_rgb = src[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

        # orient brush perpendicular to local gradient
        gradient_angle = np.degrees(np.arctan2(gy[cy, cx], gx[cy, cx]))
        brush_angle = gradient_angle + 90.0 + rng.normal(0, 10)

        brush = brushes[rng.integers(0, len(brushes))]
        scale = rng.uniform(*stroke_scale_range)

        # tentative paint
        cand_canvas = canvas.copy()
        cand_edge = edge_map.copy()
        cand_height = height_field.copy()
        cand_canvas, cand_edge, cand_height = _stamp_brush(
            cand_canvas, cand_edge, cand_height, brush,
            cy, cx, color_rgb, brush_angle, scale,
        )

        new_diff = np.abs(src - cand_canvas).sum(axis=-1)
        if new_diff.sum() < diff.sum():
            canvas, edge_map, height_field = cand_canvas, cand_edge, cand_height
            diff = new_diff
            accepted += 1

    if phong_light and height_field.max() > 0:
        canvas = render_height_field(canvas, height_field)

    return canvas.astype(np.float32), edge_map.astype(np.float32), height_field.astype(np.float32)


# --------------------------------------------------------------------------
# 4. Phong-style height-field shading
# --------------------------------------------------------------------------

def render_height_field(
    albedo_rgb: np.ndarray,
    height: np.ndarray,
    light_dir: tuple[float, float, float] = (-0.4, -0.4, 1.0),
    ambient: float = 0.55,
    diffuse: float = 0.55,
    specular: float = 0.25,
    shininess: float = 24.0,
) -> np.ndarray:
    """Shade `albedo_rgb` with Phong over a relief `height` field."""
    # normals from gradient
    hy, hx = np.gradient(height * 8.0)  # exaggerate relief
    normals = np.stack([-hx, -hy, np.ones_like(height)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8

    ld = np.array(light_dir, dtype=np.float32)
    ld = ld / (np.linalg.norm(ld) + 1e-8)
    ndotl = np.clip(normals @ ld, 0, 1)

    # blinn-phong specular
    view = np.array([0, 0, 1], dtype=np.float32)
    halfway = (ld + view) / np.linalg.norm(ld + view)
    spec = np.clip(normals @ halfway, 0, 1) ** shininess

    shaded = albedo_rgb * (ambient + diffuse * ndotl[..., None]) \
        + specular * spec[..., None]
    return np.clip(shaded, 0, 1)


# --------------------------------------------------------------------------
# 5. Trapped-ball region growing → binary T(x)
# --------------------------------------------------------------------------

def trapped_ball_merge(
    intensity_edge: np.ndarray,
    ball_radius: int = 3,
    edge_threshold: float = 0.3,
) -> np.ndarray:
    """Closed-boundary binary edge map from an intensity edge map.

    Simplified trapped-ball [Zhang et al. 2009]:
      1. Threshold the intensity edge map.
      2. Dilate with a `ball_radius` disk (closes small gaps).
      3. Watershed the complement using local minima as seeds.
      4. Emit the watershed boundaries as the final T(x).

    This is the paper's T(x) post-processing in §3.2.
    """
    eb = intensity_edge >= edge_threshold
    if ball_radius > 0:
        eb = morphology.closing(eb, morphology.disk(ball_radius))

    # region seeds = connected components of the un-edged area
    interior = ~eb
    seeds, n = ndi.label(interior)
    if n == 0:
        return eb.astype(np.uint8)

    # watershed over the distance transform of the interior
    dist = ndi.distance_transform_edt(interior)
    labels = segmentation.watershed(-dist, markers=seeds, mask=np.ones_like(interior))
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    return boundaries.astype(np.uint8)


# --------------------------------------------------------------------------
# 6. Guided filter (He et al. 2010)
# --------------------------------------------------------------------------

def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Simple separable box filter, constant border."""
    kernel_size = 2 * r + 1
    return ndi.uniform_filter(img, size=kernel_size, mode="nearest")


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 1e-4,
) -> np.ndarray:
    """O(n) guided filter; `guide` can be RGB or gray. `src` is 2-D."""
    if guide.ndim == 3:
        g = color.rgb2gray(guide).astype(np.float32)
    else:
        g = guide.astype(np.float32)
    p = src.astype(np.float32)

    mean_g = _box_filter(g, radius)
    mean_p = _box_filter(p, radius)
    corr_gp = _box_filter(g * p, radius)
    var_g = _box_filter(g * g, radius) - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    return (mean_a * g + mean_b).astype(np.float32)


# --------------------------------------------------------------------------
# 7. Porter-Duff stroke ordering / coloring (Algorithm 2)
# --------------------------------------------------------------------------

def _solve_porter_duff(
    observed_rgb: np.ndarray,   # (N, 3)
    alpha_A: np.ndarray,        # (N,)
    alpha_B: np.ndarray,        # (N,)
) -> np.ndarray:
    """Solve (A over B) Equation (8) as a least-squares system for (A_rgb, B_rgb).

    (A over B)_rgb = (α_A A_rgb + (1-α_A) α_B B_rgb) / (α_A + (1-α_A) α_B)

    With N pixels we get 3N equations in 6 unknowns (one 3-vector per stroke).
    Each channel is solved independently.
    """
    denom = alpha_A + (1 - alpha_A) * alpha_B                 # (N,)
    denom = np.clip(denom, 1e-6, None)
    wA = alpha_A / denom                                      # (N,)
    wB = ((1 - alpha_A) * alpha_B) / denom                    # (N,)

    colors = np.zeros((2, 3), dtype=np.float32)
    for ch in range(3):
        M = np.stack([wA, wB], axis=1)  # (N, 2)
        y = observed_rgb[:, ch]
        sol, *_ = np.linalg.lstsq(M, y, rcond=None)
        colors[:, ch] = sol
    return np.clip(colors, 0, 1)  # rows = [A_rgb, B_rgb]


def porter_duff_ordering(
    painting_rgb: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    alpha_A: np.ndarray,
    alpha_B: np.ndarray,
) -> dict:
    """Paper Algorithm 2.

    Given two (possibly overlapping) brushstroke masks and their alpha mattes,
    figure out which one sits on top and what each one's pigment RGB is.
    """
    union = mask_A | mask_B
    pts = np.argwhere(union)
    if len(pts) < 6:
        raise ValueError("Overlap region too small for a stable fit.")

    obs = painting_rgb[pts[:, 0], pts[:, 1]]
    aA = alpha_A[pts[:, 0], pts[:, 1]]
    aB = alpha_B[pts[:, 0], pts[:, 1]]

    # Scenario 1: A over B
    c_AB = _solve_porter_duff(obs, aA, aB)
    A1, B1 = c_AB[0], c_AB[1]
    denom1 = np.clip(aA + (1 - aA) * aB, 1e-6, None)
    recon1 = (aA[:, None] * A1 + ((1 - aA) * aB)[:, None] * B1) / denom1[:, None]
    err1 = float(((recon1 - obs) ** 2).mean())

    # Scenario 2: B over A
    c_BA = _solve_porter_duff(obs, aB, aA)
    B2, A2 = c_BA[0], c_BA[1]
    denom2 = np.clip(aB + (1 - aB) * aA, 1e-6, None)
    recon2 = (aB[:, None] * B2 + ((1 - aB) * aA)[:, None] * A2) / denom2[:, None]
    err2 = float(((recon2 - obs) ** 2).mean())

    if err1 <= err2:
        return {"order": "A_over_B", "color_A": A1, "color_B": B1,
                "residual_correct": err1, "residual_reversed": err2}
    else:
        return {"order": "B_over_A", "color_A": A2, "color_B": B2,
                "residual_correct": err2, "residual_reversed": err1}


# --------------------------------------------------------------------------
# 8. SAM 2 helper (Strategy C) — lazy import
# --------------------------------------------------------------------------

def run_sam2_automask(
    image_rgb: np.ndarray,
    mask_generator,
    min_area: int = 25,
) -> tuple[np.ndarray, list[dict]]:
    """Run SAM 2's automatic mask generator and return (edge_map, masks).

    `mask_generator` is an instance of `sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator`
    built by the calling notebook (keeps this module torch-free).
    """
    img_u8 = (image_rgb * 255).astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb
    masks = mask_generator.generate(img_u8)
    edge = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    keep = []
    for m in masks:
        seg = m["segmentation"]
        if seg.sum() < min_area:
            continue
        boundary = segmentation.find_boundaries(seg, mode="outer")
        edge = np.maximum(edge, boundary.astype(np.float32))
        keep.append(m)
    return edge, keep


# --------------------------------------------------------------------------
# 9. Metrics
# --------------------------------------------------------------------------

def compute_edge_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: int = 2,
) -> dict:
    """Binary edge IoU/F1/precision/recall with `tolerance` pixels of slack.

    We dilate `gt` by `tolerance` for the precision numerator and dilate `pred`
    by `tolerance` for the recall numerator — standard BSDS-style fair eval.
    """
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    selem = morphology.disk(tolerance)

    gt_thick = morphology.dilation(gt_b, selem)
    pred_thick = morphology.dilation(pred_b, selem)

    tp_p = np.logical_and(pred_b, gt_thick).sum()
    tp_r = np.logical_and(gt_b, pred_thick).sum()
    fp = pred_b.sum() - tp_p
    fn = gt_b.sum() - tp_r

    precision = tp_p / max(tp_p + fp, 1)
    recall = tp_r / max(tp_r + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = np.logical_and(pred_b, gt_b).sum() / max(np.logical_or(pred_b, gt_b).sum(), 1)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


# --------------------------------------------------------------------------
# 10. Matplotlib comparison grid
# --------------------------------------------------------------------------

def build_comparison_grid(
    rows: list[dict],
    save_path: Path | str | None = None,
    col_order: Sequence[str] = ("input", "A", "B", "C", "D", "gt"),
    col_titles: dict[str, str] | None = None,
) -> "matplotlib.figure.Figure":  # noqa: F821
    """Build a grid: one row per painting, one column per strategy.

    `rows` is a list of dicts like:
        {"input": rgb_array, "A": edge_map, "B": edge_map, ..., "gt": optional}
    Missing columns are skipped (drawn as blank).
    """
    import matplotlib.pyplot as plt

    titles = {
        "input": "Input",
        "A": "A: Pix2Pix ft",
        "B": "B: Pix2Pix scratch",
        "C": "C: SAM 2",
        "D": "D: SegFormer",
        "gt": "Ground truth",
    }
    if col_titles:
        titles.update(col_titles)

    present_cols = [c for c in col_order if any(c in r for r in rows)]

    fig, axes = plt.subplots(
        nrows=len(rows), ncols=len(present_cols),
        figsize=(2.2 * len(present_cols), 2.2 * len(rows)),
        squeeze=False,
    )
    for i, row in enumerate(rows):
        for j, c in enumerate(present_cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles.get(c, c), fontsize=9)
            img = row.get(c)
            if img is None:
                ax.set_facecolor("#eeeeee")
                continue
            if img.ndim == 2:
                ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
