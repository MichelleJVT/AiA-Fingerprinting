"""
Spontaneity & line analysis — van Dantzig Chapter I.

Implements computational proxies for:
- Spontaneous vs inhibited line detection (speed, pressure, uniformity)
- Contrast analysis (light/dark, warm/cool)
- Detail density in major vs minor parts
"""

import numpy as np
from scipy import ndimage
from skimage import feature, morphology, filters


def extract_lines(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Detect edges/lines using Canny; returns binary edge map."""
    return feature.canny(gray, sigma=sigma).astype(np.float32)


def line_spontaneity_score(gray: np.ndarray, patch_size: int = 64) -> dict:
    """
    Measure spontaneity of lines in patches across the image.

    Van Dantzig's spontaneous line (Fig. 3) has:
      - Gradual tone variation (beginning → middle → end)
      - Smooth width transitions
      - No "crumbling" (repeated re-starts)

    An inhibited line (Figs. 4-5) has:
      - Coarser strokes
      - Irregular middle section with re-starts
      - Disturbed tone relations

    We approximate this by measuring:
      1. Edge curvature smoothness (spontaneous → smooth)
      2. Stroke width consistency (spontaneous → gradual change)
      3. Junction density (inhibited → more junctions from re-starts)
    """
    edges = extract_lines(gray)

    # Skeleton = thinned line representation
    skel = morphology.skeletonize(edges > 0)

    h, w = gray.shape
    n_patches_y = max(1, h // patch_size)
    n_patches_x = max(1, w // patch_size)

    curvature_scores = []
    junction_densities = []
    width_consistencies = []

    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            y0, y1 = iy * patch_size, min((iy + 1) * patch_size, h)
            x0, x1 = ix * patch_size, min((ix + 1) * patch_size, w)

            patch_skel = skel[y0:y1, x0:x1]
            patch_edges = edges[y0:y1, x0:x1]

            if patch_skel.sum() < 10:
                continue

            # Curvature smoothness via Laplacian of skeleton
            laplacian = ndimage.laplace(patch_skel.astype(np.float32))
            curvature_scores.append(np.std(laplacian[patch_skel]))

            # Junction density: pixels with >2 skeleton neighbors
            kernel = np.ones((3, 3))
            kernel[1, 1] = 0
            neighbor_count = ndimage.convolve(
                patch_skel.astype(np.float32), kernel, mode="constant"
            )
            junctions = (neighbor_count > 2) & patch_skel
            area = max(1, patch_skel.sum())
            junction_densities.append(junctions.sum() / area)

            # Width consistency: distance transform on edges
            if patch_edges.sum() > 0:
                dist = ndimage.distance_transform_edt(~(patch_edges > 0))
                on_skel = dist[patch_skel]
                if len(on_skel) > 1:
                    width_consistencies.append(np.std(on_skel) / max(1e-8, np.mean(on_skel)))

    return {
        "curvature_smoothness": float(np.mean(curvature_scores)) if curvature_scores else 0.0,
        "junction_density": float(np.mean(junction_densities)) if junction_densities else 0.0,
        "width_consistency_cv": float(np.mean(width_consistencies)) if width_consistencies else 0.0,
        # Higher = more spontaneous (fewer junctions, smoother curvature, consistent width)
        "spontaneity_index": _spontaneity_index(
            curvature_scores, junction_densities, width_consistencies
        ),
    }


def _spontaneity_index(curvatures, junctions, widths) -> float:
    """
    Composite score: 0 = maximally inhibited, 1 = maximally spontaneous.
    """
    if not curvatures:
        return 0.0
    c = 1.0 - min(1.0, np.mean(curvatures) / 0.5)   # smooth curvature → high
    j = 1.0 - min(1.0, np.mean(junctions) / 0.15)    # few junctions → high
    w = 1.0 - min(1.0, np.mean(widths) / 1.0)         # consistent width → high
    return float(np.clip((c + j + w) / 3.0, 0.0, 1.0))


def contrast_analysis(img_rgb: np.ndarray) -> dict:
    """
    Van Dantzig Ch. I & III: Contrast characteristics.

    Great masters:
      - Strong light/dark contrasts in MAIN parts
      - Warm colors foreground, cool background
      - Contrast logically connected to content

    Inhibited painters:
      - Weak contrasts, or contrasts in unimportant places
    """
    # Light/dark analysis
    gray = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    global_contrast = float(np.std(gray))
    dynamic_range = float(np.ptp(gray))

    # Warm/cool decomposition
    # Warm: R+Y channels; Cool: B+G channels (simplified)
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    warmth = (r + 0.5 * g) / 1.5 - (b + 0.5 * g) / 1.5
    warm_cool_range = float(np.ptp(warmth))
    warm_cool_std = float(np.std(warmth))

    # Spatial distribution: center vs periphery contrast
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    margin_y, margin_x = h // 4, w // 4
    center = gray[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x]
    periphery_mask = np.ones_like(gray, dtype=bool)
    periphery_mask[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x] = False
    periphery = gray[periphery_mask]

    center_contrast = float(np.std(center))
    periphery_contrast = float(np.std(periphery))

    return {
        "global_contrast_std": global_contrast,
        "dynamic_range": dynamic_range,
        "warm_cool_range": warm_cool_range,
        "warm_cool_std": warm_cool_std,
        "center_contrast": center_contrast,
        "periphery_contrast": periphery_contrast,
        # Masters: center > periphery contrast (detail in main parts)
        "center_to_periphery_ratio": center_contrast / max(1e-8, periphery_contrast),
    }


def detail_density_analysis(gray: np.ndarray, n_zones: int = 3) -> dict:
    """
    Van Dantzig Ch. I: detail distribution in main vs minor parts.

    Great masters: more detail in main (central light) parts,
    fewer details in minor (peripheral dark) parts.
    """
    h, w = gray.shape
    zone_h, zone_w = h // n_zones, w // n_zones

    # Edge density per zone (proxy for detail)
    edges = filters.sobel(gray)
    zone_densities = np.zeros((n_zones, n_zones))

    for iy in range(n_zones):
        for ix in range(n_zones):
            y0, y1 = iy * zone_h, min((iy + 1) * zone_h, h)
            x0, x1 = ix * zone_w, min((ix + 1) * zone_w, w)
            zone = edges[y0:y1, x0:x1]
            zone_densities[iy, ix] = float(np.mean(zone))

    # Center zone vs border zones
    center_detail = zone_densities[1, 1] if n_zones >= 3 else zone_densities.mean()
    border_mask = np.ones((n_zones, n_zones), dtype=bool)
    if n_zones >= 3:
        border_mask[1, 1] = False
    border_detail = zone_densities[border_mask].mean()

    return {
        "zone_detail_map": zone_densities.tolist(),
        "center_detail_density": float(center_detail),
        "border_detail_density": float(border_detail),
        "center_to_border_ratio": float(center_detail / max(1e-8, border_detail)),
    }
