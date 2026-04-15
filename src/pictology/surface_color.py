"""
Surface, color, texture, and spatial analysis — van Dantzig Chapter II, Elements 2, 5, 12-14.

Implements computational proxies for:
- Organization of surface (zonal composition preferences)
- Color warmth/coolness distributions
- Light/dark spatial relationships (depth cues)
- Texture rendering characteristics
- Perspective/projection analysis
"""

import numpy as np
from scipy import ndimage
from skimage import filters, color as skcolor


def surface_organization(img_rgb: np.ndarray, n_zones: int = 3) -> dict:
    """
    Van Dantzig Element 2: Organization of the surface.

    "The surface of a picture can be divided into zones or compartments...
    Most painters have preferences for placing particular shapes and colors
    in particular zones."

    Divides image into 9 compartments (3×3) and computes per-zone:
    - Mean luminance and color warmth
    - Edge density (detail level)
    - Color variance (visual complexity)
    """
    gray = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    h, w = gray.shape
    zone_h, zone_w = h // n_zones, w // n_zones

    edges = filters.sobel(gray)
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    warmth = (r + 0.5 * g) / 1.5 - (b + 0.5 * g) / 1.5

    luminance_map = np.zeros((n_zones, n_zones))
    warmth_map = np.zeros((n_zones, n_zones))
    detail_map = np.zeros((n_zones, n_zones))
    color_variance_map = np.zeros((n_zones, n_zones))

    for iy in range(n_zones):
        for ix in range(n_zones):
            y0, y1 = iy * zone_h, min((iy + 1) * zone_h, h)
            x0, x1 = ix * zone_w, min((ix + 1) * zone_w, w)

            luminance_map[iy, ix] = np.mean(gray[y0:y1, x0:x1])
            warmth_map[iy, ix] = np.mean(warmth[y0:y1, x0:x1])
            detail_map[iy, ix] = np.mean(edges[y0:y1, x0:x1])
            color_variance_map[iy, ix] = np.mean(np.std(img_rgb[y0:y1, x0:x1], axis=(0, 1)))

    # Identify main subject zone (highest detail + contrast)
    main_zone = np.unravel_index(np.argmax(detail_map), detail_map.shape)

    return {
        "luminance_zones": luminance_map.tolist(),
        "warmth_zones": warmth_map.tolist(),
        "detail_zones": detail_map.tolist(),
        "color_variance_zones": color_variance_map.tolist(),
        "main_subject_zone": list(main_zone),
        "center_luminance": float(luminance_map[1, 1]) if n_zones >= 3 else float(luminance_map.mean()),
        "luminance_gradient_tb": float(luminance_map[0].mean() - luminance_map[-1].mean()),
        "warmth_gradient_tb": float(warmth_map[-1].mean() - warmth_map[0].mean()),
    }


def color_analysis(img_rgb: np.ndarray) -> dict:
    """
    Van Dantzig Element 5: Colors.

    "Colors may be designated as warm, cool, or neutral."
    "Some painters have a preference for particular colors in particular places."

    Measures:
    - Warm/cool color balance
    - Color palette range and dominance
    - Spatial warm→cool gradient (foreground→background)
    """
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    # Warm/cool index: +1 = fully warm, -1 = fully cool
    warmth = (2.0 * r + g - b - g) / 3.0  # simplified warm-cool axis
    mean_warmth = float(np.mean(warmth))

    # Convert to HSV for hue analysis
    hsv = skcolor.rgb2hsv(img_rgb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Hue histogram (12 bins = color wheel)
    hue_hist, _ = np.histogram(hue[sat > 0.1], bins=12, range=(0, 1), density=True)
    dominant_hue_bin = int(np.argmax(hue_hist))

    # Saturation characteristics
    mean_saturation = float(np.mean(sat))
    saturation_range = float(np.ptp(sat))

    # Foreground (bottom half) vs background (top half) warmth
    h = img_rgb.shape[0]
    fg_warmth = float(np.mean(warmth[h // 2:]))
    bg_warmth = float(np.mean(warmth[:h // 2]))

    return {
        "mean_warmth": mean_warmth,
        "warmth_std": float(np.std(warmth)),
        "fg_warmth": fg_warmth,
        "bg_warmth": bg_warmth,
        "fg_bg_warmth_diff": fg_warmth - bg_warmth,
        "hue_histogram": hue_hist.tolist(),
        "dominant_hue_bin": dominant_hue_bin,
        "mean_saturation": mean_saturation,
        "saturation_range": saturation_range,
        "value_mean": float(np.mean(val)),
        "value_std": float(np.std(val)),
    }


def light_dark_spatial(img_rgb: np.ndarray) -> dict:
    """
    Van Dantzig Element 12: Relation between light and dark.

    "The dark, black lines have a stronger effect than the lighter, gray ones.
    The contrast between white and lines has an effect at least as strong as
    linear perspective."

    Also Element 14: Projection (perspective).

    Masters use:
    - Warm fg + cool bg → depth
    - Strong contrast in main parts → spatial prominence
    - Light/dark → form and contour (Rembrandt #21-24)
    """
    gray = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    h, w = gray.shape

    # Vertical luminance gradient (top → bottom)
    row_means = np.mean(gray, axis=1)
    vertical_gradient = np.polyfit(np.arange(h), row_means, 1)[0]

    # Horizontal luminance gradient (left → right)
    col_means = np.mean(gray, axis=0)
    horizontal_gradient = np.polyfit(np.arange(w), col_means, 1)[0]

    # Light/dark clustering (bimodality)
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 1), density=True)
    # Bimodality suggests strong light/dark contrast (master-like)
    hist_norm = hist / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    # Warmth depth cue
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    warmth = (r + 0.5 * g) / 1.5 - (b + 0.5 * g) / 1.5
    warmth_vertical = np.polyfit(np.arange(h), np.mean(warmth, axis=1), 1)[0]

    return {
        "vertical_luminance_gradient": float(vertical_gradient),
        "horizontal_luminance_gradient": float(horizontal_gradient),
        "luminance_entropy": float(entropy),
        "warmth_depth_gradient": float(warmth_vertical),
        # Positive warmth gradient (bottom warmer) = proper depth cue (master-like)
        "depth_cue_coherent": float(warmth_vertical) > 0,
    }


def texture_rendering(gray: np.ndarray) -> dict:
    """
    Van Dantzig Element 13: Rendering of texture.

    Different textures (flesh, metal, silk, cloth, linen) rendered differently
    by masters. We measure local texture statistics to characterize variety.
    """
    # Multi-scale texture via Gaussian pyramids
    scales = [1, 2, 4, 8]
    texture_stats = {}

    for s in scales:
        if s > 1:
            smoothed = ndimage.gaussian_filter(gray, sigma=s)
        else:
            smoothed = gray

        grad = filters.sobel(smoothed)
        texture_stats[f"texture_energy_scale_{s}"] = float(np.mean(grad ** 2))
        texture_stats[f"texture_variance_scale_{s}"] = float(np.var(grad))

    # Texture variety across patches (masters show different texture for different materials)
    patch_size = max(32, min(gray.shape) // 8)
    patch_energies = []
    h, w = gray.shape
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            patch_energies.append(np.mean(filters.sobel(patch) ** 2))

    texture_stats["texture_variety"] = float(np.std(patch_energies)) if patch_energies else 0.0
    texture_stats["texture_range"] = float(np.ptp(patch_energies)) if patch_energies else 0.0

    return texture_stats
