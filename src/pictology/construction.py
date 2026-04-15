"""
Sequence of development & construction analysis — van Dantzig Elements 6, 15, 19.

Implements computational proxies for:
- Paint layering order (background-first = correct sequence)
- Saved contours detection (forgery indicator)
- Major/minor part discrimination
- Anatomy proportions (head-to-body ratio, hand proportions)
"""

import numpy as np
from scipy import ndimage
from skimage import filters, segmentation, measure, feature


def sequence_of_development(gray: np.ndarray) -> dict:
    """
    Van Dantzig Element 6: Sequence of Development.

    "Most great masters will paint the background first. Each succeeding layer
    brings the subject matter on planes increasingly closer to the viewer."

    "Should the foreground be painted first, the background would have to be
    filled in very carefully... a 'saved' space along this contour."

    We detect:
    - Saved contours (gap between foreground object and background)
    - Layer overlapping direction (foreground overlapping background = correct)
    """
    edges = feature.canny(gray, sigma=2.0)

    # Detect "saved contours": thin bright/dark gaps next to edges
    # These appear as a halo of different intensity along object boundaries
    dilated_1 = ndimage.binary_dilation(edges, iterations=1)
    dilated_3 = ndimage.binary_dilation(edges, iterations=3)
    halo_band = dilated_3 & ~dilated_1

    if halo_band.sum() < 10 or edges.sum() < 10:
        return {
            "saved_contour_score": 0.0,
            "contour_sharpness": 0.0,
            "layer_coherence": 0.0,
        }

    # Saved contour: unusual intensity in the halo band
    edge_intensity = gray[edges]
    halo_intensity = gray[halo_band]
    general_intensity = gray[~dilated_3] if (~dilated_3).sum() > 0 else gray.ravel()

    # A "saved" contour has halo intensity closer to background than to edge
    edge_mean = np.mean(edge_intensity)
    halo_mean = np.mean(halo_intensity)
    bg_mean = np.mean(general_intensity)

    # If halo is more like background than edge → saved contour (bad sign)
    dist_to_edge = abs(halo_mean - edge_mean)
    dist_to_bg = abs(halo_mean - bg_mean)
    saved_contour_score = dist_to_bg / max(1e-8, dist_to_edge + dist_to_bg)

    # Contour sharpness: how abrupt is the edge transition?
    grad_mag = filters.sobel(gray)
    contour_sharpness = float(np.mean(grad_mag[edges]))

    # Layer coherence: gradual falloff from edges into interior (good layering)
    # vs abrupt uniform areas (flat, all-at-once painting)
    distances = [1, 2, 3, 5, 8]
    intensity_falloff = []
    for d in distances:
        dilated = ndimage.binary_dilation(edges, iterations=d)
        ring = dilated & ~ndimage.binary_dilation(edges, iterations=max(0, d - 1))
        if ring.sum() > 0:
            intensity_falloff.append(np.mean(gray[ring]))

    if len(intensity_falloff) >= 3:
        # Smooth monotonic falloff = good layering
        diffs = np.diff(intensity_falloff)
        layer_coherence = float(1.0 - np.std(np.sign(diffs)))
    else:
        layer_coherence = 0.5

    return {
        "saved_contour_score": float(saved_contour_score),
        "contour_sharpness": float(contour_sharpness),
        "layer_coherence": float(layer_coherence),
    }


def major_minor_analysis(img_rgb: np.ndarray) -> dict:
    """
    Van Dantzig Element 19: Major and Minor parts.

    "Every great artist will make his greatest effort in those parts he considers
    most important."

    "A forger doing his utmost to produce impressive work will be hindered by
    this very effort, and the important parts are bound to be more inhibited."

    Detects the main subject region and compares detail/contrast between
    major and minor parts.
    """
    gray = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    h, w = gray.shape

    # Saliency-based main subject detection
    # Use center-surround difference at multiple scales
    saliency = np.zeros_like(gray)
    for sigma in [2, 4, 8, 16]:
        blurred = ndimage.gaussian_filter(gray, sigma=sigma)
        saliency += np.abs(gray - blurred)
    saliency /= 4

    # Threshold to get major region
    threshold = np.percentile(saliency, 70)
    major_mask = saliency > threshold
    minor_mask = ~major_mask

    if major_mask.sum() < 100 or minor_mask.sum() < 100:
        return {"major_minor_detail_ratio": 1.0, "major_minor_contrast_ratio": 1.0}

    # Detail (edge density) in each region
    edges = filters.sobel(gray)
    major_detail = np.mean(edges[major_mask])
    minor_detail = np.mean(edges[minor_mask])

    # Contrast in each region
    major_contrast = np.std(gray[major_mask])
    minor_contrast = np.std(gray[minor_mask])

    # Color variance in each region
    major_color_var = np.mean([np.std(img_rgb[..., c][major_mask]) for c in range(3)])
    minor_color_var = np.mean([np.std(img_rgb[..., c][minor_mask]) for c in range(3)])

    return {
        "major_detail_density": float(major_detail),
        "minor_detail_density": float(minor_detail),
        "major_minor_detail_ratio": float(major_detail / max(1e-8, minor_detail)),
        "major_contrast": float(major_contrast),
        "minor_contrast": float(minor_contrast),
        "major_minor_contrast_ratio": float(major_contrast / max(1e-8, minor_contrast)),
        "major_color_variance": float(major_color_var),
        "minor_color_variance": float(minor_color_var),
        "major_fraction": float(major_mask.sum() / (h * w)),
    }
