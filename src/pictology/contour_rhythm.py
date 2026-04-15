"""
Contour & rhythm analysis — van Dantzig Chapter II, Elements 16, 20, 21.

Implements computational proxies for:
- Contour–interior relationship (wire vs connected vs harmonious)
- Rhythm in single, parallel, and crossing lines
- Totality (interrelatedness of all elements)
"""

import numpy as np
from scipy import ndimage
from skimage import feature, filters, segmentation, morphology


def contour_interior_relationship(gray: np.ndarray) -> dict:
    """
    Van Dantzig Element 16, Ch. III (fig. 26a-d): Contour and Interior.

    Four types:
    a) Contour tone constant, interior varies → shape connection only
    b) Contour & interior tones connected, shape approximately follows
    c) Wire contour: no connection of color or shape (forgery indicator)
    d) Harmonious: shape and color intimately connected (master indicator)

    We measure:
    - Intensity correlation between contour pixels and adjacent interior
    - Gradient continuity from contour into interior
    - Contour "wire" score (isolated sharp contour = high)
    """
    edges = feature.canny(gray, sigma=1.5)

    if edges.sum() < 10:
        return {
            "contour_interior_correlation": 0.0,
            "contour_wire_score": 0.0,
            "gradient_continuity": 0.0,
        }

    # Dilate edges to get near-contour interior pixels
    dilated = ndimage.binary_dilation(edges, iterations=3)
    interior_band = dilated & ~edges

    if interior_band.sum() < 10:
        return {
            "contour_interior_correlation": 0.0,
            "contour_wire_score": 0.0,
            "gradient_continuity": 0.0,
        }

    # Intensity at contour vs adjacent interior
    contour_intensity = gray[edges]
    interior_intensity = gray[interior_band]

    # Correlation: do intensities covary? (harmonious → high correlation)
    min_len = min(len(contour_intensity), len(interior_intensity))
    if min_len > 10:
        # Sample to equal lengths
        rng = np.random.RandomState(42)
        ci = rng.choice(contour_intensity, size=min_len, replace=True)
        ii = rng.choice(interior_intensity, size=min_len, replace=True)
        correlation = float(np.corrcoef(ci, ii)[0, 1])
    else:
        correlation = 0.0

    # Wire score: how isolated is the contour from the interior?
    # Sharp edges with big intensity jump = wire contour
    grad_at_edge = filters.sobel(gray)[edges]
    wire_score = float(np.mean(grad_at_edge) / max(1e-8, np.std(gray)))

    # Gradient continuity: does the gradient direction remain consistent
    # across the contour boundary?
    gx = ndimage.sobel(gray, axis=1).astype(np.float64)
    gy = ndimage.sobel(gray, axis=0).astype(np.float64)
    angle_at_edge = np.arctan2(gy[edges], gx[edges])
    angle_at_interior = np.arctan2(gy[interior_band], gx[interior_band])
    gradient_continuity = 1.0 - float(
        np.abs(np.mean(np.cos(angle_at_edge)) - np.mean(np.cos(angle_at_interior)))
    )

    return {
        "contour_interior_correlation": correlation,
        "contour_wire_score": wire_score,
        "gradient_continuity": float(np.clip(gradient_continuity, 0, 1)),
        # Classification hint
        "contour_type": _classify_contour(correlation, wire_score, gradient_continuity),
    }


def _classify_contour(correlation: float, wire_score: float, continuity: float) -> str:
    """Classify contour type per van Dantzig fig. 26."""
    if wire_score > 2.0 and correlation < 0.3:
        return "wire_contour"         # Type c — forgery indicator
    if correlation > 0.6 and continuity > 0.7:
        return "harmonious"           # Type d — master indicator
    if correlation > 0.4:
        return "shape_connected"      # Type a
    return "partially_connected"      # Type b


def rhythm_analysis(gray: np.ndarray) -> dict:
    """
    Van Dantzig Element 20, Ch. III: Line and Rhythm.

    "Single lines: generally mechanical; most vary little in tone, width, or direction"
    vs spontaneous: harmonious variation.

    Measures:
    - Local stroke orientation variance (mechanical = low, spontaneous = moderate)
    - Tone variation along strokes
    - Regularity / repetitiveness of patterns
    """
    # Orientation field via structure tensor
    axx, axy, ayy = feature.structure_tensor(gray, sigma=1.5)
    orientation = 0.5 * np.arctan2(2 * axy, axx - ayy)

    # Coherence (local agreement of orientation)
    coherence_map = np.sqrt((axx - ayy) ** 2 + 4 * axy ** 2) / (axx + ayy + 1e-8)

    # Patch-wise rhythm analysis
    patch_size = 64
    h, w = gray.shape
    local_coherences = []
    local_orientation_vars = []
    tone_variations = []

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch_orient = orientation[y:y + patch_size, x:x + patch_size]
            patch_coh = coherence_map[y:y + patch_size, x:x + patch_size]
            patch_gray = gray[y:y + patch_size, x:x + patch_size]

            local_coherences.append(np.mean(patch_coh))
            # Circular variance of orientation
            sin_sum = np.mean(np.sin(2 * patch_orient))
            cos_sum = np.mean(np.cos(2 * patch_orient))
            circ_var = 1 - np.sqrt(sin_sum ** 2 + cos_sum ** 2)
            local_orientation_vars.append(circ_var)
            tone_variations.append(np.std(patch_gray))

    return {
        "mean_coherence": float(np.mean(local_coherences)) if local_coherences else 0.0,
        "coherence_std": float(np.std(local_coherences)) if local_coherences else 0.0,
        "mean_orientation_variance": float(np.mean(local_orientation_vars)) if local_orientation_vars else 0.0,
        "tone_variation_mean": float(np.mean(tone_variations)) if tone_variations else 0.0,
        "tone_variation_std": float(np.std(tone_variations)) if tone_variations else 0.0,
        # Mechanical: high coherence + low orientation variance
        # Spontaneous: moderate coherence + moderate orientation variance
        "rhythm_type": _classify_rhythm(
            np.mean(local_coherences) if local_coherences else 0,
            np.mean(local_orientation_vars) if local_orientation_vars else 0,
        ),
    }


def _classify_rhythm(coherence: float, orient_var: float) -> str:
    if coherence > 0.7 and orient_var < 0.3:
        return "mechanical"
    if coherence > 0.4 and orient_var > 0.3:
        return "spontaneous_harmonious"
    if orient_var > 0.6:
        return "spontaneous_disharmonious"
    return "mixed"


def totality_score(feature_dict: dict) -> float:
    """
    Van Dantzig Element 21: Totality.

    "A work of high quality had to be more or less a totality... the extent to
    which the composing elements are interrelated."

    This is a meta-score: how coherent are all the individual element scores?
    A great work shows internal consistency; a forgery shows contradictions.

    We measure the variance of normalized quality indicators — low variance
    means all indicators point the same direction (high totality).
    """
    quality_indicators = []

    # Collect all indicators that signal "quality" vs "forgery"
    if "spontaneity_index" in feature_dict:
        quality_indicators.append(feature_dict["spontaneity_index"])

    if "center_to_periphery_ratio" in feature_dict:
        # Normalize: ratio > 1 is good (more detail in center)
        ratio = feature_dict["center_to_periphery_ratio"]
        quality_indicators.append(min(1.0, ratio / 2.0))

    if "contour_interior_correlation" in feature_dict:
        quality_indicators.append(max(0, feature_dict["contour_interior_correlation"]))

    if "gradient_continuity" in feature_dict:
        quality_indicators.append(feature_dict["gradient_continuity"])

    if "pressure_variation_index" in feature_dict:
        quality_indicators.append(min(1.0, feature_dict["pressure_variation_index"]))

    if len(quality_indicators) < 2:
        return 0.5

    arr = np.array(quality_indicators)
    # Totality = mean quality × consistency (low variance)
    mean_q = np.mean(arr)
    consistency = 1.0 - np.std(arr)
    return float(np.clip(mean_q * consistency, 0.0, 1.0))
