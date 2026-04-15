"""
Brushstroke & manipulation analysis — van Dantzig Chapter II, Elements 3-4, 8.

Implements computational proxies for:
- Brushstroke shape characterization (length, width, form)
- Pressure/manipulation variation
- Line and plane relationships
- Hatching pattern analysis (parallel, crossing, harmonious/mechanical)
"""

import numpy as np
from scipy import ndimage
from skimage import feature, filters, morphology, measure


def extract_brushstrokes(gray: np.ndarray, sigma: float = 1.5) -> dict:
    """
    Characterize brushstroke properties.

    Van Dantzig Element 8: "Everyone who depicts produces lines or brushstrokes
    which are typical of him."

    Measures:
    - Length distribution of connected stroke segments
    - Width (from distance transform on thinned strokes)
    - Curvature distribution (straight vs curved vs angular)
    - Dominant orientations
    """
    edges = feature.canny(gray, sigma=sigma)
    labeled, n_strokes = ndimage.label(edges)

    lengths = []
    widths = []
    curvatures = []
    orientations = []

    for label_id in range(1, min(n_strokes + 1, 500)):  # cap for performance
        mask = labeled == label_id
        pixels = np.argwhere(mask)

        if len(pixels) < 5:
            continue

        # Length: number of pixels in skeleton
        skel = morphology.skeletonize(mask)
        length = skel.sum()
        lengths.append(length)

        # Width: mean distance from skeleton to edge boundary
        if length > 0:
            dist = ndimage.distance_transform_edt(mask)
            mean_width = np.mean(dist[skel]) * 2  # diameter
            widths.append(mean_width)

        # Curvature: fit line, measure residual (straight = low residual)
        if len(pixels) >= 5:
            ys, xs = pixels[:, 0].astype(float), pixels[:, 1].astype(float)
            if np.std(xs) > 1e-3:
                coeffs = np.polyfit(xs, ys, 1)
                fitted = np.polyval(coeffs, xs)
                residual = np.std(ys - fitted) / max(1e-8, np.std(ys))
                curvatures.append(residual)

            # Orientation: angle of best-fit line
            angle = np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])
            orientations.append(np.degrees(angle) % 180)

    return {
        "n_strokes_detected": n_strokes,
        "stroke_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "stroke_length_std": float(np.std(lengths)) if lengths else 0.0,
        "stroke_width_mean": float(np.mean(widths)) if widths else 0.0,
        "stroke_width_std": float(np.std(widths)) if widths else 0.0,
        "stroke_curvature_mean": float(np.mean(curvatures)) if curvatures else 0.0,
        "stroke_curvature_std": float(np.std(curvatures)) if curvatures else 0.0,
        "orientation_hist": _orientation_histogram(orientations),
        "length_to_width_ratio": (
            float(np.mean(lengths) / max(1e-8, np.mean(widths)))
            if lengths and widths else 0.0
        ),
    }


def _orientation_histogram(angles: list, n_bins: int = 12) -> list:
    """Histogram of stroke orientations (0-180°)."""
    if not angles:
        return [0.0] * n_bins
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 180), density=True)
    return hist.tolist()


def hatching_analysis(gray: np.ndarray) -> dict:
    """
    Van Dantzig Ch. III Elucidation: Hatching rhythm classification.

    Types (fig. 14-20):
    1. Parallel harmonious: gradual variation in tone/shape/direction
    2. Parallel disharmonious: each line ok, but disordered sequence
    3. Parallel mechanical: monotonously constant shape/color/distance

    4. Crossing harmonious: polygons vary but remain similar
    5. Crossing disharmonious: polygons change suddenly
    6. Crossing mechanical: constant polygon shape, monotonous

    We measure:
    - Orientation entropy (mechanical = low, spontaneous = high)
    - Parallelism index (fraction of strokes with similar orientation)
    - Crossing density
    """
    # Gabor filter bank for orientation detection
    orientations = np.linspace(0, np.pi, 12, endpoint=False)
    responses = []
    for theta in orientations:
        filt_real, filt_imag = filters.gabor(gray, frequency=0.1, theta=theta)
        responses.append(np.mean(filt_real ** 2))
    responses = np.array(responses)

    # Normalize to probability distribution
    total = responses.sum()
    if total > 0:
        probs = responses / total
    else:
        probs = np.ones_like(responses) / len(responses)

    # Orientation entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(orientations))

    # Dominant orientation strength (mechanical → strong peak)
    dominant_strength = float(np.max(probs))

    # Parallelism: concentration around dominant orientation
    dominant_idx = np.argmax(probs)
    neighborhood = [
        (dominant_idx - 1) % len(probs),
        dominant_idx,
        (dominant_idx + 1) % len(probs),
    ]
    parallelism = sum(probs[i] for i in neighborhood)

    # Crossing detection: strong responses at multiple non-adjacent orientations
    sorted_resp = np.sort(probs)[::-1]
    crossing_index = sorted_resp[1] / max(1e-8, sorted_resp[0]) if len(sorted_resp) > 1 else 0.0

    return {
        "orientation_entropy": float(entropy),
        "orientation_entropy_normalized": float(entropy / max_entropy),
        "dominant_orientation_strength": dominant_strength,
        "parallelism_index": float(parallelism),
        "crossing_index": float(crossing_index),
        "orientation_response_profile": responses.tolist(),
        # Classification hint
        "hatching_type": _classify_hatching(entropy / max_entropy, parallelism, crossing_index),
    }


def _classify_hatching(norm_entropy: float, parallelism: float, crossing: float) -> str:
    """Classify hatching pattern per van Dantzig types."""
    if parallelism > 0.6 and norm_entropy < 0.5:
        return "parallel_mechanical"
    if parallelism > 0.5 and norm_entropy > 0.7:
        return "parallel_harmonious"
    if crossing > 0.7 and norm_entropy > 0.7:
        return "crossing_harmonious"
    if crossing > 0.7 and norm_entropy < 0.5:
        return "crossing_mechanical"
    if norm_entropy > 0.8:
        return "parallel_disharmonious_or_crossing_spontaneous"
    return "mixed"


def pressure_variation(gray: np.ndarray) -> dict:
    """
    Van Dantzig Element 4: Variations in manipulation.

    "We distinguish the variations in pressure, the way in which the instrument
    is held, the speed and direction at which it is moved."

    Measures:
    - Local intensity variance along edges (pressure variation)
    - Gradient magnitude distribution (force of application)
    - Stroke tapering (pressure lift at ends vs middle)
    """
    edges = feature.canny(gray, sigma=1.5)
    grad_mag = filters.sobel(gray)

    # Intensity along edges
    if edges.sum() > 0:
        edge_intensities = gray[edges]
        edge_gradients = grad_mag[edges]
    else:
        edge_intensities = np.array([0.0])
        edge_gradients = np.array([0.0])

    return {
        "edge_intensity_mean": float(np.mean(edge_intensities)),
        "edge_intensity_std": float(np.std(edge_intensities)),
        "edge_gradient_mean": float(np.mean(edge_gradients)),
        "edge_gradient_std": float(np.std(edge_gradients)),
        # High std → varied pressure (spontaneous); Low std → uniform (mechanical/inhibited)
        "pressure_variation_index": float(
            np.std(edge_intensities) / max(1e-8, np.mean(edge_intensities))
        ),
    }
