# Pictology — Computational Art Authentication

A Python implementation of **M.M. van Dantzig's (1973) *Pictology: An Analytical Method for Attribution and Evaluation of Pictures***, translating his 21 pictorial elements into quantifiable image-analysis features.

Van Dantzig's method works by building a **characteristic list** for each artist from verified authentic works, then scoring candidate paintings against that list. A match of **≥ 75 %** of characteristics indicates authenticity; **< 50 %** suggests another hand (pp. 5–6, validated at p. 51 with the Van Gogh corpus).

The theoretical framework spans four chapters:

| Chapter | Title | Pages | Scope |
|---|---|---|---|
| I | Fundamentals of Pictology | pp. 3–11 | Spontaneous vs inhibited execution; identity; quality; the 75 %/50 % scoring rule |
| II | Pictorial Elements in Pictology | pp. 12–29 | Definition of the 21 pictorial elements |
| III | Elucidation | pp. 30–40 | Detailed figures and worked examples (rhythm, contrast, contour types, Picasso) |
| IV | The Application of Pictology | pp. 41–51 | Vermeer/Van Meegeren case; building & applying characteristic lists |

Characteristic lists for three masters are provided in the book: **Van Gogh** (pp. 52–70, 85 characteristics), **Rembrandt** (pp. 71–77, 65 characteristics), and **Frans Hals** (pp. 78–93, 142 characteristics), followed by **Leonardo da Vinci** (pp. 94+).

The package extracts **117 numerical features** from a single painting image and exposes a two-step workflow: profile building and authentication.

---

## Quick Start

```python
from src.pictology import PictologyPipeline, CharacteristicList

pipe = PictologyPipeline(max_size=1024)

# 1. Build an artist profile from verified works
verified_paths = ["data/raw/rembrandt/authenticated/work_01.jpg", ...]
profile = pipe.build_artist_profile("Rembrandt", verified_paths)
profile.save("profiles/rembrandt.json")

# 2. Authenticate a candidate
result = pipe.authenticate("candidate.jpg", profile)
print(result["verdict"], result["positive_pct"])
# → "authentic"  82.3
```

---

## Architecture Overview

```
pictology/
├── __init__.py               # Public API: PictologyPipeline, CharacteristicList
├── pipeline.py               # Orchestrator — runs all analyses, returns feature dict
├── spontaneity.py            # Ch. I   — line spontaneity, contrast, detail density
├── brushstroke.py            # Ch. II  — stroke shape, hatching, pressure
├── surface_color.py          # Ch. II  — surface zones, color, light/dark, texture
├── contour_rhythm.py         # Ch. II  — contour–interior, rhythm, totality
├── construction.py           # Ch. II  — paint sequence, major/minor parts
└── characteristic_list.py    # Appendix — per-artist profiles, scoring, verdicts
```

---

## Module Reference

### `pipeline.py` — Orchestrator

The central entry point. Loads an image, runs every analysis module, and assembles the unified feature dictionary.

| Function / Method | Description |
|---|---|
| `PictologyPipeline(max_size=1024)` | Constructor. `max_size` caps the longest image dimension for processing speed. |
| `PictologyPipeline.load_image(path)` | Load an image file as a float32 RGB array `[0, 1]`, resized to `max_size`. |
| `PictologyPipeline.to_gray(img_rgb)` | Convert RGB array to luminance using Rec. 709 coefficients. |
| `PictologyPipeline.extract_features(image)` | **Core method.** Accepts a file path or RGB array. Calls every analysis function below, prefixes their keys, and returns a flat `dict` of ~117 numerical features. |
| `PictologyPipeline.build_artist_profile(artist_name, image_paths)` | Extract features from a list of verified works and build a `CharacteristicList`. Van Dantzig recommended ≥ 50 works; minimum practical: 10+. |
| `PictologyPipeline.authenticate(candidate, characteristic_list)` | Extract features from a candidate image and score them against an existing `CharacteristicList`. Returns verdict dict. |
| `_prefix(prefix, d)` | Helper — prefixes all numeric keys in a dict (e.g. `"stroke"` → `"stroke_length_mean"`). List-valued features are exploded into indexed keys. |

---

### `spontaneity.py` — Line Spontaneity, Contrast & Detail Density

Implements van Dantzig **Chapter I** (pp. 3–11): the distinction between *spontaneous* (master) and *inhibited* (forger) mark-making.

| Function | Van Dantzig Reference | Description |
|---|---|---|
| `extract_lines(gray, sigma=1.0)` | — | Canny edge detection; returns a binary edge map used by downstream functions. |
| `line_spontaneity_score(gray, patch_size=64)` | Ch. I, pp. 6–11, Figs. 3–5 | Divides the image into patches and measures three spontaneity proxies per patch: **curvature smoothness** (Laplacian of skeleton), **junction density** (skeleton branch-points — re-starts indicate inhibition), and **stroke-width consistency** (distance-transform CV). Returns a composite `spontaneity_index` ∈ [0, 1]. |
| `_spontaneity_index(curvatures, junctions, widths)` | — | Internal composite: averages normalised curvature-smoothness, junction-density, and width-consistency into a single spontaneity score. |
| `contrast_analysis(img_rgb)` | Ch. I pp. 10–11 & Ch. III pp. 32–34 | Measures **light/dark contrast** (global std, dynamic range), **warm/cool contrast** (range and std of a warm–cool axis), and **spatial distribution** (center vs periphery contrast). Masters have stronger contrast in central/main parts. |
| `detail_density_analysis(gray, n_zones=3)` | Ch. I p. 11, Element 19 pp. 27–29 | Divides the image into an `n×n` grid, computes edge density (Sobel) per zone, and compares **center detail** to **border detail**. Masters concentrate detail on the main subject. |

**Features produced** (prefixed `spont_*` and `detail_*`):
`curvature_smoothness`, `junction_density`, `width_consistency_cv`, `spontaneity_index`, `zone_detail_map`, `center_detail_density`, `border_detail_density`, `center_to_border_ratio`

**Features produced** (prefixed `contrast_*`):
`global_contrast_std`, `dynamic_range`, `warm_cool_range`, `warm_cool_std`, `center_contrast`, `periphery_contrast`, `center_to_periphery_ratio`

---

### `brushstroke.py` — Stroke Shape, Hatching & Pressure

Implements van Dantzig **Chapter II, Elements 3–4, 8** (pp. 14–22) and **Chapter III** hatching elucidation (pp. 30–32): brushstroke characterisation, hatching patterns, and manipulation pressure.

| Function | Van Dantzig Reference | Description |
|---|---|---|
| `extract_brushstrokes(gray, sigma=1.5)` | Element 8, pp. 19–22 | Detects connected edge components (capped at 500 for performance). For each stroke: skeleton **length**, distance-transform **width**, polyfit-residual **curvature**, and best-fit **orientation**. Also computes a 12-bin orientation histogram and an overall length-to-width ratio. |
| `_orientation_histogram(angles, n_bins=12)` | — | Normalised histogram of stroke orientations over [0°, 180°]. |
| `hatching_analysis(gray)` | Ch. III, pp. 30–32, Figs. 14–20 | Runs a 12-orientation **Gabor filter bank** and derives: **orientation entropy** (mechanical = low, spontaneous = high), **dominant orientation strength**, **parallelism index** (concentration around peak), and **crossing index** (secondary peak strength). Classifies the hatching as one of six van Dantzig types. |
| `_classify_hatching(norm_entropy, parallelism, crossing)` | pp. 30–32, Figs. 14–20 | Rule-based classifier: `parallel_mechanical`, `parallel_harmonious`, `parallel_disharmonious_or_crossing_spontaneous`, `crossing_harmonious`, `crossing_mechanical`, or `mixed`. |
| `pressure_variation(gray)` | Element 4, pp. 15–16 | Measures **intensity** and **gradient magnitude** statistics along detected edges. High standard deviation signals varied pressure (spontaneous); low std signals mechanical/inhibited work. Returns a composite `pressure_variation_index` (CV of edge intensity). |

**Features produced** (prefixed `stroke_*`, `hatch_*`, `pressure_*`):
`n_strokes_detected`, `stroke_length_mean/_std`, `stroke_width_mean/_std`, `stroke_curvature_mean/_std`, `orientation_hist_0..11`, `length_to_width_ratio`, `orientation_entropy`, `orientation_entropy_normalized`, `dominant_orientation_strength`, `parallelism_index`, `crossing_index`, `orientation_response_profile_0..11`, `hatching_type`, `edge_intensity_mean/_std`, `edge_gradient_mean/_std`, `pressure_variation_index`

---

### `surface_color.py` — Surface Organisation, Color, Light/Dark & Texture

Implements van Dantzig **Chapter II, Elements 2, 5, 12–14** (pp. 13–25): spatial composition, colour warmth, luminance relationships, and texture rendering.

| Function | Van Dantzig Reference | Description |
|---|---|---|
| `surface_organization(img_rgb, n_zones=3)` | Element 2, pp. 13–14 | Divides the canvas into a 3 × 3 grid and computes per-zone: **mean luminance**, **colour warmth**, **edge density** (detail level), and **colour variance**. Identifies the **main subject zone** (highest detail). Also measures top-to-bottom luminance and warmth gradients. |
| `color_analysis(img_rgb)` | Element 5, pp. 17–18 | Computes **warm/cool balance** (simplified RGB axis), a 12-bin **hue histogram** (via HSV), **saturation** stats, and **foreground vs background warmth** (bottom-half vs top-half). Masters typically use warm foregrounds and cool backgrounds. |
| `light_dark_spatial(img_rgb)` | Elements 12 & 14, pp. 23–25 | Fits linear gradients to row-wise and column-wise luminance means (**vertical/horizontal luminance gradient**). Computes a 64-bin luminance histogram **entropy** (bimodal = strong contrast). Derives a **warmth depth gradient** — positive = bottom warmer than top, coherent with depth perspective (master-like). |
| `texture_rendering(gray)` | Element 13, pp. 24–25 | Multi-scale texture analysis via Gaussian smoothing at σ = {1, 2, 4, 8}. Computes **texture energy** and **variance** at each scale. Then divides into patches and measures cross-patch **texture variety** and **range** — masters render different materials (flesh, metal, silk) with distinct textures. |

**Features produced** (prefixed `surface_*`, `color_*`, `lightdark_*`, `texture_*`):
`luminance_zones_*`, `warmth_zones_*`, `detail_zones_*`, `color_variance_zones_*`, `main_subject_zone_*`, `center_luminance`, `luminance_gradient_tb`, `warmth_gradient_tb`, `mean_warmth`, `warmth_std`, `fg_warmth`, `bg_warmth`, `fg_bg_warmth_diff`, `hue_histogram_0..11`, `dominant_hue_bin`, `mean_saturation`, `saturation_range`, `value_mean`, `value_std`, `vertical_luminance_gradient`, `horizontal_luminance_gradient`, `luminance_entropy`, `warmth_depth_gradient`, `depth_cue_coherent`, `texture_energy_scale_1/2/4/8`, `texture_variance_scale_1/2/4/8`, `texture_variety`, `texture_range`

---

### `contour_rhythm.py` — Contour–Interior, Rhythm & Totality

Implements van Dantzig **Chapter II, Elements 16, 20, 21** (pp. 26–29) and **Chapter III** contour elucidation (p. 37, Figs. 26a–d): how contours relate to their interiors, rhythmic quality of strokes, and overall coherence.

| Function | Van Dantzig Reference | Description |
|---|---|---|
| `contour_interior_relationship(gray)` | Element 16, pp. 26–27 & p. 37, Figs. 26a–d | Detects Canny edges, dilates them to create a near-contour interior band. Measures: **contour–interior intensity correlation** (harmonious = high), **wire score** (sharp isolated edges / global std — high = forgery indicator), and **gradient continuity** (consistent gradient direction across contour boundary). Classifies the contour into one of four van Dantzig types. |
| `_classify_contour(correlation, wire_score, continuity)` | p. 37, Fig. 26a–d | Rule-based classifier: `wire_contour` (forgery indicator), `harmonious` (master indicator), `shape_connected`, or `partially_connected`. |
| `rhythm_analysis(gray)` | Element 20, pp. 28–29 & Ch. III pp. 30–32 | Computes the **structure tensor** orientation field. Analyses patches for: **local coherence** (direction agreement), **circular orientation variance**, and **tone variation**. Mechanical strokes show high coherence + low variance; spontaneous harmonious strokes show moderate values. |
| `_classify_rhythm(coherence, orient_var)` | pp. 30–32 | Classifier: `mechanical`, `spontaneous_harmonious`, `spontaneous_disharmonious`, or `mixed`. |
| `totality_score(feature_dict)` | Element 21, pp. 28–29 | **Meta-score** across the entire feature vector. Collects normalised quality indicators (`spontaneity_index`, `center_to_periphery_ratio`, `contour_interior_correlation`, `gradient_continuity`, `pressure_variation_index`), then computes `mean_quality × consistency` (1 − std). A great work shows all indicators pointing the same direction; a forgery shows contradictions. Returns a single float ∈ [0, 1]. |

**Features produced** (prefixed `contour_*`, `rhythm_*`, plus `totality_score`):
`contour_interior_correlation`, `contour_wire_score`, `gradient_continuity`, `contour_type`, `mean_coherence`, `coherence_std`, `mean_orientation_variance`, `tone_variation_mean`, `tone_variation_std`, `rhythm_type`, `totality_score`

---

### `construction.py` — Sequence of Development & Major/Minor Parts

Implements van Dantzig **Elements 6, 19** (pp. 18–19 & 27–29): paint layering order, saved contours, and the distribution of effort between important and subordinate regions.

| Function | Van Dantzig Reference | Description |
|---|---|---|
| `sequence_of_development(gray)` | Element 6, pp. 18–19 | Detects **saved contours** — thin halos along edges where the background was painted *around* the foreground (incorrect sequence, forgery indicator). Measures a `saved_contour_score` (halo intensity closer to background than edge = bad sign), `contour_sharpness`, and `layer_coherence` (monotonic intensity falloff from edge into interior indicates correct background-first layering). |
| `major_minor_analysis(img_rgb)` | Element 19, pp. 27–29 | Identifies the **main subject** via multi-scale centre–surround saliency. Thresholds at the 70th percentile to separate major and minor regions. Compares **edge density** (detail), **luminance contrast**, and **colour variance** between the two regions. Masters invest the most effort in the main subject; forgers' effort is inhibited precisely in the important parts. |

**Features produced** (prefixed `sequence_*`, `majmin_*`):
`saved_contour_score`, `contour_sharpness`, `layer_coherence`, `major_detail_density`, `minor_detail_density`, `major_minor_detail_ratio`, `major_contrast`, `minor_contrast`, `major_minor_contrast_ratio`, `major_color_variance`, `minor_color_variance`, `major_fraction`

---

### `characteristic_list.py` — Artist Profiles & Authentication Scoring

Implements the scoring system from **van Dantzig's Chapter IV** (pp. 41–51) and the **characteristic list format** used throughout the application section (pp. 52–93): per-artist characteristic lists with threshold-based verdicts.

| Class / Function | Description |
|---|---|
| `Verdict` | Enum: `POSITIVE` (+), `NEGATIVE` (−), `NOT_APPLICABLE` (0). |
| `Characteristic` | Dataclass for a single list entry: element category, feature key, low/high thresholds, optional inversion. |
| `CharacteristicList(artist)` | Container for an artist's full set of characteristics. |
| `CharacteristicList.build_from_features(feature_dicts)` | Compute mean ± 1.5σ thresholds from a list of verified-work feature dicts. Automatically creates one `Characteristic` per numeric feature key. |
| `CharacteristicList.evaluate(feature_dict)` | Score a candidate's features against every characteristic. Returns a list of `(Characteristic, Verdict)` tuples. |
| `CharacteristicList.authenticate(feature_dict)` | Full assessment: computes `positive_pct`, selects `verdict` ("authentic" ≥ 75 %, "doubtful" 50–75 %, "other_hand" < 50 %), and returns per-characteristic details. |
| `CharacteristicList.save(path)` | Serialise the list (thresholds + stats) to JSON. |
| `CharacteristicList.load(path)` | Deserialise from JSON, returning a ready-to-use `CharacteristicList`. |
| `_infer_element(feature_key)` | Maps a feature key (e.g. `"stroke_length_mean"`) back to its van Dantzig element name (e.g. `"Brushstrokes (Element 8)"`). |

---

## Van Dantzig's 21 Pictorial Elements → Code Mapping

| # | Element | Pages | Computed? | Primary Function(s) |
|---|---|---|---|---|
| 1 | Subject / Representation | p. 13 | ✗ | Requires semantic understanding |
| 2 | Surface Organisation | pp. 13–14 | ✓ | `surface_organization()` |
| 3 | Materials / Technique | p. 14 | ◐ | `texture_rendering()` (partial) |
| 4 | Manipulation Variations | pp. 15–16 | ✓ | `pressure_variation()` |
| 5 | Colors | pp. 17–18 | ✓ | `color_analysis()` |
| 6 | Sequence of Development | pp. 18–19 | ✓ | `sequence_of_development()` |
| 7 | Drawn or Painted | p. 19 | ◐ | Encoded in brushstroke statistics |
| 8 | Lines / Planes / Relations | pp. 19–22 | ✓ | `extract_brushstrokes()` |
| 9 | Regularity | p. 22 | ◐ | Encoded in `rhythm_analysis()` |
| 10 | Negative Shapes | p. 22 | ◐ | Partially via contour analysis |
| 11 | Adornments | pp. 22–23 | ✗ | Requires semantic understanding |
| 12 | Light / Dark Relations | pp. 23–24 | ✓ | `light_dark_spatial()`, `contrast_analysis()` |
| 13 | Texture Rendering | pp. 24–25 | ✓ | `texture_rendering()` |
| 14 | Perspective / Projection | p. 25 | ✓ | `light_dark_spatial()` depth cues |
| 15 | Anatomy | pp. 25–26 | ✗ | Requires object detection |
| 16 | Contour / Interior | pp. 26–27 | ✓ | `contour_interior_relationship()` |
| 17 | Special Items | p. 27 | ✗ | Artist-specific, not generalisable |
| 18 | Style | p. 27 | ◐ | Aggregate of all features |
| 19 | Major / Minor Parts | pp. 27–29 | ✓ | `major_minor_analysis()`, `detail_density_analysis()` |
| 20 | Line and Rhythm | pp. 28–29 | ✓ | `rhythm_analysis()`, `hatching_analysis()` |
| 21 | Totality | pp. 28–29 | ✓ | `totality_score()` |

**Legend:** ✓ = fully implemented, ◐ = partially captured, ✗ = not computed

---

## Authentication Thresholds

Van Dantzig's decision boundaries (pp. 5–6), as implemented in `CharacteristicList.authenticate()`:

> *"If the number of characteristics which corresponded with the list was 75% or more of the maximum number available for comparison, the work was almost certainly by the painter to whom the list applied. If the fraction was less than 50 percent the work was by another hand provided sufficient integrity of surface. Between these two fractions further investigation is required or doubt will remain."* — p. 5

These thresholds were empirically validated against the Van Gogh corpus (p. 51): genuine works showed ≥ 76 % agreement, while the closest forgeries peaked at 45 %, confirming a clear separation.

| Positive % | Verdict | Interpretation |
|---|---|---|
| **≥ 75 %** | `authentic` | Almost certainly by the artist |
| **50 – 75 %** | `doubtful` | Needs further investigation |
| **< 50 %** | `other_hand` | By another hand |

Example applications in the book: Rembrandt's *Saint Bartholomew* scored 37.5 % positive → other hand (pp. 74–76); the Vienna *Lady* by Frans Hals scored 96 % positive → authentic, vs the Cincinnati *Lady* at 35 % → other hand (pp. 90–93).

---

## Dependencies

`numpy`, `scipy`, `scikit-image`, `Pillow` — all listed in the project's `pyproject.toml`.
