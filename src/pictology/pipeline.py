"""
PictologyPipeline — Orchestrates the full van Dantzig analysis.

Takes an image (path or array) and runs all 21 pictorial element analyses,
returning a unified feature dictionary and optional authentication verdict.
"""

from pathlib import Path
import numpy as np
from PIL import Image

from .spontaneity import line_spontaneity_score, contrast_analysis, detail_density_analysis
from .brushstroke import extract_brushstrokes, hatching_analysis, pressure_variation
from .surface_color import surface_organization, color_analysis, light_dark_spatial, texture_rendering
from .contour_rhythm import contour_interior_relationship, rhythm_analysis, totality_score
from .construction import sequence_of_development, major_minor_analysis
from .characteristic_list import CharacteristicList


class PictologyPipeline:
    """
    End-to-end pictological analysis of a painting image.

    Implements a computational translation of van Dantzig's 21 pictorial
    elements into measurable features. Designed for high-resolution
    digitized artworks.

    Workflow:
        1. Extract features from verified works → build CharacteristicList
        2. Extract features from candidate work
        3. Authenticate candidate against the list

    Example:
        >>> pipe = PictologyPipeline()
        >>> features = pipe.extract_features("painting.jpg")
        >>> # Build list from known works
        >>> clist = CharacteristicList(artist="Van Gogh")
        >>> verified = [pipe.extract_features(p) for p in verified_paths]
        >>> clist.build_from_features(verified)
        >>> # Authenticate
        >>> result = clist.authenticate(features)
        >>> print(result["verdict"], result["positive_pct"])
    """

    def __init__(self, max_size: int | None = 1024):
        """
        Args:
            max_size: Maximum dimension for processing. Larger preserves more
                      texture detail but is slower. Set to None to skip resizing.
        """
        self.max_size = max_size

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load image as float32 RGB [0,1], resized to max_size."""
        # Allow loading large images by increasing PIL's decompression bomb limit
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if self.max_size is not None and max(w, h) > self.max_size:
            scale = self.max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    def to_gray(self, img_rgb: np.ndarray) -> np.ndarray:
        return 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]

    def extract_features(self, image: str | Path | np.ndarray) -> dict:
        """
        Extract all pictological features from an image.

        Maps to van Dantzig's 21 pictorial elements:

          Identity elements (1-11):
            1. Subject/representation     — not computed (requires semantic understanding)
            2. Surface organization        → surface_organization()
            3. Materials/technique         — partially via texture_rendering()
            4. Manipulation variations     → pressure_variation()
            5. Colors                      → color_analysis()
            6. Sequence of development     → sequence_of_development()
            7. Drawn or painted            — encoded in brushstroke stats
            8. Lines/planes/relationships  → extract_brushstrokes()
            9. Regularity                  — encoded in rhythm_analysis()
           10. Negative shapes             — partially via contour analysis
           11. Adornments                  — not computed (requires semantic understanding)

          Identity + Quality (12-17):
           12. Light/dark relations         → light_dark_spatial(), contrast_analysis()
           13. Texture rendering            → texture_rendering()
           14. Perspective/projection       → light_dark_spatial() depth cues
           15. Anatomy                      — not computed (requires object detection)
           16. Contour/interior             → contour_interior_relationship()
           17. Special items                — not computed (artist-specific)

          Quality (18-21):
           18. Style                        — aggregate of all features
           19. Major/minor parts            → major_minor_analysis(), detail_density_analysis()
           20. Line and rhythm              → rhythm_analysis(), hatching_analysis()
           21. Totality                     → totality_score()

        Returns:
            Dictionary of ~60 numerical features keyed by descriptive name.
        """
        if isinstance(image, (str, Path)):
            img_rgb = self.load_image(image)
        else:
            img_rgb = image

        gray = self.to_gray(img_rgb)
        features = {}

        # Element 2: Surface organization
        features.update(_prefix("surface", surface_organization(img_rgb)))

        # Elements 3-4: Materials/technique, manipulation
        features.update(_prefix("pressure", pressure_variation(gray)))

        # Element 5: Colors
        features.update(_prefix("color", color_analysis(img_rgb)))

        # Element 6: Sequence of development
        features.update(_prefix("sequence", sequence_of_development(gray)))

        # Element 8: Lines, planes, brushstrokes
        features.update(_prefix("stroke", extract_brushstrokes(gray)))

        # Element 12: Light/dark relations + contrast
        features.update(_prefix("lightdark", light_dark_spatial(img_rgb)))
        features.update(_prefix("contrast", contrast_analysis(img_rgb)))

        # Element 13: Texture rendering
        features.update(_prefix("texture", texture_rendering(gray)))

        # Element 16: Contour/interior
        features.update(_prefix("contour", contour_interior_relationship(gray)))

        # Element 19: Major/minor parts
        features.update(_prefix("majmin", major_minor_analysis(img_rgb)))
        features.update(_prefix("detail", detail_density_analysis(gray)))

        # Element 20: Rhythm + hatching
        features.update(_prefix("rhythm", rhythm_analysis(gray)))
        features.update(_prefix("hatch", hatching_analysis(gray)))

        # Quality: Spontaneity (Chapter I)
        features.update(_prefix("spont", line_spontaneity_score(gray)))

        # Element 21: Totality (meta-score across all elements)
        features["totality_score"] = totality_score(features)

        return features

    def build_artist_profile(
        self, artist_name: str, image_paths: list[str | Path]
    ) -> CharacteristicList:
        """
        Build a characteristic list from verified authentic works.

        Van Dantzig used ~50 undoubtedly genuine works per artist.
        Minimum recommended: 10+ high-quality images.
        """
        feature_dicts = [self.extract_features(p) for p in image_paths]
        clist = CharacteristicList(artist=artist_name)
        clist.build_from_features(feature_dicts)
        return clist

    def authenticate(
        self,
        candidate: str | Path | np.ndarray,
        characteristic_list: CharacteristicList,
    ) -> dict:
        """
        Authenticate a candidate work against an artist's characteristic list.

        Returns dict with:
            artist: artist name
            positive_pct: % matching characteristics
            verdict: "authentic" / "doubtful" / "other_hand"
            details: per-characteristic breakdown
        """
        features = self.extract_features(candidate)
        return characteristic_list.authenticate(features)


def _prefix(prefix: str, d: dict) -> dict:
    """Prefix all keys in a dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool)):
            result[f"{prefix}_{k}"] = v
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            # Store list features as individual indexed entries
            for i, val in enumerate(v):
                result[f"{prefix}_{k}_{i}"] = val
        # Skip non-numeric values (strings, nested lists) for the feature vector
    return result
