"""
Characteristic list — van Dantzig's Appendix scoring system.

Manages the per-artist list of characteristics and provides the
quantitative scoring framework (75% threshold = authentic, <50% = other hand).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class Verdict(Enum):
    POSITIVE = "+"      # Feature matches the artist's list
    NEGATIVE = "-"      # Feature does not match
    NOT_APPLICABLE = "0"  # Feature cannot be evaluated for this work


@dataclass
class Characteristic:
    """A single entry in an artist's characteristic list."""
    id: int
    element: str               # Pictorial element category (Element 1-21)
    description: str           # Human-readable description
    feature_key: Optional[str] = None  # Key in computed feature dict
    threshold_low: Optional[float] = None   # Below = negative
    threshold_high: Optional[float] = None  # Above = positive
    invert: bool = False       # If True, below threshold = positive


@dataclass
class CharacteristicList:
    """
    Van Dantzig's characteristic list for a specific artist.

    Usage:
        1. Build from known authentic works (build_from_features)
        2. Score a candidate work (evaluate)
        3. Get authentication verdict (authenticate)

    Thresholds (van Dantzig):
        ≥75% positive → almost certainly by the artist
        50-75% → doubtful, needs investigation
        <50%  → by another hand
    """
    artist: str
    characteristics: list[Characteristic] = field(default_factory=list)
    _feature_stats: dict = field(default_factory=dict, repr=False)

    def build_from_features(self, feature_dicts: list[dict]) -> None:
        """
        Build characteristic thresholds from a set of verified authentic works.

        For each computed feature, establishes the range seen in authentic works
        and sets thresholds at ±1.5 std from the mean.
        """
        if not feature_dicts:
            return

        # Collect all feature keys
        all_keys = set()
        for fd in feature_dicts:
            all_keys.update(k for k, v in fd.items() if isinstance(v, (int, float)))

        self._feature_stats = {}
        self.characteristics = []

        for idx, key in enumerate(sorted(all_keys), start=1):
            values = [fd[key] for fd in feature_dicts if key in fd and isinstance(fd[key], (int, float))]
            if len(values) < 2:
                continue

            import numpy as np
            mean = float(np.mean(values))
            std = float(np.std(values))

            self._feature_stats[key] = {"mean": mean, "std": std, "n": len(values)}

            self.characteristics.append(Characteristic(
                id=idx,
                element=_infer_element(key),
                description=f"Computed: {key}",
                feature_key=key,
                threshold_low=mean - 1.5 * std,
                threshold_high=mean + 1.5 * std,
            ))

    def evaluate(self, feature_dict: dict) -> list[tuple[Characteristic, Verdict]]:
        """
        Score a candidate work against the characteristic list.

        Returns list of (characteristic, verdict) tuples.
        """
        results = []
        for char in self.characteristics:
            if char.feature_key is None or char.feature_key not in feature_dict:
                results.append((char, Verdict.NOT_APPLICABLE))
                continue

            val = feature_dict[char.feature_key]
            if not isinstance(val, (int, float)):
                results.append((char, Verdict.NOT_APPLICABLE))
                continue

            if char.threshold_low is not None and char.threshold_high is not None:
                in_range = char.threshold_low <= val <= char.threshold_high
                if char.invert:
                    in_range = not in_range
                verdict = Verdict.POSITIVE if in_range else Verdict.NEGATIVE
            else:
                verdict = Verdict.NOT_APPLICABLE

            results.append((char, verdict))

        return results

    def authenticate(self, feature_dict: dict) -> dict:
        """
        Full authentication assessment.

        Returns:
            artist: the artist name
            positive_pct: percentage of matching characteristics
            verdict: "authentic" / "doubtful" / "other_hand"
            details: per-characteristic verdicts
        """
        results = self.evaluate(feature_dict)

        applicable = [(c, v) for c, v in results if v != Verdict.NOT_APPLICABLE]
        if not applicable:
            return {
                "artist": self.artist,
                "positive_pct": 0.0,
                "verdict": "insufficient_data",
                "n_evaluated": 0,
                "n_positive": 0,
                "n_negative": 0,
                "n_not_applicable": len(results),
            }

        n_pos = sum(1 for _, v in applicable if v == Verdict.POSITIVE)
        n_neg = sum(1 for _, v in applicable if v == Verdict.NEGATIVE)
        pct = 100.0 * n_pos / len(applicable)

        if pct >= 75:
            verdict = "authentic"
        elif pct >= 50:
            verdict = "doubtful"
        else:
            verdict = "other_hand"

        return {
            "artist": self.artist,
            "positive_pct": round(pct, 1),
            "verdict": verdict,
            "n_evaluated": len(applicable),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_not_applicable": len(results) - len(applicable),
            "details": [
                {
                    "id": c.id,
                    "element": c.element,
                    "description": c.description,
                    "feature_key": c.feature_key,
                    "verdict": v.value,
                }
                for c, v in results
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save characteristic list to JSON."""
        data = {
            "artist": self.artist,
            "feature_stats": self._feature_stats,
            "characteristics": [
                {
                    "id": c.id,
                    "element": c.element,
                    "description": c.description,
                    "feature_key": c.feature_key,
                    "threshold_low": c.threshold_low,
                    "threshold_high": c.threshold_high,
                    "invert": c.invert,
                }
                for c in self.characteristics
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CharacteristicList":
        """Load characteristic list from JSON."""
        data = json.loads(Path(path).read_text())
        clist = cls(artist=data["artist"])
        clist._feature_stats = data.get("feature_stats", {})
        clist.characteristics = [
            Characteristic(**c) for c in data["characteristics"]
        ]
        return clist


def _infer_element(feature_key: str) -> str:
    """Infer the van Dantzig pictorial element from the feature key."""
    mapping = {
        "spontan": "Quality/Spontaneity",
        "curvature": "Lines (Element 8)",
        "junction": "Lines (Element 8)",
        "width": "Lines/Brushstrokes (Element 8)",
        "contrast": "Light/Dark (Element 12)",
        "warm": "Colors (Element 5)",
        "cool": "Colors (Element 5)",
        "detail": "Major/Minor Parts (Element 19)",
        "contour": "Contour/Interior (Element 16)",
        "rhythm": "Rhythm (Element 20)",
        "coherence": "Rhythm (Element 20)",
        "orientation": "Hatching (Element 20)",
        "texture": "Texture (Element 13)",
        "pressure": "Manipulation (Element 4)",
        "stroke": "Brushstrokes (Element 8)",
        "hue": "Colors (Element 5)",
        "saturation": "Colors (Element 5)",
        "luminance": "Light/Dark (Element 12)",
        "layer": "Sequence (Element 6)",
        "saved": "Sequence (Element 6)",
        "major": "Major/Minor (Element 19)",
        "minor": "Major/Minor (Element 19)",
        "surface": "Surface Organization (Element 2)",
        "zone": "Surface Organization (Element 2)",
        "depth": "Perspective (Element 14)",
    }
    key_lower = feature_key.lower()
    for pattern, element in mapping.items():
        if pattern in key_lower:
            return element
    return "General"
