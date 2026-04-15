"""
Pictology — Computational implementation of van Dantzig's art authentication method.

Translates the 21 pictorial elements from M.M. van Dantzig (1973) "Pictology:
An Analytical Method for Attribution and Evaluation of Pictures" into
quantifiable image-analysis features.
"""

from .pipeline import PictologyPipeline
from .characteristic_list import CharacteristicList

__all__ = ["PictologyPipeline", "CharacteristicList"]
