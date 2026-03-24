"""
Shared utilities for Part A research notebooks.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Image loading ────────────────────────────────────────────────────────────

def load_image(path: str | Path, max_size: int = 1024) -> np.ndarray:
    """
    Load an image as a float32 RGB array in [0, 1].
    Downsamples the long edge to max_size to keep computation fast
    while retaining texture structure.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance: standard ITU-R BT.709 weights."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


# ── Display helpers ───────────────────────────────────────────────────────────

def show_pair(img_a, img_b, title_a="Manet", title_b="Contemporary", figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [img_a, img_b], [title_a, title_b]):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    plt.tight_layout()
    return fig


def heatmap_overlay(img: np.ndarray, heat: np.ndarray, title="", alpha=0.6, figsize=(8, 6)):
    """Overlay a heatmap on an image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    h = ax.imshow(heat, alpha=alpha, cmap="plasma",
                  extent=[0, img.shape[1], img.shape[0], 0])
    plt.colorbar(h, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    return fig
