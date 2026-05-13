"""
Create training patches from high-res Manet images.

Collects Manet images, slices them into 256x256 patches with 50% overlap,
and saves them to data/manet/patches/ for training.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from tqdm import tqdm

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ──
MANET_SOURCE = Path(r"C:\Users\MichelleJacobs\OneDrive - Jongens van Techniek\JVT _ Engineering - Authentication in Art\AIA - AI Tool - Fingerprinting\Manet Website download")
OUTPUT_DIR = ROOT / "data" / "manet" / "authentic" / "patches"
PATCH_SIZE = 256
OVERLAP_RATIO = 0.5  # 50% overlap
MIN_IMAGE_SIZE = 512  # Skip images smaller than this (too small for meaningful patches)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Derived
STRIDE = int(PATCH_SIZE * (1 - OVERLAP_RATIO))  # 128 pixels


def extract_patches(img: np.ndarray, patch_size: int = 256, stride: int = 128) -> Iterator[np.ndarray]:
    """
    Extract overlapping patches from an image.
    
    Args:
        img: RGB image array of shape (H, W, 3)
        patch_size: Size of square patches
        stride: Step size between patches (smaller = more overlap)
        
    Yields:
        Patches of shape (patch_size, patch_size, 3)
    """
    h, w = img.shape[:2]
    
    # Calculate number of patches
    n_rows = (h - patch_size) // stride + 1
    n_cols = (w - patch_size) // stride + 1
    
    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            
            # Ensure we don't go over the edge
            if y + patch_size <= h and x + patch_size <= w:
                patch = img[y:y+patch_size, x:x+patch_size]
                yield patch


def process_image(img_path: Path, output_dir: Path) -> int:
    """
    Process a single image: load, extract patches, save.
    
    Returns:
        Number of patches extracted
    """
    try:
        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Skip small images
        if min(w, h) < MIN_IMAGE_SIZE:
            return 0
        
        img_array = np.array(img)
        
        # Extract patches
        patch_count = 0
        stem = img_path.stem
        
        for idx, patch in enumerate(extract_patches(img_array, PATCH_SIZE, STRIDE)):
            # Save patch
            patch_name = f"{stem}_patch_{idx:04d}.png"
            patch_path = output_dir / patch_name
            
            patch_img = Image.fromarray(patch)
            patch_img.save(patch_path, "PNG")
            patch_count += 1
        
        return patch_count
    
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return 0


def main():
    print("=== Manet Patch Extraction ===")
    print(f"Source: {MANET_SOURCE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Overlap: {OVERLAP_RATIO*100:.0f}% (stride: {STRIDE}px)")
    print()
    
    # Check source directory exists
    if not MANET_SOURCE.exists():
        print(f"ERROR: Source directory not found: {MANET_SOURCE}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    image_files = sorted([
        f for f in MANET_SOURCE.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()
    ])
    
    print(f"Found {len(image_files)} images in source directory")
    
    if not image_files:
        print("No images found!")
        return
    
    # Process each image
    total_patches = 0
    processed_images = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        patch_count = process_image(img_path, OUTPUT_DIR)
        if patch_count > 0:
            total_patches += patch_count
            processed_images += 1
    
    print()
    print(f"✓ Processing complete!")
    print(f"  Images processed: {processed_images}/{len(image_files)}")
    print(f"  Total patches extracted: {total_patches}")
    print(f"  Average patches per image: {total_patches/processed_images:.1f}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Create a metadata file
    metadata_path = OUTPUT_DIR / "README.txt"
    with open(metadata_path, "w") as f:
        f.write(f"Manet Training Patches\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Source: {MANET_SOURCE}\n")
        f.write(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n")
        f.write(f"Overlap: {OVERLAP_RATIO*100:.0f}% (stride: {STRIDE}px)\n")
        f.write(f"Images processed: {processed_images}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Created: {Path(__file__).name}\n")


if __name__ == "__main__":
    main()
