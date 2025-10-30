#!/usr/bin/env python3
"""
Rotate images by 90°, 180°, or 270° counterclockwise and save them into separate folders.

Input structure:
    source/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

Output structure:
    output/
    ├── 90/
    │   ├── image1.jpg
    │   └── ...
    ├── 180/
    │   ├── image1.jpg
    │   └── ...
    └── 270/
        ├── image1.jpg
        └── ...

Usage:
    python rotate-images.py --src /path/to/source --out /path/to/output
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def ensure_dir(directory: Path):
    """Ensure a directory exists."""
    directory.mkdir(parents=True, exist_ok=True)


def list_images(root: Path):
    """List all image files in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def rotate_and_save(image_path: Path, output_dir: Path, angles=(0, 90, 180, 270)):
    """
    Rotate an image by specified angles and save to corresponding folders.

    Args:
        image_path (Path): Path to the input image.
        output_dir (Path): Path to the output directory.
        angles (tuple): Angles to rotate the image (default: 90, 180, 270).
    """
    img = Image.open(image_path)
    for angle in angles:
        rotated_img = img.rotate(angle, expand=True)
        angle_dir = output_dir / str(angle)
        ensure_dir(angle_dir)
        rotated_img.save(angle_dir / image_path.name, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Rotate images and save to separate folders.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the source folder containing images.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output folder.")
    parser.add_argument("--angles", type=int, nargs="+", default=[0, 90, 180, 270],
                        help="Angles to rotate the images (default: 90, 180, 270).")
    args = parser.parse_args()

    # Ensure source folder exists
    if not args.src.exists():
        raise FileNotFoundError(f"Source folder not found: {args.src}")

    # List all images in the source folder
    images = list_images(args.src)
    if not images:
        raise ValueError(f"No images found in source folder: {args.src}")

    # Rotate and save images
    print(f"Rotating {len(images)} images...")
    for image_path in tqdm(images, desc="Processing"):
        rotate_and_save(image_path, args.out, angles=args.angles)

    print(f"Done! Rotated images saved to: {args.out}")


if __name__ == "__main__":
    main()