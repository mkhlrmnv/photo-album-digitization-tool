#!/usr/bin/env python3
"""
Visualize YOLO Classification Model Predictions

This script uses a YOLO classification model to predict the class of images
and displays the predicted class label in the top-left corner of each image.

Input structure:
    input/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

Usage:
    python visualize.py --model yolov11-cls.pt --input /path/to/input
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def list_images(root: Path):
    """List all image files in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def draw_label(image, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 255, 0), thickness=2):
    """
    Draw a label in the top-left corner of the image.

    Args:
        image: The image to draw on.
        label: The text label to draw.
        font: The font to use (default: cv2.FONT_HERSHEY_SIMPLEX).
        font_scale: The scale of the font (default: 1).
        color: The color of the text (default: green).
        thickness: The thickness of the text (default: 2).
    """
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x, text_y = 10, 10 + text_size[1]  # Top-left corner
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO classification model predictions.")
    parser.add_argument("--model", type=str, default='models/first-version.pt', help="Path to the YOLO classification model weights.")
    parser.add_argument("--input", type=Path, default='../../datasets/only-images/processed-by-segmentation', help="Path to the input folder containing images.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for inference (default: 224).")
    args = parser.parse_args()

    # Ensure input folder exists
    if not args.input.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input}")

    # List all images in the input folder
    images = list_images(args.input)
    if not images:
        raise ValueError(f"No images found in input folder: {args.input}")

    # Load YOLO classification model
    print(f"Loading YOLO classification model from {args.model}...")
    model = YOLO(args.model)

    # Process each image
    print(f"Processing {len(images)} images...")
    for image_path in tqdm(images, desc="Processing"):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠ Warning: Could not read image: {image_path}")
            continue

        # Run inference
        results = model.predict(source=str(image_path), imgsz=args.img_size, save=False, verbose=False)

        # Get the predicted class label
        if results and len(results) > 0:
            # breakpoint()
            predicted_class = results[0].names[results[0].probs.top1]
        else:
            predicted_class = "Unknown"

        # Draw the label on the image
        draw_label(image, predicted_class)

        # Display the image
        cv2.imshow("YOLO Classification", image)

        # Wait for a key press to move to the next image
        key = cv2.waitKey(0)
        if key == 27:  # Press 'Esc' to exit
            print("Exiting visualization...")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()