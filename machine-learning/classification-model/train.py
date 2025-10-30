#!/usr/bin/env python3
"""
Train a YOLOv11 classification model on a dataset.

Dataset structure:
    dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   ├── image1.jpg
    │   │   └── ...
    ├── val/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/ (optional)
        ├── class1/
        ├── class2/
        └── ...

Usage:
    python training.py --data dataset --epochs 50 --batch-size 32 --img-size 224
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv11 classification model.")
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32).")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for training (default: 224).")
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt", help="Path to the YOLOv11 classification model weights.")
    parser.add_argument("--project", type=str, default="runs/train", help="Project folder to save training results.")
    parser.add_argument("--name", type=str, default="rotation-model", help="Name of the experiment folder.")
    args = parser.parse_args()

    # Initialize YOLO model
    print(f"Loading YOLOv11 classification model from {args.model}...")
    model = YOLO(args.model)  # Load YOLOv11 classification model

    # Train the model
    print("Starting training...")
    model.train(
        data=args.data,  # Path to dataset
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.project,
        name=args.name,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()