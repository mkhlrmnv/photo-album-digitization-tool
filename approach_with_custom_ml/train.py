"""
YOLO11 Segmentation Model Training Script

This script trains a YOLO11 segmentation model on a custom dataset.
Make sure you have ultralytics installed: pip install ultralytics
"""

from ultralytics import YOLO
import os
from pathlib import Path

# Configuration
MODEL_SIZE = "yolo11n-seg.pt"  # Options: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt
DATA_YAML = "test-dataset-for-segmentation/data.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0  # GPU device, use 'cpu' for CPU training
PROJECT_NAME = "runs/segment"
EXPERIMENT_NAME = "photo_segmentation"

# Hyperparameters
LEARNING_RATE = 0.01
PATIENCE = 50  # Early stopping patience
SAVE_PERIOD = 10  # Save checkpoint every N epochs

def train_model():
    """
    Train YOLO11 segmentation model
    """
    print("=" * 60)
    print("YOLO11 Segmentation Training")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Data configuration file not found: {DATA_YAML}")
    
    # Load pretrained YOLO11 segmentation model
    print("\nLoading model...")
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        
        # Hyperparameters
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,    # Scale augmentation
        shear=0.0,    # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,   # Flip up-down augmentation
        fliplr=0.5,   # Flip left-right augmentation
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
        
        # Validation
        val=True,
        plots=True,
        save=True,
        save_json=False,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Last model saved at: {results.save_dir}/weights/last.pt")
    print("=" * 60)
    
    return results

def validate_model(model_path):
    """
    Validate the trained model
    """
    print("\nValidating model...")
    model = YOLO(model_path)
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Train the model
    results = train_model()
    
    # Validate the best model
    best_model_path = f"{PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt"
    if os.path.exists(best_model_path):
        validate_model(best_model_path)
    else:
        print(f"\nWarning: Best model not found at {best_model_path}")