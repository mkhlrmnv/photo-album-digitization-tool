"""
Dataset Split Script for Classification YOLO Models

This script splits a dataset into train/val/test sets for classification tasks.
It assumes the dataset is organized into subdirectories, where each subdirectory
represents a class label.

Input structure:
    dataset/
    ├── class1/
    │   └── (all images for class1)
    ├── class2/
    │   └── (all images for class2)
    └── ...

Output structure:
    dataset-split/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── val/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...

Usage:
    python dataset-split.py --source dataset --output dataset-split
    python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def create_folder_structure(base_folder, classes, splits=['train', 'val', 'test']):
    """Create the folder structure for train/val/test splits"""
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_folder, split, class_name), exist_ok=True)
    print("✓ Created folder structure")


def get_class_folders(folder_path):
    """Get all class folders from the dataset"""
    class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    return class_folders


def get_image_files(folder_path):
    """Get all image files from the folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    files = []
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(filename)
    
    return files


def split_dataset(files, train_ratio, val_ratio, test_ratio):
    """Split files into train/val/test sets"""
    # Shuffle files
    random.shuffle(files)
    
    # Calculate split indices
    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def transfer_files(files, source_folder, dest_folder, move=True):
    """Move or copy image files to destination folders"""
    transfer_func = shutil.move if move else shutil.copy2
    
    for filename in files:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(dest_folder, filename)
        transfer_func(src, dst)


def split_dataset_flexible(source_path, output_path, train_ratio=0.7, val_ratio=0.15, 
                           test_ratio=0.15, seed=42, move=True):
    """
    Split a dataset into train/val/test sets for classification tasks
    
    Args:
        source_path: Path to source dataset
        output_path: Path to output dataset
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
        move: If True, move files; if False, copy files (default: True)
    """
    
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    print("=" * 60)
    print("Dataset Split for Classification")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%")
    print(f"Mode: {'Move' if move else 'Copy'}")
    print(f"Random seed: {seed}")
    print("-" * 60)
    
    # Get class folders
    class_folders = get_class_folders(source_path)
    if not class_folders:
        raise ValueError("No class folders found in the source dataset")
    
    print(f"✓ Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    # Create output folder structure
    create_folder_structure(output_path, class_folders)
    
    # Process each class folder
    for class_name in class_folders:
        print(f"\nProcessing class: {class_name}")
        source_class_folder = source_path / class_name
        
        # Get all image files for this class
        all_files = get_image_files(source_class_folder)
        print(f"  ✓ Found {len(all_files)} images")
        
        if len(all_files) == 0:
            print(f"  ⚠ Warning: No images found for class {class_name}")
            continue
        
        # Split dataset
        train_files, val_files, test_files = split_dataset(
            all_files, train_ratio, val_ratio, test_ratio
        )
        
        print(f"  ✓ Split into: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Transfer files
        for split_name, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            dest_folder = output_path / split_name / class_name
            transfer_files(files, source_class_folder, dest_folder, move=move)
            print(f"  ✓ Transferred {len(files)} {split_name} files")
    
    # Print summary
    print("\n" + "-" * 60)
    print("✓ Dataset split complete!")
    print(f"\nOutput location: {output_path.absolute()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets for classification tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split ratios (70% train, 15% val, 15% test)
  python dataset-split.py --source dataset --output dataset-split
  
  # Custom split ratios
  python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
  
  # Copy files instead of moving
  python dataset-split.py --source dataset --output dataset-split --copy
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output dataset')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios for train/val/test (default: 0.7 0.15 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of moving them')
    
    args = parser.parse_args()
    
    # Validate ratios
    train_ratio, val_ratio, test_ratio = args.ratios
    if abs(sum(args.ratios) - 1.0) > 0.001:
        parser.error("Ratios must sum to 1.0")
    
    # Split dataset
    split_dataset_flexible(
        source_path=args.source,
        output_path=args.output,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
        move=not args.copy
    )


if __name__ == "__main__":
    main()