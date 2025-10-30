"""
Dataset Split Script

This script splits a dataset with images and labels into train/val/test sets.
It supports both YOLO and CVAT format inputs and outputs.

Input structure (YOLO format):
    dataset/
    ├── images/
    │   └── (all images)
    └── labels/
        └── (all labels)

Input structure (CVAT format):
    dataset/
    ├── images/
    │   └── train/
    │       └── (all images)
    └── labels/
        └── train/
            └── (all labels)

Output structure (YOLO format):
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Output structure (CVAT format):
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── train.txt
    ├── val.txt
    └── test.txt

Usage:
    python dataset-split.py --source dataset --output dataset-split
    python dataset-split.py --source dataset --output dataset-split --format cvat
    python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def create_folder_structure(base_folder, output_format='yolo', splits=['train', 'val', 'test']):
    """Create the folder structure for train/val/test splits"""
    if output_format == 'yolo':
        for split in splits:
            os.makedirs(os.path.join(base_folder, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_folder, split, 'labels'), exist_ok=True)
    elif output_format == 'cvat':
        for split in splits:
            os.makedirs(os.path.join(base_folder, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(base_folder, 'labels', split), exist_ok=True)
    print("✓ Created folder structure")


def get_image_files(folder_path):
    """Get all image files from the folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    files = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Images folder not found: {folder_path}")
    
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


def transfer_files(files, source_img_folder, source_lbl_folder, 
                   dest_img_folder, dest_lbl_folder, move=True):
    """Move or copy image and label files to destination folders"""
    copied_count = 0
    missing_labels = []
    
    transfer_func = shutil.move if move else shutil.copy2
    
    for filename in files:
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Transfer image
        src_img = os.path.join(source_img_folder, filename)
        dst_img = os.path.join(dest_img_folder, filename)
        
        if os.path.exists(src_img):
            transfer_func(src_img, dst_img)
        else:
            print(f"⚠ Warning: Image not found: {src_img}")
            continue
        
        # Transfer label (assuming .txt extension)
        label_filename = base_name + '.txt'
        src_lbl = os.path.join(source_lbl_folder, label_filename)
        dst_lbl = os.path.join(dest_lbl_folder, label_filename)
        
        if os.path.exists(src_lbl):
            transfer_func(src_lbl, dst_lbl)
            copied_count += 1
        else:
            missing_labels.append(filename)
    
    return copied_count, missing_labels


def create_cvat_txt_file(output_path, split, filenames):
    """Create CVAT .txt file with image paths"""
    txt_path = output_path / f"{split}.txt"
    with open(txt_path, 'w') as f:
        for filename in sorted(filenames):
            f.write(f"data/images/{split}/{filename}\n")
    print(f"✓ Created {split}.txt with {len(filenames)} entries")


def detect_input_format(source_path):
    """Detect if input is in YOLO or CVAT format"""
    source_path = Path(source_path)
    
    # Check for CVAT format (images/train and labels/train exist)
    cvat_images = source_path / "images" / "train"
    cvat_labels = source_path / "labels" / "train"
    
    if cvat_images.exists() and cvat_labels.exists():
        return 'cvat'
    
    # Check for YOLO format (images/ and labels/ directly contain files)
    yolo_images = source_path / "images"
    yolo_labels = source_path / "labels"
    
    if yolo_images.exists() and yolo_labels.exists():
        # Check if they contain files directly (not subdirectories with files)
        has_image_files = any(
            f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            for f in yolo_images.iterdir() if f.is_file()
        )
        if has_image_files:
            return 'yolo'
    
    raise ValueError("Could not detect input format. Expected either:\n"
                    "  YOLO: images/ and labels/ with files directly\n"
                    "  CVAT: images/train/ and labels/train/ with files")


def split_dataset_flexible(source_path, output_path, train_ratio=0.7, val_ratio=0.15, 
                           test_ratio=0.15, seed=42, move=True, input_format=None, 
                           output_format='yolo'):
    """
    Split a dataset into train/val/test sets with flexible input/output formats
    
    Args:
        source_path: Path to source dataset
        output_path: Path to output dataset
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
        move: If True, move files; if False, copy files (default: True)
        input_format: Input format ('yolo' or 'cvat', auto-detect if None)
        output_format: Output format ('yolo' or 'cvat', default: 'yolo')
    """
    
    source_path = Path(source_path)
    output_path = Path(output_path)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    # Auto-detect input format if not specified
    if input_format is None:
        input_format = detect_input_format(source_path)
    
    print("=" * 60)
    print("Dataset Split")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Input format: {input_format.upper()}")
    print(f"Output format: {output_format.upper()}")
    print(f"Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {test_ratio*100}%")
    print(f"Mode: {'Move' if move else 'Copy'}")
    print(f"Random seed: {seed}")
    print("-" * 60)
    
    # Define source folders based on input format
    if input_format == 'cvat':
        source_images_folder = source_path / "images" / "train"
        source_labels_folder = source_path / "labels" / "train"
    else:  # yolo
        source_images_folder = source_path / "images"
        source_labels_folder = source_path / "labels"
    
    # Check if source folders exist
    if not source_images_folder.exists():
        raise FileNotFoundError(f"Source images folder not found: {source_images_folder}")
    if not source_labels_folder.exists():
        print(f"⚠ Warning: Source labels folder not found: {source_labels_folder}")
    
    # Create output folder structure
    create_folder_structure(output_path, output_format)
    
    # Get all image files
    all_files = get_image_files(source_images_folder)
    print(f"✓ Found {len(all_files)} images")
    
    if len(all_files) == 0:
        raise ValueError("No images found in source folder")
    
    # Split dataset
    train_files, val_files, test_files = split_dataset(
        all_files, train_ratio, val_ratio, test_ratio
    )
    
    print(f"✓ Split into: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    print("-" * 60)
    
    # Define output paths based on output format
    if output_format == 'cvat':
        output_paths = {
            'train': (output_path / 'images' / 'train', output_path / 'labels' / 'train'),
            'val': (output_path / 'images' / 'val', output_path / 'labels' / 'val'),
            'test': (output_path / 'images' / 'test', output_path / 'labels' / 'test')
        }
    else:  # yolo
        output_paths = {
            'train': (output_path / 'train' / 'images', output_path / 'train' / 'labels'),
            'val': (output_path / 'val' / 'images', output_path / 'val' / 'labels'),
            'test': (output_path / 'test' / 'images', output_path / 'test' / 'labels')
        }
    
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Transfer files for each split
    for split_name, files in splits_data.items():
        print(f"\nProcessing {split_name} files...")
        dest_images, dest_labels = output_paths[split_name]
        
        copied, missing = transfer_files(
            files, source_images_folder, source_labels_folder,
            dest_images, dest_labels, move=move
        )
        
        print(f"✓ Transferred {copied} {split_name} image-label pairs")
        if missing:
            print(f"⚠ Warning: {len(missing)} {split_name} images missing labels")
    
    # Create .txt files for CVAT format
    if output_format == 'cvat':
        print("\n" + "-" * 60)
        print("Creating CVAT .txt files...")
        for split_name, files in splits_data.items():
            create_cvat_txt_file(output_path, split_name, files)
    
    # Print summary
    print("\n" + "-" * 60)
    print("✓ Dataset split complete!")
    print(f"\nSummary:")
    print(f"  Train: {len(train_files)} images ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_files)} images ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_files)} images ({test_ratio*100:.1f}%)")
    print(f"  Total: {len(all_files)} images")
    print(f"\nOutput location: {output_path.absolute()}")
    print("=" * 60)
    
    return {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'total': len(all_files)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect input format, output in YOLO format
  python dataset-split.py --source dataset --output dataset-split
  
  # Specify CVAT input format, output in CVAT format
  python dataset-split.py --source dataset --output dataset-split --input-format cvat --output-format cvat
  
  # Custom split ratios
  python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
  
  # Copy files instead of moving
  python dataset-split.py --source dataset --output dataset-split --copy
  
  # CVAT input to YOLO output
  python dataset-split.py --source cvat-dataset --output yolo-dataset --input-format cvat --output-format yolo
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output dataset')
    parser.add_argument('--input-format', type=str, choices=['yolo', 'cvat'],
                        help='Input format (auto-detect if not specified)')
    parser.add_argument('--output-format', type=str, choices=['yolo', 'cvat'], default='yolo',
                        help='Output format (default: yolo)')
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
        move=not args.copy,
        input_format=args.input_format,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()