"""
CVAT TXT File Generator

This script generates .txt files for CVAT dataset format.
It scans the images directory and creates corresponding .txt files
with paths to all images in each split (train/val/test).

Usage:
    python cvat-txt-file-generator.py --dataset path/to/dataset
    python cvat-txt-file-generator.py --dataset path/to/dataset --splits train val test
"""

import os
import argparse
from pathlib import Path


def get_image_files(folder_path):
    """Get all image files from a folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    files = []
    
    if not os.path.exists(folder_path):
        print(f"⚠ Warning: Folder not found: {folder_path}")
        return files
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(filename)
    
    return sorted(files)


def generate_txt_file(dataset_path, split):
    """Generate .txt file for a specific split (train/val/test)"""
    
    # Define paths
    images_folder = os.path.join(dataset_path, "images", split)
    output_file = os.path.join(dataset_path, f"{split}.txt")
    
    # Get all image files
    image_files = get_image_files(images_folder)
    
    if not image_files:
        print(f"⚠ No images found in {images_folder}")
        return 0
    
    # Write to output file
    with open(output_file, 'w') as f:
        for filename in image_files:
            # Write the relative path format for CVAT
            f.write(f"data/images/{split}/{filename}\n")
    
    print(f"✓ Created {split}.txt with {len(image_files)} images")
    return len(image_files)


def generate_all_txt_files(dataset_path, splits=['train', 'val', 'test']):
    """Generate .txt files for all specified splits"""
    
    print("=" * 60)
    print("CVAT TXT File Generator")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Generating files for: {', '.join(splits)}")
    print("-" * 60)
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Check if images folder exists
    images_path = os.path.join(dataset_path, "images")
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images folder not found: {images_path}")
    
    # Generate txt files for each split
    total_images = 0
    results = {}
    
    for split in splits:
        count = generate_txt_file(dataset_path, split)
        results[split] = count
        total_images += count
    
    # Print summary
    print("-" * 60)
    print("✓ Generation complete!")
    print(f"\nSummary:")
    for split, count in results.items():
        if count > 0:
            print(f"  {split}: {count} images")
    print(f"  Total: {total_images} images")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate CVAT .txt files with image paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate txt files for all splits (train, val, test)
  python cvat-txt-file-generator.py --dataset test-dataset-cvat
  
  # Generate only for specific splits
  python cvat-txt-file-generator.py --dataset test-dataset-cvat --splits train val
  
  # Generate only for training set
  python cvat-txt-file-generator.py --dataset test-dataset-cvat --splits train
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset root folder')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Splits to generate txt files for (default: train val test)')
    
    args = parser.parse_args()
    
    # Generate txt files
    generate_all_txt_files(args.dataset, args.splits)


if __name__ == "__main__":
    main()