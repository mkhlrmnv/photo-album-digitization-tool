"""
Dataset Format Converter

Converts between CVAT-ready and YOLO-ready dataset formats.

CVAT Format:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── data.yaml
    ├── train.txt
    ├── val.txt
    └── test.txt

YOLO Format:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import os
import shutil
import yaml
import zipfile
from pathlib import Path

class DatasetConverter:
    def __init__(self, source_path, target_path):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
    
    def cvat_to_yolo(self):
        """Convert CVAT format to YOLO format"""
        print("=" * 60)
        print("Converting CVAT format to YOLO format")
        print("=" * 60)
        
        # Read CVAT data.yaml
        cvat_yaml_path = self.source_path / "data.yaml"
        if not cvat_yaml_path.exists():
            raise FileNotFoundError(f"CVAT data.yaml not found at {cvat_yaml_path}")
        
        with open(cvat_yaml_path, 'r') as f:
            cvat_config = yaml.safe_load(f)
        
        # Determine which splits exist
        splits = []
        if (self.source_path / "train.txt").exists():
            splits.append('train')
        if (self.source_path / "val.txt").exists():
            splits.append('val')
        if (self.source_path / "test.txt").exists():
            splits.append('test')
        
        print(f"Found splits: {', '.join(splits)}")
        
        # Create YOLO directory structure
        for split in splits:
            (self.target_path / split / "images").mkdir(parents=True, exist_ok=True)
            (self.target_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy files for each split
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            source_images = self.source_path / "images" / split
            source_labels = self.source_path / "labels" / split
            target_images = self.target_path / split / "images"
            target_labels = self.target_path / split / "labels"
            
            if not source_images.exists():
                print(f"  ⚠ Warning: {source_images} not found, skipping")
                continue
            
            # Copy images
            image_files = list(source_images.glob("*"))
            for img_file in image_files:
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    shutil.copy2(img_file, target_images / img_file.name)
            
            # Copy labels
            if source_labels.exists():
                label_files = list(source_labels.glob("*.txt"))
                for lbl_file in label_files:
                    shutil.copy2(lbl_file, target_labels / lbl_file.name)
                print(f"  ✓ Copied {len(image_files)} images and {len(label_files)} labels")
            else:
                print(f"  ⚠ Warning: {source_labels} not found")
        
        # Create YOLO data.yaml
        yolo_config = {
            'path': str(self.target_path.absolute()),
            'train': 'train/images' if 'train' in splits else None,
            'val': 'val/images' if 'val' in splits else 'train/images', # <- if there isn't val sets train set to be val as well
            'test': 'test/images' if 'test' in splits else None,
            'names': cvat_config.get('names', ['Photo']),
            'nc': len(cvat_config.get('names', ['Photo']))
        }
        
        # Remove None values
        yolo_config = {k: v for k, v in yolo_config.items() if v is not None}
        
        yolo_yaml_path = self.target_path / "data.yaml"
        with open(yolo_yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False)
        
        print("\n" + "=" * 60)
        print("✓ Conversion complete!")
        print(f"YOLO dataset saved at: {self.target_path}")
        print(f"data.yaml created at: {yolo_yaml_path}")
        print("=" * 60)
    
    def yolo_to_cvat(self, zip_output=True, remove_temp=True):
        """Convert YOLO format to CVAT format"""
        print("=" * 60)
        print("Converting YOLO format to CVAT format")
        print("=" * 60)
        
        # Use a temporary directory for conversion
        temp_dir = self.target_path
        if zip_output:
            temp_dir = Path(str(self.target_path) + "_temp")
        
        # Read YOLO data.yaml
        yolo_yaml_path = self.source_path / "data.yaml"
        if not yolo_yaml_path.exists():
            raise FileNotFoundError(f"YOLO data.yaml not found at {yolo_yaml_path}")
        
        with open(yolo_yaml_path, 'r') as f:
            yolo_config = yaml.safe_load(f)
        
        # Determine which splits exist
        splits = []
        if (self.source_path / "train").exists():
            splits.append('train')
        if (self.source_path / "val").exists():
            splits.append('val')
        if (self.source_path / "test").exists():
            splits.append('test')
        
        print(f"Found splits: {', '.join(splits)}")
        
        # Create CVAT directory structure
        (temp_dir / "images").mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy files for each split
        txt_files = {}
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            source_images = self.source_path / split / "images"
            source_labels = self.source_path / split / "labels"
            target_images = temp_dir / "images" / split
            target_labels = temp_dir / "labels" / split
            
            target_images.mkdir(parents=True, exist_ok=True)
            target_labels.mkdir(parents=True, exist_ok=True)
            
            if not source_images.exists():
                print(f"  ⚠ Warning: {source_images} not found, skipping")
                continue
            
            # Copy images and collect filenames
            image_files = []
            for img_file in source_images.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    shutil.copy2(img_file, target_images / img_file.name)
                    image_files.append(img_file.name)
            
            # Copy labels
            label_count = 0
            if source_labels.exists():
                for lbl_file in source_labels.glob("*.txt"):
                    shutil.copy2(lbl_file, target_labels / lbl_file.name)
                    label_count += 1
            
            print(f"  ✓ Copied {len(image_files)} images and {label_count} labels")
            
            # Create .txt file with image paths
            txt_content = [f"data/images/{split}/{img}\n" for img in sorted(image_files)]
            txt_files[split] = txt_content
        
        # Write .txt files
        for split, content in txt_files.items():
            txt_path = temp_dir / f"{split}.txt"
            with open(txt_path, 'w') as f:
                f.writelines(content)
            print(f"\n✓ Created {split}.txt with {len(content)} entries")
        
        # Create CVAT data.yaml
        cvat_config = {
            'names': {i: name for i, name in enumerate(yolo_config.get('names', ['Photo']))},
            'path': '.',
            'train': 'train.txt' if 'train' in splits else None,
            'val': 'val.txt' if 'val' in splits else None,
            'test': 'test.txt' if 'test' in splits else None
        }
        
        # Remove None values
        cvat_config = {k: v for k, v in cvat_config.items() if v is not None}
        
        cvat_yaml_path = temp_dir / "data.yaml"
        with open(cvat_yaml_path, 'w') as f:
            yaml.dump(cvat_config, f, default_flow_style=False)
        
        # Zip the output if requested
        if zip_output:
            print("\n" + "=" * 60)
            print("Creating zip archive...")
            zip_path = Path(str(self.target_path) + ".zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
            
            print(f"✓ Created zip archive: {zip_path}")
            
            # Remove temporary directory if requested
            if remove_temp:
                print("Removing temporary files...")
                shutil.rmtree(temp_dir)
                print("✓ Temporary files removed")
            
            print("\n" + "=" * 60)
            print("✓ Conversion complete!")
            print(f"CVAT dataset zipped at: {zip_path}")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✓ Conversion complete!")
            print(f"CVAT dataset saved at: {temp_dir}")
            print(f"data.yaml created at: {cvat_yaml_path}")
            print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert between CVAT and YOLO dataset formats')
    parser.add_argument('--mode', type=str, required=True, choices=['cvat2yolo', 'yolo2cvat'],
                        help='Conversion mode: cvat2yolo or yolo2cvat')
    parser.add_argument('--source', type=str, required=True,
                        help='Source dataset path')
    parser.add_argument('--target', type=str, required=True,
                        help='Target dataset path')
    
    args = parser.parse_args()
    
    converter = DatasetConverter(args.source, args.target)
    
    if args.mode == 'cvat2yolo':
        converter.cvat_to_yolo()
    elif args.mode == 'yolo2cvat':
        converter.yolo_to_cvat()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()