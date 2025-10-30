import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library
import cv2

def run_yolo_and_save_cvat(model_path, input_folder, output_folder):
    """
    Run a YOLO segmentation model on all images in a folder and save results in CVAT format.

    Args:
        model_path (str): Path to the YOLO model.
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where the CVAT dataset will be saved.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Define paths
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    images_train_folder = output_folder / "images" / "train"
    labels_train_folder = output_folder / "labels" / "train"

    # Create CVAT directory structure
    images_train_folder.mkdir(parents=True, exist_ok=True)
    labels_train_folder.mkdir(parents=True, exist_ok=True)

    # Process images
    image_files = list(input_folder.glob("*"))
    train_txt_content = []

    for img_file in image_files:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            print(f"Processing {img_file.name}...")
            results = model.predict(source=str(img_file), save=False, save_txt=False)

            # Save image to train folder
            target_image_path = images_train_folder / img_file.name
            shutil.copy2(img_file, target_image_path)

            # Save labels
            for result in results:
                label_file_name = img_file.stem + ".txt"
                label_file_path = labels_train_folder / label_file_name

                # Load the image to get its dimensions
                img = cv2.imread(str(img_file))
                img_height, img_width = img.shape[:2]

                with open(label_file_path, 'w') as f:
                    for mask in result.masks.xy:  # Extract polygons for each mask
                        # Calculate the bounding box of the mask
                        x_coords = [x for x, y in mask]
                        y_coords = [y for x, y in mask]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        # Normalize the bounding box coordinates to fractions of the image dimensions
                        x_min /= img_width
                        x_max /= img_width
                        y_min /= img_height
                        y_max /= img_height

                        # Save the bounding box as a mask with four points
                        mask_points = f"{x_min:.6f} {y_min:.6f} {x_max:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {x_min:.6f} {y_max:.6f}"
                        f.write(f"0 {mask_points}\n")  # Class ID is assumed to be 0


            # Add image path to train.txt
            train_txt_content.append(f"data/images/train/{img_file.name}\n")

    # Write train.txt
    train_txt_path = output_folder / "train.txt"
    with open(train_txt_path, 'w') as f:
        f.writelines(train_txt_content)

    # Create data.yaml
    data_yaml = {
        'names': {0: 'Photo'},  # Update with your class names
        'path': '.',
        'train': 'train.txt',
    }
    data_yaml_path = output_folder / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\nConversion complete!")
    print(f"CVAT dataset saved at: {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO model on images and save results in CVAT format")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output", type=str, required=True, help="Path to the folder where the CVAT dataset will be saved")

    args = parser.parse_args()

    model = 'models/combination-with-synthetic-v2.pt'

    run_yolo_and_save_cvat(model, args.input, args.output)