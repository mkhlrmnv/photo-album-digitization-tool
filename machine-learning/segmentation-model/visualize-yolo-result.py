import cv2
from pathlib import Path
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library
import numpy as np
import logging

def visualize_yolo_results(model_path, input_folder, draw_rect=False, save_crops=False, output_folder=None):
    """
    Run a YOLO segmentation model on all images in a folder and visualize or save the results.

    Args:
        model_path (str): Path to the YOLO model.
        input_folder (str): Path to the folder containing input images.
        draw_rect (bool): If True, draw the smallest rectangle around the mask instead of the mask itself.
        save_crops (bool): If True, save the cropped results instead of displaying them.
        output_folder (str): Path to the folder where cropped results will be saved (required if save_crops is True).
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Define input folder
    input_folder = Path(input_folder)

    # Ensure output folder exists if saving crops
    if save_crops and output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # Process images
    image_files = list(input_folder.glob("*"))

    for img_file in image_files:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            print(f"Processing {img_file.name}...")
            results = model.predict(source=str(img_file), save=False, save_txt=False)

            # Load the image for visualization or cropping
            image = cv2.imread(str(img_file))
            img_height, img_width = image.shape[:2]

            crop_idx = 0  # Counter for multiple crops per image

            for result in results:
                if draw_rect or save_crops:
                    # Process each mask
                    for mask in getattr(result, "masks", []).xy if getattr(result, "masks", None) else []:
                        # Convert mask points to a NumPy array
                        points = np.array(mask, dtype=np.float32)

                        # Validate points
                        if points.size == 0 or len(points) < 3:
                            continue

                        try:
                            # Calculate the minimum area rectangle
                            rotated_rect = cv2.minAreaRect(points)
                            box = cv2.boxPoints(rotated_rect)  # Get the 4 corner points of the rectangle
                            box = np.int32(box)  # Convert to integer coordinates

                            # Crop the region based on the rectangle
                            (cx, cy), (w, h), angle = rotated_rect
                            if w <= 0 or h <= 0:
                                continue

                            # Ensure width >= height for consistent rotation handling
                            if w < h:
                                angle += 90
                                w, h = h, w

                            # Rotation matrix around the center
                            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                            rotated = cv2.warpAffine(image, M, (img_width, img_height), flags=cv2.INTER_CUBIC)

                            # Compute crop coordinates
                            x = int(round(cx - w / 2.0))
                            y = int(round(cy - h / 2.0))
                            w_i = int(round(w))
                            h_i = int(round(h))

                            # Clamp to image bounds
                            x = max(0, x)
                            y = max(0, y)
                            if x + w_i > img_width:
                                w_i = img_width - x
                            if y + h_i > img_height:
                                h_i = img_height - y
                            if w_i <= 0 or h_i <= 0:
                                continue

                            crop = rotated[y:y + h_i, x:x + w_i]

                            if save_crops:
                                # Save the cropped region
                                crop_filename = f"{img_file.stem}_crop_{crop_idx}.jpg"
                                crop_path = output_folder / crop_filename
                                cv2.imwrite(str(crop_path), crop)
                                crop_idx += 1
                            elif draw_rect:
                                # Draw the rotated rectangle on the image
                                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error processing mask for {img_file.name}: {e}")
                            continue

            if not save_crops: # and not draw_rect:
                # Display the image
                cv2.imshow("YOLO Segmentation Results", image)

                # Wait for a key press to move to the next image
                key = cv2.waitKey(0)
                if key == 27:  # Press 'Esc' to exit
                    print("Exiting visualization...")
                    cv2.destroyAllWindows()
                    return

    if not save_crops:
        cv2.destroyAllWindows()
    print("Processing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO model on images and visualize or save results")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--rect", action="store_true", help="Draw the smallest rectangle around the mask instead of the mask itself")
    parser.add_argument("--save-crops", action="store_true", help="Save the cropped results instead of displaying them")
    parser.add_argument("--output", type=str, help="Path to the folder where cropped results will be saved (required if --save-crops is used)")

    args = parser.parse_args()

    if args.save_crops and not args.output:
        parser.error("--output is required if --save-crops is used")

    model = 'models/combination-with-synthetic-v2.pt'

    visualize_yolo_results(model, args.input, draw_rect=args.rect, save_crops=args.save_crops, output_folder=args.output)