import cv2
from pathlib import Path
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library
import numpy as np
import logging

def visualize_yolo_results(model_path, input_folder, draw_rect=False):
    """
    Run a YOLO segmentation model on all images in a folder and visualize the results.

    Args:
        model_path (str): Path to the YOLO model.
        input_folder (str): Path to the folder containing input images.
        draw_rect (bool): If True, draw the smallest rectangle around the mask instead of the mask itself.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Define input folder
    input_folder = Path(input_folder)

    # Process images
    image_files = list(input_folder.glob("*"))

    for img_file in image_files:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            print(f"Processing {img_file.name}...")
            results = model.predict(source=str(img_file), save=False, save_txt=False)

            # Load the image for visualization
            image = cv2.imread(str(img_file))
            img_height, img_width = image.shape[:2]

            for result in results:
                if draw_rect:
                    # Draw the smallest rotated rectangle around each mask
                    for mask in getattr(result, "masks", []).xy if getattr(result, "masks", None) else []:
                        # Convert mask points to a NumPy array
                        points = np.array(mask, dtype=np.float32)

                        # Validate points
                        if points.size == 0 or len(points) < 3:
                            logging.warning(f"Skipping mask with insufficient points (len={len(points)}) for image {img_file.name}")
                            continue

                        try:
                            # Calculate the minimum area rectangle
                            rotated_rect = cv2.minAreaRect(points)
                            box = cv2.boxPoints(rotated_rect)  # Get the 4 corner points of the rectangle

                            if box is None or box.size == 0:
                                logging.warning(f"boxPoints returned empty for image {img_file.name}")
                                continue

                            if np.isnan(box).any():
                                logging.warning(f"box contains NaN for image {img_file.name}")
                                continue

                            # Ensure the rectangle has positive area
                            if cv2.contourArea(box) <= 0:
                                logging.warning(f"Generated box has zero area for image {img_file.name}")
                                continue

                            box = np.int32(box)  # Convert to integer coordinates

                            # Draw the rotated rectangle on the image
                            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                        except Exception as e:
                            logging.exception(f"Failed to compute/draw rotated rect for image {img_file.name}: {e}")
                else:
                    # Draw the masks and bounding boxes
                    annotated_frame = result.plot()
                    image = cv2.addWeighted(image, 0.5, annotated_frame, 0.5, 0)

            # Display the image
            cv2.imshow("YOLO Segmentation Results", image)

            # Wait for a key press to move to the next image
            key = cv2.waitKey(0)
            if key == 27:  # Press 'Esc' to exit
                print("Exiting visualization...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Visualization complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO model on images and visualize results")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--rect", action="store_true", help="Draw the smallest rectangle around the mask instead of the mask itself")

    args = parser.parse_args()

    model = 'models/combination-with-synthetic-v2.pt'

    visualize_yolo_results(model, args.input, draw_rect=args.rect)