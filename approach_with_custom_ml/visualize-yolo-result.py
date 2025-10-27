import cv2
from pathlib import Path
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library

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
                    # Draw the smallest rectangle around each mask
                    for mask in result.masks.xy:
                        # Calculate the bounding box of the mask
                        x_coords = [x for x, y in mask]
                        y_coords = [y for x, y in mask]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))

                        # Draw the rectangle on the image
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)
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

    model = 'training-runs/combination-with-synthetic-v2/epoch80.pt'

    visualize_yolo_results(model, args.input, draw_rect=args.rect)