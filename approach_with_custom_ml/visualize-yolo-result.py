import cv2
from pathlib import Path
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library

def visualize_yolo_results(model_path, input_folder):
    """
    Run a YOLO segmentation model on all images in a folder and visualize the results.

    Args:
        model_path (str): Path to the YOLO model.
        input_folder (str): Path to the folder containing input images.
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
            results = model.predict(source=str(img_file), save=False, save_txt=False, show=True)

            # Visualize results
            for result in results:
                annotated_frame = result.plot()  # Annotate the image with masks and bounding boxes
                cv2.imshow("YOLO Segmentation Results", annotated_frame)

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

    args = parser.parse_args()

    model = 'training-runs/combination-with-synthetic/segment/photo_segmentation/weights/best.pt'

    visualize_yolo_results(model, args.input)