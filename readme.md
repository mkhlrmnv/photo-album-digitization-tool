# Image Processing Project Overview

This project leverages the power of OpenCV, a leading library in computer vision, to perform various image processing tasks. It's designed to manipulate images through operations like background removal, image cropping, and feature extraction. The core of the project is divided into two segments: the functional implementations within `functions.py` and their validation through unit tests in `test_functions.py`.

## Key Features

### 1. **Remove White Background** (`remove_white`)
- **Description**: Isolates the main subject by removing the white background from an image. Ideal for images where the subject is set against a white backdrop.
- **Input**: Path to the image file.
- **Output**: The image with the white background removed, or the original image if no contours are detected.

### 2. **Crop Image into Four Equal Pieces** (`crop_in_four_pieces`)
- **Description**: Divides an image into four equal sections. Useful for detailed analysis of specific image areas or for processing smaller data chunks.
- **Input**: Path to the image file.
- **Output**: Four cropped sections of the original image.

### 3. **Get Pictures** (`get_pictures`)
- **Description**: Processes an image by converting it to grayscale, applying binary thresholding, and identifying contours. It then filters these contours based on set criteria and returns the corresponding cropped images.
- **Input**: Path to the image file.
- **Output**: Cropped images based on the filtered contours.

## Testing

A comprehensive suite of unit tests in `test_functions.py` ensures the reliability and accuracy of the implemented functions. These tests include scenarios such as:
- Removing white backgrounds, with and without detected contours.
- Cropping an image into four equal parts and verifying each segment's integrity.
- Extracting images through contour filtering and validating the expected number of cropped images.

## Getting Started

### Prerequisites
Ensure Python is installed on your system, along with the OpenCV library. Install OpenCV using pip:

```bash
pip install opencv-python-headless