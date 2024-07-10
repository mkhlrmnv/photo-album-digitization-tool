import cv2
import numpy as np
import os

# Directory containing your scanned images
input_dir = "input"
output_dir = "output"

test_img = 'input/Skannaus 7.jpeg'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to crop the image based on the two largest contours
def crop_image(image_path, output_path_prefix):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Process the two largest contours
    for i, contour in enumerate(contours[:2]):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y+h, x:x+w]
        output_path = f"{output_path_prefix}_{i+1}.jpg"
        cv2.imwrite(output_path, cropped_image)

# Process all images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        input_path = os.path.join(input_dir, filename)
        output_path_prefix = os.path.join(output_dir, os.path.splitext(filename)[0])
        crop_image(input_path, output_path_prefix)
