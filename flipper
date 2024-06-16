import os
from PIL import Image

inputDir = 'input'
outputDir = 'output'

# Ensure output directory exists
os.makedirs(outputDir, exist_ok=True)

def rotate_images(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct full file path
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            
            # Open the image
            with Image.open(input_path) as img:
                # Rotate the image 90 degrees clockwise
                rotated_img = img.rotate(-90, expand=True)
                # Save the rotated image to the output directory
                rotated_img.save(output_path)
                # print(f"Rotated and saved {filename}")

# Call the function
rotate_images(inputDir, outputDir)