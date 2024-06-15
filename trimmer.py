import os
from PIL import Image
import numpy as np

def trim_white_borders(image_path, output_folder):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Check for fully white rows and columns
    # non_white_rows = np.where(np.any(image_array < 250, axis=(1, 2)))[0]
    # non_white_cols = np.where(np.any(image_array < 250, axis=(0, 2)))[0]

    white_rows = np.where()
    
    # Determine the cropping box
    if non_white_rows.size > 0 and non_white_cols.size > 0:
        top, bottom = non_white_rows[0], non_white_rows[-1]
        left, right = non_white_cols[0], non_white_cols[-1]
        
        # Crop the image
        trimmed_image = image.crop((left, top, right + 1, bottom + 1))
        
        # Save the trimmed image
        base_name = os.path.basename(image_path)
        trimmed_image.save(os.path.join(output_folder, base_name))
    else:
        # If the image is completely white, save the original
        image.save(os.path.join(output_folder, os.path.basename(image_path)))

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpeg'):
            image_path = os.path.join(input_folder, filename)
            trim_white_borders(image_path, output_folder)

if __name__ == "__main__":
    input_folder = 'input'
    output_folder = 'output'
    
    process_folder(input_folder, output_folder)
