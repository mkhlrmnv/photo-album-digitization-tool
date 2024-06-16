import os
from PIL import Image

from trimmer import getCols
import numpy as np

# VARIABLES
JUMP = 20

def crop_image(image_path, output_folder, images_per_page, start_page, start_picture):
    image = Image.open(image_path)
    width, height = image.size
    
    # Calculate dimensions of each quadrant
    new_width = width // 2
    new_height = height // 2
    
    page = start_page
    picture_number = start_picture

    for i in range(2):
        for j in range(2):
            left = i * new_width
            top = j * new_height
            right = (i + 1) * new_width
            bottom = (j + 1) * new_height

            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))

            # Save the cropped image with the new naming convention
            cropped_image.save(os.path.join(output_folder, f"page_{page}_picture_{picture_number}.jpeg"))

            picture_number += 1
            if picture_number > images_per_page:
                picture_number = 1
                page += 1

    return page, picture_number

def split(array):
    whiteRows = getCols(array)
    jump = 0

    for i in range(len(whiteRows)):
        if i + 1 < len(whiteRows):
            if whiteRows[i + 1] > (whiteRows[i] + JUMP): 
                jump = whiteRows[i + 1]
                break

    img = Image.fromarray(array)

    if jump > 0 and jump < img.height:
        top = img.crop((0, 0, img.width, jump))
        bottom = img.crop((0, jump, img.width, img.height))
    else:
        print(f"Invalid jump value: {jump}. Image height is {img.height}.")
        top = img
        bottom = None

    if top and bottom:
        return top, bottom
    else:
        print("Cropping was not successful. Check the jump value.")
        return None, None

def process_folder(input_folder, output_folder, images_per_page):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    page = 1
    picture_number = 1
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpeg'):
            image_path = os.path.join(input_folder, filename)
            page, picture_number = crop_image(image_path, output_folder, images_per_page, page, picture_number)

def process_folder2():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 0
    for f in os.listdir(input_folder):
        if f.lower().endswith('.jpeg'):
            counter += 1
            path = os.path.join(input_folder, f)
            imgArr = np.array(Image.open(path))
            top, bottom = split(imgArr)
            if top and bottom:
                top.save(os.path.join(output_folder, f))
                bottom.save(os.path.join(output_folder, f"{f}_2.jpeg"))
                print(f"Done: {counter} / {len(os.listdir(input_folder))}")
            else:
                print(f"Picture {f} failed")



if __name__ == "__main__":
    input_folder = 'input'
    output_folder = 'output'
    images_per_page = 2  # Specify how many images can be on one page
    
    # process_folder(input_folder, output_folder, images_per_page)
    process_folder2()
