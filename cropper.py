import os
from PIL import Image

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

def process_folder(input_folder, output_folder, images_per_page):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    page = 1
    picture_number = 1
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpeg'):
            image_path = os.path.join(input_folder, filename)
            page, picture_number = crop_image(image_path, output_folder, images_per_page, page, picture_number)

if __name__ == "__main__":
    input_folder = 'input'
    output_folder = 'output'
    images_per_page = 2  # Specify how many images can be on one page
    
    process_folder(input_folder, output_folder, images_per_page)
