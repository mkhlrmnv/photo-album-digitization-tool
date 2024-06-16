from PIL import Image
import numpy as np

from trimmer import getCols

# VARIABLES
JUMP = 20

def find_split_row(image, threshold=250):
    """Find the row to split the image on, based on significant changes in whiteness."""
    grayscale = image.convert("L")  # Convert image to grayscale
    img_array = np.array(grayscale)  # Convert grayscale image to numpy array

    # Calculate the absolute difference between the whiteness of adjacent rows
    row_diffs = np.abs(np.diff(np.mean(img_array, axis=1)))
    # Find the row with the maximum difference in whiteness
    split_row = np.argmax(row_diffs)

    return split_row

def split_image_horizontally(image_path, output_path1, output_path2):
    """Split the image at the row with the most significant change in whiteness."""
    image = Image.open(image_path)
    split_row = find_split_row(image)

    # Crop the image into top and bottom parts
    top_image = image.crop((0, 0, image.width, split_row))
    bottom_image = image.crop((0, split_row, image.width, image.height))

    # Save the two new images
    top_image.save(output_path1)
    bottom_image.save(output_path2)

def split(array):
    print(len(array))
    print(len(array[0]))
    whiteRows = getCols(array)
    jump = 0

    print(len(whiteRows))

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
        top.show()
        bottom.show()
    else:
        print("Cropping was not successful. Check the jump value.")

# Example usage
image_path = "input/Skannaus 3.jpeg"
output_path1 = "output/top_image.jpeg"
output_path2 = "output/bottom_image.jpg"
split(np.array(Image.open(image_path)))