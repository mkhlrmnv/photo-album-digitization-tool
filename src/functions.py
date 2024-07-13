import cv2

def remove_white(path):
    """
    Removes white background from an image.

    Args:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The cropped image with the white background removed, or the original image if no contours are found.
    """

    # Opens picture from path and makes it gray scale
    image = cv2.imread(path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applies binary threshold and find picture contours from it
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Returns bigest contour that it found as image or if not any contours
    # were found returns original picture
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the image
        # for some reason w and h is +1 from what it's need to be
        cropped_image = image[y:y+h - 1, x:x+w - 1]
        return cropped_image
    else:
        return image
    
def crop_in_four_pieces(path):
    """
    Crops an image into four equal pieces.

    Args:
        path (str): The path to the image file.

    Returns:
        list: A list containing the four cropped image pieces.
    """
    # Opens the image
    image = cv2.imread(path)

    # Takes its dimensions and calculates what is half of them
    width, height, _ = image.shape
    half_width = width // 2  # two of // divides and floors the result
    half_height = height // 2

    result = []

    for i in range(2):
        for j in range(2):
            left = i * half_width
            top = j * half_height
            right = (i + 1) * half_width
            bottom = (j + 1) * half_height

            result.append(image[top:bottom, left:right])

    return result

def get_pictures(path):
    """
    Opens an image from the given path and converts it to grayscale.
    Applies binary thresholding and finds contours in the image.
    Filters the contours based on minimum width and height.
    Returns a list of cropped images corresponding to the filtered contours.

    Args:
        path (str): The path to the image file.

    Returns:
        list: A list of cropped images.

    """
    # Opens picture from path and makes it gray scale
    image = cv2.imread(path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applies binary threshold and find picture contours from it
    _, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    result = []

    min_x = image.shape[0] // 4
    min_y = image.shape[1] // 4

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_x and h > min_y:
            result.append(image[y:y+h - 1, x:x+w - 1])

    return result

# For debugging

image_path = 'input/Skannaus 22.jpeg'
cropped_images = get_pictures(image_path)
for i, img in enumerate(cropped_images):
    cv2.imshow(f"Cropped Image {i}", img)
cv2.imshow("original", cv2.imread(image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()