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


# For debugging
image_path = 'input/page_2_picture_2.jpeg'
cropped_image = remove_white(image_path)
cv2.imshow("Cropped Image", cropped_image)
cv2.imshow("original", cv2.imread(image_path))
cv2.waitKey(0)
cv2.destroyAllWindows()