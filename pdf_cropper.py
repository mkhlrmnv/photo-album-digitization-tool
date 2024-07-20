import cv2
import pdf2image
import os
import numpy

def pdf2img(path):
    """
    Convert a PDF file to a list of OpenCV images.

    Args:
        path (str): The path to the PDF file.

    Returns:
        list: A list of OpenCV images converted from the PDF pages.
    """
    images = pdf2image.convert_from_path(path, dpi=300)
    list = []
    for img in images:
        list.append(cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR))
    return list

def get_pictures_from_pdf(path):
    """
    Extracts pictures from a PDF file.

    Args:
        path (str): The path to the PDF file.

    Returns:
        list: A list of images extracted from the PDF file.
    """
    # empty array for returning contours
    result = []
    for image in pdf2img(path):
        # calculating max and min values so too small or too big contours doesn't been
        # added to result list
        min_x = image.shape[0] // 4
        min_y = image.shape[1] // 4
        max_x = image.shape[0] - min_x
        max_y = image.shape[1] - min_y

        denoised = cv2.fastNlMeansDenoisingColored(image)
        grey = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)

        # Applies binary threshold and find picture contours from it
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # going through all contours, and if they match size criterias, contour gets added
        # into result list
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if (w > min_x and h > min_y) and w < max_x and h < max_y:
                result.append(image[y:y+h - 1, x:x+w - 1])

    # returning all contours that matched criterias
    return result

for f in os.listdir('test_pictures/test_get_pictures_from_pdf'):
    if f.endswith('.pdf'):
        path = os.path.join('test_pictures/test_get_pictures_from_pdf', f)
        res = get_pictures_from_pdf(path)
        for i, j in enumerate(res):
            cv2.imshow(f"{f} {i}", j)
cv2.waitKey(0)
cv2.destroyAllWindows()