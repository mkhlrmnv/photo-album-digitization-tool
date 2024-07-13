import numpy as np
import cv2
import unittest
import os
from functions import *

class TestRemoveWhite(unittest.TestCase):
    def test_remove_white_with_contour(self):
        # Create a test image with a white background and a black square contour
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 0), -1)

        # Save the test image to a temporary file
        temp_file = 'test_image.png'
        cv2.imwrite(temp_file, image)

        # Call the remove_white function
        result = remove_white(temp_file)

        # Assert that the result is the cropped image without the white background
        expected_result = image[20:80, 20:80]
        self.assertTrue(np.array_equal(result, expected_result))

        # Clean up the temporary file
        os.remove(temp_file)

    def test_remove_white_without_contour(self):
        # Create a test image with a white background
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)

        # Save the test image to a temporary file
        temp_file = 'test_image.png'
        cv2.imwrite(temp_file, image)

        # Call the remove_white function
        result = remove_white(temp_file)

        # Assert that the result is the original image
        self.assertTrue(np.array_equal(result, image))

        # Clean up the temporary file
        os.remove(temp_file)

if __name__ == '__main__':
    unittest.main()

