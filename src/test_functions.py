import numpy as np
import cv2
import unittest
import os
from functions import *

class TestFunctions(unittest.TestCase):
    def test_remove_white_with_contour(self):
        # Create a test image with a white background and a black square contour
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 0), -1)

        # Save the test image to a temporary file
        temp_file = 'test_pictures/test_remove_with_contours.png'

        # Call the remove_white function
        result = remove_white(temp_file)

        # Assert that the result is the cropped image without the white background
        expected_result = image[20:80, 20:80]
        self.assertTrue(np.array_equal(result, expected_result))


    def test_remove_white_without_contour(self):
        # Create a test image with a white background
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)

        # Save the test image to a temporary file
        temp_file = 'test_pictures/test_remove_white_without_contours.png'

        # Call the remove_white function
        result = remove_white(temp_file)

        # Assert that the result is the original image
        self.assertTrue(np.array_equal(result, image))


    def test_crop_in_four_pieces(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)
        cv2.rectangle(image, (50, 0), (100, 50), (0, 0, 0), -1)
        cv2.rectangle(image, (0, 50), (50, 100), (255, 0, 0), -1)
        cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), -1)

        temp_file = 'test_pictures/test_crop_in_four.png'

        # Call the crop_in_four_pieces function
        result = crop_in_four_pieces('test_pictures/test_crop_in_four.png')

        # Assert that the result is a list containing four cropped image pieces
        self.assertEqual(len(result), 4)
        for i in range(2):
            for j in range(2):
                self.assertTrue(isinstance(result[i * 2 + j], np.ndarray))
                self.assertEqual(result[i * 2 + j].shape, (50, 50, 3))
                self.assertTrue(np.array_equal(result[i * 2 + j], image[j * 50:(j + 1) * 50, i * 50:(i + 1) * 50]))

    def test_get_picture(self):
        for f in os.listdir('test_pictures/test_get_pictures'):
            if f.endswith(".jpg") or f.endswith(".jpeg"):
                path = os.path.join('test_pictures/test_get_pictures', f)
                self.assertEqual(2, len(get_pictures(path)))

    def test_get_pictures_from_pdf(self):
        res = get_pictures_from_pdf("test_pictures/test_get_pictures_from_pdf/test_1.pdf")

        self.assertEqual(5, len(get_pictures_from_pdf("test_pictures/test_get_pictures_from_pdf/test_1.pdf")))
        self.assertEqual(1, len(get_pictures_from_pdf("test_pictures/test_get_pictures_from_pdf/test_2.pdf")))


    def test_flip_counter_clockwise(self):
        # Call the flip_counter_clockwise function
        result = flip_counter_clockwise('test_pictures/test_flip/test_flip_original.png')
        
        # Assert that the result is the counter-clockwise flipped image
        expected_result = cv2.imread('test_pictures/test_flip/test_flip_counter_clockwise.png')
        self.assertTrue(np.array_equal(result, expected_result))

    def test_flip_clockwise(self):
        # Call the flip_clockwise function
        result = flip_clockwise('test_pictures/test_flip/test_flip_original.png')

        # Assert that the result is the counter-clockwise flipped image
        expected_result = cv2.imread('test_pictures/test_flip/test_flip_clockwise.png')

        self.assertTrue(np.array_equal(result, expected_result))

    def test_pdf2img(self):
        result = pdf2img('test_pictures/test_get_pictures_from_pdf/test_2.pdf')
        self.assertEqual(1, len(result))
        self.assertEqual("<class 'numpy.ndarray'>", f"{type(result[0])}")

if __name__ == '__main__':
    unittest.main()