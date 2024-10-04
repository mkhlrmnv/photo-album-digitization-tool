import numpy as np
import cv2
import unittest
import os
from functions import ImageProcessing as im

class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.test_pictures_dir = 'test_pictures'

    def test_remove_white_with_contour(self):
        # Create a test image with a white background and a black square contour
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 0), -1)

        # Call the remove_white function
        result = im.remove_white(os.path.join(self.test_pictures_dir, 'test_remove_with_contours.png'))

        # Assert that the result is the cropped image without the white background
        expected_result = image[20:80, 20:80]
        self.assertTrue(np.array_equal(result, expected_result))


    def test_remove_white_without_contour(self):
        # Create a test image with a white background
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)

        # Call the remove_white function
        result = im.remove_white(os.path.join(self.test_pictures_dir, 'test_remove_white_without_contours.png'))

        # Assert that the result is the original image
        self.assertTrue(np.array_equal(result, image))


    def test_crop_in_four_pieces(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image.fill(255)
        cv2.rectangle(image, (50, 0), (100, 50), (0, 0, 0), -1)
        cv2.rectangle(image, (0, 50), (50, 100), (255, 0, 0), -1)
        cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), -1)

        # Call the crop_in_four_pieces function
        result = im.crop_in_four_pieces(os.path.join(self.test_pictures_dir, 'test_crop_in_four.png'))
        
        # Assert that the result is a list containing four cropped image pieces
        self.assertEqual(len(result), 4)
        for i in range(2):
            for j in range(2):
                self.assertTrue(isinstance(result[i * 2 + j], np.ndarray))
                self.assertEqual(result[i * 2 + j].shape, (50, 50, 3))
                self.assertTrue(np.array_equal(result[i * 2 + j], image[j * 50:(j + 1) * 50, i * 50:(i + 1) * 50]))

    def test_get_picture(self):
        test_dir = os.path.join(self.test_pictures_dir, 'test_get_pictures')
        for f in os.listdir(test_dir):
            if f.endswith(".jpg") or f.endswith(".jpeg"):
                path = os.path.join(test_dir, f)
                self.assertEqual(2, len(im.get_pictures(path)))

    def test_get_pictures_from_pdf(self):
        pdf_dir = os.path.join(self.test_pictures_dir, 'test_get_pictures_from_pdf')
        self.assertEqual(5, len(im.get_pictures_from_pdf(os.path.join(pdf_dir, "test_1.pdf"))))
        self.assertEqual(1, len(im.get_pictures_from_pdf(os.path.join(pdf_dir, "test_2.pdf"))))

    def test_flip_counter_clockwise(self):
        flip_dir = os.path.join(self.test_pictures_dir, 'test_flip')
        # Call the flip_counter_clockwise function
        result = im.flip_counter_clockwise(os.path.join(flip_dir, 'test_flip_original.png'))
        
        # Assert that the result is the counter-clockwise flipped image
        expected_result = cv2.imread(os.path.join(flip_dir, 'test_flip_counter_clockwise.png'))
        self.assertTrue(np.array_equal(result, expected_result))

    def test_flip_clockwise(self):
        flip_dir = os.path.join(self.test_pictures_dir, 'test_flip')
        # Call the flip_clockwise function
        result = im.flip_clockwise(os.path.join(flip_dir, 'test_flip_original.png'))

        # Assert that the result is the clockwise flipped image
        expected_result = cv2.imread(os.path.join(flip_dir, 'test_flip_clockwise.png'))

        self.assertTrue(np.array_equal(result, expected_result))

    def test_pdf2img(self):
        pdf_dir = os.path.join(self.test_pictures_dir, 'test_get_pictures_from_pdf')
        result = im.pdf2img(os.path.join(pdf_dir, 'test_2.pdf'))
        self.assertEqual(1, len(result))
        self.assertEqual("<class 'numpy.ndarray'>", f"{type(result[0])}")


if __name__ == '__main__':
    unittest.main()