import cv2
import pdf2image
import os
import numpy

kuva = 'test_pictures/test_get_pictures_from_pdf/test_2.pdf'


images = pdf2image.convert_from_path(kuva, dpi=300)

list = [cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR) for img in images]

denoised = cv2.fastNlMeansDenoisingColored(list[0])

ksize = (2, 2) 
blur = cv2.blur(denoised, ksize=ksize)

grey_blur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

thresh_with_blur = cv2.adaptiveThreshold(grey_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

grey = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)

thresh_without = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

resized = cv2.resize(thresh_without, (0, 0), fx=0.05, fy=0.05)

resized_blur = cv2.resize(thresh_with_blur, (0, 0), fx=0.05, fy=0.05)
print(resized.shape)

cv2.imshow("blurred", thresh_with_blur)
cv2.imshow("without", thresh_without)
cv2.imshow("resized without blur", resized)
cv2.imshow("resized with blur", resized_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
