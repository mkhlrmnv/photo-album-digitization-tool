from src.functions import ImageProcessing as im
import cv2
from PIL import Image
import os


path = 'test_pictures/test_get_pictures_from_pdf/test_1.pdf'
output = 'output/finished'

pictures = im.get_pictures_from_pdf(path)

print(f"lne: {len(pictures)}")

for i, p in enumerate(pictures):
    name = os.path.join(output, f'kuva_{i}')
    cv2.imshow(f"{i}", p)

cv2.waitKey(0)
cv2.destroyAllWindows()


