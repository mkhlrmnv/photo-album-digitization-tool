import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import glob

def convert_pdf_to_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for f in os.listdir(input_dir):
        if f.lower().endswith('.pdf'):
            path = os.path.join(input_dir, f)
            pages = convert_from_path(path, dpi=300)
            for i, page in enumerate(pages):
                page_filename = f'{f.split(".")[0]}_{i}.jpg'
                page.save(os.path.join(output_dir, page_filename), 'JPEG')

def extract_pictures_from_page(page_image_path, output_dir):
    image = cv2.imread(page_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    picture_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:  # Filter out small contours
            picture = image[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/picture_{picture_count}.jpg", picture)
            picture_count += 1

import cv2

def draw_contours_on_picture(page_image_path):
    image = cv2.imread(page_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply additional preprocessing to better isolate large features
    # For example, a larger kernel size for GaussianBlur might help
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    edged = cv2.Canny(blurred, 100, 200, apertureSize=3, L2gradient=True)

    cv2.imshow('canny', edged)
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on contour area, largest to smallest
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Iterate over the sorted contours, drawing only the largest ones
    # Adjust the criteria as needed to better match your images
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_pdfs_from_folder(input_folder, img_output, final_output):
    os.makedirs(final_output, exist_ok=True)
    # pdf_files = glob.glob(os.path.join(input_folder, '*.pdf'))

    convert_pdf_to_images(input_folder, img_output)
    
    page_images = glob.glob(os.path.join(img_output, '*.jpg'))
    for page_image in page_images:
        extract_pictures_from_page(page_image, final_output)


input_folder = 'input'
output_img = 'output/pdf2image'
output_crop = 'output/finished'

test_img = 'output/pdf2image/kuva albumi_1.jpg'

draw_contours_on_picture(test_img)