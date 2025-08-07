from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import os
from dotenv import load_dotenv
load_dotenv() 

import glob

model = YOLO("/Users/mkhlrmnv/Desktop/kuva-prosessointi/custom_ml/finetuned-yolov8m-obb-200epoch.pt")

image_extension = "*.jpeg"
image_folder = "/Users/mkhlrmnv/Desktop/kuva albumit/2000/original"

imgs = []
imgs.extend(glob.glob(os.path.join(image_folder, image_extension)))

# print(os.listdir(image_folder))
# print(imgs)

for i, im in enumerate(imgs):
    # Load the image
    image = cv2.imread(im)
    
    # Convert to grayscale (black and white)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert back to 3-channel for YOLO (YOLO expects 3-channel input)
    gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    res = model.predict(gray_image_3ch, conf=0.4)
    cv2.imshow(f"img {i}", res[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()