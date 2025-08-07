import os, glob
import cv2
import numpy as np
from PIL import Image

# Install python-doctr: pip install python-doctr[torch]
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# 1. Load the pre-trained OCR-predictor (which also does layout detection)
model = ocr_predictor(pretrained=True)  # uses DB-ResNet50 under the hood
model.eval()

# 2. Helper to run doctr and return all "figure" boxes
def get_figure_boxes(img_bgr):
    # Convert to PIL and wrap in a DocumentFile
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # print(Image.fromarray(img_rgb))
    doc = DocumentFile.from_images([img_bgr])
    # Run the model
    result = model(doc)  
    page = result.pages[0]
    # Collect all non-text blocks
    figures = [b for b in page.blocks]
    # Convert to pixel coordinates
    boxes = [b.geometry for b in figures]
    return boxes

# 3. Your processing loop
image_extension = "*.jpeg"
image_folder  = "/Users/mkhlrmnv/Desktop/kuva-prosessointi/approach_with_custom_ml/with_pre_processing/data_set/orig"
output_folder = "/Users/mkhlrmnv/Desktop/kuva-prosessointi/approach_with_custom_ml/with_pre_processing/data_set/doctr_processed"
os.makedirs(output_folder, exist_ok=True)

imgs = glob.glob(os.path.join(image_folder, image_extension))
for image_path in imgs:
    print(image_path)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    boxes = get_figure_boxes(image_path)

    print(boxes)

    for i, box in enumerate(boxes):
        # Crop axis-aligned first
        # breakpoint()
        (x1, y1), (x2, y2) = box
        crop = img[y1:y2, x1:x2]
        # Optional: deskew rotated regions using minAreaRect on the mask of that region
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bin_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(int)
            w2, h2 = map(int, rect[1])
            src = box.astype("float32")
            dst = np.array([[0,h2-1],[0,0],[w2-1,0],[w2-1,h2-1]], dtype="float32")
            M   = cv2.getPerspectiveTransform(src, dst)
            crop = cv2.warpPerspective(crop, M, (w2, h2))

        # Show & save
        cv2.imshow(f"crop {i}", crop)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_folder, f"{base}_crop{i}.jpg")
        # cv2.imwrite(out_path, crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Processed {os.path.basename(image_path)}")
    break  # remove if you want to process all