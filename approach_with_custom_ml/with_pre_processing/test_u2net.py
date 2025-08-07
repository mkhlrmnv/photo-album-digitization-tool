import os, glob
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# 1. Install & import U2NET
#    You need the official repo: https://github.com/xuebinqin/U-2-Net
#    Clone it once, then point to its u2net.py and model files (u2net.pth)
from u2net import U2NET  # adjust your PYTHONPATH or copy u2net.py into your folder

# 2. Load the pretrained model
model = U2NET(3,1)
model.load_state_dict(torch.load("/Users/mkhlrmnv/Desktop/kuva-prosessointi/approach_with_custom_ml/with_pre_processing/u2net_portrait.pth", map_location="cpu"))
model.eval()

# 3. Define a helper to get a binary mask from U2NET
to_tensor = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

def get_all_saliency_masks(img_bgr):
    # Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×320×320

    # Inference
    with torch.no_grad():
        # model(...) returns a tuple: (d1, d2, d3, d4, d5, d6, d7)
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)

    # Convert each to a mask, resize back to original image size, threshold
    masks = []
    for d in (d1, d2, d3, d4, d5, d6, d7):
        sal = d[0,0].cpu().numpy()
        sal = cv2.resize(sal, (img_bgr.shape[1], img_bgr.shape[0]))
        _, mask = cv2.threshold((sal * 255).astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
        masks.append(mask)

    return masks  # list of seven binary masks

# 4. Your original loop, now using U2NET masks
image_extension = "*.jpeg"
image_folder = "/Users/mkhlrmnv/Desktop/kuva-prosessointi/approach_with_custom_ml/with_pre_processing/data_set/orig"
output_folder = "/Users/mkhlrmnv/Desktop/kuva-prosessointi/approach_with_custom_ml/with_pre_processing/data_set/u2net_processed"

imgs = glob.glob(os.path.join(image_folder, image_extension))
for image_path in imgs:
    img = cv2.imread(image_path)
    masks = get_all_saliency_masks(img)

    for i, mask in enumerate(masks):

        cv2.imshow(f"mask {i}", mask)
    cv2.imshow("img", img)

    

    # Optional: clean it up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find photo contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10000:
            continue  # skip small noise

        # get rotated rect & crop it
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        width, height = map(int, rect[1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],[0,0],[width-1,0],[width-1,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        crop = cv2.warpPerspective(img, M, (width, height))

        cv2.imshow(f"crop {i}", crop)

        # Save each cropped photo
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_folder, f"{base}_crop{i}.jpg")
        # cv2.imwrite(out_path, crop)

    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()

    print(f"Processed {os.path.basename(image_path)}")