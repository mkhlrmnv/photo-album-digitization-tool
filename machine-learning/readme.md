# Classification Model Tools

This folder contains all the scripts and tools required for working with classification models. The classification models are designed to classify or correct the orientation of cropped photos (e.g., 0°, 90°, 180°, 270°). These tools include scripts for preparing datasets, splitting datasets, training YOLO-based classification models, and visualizing the results.

---

## **Folder Structure**
```
classification-model/
├── dataset-split.py                # Split datasets into train/val/test sets
├── synthetic-dataset-generator.py  # Generates synthetic dataset
├── train.py                        # Train YOLO-based classification models
├── visualize.py                    # Visualize YOLO classification results
├── README.md                       # You are here
```

---

## **Scripts Overview**

### 1. dataset-split.py
- **Purpose**: Split a dataset into train/val/test sets.

- ***Features**:
    - Allows custom train/val/test split ratios.
    - Can move or copy files during splitting.

- **Usage**
    ```bash
    python dataset-split.py --source dataset --output dataset-split
    python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
    ```

### 2. **`synthetic-dataset-generator.py`**
- **Purpose**: Generate synthetic datasets randomly rotating images 0, 90, 180 or 270 degreed counter clock wise and moving into corresponding folder. This allows to quickly generate large datasets on which model can be trained. For example latest model was trained on 1000 synthetic photos with combination of ~280 real annotated images.

- **Features**:
  - Randomly rotates and labels images

- **Usage**:
  ```bash
  python rotate-images.py --src /path/to/source --out /path/to/output
  ````

### 3. **`train.py`**
- **Purpose**: Train YOLO-based classification models on your dataset.

- **Features**:
    - Supports custom model weights, batch sizes, and image sizes.
    - Saves training results, including logs and checkpoints.

- **Usage**:
  ```bash
  python train.py --data /path/to/dataset
  python train.py --data /path/to/dataset --epochs 50 --batch-size 32 --img-size 224
  ````

### 4. **`visualize.py`**
- **Purpose**: Visualize YOLO classification model predictions.

- **Features**:
    - Displays the predicted class label in the top-left corner of each image.
    - Allows users to view images one by one without saving them.

- **Usage**:
  ```bash
  python visualize.py --model your-finetuned-model.pt --input /path/to/input
  ````

---

## **How to Use**

1. Prepare Dataset:
    - Use CVAT to generate base dataset from real labeled images.
    - Use `rotate-images.py` to create a synthetic dataset with rotated images for classification task to combine with real image dataset
    
2. Split Dataset:
    - Use `dataset-split.py` to split your dataset into train/val/test sets.

3. Train Model:
    - Use `train.py` to train YOLO-based classification models.

4. Visualize Results:
    - Use `visualize.py` to visualize the classification results.

---

## **Dependencies**
- Python 3.8+
- Required Python packages:
```bash 
pip install ultralytics opencv-python pillow tqdm numpy```