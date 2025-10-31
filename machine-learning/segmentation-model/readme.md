# Segmentation Model Tools

This folder contains all the scripts and tools required for working with segmentation models. The segmentation models are designed to detect and segment individual photos from scanned photo album pages. These tools include scripts for formatting CVAT datasets into yolo friendly ones, generating synthetic datasets, splitting datasets, training YOLO-based segmentation models, and visualizing the result.

---

## **Folder Structure**

```
segmentation-model/
├── synthetic-dataset-creator.py    # Generate synthetic datasets for segmentation tasks
├── dataset-split.py                # Split datasets into train/val/test sets
├── train.py                        # Train YOLO-based segmentation models
├── visualize-yolo-result.py        # Visualize YOLO segmentation results
├── run-yolo-to-cvat.py             # Convert YOLO results to CVAT format
├── dataset-format-converter.py     # Convert datasets between YOLO and CVAT formats
├── cvat-txt-file-generator.py      # Generate CVAT-style .txt files
├── remove-substring.py             # Utility to clean filenames
└── README.md                       # You are here
```

---

## **Scripts Overview**

### 1. **`synthetic-dataset-creator.py`**
- **Purpose**: Generate synthetic datasets by pasting rectangular photos onto a white canvas. This allows to quickly generate large datasets on which model can be trained. For example latest model was trained on 1400 synthetic photos with combination of ~140 real annotated images.

- **Features**:
  - Randomly rotates, scales, and positions photos.
  - Optionally applies vintage effects like grain, vignette, and fade.
  - Outputs datasets in YOLO or CVAT format.

- **Usage**:
  ```bash
  # standard layout (images/, labels/, data.yaml)
  python synthetic-dataset-creator.py --src /path/to/photos --out ./data --n 500

  # CVAT-ready layout (images/train, labels/train, train.txt, data.yaml)
  python synthetic-dataset-creator.py --src /path/to/photos --out ./data --n 500 --cvat-ready
  ````

### 2. dataset-split.py
- **Purpose**: Split a dataset into train/val/test sets.

- ***Features**:
    - Supports both YOLO and CVAT dataset formats.
    - Allows custom train/val/test split ratios.
    - Can move or copy files during splitting.

- **Usage**
    ```bash
    python dataset-split.py --source dataset --output dataset-split
    python dataset-split.py --source dataset --output dataset-split --format cvat
    python dataset-split.py --source dataset --output dataset-split --ratios 0.8 0.1 0.1
    ```

### 3. train.py
- **Purpose**: Train YOLO-based segmentation models on your dataset.

- **Features**:
    - Supports custom model weights, batch sizes, and image sizes.
    - Saves training results, including logs and checkpoints.

- **Usage**:
    ```bash
    python train.py
    ```

### 4. visualize-yolo-result.py
- **Purpose**: Visualize YOLO segmentation results on images.

- **Features**:
    - Draws bounding boxes or masks on images.
    - Optionally saves cropped regions.

- **Usage**:
    ```bash
    python visualize-yolo-result.py --input input/ --rect 
    python visualize-yolo-result.py --input input/ --rect --save-crops --output output/  
    ```

### 5. run-yolo-to-cvat.py
- **Purpose**: Convert YOLO segmentation results into CVAT format for annotation or further processing. This makes process of getting results from current model, correcting them, and 
finetunning model even more faster. As example, this project trained first model on 700 synthetic + 70 real images, then segmented another 70 real images with resulting model. This segments were then corrected by hand, and final model was trained on 140 real images. 

- **Usage**
    ```bash
    python run-yolo-to-cvat.py --input /path/to/original-images --output /path/to/output-folder
    ```

### 6. dataset-format-converter.py
- **Purpose**: Convert datasets between YOLO and CVAT formats. Cruisal because, manually converting them is really time consuming

- **Note**: Works only for segmentation datasets

- **Usage**:
    ```bash
    python dataset-format-converter.py --mode yolo2cvat --source /path/to/yolo-dataset --output /path/to/cvat-dataset
    python dataset-format-converter.py --mode cvat2yolo --source /path/to/yolo-dataset --output /path/to/cvat-dataset
    ```

### 7. cvat-txt-file-generator.py
- **Purpose**: Generate CVAT-style .txt files for datasets.

- **Usage**:
    ```bash
    python cvat-txt-file-generator.py --input /path/to/images --output /path/to/labels```

---

## **How to Use**

1. Create dataset from real images, with CVAT
    - CVAT website has tutorial how to do this
    - Use `dataset-format-converter.py` to convert datasets to YOLO format for training.

2. Generate Synthetic Datasets:
    - Use `synthetic-dataset-creator.py` to create synthetic datasets for training.

3. Split Datasets:
    - Use `dataset-split.py` to split your dataset into train/val/test sets.

4. Train Models:
    - Use `train.py` to train YOLO-based segmentation models.

5. Visualize Results:
    - Use `visualize-yolo-result.py` to visualize the segmentation results.

6. Finetune model
    - Use `run-yolo-to-cvat.py` to create larger dataset from real images with already trained model.
    - Correct results CVAT
    - Train model even furter

---

## **Dependencies**

Python 3.8+
Required Python packages:
```bash
pip install ultralytics opencv-python pillow tqdm numpy```

