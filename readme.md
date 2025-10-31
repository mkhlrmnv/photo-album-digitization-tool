# Photo Album Digitization Tool: Image Processing and Machine Learning Platform

## **Core Idea**
The **Kuva-Prosessointi** project is a comprehensive platform for processing scanned photo album pages. It combines a web-based interface for user interaction with machine learning models for image segmentation and classification. The platform allows users to upload large images, crop individual photos, and process them using advanced machine learning techniques. Additionally, it provides tools for generating synthetic datasets, splitting datasets, and training models for two distinct tasks: **segmentation** and **classification**.

### Example of the UI
<a href="examples/sample-photo.jpg">
    <img src="./examples/Screenshot 2025-10-31 at 15.24.33.png" alt="Sample scanned photo" style="max-width:500px; width:100%; height:auto; display:block;" />
</a>


### Usage example
<a href="web-interface/static/demo.gif">
    <img src="./examples/Screen%20Recording%202025-10-30%20at%2020.30.02.gif" alt="Demo: Photo Album Digitization (small)" style="max-width:500px; width:100%; height:auto; display:block;" />
</a>

---

## **Project Structure**
The project is organized into the following main components:

```
kuva-prosessointi/
├── web-interface/                      # Web application for user interaction
│ ├── app/                              # Application logic and components
│ │ ├── app.py                          # Main app components (UI logic)
│ │ ├── states/                         # Application state management
│ │ │  └── state.py                      # Handles uploads, processing, and results
│ │ │
│ ├── rxconfig.py                       # Reflex configuration
│ ├── static/                           # Static assets (CSS, JS, images)
│ └── templates/                        # HTML templates (if applicable)
│ 
├── machine-learning/                   # Machine learning scripts and models
│ ├── segmentation-model/               # Scripts and models for segmentation tasks
│ │ ├── synthetic-dataset-creator.py    # Generate synthetic datasets
│ │ ├── dataset-split.py                # Split datasets into train/val/test
│ │ ├── train.py                        # Train segmentation models
│ │ ├── cvat-txt-file-generator.py      # Generates required .txt files to import dataset into cvat
│ │ ├── dataset-format-converted.py     # Converts yolo dataset into cvat and via versa
│ │ └── README.md                       # TODO
│ │
│ ├── classification-model/             # Scripts and models for classification tasks
│ │ ├── synthetic-dataset-generator.py  # Generate synthetic datasets
│ │ ├── dataset-split.py                # Split datasets into train/val/test
│ │ ├── train.py                        # Train classification models
│ │ └── README.md                       # TODO
│ 
├── datasets/                           # Datasets for training and testing
│ └── README.md                         # TODO
└── README.md                           # Project overview and instructions
```


---

## **How to Start the Web Application**
The web application provides an interface for uploading and processing images. Follow these steps to start the app:

1. **Install Dependencies**:
   - Ensure you have Python 3.8+ installed.
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Web Application**:
   - Navigate to the `web-interface/` folder:
     ```bash
     cd web-interface
     ```
   - Start the Reflex app:
     ```bash
     reflex run
     ```
   - Open your browser and navigate to `http://localhost:3000` to access the app.

3. **Features**:
   - **Upload Images**: Upload large scanned images of photo album pages.
   - **Process Images**: Automatically crop individual photos and process them using machine learning models.
   - **Download Results**: Download processed images and results as a zip file.

---

## **Machine Learning Folder**
The [machine-learning](http://_vscodecontentref_/4) folder contains all the scripts and tools for working with datasets and training models. It is divided into two subfolders for the two main tasks:

### **1. Segmentation Model**
- **Purpose**: Detect and segment individual photos from scanned album pages.
- **Key Scripts**:
  - `synthetic-dataset-creator.py`: Generate synthetic datasets for segmentation tasks.
  - `dataset-split.py`: Split datasets into train/val/test sets for segmentation.
  - `train.py`: Train YOLO-based segmentation models.
  - `cvat-txt-file-generator.py`: Generates required .txt files to import dataset into cvat
  - `dataset-format-converted.py': Converts yolo dataset into cvat format and via versa
- **Details**: Refer to the `segmentation-model/README.md` for a detailed explanation of each script.

### **2. Classification Model**
- **Purpose**: Classify or correct the orientation of cropped photos (e.g., 0°, 90°, 180°, 270°).
- **Key Scripts**:
  - `synthetic-dataset-generator.py`: Generate synthetic datasets for classification tasks.
  - `dataset-split.py`: Split datasets into train/val/test sets for classification.
  - `train.py`: Train YOLO-based classification models.
- **Details**: Refer to the `classification-model/README.md` for a detailed explanation of each script.

---

## **Key Features**
1. **Web Interface**:
   - User-friendly interface for uploading and processing images.
   - Real-time progress updates and downloadable results.

2. **Machine Learning Tools**:
   - Scripts for generating synthetic datasets, splitting datasets, and training models.
   - Support for both segmentation and classification tasks.

---

For more detailed instructions on specific tasks, refer to the [README.md](http://_vscodecontentref_/5) files in the respective folders.