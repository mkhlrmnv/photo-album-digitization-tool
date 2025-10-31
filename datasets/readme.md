## Datasets

This folder contains all the datasets used for training and testing the machine learning models. The datasets are organized into subdirectories for raw, processed, and synthetic datasets. These datasets are used for both segmentation and classification tasks.

---

## **Folder Structure**
```
├── you-dataset1/       # downloaded or your own datasets
├── you-dataset2/
├── ...
└── README.md           # You are here
```

---

## Datasets used in project

You can access all the dataset used in this project [here](https://drive.google.com/file/d/1u4P-gXudEpjpQcIh559YQul4N5iyYNkw/view?usp=sharing)

### Sctructure of zipped folder
```
├── only-images/                            # folder contraining unannotated images
│ ├── aidin-lapsuus-dataset-only-images/    # real photo album scans
│ ├── second-dataset/                       # real photo album scans
│ ├── processed-by-segmentation/            # folder of crops outputted by segmentation model
│ └── coco/                                 # some folder with real images, for syhtetic dataset generation (for example coco-val2017)
│
├── rotation-model/
│ ├── synthetic-dataset/                    # synthetically generated dataset
│ │ ├── 0                                   # folder with all images with label '0'
│ │ └── ...
│ │
│ ├── real-dataset/                         # real labeled images
│ │ ├── 0                                   # folder with all images with label '0'
│ │ └── ...
│ │
│ ├── real-dataset/                         # same as above but split into train, val, test parts
│ │ ├── train                               # folder for training part
│ │ │ └── 0                                 # folder with all images with label '0'
│ │ └── ...
│ │
│ ├── training-dataset/                     # dataset on which the model was trained
│ │ ├── 0                                   # # folder with all images with label '0'
│ │ ├── ...
│ │
│ ├── training-dataset-split/               # same as above but split into train, val, test parts
│ │ ├── train                               # folder for training part
│ │ │ └── 0                                 # folder with all images with label '0'
│ │ └── ...
│
├── segmentation-model/
│ ├── 2000-matka-dataset/                   # first real photo scan dataset in CVAT format
│ │ └── images/                             # all the images
│ │ └── labels/                             # segments for each image
│ │ └── data.yaml                           # yaml file contraing info about labels
│ │ └── train.txt                           # CVAT file with paths to all images, in training set (basically all of images in this dataste)
│ │
│ ├── aidin-lapsuus-dataset/                # second real photo scan dataset in YOLO format
│ │ └── train/                              # all the images
│ │ │ ├── images/                           # all the images
│ │ │ └── images/                           # all the segments
│ │ ├── ...
│ │ └── data.yaml                           # yaml file contraing info about labels
│ │
│ ├── example-cvat-dataset/                 # Example of CVAT format
│ │
│ ├── synthetic-dataset/                    # first part of synthetic dataset
│ │
│ ├── synthetic-dataset-with-fade/          # second part of synthetic dataset (here rando fade was applied on each image)
│ │
│ └── training-dataset/                     # same as above but split into train, val, test parts
│
└── README.md                               # Same as this
```