# Brain Tumor Segmentation Project
![image](https://github.com/user-attachments/assets/fe1509e6-b736-40da-9e91-fd13746a6d55)

## Overview

This project implements a deep learning model for brain tumor segmentation using the BraTS2020 dataset. The model processes multi-modal MRI scans to identify and segment different types of brain tumors.

## Data

The project uses the BraTS2020_TrainingData dataset, which includes the following MRI modalities:
- T1-weighted (T1)
- T2-weighted (T2)
- T1 with contrast enhancement (T1ce)
- Fluid Attenuated Inversion Recovery (FLAIR)

## Requirements

All required dependencies are listed in the `requirements.txt` file. To install them, run:

```
pip install -r requirements.txt
```


## Data Preprocessing

The preprocessing pipeline includes the following steps:
1. Loading and scaling of multi-modal MRI images (T1, T2, T1ce, FLAIR)
2. Loading and processing of segmentation masks
3. Combining multiple modalities into a single 4-channel image
4. Cropping images and masks to a specified region of interest
5. One-hot encoding of segmentation masks
6. Conversion to PyTorch tensors
7. Saving processed data as PyTorch tensor files

## Dataset Class

The `BrainTumorDataset` class is implemented to facilitate data loading during training. It inherits from PyTorch's `Dataset` class and provides methods for accessing the preprocessed image and mask pairs.

## Model Training

The training process includes:
1. Data loading using `DataLoader`
2. Model optimization using a combination of Dice loss and Focal loss
3. Training and validation loops
4. Logging of training and validation losses

## Usage

1. Preprocess the BraTS2020 data using the provided preprocessing script.
2. Create instances of the `BrainTumorDataset` class for training and validation sets.
3. Initialize the model, optimizer, and loss functions.
4. Run the training loop, calling `train_epoch` and `validate` functions for each epoch.

## Note

This project is based on BraTS2020 dataset. Ensure you have the correct data structure and file paths before running the preprocessing and training scripts.
