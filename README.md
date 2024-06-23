# Image Classification Project README

## Overview
This project focuses on semantic segmentation with new optimizer--Bregman method.

I have not finished all the numerical experiments. The results are on the way.

## Model

UNet is a convolutional network architecture designed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path for precise localization.

## Regularizers

### Sparse Group Lasso Regularizer

The sparse Group Lasso regularizer is defined as:

$$ J(\theta) = \sum_{l=1}^L \left((1-\alpha) \lambda_l \sqrt{P_l} \sum_{n=1}^{N_l} \left\|\theta_l^n\right\|_2 + \alpha \lambda_l \left\|\theta_l\right\|_1\right) $$

where \( \alpha \in [0, 1] \) sets the relative influence of both terms.


# Prerequisites

## Python Packages

### Core Packages

- **torch**: 1.10.0
- **torchvision**: 0.11.1
- **sys**: System-specific parameters and functions.
- **os**: Operating system interfaces.
- **math**: Mathematical functions.

### Custom Packages

- **models.aux_funs**: Custom auxiliary functions for models.
- **models.unet**: Implementation of UNet architecture for image segmentation.
- **models.segnet**: Implementation of SegNet architecture for image segmentation.
- **optimizers**: Custom optimization algorithms.
- **regularizers**: Implementations of regularization techniques.
- **train**: Modules for training neural networks.
- **utils.configuration**: Utilities for configuration settings.
- **utils.datasets**: Utilities for handling datasets.

### Additional Libraries

- **torchmetrics**: Metrics computation for PyTorch.
- **matplotlib.pyplot**: Plotting library for visualizing data.
- **pickle**: Python object serialization.

## Python Version

- **Python**: 3.9.12

## Framework Versions

- **TensorFlow**: 2.15.0
- **torch**: 1.10.0

## Additional Notes

Make sure to install these packages using `pip` or another package manager before running the project.


## Usage



Feel free to modify the content based on your specific project requirements and information.
