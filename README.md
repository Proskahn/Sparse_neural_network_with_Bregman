# Bregman semantic segmentation
## Overview
This project focuses on semantic segmentation with new optimizer--Bregman method.

I have not finished all the numerical experiments. The results are on the way.

## Model

UNet is a convolutional network architecture designed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path for precise localization.

## Regularizers

### Sparse Group Lasso Regularizer

The sparse Group Lasso regularizer is defined as:

$$
J(\theta) = \sum_{l=1}^L \lambda_1 \left(\lambda_0 \sqrt{P_l} \sum_{n=1}^{N_l} \left\|\theta_l^n\right\|_2 + (1-\lambda_0) \left\|\theta_l\right\|_1\right),
$$
where $ \lambda_0 \in [0, 1]$ sets the relative influence of both terms.



# Prerequisites

## Python Packages

### Core Packages

- **torch**: 1.10.0
- **torchvision**: 0.11.1
- **sys**: System-specific parameters and functions.
- **os**: Operating system interfaces.
- **math**: Mathematical functions.


### Additional Libraries

- **torchmetrics**: Metrics computation for PyTorch.
- **matplotlib.pyplot**: Plotting library for visualizing data.
- **pickle**: Python object serialization.

## Python Version

- **Python**: 3.9.12

## Framework Versions

- **torch**: 1.10.0





Feel free to modify the content based on your specific project requirements and information.
