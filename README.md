# Bregman semantic segmentation
## Overview
This project test new optimizer--Bregman method in neural network training. While $\ell_1$-regularization doesn't significantly reduce the number of paramters in the neural network, we applied another approach, named Linearized Bregman method.

It is an iteration method that starts from sparse initializaiton and recovers the important parameters first(Here important means large gradient norm) while defers other parameters. Hence it keeps the sparisity all along the training.

We tested on U-Net semantic segmentation, CT reconstrution and CT denosing. All the experiments demonstrate that Bregman method is able to significantly reduce the number of active parameters with negligible accuracy drop.

## Model

Semantic segmentation:

UNet is a convolutional network architecture designed for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path for precise localization.

CT Reconstruction:

The network, named FullRecoNet, is structured as a fully connected architecture with 4 layers. Input Layer accepts a flattened sinogram as input, with dimensions set to match the flattened product of the original sinogram’s width and height ($5\times 41$ in this case).

CT denosing:

The 3-layer denoising network operates directly on $28 \times 28$ image reconstructions obtained from FBP. It takes an FBP reconstruction as input and learns to suppress noise and streak artifacts.

## Regularizers

### Sparse Group Lasso Regularizer

The sparse Group Lasso regularizer is defined as:

$$
J(\theta) = \sum_{l=1}^L \lambda_1 \left(\lambda_0 \sqrt{P_l} \sum_{n=1}^{N_l} \left\|\theta_l^n\right\|_2 + (1-\lambda_0) \left\|\theta_l\right\|_1\right),
$$

where $ \lambda_0 \in [0, 1] $ sets the relative influence of both terms.



# Prerequisites

## Python Packages

### Core Packages

- **torch**: 1.10.0
- **torchvision**: 0.11.1
- **sys**: System-specific parameters and functions.
- **os**: Operating system interfaces.
- **math**: Mathematical functions.
- **odl**: 1.0.0dev.


### Additional Libraries

- **torchmetrics**: Metrics computation for PyTorch.
- **matplotlib.pyplot**: Plotting library for visualizing data.
- **pickle**: Python object serialization.

## Python Version

- **Python**: 3.9.12

## Framework Versions

- **torch**: 1.10.0





Feel free to modify the content based on your specific project requirements and information.
