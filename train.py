import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.optim as optim
import torchmetrics
import torchvision
import matplotlib.pyplot as plt

from regularizers import reg_l1_l2
import models.aux_funs as maf
import optimizers as op
import regularizers as reg
import train
import math
import utils.configuration as cf
import utils.datasets as ud
import torch.nn.utils.prune as prune
import os

# Convert a PyTorch tensor into a PIL image
t2img = T.ToPILImage()

# Convert a PIL image into a PyTorch tensor
img2t = T.ToTensor()

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()

# Classification metrics imports
from torchmetrics.classification import Dice, JaccardIndex, MulticlassAccuracy

# Train step function
def train_step(conf, model, opt, train_loader, l1_lambda=0.01):
    tot_loss = 0.0
    iou_metric = JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
    dice_metric = Dice(num_classes=3, average='micro', ignore_index=1)
    criterion = nn.CrossEntropyLoss()

    for batch_id, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        logits = model(x)
        y = y.squeeze(1).long()
        loss = criterion(logits, y)
        
        # if isinstance(opt, optim.SGD):
        #     l1_l2_regularization = 0.0
        #     for param in model.parameters():
        #         if param.requires_grad:
        #             l1_norm = torch.norm(param, p=1)
        #             l2_norm = torch.norm(param, p=2)
        #             regularization_term = conf.lamda_1 * ((1-conf.lamda_0) * l1_norm + conf.lamda_0 * torch.sqrt(torch.tensor(param.shape[-1], dtype=torch.float)) * l2_norm)
        #             l1_l2_regularization += regularization_term
            
        #     loss += l1_l2_regularization
        
        loss.backward()
        opt.step()
        
        tot_loss += loss.item()

        with torch.no_grad():
            probs = nn.Softmax(dim=1)(logits)
            pred_labels = probs.argmax(dim=1)
            iou_metric.update(pred_labels, y)
            dice_metric.update(pred_labels, y)
    
    avg_loss = tot_loss / (batch_id + 1)
    iou_accuracy = iou_metric.compute().item()
    dice_accuracy = dice_metric.compute().item()
    iou_metric.reset()
    dice_metric.reset()
    
    print(f"Train Loss: {avg_loss:.4f}, IOU Accuracy: {iou_accuracy:.4f}, Dice Accuracy: {dice_accuracy:.4f}")
    return {'loss': avg_loss, 'IOU Accuracy': iou_accuracy, 'Dice Accuracy': dice_accuracy}

# Validation step function
def validation_step(model, val_loader):
    tot_loss = 0.0
    pixel_metric = MulticlassAccuracy(num_classes=3, average='micro', ignore_index=1)
    iou_metric = JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
    dice_metric = Dice(num_classes=3, average='micro', ignore_index=1)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(val_loader):
            logits = model(x)
            y = y.squeeze(1).long()
            loss = criterion(logits, y)

            tot_loss += loss.item()

            probs = nn.Softmax(dim=1)(logits)
            pred_labels = probs.argmax(dim=1)
            iou_metric.update(pred_labels, y)
            dice_metric.update(pred_labels, y)
            pixel_metric.update(pred_labels, y)
    
    avg_loss = tot_loss / (batch_id + 1)
    iou_accuracy = iou_metric.compute().item()
    dice_accuracy = dice_metric.compute().item()
    pixel_accuracy = pixel_metric.compute().item()
    iou_metric.reset()
    dice_metric.reset()
    pixel_metric.reset()
    
    net_sparse = maf.net_sparsity(model)
    node_sparse = maf.node_sparsity(model)
    node_sparsity_str = ', '.join([f'{sparsity:.4f}' for sparsity in node_sparse])

    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}, IOU Accuracy: {iou_accuracy:.4f}, Dice Accuracy: {dice_accuracy:.4f}, Net: {net_sparse:.4f}, Node: {node_sparsity_str}")
    return {'loss': avg_loss, 'IOU Accuracy': iou_accuracy, 'Dice Accuracy': dice_accuracy, 'Pixel Accuracy': pixel_accuracy, 'Net Sparsity': net_sparse, 'Node': node_sparsity_str}

# Pruning step function
def prune_step(model, a1=0.01, a2=0.01, conv_group=True):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and conv_group:
            prune.ln_structured(module, name='weight', amount=a1, n=2, dim=0)
            prune.ln_structured(module, name='weight', amount=a1, n=2, dim=1)
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=a2)

# Function to print test dataset masks
def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch, save_path, show_plot):
    model.eval()
    predictions = model(test_pets_targets)
    pred = nn.Softmax(dim=1)(predictions)
    pred_labels = pred.argmax(dim=1).unsqueeze(1)
    pred_mask = pred_labels.to(torch.float)

    iou = JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
    iou_accuracy = iou(pred_mask, test_pets_labels)
    net_sparse = maf.net_sparsity(model)
    node_sparse = maf.node_sparsity(model)
    node_sparsity_str = ', '.join([f'{sparsity:.4f}' for sparsity in node_sparse])
    
    title = (f'Epoch: {epoch:02d}, IoU: {iou_accuracy:.4f}, '
             f'Net: {net_sparse:.4f}, Node: {node_sparsity_str}')
    print(title)

    close_figures()

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_targets, nrow=7)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(pred_mask / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Predicted Labels")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"epoch_{epoch:02}.png"), format="png", bbox_inches="tight", pad_inches=0.4)

    if not show_plot:
        close_figures()
    else:
        plt.show()
