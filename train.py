import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.optim as optim
import torchmetrics as TM
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
# Convert a PyTorch tensor into a PIL image
t2img = T.ToPILImage()

# Convert a PIL image into a PyTorch tensor
img2t = T.ToTensor()

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
from torchmetrics.classification import Dice
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
iou_metric = TM.JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
def train_step(conf, model, opt, train_loader, l1_lambda=0.01):
    tot_loss = 0.0
    # Initialize the Jaccard Index (IOU) metric with 3 classes, ignoring class 1
    iou_metric = TM.JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
    # Initialize the Dice metric with 3 classes, average='micro' and ignoring class 1
    dice_metric = TM.Dice(num_classes=3, average='micro', ignore_index=1)
    # Use Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    for batch_id, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        logits = model(x)
        y = y.squeeze(1).long()
        loss = criterion(logits, y)
        
        if isinstance(opt, optim.SGD):
            l1_l2_regularization = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    l1_norm = torch.norm(param, p=1)  # L1 norm of parameter tensor
                    l2_norm = torch.norm(param, p=2)  # L2 norm of parameter tensor
                    regularization_term = conf.lamda_1 * ((conf.lamda_0) * l1_norm + (1 - conf.lamda_0) * torch.sqrt(torch.tensor(param.shape[-1], dtype=torch.float)) * l2_norm)
                    l1_l2_regularization += regularization_term
            
            # Add regularization term to the loss
            loss += l1_l2_regularization
        
        loss.backward()
        opt.step()
        
        tot_loss += loss.item()

        # Update IOU and Dice metrics
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


def validation_step(model, val_loader):
    tot_loss = 0.0
    pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro', ignore_index=1)
    iou_metric = TM.JaccardIndex(num_classes=3, task='multiclass', average='micro', ignore_index=1)
    dice_metric = Dice(num_classes=3, average='micro', ignore_index=1)
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(val_loader):
            logits = model(x)
            y = y.squeeze(1).long()
            loss = criterion(logits, y)

            tot_loss += loss.item()

            # Update IOU, Dice and Pixel metrics
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
    net_sparse = maf.net_sparsity(model)
    node_sparse = maf.node_sparsity(model)
    node_sparsity_str = ', '.join([f'{sparsity:.4f}' for sparsity in node_sparse])

    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}, IOU Accuracy: {iou_accuracy:.4f}, Dice Accuracy: {dice_accuracy:.4f}, Net: {net_sparse:.4f}, Node: {node_sparsity_str}")
    return {'loss': avg_loss, 'IOU Accuracy': iou_accuracy, 'Dice Accuracy': dice_accuracy, 'Net Sparsity': net_sparse, 'Node': node_sparsity_str}


def prune_step(model, a1=0.01, a2=0.01, conv_group=True):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and conv_group:
            prune.ln_structured(module, name='weight', amount=a1, n=2, dim=0)
            prune.ln_structured(module, name='weight', amount=a1, n=2, dim=1)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=a2)


def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch, save_path, show_plot):
    model.eval()
    predictions = model(test_pets_targets)
    test_pets_labels = test_pets_labels
    # print("Predictions Shape: {}".format(predictions.shape))
    pred = nn.Softmax(dim=1)(predictions)

    pred_labels = pred.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1)
    # print("pred_labels.shape: {}".format(pred_labels.shape))
    pred_mask = pred_labels.to(torch.float)

    iou =TM.classification.MulticlassJaccardIndex(3, average='micro',ignore_index=1)
    iou_accuracy = iou(pred_mask, test_pets_labels)
    net_sparse = maf.net_sparsity(model)
    node_sparse = maf.node_sparsity(model)
    node_sparsity_str = ', '.join([f'{sparsity:.4f}' for sparsity in node_sparse])
    title = (f'Epoch: {epoch:02d}, IoU: {iou_accuracy:.4f},], '
         f'Net: {net_sparse:.4f}, Node: {node_sparsity_str}]')
    print(title)

    # Close all previously open figures.
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
    # end if

    if show_plot is False:
        close_figures()
    else:
        plt.show()
    # end if
# end def