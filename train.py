import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# torchvision
from torchvision import datasets, transforms

# Define a custom IoU Metric for validating the model.
def IoUMetric(pred, gt, softmax=False):
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    # end if
    
    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)
    # print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax).log())
    # end def
# end class


# Train step
def train_step(conf, model, opt, train_loader):
    tot_loss = 0.0
    tot_steps = 0
    
    for batch_id, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        logits = model(x)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # Ensure the target tensor has the correct shape
        y = y.squeeze(1).long()
        # Calculate loss
        loss = criterion(logits, y)

        if hasattr(conf, "weight_reg"):
            loss += conf.weight_reg(model)
        loss.backward()
        opt.step()
        tot_loss += loss.item()
    
    # Print the current Dice score and loss after each epoch
    print("Train Loss:", tot_loss/(batch_id+1))
    
    return {'loss': tot_loss}            


class best_model:
    '''saves the best model'''
    def __init__(self, best_model=None, gamma = 0.0, goal_acc = 0.0):
        # stores best seen score and model
        self.best_score = 0.0
        
        # if specified, a copy of the model gets saved into this variable
        self.best_model = best_model

        # score function
        def score_fun(train_acc, test_acc):
            return gamma * train_acc + (1-gamma) * test_acc + (train_acc > goal_acc)
        self.score_fun = score_fun
        
    
    def __call__(self, train_acc, val_acc, model=None):
        # evaluate score
        score = self.score_fun(train_acc, val_acc)
        if score >= self.best_score:
            self.best_score = score
            # store model
            if self.best_model is not None:
                self.best_model.load_state_dict(model.state_dict())