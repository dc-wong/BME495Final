import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = torch.sum(pred * target, dim=(2, 3, 4))
        pred_sum = torch.sum(pred, dim=(2, 3, 4))
        target_sum = torch.sum(target, dim=(2, 3, 4))

        dice_score = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1 - dice_score 

        return dice_loss.mean(dim=1).mean() 

class ChannelWiseBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        BCEloss = F.binary_cross_entropy(preds, target, reduction='none').mean(dim=(2, 3, 4)).mean(dim=1).mean()
        return BCEloss

### find the hybrid loss, which is the sume of Dice and BCE
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce = ChannelWiseBCELoss()
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target) 
        bce = self.bce(pred, target)
        hybrid_loss = bce + dice
        return hybrid_loss

### The loss function
### the primary purpose is a wrawpped for HybridLoss() that reshapes the mask label
class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = HybridLoss()

    def forward(self, preds, target):
        ### expand the labels to have the same number of channels as the output
        target = target.expand(-1, preds.shape[1], -1, -1, -1)  

        ### calculate the loss
        loss = self.loss(preds, target)  
        return loss
