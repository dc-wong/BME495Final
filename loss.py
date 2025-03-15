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
        dice_loss = 1 - dice_score  # Shape: (B, C), per-channel loss

        return dice_loss.mean(dim=1).mean()  # Average over channels first, then over batch


class TverskyLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=0.3, beta=0.7):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        TP = torch.sum(pred * target, dim=(2, 3, 4))
        FP = torch.sum(pred * (1 - target), dim=(2, 3, 4))
        FN = torch.sum((1 - pred) * target, dim=(2, 3, 4))

        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - tversky_index  # Shape: (B, C), per-channel loss

        return tversky_loss.mean(dim=1).mean()  # Average over channels first, then over batch

class ChannelWiseBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, target):
        BCEloss = F.binary_cross_entropy(preds, target, reduction='none').mean(dim=(2, 3, 4)).mean(dim=1).mean()
        return BCEloss


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.dice_loss = DiceLoss()
        #self.tversky_loss = TverskyLoss(alpha, beta)
        self.bce = ChannelWiseBCELoss()
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)  # Scalar
        #tversky = self.tversky_loss(pred, target)  # Scalar
        bce = self.bce(pred, target)

        hybrid_loss = bce + dice
        return hybrid_loss


class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = HybridLoss()

    def forward(self, preds, target):
        """
        Args:
        preds: Tensor of shape (B, C, D, H, W), where C is the number of segmentation outputs.
        target: Tensor of shape (B, 1, D, H, W), representing the single-channel ground truth.

        Returns:
        Scalar loss averaged across all channels and batches.
        """
        # target = F.interpolate(target, size=preds.shape[2:], mode='trilinear', align_corners=False)
        target = target.expand(-1, preds.shape[1], -1, -1, -1)  # Expand to (B, C, D, H, W)

        loss = self.loss(preds, target)  # Scalar loss
        return loss
