import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
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
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        TP = torch.sum(pred * target, dim=(2, 3, 4))
        FP = torch.sum(pred * (1 - target), dim=(2, 3, 4))
        FN = torch.sum((1 - pred) * target, dim=(2, 3, 4))

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - tversky_index  # Shape: (B, C), per-channel loss

        return tversky_loss.mean(dim=1).mean()  # Average over channels first, then over batch


class DiceTverskyHybridLoss(nn.Module):
    def __init__(self, dice_weight=0.7, tversky_weight=0.3, alpha=0.3, beta=0.7):
        super(DiceTverskyHybridLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha, beta)
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)  # Scalar
        tversky = self.tversky_loss(pred, target)  # Scalar

        hybrid_loss = self.dice_weight * dice + self.tversky_weight * tversky
        return hybrid_loss


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.dice_tversky_loss = DiceTverskyHybridLoss()

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

        loss = self.dice_tversky_loss(preds, target)  # Scalar loss
        return loss
