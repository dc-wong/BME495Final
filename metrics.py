import torch
import torch.nn.functional as F

def JaccardIndex(preds, target, smooth=1e-6):
    intersection = (preds * target).sum(dim=(2,3,4))
    union = preds.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean(dim=1).mean()

def Dice(preds, target, smooth=1e-6):
    intersection = (preds * target).sum(dim=(2,3,4))
    dice = (2 * intersection + smooth) / (preds.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + smooth)
    return dice.mean(dim=1).mean()

def Precision(preds, target, smooth=1e-6):
    tp = (preds * target).sum(dim=(2,3,4)).float()
    fp = (preds * (1 - target)).sum(dim=(2,3,4)).float()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.mean(dim=1).mean()

def Recall(preds, target, smooth=1e-6):
    tp = (preds * target).sum(dim=(2,3,4)).float()
    fn = ((1 - preds) * target).sum(dim=(2,3,4)).float()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.mean(dim=1).mean()

### find the average accuracy across the 4 metrics
def MultiAccuracy(preds, target, weights=None):
    ### weights is the weight given to each metric
    if weights is None:
        weights = torch.Tensor([0.25, 0.25, 0.25, 0.25])
    device = preds.device
    weights = weights.to(device)
    
    ### Ensure target has the same number of channels as preds
    target = target.expand(-1, preds.shape[1], -1, -1, -1)
    ### Threshold predictions to obtain binary outputs
    preds_thresh = (preds > 0.5).float()

    ### calcuate each metric
    J = JaccardIndex(preds_thresh, target)
    D = Dice(preds_thresh, target)
    P = Precision(preds_thresh, target)
    R = Recall(preds_thresh, target)

    ### return the weighted sum of the metrics
    metrics = torch.stack([J, D, P, R])
    return torch.dot(weights, metrics)
