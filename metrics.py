import torch
import torch.nn.functional as F

def JaccardIndex(preds, target, smooth=1e-6):
    """ Computes Jaccard Index (IoU) per channel, then averages over channels and batch. """
    target = target.long()
    
    intersection = torch.logical_and(preds, target).sum(dim=(2,3,4))  # Sum over spatial dims
    union = torch.logical_or(preds, target).sum(dim=(2,3,4)) 
    
    iou = (intersection + smooth) / (union + smooth)  # Avoid division by zero
    return iou.mean(dim=1).mean()  # Mean over channels, then batch

def Dice(preds, target, smooth=1e-6):
    """ Computes Dice Coefficient per channel, then averages over channels and batch. """
    preds = preds.float()
    target = target.float()

    intersection = torch.logical_and(preds, target).sum(dim=(2,3,4))
    dice = (2 * intersection + smooth) / (preds.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + smooth)
    
    return dice.mean(dim=1).mean()  # Mean over channels, then batch

def Precision(preds, target, smooth=1e-6):
    """ Computes Precision per channel, then averages over channels and batch. """
    tp = torch.logical_and(preds, target).sum(dim=(2,3,4)).float()
    fp = torch.logical_and(preds, torch.logical_not(target)).sum(dim=(2,3,4)).float()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.mean(dim=1).mean()  # Mean over channels, then batch

def Recall(preds, target, smooth=1e-6):
    """ Computes Recall per channel, then averages over channels and batch. """
    tp = torch.logical_and(preds, target).sum(dim=(2,3,4)).float()
    fn = torch.logical_and(torch.logical_not(preds), target).sum(dim=(2,3,4)).float()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.mean(dim=1).mean()  # Mean over channels, then batch

def UnifiedAccuracy(preds, target, weights=None):
    """ Computes a weighted accuracy score per channel, then averages over channels and batch. """
    if weights is None:
        weights = torch.Tensor([0.25, 0.25, 0.25, 0.25])
    device = preds.device
    weights = weights.to(device)

    J = JaccardIndex(preds, target)
    D = Dice(preds, target)
    P = Precision(preds, target)
    R = Recall(preds, target)
    
    metrics = torch.stack([J, D, P, R])
    return torch.dot(weights, metrics)

def MeanPoolOverChannels(tensor):
    """ Computes mean pooling over the channel dimension. """
    return torch.mean(tensor, dim=1, keepdim=True)  # Reduce channel dimension

def MultiAccuracy(preds, target, weights=None):
    """
    Computes a weighted accuracy score per channel, then averages over channels and batch.
    Handles multi-channel predictions by thresholding and mean pooling.
    """
    if weights is None:
        weights = torch.Tensor([0.25, 0.25, 0.25, 0.25])
    device = preds.device
    weights = weights.to(device)

    # Ensure target has the same shape as predictions
    target = target.expand_as(preds)

    # Mean pool over channels and threshold predictions
    preds = MeanPoolOverChannels(preds)
    preds = (preds > 0.5).long()

    # Compute metrics
    J = JaccardIndex(preds, target)
    D = Dice(preds, target)
    P = Precision(preds, target)
    R = Recall(preds, target)

    metrics = torch.stack([J, D, P, R])
    return torch.dot(weights, metrics)

# def ASSD(preds, target):
#     preds = preds.float()
#     target = target.float()

#     batch_size = preds.shape[0]
#     assd_list = []

#     for i in range(batch_size):
#         pred_surface = torch.nonzero(preds[i])
#         target_surface = torch.nonzero(target[i])

#         if pred_surface.numel() == 0 or target_surface.numel() == 0:
#             assd_list.append(torch.tensor(float('inf'), device=preds.device))
#             continue

#         # Compute distances
#         dists_pred_to_target = torch.cdist(pred_surface.float(), target_surface.float(), p=2)
#         dists_target_to_pred = torch.cdist(target_surface.float(), pred_surface.float(), p=2)

#         # Compute ASSD (mean of min distances)
#         assd = (dists_pred_to_target.min(dim=1)[0].mean() + dists_target_to_pred.min(dim=1)[0].mean()) / 2
#         assd_list.append(assd)

#     return torch.stack(assd_list).mean()

# def HD95(preds, target):
#     """
#     Computes the 95th percentile of the Hausdorff Distance (HD95).
#     """
#     preds = preds.float()
#     target = target.float()

#     batch_size = preds.shape[0]
#     hd95_list = []

#     for i in range(batch_size):
#         pred_surface = torch.nonzero(preds[i])
#         target_surface = torch.nonzero(target[i])

#         if pred_surface.numel() == 0 or target_surface.numel() == 0:
#             hd95_list.append(torch.tensor(float('inf'), device=preds.device))
#             continue

#         # Compute pairwise distances efficiently with PyTorch
#         dists_pred_to_target = torch.cdist(pred_surface.float(), target_surface.float(), p=2)
#         dists_target_to_pred = torch.cdist(target_surface.float(), pred_surface.float(), p=2)

#         # Get the 95th percentile
#         hd95 = torch.quantile(torch.cat([dists_pred_to_target.min(dim=1)[0], dists_target_to_pred.min(dim=1)[0]]), 0.95)
#         hd95_list.append(hd95)

#     return torch.stack(hd95_list).mean()