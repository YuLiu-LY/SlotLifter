# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from piq import ssim 
from piq import psnr 
import lpips


def average_ari(masks, masks_gt, fg_only=False, reduction='mean'):
    r'''
    Input:
        masks: (B, K, N)
        masks_gt: (B, N)
    '''
    ari = []
    masks = masks.argmax(dim=1)
    B = masks.shape[0]
    for i in range(B):
        m = masks[i].cpu().numpy()
        m_gt = masks_gt[i].cpu().numpy()
        if fg_only:
            m = m[np.where(m_gt > 0)]
            m_gt = m_gt[np.where(m_gt > 0)]
        score = adjusted_rand_score(m, m_gt)
        ari.append(score)
    if reduction == 'mean':
        return torch.Tensor(ari).mean()
    else:
        return torch.Tensor(ari)


def imask2bmask(imasks, ignore_index=None):
    r"""Convert index mask to binary mask.
    Args:
        imask: index mask, shape (B, N)
    Returns:
        bmasks: # a list of (K, N), len = B
    """
    B, N = imasks.shape
    bmasks = []
    for i in range(B):
        imask = imasks[i:i+1] # (1, N)
        classes = imask.unique().tolist()
        if ignore_index in classes:
            classes.remove(ignore_index)
        bmask = [imask == c for c in classes]
        bmask = torch.cat(bmask, dim=0) # (K, N)
        bmasks.append(bmask.float())
    # can't use torch.stack because of different K
    return bmasks 


def mean_best_overlap(masks, masks_gt, fg_only=False, reduction='mean'):
    r"""Compute the best overlap between predicted and ground truth masks.
    Args:
        masks: predicted masks, shape (B, K, N), binary N = H*W
        masks_gt: ground truth masks, shape (B, N), index
    """
    B = masks.shape[0]
    ignore_index = None
    if fg_only:
        ignore_index = 0
    bmasks_gt = imask2bmask(masks_gt, ignore_index=ignore_index)  # a list of (K, N), len = B
    mean_best_overlap = []
    mOR = []
    for i in range(B):
        mask = masks[i].unsqueeze(0) > 0.5 # (1, K, N)
        mask_gt = bmasks_gt[i].unsqueeze(1) > 0.5 # (K_gt, 1, N)
        # Compute IOU
        eps = 1e-8
        intersection = (mask * mask_gt).sum(-1)
        union = (mask + mask_gt).sum(-1)
        iou = intersection / (union + eps) # (K_gt, K)
        # Compute best overlap
        best_overlap, _ = torch.max(iou, dim=1)
        # Compute mean best overlap
        mean_best_overlap.append(best_overlap.mean())
        mOR.append((best_overlap > 0.5).float().mean())
    if reduction == 'mean':
        return torch.stack(mean_best_overlap).mean()
    # , torch.stack(mOR).mean()
    else:
        return torch.stack(mean_best_overlap)
    # , torch.stack(mOR)


def iou_loss(pred, target):
    """
    Compute the iou loss: 1 - iou
    pred: [K, N]
    targets: [Kt, N]
    """
    eps = 1e-8
    pred = pred > 0.5 # [K, N]
    target = target > 0.5
    intersection = (pred[:, None] & target[None]).sum(-1).float() # [K, Kt]
    union = (pred[:, None] | target[None]).sum(-1).float() + eps # [K, Kt]
    loss = 1 - (intersection / union) # [K, Kt]
    return loss # [K, Kt]
iou_loss_jit = torch.jit.script(iou_loss) 


class Matcher():
    @torch.no_grad()
    def forward(self, pred, target):
        r"""
        pred: [K, N]
        targets: [Kt, N]
        """
        loss = iou_loss_jit(pred, target)
        row_ind, col_ind = linear_sum_assignment(loss.cpu().numpy())
        return torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)
    
    @torch.no_grad()
    def batch_forward(self, pred, targets):
        """
        pred: [B, K, N]
        targets: list of B x [Kt, N] Kt can be different for each target
        """
        indices = []
        for i in range(pred.shape[0]):
            indices.append(self.forward(pred[i], targets[i]))
        return indices


@torch.no_grad()
def compute_iou(pred, target):
    """
    Input:
        x: [K, N]
        y: [K, N]
    Return:
        iou: [K, N]
    """
    eps = 1e-8
    pred = pred > 0.5 # [K, N]
    target = target > 0.5
    intersection = (pred & target).sum(-1).float()
    union = (pred | target).sum(-1).float() + eps # [K]
    return (intersection / union).mean()
compute_iou_jit = torch.jit.script(compute_iou)


def matchedIoU(preds, targets, matcher, fg_only=False, reduction="mean"):
    r"""
    Input:
        pred: [B, K, N]
        targets: [B, N] 
    Return:
        IoU: [1] or [B]
    """
    if preds.dim() == 2: # [K, N]
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)
    
    ious = []
    B = preds.shape[0]
    ignore_index = None
    if fg_only:
        ignore_index = 0
    targets = imask2bmask(targets, ignore_index) # a list of [K1, N], len = B
    for i in range(B):
        tgt = targets[i]
        pred = preds[i] # [K, N]
        src_idx, tgt_idx = matcher.forward(pred, tgt) 
        src_pred = pred[src_idx] # [K1, N]
        tgt_mask = tgt[tgt_idx] # [K1, N]
        ious.append(compute_iou_jit(src_pred, tgt_mask))
    ious = torch.stack(ious)
    if reduction == "mean":
        return ious.mean()
    else:
        return ious


matcher = Matcher()
SEGMETRICS = {
    "hiou": partial(matchedIoU, matcher=matcher), # hungarian matched iou
    "hiou_fg": partial(matchedIoU, fg_only=True, matcher=matcher),
    "mbo": mean_best_overlap, # mean best overlap
    "mbo_fg": partial(mean_best_overlap, fg_only=True),
    "ari": average_ari,
    "ari_fg": partial(average_ari, fg_only=True),
}
class SegMetrics(nn.Module):
    def __init__(self, metrics=["hiou", "ari", "ari_fg"]):
        super().__init__()
        self.metrics = {}
        for m in metrics:
            self.metrics[m] = SEGMETRICS[m]
    
    def forward(self, preds, targets):
        r"""
        Input:
            preds: [B, N, K]
            targets: [B, N]
        Return:
            metrics: dict of metrics
        """
        metrics = {}
        valid = targets.sum(-1) > 0
        preds = preds[valid]
        targets = targets[valid]
        preds = F.one_hot(preds.argmax(dim=-1), num_classes=preds.shape[-1]).permute(0, 2, 1).float() # [B, K, N]
        for k, v in self.metrics.items():
            if valid.sum() > 0:
                metrics[k] = v(preds, targets)
            else:
                metrics[k] = torch.tensor(1).to(preds.device)
        return metrics
    
    def compute(self, preds, targets, metric='hiou'):
        return self.metrics[metric](preds, targets)
    
    def metrics_name(self):
        return list(self.metrics.keys())


class ReconMetrics(nn.Module):
    def __init__(self, lpips_net='vgg'):
        super().__init__()
        self.metrics = {
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips.LPIPS(net=lpips_net),
        }

    def forward(self, preds, targets):
        r"""
        Input:
            preds: [B, C, H, W]
            targets: [B, C, H, W]
        Return:
            metrics: dict of metrics
        """
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = v(preds, targets).mean()
        return metrics
        
    def compute(self, preds, targets, metric='psnr'):
        return self.metrics[metric](preds, targets)
    
    def metrics_name(self):
        return list(self.metrics.keys())
    
    def set_divice(self, device):
        self.metrics["lpips"].to(device)