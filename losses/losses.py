import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import t2np


def normal_fl_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Normal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0 
    weights[mask_lower] = fl_weights[mask_lower]

    return weights


def abnormal_fl_weighting(logps, gamma=2, alpha=0.53, normalizer=10):
    """
    Abnormal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -1.0
    mask_lower = logps <= -1.0
    probs = torch.exp(logps)
    fl_weights = alpha * (1 + probs).pow(gamma) * (1 / torch.abs(logps))
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_lower] = 1.0 
    weights[mask_larger] = fl_weights[mask_larger]

    return weights


def get_logp_boundary(logps, mask, pos_beta=0.05, margin_tau=0.1, normalizer=10):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    normal_logps = logps[mask == 0].detach()
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]
    
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary

    return b_n, b_a

#Proviamo il pos_beta adattivo

'''
def get_logp_boundary(
    logps,
    mask,
    pos_beta=0.05,
    margin_tau=0.1,
    normalizer=10,
    adaptive=False,
    min_eps=0.01,
    max_eps=0.1,
    n_steps=10,
    epoch=None,
    warmup_epochs=5
):
    """
    Adaptive or fixed boundary from normal log-likelihoods.
    If adaptive=True and epoch >= warmup_epochs, finds optimal pos_beta by maximizing gap.

    Args:
        logps: tensor of log-likelihoods
        mask: tensor (0 = normal, 1 = anomaly)
        pos_beta: used during warm-up or if adaptive=False
        margin_tau: b_a = b_n - tau
        adaptive: enable adaptive pos_beta
        epoch: current epoch number
        warmup_epochs: fixed pos_beta used before this
    """
    normal_logps = logps[mask == 0].detach()

    use_adaptive = adaptive and (epoch is None or epoch >= warmup_epochs)
    
    if use_adaptive:
        best_eps = pos_beta
        best_gap = -np.inf
        for eps in np.linspace(min_eps, max_eps, n_steps):
            threshold = np.percentile(t2np(normal_logps), eps * 100)
            inside = normal_logps[normal_logps >= threshold]
            outside = normal_logps[normal_logps < threshold]
            if len(inside) == 0 or len(outside) == 0:
                continue
            gap = torch.min(inside) - torch.max(outside)
            if gap > best_gap:
                best_gap = gap
                best_eps = eps
                #print(f"nuovo epsilon trovato pari a: {eps}")
        pos_beta = best_eps

    # Calcolo soglia finale
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]
    b_n = b_n / normalizer
    b_a = b_n - margin_tau
    return b_n, b_a

'''
def calculate_bg_spp_loss(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    b_a = boundaries[1]
    anomaly_logps = logps[mask == 1]    
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
    loss_a = anomaly_logps_inter - b_a

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
        ano_weights = weights[mask == 1][anomaly_logps >= b_a]
        loss_a = loss_a * ano_weights
    
    loss_n = torch.mean(loss_n)
    loss_a = torch.mean(loss_a)

    return loss_n, loss_a


def calculate_bg_spp_loss_normal(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
    
    loss_n = torch.mean(loss_n)

    return loss_n