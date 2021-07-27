import torch
import torch.nn.functional as F
import math


def quantile_calibration(mean, std, targets):
    num_quantiles = 10
    quantiles = torch.linspace(0.05, 0.95, num_quantiles).to(mean.device)  # [num_quantiles]
    quantiles = quantiles.view(-1, 1, 1)   # [num_quantiles x 1 x 1]

    z_dist = torch.distributions.Normal(
        torch.tensor((0.0,), device=mean.device),
        torch.tensor((1.0,), device=mean.device),
    )
    tail_probs = (1 - quantiles) / 2
    z_scores = z_dist.icdf(1 - tail_probs)

    targets = targets.unsqueeze(0)  # [1 x num_samples x num_tasks]
    pred_mean = mean.unsqueeze(0)  # [1 x num_samples x num_tasks]
    pred_std = std.unsqueeze(0)  # [1 x num_samples x num_tasks]
    lb = pred_mean - z_scores * pred_std
    ub = pred_mean + z_scores * pred_std

    targets_in_region = torch.le(lb, targets) * torch.le(targets, ub)
    occupancy_rates = targets_in_region.float().mean(1, keepdim=True)  # [1 x 1 x num_tasks]
    ece = (occupancy_rates - quantiles).abs().mean().item()
    calibration_metrics = {
        f"{quantile.item():.2f}_quantile": occ_rate.mean().item()
        for quantile, occ_rate in zip(quantiles, occupancy_rates)
    }
    calibration_metrics["ece"] = ece
    return calibration_metrics


def top_k_accuracy(logits, targets, k=1):
    if k > 1:
        raise NotImplementedError

    pred_class = logits.argmax(dim=-1)
    num_correct = (pred_class == targets).float().sum().item()
    return num_correct / logits.shape[0]
