"""
Loss functions for the hybrid GAT+VAE model.

L_total = L_classification (Focal Loss)
        + lambda_1 * L_reconstruction (MSE on raw features)
        + lambda_2 * L_KL (KL divergence)
"""
import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for imbalanced classification.
    alpha is the weight for the POSITIVE (fraud) class.
    For fraud detection (~3% positive rate), alpha=0.75 upweights the minority class.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal = alpha_t * (1 - pt) ** gamma * bce
    return focal.mean() if reduction == "mean" else focal


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x)


def total_loss(
    outputs: dict,
    targets: torch.Tensor,
    lambda1: float = 0.1,
    lambda2: float = 0.01,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss. Uses raw_txn (not h_t) for VAE reconstruction.
    """
    l_cls = focal_loss(outputs["logit"], targets, gamma=focal_gamma, alpha=focal_alpha)

    # VAE reconstructs raw features, not GAT embeddings
    recon_target = outputs.get("raw_txn", outputs.get("h_t"))
    l_recon = reconstruction_loss(recon_target, outputs["x_recon"])
    l_kl = kl_divergence(outputs["mu"], outputs["logvar"])

    loss = l_cls + lambda1 * l_recon + lambda2 * l_kl

    return loss, {
        "total": loss.item(),
        "classification": l_cls.item(),
        "reconstruction": l_recon.item(),
        "kl": l_kl.item(),
    }
