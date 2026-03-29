"""
Hybrid GAT + VAE fraud detection model (v2 — Dual-Branch).

Architecture:
    Branch A (Graph):  HeteroData -> HeteroGATEncoder -> h_t (gat_out dim)
    Branch B (Anomaly): raw_features -> TransactionVAE -> recon_err (normalized)

    Fusion: classifier([h_t || normalized_recon_err]) -> fraud_score
"""
import torch
import torch.nn as nn
from .gat_encoder import HeteroGATEncoder
from .vae import TransactionVAE


class HybridGATVAE(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        in_channels: dict,
        raw_txn_dim: int,             # raw transaction feature dim (e.g. 383)
        gat_hidden: int = 128,
        gat_out: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        vae_latent: int = 32,
        vae_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.raw_txn_dim = raw_txn_dim

        # Branch A: Graph encoder
        self.gat = HeteroGATEncoder(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=gat_hidden,
            out_channels=gat_out,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        # Branch B: VAE on raw features (NOT GAT embeddings)
        self.vae = TransactionVAE(
            input_dim=raw_txn_dim,
            latent_dim=vae_latent,
            hidden_dim=vae_hidden,
        )

        # Normalize recon_err before classifier
        self.recon_bn = nn.BatchNorm1d(1)

        # Classifier: [h_t (gat_out) || normalized_recon_err (1)] -> fraud logit
        cls_input_dim = gat_out + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_dict, edge_index_dict, raw_txn_features=None):
        """
        Args:
            x_dict: node feature dict for HeteroConv
            edge_index_dict: edge index dict for HeteroConv
            raw_txn_features: raw transaction features for VAE (before GAT).
                              If None, falls back to x_dict["transaction"].
        """
        # Branch A: Graph encoding
        h_dict = self.gat(x_dict, edge_index_dict)
        h_t = h_dict["transaction"]

        # Branch B: VAE on raw features
        if raw_txn_features is None:
            raw_txn_features = x_dict["transaction"]
        x_recon, mu, logvar = self.vae(raw_txn_features)

        # Reconstruction error (per-sample MSE)
        recon_err = (raw_txn_features - x_recon).pow(2).mean(dim=1, keepdim=True)
        recon_err_norm = self.recon_bn(recon_err)

        # Fusion: concat graph embedding + normalized anomaly score
        cls_input = torch.cat([h_t, recon_err_norm], dim=1)
        logit = self.classifier(cls_input).squeeze(-1)
        fraud_score = torch.sigmoid(logit)

        return {
            "logit": logit,
            "fraud_score": fraud_score,
            "h_t": h_t,
            "raw_txn": raw_txn_features,
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "recon_err": recon_err,
        }
