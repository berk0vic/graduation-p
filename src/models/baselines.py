"""
Baseline models for comparison:
  - XGBoost (tabular)
  - Random Forest (tabular)
  - GCN (homogeneous graph, no attention)
  - GAT-only (heterogeneous, no VAE)
  - VAE-only (unsupervised anomaly detection)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, GATConv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ── Tabular baselines ──────────────────────────────────────────────────────────

def get_xgboost(scale_pos_weight: float = 100.0, **kwargs) -> XGBClassifier:
    """scale_pos_weight = n_neg / n_pos to handle imbalance."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        tree_method="hist",
        **kwargs,
    )


def get_random_forest(class_weight: str = "balanced", **kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight=class_weight,
        n_jobs=-1,
        **kwargs,
    )


# ── GCN baseline (homogeneous graph) ─────────────────────────────────────────

class GCNBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, out: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out)
        self.classifier = nn.Linear(out, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        return self.classifier(h).squeeze(1)


# ── GAT-only baseline (no VAE) ────────────────────────────────────────────────

class GATOnlyBaseline(nn.Module):
    def __init__(self, metadata: tuple, in_channels: dict, hidden: int = 64, out: int = 32, heads: int = 4):
        super().__init__()
        node_types, edge_types = metadata

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], hidden) for nt in node_types if nt in in_channels
        })

        self.conv1 = HeteroConv({
            et: GATConv(hidden, hidden, heads=heads, concat=True, add_self_loops=False)
            for et in edge_types
        }, aggr="sum")

        self.conv2 = HeteroConv({
            et: GATConv(hidden * heads, out, heads=1, concat=False, add_self_loops=False)
            for et in edge_types
        }, aggr="sum")

        self.classifier = nn.Linear(out, 1)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {nt: F.elu(self.input_proj[nt](x)) for nt, x in x_dict.items() if nt in self.input_proj}
        h = {nt: F.elu(t) for nt, t in self.conv1(h, edge_index_dict).items()}
        h = {nt: F.elu(t) for nt, t in self.conv2(h, edge_index_dict).items()}
        return self.classifier(h["transaction"]).squeeze(1)


# ── VAE-only baseline (unsupervised) ─────────────────────────────────────────

class VAEOnlyBaseline(nn.Module):
    """Anomaly score = reconstruction error from VAE on raw tabular features."""
    def __init__(self, input_dim: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        from .vae import TransactionVAE
        self.vae = TransactionVAE(input_dim, latent, hidden)

    def forward(self, x: torch.Tensor):
        x_recon, mu, logvar = self.vae(x)
        recon_err = F.mse_loss(x_recon, x, reduction="none").mean(dim=1)
        return recon_err  # higher = more anomalous
