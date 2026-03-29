"""
Training loop for HybridGATVAE with two-phase training.

Phase 1: Pre-train VAE (freeze GAT + classifier) — learn normal patterns
Phase 2: End-to-end fine-tune with differential LR — VAE gets lower LR
"""
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .losses import total_loss, reconstruction_loss, kl_divergence


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        device: str = "auto",
    ):
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lr = lr
        self.weight_decay = weight_decay

    def _get_raw_txn(self, data):
        """Get raw transaction features for VAE input."""
        return data["transaction"].x.to(self.device)

    # ───────────────────────────────────
    # Phase 1: VAE pre-training only
    # ───────────────────────────────────
    def pretrain_vae(self, data, epochs: int = 50, lr: float = 1e-3) -> list[dict]:
        """Pre-train VAE on raw transaction features (normal pattern learning)."""
        data = data.to(self.device)
        raw_txn = self._get_raw_txn(data)

        # Freeze GAT + classifier, only train VAE
        for p in self.model.gat.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = False
        for p in self.model.vae.parameters():
            p.requires_grad = True

        optimizer = Adam(self.model.vae.parameters(), lr=lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        mask = data["transaction"].train_mask if hasattr(data["transaction"], "train_mask") else None

        history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            x = raw_txn[mask] if mask is not None else raw_txn
            x_recon, mu, logvar = self.model.vae(x)

            l_recon = reconstruction_loss(x, x_recon)
            l_kl = kl_divergence(mu, logvar)
            loss = l_recon + self.lambda2 * l_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.vae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss.item())

            row = {
                "epoch": epoch,
                "recon_loss": l_recon.item(),
                "kl_loss": l_kl.item(),
                "total": loss.item(),
            }
            history.append(row)
            if epoch % 10 == 0:
                print(f"[VAE Phase 1] Epoch {epoch:3d} | recon: {l_recon.item():.4f} | kl: {l_kl.item():.4f}")

        # Unfreeze everything for Phase 2
        for p in self.model.parameters():
            p.requires_grad = True

        print(f"[VAE Phase 1] Done. Final recon: {history[-1]['recon_loss']:.4f}")
        return history

    # ───────────────────────────────────
    # Phase 2: End-to-end with differential LR
    # ───────────────────────────────────
    def _build_optimizer_phase2(self):
        """Differential LR: VAE gets 0.1x learning rate to prevent catastrophic forgetting."""
        return Adam([
            {"params": self.model.gat.parameters(), "lr": self.lr},
            {"params": self.model.vae.parameters(), "lr": self.lr * 0.1},  # 10x lower
            {"params": self.model.recon_bn.parameters(), "lr": self.lr},
            {"params": self.model.classifier.parameters(), "lr": self.lr * 2},  # faster
        ], weight_decay=self.weight_decay)

    def train_epoch(self, data) -> dict:
        self.model.train()
        data = data.to(self.device)
        raw_txn = self._get_raw_txn(data)

        self.optimizer.zero_grad()
        outputs = self.model(data.x_dict, data.edge_index_dict, raw_txn_features=raw_txn)

        mask = data["transaction"].train_mask
        targets = data["transaction"].y

        masked_outputs = {
            "logit": outputs["logit"][mask],
            "raw_txn": outputs["raw_txn"][mask],
            "x_recon": outputs["x_recon"][mask],
            "mu": outputs["mu"][mask],
            "logvar": outputs["logvar"][mask],
        }
        loss, loss_components = total_loss(
            masked_outputs, targets[mask],
            self.lambda1, self.lambda2,
            self.focal_gamma, self.focal_alpha,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_components

    @torch.no_grad()
    def evaluate(self, data) -> dict:
        self.model.eval()
        data = data.to(self.device)
        raw_txn = self._get_raw_txn(data)

        outputs = self.model(data.x_dict, data.edge_index_dict, raw_txn_features=raw_txn)

        if hasattr(data["transaction"], "val_mask"):
            mask = data["transaction"].val_mask
        else:
            mask = torch.ones(data["transaction"].y.shape[0], dtype=torch.bool)
        mask = mask.to(self.device)

        targets = data["transaction"].y
        masked_outputs = {
            "logit": outputs["logit"][mask],
            "raw_txn": outputs["raw_txn"][mask],
            "x_recon": outputs["x_recon"][mask],
            "mu": outputs["mu"][mask],
            "logvar": outputs["logvar"][mask],
        }
        _, loss_components = total_loss(
            masked_outputs, targets[mask],
            self.lambda1, self.lambda2,
            self.focal_gamma, self.focal_alpha,
        )

        return {
            **loss_components,
            "fraud_scores": outputs["fraud_score"].cpu(),
            "recon_err": outputs["recon_err"].cpu(),
            "targets": targets.cpu(),
        }

    def fit(
        self,
        train_data,
        val_data=None,
        epochs: int = 100,
        patience: int = 15,
        vae_pretrain_epochs: int = 50,
    ) -> list[dict]:
        """
        Full training pipeline:
        1. Phase 1: VAE pre-training
        2. Phase 2: End-to-end with differential LR + early stopping
        """
        # Phase 1: VAE pre-training
        print("=" * 60)
        print("PHASE 1: VAE Pre-training")
        print("=" * 60)
        vae_history = self.pretrain_vae(train_data, epochs=vae_pretrain_epochs)

        # Phase 2: End-to-end
        print("\n" + "=" * 60)
        print("PHASE 2: End-to-end training (differential LR)")
        print("=" * 60)
        self.optimizer = self._build_optimizer_phase2()
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=7, factor=0.5)

        history = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_data)
            row = {"epoch": epoch, "train_loss": train_metrics["total"],
                   "train_cls": train_metrics["classification"],
                   "train_recon": train_metrics["reconstruction"]}

            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                row["val_loss"] = val_metrics["total"]
                row["val_recon"] = val_metrics["reconstruction"]
                self.scheduler.step(val_metrics["total"])

                # Early stopping
                if val_metrics["total"] < best_val_loss:
                    best_val_loss = val_metrics["total"]
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

            history.append(row)
            if epoch % 5 == 0:
                parts = [f"Epoch {epoch:3d}"]
                parts.append(f"cls: {train_metrics['classification']:.4f}")
                parts.append(f"recon: {train_metrics['reconstruction']:.4f}")
                if "val_loss" in row:
                    parts.append(f"val: {row['val_loss']:.4f}")
                    parts.append(f"val_recon: {row['val_recon']:.4f}")
                print(" | ".join(parts))

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model (val_loss={best_val_loss:.4f})")

        return history


# ───────────────────────────────────
# Temporal train/val/test split
# ───────────────────────────────────
def temporal_split(data, time_col_values, val_ratio=0.15, test_ratio=0.15):
    """
    Split by time: oldest -> train, middle -> val, newest -> test.
    Prevents temporal leakage (future info in training).

    Args:
        data: HeteroData with transaction nodes
        time_col_values: array of timestamps (TransactionDT) for each transaction
        val_ratio: fraction for validation
        test_ratio: fraction for test
    Returns:
        data with train_mask, val_mask, test_mask on transaction nodes
    """
    n = len(time_col_values)
    sorted_indices = np.argsort(time_col_values)

    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[sorted_indices[:train_end]] = True
    val_mask[sorted_indices[train_end:val_end]] = True
    test_mask[sorted_indices[val_end:]] = True

    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask

    n_train = train_mask.sum().item()
    n_val = val_mask.sum().item()
    n_test = test_mask.sum().item()
    print(f"Temporal split: train={n_train:,} | val={n_val:,} | test={n_test:,}")

    return data
