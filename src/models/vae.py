"""
Variational Autoencoder for anomaly detection on raw transaction features.
Deeper architecture for high-dimensional input (383+ features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransactionVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()

        # Deeper encoder: input -> hidden -> hidden/2 -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
        )
        self.enc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Deeper decoder: latent -> hidden/2 -> hidden -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (used as anomaly score)."""
        return F.mse_loss(x_recon, x, reduction="none").mean(dim=1)
