"""
SHAP-based explainability for flagged transactions.
Uses KernelExplainer as a model-agnostic wrapper.
For GNN-specific explanations, see gnn_explainer.py.
"""
import numpy as np
import shap
import torch


class FraudSHAPExplainer:
    def __init__(self, model: torch.nn.Module, background_embeddings: np.ndarray):
        """
        Parameters
        ----------
        model                 : trained HybridGATVAE (or any callable)
        background_embeddings : sample of transaction embeddings h_t for SHAP background
        """
        self.background = background_embeddings

        def predict_fn(h_t_np: np.ndarray) -> np.ndarray:
            h_t = torch.tensor(h_t_np, dtype=torch.float32)
            with torch.no_grad():
                # Direct classifier path (embeddings already computed)
                recon_err = torch.zeros(len(h_t), 1)
                combined = torch.cat([h_t, recon_err], dim=1)
                logits = model.classifier(combined).squeeze(1)
                return torch.sigmoid(logits).numpy()

        self.explainer = shap.KernelExplainer(predict_fn, background_embeddings)

    def explain(self, embeddings: np.ndarray, nsamples: int = 200) -> shap.Explanation:
        return self.explainer.shap_values(embeddings, nsamples=nsamples)

    def plot_waterfall(self, idx: int, shap_values: np.ndarray, feature_names: list) -> None:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=self.explainer.expected_value,
                feature_names=feature_names,
            )
        )
