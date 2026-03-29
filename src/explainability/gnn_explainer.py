"""
GNNExplainer wrapper for subgraph-level explanations.
Identifies which neighbor nodes and edges contributed most to a fraud prediction.
"""
import torch
from torch_geometric.explain import Explainer, GNNExplainer


def build_gnn_explainer(model: torch.nn.Module) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        ),
    )


def explain_transaction(
    explainer: Explainer,
    data,
    transaction_idx: int,
) -> dict:
    """
    Returns node and edge importance masks for a single flagged transaction.
    """
    explanation = explainer(
        x=data.x_dict,
        edge_index=data.edge_index_dict,
        index=transaction_idx,
    )
    return {
        "node_mask": explanation.node_mask,
        "edge_mask": explanation.edge_mask,
    }
