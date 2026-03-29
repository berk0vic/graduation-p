"""
Elliptic Bitcoin Dataset loader.
Kaggle: https://www.kaggle.com/ellipticco/elliptic-data-set
Files expected in data/raw/elliptic/:
    elliptic_txs_features.csv
    elliptic_txs_edgelist.csv
    elliptic_txs_classes.csv
Labels: 1 = illicit (fraud), 2 = licit, unknown = unlabeled
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data/raw/elliptic/elliptic_bitcoin_dataset"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)
    edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
    return features, edges, classes


def build_pyg_graph() -> Data:
    """Convert Elliptic dataset to a PyG homogeneous graph."""
    features, edges, classes = load_raw()

    # Node features: columns 1-165 are features, column 0 is node id, column 1 is time step
    node_ids = features.iloc[:, 0].values
    node_id_map = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)

    # Labels: 1=illicit→1, 2=licit→0, unknown→-1
    classes = classes.set_index("txId")
    labels = []
    for nid in node_ids:
        c = classes.loc[nid, "class"] if nid in classes.index else "unknown"
        if c == "1":
            labels.append(1)
        elif c == "2":
            labels.append(0)
        else:
            labels.append(-1)
    y = torch.tensor(labels, dtype=torch.long)

    src = [node_id_map[n] for n in edges.iloc[:, 0].values if n in node_id_map]
    dst = [node_id_map[n] for n in edges.iloc[:, 1].values if n in node_id_map]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
