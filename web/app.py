"""
Flask app for Fraud Detection Demo.

Usage:
    cd web
    python app.py

Then open http://localhost:5001 in your browser.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from xgboost import XGBClassifier
from src.graph.builder import build_hetero_graph
from src.models.hybrid_model import HybridGATVAE

app = Flask(__name__)

# ── Global state ──
graph_data = None
model = None
fraud_scores = None     # hybrid model scores
xgb_scores = None       # xgboost scores
device = "cpu"


def load_model(data):
    """Load the trained HybridGATVAE model."""
    global model

    in_channels = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    raw_dim = data["transaction"].x.shape[1]

    model = HybridGATVAE(
        metadata=data.metadata(),
        in_channels=in_channels,
        raw_txn_dim=raw_dim,
        gat_hidden=128, gat_out=64,
        gat_heads=4, gat_layers=2,
        vae_latent=32, vae_hidden=128,
    )

    weights_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'models', 'hybrid_gatvae_ieee_cis.pt')
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Model loaded from {weights_path}")
        except Exception:
            print(f"WARNING: Checkpoint architecture mismatch, using random weights (re-train to fix)")
    else:
        print(f"WARNING: No weights found at {weights_path}, using random weights")

    model.eval()


def run_inference(data):
    """Run Hybrid model on all transactions."""
    global fraud_scores

    model.eval()
    with torch.no_grad():
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}
        edge_dict = {etype: data[etype].edge_index for etype in data.edge_types}
        raw_txn = data["transaction"].x
        outputs = model(x_dict, edge_dict, raw_txn_features=raw_txn)
        fraud_scores = outputs["fraud_score"].numpy()

    print(f"Hybrid inference done: {len(fraud_scores)} transactions scored")


def run_xgboost(data):
    """Train XGBoost on transaction features and score all transactions."""
    global xgb_scores

    X = data["transaction"].x.numpy()
    y = data["transaction"].y.numpy()

    # Simple train/test: use 70% for train, score everything
    n = len(y)
    train_n = int(n * 0.7)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:train_n]

    X_train, y_train = X[train_idx], y[train_idx]

    fraud_ratio = max(y_train.sum(), 1)
    scale = (len(y_train) - fraud_ratio) / fraud_ratio

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric='aucpr', tree_method='hist',
        n_jobs=-1, random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train)

    # Score ALL transactions
    xgb_scores = xgb.predict_proba(X)[:, 1]
    print(f"XGBoost done: {len(xgb_scores)} transactions scored")


def build_graph_json(data, center_indices, max_nodes=150):
    """Build a JSON subgraph around given transaction indices."""
    nodes = []
    edges = []
    seen_nodes = set()

    labels = data["transaction"].y.numpy()
    scores = fraud_scores if fraud_scores is not None else np.zeros(len(labels))

    for idx in center_indices[:max_nodes]:
        node_id = f"txn_{idx}"
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)

        is_fraud = int(labels[idx]) == 1
        score = float(scores[idx])

        nodes.append({
            "id": node_id,
            "label": f"T{idx}\n{score:.2f}",
            "group": "fraud" if is_fraud else "transaction",
            "title": f"Transaction {idx}\nHybrid: {score:.3f}\nActual: {'FRAUD' if is_fraud else 'Legit'}",
        })

    # Add connected accounts
    if ("account", "initiates", "transaction") in data.edge_types:
        acct_edges = data["account", "initiates", "transaction"].edge_index
        center_set = set(center_indices[:max_nodes].tolist() if hasattr(center_indices, 'tolist') else list(center_indices[:max_nodes]))

        for acct_idx, txn_idx in zip(acct_edges[0].tolist(), acct_edges[1].tolist()):
            if txn_idx in center_set:
                acct_id = f"acct_{acct_idx}"
                txn_id = f"txn_{txn_idx}"
                if acct_id not in seen_nodes:
                    seen_nodes.add(acct_id)
                    nodes.append({"id": acct_id, "label": f"A{acct_idx}", "group": "account", "title": f"Account {acct_idx}"})
                edges.append({"from": acct_id, "to": txn_id, "label": "initiates"})

    # Add connected merchants
    if ("transaction", "paid_to", "merchant") in data.edge_types:
        merch_edges = data["transaction", "paid_to", "merchant"].edge_index
        center_set = set(center_indices[:max_nodes].tolist() if hasattr(center_indices, 'tolist') else list(center_indices[:max_nodes]))

        for txn_idx, merch_idx in zip(merch_edges[0].tolist(), merch_edges[1].tolist()):
            if txn_idx in center_set:
                merch_id = f"merch_{merch_idx}"
                txn_id = f"txn_{txn_idx}"
                if merch_id not in seen_nodes:
                    seen_nodes.add(merch_id)
                    nodes.append({"id": merch_id, "label": f"M{merch_idx}", "group": "merchant", "title": f"Merchant {merch_idx}"})
                edges.append({"from": txn_id, "to": merch_id, "label": "paid_to"})

    return {"nodes": nodes, "edges": edges}


# ── Routes ──

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/upload", methods=["POST"])
def upload_csv():
    """Upload a CSV and build graph + run both models."""
    global graph_data

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    df = pd.read_csv(file)

    if "card1" in df.columns:
        dataset = "ieee_cis"
    elif "nameOrig" in df.columns:
        dataset = "paysim"
    else:
        return jsonify({"error": "CSV not recognized. Need IEEE-CIS or PaySim format."}), 400

    if "isFraud" not in df.columns:
        df["isFraud"] = 0

    graph_data = build_hetero_graph(df, dataset=dataset)
    load_model(graph_data)
    run_inference(graph_data)
    run_xgboost(graph_data)

    return jsonify({"status": "ok", "transactions": len(fraud_scores), "dataset": dataset})


@app.route("/load_existing", methods=["POST"])
def load_existing():
    """Load the pre-built graph from data/processed/."""
    global graph_data

    graph_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'ieee_cis', 'hetero_graph_v3.pt')
    if not os.path.exists(graph_path):
        return jsonify({"error": f"Graph not found at {graph_path}"}), 404

    try:
        graph_data = torch.load(graph_path, map_location=device, weights_only=False)
    except Exception as e:
        return jsonify({"error": f"Failed to load graph: {e}"}), 500

    load_model(graph_data)
    run_inference(graph_data)
    run_xgboost(graph_data)

    return jsonify({"status": "ok", "transactions": len(fraud_scores), "dataset": "ieee_cis (pre-built)"})


@app.route("/results")
def get_results():
    """Return fraud detection results."""
    if fraud_scores is None:
        return jsonify({"error": "No results yet. Upload data first."}), 400

    threshold = float(request.args.get("threshold", 0.5))
    labels = graph_data["transaction"].y.numpy()
    preds = (fraud_scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    top_indices = np.argsort(fraud_scores)[::-1][:100]
    top_list = [{"index": int(i), "score": round(float(fraud_scores[i]), 4),
                 "actual": "FRAUD" if labels[i] == 1 else "Legit"} for i in top_indices]

    return jsonify({
        "threshold": threshold, "total": len(fraud_scores), "flagged": int(preds.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "top_transactions": top_list,
    })


@app.route("/comparison")
def comparison():
    """Compare XGBoost vs Hybrid — who catches what."""
    if fraud_scores is None or xgb_scores is None:
        return jsonify({"error": "Models not ready yet."}), 400

    threshold = float(request.args.get("threshold", 0.5))
    labels = graph_data["transaction"].y.numpy()

    hybrid_pred = (fraud_scores >= threshold).astype(int)
    xgb_pred = (xgb_scores >= threshold).astype(int)

    total_fraud = int((labels == 1).sum())

    both_catch  = int(((hybrid_pred == 1) & (xgb_pred == 1) & (labels == 1)).sum())
    only_hybrid = int(((hybrid_pred == 1) & (xgb_pred == 0) & (labels == 1)).sum())
    only_xgb    = int(((hybrid_pred == 0) & (xgb_pred == 1) & (labels == 1)).sum())
    both_miss   = int(((hybrid_pred == 0) & (xgb_pred == 0) & (labels == 1)).sum())

    # Get indices of "only hybrid catches" for graph view
    only_hybrid_mask = (hybrid_pred == 1) & (xgb_pred == 0) & (labels == 1)
    only_hybrid_indices = np.where(only_hybrid_mask)[0]

    # Top examples from "only hybrid catches"
    if len(only_hybrid_indices) > 0:
        # Sort by hybrid score (highest first)
        sorted_idx = only_hybrid_indices[np.argsort(fraud_scores[only_hybrid_indices])[::-1]]
        examples = [{"index": int(i),
                      "hybrid_score": round(float(fraud_scores[i]), 4),
                      "xgb_score": round(float(xgb_scores[i]), 4)}
                     for i in sorted_idx[:30]]
    else:
        examples = []

    return jsonify({
        "total_fraud": total_fraud,
        "both_catch": both_catch,
        "only_hybrid": only_hybrid,
        "only_xgb": only_xgb,
        "both_miss": both_miss,
        "only_hybrid_examples": examples,
    })


@app.route("/graph")
def get_graph():
    """Return subgraph JSON for visualization."""
    if graph_data is None:
        return jsonify({"error": "No graph loaded"}), 400

    mode = request.args.get("mode", "top_fraud")
    threshold = float(request.args.get("threshold", 0.5))
    labels = graph_data["transaction"].y.numpy()

    if mode == "top_fraud":
        indices = np.argsort(fraud_scores)[::-1][:50]
    elif mode == "flagged":
        indices = np.where(fraud_scores >= threshold)[0][:80]
    elif mode == "missed":
        indices = np.where((labels == 1) & (fraud_scores < threshold))[0][:50]
    elif mode == "only_hybrid":
        # Fraud that only Hybrid catches (XGBoost misses)
        hybrid_pred = (fraud_scores >= threshold).astype(int)
        xgb_pred = (xgb_scores >= threshold).astype(int)
        mask = (hybrid_pred == 1) & (xgb_pred == 0) & (labels == 1)
        indices = np.where(mask)[0][:60]
    else:
        indices = np.arange(min(50, len(fraud_scores)))

    graph_json = build_graph_json(graph_data, indices)
    return jsonify(graph_json)


if __name__ == "__main__":
    print("Starting Fraud Detection Demo...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=False, port=5001)
