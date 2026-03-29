# Research Plan: Hybrid Graph Neural Networks for Financial Fraud Detection
**CSE492 — Berk Tahir Kılıç**

---

## 1. The Problem (Why This Matters)

Traditional fraud detection treats every transaction as an independent row in a table.
This fails for organized fraud rings, where the fraud signal is not in a single transaction
but in the *relationships* between accounts, merchants, and transactions across time.

Example: Account A sends money to Account B which pays Merchant C — none of these
individually look suspicious, but the 3-hop pattern matches 50 other fraud cases.
Tabular models (XGBoost, Random Forest) are blind to this.

---

## 2. Research Questions

### Primary Question
> Can a hybrid GAT+VAE model operating on a heterogeneous transaction graph
> outperform both standalone GNN and tabular baselines for financial fraud detection,
> particularly on imbalanced datasets?

### Sub-Questions
1. Does modeling transactions as a heterogeneous graph (Account, Transaction, Merchant)
   capture fraud patterns that tabular models miss?

2. Does adding a VAE (unsupervised) component on top of a supervised GAT improve
   detection of *novel, unseen* fraud patterns that weren't in training labels?

3. Can the model provide human-interpretable explanations (via SHAP) that are
   actionable for banking analysts — not just a score, but a reason?

4. Does the model scale to 6M+ transactions (PaySim) while maintaining sub-second
   inference latency suitable for real-time banking alerts?

---

## 3. Hypothesis

A hybrid GAT+VAE model will:
- Achieve higher F1-Score and Recall on minority fraud class than all baselines
- Detect novel fraud patterns (zero-shot anomaly detection) via VAE reconstruction error
- Provide SHAP-based explanations that identify top contributing graph features

---

## 4. Related Work (Your 11 Papers — What Each Contributes)

| Paper | What you take from it |
|---|---|
| Detecting Imbalanced Fraud via Hybrid GAT and VAE | Core architecture reference — your model extends this |
| CaT-GNN: Causal Temporal GNNs | Temporal edge features idea (time-ordered transactions) |
| Detecting Fraud via Heterogeneous Graph Neural Networks | Heterogeneous graph construction (Account/Merchant/Transaction) |
| Explainable AI (XAI) for Fraud | SHAP integration methodology |
| Integrated Temporal GNN and Autoencoder | VAE+GNN combination patterns |
| FinGraphFL: Financial Graph Federated Learning | Discussion point: privacy-preserving future work |
| Enhanced Fraud Detection Using LSTM and SMOTE | Imbalance handling comparison (SMOTE vs focal loss) |
| Deep Forest (gcForest) for Imbalanced Fraud | Baseline comparison candidate |
| Credit Card Fraud Detection Using TabNet | Tabular baseline for comparison |
| Combining Unsupervised and Supervised Learning in CCF | Motivation for hybrid supervised+unsupervised |
| Deep Learning via Continuous-Coupled Neural Nets | Background on deep learning for fraud |

---

## 5. Datasets

### Primary Dataset
- **PaySim** (Kaggle): 6.3M synthetic mobile money transactions, labeled fraud/non-fraud
  - Use for: scalability testing + main experiments
  - Imbalance ratio: ~0.1% fraud (highly imbalanced — realistic)

### Secondary Dataset
- **Elliptic Bitcoin Dataset**: 200K transactions as a native graph with time steps
  - Use for: graph-native fraud, temporal GNN comparison
  - Already structured as a graph — easier to build on

### Baseline Comparison Dataset
- **IEEE-CIS Fraud Detection** (Kaggle): real credit card transactions
  - Use for: tabular baseline comparison (XGBoost vs your model)

---

## 6. Methodology — Step by Step

### Step 1: Graph Construction
Convert tabular transaction data into a PyG HeteroData graph.

**Node types:**
- `Account`: source/destination accounts
- `Transaction`: each transaction event
- `Merchant`: merchant/recipient entities

**Edge types:**
- `Account -[initiates]-> Transaction`
- `Transaction -[paid_to]-> Merchant`
- `Account -[shares_device_with]-> Account` (same IP/device = suspicious link)
- `Transaction -[followed_by]-> Transaction` (temporal chain per account)

**Node features:**
- Account: transaction count, avg amount, fraud history ratio, days active
- Transaction: amount, hour of day, day of week, velocity (txns in last 1h/24h)
- Merchant: category, avg transaction size, fraud rate from training set

### Step 2: Model Architecture

```
Input: HeteroData Graph
       |
       v
[GAT Encoder]
- HeteroConv with GATConv per edge type
- 2-3 attention layers
- Outputs: node embeddings h_i for each node
       |
       v
[Transaction Node Embeddings h_t]
       |
      / \
     /   \
    v     v
[VAE]   [Classifier Head]
Encoder  Takes h_t directly
z ~ N(mu, sigma)
Decoder
Reconstruction error r_t
     \   /
      \ /
       v
[Combined Score]
fraud_score = sigmoid(W * [h_t || r_t])
```

**Loss function:**
```
L_total = L_classification (Focal Loss)
        + lambda_1 * L_reconstruction (MSE)
        + lambda_2 * L_KL (KL divergence)
```

Why Focal Loss: down-weights easy negatives (legitimate transactions),
forces model to focus on hard-to-detect minority fraud cases.

### Step 3: Handling Class Imbalance
- **Focal Loss** (primary): gamma=2, alpha tuned per dataset
- **Node-level class weights**: weight fraud nodes higher in loss
- **Threshold tuning**: optimize decision threshold for F1, not 0.5 default

### Step 4: Explainability
After getting fraud_score for a transaction:
- Use **SHAP TreeExplainer** or **GNNExplainer** to identify:
  - Which graph neighbors contributed most to the score
  - Which node features pushed the score up
- Output to analyst: "Flagged because: amount 5x account average,
  shared merchant with 3 known fraud accounts, 2nd transaction in 10 minutes"

---

## 7. Baselines (What You Compare Against)

| Model | Type | Purpose |
|---|---|---|
| XGBoost | Tabular, supervised | No-graph baseline |
| Random Forest | Tabular, supervised | Classic fraud detection |
| GCN | Homogeneous graph | Graph without attention |
| GAT-only | Heterogeneous graph | Graph with attention, no VAE |
| VAE-only | Unsupervised | Anomaly detection, no graph |
| **GAT+VAE (yours)** | Hybrid | Proposed model |

This lets you show clearly what each component adds.

---

## 8. Evaluation Metrics

| Metric | Why |
|---|---|
| **F1-Score (minority class)** | Primary metric — balances precision & recall for fraud |
| **Recall (fraud catch rate)** | Missing fraud = costly; maximize this |
| **Precision** | Avoid too many false alerts (analyst fatigue) |
| **AUC-ROC** | Overall discrimination ability |
| **Inference latency (ms)** | Sub-second target for real-time alerts |
| **Training time** | Scalability on PaySim 6M+ |

---

## 9. Expected Contributions

1. A working hybrid GAT+VAE implementation on a heterogeneous financial graph
2. Empirical comparison showing each component's contribution (ablation study)
3. SHAP-based explanation pipeline for fraud alerts
4. Scalability analysis on PaySim (6M+ transactions)

---

## 10. Timeline (16 Weeks)

| Weeks | Milestone |
|---|---|
| 1-2 | Literature review, finalize related work section, set up environment |
| 3-4 | Download datasets, EDA (distributions, imbalance ratio, graph stats) |
| 5-6 | Graph construction (PyG HeteroData), verify graph structure |
| 7 | Baseline models (XGBoost, GCN) — establish benchmark numbers |
| 8-10 | Implement GAT encoder + VAE + combined model |
| 11 | Training runs, hyperparameter tuning (focal loss weights, GAT heads) |
| 12 | Ablation study (GAT-only vs VAE-only vs hybrid) |
| 13 | SHAP explainability pipeline |
| 14 | Latency benchmarking, scalability tests on PaySim |
| 15 | Results analysis, tables, figures |
| 16 | Final report writing + presentation prep |

---

## 11. Tech Stack Summary

| Component | Tool |
|---|---|
| Graph construction | PyTorch Geometric (PyG) `HeteroData` |
| GAT layers | `HeteroConv` + `GATConv` from PyG |
| VAE | Custom PyTorch `nn.Module` |
| Baselines | Scikit-learn (XGBoost, RF), PyG (GCN) |
| Imbalance | `focal_loss` or manual weighted BCE |
| Explainability | SHAP + `torch_geometric.explain.GNNExplainer` |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, NetworkX (for graph vis) |
| Environment | Python 3.10, venv or conda |
| Hardware | MacBook M1 Pro (MPS backend for PyTorch) |
