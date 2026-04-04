# Theoretical Background and Methods

**Hybrid Graph Neural Networks for Financial Fraud Detection**
**Berk Tahir Kilic — CSE492 Graduation Project**

---

## Table of Contents

1. [Graph Neural Networks (GNN)](#1-graph-neural-networks-gnn)
2. [Variational Autoencoder (VAE)](#2-variational-autoencoder-vae)
3. [Hybrid Model Architecture (HybridGATVAE)](#3-hybrid-model-architecture-hybridgatvae)
4. [Loss Functions](#4-loss-functions)
5. [Traditional Machine Learning Models](#5-traditional-machine-learning-models)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Data Preprocessing Techniques](#7-data-preprocessing-techniques)
8. [Training Techniques](#8-training-techniques)
9. [Activation Functions](#9-activation-functions)
10. [Explainability Methods](#10-explainability-methods)
11. [Graph Construction Techniques](#11-graph-construction-techniques)
12. [Datasets](#12-datasets)
13. [Class Imbalance Handling](#13-class-imbalance-handling)
14. [References](#14-references)

---

## 1. Graph Neural Networks (GNN)

### 1.1 Introduction to Graph Neural Networks

Graph Neural Networks (GNNs) are a class of deep learning models designed to operate on graph-structured data. While traditional neural networks (CNNs, RNNs, MLPs) only process regular grids or sequential structures, GNNs can handle relational data composed of nodes and edges.

The fundamental operating principle is based on the **message passing paradigm**: each node collects information from its neighbors (aggregate), combines it with its own representation (combine), and produces an updated embedding. This process is repeated across multiple layers, allowing each node to gather information from an increasingly wider neighborhood.

**Mathematical formulation:**

```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

Here, `h_v^(l)` is the representation of node `v` at layer `l`, and `N(v)` is the set of neighbors of node `v`.

**Why GNNs?** Financial transaction networks are inherently graph-structured: accounts (nodes), transactions (nodes), merchants (nodes), and relationships between them (edges). Fraud often involves abnormal patterns within a network — one must look at inter-transaction relationships, not just individual transactions.

---

### 1.2 Graph Convolutional Networks (GCN)

**Reference:** Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017

Graph Convolutional Network (GCN) is one of the first effective methods to apply convolution operations on graph-structured data. It extends the success of CNNs in image processing to the graph domain.

**How it works:**

GCN draws inspiration from spectral graph theory. Instead of computing the full Fourier transform of signals on a graph, it uses a first-order approximation. At each layer, every node takes a weighted average of its neighbors' features and transforms them through a learnable weight matrix.

**Formula:**

```
H^(l+1) = σ(D̃^(-1/2) · Ã · D̃^(-1/2) · H^(l) · W^(l))
```

Where:
- `Ã = A + I_N` — adjacency matrix with added self-loops
- `D̃` — degree matrix of `Ã`, where `D̃_ii = Σ_j Ã_ij`
- `H^(l)` — node representation matrix at layer `l` (N × d)
- `W^(l)` — learnable weight matrix at layer `l` (d × d')
- `σ` — nonlinear activation function (e.g., ReLU)

The term `D̃^(-1/2) · Ã · D̃^(-1/2)` is called **symmetric normalization** and scales by neighbor count — this prevents high-degree nodes from having disproportionate influence.

**Advantages:**
- Simple and efficient
- Effective for semi-supervised learning
- Strong baseline for node classification

**Limitations:**
- Assigns equal weight to all neighbors (no attention mechanism)
- Designed only for homogeneous graphs

**Usage in project:** GCN is used as a baseline model to measure how much advantage GAT's attention-based approach provides over GCN.

---

### 1.3 Graph Attention Networks (GAT)

**Reference:** Veličković et al., "Graph Attention Networks", ICLR 2018

Graph Attention Network (GAT) overcomes GCN's limitation of assigning equal weights to all neighbors by introducing an **attention mechanism** that dynamically learns the contribution of each neighbor.

**How it works:**

An attention coefficient is computed for each node pair (i, j). This coefficient determines how much node j contributes to the update of node i. The formula consists of three stages:

**Stage 1 — Linear transformation:**

Each node's features are transformed with a learnable weight matrix `W`:

```
z_i = W · h_i
```

**Stage 2 — Attention coefficients:**

```
e_ij = LeakyReLU(a^T · [z_i ∥ z_j])
```

Where `∥` denotes concatenation and `a` is a learnable attention vector.

**Stage 3 — Softmax normalization:**

```
α_ij = exp(e_ij) / Σ_{k ∈ N(i)} exp(e_ik)
```

**Stage 4 — Weighted aggregation:**

```
h_i' = σ(Σ_{j ∈ N(i)} α_ij · z_j)
```

**Multi-head attention:**

Instead of a single attention function, `K` different attention heads run in parallel and their results are concatenated:

```
h_i' = ∥_{k=1}^{K} σ(Σ_{j ∈ N(i)} α_ij^k · W^k · h_j)
```

This approach enables the model to capture different relational patterns simultaneously.

**Advantages:**
- Different importance can be assigned to different neighbors
- Structures the graph through the learning process
- More flexible and powerful than GCN

**Usage in project:** GAT is used in the graph branch of HybridGATVAE as `HeteroGATEncoder`. It is configured with 4 attention heads and 2 GAT layers. Hidden dimension: 128, output dimension: 64.

---

### 1.4 Heterogeneous Graph Neural Networks

Most real-world graphs are **heterogeneous** — they contain multiple node types and edge types. For example, in a financial network, accounts, transactions, and merchants are different types of nodes; "initiates", "paid_to", "followed_by" are different edge types.

The **HeteroConv** approach applies a separate convolution layer for each edge type and combines the results to process heterogeneous information:

```
h_v^(l+1) = AGG_{r ∈ R}(CONV_r({h_u^(l) : u ∈ N_r(v)}))
```

Where `R` is the set of edge types and `N_r(v)` is the set of neighbors of node `v` connected via edge type `r`.

**Node types defined in the project:**

| Node Type | Description | Feature Dimension |
|---|---|---|
| `transaction` | Financial transactions | ~383 (IEEE-CIS) |
| `account` | Accounts (card-based) | 5 (count, mean, std, min, max) |
| `merchant` | Merchants | 3 (count, mean, std) |

**Edge types defined in the project:**

| Edge Type | Direction | Description |
|---|---|---|
| `initiates` | account → transaction | Account initiating a transaction |
| `paid_to` | transaction → merchant | Transaction made to a merchant |
| `followed_by` | transaction → transaction | Sequential transactions on the same card (within 1 hour) |
| `shares_identity` | transaction ↔ transaction | Shared device/browser information |

**Why heterogeneous graphs?** Homogeneous graphs treat all nodes as the same type and lose relational richness. Financial networks inherently contain multi-typed entities — the heterogeneous approach preserves this structure.

---

## 2. Variational Autoencoder (VAE)

### 2.1 Autoencoders

An autoencoder is an unsupervised neural network model that learns to reconstruct its input data as output. It consists of two main components:

- **Encoder:** Compresses the input into a lower-dimensional **latent representation**
- **Decoder:** Reconstructs the original input from the latent representation

```
Input x → [Encoder] → Latent representation z → [Decoder] → Reconstructed x̂
```

**Bottleneck architecture:** The latent layer dimension must be smaller than the input dimension — this forces the model to learn the most important features of the data.

Autoencoders are used in anomaly detection: the model is trained on normal data, and abnormal data (fraud) produces high reconstruction error.

---

### 2.2 Variational Autoencoder (VAE)

**Reference:** Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014

The Variational Autoencoder (VAE) is a **probabilistic** generalization of the classical autoencoder. Instead of a fixed vector as a latent representation, it learns the **parameters of a probability distribution** (mean μ and variance σ²).

**How it works:**

1. The **encoder** takes input `x` and produces the parameters of the latent distribution:
   - `μ = f_μ(x)` — mean vector
   - `log_var = f_logvar(x)` — log-variance vector

2. **Reparameterization trick:**
   Sampling directly from the distribution would block gradient computation. Instead:
   ```
   z = μ + exp(0.5 · log_var) · ε,    ε ~ N(0, I)
   ```
   This transforms the stochastic operation into a deterministic function, enabling backpropagation.

3. The **decoder** takes `z` and reconstructs `x̂`.

**ELBO (Evidence Lower Bound) loss function:**

```
L_VAE = E_{q(z|x)}[log p(x|z)] − KL(q(z|x) ∥ p(z))
```

- First term: **Reconstruction loss** — how well the decoder reconstructs the input
- Second term: **KL Divergence** — how close the latent distribution is to the target distribution (standard normal)

**Usage in project:** The `TransactionVAE` module learns the distribution of normal financial transactions. Since fraudulent transactions deviate from normal patterns, they produce high reconstruction error — this error forms the anomaly branch of the hybrid model.

**Architecture:**
```
Encoder: 383 → 128 → 64 → 32 (latent dim)
Decoder: 32 → 64 → 128 → 383
Activation: ELU
Dropout: 0.2
```

---

### 2.3 KL Divergence (Kullback-Leibler Divergence)

**Reference:** Kullback & Leibler, "On Information and Sufficiency", 1951

KL Divergence is a fundamental measure from information theory that quantifies the "distance" (asymmetric) between two probability distributions. It measures how much one distribution differs from another.

**General formula:**

```
KL(Q ∥ P) = Σ_x Q(x) · log(Q(x) / P(x))
```

**Closed-form for VAE (two Gaussians):**

Since in VAE `q(z|x) = N(μ, σ²)` and `p(z) = N(0, I)`:

```
KL = −0.5 · Σ(1 + log(σ²) − μ² − σ²)
```

**Usage in project:** KL divergence ensures the VAE's latent space stays close to a standard normal distribution. It is added to the total loss with weight coefficient `λ₂ = 0.01`.

---

## 3. Hybrid Model Architecture (HybridGATVAE)

### 3.1 Dual-Branch Fusion Architecture

HybridGATVAE is the **dual-branch fusion architecture** that constitutes the main contribution of this project. It detects fraud by combining two different information sources:

**Branch A — Structural/Relational Branch (Graph Branch):**
```
HeteroData → HeteroGATEncoder (2 layers, 4 heads) → h_t ∈ ℝ^64
```
Captures the transaction's position in the graph structure, neighborhood relationships, and structural patterns.

**Branch B — Anomaly Branch (VAE Branch):**
```
Raw features x → TransactionVAE → reconstruction error (scalar)
```
Measures the degree to which a transaction deviates from normal transaction patterns.

**Fusion and Classification:**
```
[h_t (64-dim) ∥ BatchNorm(recon_err) (1-dim)] → MLP(65 → 64 → 32 → 1) → sigmoid → fraud score
```

The strengths of this dual approach:
- Models using only graph structure may miss individual transaction anomalies
- Models using only VAE cannot capture relational fraud patterns between transactions
- The hybrid model combines both information sources for more comprehensive detection

---

### 3.2 Two-Phase Training Strategy

Model training is carried out in two stages:

**Phase 1 — VAE Pre-training (50 epochs):**
- No gradients flow to the GAT encoder and classifier layers (frozen)
- Only the VAE learns normal transaction patterns
- Goal: VAE achieves good reconstruction capability

**Phase 2 — End-to-End Fine-tuning (50 epochs):**
- All components are unfrozen
- **Differential learning rates:**
  - GAT: `lr = 0.001`
  - VAE: `lr × 0.1 = 0.0001` (to preserve Phase 1 learning)
  - Classifier: `lr × 2 = 0.002` (for faster adaptation)

This two-phase strategy prevents the **catastrophic forgetting** problem — i.e., it prevents the normal transaction patterns learned by the VAE during pre-training from being destroyed during end-to-end training.

---

## 4. Loss Functions

### 4.1 Binary Cross-Entropy (BCE)

The standard loss function for binary classification problems. It measures the difference between the model's predicted probability `p` and the true label `y`.

**Formula:**

```
BCE(y, p) = −[y · log(p) + (1 − y) · log(1 − p)]
```

- For `y = 1` (fraud): `−log(p)` → model should give high `p`
- For `y = 0` (normal): `−log(1−p)` → model should give low `p`

BCE serves as the foundation for Focal Loss.

---

### 4.2 Focal Loss

**Reference:** Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), ICCV 2017

Focal Loss is a loss function designed for problems with extreme class imbalance. Although originally proposed for object detection, it is highly effective in rare event classification tasks such as fraud detection.

**Problem:** With standard cross-entropy, the model learns easy examples (99% normal transactions) very quickly while failing to focus sufficiently on hard examples (fraud).

**Solution:** Focal Loss reduces (down-weights) the loss for easy examples, forcing the model to focus on hard examples.

**Formula:**

```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
```

Where:
- `p_t` = probability the model predicts for the correct class
- `γ` = **focusing parameter**. When `γ = 0`, it equals standard CE. As `γ` increases, the loss for easy examples is suppressed more.
- `α_t` = class weight parameter. Assigns higher weight to the minority class.

**Project values:** `γ = 2.0`, `α = 0.75`

**Example impact:** If the model predicts 95% correctly for a normal transaction:
- CE loss: `−log(0.95) = 0.05`
- Focal Loss: `−(1−0.95)² · log(0.95) = 0.05 × 0.0025 = 0.000125` (reduced by 200×!)

This way the model focuses on hard examples (fraud) that it predicts incorrectly, rather than easy examples it already handles well.

---

### 4.3 MSE Reconstruction Loss

Mean Squared Error (MSE) measures how well the VAE can reconstruct the input.

**Formula:**

```
L_recon = (1/n) · Σ_{i=1}^{n} (x_i − x̂_i)²
```

- `x_i` — original input feature
- `x̂_i` — feature reconstructed by the VAE
- `n` — number of features

Normal transactions produce low MSE (VAE has learned them well), fraudulent transactions produce high MSE (deviation from normal patterns).

**Project weight:** `λ₁ = 0.1 – 0.3`

---

### 4.4 Total Loss Function

The model's total loss consists of three components:

```
L_total = L_focal + λ₁ · L_reconstruction + λ₂ · L_KL
```

| Component | Weight | Purpose |
|---|---|---|
| `L_focal` | 1.0 | Fraud classification performance |
| `L_reconstruction` | λ₁ = 0.1–0.3 | VAE reconstruction quality |
| `L_KL` | λ₂ = 0.01 | Latent space regularization |

These weights balance ensuring dominant classification performance, preserving VAE reconstruction capacity, and preventing excessive latent space dispersion.

---

## 5. Traditional Machine Learning Models

### 5.1 XGBoost (Extreme Gradient Boosting)

**Reference:** Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016

XGBoost is a highly optimized implementation of the gradient boosting framework. It consists of an ensemble of decision trees trained sequentially, where each new tree attempts to correct the errors of the previous trees.

**How it works:**

1. The initial model starts with a simple prediction (e.g., log of class proportions)
2. At each step, the residual error of the current model is computed
3. A new decision tree is trained to predict this residual error
4. The new tree is scaled by a learning rate and added to the ensemble
5. The process repeats until a specified number of trees are trained or early stopping criteria are met

**Objective function:**

```
L = Σ_{i=1}^{n} l(y_i, ŷ_i) + Σ_{k=1}^{K} Ω(f_k)
```

Where the regularization term is:

```
Ω(f) = γ · T + (1/2) · λ · ∥w∥²
```

- `l(y_i, ŷ_i)` — prediction loss (e.g., log loss)
- `T` — number of leaves in the tree
- `w` — leaf weights
- `γ` — leaf complexity penalty
- `λ` — L2 regularization coefficient

**Usage in project:**
- 500 trees, max depth 6, learning rate 0.05
- `scale_pos_weight = n_negative / n_positive` for class imbalance handling
- Histogram-based tree method (`tree_method: hist`) for fast training
- Used as the tabular-based baseline model

---

### 5.2 Random Forest

**Reference:** Breiman, "Random Forests", Machine Learning, 2001

Random Forest is an ensemble learning method where independently trained decision trees produce results through majority voting. It uses **Bagging** (Bootstrap Aggregating) techniques.

**How it works:**

1. `B` subsets are created from the original training data via **bootstrap sampling** (sampling with replacement)
2. A decision tree is trained on each subset
3. At each split point, each tree selects the best split from a **random feature subset** (`√p` features, where `p` is the total number of features)
4. For classification, the final prediction is the **majority vote** of all trees

**Prediction:**

```
ŷ = mode{h_b(x) : b = 1, ..., B}
```

Where `h_b` is the b-th decision tree.

**Why Random Forest?**
- **Variance reduction:** Bagging reduces the overfitting tendency of individual trees
- **Feature randomness:** Reduces correlation between trees
- **Robust:** Resilient to outliers and missing data

**Usage in project:**
- 300 trees, unlimited depth
- `class_weight: balanced` for class imbalance handling
- Parallel processing (`n_jobs=-1`)

---

## 6. Evaluation Metrics

### 6.1 Confusion Matrix

The fundamental visualization tool for classification performance. In binary classification, it is divided into four categories:

```
                      Predicted
                  Positive    Negative
Actual Positive |   TP     |    FN    |
Actual Negative |   FP     |    TN    |
```

- **TP (True Positive):** Fraud correctly detected
- **FP (False Positive):** Normal transaction falsely flagged as fraud
- **FN (False Negative):** Fraud missed (most dangerous!)
- **TN (True Negative):** Normal transaction correctly approved

---

### 6.2 Precision

**"Of all transactions flagged as fraud, how many are actually fraud?"**

```
Precision = TP / (TP + FP)
```

High precision means low false alarm rate. Important for banks — every alarm requires manual review.

---

### 6.3 Recall (Sensitivity)

**"Of all actual fraud cases, how many did we catch?"**

```
Recall = TP / (TP + FN)
```

High recall means few missed fraud cases. In fraud detection, recall is generally more critical than precision — the cost of missing a fraud is much higher than the cost of a false alarm.

---

### 6.4 F1 Score

The **harmonic mean** of Precision and Recall. Used when both metrics need to be balanced.

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

**Why harmonic mean?** The harmonic mean is used instead of the arithmetic mean because if one metric is very low (e.g., Recall = 0.01), F1 will also be low — it is not possible to inflate F1 by having only one metric be good.

**Usage in project:** F1 score is computed for the fraud class (`pos_label=1`) and is used as the primary metric for optimal threshold search.

---

### 6.5 AUC-ROC (Receiver Operating Characteristic)

**ROC Curve:** A curve showing the relationship between **True Positive Rate (TPR)** and **False Positive Rate (FPR)** at different threshold values.

```
TPR = TP / (TP + FN)     (= Recall)
FPR = FP / (FP + TN)
```

**AUC (Area Under the Curve):** The area under the ROC curve.

- `AUC = 1.0` → Perfect classifier
- `AUC = 0.5` → Random prediction (worthless)
- `AUC < 0.5` → Worse than random (classes should be inverted)

**Advantage:** Threshold-independent — measures the model's overall discrimination capability.

---

### 6.6 Average Precision (AP) / PR-AUC

The area under the Precision-Recall curve.

```
AP = Σ_n (R_n − R_{n-1}) · P_n
```

**Why AP?** In imbalanced datasets, AUC-ROC can be misleading because when TN count is very high, FPR always remains low. AP directly focuses on the positive class performance and is more informative for imbalanced data.

**Usage in project:** Monitored with `eval_metric: aucpr` for XGBoost.

---

### 6.7 Optimal Threshold Search

Classification models produce a probability score (0–1). A threshold is needed to convert this score into a binary label (0 or 1). The default threshold of 0.5 is not optimal for imbalanced data.

**Method:**
- 200 threshold values between 0.01 and 0.99 are tested
- F1 score is computed for each threshold
- The threshold yielding the highest F1 is selected as the optimal threshold

```
optimal_threshold = argmax_{t ∈ [0.01, 0.99]} F1(y_true, ŷ > t)
```

---

## 7. Data Preprocessing Techniques

### 7.1 IQR Normalization

Interquartile Range (IQR) based normalization is a scaling method that is robust to outliers.

**Formula:**

```
x_norm = (x − median) / IQR

IQR = Q3 − Q1
```

Where:
- `Q1` — 25th percentile (1st quartile)
- `Q3` — 75th percentile (3rd quartile)
- `median` — 50th percentile
- The result is clipped to `[−5, 5]`

**Why IQR?** Standard z-score normalization `(x − mean) / std` is sensitive to outliers because the mean and standard deviation are affected by them. IQR-based normalization uses median and quartiles, reducing the impact of outliers.

**Usage in project:** Transaction, account, and merchant features are each separately scaled with IQR normalization.

---

### 7.2 One-Hot Encoding

A preprocessing technique that converts categorical variables into binary vectors.

**Example:**

```
ProductCD = {W, H, C, S, R}

W → [1, 0, 0, 0, 0]
H → [0, 1, 0, 0, 0]
C → [0, 0, 1, 0, 0]
S → [0, 0, 0, 1, 0]
R → [0, 0, 0, 0, 1]
```

**Why?** Assigning numerical values to categorical variables (e.g., W=1, H=2, C=3) causes the model to assume a false ordinal relationship between categories.

**Columns applied in project:** `ProductCD`, `card4`, `card6`

---

### 7.3 Missing Value Handling

Real-world data frequently contains missing values. A graduated strategy is applied in the project:

| Missing Rate | Strategy |
|---|---|
| < 30% | Keep, impute with median |
| 30%–80% | Median imputation |
| > 80% | Drop feature |
| Categorical | Mode (most frequent value) imputation |

**Median imputation** is more robust to outliers compared to mean imputation.

---

### 7.4 Feature Engineering

The process of deriving new, informative features from raw data.

**Velocity features:**
- `txn_cumcount` — Cumulative transaction count on the same card
- `time_since_last` — Time elapsed since the last transaction on the same card (seconds)

**Rolling statistics:**
- `amt_rolling_mean` — Mean amount up to the current transaction on the same card
- `amt_rolling_std` — Standard deviation of amounts up to the current transaction on the same card
- `amount_zscore` — Z-score of the transaction amount for that card

**Cyclical time encoding:**

Encodes hour information as continuous cyclical features (23:59 and 00:00 should be close):

```
hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
```

---

### 7.5 Temporal Train/Val/Test Split

Random splitting is inappropriate for time series data — **temporal splitting** is used to prevent future information from leaking into the training set (data leakage).

```
|-------- Train (70%) --------|-- Validation (15%) --|-- Test (15%) --|
Oldest _____________________________________________________ Newest
```

This approach correctly simulates the scenario of learning from past data to predict future transactions.

---

## 8. Training Techniques

### 8.1 Adam Optimizer

**Reference:** Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015

Adam (Adaptive Moment Estimation) is the most widely used optimization algorithm in deep learning. It combines the advantages of Momentum and RMSProp.

**How it works:**

It computes an adaptive learning rate for each parameter:

1. First moment estimate (momentum):
   ```
   m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
   ```

2. Second moment estimate (RMSProp):
   ```
   v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
   ```

3. Bias correction:
   ```
   m̂_t = m_t / (1 − β₁^t)
   v̂_t = v_t / (1 − β₂^t)
   ```

4. Parameter update:
   ```
   θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)
   ```

**Project values:** `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `lr = 0.001`

---

### 8.2 Learning Rate Scheduling

**ReduceLROnPlateau:** Automatically reduces the learning rate when it detects that validation loss is no longer improving.

```
If val_loss does not improve for 7 epochs:
    lr_new = lr_old × 0.5
```

This approach enables fast learning with large steps at the beginning, and fine-tuning with smaller steps toward the end.

---

### 8.3 Early Stopping

Monitors validation set performance and terminates training early when the model begins overfitting.

```
If val_loss does not improve for patience=15 epochs:
    Stop training
    Load best model weights
```

**Why?** Deep learning models begin memorizing training data (overfitting) when trained long enough. Early stopping captures the point where the model's generalization performance is best, thus preventing overfitting.

---

### 8.4 Gradient Clipping

In deep networks, gradients can sometimes reach very large values (exploding gradients). Gradient clipping rescales the gradient when its norm exceeds a threshold.

**Formula:**

```
If ∥g∥ > max_norm:
    g = g · max_norm / ∥g∥
```

**Project value:** `max_norm = 1.0`

---

### 8.5 Dropout

**Reference:** Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014

Dropout is a regularization technique that randomly deactivates neurons with probability `p` during training.

**How it works:**

During training:
```
h_dropped = h · mask,    mask ~ Bernoulli(1−p)
```

During testing:
```
h_test = h · (1 − p)     (or scaling by 1/(1−p) during training)
```

**Why?** It prevents neurons from becoming overly dependent on each other (co-adaptation). Each neuron is forced to learn more robust features by assuming other neurons "might not be there."

**Project values:** GAT: `p=0.3`, VAE: `p=0.2`, Classifier: `p=0.3`

---

### 8.6 Batch Normalization

**Reference:** Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training", ICML 2015

Batch normalization accelerates training and improves stability by normalizing activations within each mini-batch.

**Formula:**

```
x̂ = (x − μ_B) / √(σ_B² + ε)
y = γ · x̂ + β
```

- `μ_B` and `σ_B²` — mini-batch mean and variance
- `γ` and `β` — learnable scale and shift parameters

**Usage in project:** `BatchNorm1d` is used for normalizing the VAE reconstruction error.

---

### 8.7 Weight Decay (L2 Regularization)

Penalizes large weight values to limit model complexity.

**Formula:**

```
L_reg = L + λ · ∥θ∥²
```

Where `λ = 1e-4` (weight_decay parameter).

**Effect:** Encourages weights to remain small, which prevents the model from overreacting to small changes in input data (smoother decision boundaries).

---

## 9. Activation Functions

Activation functions introduce **nonlinearity** into neural networks. Without them, a multi-layer network would reduce to a single linear transformation.

### 9.1 ELU (Exponential Linear Unit)

**Reference:** Clevert et al., "Fast and Accurate Deep Network Learning by Exponential Linear Units", ICLR 2016

```
f(x) = x                  if x > 0
f(x) = α · (e^x − 1)     if x ≤ 0
```

(`α = 1.0` default)

**Advantages:**
- Saturates to a negative value in the negative region, not zero — prevents the "dead neuron" problem
- Produces zero-centered output — improves gradient flow
- Preserves all advantages of ReLU

**Usage in project:** Used as the main activation function in GAT encoder, VAE, and classifier layers.

---

### 9.2 Sigmoid

```
σ(x) = 1 / (1 + e^(−x))
```

- Output range: `(0, 1)` — suitable for probability interpretation
- Derivative: `σ'(x) = σ(x) · (1 − σ(x))`

**Usage in project:** Used in the final classification layer to produce fraud probability in the [0, 1] range.

---

### 9.3 LeakyReLU

```
f(x) = x          if x > 0
f(x) = α · x      if x ≤ 0    (α = 0.2)
```

Mitigates the dead neuron problem by providing a small slope in the negative region instead of zero, unlike ReLU.

**Usage in project:** Used in the GAT attention mechanism (computation of `e_ij`).

---

### 9.4 Softmax

```
softmax(z_i) = e^(z_i) / Σ_{j=1}^{K} e^(z_j)
```

Converts a vector into a probability distribution that sums to 1.

**Usage in project:** Used in GAT to normalize attention coefficients — the sum of attention weights over each node's neighbors equals 1.

---

## 10. Explainability Methods

### 10.1 SHAP (SHapley Additive exPlanations)

**Reference:** Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

SHAP is a model explanation method based on **Shapley values** from cooperative game theory. It measures the **marginal contribution** of each feature to the model's prediction.

**Shapley value formula:**

```
ϕ_i = Σ_{S ⊆ N\{i}} [|S|! · (|N| − |S| − 1)! / |N|!] · [f(S ∪ {i}) − f(S)]
```

Where:
- `N` — set of all features
- `S` — a subset of features (excluding i)
- `f(S)` — model output when only features in set `S` are used
- `ϕ_i` — contribution of feature `i`

**How to interpret:**
- `ϕ_i > 0` → Feature `i` increases fraud probability
- `ϕ_i < 0` → Feature `i` decreases fraud probability
- `|ϕ_i|` magnitude indicates the strength of the effect

**KernelExplainer:** A model-agnostic approach that works for any model. It estimates SHAP values comparatively using background samples.

**Usage in project:** Feature importance analysis on transaction embeddings using 50–100 background samples and 200 coalition samples (`nsamples`). Visualized with waterfall and bar plots.

---

### 10.2 GNNExplainer

**Reference:** Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks", NeurIPS 2019

GNNExplainer is a method specifically designed to explain GNN model predictions. It learns edge masks and feature masks to determine which subgraph and which features contribute most to the prediction.

**Objective function:**

```
max MI(Y, (G_s, X_s))
```

Where:
- `Y` — model prediction
- `G_s` — explanatory subgraph
- `X_s` — explanatory feature subset
- `MI` — mutual information

**How it works:**
1. A soft mask (0–1) is learned for each edge
2. A soft mask is learned for each feature
3. These masks are optimized over 200 epochs
4. Edges and features with high mask values = important components

**Usage in project:** When a transaction is flagged as fraud, GNNExplainer is used to visualize which neighbor nodes (accounts, other transactions) and which edges (relationships) contributed to this decision. This transforms the decision from a "black box" into explainable results for auditors.

---

## 11. Graph Construction Techniques

### 11.1 Heterogeneous Graph Structure

The transformation from tabular data to graph structure is one of the critical stages of the project. The IEEE-CIS and PaySim datasets do not contain a natural graph structure — this structure is derived from the data.

**Node creation:**
- Each transaction becomes a transaction node
- Each unique card number (card1) becomes an account node
- Each unique merchant (addr1 + ProductCD) becomes a merchant node

**Edge creation:**
- An `initiates` edge between each transaction and the account that initiated it
- A `paid_to` edge between each transaction and the relevant merchant
- Reverse edges are added for bidirectional message passing

---

### 11.2 Temporal Edges

Temporal edges are created between sequential transactions on the same card:

```
followed_by: transaction_i → transaction_j
Conditions:
  - Same card (card1)
  - Time difference < 3600 seconds (1 hour)
  - max_neighbors = 5 (per transaction)
```

**Why?** Fraud often involves short-term transaction patterns: rapid successive transactions, increasing amounts, transactions distributed across different merchants. Temporal edges capture these sequential patterns.

---

### 11.3 Identity Edges

Edges are created between transactions that share the same device or browser information:

```
shares_identity: transaction_i ↔ transaction_j
Shared features: DeviceType, DeviceInfo, id_30 (OS), id_31 (browser)
max_group_size = 50 (to prevent excessively large groups)
max_connections = 10 (within each group)
```

**Purpose:** Fraud rings typically share the same device or software. Identity edges reveal these hidden connections.

---

### 11.4 NeighborLoader Sampling

Large graphs do not fit in GPU memory. NeighborLoader samples each node's neighborhood for mini-batch training:

```
num_neighbors = [10, 5]  → Layer 1: 10 neighbors, Layer 2: 5 neighbors
batch_size = 2048
```

This approach enables scalable training on graphs with millions of nodes.

---

## 12. Datasets

### 12.1 IEEE-CIS Fraud Detection

| Property | Value |
|---|---|
| **Source** | Kaggle IEEE-CIS Fraud Detection competition |
| **Provider** | Vesta Corporation |
| **Transaction count** | 590,540 |
| **Raw feature count** | 434 (after filtering ~360) |
| **Fraud rate** | 3.5% |
| **Feature types** | Transaction amount, card info, address, email, device info, V1–V339 anonymous features |
| **Usage in project** | Primary dataset, training of both tabular and graph-based models |

---

### 12.2 Elliptic Bitcoin Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle Elliptic Dataset |
| **Reference** | Weber et al., "Anti-Money Laundering in Bitcoin", 2019 |
| **Transaction count** | 203,769 (nodes) |
| **Feature count** | 166 (93 local + 72 aggregated) |
| **Temporal steps** | 49 time steps |
| **Labeled nodes** | 46,564 (9.8% illicit) |
| **Unlabeled nodes** | 157,205 |
| **Graph structure** | Natural homogeneous graph (Bitcoin transaction network) |
| **Usage in project** | Model performance comparison with natural graph structure |

---

### 12.3 PaySim

| Property | Value |
|---|---|
| **Source** | Kaggle PaySim 1 |
| **Reference** | Lopez-Rojas et al., "PaySim: A financial mobile money simulator", 2016 |
| **Transaction count** | 6,362,620 (filtered: 767,409 TRANSFER + CASH_OUT) |
| **Raw feature count** | 11 core attributes |
| **Fraud rate** | 0.3% |
| **Data type** | Synthetic, based on real African mobile money logs |
| **Usage in project** | Scalability testing with large-scale dataset |

---

## 13. Class Imbalance Handling

Fraud detection is inherently an **extremely imbalanced** problem — 96–99.7% of transactions are normal. If this imbalance is not managed, the model achieves 96%+ accuracy by predicting "everything is normal" while catching zero fraud.

**Methods used in the project:**

| Method | Model | Mechanism |
|---|---|---|
| Focal Loss (`α=0.75`) | HybridGATVAE | Suppresses loss for easy examples, focuses on hard examples |
| `scale_pos_weight` | XGBoost | Weights positive class by `n_neg / n_pos` ratio |
| `class_weight: balanced` | Random Forest | Weights inversely proportional to class frequencies: `w_j = n / (C · n_j)` |
| Stratified sampling | All models | Preserves class proportions during splitting |

---

## 14. References

1. **Kipf, T. N. & Welling, M.** (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.

2. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y.** (2018). Graph Attention Networks. *ICLR 2018*.

3. **Kingma, D. P. & Welling, M.** (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.

4. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

5. **Chen, T. & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.

6. **Breiman, L.** (2001). Random Forests. *Machine Learning, 45(1), 5–32*.

7. **Kingma, D. P. & Ba, J.** (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

8. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.** (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR 15, 1929–1958*.

9. **Ioffe, S. & Szegedy, C.** (2015). Batch Normalization: Accelerating Deep Network Training. *ICML 2015*.

10. **Clevert, D. A., Unterthiner, T., & Hochreiter, S.** (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). *ICLR 2016*.

11. **Lundberg, S. M. & Lee, S. I.** (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.

12. **Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J.** (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *NeurIPS 2019*.

13. **Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E.** (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. *KDD Workshop*.

14. **Lopez-Rojas, E. A., Elmir, A., & Axelsson, S.** (2016). PaySim: A Financial Mobile Money Simulator for Fraud Detection. *EMSS 2016*.

15. **Kullback, S. & Leibler, R. A.** (1951). On Information and Sufficiency. *Annals of Mathematical Statistics, 22(1), 79–86*.

---

*This document contains a comprehensive overview of all theoretical methods and techniques used within the scope of the CSE492 Graduation Project.*
