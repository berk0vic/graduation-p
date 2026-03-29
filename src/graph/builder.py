"""
Converts tabular transaction data (IEEE-CIS / PaySim) into a
PyG HeteroData graph with proper feature engineering.

Node types:  account | transaction | merchant
Edge types:
  account  -[initiates]->   transaction
  transaction -[paid_to]->  merchant
  (+ reverse edges for bidirectional message passing)
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_hetero_graph(df: pd.DataFrame, dataset: str = "ieee_cis") -> HeteroData:
    data = HeteroData()

    if dataset == "ieee_cis":
        return _build_ieee_cis(df, data)
    elif dataset == "paysim":
        return _build_paysim(df, data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ──────────────────────────────────────────────
# IEEE-CIS graph builder (improved preprocessing)
# ──────────────────────────────────────────────
def _build_ieee_cis(df: pd.DataFrame, data: HeteroData) -> HeteroData:
    # Reset index to ensure 0-based contiguous indices
    df = df.reset_index(drop=True)

    # ─── 1. Feature Engineering for Transaction Nodes ───
    txn_features = _ieee_txn_features(df)
    n_txn = len(df)

    # ─── 2. Account nodes (card1 as proxy) ───
    accounts = pd.unique(df["card1"].dropna()).tolist()
    acct_map = {a: i for i, a in enumerate(accounts)}

    # Account features: aggregate statistics per account
    acct_features = _ieee_account_features(df, accounts, acct_map)

    # ─── 3. Merchant nodes (addr1 + ProductCD as proxy) ───
    df["merchant_key"] = df["addr1"].astype(str) + "_" + df["ProductCD"].astype(str)
    merchants = pd.unique(df["merchant_key"].dropna()).tolist()
    merch_map = {m: i for i, m in enumerate(merchants)}

    # Merchant features: aggregate statistics per merchant
    merch_features = _ieee_merchant_features(df, merchants, merch_map)

    # ─── 4. Set node features and labels ───
    data["account"].x = acct_features
    data["merchant"].x = merch_features
    data["transaction"].x = txn_features
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)

    # ─── 5. Edges: account -[initiates]-> transaction ───
    valid_mask = df["card1"].isin(acct_map)
    valid_df = df[valid_mask]
    src_acct = [acct_map[a] for a in valid_df["card1"]]
    dst_txn = valid_df.index.tolist()
    data["account", "initiates", "transaction"].edge_index = torch.tensor(
        [src_acct, dst_txn], dtype=torch.long
    )
    data["transaction", "initiated_by", "account"].edge_index = torch.tensor(
        [dst_txn, src_acct], dtype=torch.long
    )

    # ─── 6. Edges: transaction -[paid_to]-> merchant ───
    src_txn_m, dst_m = [], []
    for idx, mk in zip(df.index, df["merchant_key"]):
        if mk in merch_map:
            src_txn_m.append(idx)
            dst_m.append(merch_map[mk])
    data["transaction", "paid_to", "merchant"].edge_index = torch.tensor(
        [src_txn_m, dst_m], dtype=torch.long
    )
    data["merchant", "received_from", "transaction"].edge_index = torch.tensor(
        [dst_m, src_txn_m], dtype=torch.long
    )

    # ─── 7. Temporal edges: same card, within time window ───
    src_temp, dst_temp = _build_temporal_edges(
        df, time_col="TransactionDT", group_col="card1",
        time_window=3600, max_neighbors=5,
    )
    if len(src_temp) > 0:
        data["transaction", "followed_by", "transaction"].edge_index = torch.tensor(
            [src_temp, dst_temp], dtype=torch.long
        )
        data["transaction", "preceded_by", "transaction"].edge_index = torch.tensor(
            [dst_temp, src_temp], dtype=torch.long
        )

    # ─── 8. Identity edges: shared device/IP between accounts ───
    src_id, dst_id = _build_identity_edges(df)
    if len(src_id) > 0:
        data["transaction", "shares_identity", "transaction"].edge_index = torch.tensor(
            [src_id, dst_id], dtype=torch.long
        )
        data["transaction", "shares_identity_rev", "transaction"].edge_index = torch.tensor(
            [dst_id, src_id], dtype=torch.long
        )

    # ─── 9. Set num_nodes (REQUIRED for NeighborLoader) ───
    data["transaction"].num_nodes = n_txn
    data["account"].num_nodes = len(accounts)
    data["merchant"].num_nodes = len(merchants)

    print(f"Graph built:")
    print(f"  Transactions: {n_txn:,} ({txn_features.shape[1]} features)")
    print(f"  Accounts: {len(accounts):,} ({acct_features.shape[1]} features)")
    print(f"  Merchants: {len(merchants):,} ({merch_features.shape[1]} features)")
    print(f"  Edges: {len(src_acct):,} (initiates + reverse)")
    print(f"  Edges: {len(src_txn_m):,} (paid_to + reverse)")
    print(f"  Edges: {len(src_temp):,} (temporal: followed_by + preceded_by)")
    print(f"  Edges: {len(src_id):,} (identity: shares_identity + reverse)")

    return data


def _build_temporal_edges(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    group_col: str = "card1",
    time_window: int = 3600,  # saniye (1 saat)
    max_neighbors: int = 5,
) -> tuple[list, list]:
    """
    Aynı karttan (group_col) time_window saniye içinde yapılan
    işlemler arasında temporal kenar oluşturur.
    Makalelerdeki yaklaşım: shared identifier + temporal proximity.
    """
    src_list, dst_list = [], []

    if time_col not in df.columns:
        print(f"  WARNING: {time_col} kolonu bulunamadı, temporal edge oluşturulamadı.")
        return src_list, dst_list

    # Zamana göre sırala
    df_sorted = df[[group_col, time_col]].copy()
    df_sorted["orig_idx"] = df_sorted.index

    # Her kart grubu için temporal kenar oluştur
    grouped = df_sorted.groupby(group_col)
    for _, group in grouped:
        if len(group) < 2:
            continue

        group = group.sort_values(time_col)
        indices = group["orig_idx"].values
        times = group[time_col].values

        for i in range(len(group)):
            count = 0
            for j in range(i + 1, len(group)):
                dt = abs(times[j] - times[i])
                if dt > time_window:
                    break
                src_list.append(indices[i])
                dst_list.append(indices[j])
                count += 1
                if count >= max_neighbors:
                    break

    print(f"  Temporal edges: {len(src_list):,} (window={time_window}s, max_neighbors={max_neighbors})")
    return src_list, dst_list


def _build_identity_edges(
    df: pd.DataFrame,
    identity_cols: list[str] | None = None,
    max_group_size: int = 50,
) -> tuple[list, list]:
    """
    Build edges between transactions that share device/IP/email identity.
    Transactions sharing the same DeviceType+DeviceInfo or id_30+id_31 are linked.
    """
    if identity_cols is None:
        identity_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31"]

    # Check which columns exist
    available = [c for c in identity_cols if c in df.columns]
    if not available:
        print("  WARNING: No identity columns found, skipping identity edges.")
        return [], []

    src_list, dst_list = [], []
    seen_pairs = set()

    # Group by available identity columns (non-null combinations)
    for cols in [available[:2], available[2:], available]:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        # Drop rows where all identity cols are null
        valid = df.dropna(subset=cols, how="all")
        if len(valid) == 0:
            continue

        # Create identity key from available columns
        key = valid[cols].fillna("__NA__").astype(str).agg("_".join, axis=1)
        valid = valid.copy()
        valid["_id_key"] = key

        for _, group in valid.groupby("_id_key"):
            if len(group) < 2 or len(group) > max_group_size:
                continue
            indices = group.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, min(i + 10, len(indices))):
                    pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        src_list.append(indices[i])
                        dst_list.append(indices[j])

    print(f"  Identity edges: {len(src_list):,} (cols={available})")
    return src_list, dst_list


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add velocity / risk features per card (card1):
    - txn_count_1h, txn_count_6h, txn_count_24h: rolling txn count
    - amt_mean_24h, amt_std_24h: rolling amount stats
    - time_since_last: seconds since last txn on same card
    - amount_zscore: how unusual is this amount vs card's history
    - hour_sin, hour_cos: cyclical time encoding
    """
    df = df.copy()

    # Time features (TransactionDT is in seconds from some reference)
    if "TransactionDT" in df.columns:
        # Cyclical hour encoding
        seconds_in_day = 86400
        hour_frac = (df["TransactionDT"] % seconds_in_day) / seconds_in_day
        df["hour_sin"] = np.sin(2 * np.pi * hour_frac)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac)

        # Sort by card+time for rolling features (preserve original index for re-sort)
        df = df.sort_values(["card1", "TransactionDT"])

        # Time since last transaction (same card)
        df["time_since_last"] = df.groupby("card1")["TransactionDT"].diff().fillna(0)

        # Rolling transaction counts (approximate via cumcount within time windows)
        # For efficiency, use expanding count per card
        df["txn_cumcount"] = df.groupby("card1").cumcount()

        # Per-card rolling amount stats
        expanding = df.groupby("card1")["TransactionAmt"].expanding()
        df["amt_rolling_mean"] = expanding.mean().reset_index(level=0, drop=True)
        df["amt_rolling_std"] = expanding.std().reset_index(level=0, drop=True).fillna(0)

        # Amount z-score vs card history
        df["amount_zscore"] = (df["TransactionAmt"] - df["amt_rolling_mean"]) / (df["amt_rolling_std"] + 1e-8)
        df["amount_zscore"] = df["amount_zscore"].clip(-10, 10).fillna(0)

        # Clean up temp columns
        df = df.drop(columns=["amt_rolling_mean", "amt_rolling_std"], errors="ignore")

        # Re-sort by original index
        df = df.sort_index()

    return df


def _ieee_txn_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Proper feature engineering:
    1. Drop columns with >90% zeros (former NaN, now noise)
    2. One-hot encode key categorical columns
    3. Add velocity/risk features
    4. Median imputation for remaining NaN
    5. IQR normalization + clip
    """
    # ── Step 0: Add velocity features ──
    df = _add_velocity_features(df)

    # ── Step 1: Select useful numeric columns ──
    drop_cols = ["isFraud", "TransactionID", "card1", "merchant_key"]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in drop_cols]

    # Drop columns with >90% zeros (these were NaN filled with 0, no info)
    num_df = df[num_cols]
    zero_pct = (num_df == 0).sum() / len(num_df)
    good_num_cols = [c for c in num_cols if zero_pct[c] < 0.90]

    # Make sure velocity features are included even if they have zeros
    velocity_cols = ["hour_sin", "hour_cos", "time_since_last", "txn_cumcount",
                     "amount_zscore"]
    for vc in velocity_cols:
        if vc in df.columns and vc not in good_num_cols:
            good_num_cols.append(vc)

    # ── Step 2: One-hot encode categorical columns ──
    cat_cols = ["ProductCD", "card4", "card6"]
    cat_dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)

    # ── Step 3: Combine numeric + categorical ──
    feat_df = pd.concat([df[good_num_cols], cat_dummies], axis=1)

    # ── Step 4: Median imputation (remaining NaN) ──
    medians = feat_df.median()
    feat_df = feat_df.fillna(medians)

    # ── Step 5: IQR normalization + clip ──
    feat_np = feat_df.values.astype("float32")
    q1 = np.percentile(feat_np, 25, axis=0)
    q3 = np.percentile(feat_np, 75, axis=0)
    median = np.median(feat_np, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    feat_np = (feat_np - median) / iqr
    feat_np = np.clip(feat_np, -5, 5)

    return torch.tensor(feat_np, dtype=torch.float32)


def _ieee_account_features(
    df: pd.DataFrame, accounts: list, acct_map: dict
) -> torch.Tensor:
    """
    Aggregate statistics per account (card1):
    - transaction count
    - mean, std, min, max of TransactionAmt
    """
    acct_stats = df.groupby("card1")["TransactionAmt"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    acct_stats["std"] = acct_stats["std"].fillna(0)

    features = []
    for acct in accounts:
        if acct in acct_stats.index:
            row = acct_stats.loc[acct]
            features.append([row["count"], row["mean"], row["std"], row["min"], row["max"]])
        else:
            features.append([0, 0, 0, 0, 0])

    feat_np = np.array(features, dtype="float32")

    # Normalize account features (IQR)
    q1 = np.percentile(feat_np, 25, axis=0)
    q3 = np.percentile(feat_np, 75, axis=0)
    median = np.median(feat_np, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    feat_np = (feat_np - median) / iqr
    feat_np = np.clip(feat_np, -5, 5)

    return torch.tensor(feat_np, dtype=torch.float32)


def _ieee_merchant_features(
    df: pd.DataFrame, merchants: list, merch_map: dict
) -> torch.Tensor:
    """
    Aggregate statistics per merchant:
    - transaction count
    - mean, std of TransactionAmt
    """
    merch_stats = df.groupby("merchant_key")["TransactionAmt"].agg(
        ["count", "mean", "std"]
    )
    merch_stats["std"] = merch_stats["std"].fillna(0)

    features = []
    for m in merchants:
        if m in merch_stats.index:
            row = merch_stats.loc[m]
            features.append([row["count"], row["mean"], row["std"]])
        else:
            features.append([0, 0, 0])

    feat_np = np.array(features, dtype="float32")

    # Normalize
    q1 = np.percentile(feat_np, 25, axis=0)
    q3 = np.percentile(feat_np, 75, axis=0)
    median = np.median(feat_np, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    feat_np = (feat_np - median) / iqr
    feat_np = np.clip(feat_np, -5, 5)

    return torch.tensor(feat_np, dtype=torch.float32)


# ──────────────────────────────────────────────
# PaySim graph builder
# ──────────────────────────────────────────────
def _build_paysim(df: pd.DataFrame, data: HeteroData) -> HeteroData:
    df = df.reset_index(drop=True)

    orig_accounts = pd.unique(df["nameOrig"]).tolist()
    dest_accounts = pd.unique(df["nameDest"]).tolist()
    all_accounts = list(set(orig_accounts + dest_accounts))
    acct_map = {a: i for i, a in enumerate(all_accounts)}

    n_txn = len(df)

    # Account features: aggregate stats
    orig_stats = df.groupby("nameOrig")["amount"].agg(["count", "mean", "std"]).fillna(0)
    acct_feats = []
    for a in all_accounts:
        if a in orig_stats.index:
            row = orig_stats.loc[a]
            acct_feats.append([row["count"], row["mean"], row["std"]])
        else:
            acct_feats.append([0, 0, 0])

    acct_np = np.array(acct_feats, dtype="float32")
    # Normalize
    q1 = np.percentile(acct_np, 25, axis=0)
    q3 = np.percentile(acct_np, 75, axis=0)
    med = np.median(acct_np, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    acct_np = np.clip((acct_np - med) / iqr, -5, 5)

    data["account"].x = torch.tensor(acct_np, dtype=torch.float32)
    data["transaction"].x = _paysim_txn_features(df)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)

    # account -[initiates]-> transaction
    src = [acct_map[a] for a in df["nameOrig"]]
    dst = list(range(n_txn))
    data["account", "initiates", "transaction"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )
    # Reverse
    data["transaction", "initiated_by", "account"].edge_index = torch.tensor(
        [dst, src], dtype=torch.long
    )

    # transaction -[received_by]-> account
    src2 = list(range(n_txn))
    dst2 = [acct_map[a] for a in df["nameDest"]]
    data["transaction", "received_by", "account"].edge_index = torch.tensor(
        [src2, dst2], dtype=torch.long
    )
    # Reverse
    data["account", "sent_to", "transaction"].edge_index = torch.tensor(
        [dst2, src2], dtype=torch.long
    )

    return data


def _paysim_txn_features(df: pd.DataFrame) -> torch.Tensor:
    cols = ["amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]
    feat_np = df[cols].fillna(0).values.astype("float32")

    # IQR normalization
    q1 = np.percentile(feat_np, 25, axis=0)
    q3 = np.percentile(feat_np, 75, axis=0)
    median = np.median(feat_np, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    feat_np = np.clip((feat_np - median) / iqr, -5, 5)

    return torch.tensor(feat_np, dtype=torch.float32)
