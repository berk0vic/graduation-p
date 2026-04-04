#!/usr/bin/env python3
"""
Export a small heterogeneous subgraph from IEEE-CIS HeteroData to JSON for web viz.

Usage:
  python scripts/export_graph_sample.py
  python scripts/export_graph_sample.py --graph data/processed/ieee_cis/hetero_graph_v3.pt --out web/graph_sample.json

Then serve the web folder (JSON must load from same origin):
  python -m http.server 8765 --directory web
  open http://localhost:8765/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def _nid(prefix: str, i: int) -> str:
    return f"{prefix}:{int(i)}"


def sample_subgraph(data, seed_txns: list[int], max_transactions: int = 100) -> tuple[set, set, set]:
    txn_set = set(seed_txns)
    acct_set: set[int] = set()
    merch_set: set[int] = set()

    ei_init = data["account", "initiates", "transaction"].edge_index
    acct_of_txn: dict[int, int] = {}
    txns_of_acct: defaultdict[int, list[int]] = defaultdict(list)
    for a, t in zip(ei_init[0].tolist(), ei_init[1].tolist()):
        acct_of_txn[t] = a
        txns_of_acct[a].append(t)

    ei_pay = data["transaction", "paid_to", "merchant"].edge_index
    merch_of_txn: dict[int, int] = {}
    for t, m in zip(ei_pay[0].tolist(), ei_pay[1].tolist()):
        merch_of_txn[t] = m

    for t in list(txn_set):
        if t in acct_of_txn:
            acct_set.add(acct_of_txn[t])
        if t in merch_of_txn:
            merch_set.add(merch_of_txn[t])

    for a in list(acct_set):
        for t in txns_of_acct.get(a, []):
            if len(txn_set) >= max_transactions:
                break
            txn_set.add(t)
        if len(txn_set) >= max_transactions:
            break

    for t in list(txn_set):
        if t in acct_of_txn:
            acct_set.add(acct_of_txn[t])
        if t in merch_of_txn:
            merch_set.add(merch_of_txn[t])

    return txn_set, acct_set, merch_set


def build_vis_payload(data, txn_set: set[int], acct_set: set[int], merch_set: set[int]) -> dict:
    y = data["transaction"].y
    nodes = []
    for t in sorted(txn_set):
        fraud = bool(y[t].item() == 1)
        nodes.append(
            {
                "id": _nid("t", t),
                "label": f"Txn {t}",
                "group": "fraud" if fraud else "transaction",
                "title": "Fraud" if fraud else "Legit",
            }
        )
    for a in sorted(acct_set):
        nodes.append(
            {
                "id": _nid("a", a),
                "label": f"Card / Acct {a}",
                "group": "account",
                "title": "Account node (card1 proxy)",
            }
        )
    for m in sorted(merch_set):
        nodes.append(
            {
                "id": _nid("m", m),
                "label": f"Merchant {m}",
                "group": "merchant",
                "title": "Merchant (addr1 + ProductCD proxy)",
            }
        )

    allowed_t = txn_set
    allowed_a = acct_set
    allowed_m = merch_set

    edges: list[dict] = []

    def add_edges(etype: tuple, src_t: str, dst_t: str, label: str) -> None:
        ei = data[etype].edge_index
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            if src_t == "t" and dst_t == "a":
                ok = s in allowed_t and d in allowed_a
                fr, to = _nid("t", s), _nid("a", d)
            elif src_t == "a" and dst_t == "t":
                ok = s in allowed_a and d in allowed_t
                fr, to = _nid("a", s), _nid("t", d)
            elif src_t == "t" and dst_t == "m":
                ok = s in allowed_t and d in allowed_m
                fr, to = _nid("t", s), _nid("m", d)
            elif src_t == "m" and dst_t == "t":
                ok = s in allowed_m and d in allowed_t
                fr, to = _nid("m", s), _nid("t", d)
            elif src_t == "t" and dst_t == "t":
                ok = s in allowed_t and d in allowed_t
                fr, to = _nid("t", s), _nid("t", d)
            else:
                ok = False
                fr, to = "", ""
            if ok:
                edges.append({"from": fr, "to": to, "label": label})

    add_edges(("account", "initiates", "transaction"), "a", "t", "initiates")
    add_edges(("transaction", "initiated_by", "account"), "t", "a", "initiated_by")
    add_edges(("transaction", "paid_to", "merchant"), "t", "m", "paid_to")
    add_edges(("merchant", "received_from", "transaction"), "m", "t", "received_from")

    if ("transaction", "followed_by", "transaction") in data.edge_types:
        add_edges(("transaction", "followed_by", "transaction"), "t", "t", "followed_by (time)")
    if ("transaction", "preceded_by", "transaction") in data.edge_types:
        add_edges(("transaction", "preceded_by", "transaction"), "t", "t", "preceded_by (time)")
    if ("transaction", "shares_identity", "transaction") in data.edge_types:
        add_edges(("transaction", "shares_identity", "transaction"), "t", "t", "same identity")
    if ("transaction", "shares_identity_rev", "transaction") in data.edge_types:
        add_edges(("transaction", "shares_identity_rev", "transaction"), "t", "t", "same identity (rev)")

    return {
        "meta": {
            "description": "Sample of IEEE-CIS heterogeneous graph (PyG export)",
            "node_counts": {
                "transaction": len(txn_set),
                "account": len(acct_set),
                "merchant": len(merch_set),
            },
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }


def pick_seeds(data, n_fraud: int, n_legit: int, rng: torch.Generator) -> list[int]:
    y = data["transaction"].y
    fraud_idx = (y == 1).nonzero(as_tuple=True)[0]
    legit_idx = (y == 0).nonzero(as_tuple=True)[0]
    perm_f = fraud_idx[torch.randperm(len(fraud_idx), generator=rng)][:n_fraud]
    perm_l = legit_idx[torch.randperm(len(legit_idx), generator=rng)][:n_legit]
    return torch.cat([perm_f, perm_l]).tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/processed/ieee_cis/hetero_graph_v3.pt"),
        help="Path to torch-saved HeteroData",
    )
    p.add_argument("--out", type=Path, default=Path("web/graph_sample.json"))
    p.add_argument("--fraud-seeds", type=int, default=8)
    p.add_argument("--legit-seeds", type=int, default=8)
    p.add_argument("--max-txn", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.graph.is_file():
        raise SystemExit(f"Graph file not found: {args.graph.resolve()}")

    try:
        data = torch.load(args.graph, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(args.graph, map_location="cpu")
    rng = torch.Generator().manual_seed(args.seed)
    seeds = pick_seeds(data, args.fraud_seeds, args.legit_seeds, rng)
    txn_set, acct_set, merch_set = sample_subgraph(data, seeds, max_transactions=args.max_txn)
    payload = build_vis_payload(data, txn_set, acct_set, merch_set)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out} ({payload['meta']})")


if __name__ == "__main__":
    main()
