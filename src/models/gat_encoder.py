"""
Heterogeneous GAT encoder.
Wraps PyG HeteroConv + GATConv to produce per-node embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGATEncoder(nn.Module):
    def __init__(
        self,
        metadata: tuple,          # (node_types, edge_types) from HeteroData.metadata()
        in_channels: dict,        # {node_type: feature_dim}
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = node_types
        self.dropout = dropout
        self.num_layers = num_layers

        # Linear projections: map each node type's raw features to hidden_channels
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            in_dim = in_channels.get(ntype, hidden_channels)
            self.input_projections[ntype] = nn.Linear(in_dim, hidden_channels)

        # Build HeteroConv layers — explicit input dimensions (no lazy)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            curr_out = out_channels if is_last else hidden_channels
            curr_heads = 1 if is_last else num_heads
            # After input_projections, all node types have hidden_channels dims
            # After layer 0 with concat, dims become hidden_channels * num_heads
            curr_in = hidden_channels if i == 0 else hidden_channels * num_heads

            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=(curr_in, curr_in),
                    out_channels=curr_out,
                    heads=curr_heads,
                    concat=(not is_last),
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        # Project raw features to common hidden dimension
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_projections:
                h_dict[ntype] = F.elu(self.input_projections[ntype](x))
            else:
                h_dict[ntype] = x

        # Message passing layers
        for i, conv in enumerate(self.convs):
            h_prev = h_dict
            h_dict = conv(h_prev, edge_index_dict)

            # Carry forward node types that weren't updated (source-only nodes)
            for ntype in h_prev:
                if ntype not in h_dict:
                    h_dict[ntype] = h_prev[ntype]

            # Activation + dropout (skip on last layer)
            is_last = (i == self.num_layers - 1)
            if not is_last:
                h_dict = {
                    k: F.dropout(F.elu(v), p=self.dropout, training=self.training)
                    for k, v in h_dict.items()
                }

        return h_dict
