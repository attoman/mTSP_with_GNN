# models/gnn_transformer_encoder.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class EnhancedGNNTransformerEncoder(nn.Module):
    """
    self-attention을 포함한 GNN Transformer 인코더.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super(EnhancedGNNTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            in_channels = hidden_channels * heads

        self.gnn_output = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, batch):
        """
        GNN Transformer 인코더를 통한 순전파.
        """
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.gnn_output[1].p, training=self.training)

        x = self.gnn_output(x)

        return x
