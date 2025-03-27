import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import BatchNorm, GCNConv

from .EdgeGenerator import EdgeGenerator
from .NBeatsNet import NBeatsNet


class CommonRelationModule(nn.Module):
    """
    Input Shape:
        - ``x``: [num_nodes, sequence_length]

    Outputs Shape:
        - ``backcast``: [num_nodes, sequence_length]
        - ``forecast``: [num_nodes, output_dim]
    """

    def __init__(self, sequence_length: int, hidden_dim: int, output_dim: int, k: int, device='cpu'):
        super(CommonRelationModule, self).__init__()

        self.g_conv = GCNConv(in_channels=sequence_length, out_channels=hidden_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, sequence_length),
            BatchNorm1d(sequence_length),
            nn.LeakyReLU(),

            Dropout(0.1)
        )
        self.doubly_residua = NBeatsNet(backcast_length=sequence_length, forecast_length=output_dim, device=device)

        self.bn_g_conv = BatchNorm(hidden_dim)

        self.relation_extractor = EdgeGenerator(k=k)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        edges, weights = self.relation_extractor(x, x)  # [2, num_nodes * k], [num_nodes * k]

        x = self.g_conv(x, edges, weights)  # [num_nodes, sequence_length] -> [num_nodes, d_hidden]
        x = self.bn_g_conv(x)
        x = F.leaky_relu(x)

        x = self.fc_layers(x)  # [num_nodes, d_hidden] -> [num_nodes, out_features]

        backcast, forecast = self.doubly_residua(x)

        return backcast, forecast
