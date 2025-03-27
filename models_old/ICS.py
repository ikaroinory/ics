import torch
from torch import nn

from .CommonRelationModule import CommonRelationModule
from .HiddenRelationModule import HiddenRelationModule


class ICS(nn.Module):
    """
    Input Shape:
        - ``x``: [num_nodes, sequence_length]

    Outputs Shape:
        - ``output``: [num_nodes, 1]
    """

    def __init__(self, sequence_length: int, hidden_dim: int, middle_output_dim: int, k: int, num_graph: int = 2, device='cpu'):
        super(ICS, self).__init__()

        self.common_relationship_module = CommonRelationModule(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            output_dim=middle_output_dim,
            k=k,
            device=device
        )
        self.hidden_relation_module = HiddenRelationModule(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            output_dim=middle_output_dim,
            k=k,
            num_graph=num_graph
        )

        self.output_layer = nn.Sequential(
            nn.Linear(middle_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, actuator_indices: torch.Tensor) -> torch.Tensor:
        common_backcast, common_forecast = self.common_relationship_module(x)

        x = x - common_backcast

        o = self.hidden_relation_module(x, actuator_indices)

        o = self.output_layer(o + common_forecast)

        return o.T
