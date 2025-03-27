import torch
from torch import nn

from models.nbeats.blocks.ThetaGenerator import ThetaGenerator


class GenericBlock(nn.Module):
    """
    Input Shape:
        - ``x``: [batch_size, num_nodes, sequence_length]

    Outputs Shape:
        - ``backcast``: [batch_size, num_nodes, backcast_length]
        - ``forecast``: [batch_size, num_nodes, forecast_length]
    """

    def __init__(self, hidden_dim: int, theta_dim: int, backcast_length: int, forecast_length: int):
        super(GenericBlock, self).__init__()

        # backcast_length must be equal to the sequence_length
        self.theta_generator = ThetaGenerator(backcast_length, hidden_dim, theta_dim)

        self.backcast_fc = nn.Linear(theta_dim, backcast_length)
        self.forecast_fc = nn.Linear(theta_dim, forecast_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_backcast, theta_forecast = self.theta_generator(x)

        backcast = self.backcast_fc(theta_backcast)
        forecast = self.forecast_fc(theta_forecast)

        return backcast, forecast
