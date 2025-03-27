import torch
from torch import nn


class ThetaGenerator(nn.Module):
    """
    Input Shape:
        - ``x``: [batch_size, num_nodes, sequence_length]

    Outputs Shape:
        - ``theta_backcast``: [batch_size, num_nodes, output_dim]
        - ``theta_forecast``: [batch_size, num_nodes, output_dim]
    """

    def __init__(self, sequence_length: int, hidden_dim: int, theta_dim: int):
        super(ThetaGenerator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.theta_backcast_fc = nn.Linear(hidden_dim, theta_dim, bias=False)
        self.theta_forecast_fc = nn.Linear(hidden_dim, theta_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(x)  # [batch_size, num_nodes, hidden_dim]

        theta_backcast = self.theta_backcast_fc(x)  # [batch_size, num_nodes, output_dim]
        theta_forecast = self.theta_forecast_fc(x)  # [batch_size, num_nodes, output_dim]

        return theta_backcast, theta_forecast
