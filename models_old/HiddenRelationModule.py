import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from models_old.EdgeGenerator import EdgeGenerator


class HiddenRelationModule(nn.Module):
    """
    Input Shape:
        - ``x``: [num_nodes, sequence_length]

    Outputs Shape:
        - ``output``: [num_nodes, output_dim]
    """

    def __init__(self, sequence_length: int, hidden_dim: int, output_dim: int, k: int, num_graph: int = 2):
        super(HiddenRelationModule, self).__init__()

        self.hidden_dim = hidden_dim

        # actuator_encoder + sensor_encoder + edge_generator = Auto Weight Function

        self.actuator_encoder = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )

        self.edge_generator = EdgeGenerator(k=k)

        self.graph_layers = nn.ModuleList()
        for _ in range(num_graph):
            self.graph_layers.append(GATConv(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU()
        )

    def forward(self, x: torch.Tensor, actuator_indices: torch.Tensor) -> torch.Tensor:
        device = x.device
        actuator_indices = actuator_indices.squeeze()

        num_nodes = x.shape[0]

        indices = torch.arange(num_nodes, device=device)
        sensor_indices = indices[~torch.isin(indices, actuator_indices)]

        x_actuator = self.actuator_encoder(x[actuator_indices])  # [num_actuator_nodes, hidden_dim]
        x_sensor = self.sensor_encoder(x[sensor_indices])  # [num_sensor_nodes, hidden_dim]

        edges, weights = self.edge_generator(x_actuator, x_sensor)  # [2, num_actuator_nodes * k], [num_actuator_nodes * k]

        new_x = torch.zeros([num_nodes, self.hidden_dim], device=device)
        new_x[actuator_indices] = x_actuator
        new_x[sensor_indices] = x_sensor

        for graph in self.graph_layers:
            new_x = graph(new_x, edges)
        new_x = F.leaky_relu(new_x)

        output = self.mlp(new_x)

        return output
