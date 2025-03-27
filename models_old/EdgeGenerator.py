import torch
from torch import nn


class EdgeGenerator(nn.Module):
    """
    Input Shape:
        - ``x_actuators``: [num_actuators, sequence_length]
        - ``x_sensors``: [num_sensors, sequence_length]

    Outputs Shape:
        - ``edges``: [2, num_actuators * k]
        - ``weights``: [num_actuators * k]
    """

    def __init__(self, k: int):
        super(EdgeGenerator, self).__init__()

        self.k = k

    @staticmethod
    def cos_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.

        Input Shape:
            - ``x``: [input_dim_x, sequence_length]
            - ``y``: [input_dim_y, sequence_length]

        Output Shape:
            - ``similarity``: [input_dim_x, input_dim_y]
        """

        x_norm = x.norm(2, dim=1).unsqueeze(1)  # [input_dim_x, 1]
        y_norm = y.norm(2, dim=1).unsqueeze(1)  # [input_dim_y, 1]

        similarity = (x @ y.T) / (x_norm @ y_norm.T)  # [input_dim_x, input_dim_y]

        return similarity

    def forward(self, x_actuators: torch.Tensor, x_sensors: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        device = x_actuators.device

        similarity = self.cos_similarity(x_actuators, x_sensors)

        values, indices = torch.topk(similarity ** 2, self.k, dim=1)
        values = torch.sqrt(values)

        target_nodes = indices.reshape(-1)  # [num_actuators * k]
        source_nodes = torch.arange(x_actuators.shape[0], device=device).repeat_interleave(self.k)

        edges = torch.stack([source_nodes, target_nodes], dim=0)
        weights = values.reshape(-1)

        return edges, weights
