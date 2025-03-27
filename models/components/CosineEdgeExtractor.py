import torch
from torch import nn


class CosineEdgeExtractor(nn.Module):
    """
    Input Shape:
        - ``x_actuators``: (batch_size, num_actuators, sequence_length)
        - ``x_sensors``: (batch_size, num_sensors, sequence_length)

    Outputs Shape:
        - ``edges``: (batch_size, 2, num_actuators * k)
        - ``weights``: (batch_size, num_actuators * k)
    """

    def __init__(self, k: int):
        super(CosineEdgeExtractor, self).__init__()

        self.k = k

    @staticmethod
    def cos_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.

        Input Shape:
            - ``x``: (batch_size, input_dim_x, sequence_length)
            - ``y``: (batch_size, input_dim_y, sequence_length)

        Output Shape:
            - ``similarity``: (batch_size, input_dim_x, input_dim_y)
        """

        x_norm = x.norm(2, dim=2).unsqueeze(1)  # (batch_size, 1, input_dim_x)
        y_norm = y.norm(2, dim=2).unsqueeze(1)  # (batch_size, 1, input_dim_y)

        similarity = x @ y.mT / (x_norm.mT @ y_norm)  # (batch_size, input_dim_x, input_dim_y)

        return similarity

    def forward(self, x_actuators: torch.Tensor, x_sensors: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        batch_size, num_actuators, _ = x_actuators.shape

        similarity = self.cos_similarity(x_actuators, x_sensors)

        indices = torch.topk(similarity ** 2, self.k, dim=2).indices
        values = similarity.gather(dim=2, index=indices)  # [batch_siz, num_actuators, k]

        target_nodes = indices.reshape(batch_size, -1)  # [batch_size, num_edges]
        source_nodes = torch.arange(num_actuators).repeat_interleave(self.k).repeat(batch_size, 1).to(target_nodes.device)

        edges = torch.stack([source_nodes, target_nodes], dim=1)  # [batch_size, 2, num_actuators * k]
        weights = values.reshape(batch_size, -1)  # [batch_size, num_actuators * k]

        return edges, weights
