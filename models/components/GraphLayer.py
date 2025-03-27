import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv


class GraphLayer(nn.Module):
    """
    Input Shape:
        - ``x``: [batch_size, num_nodes, sequence_length]
        - ``edges``: [batch_size, 2, num_edges]
        - ``weights``: [batch_size, num_edges]

    Outputs Shape:
        - ``output``: [batch_size, num_nodes, output_dim]
    """

    def __init__(self, sequence_length: int, output_dim: int):
        super(GraphLayer, self).__init__()

        self.output_dim = output_dim

        self.gat = GCNConv(in_channels=sequence_length, out_channels=output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor = None):
        batch_size = x.shape[0]
        data_list = [Data(x=x[i], edge_index=edges[i], edge_attr=weights[i].unsqueeze(1)) for i in range(batch_size)]
        batch = Batch.from_data_list(data_list)

        # [batch_size * num_nodes, output_dim], [2, batch_size * num_edges], [batch_size * num_edges]
        out = self.gat(batch['x'], batch['edge_index'], batch['edge_attr'])
        out = self.bn(out)
        out = self.activation(out)
        out = out.reshape(batch_size, -1, self.output_dim)  # [batch_size * num_nodes, output_dim] -> [batch_size, num_nodes, output_dim]

        return out
