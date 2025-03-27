import torch
from torch import nn
from torch_geometric.data import Batch, Data

from models.components import CosineEdgeExtractor, GraphLayer, NBeatsNet


class BasicRelationModule(nn.Module):
    """
    Input Shape:
        - ``x``: [batch_size, num_nodes, sequence_length]

    Outputs Shape:
        - ``backcast``: [batch_size, num_nodes, sequence_length]
        - ``forecast``: [batch_size, num_nodes, output_dim]
    """

    def __init__(self, sequence_length: int, hidden_dim: int, output_dim: int, k: int, device='cpu'):
        super(BasicRelationModule, self).__init__()

        self.edge_extractor = CosineEdgeExtractor(k=k)

        self.graph_layer = GraphLayer(sequence_length=sequence_length, output_dim=hidden_dim)

        self.doubly_residua = NBeatsNet(backcast_length=sequence_length, forecast_length=output_dim)

        self.to(device)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        edges, weights = self.edge_extractor(x, x)  # [batch_size, 2, num_nodes * k], [batch_size, num_nodes * k]

        out = self.graph_layer(x, edges, weights)

        return out, out


if __name__ == '__main__':
    device = 'cpu'
    x = torch.rand([32, 51, 100], device=device).double()

    # model = BasicRelationModule(sequence_length=100, hidden_dim=128, output_dim=10, k=3, device=device).to(device).double()
    # b, f = model(x)

    model = NBeatsNet(backcast_length=100, forecast_length=10).to(device).double()
    b, f = model(x)
    print(b.shape, f.shape)
