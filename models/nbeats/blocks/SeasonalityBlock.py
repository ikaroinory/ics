import numpy as np
import torch
from torch import nn


class SeasonalityBlock(nn.Module):
    def __init__(self):
        super(SeasonalityBlock, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta_backcast, theta_forecast = self.theta_generator(x)

        return x, x


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return torch.arange(0, horizon) / horizon


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


if __name__ == '__main__':
    o = linear_space(100, 10, True)
    print(o)
