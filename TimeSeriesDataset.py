import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, actuator_indices: np.ndarray, slide_window_size: int, slide_stride: int):
        x = []
        y = []
        for i in range(0, len(data) - slide_window_size, slide_stride):
            x.append(data[i:i + slide_window_size])
            y.append(data[i + slide_window_size])

        x = np.array(x)
        y = np.array(y)

        self.x = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)
        self.actuator_indices = torch.tensor(actuator_indices)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.actuator_indices, self.y[idx]
