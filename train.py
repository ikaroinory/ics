import argparse
import copy
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import L1Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from Logger import Logger
from TimeSeriesDataset import TimeSeriesDataset
from models import ICS


def parse_arguments():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def resample(data: pd.DataFrame) -> pd.DataFrame:
    return data.resample('5s').mean()


def get_dataloader(seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    with open('data/processed/a2_normal_values_df.pkl', 'rb') as file:
        values_df: pd.DataFrame = pickle.load(file)
    with open('data/processed/a2_actuator_indices_np.pkl', 'rb') as file:
        actuator_indices = pickle.load(file)

    values_df = resample(values_df)

    x = values_df.values

    # 7:1:2
    x_train, x_others = train_test_split(x, test_size=0.3, random_state=seed)
    x_valid, x_test = train_test_split(x_others, test_size=0.6, random_state=seed)

    train_dataset = TimeSeriesDataset(x_train, actuator_indices, 15, 5)
    valid_dataset = TimeSeriesDataset(x_valid, actuator_indices, 15, 5)
    test_dataset = TimeSeriesDataset(x_test, actuator_indices, 15, 5)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def train_epoch(train_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer, device: str) -> float:
    model.to(device)
    model.train()

    total_train_loss = 0
    for x, actuator_indices, y in tqdm(train_loader):
        x = x.squeeze()

        x = x.to(device)
        actuator_indices = actuator_indices.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x, actuator_indices)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def valid_epoch(valid_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str) -> float:
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_valid_loss = 0
        for x, actuator_indices, y in tqdm(valid_loader):
            x = x.squeeze()

            x = x.to(device)
            actuator_indices = actuator_indices.to(device)
            y = y.to(device)

            output = model(x, actuator_indices)

            loss = loss_fn(output, y)

            total_valid_loss += loss.item()

    return total_valid_loss / len(valid_loader)


def train(
    train_loader: DataLoader, valid_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer, epochs: int,
    early_stop: int, device: str
):
    best_epoch = -1
    best_train_loss_with_best_epoch = float('inf')
    best_valid_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    no_improvement_count = 0

    for epoch in tqdm(range(epochs), desc='Train'):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)

        valid_loss = valid_epoch(valid_loader, model, loss_fn, device)

        Logger.info(f'Epoch {epoch + 1}:')
        Logger.info(f' - Train loss: {train_loss}')
        Logger.info(f' - Valid loss: {valid_loss}')

        if valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            best_train_loss_with_best_epoch = train_loss
            best_valid_loss = valid_loss
            # Some metrics......
            best_model_weights = copy.deepcopy(model.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop:
            Logger.info('Early stop.')
            break

    torch.save(best_model_weights, f'saves/model_{best_epoch}.pth')

    Logger.info(f'Best epoch: {best_epoch}')
    Logger.info(f' - Train loss: {best_train_loss_with_best_epoch}')
    Logger.info(f' - Valid loss: {best_valid_loss}')
    Logger.info(f'Model save to saves/model_{best_epoch}.pth')


def ensure_dir_exists() -> None:
    Path('saves').mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    seed = 42
    epochs = 1000
    early_stop = 20
    device = 'cuda'

    Logger.init('first_train.log')

    torch.manual_seed(seed)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(seed)

    model = ICS(sequence_length=15, hidden_dim=128, middle_output_dim=64, k=3, num_graph=3, device=device).to(device)

    loss_fn = L1Loss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, epochs, early_stop, device)
