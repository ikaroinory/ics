import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from Logger import Logger


def _a2():
    df = pd.read_csv('data/original/SWaT_Dataset_Normal_v1.csv')

    # Set datetime index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=' %d/%m/%Y %I:%M:%S %p')
    df.set_index('Timestamp', inplace=True)

    # Remove the last five hours of data
    df = df[5 * 60 * 60 + 1:]

    df = df / df.max().abs()

    value_data_path = 'data/processed/a2_normal_values_df.pkl'
    with open(value_data_path, 'wb') as file:
        file.write(pickle.dumps(df))

    Logger.info(f'Saved A2 value data to {value_data_path}')

    actuators = ('MV', 'P', 'UV')
    actuator_indices = [i for i, column in enumerate(df.columns) if str(column).startswith(actuators)]
    actuator_indices = np.array(actuator_indices)

    actuator_index_path = 'data/processed/a2_actuator_indices_np.pkl'
    with open(actuator_index_path, 'wb') as file:
        file.write(pickle.dumps(actuator_indices))

    Logger.info(f'Saved A2 actuator indices to {actuator_index_path}')


if __name__ == '__main__':
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    Logger.init()

    _a2()
