import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from typing import Tuple, List, Optional

class LoadDataset(Dataset):

    def __init__(self, data_path: str, sequence_length: int = 96, 
                 scaler_type: str = 'standard', target_column: str = 'load',
                 use_time_features: bool = True):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.use_time_features = use_time_features

        self.data = pd.read_csv(data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.sort_values('datetime').reset_index(drop=True)

        if self.use_time_features:
            self._add_time_features()

        self.feature_columns = ['load', 'temp_avg', 'humidity', 'rain']
        if self.use_time_features:
            time_feature_cols = [
                'quarter_hour'
            ]
            self.feature_columns.extend(time_feature_cols)

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")

        self._preprocess_data()

    def _add_time_features(self):

        self.data['quarter_hour'] = ((self.data['datetime'].dt.hour * 4 + 
                                     self.data['datetime'].dt.minute // 15)) 

        print(f"时间特征添加完成:")
        print(f"- quarter_hour: {self.data['quarter_hour'].min()}-{self.data['quarter_hour'].max()}")

    def _preprocess_data(self):

        features = self.data[self.feature_columns].values

        self.scaled_features = self.scaler.fit_transform(features)

        self.sequences = []
        self.targets = []

        for i in range(len(self.scaled_features) - self.sequence_length):

            sequence = self.scaled_features[i:i + self.sequence_length]

            target = self.scaled_features[i + self.sequence_length, 0]  

            self.sequences.append(sequence)
            self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

        print(f"数据预处理完成:")
        print(f"- 特征维度: {len(self.feature_columns)}")
        print(f"- 序列形状: {self.sequences.shape}")
        print(f"- 目标形状: {self.targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

    def inverse_transform_target(self, scaled_target):

        if isinstance(scaled_target, (list, tuple)):
            scaled_array = np.array(scaled_target)
        elif isinstance(scaled_target, np.ndarray):
            scaled_array = scaled_target
        else:
            scaled_array = np.array([scaled_target])

        if scaled_array.ndim > 1:
            scaled_array = scaled_array.flatten()

        dummy = np.zeros((len(scaled_array), len(self.feature_columns)))
        dummy[:, 0] = scaled_array
        inverse_transformed = self.scaler.inverse_transform(dummy)
        return inverse_transformed[:, 0]

    def get_data_info(self):

        return {
            'data_path': self.data_path,
            'total_samples': len(self),
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'data_shape': self.sequences.shape,
            'date_range': (self.data['datetime'].min(), self.data['datetime'].max()),
            'scaler_type': type(self.scaler).__name__,
            'use_time_features': self.use_time_features
        }

def create_data_loaders(train_path: str, val_path: str, test_path: str,
                       sequence_length: int = 96, batch_size: int = 32,
                       scaler_type: str = 'standard', num_workers: int = 0,
                       use_time_features: bool = True):

    train_dataset = LoadDataset(train_path, sequence_length, scaler_type, 
                               use_time_features=use_time_features)
    val_dataset = LoadDataset(val_path, sequence_length, scaler_type,
                             use_time_features=use_time_features)
    test_dataset = LoadDataset(test_path, sequence_length, scaler_type,
                              use_time_features=use_time_features)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    return train_loader, val_loader, test_loader, datasets

def get_area_data_paths(area: str, data_dir: str = 'data/processed'):

    train_path = os.path.join(data_dir, f'{area}_train.csv')
    val_path = os.path.join(data_dir, f'{area}_val.csv')
    test_path = os.path.join(data_dir, f'{area}_test.csv')

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

    return train_path, val_path, test_path