import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

from utils.metrics import calculate_metrics, MetricsTracker, format_metrics

class BaseTrainer(ABC):

    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, test_loader: DataLoader,
                 criterion: nn.Module = None, optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 device: str = 'auto', save_dir: str = 'checkpoints'):

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset
        self.test_dataset = test_loader.dataset

        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.current_epoch = 0
        self.metrics_tracker = MetricsTracker()

        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

    @abstractmethod
    def train_epoch(self) -> float:

        pass

    @abstractmethod
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:

        pass

    def train(self, num_epochs: int, early_stopping_patience: int = 10,
              save_best: bool = True, print_every: int = 10) -> Dict:

        print("开始训练...")
        print("=" * 60)

        start_time = time.time()
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss = self.train_epoch()

            val_loss, val_metrics = self.validate_epoch()

            if val_loss < self.metrics_tracker.best_val_loss:
                patience_counter = 0
                if save_best:
                    self.save_model('best_model.pth')
            else:
                patience_counter += 1

            self.metrics_tracker.update(train_loss, val_loss, val_metrics, epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAPE: {val_metrics.get('MAPE', 0):.2f}% | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s")

            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break

        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.1f}秒")
        print(f"最佳模型在第 {self.metrics_tracker.best_epoch + 1} 轮")

        if save_best:
            self.save_model('final_model.pth')

        return self.metrics_tracker.get_training_history()

    def evaluate(self, data_loader: DataLoader = None, 
                 dataset_name: str = "Test") -> Dict[str, float]:

        if data_loader is None:
            data_loader = self.test_loader
            dataset = self.test_dataset
        elif data_loader == self.train_loader:
            dataset = self.train_dataset
        elif data_loader == self.val_loader:
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        self.model.eval()
        all_predictions = []
        all_targets = []

        print(f"正在评估 {dataset_name} 数据集...")

        with torch.no_grad():
            for batch_idx, (sequences, targets) in enumerate(data_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self._model_forward(sequences)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        all_predictions_original = dataset.inverse_transform_target(all_predictions)
        all_targets_original = dataset.inverse_transform_target(all_targets)

        metrics = calculate_metrics(all_targets_original, all_predictions_original)

        print(f"\n{dataset_name}数据集评估结果:")
        print(format_metrics(metrics))
        print(f"样本数量: {len(all_targets_original)}")
        print(f"真实值范围: {all_targets_original.min():.2f} - {all_targets_original.max():.2f}")
        print(f"预测值范围: {all_predictions_original.min():.2f} - {all_predictions_original.max():.2f}")

        return metrics

    def _model_forward(self, sequences):

        outputs = self.model(sequences)
        if isinstance(outputs, tuple):
            return outputs[0]  
        return outputs

    def save_model(self, filename: str):

        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_metrics': self.metrics_tracker.get_best_metrics(),
            'model_config': getattr(self.model, 'get_model_info', lambda: {})()
        }, filepath)
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath: str, load_optimizer: bool = True):

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"警告: 使用安全模式加载失败 ({e})，尝试兼容模式...")
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        print(f"模型已从 {filepath} 加载")

        return checkpoint.get('best_metrics', {})

    def predict(self, data_loader: DataLoader, return_attention: bool = False):

        self.model.eval()
        predictions = []
        attention_weights = [] if return_attention else None

        with torch.no_grad():
            for sequences, _ in data_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)

                if isinstance(outputs, tuple):
                    pred, attn = outputs
                    predictions.extend(pred.cpu().numpy())
                    if return_attention:
                        attention_weights.extend(attn.cpu().numpy())
                else:
                    predictions.extend(outputs.cpu().numpy())

        if return_attention and attention_weights:
            return np.array(predictions), np.array(attention_weights)
        return np.array(predictions)

    def get_model_summary(self):

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary = {
            'model_class': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'optimizer': self.optimizer.__class__.__name__,
            'criterion': self.criterion.__class__.__name__
        }

        if hasattr(self.model, 'get_model_info'):
            summary.update(self.model.get_model_info())

        return summary