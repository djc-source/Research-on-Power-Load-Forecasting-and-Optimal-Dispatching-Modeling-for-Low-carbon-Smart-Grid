import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Union

def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:

    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mean_true = np.mean(y_true)
    nrmse = rmse / mean_true * 100 if mean_true != 0 else float('inf')

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'nRMSE': nrmse
    }

    return metrics

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:

    formatted_lines = []
    formatted_lines.append("=" * 50)
    formatted_lines.append("模型评估指标")
    formatted_lines.append("=" * 50)

    for metric_name, value in metrics.items():
        if metric_name in ['MAPE', 'nRMSE']:
            formatted_lines.append(f"{metric_name:>8}: {value:.{precision}f}%")
        else:
            formatted_lines.append(f"{metric_name:>8}: {value:.{precision}f}")

    formatted_lines.append("=" * 50)

    return "\n".join(formatted_lines)

def compare_models(model_metrics: Dict[str, Dict[str, float]]) -> str:

    if not model_metrics:
        return "无模型数据可比较"

    all_metrics = set()
    for metrics in model_metrics.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(list(all_metrics))

    lines = []
    lines.append("=" * 80)
    lines.append("模型性能比较")
    lines.append("=" * 80)

    header = f"{'指标':<10}"
    for model_name in model_metrics.keys():
        header += f"{model_name:>15}"
    header += f"{'最佳模型':>15}"
    lines.append(header)
    lines.append("-" * 80)

    for metric in all_metrics:
        line = f"{metric:<10}"
        metric_values = {}

        for model_name, metrics in model_metrics.items():
            value = metrics.get(metric, float('nan'))
            metric_values[model_name] = value
            if metric in ['MAPE', 'RMSE', 'MAE', 'MSE', 'nRMSE']:
                line += f"{value:>14.2f}%"
            else:
                line += f"{value:>15.4f}"

        if not all(np.isnan(v) for v in metric_values.values()):
            if metric in ['MAPE', 'RMSE', 'MAE', 'MSE', 'nRMSE']:
                best_model = min(metric_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))[0]
            else:  
                best_model = max(metric_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('-inf'))[0]
            line += f"{best_model:>15}"
        else:
            line += f"{'N/A':>15}"

        lines.append(line)

    lines.append("=" * 80)

    return "\n".join(lines)

class MetricsTracker:

    def __init__(self):
        self.reset()

    def reset(self):

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}

    def update(self, train_loss: float, val_loss: float, val_metrics: Dict[str, float], epoch: int):

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_metrics.append(val_metrics.copy())

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_metrics = val_metrics.copy()

    def get_best_metrics(self) -> Dict[str, Union[float, int]]:

        result = self.best_metrics.copy()
        result['best_epoch'] = self.best_epoch
        result['best_val_loss'] = self.best_val_loss
        return result

    def get_training_history(self) -> Dict[str, list]:

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }