import torch
import numpy as np
from typing import Dict, Tuple

from trainers.base_trainer import BaseTrainer
from utils.metrics import calculate_metrics

class LSTMTransformerTrainer(BaseTrainer):

    def train_epoch(self) -> float:

        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (sequences, targets) in enumerate(self.train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs, attention_weights = self.model(sequences)

            loss = self.criterion(outputs, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs, attention_weights = self.model(sequences)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        all_predictions_original = self.val_dataset.inverse_transform_target(all_predictions)
        all_targets_original = self.val_dataset.inverse_transform_target(all_targets)

        metrics = calculate_metrics(all_targets_original, all_predictions_original)

        return avg_loss, metrics

    def _model_forward(self, sequences):

        outputs, attention_weights = self.model(sequences)
        return outputs