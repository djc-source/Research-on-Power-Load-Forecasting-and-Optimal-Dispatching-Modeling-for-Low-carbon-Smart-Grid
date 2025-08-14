import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import warnings

from dispatching.optimal_dispatch import OptimalDispatcher
from utils.dataset import LoadDataset
from sklearn.preprocessing import StandardScaler
import pickle

class ModelPredictor:

    def __init__(self, model_path: str, model_type: str = 'lstm_transformer'):

        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_model()

    def _load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        weights_only_mode = False
        try:

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            weights_only_mode = True
        except Exception:

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            weights_only_mode = False

        if weights_only_mode:

            model_config = {}
        else:

            model_config = checkpoint.get('model_config', {})

        if self.model_type == 'lstm_baseline':
            from models.lstm_baseline import LSTMBaseline
            self.model = LSTMBaseline(
                input_dim=model_config.get('input_dim', 5),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2),
                output_dim=1,
                dropout=0.1
            )
        elif self.model_type == 'lstm_transformer':
            from models.lstm_transformer import LSTMTransformerModel
            self.model = LSTMTransformerModel(
                input_dim=model_config.get('input_dim', 5),
                lstm_hidden_dim=model_config.get('lstm_hidden_dim', 64),
                lstm_layers=model_config.get('lstm_layers', 2),
                transformer_dim=model_config.get('transformer_dim', 128),
                transformer_layers=model_config.get('transformer_layers', 2),
                num_heads=model_config.get('num_heads', 8),
                sequence_length=model_config.get('sequence_length', 96),
                output_dim=1,
                dropout=0.1
            )
        elif self.model_type == 'de_lstm_baseline':
            from models.de_lstm_baseline import DeLSTMBaseline
            self.model = DeLSTMBaseline(
                input_dim=model_config.get('input_dim', 5),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2),
                output_dim=1,
                dropout=0.1
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        if weights_only_mode:

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:

            state_dict = checkpoint['model_state_dict']

        state_dict = self._adapt_state_dict(state_dict)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print(f"成功加载模型: {self.model_path}")
        print(f"模型类型: {self.model_type}")
        print(f"加载模式: {'安全模式' if weights_only_mode else '兼容模式'}")
        if not weights_only_mode:
            print(f"模型参数: {model_config}")

    def _adapt_state_dict(self, state_dict: dict) -> dict:

        adapted_state_dict = state_dict.copy()

        if self.model_type == 'lstm_baseline':

            if 'fc.weight' in state_dict and 'fc1.weight' not in state_dict:
                print("检测到单层输出结构，转换为双层结构...")

                fc_weight = state_dict['fc.weight']  
                fc_bias = state_dict['fc.bias']      

                hidden_dim = fc_weight.shape[1]
                output_dim = fc_weight.shape[0]

                fc1_out_dim = hidden_dim // 2

                adapted_state_dict['fc1.weight'] = torch.randn(fc1_out_dim, hidden_dim) * 0.1
                adapted_state_dict['fc1.bias'] = torch.zeros(fc1_out_dim)

                adapted_state_dict['fc2.weight'] = torch.randn(output_dim, fc1_out_dim) * 0.1
                adapted_state_dict['fc2.bias'] = fc_bias  

                del adapted_state_dict['fc.weight']
                del adapted_state_dict['fc.bias']

                print(f"转换完成: fc({hidden_dim}->{output_dim}) -> fc1({hidden_dim}->{fc1_out_dim}) + fc2({fc1_out_dim}->{output_dim})")

            elif 'fc1.weight' in state_dict and hasattr(self.model, 'fc') and not hasattr(self.model, 'fc1'):
                print("检测到双层输出结构，转换为单层结构...")

                fc1_weight = state_dict['fc1.weight']  
                fc1_bias = state_dict['fc1.bias']      
                fc2_weight = state_dict['fc2.weight']  
                fc2_bias = state_dict['fc2.bias']      

                hidden_dim = fc1_weight.shape[1]
                fc1_out_dim = fc1_weight.shape[0]
                output_dim = fc2_weight.shape[0]

                fc_weight = torch.zeros(output_dim, hidden_dim)
                fc_weight[:, :fc1_out_dim] = fc2_weight

                adapted_state_dict['fc.weight'] = fc_weight
                adapted_state_dict['fc.bias'] = fc2_bias

                del adapted_state_dict['fc1.weight']
                del adapted_state_dict['fc1.bias'] 
                del adapted_state_dict['fc2.weight']
                del adapted_state_dict['fc2.bias']

                print(f"转换完成: fc1({hidden_dim}->{fc1_out_dim}) + fc2({fc1_out_dim}->{output_dim}) -> fc({hidden_dim}->{output_dim})")

            elif 'output_layer.weight' in state_dict and 'fc1.weight' not in state_dict:
                print("检测到de_lstm_baseline的output_layer结构，转换为双层结构...")

                output_weight = state_dict['output_layer.weight']  
                output_bias = state_dict['output_layer.bias']      

                hidden_dim = output_weight.shape[1]
                output_dim = output_weight.shape[0]

                fc1_out_dim = hidden_dim // 2

                adapted_state_dict['fc1.weight'] = torch.randn(fc1_out_dim, hidden_dim) * 0.1
                adapted_state_dict['fc1.bias'] = torch.zeros(fc1_out_dim)

                adapted_state_dict['fc2.weight'] = torch.randn(output_dim, fc1_out_dim) * 0.1
                adapted_state_dict['fc2.bias'] = output_bias  

                del adapted_state_dict['output_layer.weight']
                del adapted_state_dict['output_layer.bias']

                print(f"转换完成: output_layer({hidden_dim}->{output_dim}) -> fc1({hidden_dim}->{fc1_out_dim}) + fc2({fc1_out_dim}->{output_dim})")

        elif self.model_type == 'de_lstm_baseline':

            if 'fc1.weight' in state_dict and 'output_layer.weight' not in state_dict:
                print("检测到双层结构，转换为de_lstm_baseline的output_layer结构...")

                fc1_weight = state_dict['fc1.weight']  
                fc1_bias = state_dict['fc1.bias']      
                fc2_weight = state_dict['fc2.weight']  
                fc2_bias = state_dict['fc2.bias']      

                hidden_dim = fc1_weight.shape[1]
                fc1_out_dim = fc1_weight.shape[0]
                output_dim = fc2_weight.shape[0]

                output_weight = torch.zeros(output_dim, hidden_dim)
                output_weight[:, :fc1_out_dim] = fc2_weight

                adapted_state_dict['output_layer.weight'] = output_weight
                adapted_state_dict['output_layer.bias'] = fc2_bias

                del adapted_state_dict['fc1.weight']
                del adapted_state_dict['fc1.bias'] 
                del adapted_state_dict['fc2.weight']
                del adapted_state_dict['fc2.bias']

                print(f"转换完成: fc1({hidden_dim}->{fc1_out_dim}) + fc2({fc1_out_dim}->{output_dim}) -> output_layer({hidden_dim}->{output_dim})")

        return adapted_state_dict

    def predict(self, input_sequence: np.ndarray) -> float:

        self.model.eval()
        with torch.no_grad():

            if len(input_sequence.shape) == 2:
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  
            else:
                input_tensor = torch.FloatTensor(input_sequence)

            input_tensor = input_tensor.to(self.device)

            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]  

            if isinstance(output, torch.Tensor):
                prediction = output.cpu().numpy().flatten()[0]
            elif isinstance(output, (list, tuple)):
                prediction = float(output[0])
            else:
                prediction = float(output)

        return prediction

    def predict_batch(self, input_sequences: np.ndarray) -> np.ndarray:

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_sequences).to(self.device)
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            if isinstance(output, torch.Tensor):
                predictions = output.cpu().numpy().flatten()
            elif isinstance(output, (list, tuple)):
                predictions = np.array([float(x) for x in output])
            else:
                predictions = np.array([float(output)])

        return predictions

class RollingOptimizer:

    def __init__(self, 
                 predictor: ModelPredictor,
                 dispatcher: OptimalDispatcher,
                 data_config: Dict,
                 window_size: int = 96,
                 step_size: int = 1):

        self.predictor = predictor
        self.dispatcher = dispatcher
        self.data_config = data_config
        self.window_size = window_size
        self.step_size = step_size

        self.beta = data_config.get('feedback_coefficient', 0.5)  

        self.prediction_history = []
        self.actual_history = []
        self.dispatch_history = []
        self.error_history = []

        self._load_dataset()

    def _load_dataset(self):

        area = self.data_config.get('area', 'Area1')
        data_dir = self.data_config.get('data_dir', 'data/processed')

        test_path = os.path.join(data_dir, f'{area}_test.csv')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试数据文件不存在: {test_path}")

        self.dataset = LoadDataset(
            data_path=test_path,
            sequence_length=96,
            scaler_type='standard',
            use_time_features=True
        )

        print(f"加载数据集: {test_path}")
        print(f"数据集大小: {len(self.dataset)} 样本")

        self.raw_data = pd.read_csv(test_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"负荷数据范围: {self.raw_data['load'].min():.2f} - {self.raw_data['load'].max():.2f} MW")

    def calculate_compensation(self, prediction_error: float) -> float:

        return self.beta * prediction_error

    def rolling_optimize(self, 
                        start_idx: int = 0, 
                        total_steps: int = 100,
                        save_results: bool = True) -> Dict:

        print("=" * 60)
        print("开始滚动优化")
        print("=" * 60)
        print(f"窗口大小: {self.window_size} 时段 (24小时)")
        print(f"步长: {self.step_size} 时段")
        print(f"总步数: {total_steps}")
        print(f"反馈系数 β: {self.beta}")
        print()

        results = {
            'predictions': [],
            'actuals': [],
            'dispatches': [],
            'errors': [],
            'compensations': [],
            'costs': [],
            'metrics': [],
            'timestamps': []
        }

        previous_error = 0.0

        for step in range(total_steps):
            current_idx = start_idx + step * self.step_size

            if current_idx + self.window_size >= len(self.raw_data):
                print(f"到达数据末尾，在第 {step} 步停止")
                break

            try:

                predictions = self._generate_forecast(current_idx)

                compensation = self.calculate_compensation(previous_error)
                compensations = np.full(self.window_size, compensation)

                dispatch_result = self._execute_dispatch(predictions, compensations)

                actual_load = self._get_actual_load(current_idx)

                prediction_error = actual_load - predictions[0]  

                results['predictions'].append(predictions[0])
                results['actuals'].append(actual_load)
                results['dispatches'].append(dispatch_result)
                results['errors'].append(prediction_error)
                results['compensations'].append(compensation)
                results['costs'].append(dispatch_result['total_cost'])

                step_metrics = self.dispatcher.calculate_metrics(
                    dispatch_result, 
                    np.array([actual_load])
                )
                results['metrics'].append(step_metrics)

                results['timestamps'].append(current_idx)

                self.prediction_history.append(predictions[0])
                self.actual_history.append(actual_load)
                self.dispatch_history.append(dispatch_result)
                self.error_history.append(prediction_error)

                previous_error = prediction_error

                if (step + 1) % 10 == 0 or step == 0:
                    print(f"Step [{step+1:3d}/{total_steps}] "
                          f"Pred: {predictions[0]:7.2f} MW | "
                          f"Actual: {actual_load:7.2f} MW | "
                          f"Error: {prediction_error:6.2f} MW | "
                          f"Cost: {dispatch_result['total_cost']:8.2f} ¥")

            except Exception as e:
                print(f"第 {step} 步执行失败: {e}")
                continue

        overall_metrics = self._calculate_overall_metrics(results)
        results['overall_metrics'] = overall_metrics

        print("\n滚动优化完成!")
        print("=" * 60)
        print("总体指标:")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        if save_results:
            self._save_results(results)

        return results

    def _generate_forecast(self, start_idx: int) -> np.ndarray:

        predictions = []

        if start_idx < 96:

            history_data = self.raw_data.iloc[0:start_idx+96].copy()
        else:
            history_data = self.raw_data.iloc[start_idx-96:start_idx].copy()

        history_data['datetime'] = pd.to_datetime(history_data['datetime'])
        history_data['quarter_hour'] = ((history_data['datetime'].dt.hour * 4 + 
                                       history_data['datetime'].dt.minute // 15))

        feature_cols = ['load', 'temp_avg', 'humidity', 'rain', 'quarter_hour']
        input_features = history_data[feature_cols].values

        input_features_scaled = self.dataset.scaler.transform(input_features)

        current_sequence = input_features_scaled[-96:]  

        for i in range(self.window_size):

            pred_scaled = self.predictor.predict(current_sequence)

            pred_original = self.dataset.inverse_transform_target([pred_scaled])[0]
            predictions.append(pred_original)

            if i < self.window_size - 1:

                next_idx = start_idx + i + 1
                if next_idx < len(self.raw_data):
                    next_row = self.raw_data.iloc[next_idx].copy()
                    next_row['load'] = pred_original  

                    next_datetime = pd.to_datetime(next_row['datetime'])
                    next_quarter_hour = next_datetime.hour * 4 + next_datetime.minute // 15

                    next_features = np.array([
                        pred_original,  
                        next_row['temp_avg'],
                        next_row['humidity'], 
                        next_row['rain'],
                        next_quarter_hour
                    ])

                    next_features_scaled = self.dataset.scaler.transform([next_features])[0]

                    current_sequence = np.vstack([current_sequence[1:], next_features_scaled])
                else:

                    last_features = current_sequence[-1].copy()
                    last_features[0] = self.dataset.scaler.transform([[pred_original, 0, 0, 0, 0]])[0][0]
                    current_sequence = np.vstack([current_sequence[1:], last_features])

        return np.array(predictions)

    def _execute_dispatch(self, load_forecast: np.ndarray, 
                         compensations: np.ndarray) -> Dict:

        model = self.dispatcher.create_model(load_forecast, len(load_forecast))

        solution = self.dispatcher.solve(model, compensations)

        return solution

    def _get_actual_load(self, idx: int) -> float:

        if idx < len(self.raw_data):
            return self.raw_data.iloc[idx]['load']
        else:
            return 0.0

    def _calculate_overall_metrics(self, results: Dict) -> Dict:

        if not results['predictions']:
            return {}

        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        errors = np.array(results['errors'])
        costs = np.array(results['costs'])

        metrics = {
            'total_steps': len(predictions),
            'avg_prediction_error': np.mean(np.abs(errors)),
            'max_prediction_error': np.max(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(np.abs(errors / actuals)) * 100,
            'total_cost': np.sum(costs),
            'avg_cost_per_step': np.mean(costs),
            'cost_std': np.std(costs)
        }

        return metrics

    def _save_results(self, results: Dict):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = f"dispatching/results/rolling_optimization_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        results_file = os.path.join(save_dir, 'rolling_results.json')

        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, list):
                json_results[key] = value
            else:
                json_results[key] = value

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        summary_file = os.path.join(save_dir, 'summary_metrics.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results['overall_metrics'], f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {save_dir}")
        print(f"详细结果: {results_file}")
        print(f"摘要指标: {summary_file}")

def create_system_config(area: str = 'Area1') -> Dict:

    if area == 'Area1':

        peak_load = 12000  
        min_load = 2200    

        total_capacity = int(peak_load * 1.25)  

        generators = {
            'Nuclear_1': {
                'p_min': int(total_capacity * 0.05 * 0.8), 
                'p_max': int(total_capacity * 0.05),       
                'cost_b': 5, 'cost_c': 30,    
                'ramp_up': 30, 'ramp_down': 30,    
                'emission_factor': 0.0  
            },
            'Hydro_1': {
                'p_min': int(total_capacity * 0.07 * 0.3), 
                'p_max': int(total_capacity * 0.07),       
                'cost_b': 3, 'cost_c': 15,    
                'ramp_up': 600, 'ramp_down': 600,
                'emission_factor': 0.0  
            },
            'Wind_1': {
                'p_min': 0,  
                'p_max': int(total_capacity * 0.20),        
                'cost_b': 2, 'cost_c': 5,     
                'ramp_up': 1000, 'ramp_down': 1000,  
                'emission_factor': 0.0  
            },
            'Solar_1': {
                'p_min': 0,  
                'p_max': int(total_capacity * 0.15),        
                'cost_b': 1, 'cost_c': 3,     
                'ramp_up': 2000, 'ramp_down': 2000,  
                'emission_factor': 0.0  
            },
            'Gas_1': {
                'p_min': int(total_capacity * 0.08 * 0.2), 
                'p_max': int(total_capacity * 0.08),       
                'cost_b': 40, 'cost_c': 100,  
                'ramp_up': 400, 'ramp_down': 400,
                'emission_factor': 0.4   
            },
            'Coal_1': {
                'p_min': int(total_capacity * 0.45 * 0.2),  
                'p_max': int(total_capacity * 0.45),        
                'cost_b': 50, 'cost_c': 200,  
                'ramp_up': 200, 'ramp_down': 200,
                'emission_factor': 0.85  
            }
        }

        storage_capacity_ratio = 0.15  

        storage = {
            'Battery_1': {
                'p_max': int(total_capacity * storage_capacity_ratio * 0.6),  
                'soc_min': int(total_capacity * storage_capacity_ratio * 0.6 * 2 * 0.1),   
                'soc_max': int(total_capacity * storage_capacity_ratio * 0.6 * 2),         
                'initial_soc': int(total_capacity * storage_capacity_ratio * 0.6 * 2 * 0.5), 
                'efficiency': 0.92,  
                'cost_per_mwh': 15   
            },
            'PumpHydro_1': {
                'p_max': int(total_capacity * storage_capacity_ratio * 0.4),  
                'soc_min': int(total_capacity * storage_capacity_ratio * 0.4 * 6 * 0.2),   
                'soc_max': int(total_capacity * storage_capacity_ratio * 0.4 * 6),         
                'initial_soc': int(total_capacity * storage_capacity_ratio * 0.4 * 6 * 0.6), 
                'efficiency': 0.78,  
                'cost_per_mwh': 8    
            }
        }

        constraints = {
            'carbon_limit': peak_load * 0.48 * 24,  
            'carbon_penalty': 50  
        }

    else:  

        max_load = 10000
        generators = {
            'Coal_1': {
                'p_min': 400, 'p_max': 2500,
                'cost_b': 28, 'cost_c': 160,
                'ramp_up': 180, 'ramp_down': 180,
                'emission_factor': 0.85
            },
            'Gas_1': {
                'p_min': 150, 'p_max': 1800,
                'cost_b': 38, 'cost_c': 85,
                'ramp_up': 350, 'ramp_down': 350,
                'emission_factor': 0.42
            },
            'Hydro_1': {
                'p_min': 80, 'p_max': 1200,
                'cost_b': 6, 'cost_c': 25,
                'ramp_up': 500, 'ramp_down': 500,
                'emission_factor': 0.0
            }
        }

        storage = {
            'Battery_1': {
                'p_max': 400,
                'soc_min': 80, 'soc_max': 1500,
                'initial_soc': 800,
                'efficiency': 0.94,
                'cost_per_mwh': 12
            }
        }

    config = {
        'generators': generators,
        'storage': storage,
        'constraints': constraints,
        'solver': {
            'name': 'glpk',
            'options': {

            }
        }
    }

    return config 

class EnhancedRollingOptimizer(RollingOptimizer):

    def __init__(self, predictor: ModelPredictor, dispatcher: OptimalDispatcher, 
                 data_config: Dict, window_size: int = 96, step_size: int = 1,
                 prediction_method: str = 'multi_step'):

        super().__init__(predictor, dispatcher, data_config, window_size, step_size)
        self.prediction_method = prediction_method

    def _generate_forecast(self, start_idx: int) -> np.ndarray:

        if self.prediction_method == 'single_step':
            return self._generate_single_step_forecast(start_idx)
        elif self.prediction_method == 'traditional':
            return self._generate_traditional_forecast(start_idx)
        else:
            return super()._generate_forecast(start_idx)

    def _generate_single_step_forecast(self, start_idx: int) -> np.ndarray:

        predictions = []
        feature_cols = ['load', 'temp_avg', 'humidity', 'rain']

        for i in range(self.window_size):
            current_idx = start_idx + i

            if current_idx < 96:

                seq_start = 0
                seq_end = max(1, current_idx)  
            else:

                seq_start = current_idx - 96
                seq_end = current_idx

            input_data = self.raw_data.iloc[seq_start:seq_end].copy()

            if len(input_data) < 96:
                last_row = input_data.iloc[-1] if len(input_data) > 0 else self.raw_data.iloc[0]
                padding_size = 96 - len(input_data)
                padding_data = pd.DataFrame([last_row] * padding_size, columns=input_data.columns)
                input_data = pd.concat([input_data, padding_data], ignore_index=True)

            input_data['datetime'] = pd.to_datetime(input_data['datetime'])
            input_data['quarter_hour'] = ((input_data['datetime'].dt.hour * 4 + 
                                         input_data['datetime'].dt.minute // 15))

            feature_cols_with_time = ['load', 'temp_avg', 'humidity', 'rain', 'quarter_hour']
            input_features = input_data[feature_cols_with_time].values
            input_features_scaled = self.dataset.scaler.transform(input_features)

            try:
                pred_scaled = self.predictor.predict(input_features_scaled)
                pred_original = self.dataset.inverse_transform_target([pred_scaled])[0]

                if i > 0 and len(self.actual_history) > 0:
                    recent_actual = self.actual_history[-min(3, len(self.actual_history)):]
                    if len(recent_actual) >= 2:
                        trend = (recent_actual[-1] - recent_actual[0]) / len(recent_actual)
                        pred_original += trend * 0.5  

                pred_original = max(1000, min(20000, pred_original))
                predictions.append(pred_original)

            except Exception as e:

                if len(predictions) > 0:
                    predictions.append(predictions[-1])
                elif len(self.actual_history) > 0:
                    predictions.append(self.actual_history[-1])
                else:
                    predictions.append(7000.0)

        return np.array(predictions)

    def _generate_traditional_forecast(self, start_idx: int) -> np.ndarray:

        predictions = []

        lookback_hours = min(7 * 24 * 4, start_idx)  
        if start_idx >= lookback_hours:
            historical_data = self.raw_data.iloc[start_idx-lookback_hours:start_idx]['load'].values
        else:
            historical_data = self.raw_data.iloc[0:start_idx]['load'].values

        if len(historical_data) > 0:
            recent_avg = np.mean(historical_data[-96:]) if len(historical_data) >= 96 else np.mean(historical_data)
            recent_trend = 0.0
            if len(historical_data) >= 8:

                x = np.arange(len(historical_data[-8:]))
                y = historical_data[-8:]
                trend_coef = np.polyfit(x, y, 1)[0]
                recent_trend = trend_coef
        else:
            recent_avg = 7000.0  
            recent_trend = 0.0

        for i in range(self.window_size):

            hour_of_day = ((start_idx + i) % (24 * 4)) / 4.0  

            if 6 <= hour_of_day <= 9 or 18 <= hour_of_day <= 21:

                pattern_factor = 1.2
            elif 0 <= hour_of_day <= 6 or 22 <= hour_of_day <= 24:

                pattern_factor = 0.8
            else:

                pattern_factor = 1.0

            noise_factor = 1.0 + np.random.normal(0, 0.05)  

            base_prediction = recent_avg + recent_trend * i
            adjusted_prediction = base_prediction * pattern_factor * noise_factor

            adjusted_prediction = max(2000, min(15000, adjusted_prediction))
            predictions.append(adjusted_prediction)

        return np.array(predictions)