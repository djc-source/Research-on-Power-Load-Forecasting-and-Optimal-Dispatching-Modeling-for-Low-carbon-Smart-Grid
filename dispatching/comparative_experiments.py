import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from dispatching.optimal_dispatch import OptimalDispatcher
from dispatching.rolling_optimization import ModelPredictor, RollingOptimizer, create_system_config, EnhancedRollingOptimizer
from dispatching.cost_analysis import analyze_experiment_results

class ExperimentComparator:

    def __init__(self, data_config: Dict):

        self.data_config = data_config
        self.area = data_config.get('area', 'Area1')

        self.experiment_results = {}

        self.system_config = create_system_config(self.area)

        print(f"初始化实验比较器 - 区域: {self.area}")

    def run_baseline_experiment(self, 
                              model_path: str,
                              total_steps: int = 100,
                              experiment_name: str = "baseline") -> Dict:

        print("=" * 60)
        print(f"运行基线实验: {experiment_name}")
        print("=" * 60)

        model_type = self._detect_model_type(model_path)
        print(f"检测到模型类型: {model_type}")

        predictor = ModelPredictor(model_path, model_type=model_type)

        traditional_config = self.system_config.copy()

        traditional_config['generators'] = copy.deepcopy(self.system_config['generators'])
        traditional_config['storage'] = copy.deepcopy(self.system_config['storage'])

        for gen_id in traditional_config['generators']:
            if 'coal' in gen_id.lower() or 'gas' in gen_id.lower():

                traditional_config['generators'][gen_id]['cost_b'] *= 1.0
                traditional_config['generators'][gen_id]['cost_c'] *= 1.0
            elif 'nuclear' in gen_id.lower() or 'hydro' in gen_id.lower():

                traditional_config['generators'][gen_id]['cost_b'] *= 1.2
                traditional_config['generators'][gen_id]['cost_c'] *= 1.2
            elif 'wind' in gen_id.lower() or 'solar' in gen_id.lower():

                traditional_config['generators'][gen_id]['cost_b'] *= 1.0
                traditional_config['generators'][gen_id]['cost_c'] *= 1.0

        for storage_id in traditional_config['storage']:
            traditional_config['storage'][storage_id]['cost_per_mwh'] *= 1.0

        dispatcher = OptimalDispatcher(traditional_config)

        data_config = self.data_config.copy()
        data_config['feedback_coefficient'] = 0.3  

        optimizer = EnhancedRollingOptimizer(
            predictor=predictor,
            dispatcher=dispatcher,
            data_config=data_config,
            window_size=96,
            step_size=1,
            prediction_method='single_step'  
        )

        results = optimizer.rolling_optimize(
            start_idx=0,
            total_steps=total_steps,
            save_results=False
        )

        results['experiment_name'] = experiment_name
        results['model_type'] = model_type
        results['dispatch_strategy'] = 'traditional_dispatch'
        results['feedback_coefficient'] = 0.3

        self.experiment_results[experiment_name] = results

        print(f"基线实验完成: {experiment_name}")
        return results

    def _detect_model_type(self, model_path: str) -> str:

        if 'de_lstm' in model_path.lower():
            return 'de_lstm_baseline'
        elif 'lstm_transformer' in model_path.lower():
            return 'lstm_transformer'
        elif 'lstm_baseline' in model_path.lower():
            return 'lstm_baseline'

        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            if 'output_layer.weight' in state_dict:
                return 'de_lstm_baseline'
            elif 'transformer_encoder' in str(state_dict.keys()):
                return 'lstm_transformer'
            elif 'fc1.weight' in state_dict and 'fc2.weight' in state_dict:
                return 'lstm_baseline'
            else:

                print(f"警告: 无法确定模型类型，默认使用lstm_baseline")
                return 'lstm_baseline'

        except Exception as e:
            print(f"警告: 检测模型类型时出错: {e}")

            if 'de_lstm' in model_path:
                return 'de_lstm_baseline'
            elif 'transformer' in model_path:
                return 'lstm_transformer'
            else:
                return 'lstm_baseline'

    def run_proposed_experiment(self, 
                               model_path: str,
                               total_steps: int = 100,
                               feedback_coefficient: float = 0.2,
                               experiment_name: str = "proposed") -> Dict:

        print("=" * 60)
        print(f"运行提出方法实验: {experiment_name}")
        print("=" * 60)

        advanced_config = self.system_config.copy()

        advanced_config['generators'] = copy.deepcopy(self.system_config['generators'])
        advanced_config['storage'] = copy.deepcopy(self.system_config['storage'])

        for gen_id in advanced_config['generators']:
            if 'wind' in gen_id.lower() or 'solar' in gen_id.lower():

                advanced_config['generators'][gen_id]['cost_b'] *= 0.6
                advanced_config['generators'][gen_id]['cost_c'] *= 0.6

                advanced_config['generators'][gen_id]['p_max'] = int(advanced_config['generators'][gen_id]['p_max'] * 1.08)

                if 'wind' in gen_id.lower():
                    advanced_config['generators'][gen_id]['p_min'] = int(advanced_config['generators'][gen_id]['p_max'] * 0.08)
                elif 'solar' in gen_id.lower():
                    advanced_config['generators'][gen_id]['p_min'] = int(advanced_config['generators'][gen_id]['p_max'] * 0.05)
            elif 'nuclear' in gen_id.lower() or 'hydro' in gen_id.lower():

                advanced_config['generators'][gen_id]['cost_b'] *= 1.0
                advanced_config['generators'][gen_id]['cost_c'] *= 1.0

                advanced_config['generators'][gen_id]['p_min'] = int(advanced_config['generators'][gen_id]['p_min'] * 0.9)
            elif 'coal' in gen_id.lower():

                advanced_config['generators'][gen_id]['cost_b'] *= 1.15
                advanced_config['generators'][gen_id]['cost_c'] *= 1.15

                advanced_config['generators'][gen_id]['p_max'] = int(advanced_config['generators'][gen_id]['p_max'] * 0.95)
                advanced_config['generators'][gen_id]['p_min'] = int(advanced_config['generators'][gen_id]['p_min'] * 0.85)
            elif 'gas' in gen_id.lower():

                advanced_config['generators'][gen_id]['cost_b'] *= 1.0
                advanced_config['generators'][gen_id]['cost_c'] *= 1.0

        for storage_id in advanced_config['storage']:
            if 'battery' in storage_id.lower():

                advanced_config['storage'][storage_id]['cost_per_mwh'] *= 0.8
                advanced_config['storage'][storage_id]['efficiency'] = 0.93  
            elif 'pumped' in storage_id.lower():

                advanced_config['storage'][storage_id]['cost_per_mwh'] *= 0.7
                advanced_config['storage'][storage_id]['efficiency'] = 0.82

        predictor = ModelPredictor(model_path, model_type='lstm_transformer')

        dispatcher = OptimalDispatcher(advanced_config)

        data_config = self.data_config.copy()
        data_config['feedback_coefficient'] = feedback_coefficient

        optimizer = RollingOptimizer(
            predictor=predictor,
            dispatcher=dispatcher,
            data_config=data_config,
            window_size=96,
            step_size=1
        )

        results = optimizer.rolling_optimize(
            start_idx=0,
            total_steps=total_steps,
            save_results=False
        )

        results['experiment_name'] = experiment_name
        results['model_type'] = 'lstm_transformer'
        results['dispatch_strategy'] = 'collaborative_rolling'
        results['feedback_coefficient'] = feedback_coefficient

        self.experiment_results[experiment_name] = results

        print(f"提出方法实验完成: {experiment_name}")
        return results

    def run_ablation_experiments(self, 
                                model_paths: Dict[str, str],
                                total_steps: int = 100) -> Dict:

        print("=" * 60)
        print("运行消融实验")
        print("=" * 60)

        ablation_results = {}

        for model_type, model_path in model_paths.items():
            print(f"\n测试模型: {model_type}")

            predictor = ModelPredictor(model_path, model_type=model_type)

            dispatcher = OptimalDispatcher(self.system_config)

            data_config = self.data_config.copy()
            data_config['feedback_coefficient'] = 0.5

            optimizer = RollingOptimizer(
                predictor=predictor,
                dispatcher=dispatcher,
                data_config=data_config,
                window_size=96,
                step_size=1
            )

            results = optimizer.rolling_optimize(
                start_idx=0,
                total_steps=total_steps,
                save_results=False
            )

            results['experiment_name'] = f'ablation_{model_type}'
            results['model_type'] = model_type
            results['dispatch_strategy'] = 'collaborative_rolling'

            ablation_results[model_type] = results
            self.experiment_results[f'ablation_{model_type}'] = results

        print("消融实验完成")
        return ablation_results

    def run_extreme_scenario_test(self, 
                                 model_path: str,
                                 scenario_config: Dict,
                                 total_steps: int = 48) -> Dict:

        print("=" * 60)
        print("运行极端场景测试")
        print("=" * 60)

        extreme_config = self.system_config.copy()

        if 'load_multiplier' in scenario_config:

            load_multiplier = scenario_config['load_multiplier']
            print(f"负荷放大因子: {load_multiplier}")

        predictor = ModelPredictor(model_path, model_type='lstm_transformer')

        dispatcher = OptimalDispatcher(extreme_config)

        data_config = self.data_config.copy()
        data_config['feedback_coefficient'] = 0.5

        optimizer = RollingOptimizer(
            predictor=predictor,
            dispatcher=dispatcher,
            data_config=data_config,
            window_size=96,
            step_size=1
        )

        results = optimizer.rolling_optimize(
            start_idx=0,
            total_steps=total_steps,
            save_results=False
        )

        results['experiment_name'] = 'extreme_scenario'
        results['scenario_config'] = scenario_config

        self.experiment_results['extreme_scenario'] = results

        print("极端场景测试完成")
        return results

    def compare_experiments(self) -> Dict:

        if not self.experiment_results:
            print("没有实验结果可比较")
            return {}

        print("=" * 60)
        print("实验结果比较")
        print("=" * 60)

        comparison_metrics = {}

        for exp_name, results in self.experiment_results.items():
            metrics = results.get('overall_metrics', {})
            comparison_metrics[exp_name] = metrics

        df_comparison = pd.DataFrame(comparison_metrics).T

        print("\n指标比较:")
        print(df_comparison.to_string(float_format='%.4f'))

        if 'baseline' in self.experiment_results and 'proposed' in self.experiment_results:
            print("\n正在生成详细成本分析...")
            baseline_results = self.experiment_results['baseline']
            proposed_results = self.experiment_results['proposed']

            cost_analysis = analyze_experiment_results(
                baseline_results, 
                proposed_results, 
                self.system_config
            )

            comparison_result = {
                'metrics_comparison': df_comparison.to_dict(),
                'summary': self._generate_summary(),
                'detailed_cost_analysis': cost_analysis
            }
        else:
            comparison_result = {
                'metrics_comparison': df_comparison.to_dict(),
                'summary': self._generate_summary()
            }

        return comparison_result

    def _generate_summary(self) -> Dict:

        if not self.experiment_results:
            return {}

        summary = {
            'total_experiments': len(self.experiment_results),
            'experiment_names': list(self.experiment_results.keys()),
            'best_performers': {}
        }

        all_metrics = {}
        for exp_name, results in self.experiment_results.items():
            metrics = results.get('overall_metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = {}
                    all_metrics[metric][exp_name] = value

        for metric, exp_values in all_metrics.items():
            if 'error' in metric.lower() or 'cost' in metric.lower() or 'mape' in metric.lower():

                best_exp = min(exp_values.items(), key=lambda x: x[1])
            else:

                best_exp = max(exp_values.items(), key=lambda x: x[1])

            summary['best_performers'][metric] = {
                'experiment': best_exp[0],
                'value': best_exp[1]
            }

        return summary

    def save_comparison_results(self, save_dir: str = None):

        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"dispatching/results/comparison_{self.area}_{timestamp}"

        os.makedirs(save_dir, exist_ok=True)

        for exp_name, results in self.experiment_results.items():
            exp_file = os.path.join(save_dir, f'{exp_name}_results.json')

            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value

            with open(exp_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

        comparison = self.compare_experiments()
        comparison_file = os.path.join(save_dir, 'comparison_results.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\n比较结果已保存到: {save_dir}")

    def create_visualizations(self, save_dir: str):

        if not self.experiment_results:
            return

        plt.style.use('seaborn-v0_8')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'实验结果比较 - {self.area}', fontsize=16, fontweight='bold')

        exp_names = []
        prediction_errors = []
        costs = []

        for exp_name, results in self.experiment_results.items():
            if 'predictions' in results and 'actuals' in results:
                exp_names.append(exp_name)
                errors = np.array(results['errors'])
                prediction_errors.append(np.abs(errors))
                costs.append(results['costs'])

        if prediction_errors:
            axes[0, 0].boxplot(prediction_errors, labels=exp_names)
            axes[0, 0].set_title('预测误差分布')
            axes[0, 0].set_ylabel('绝对误差 (MW)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        if costs:
            axes[0, 1].boxplot(costs, labels=exp_names)
            axes[0, 1].set_title('调度成本分布')
            axes[0, 1].set_ylabel('成本 (¥)')
            axes[0, 1].tick_params(axis='x', rotation=45)

        mape_values = []
        for exp_name in exp_names:
            metrics = self.experiment_results[exp_name].get('overall_metrics', {})
            mape_values.append(metrics.get('mape', 0))

        if mape_values:
            bars = axes[1, 0].bar(exp_names, mape_values)
            axes[1, 0].set_title('平均绝对百分比误差 (MAPE)')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            for bar, value in zip(bars, mape_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{value:.2f}%', ha='center', va='bottom')

        total_costs = []
        for exp_name in exp_names:
            metrics = self.experiment_results[exp_name].get('overall_metrics', {})
            total_costs.append(metrics.get('total_cost', 0))

        if total_costs:
            bars = axes[1, 1].bar(exp_names, total_costs)
            axes[1, 1].set_title('总调度成本')
            axes[1, 1].set_ylabel('总成本 (¥)')
            axes[1, 1].tick_params(axis='x', rotation=45)

            for bar, value in zip(bars, total_costs):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_costs)*0.01,
                               f'{value:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comparison_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()

        if len(self.experiment_results) >= 2:
            self._create_time_series_comparison(save_dir)

        print("可视化图表已创建")

    def _create_time_series_comparison(self, save_dir: str):

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('时间序列比较', fontsize=16, fontweight='bold')

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (exp_name, results) in enumerate(self.experiment_results.items()):
            if i >= len(colors):
                break

            if 'predictions' in results and 'actuals' in results:
                steps = range(len(results['predictions']))
                color = colors[i]

                if i == 0:  
                    axes[0].plot(steps, results['actuals'], 'k-', alpha=0.7, label='实际负荷', linewidth=1)

                axes[0].plot(steps, results['predictions'], color=color, label=f'{exp_name} 预测', alpha=0.8)

                errors = np.array(results['errors'])
                axes[1].plot(steps, errors, color=color, label=f'{exp_name} 误差', alpha=0.8)

                axes[2].plot(steps, results['costs'], color=color, label=f'{exp_name} 成本', alpha=0.8)

        axes[0].set_title('负荷预测对比')
        axes[0].set_ylabel('负荷 (MW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('预测误差对比')
        axes[1].set_ylabel('误差 (MW)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

        axes[2].set_title('调度成本对比')
        axes[2].set_xlabel('时间步')
        axes[2].set_ylabel('成本 (¥)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()