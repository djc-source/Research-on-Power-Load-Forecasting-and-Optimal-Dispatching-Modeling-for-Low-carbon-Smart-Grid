import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dispatching.optimal_dispatch import OptimalDispatcher
from dispatching.rolling_optimization import ModelPredictor, RollingOptimizer, create_system_config
from dispatching.comparative_experiments import ExperimentComparator

def parse_args():

    parser = argparse.ArgumentParser(description='运行优化调度实验')

    parser.add_argument('--area', type=str, default='Area1', choices=['Area1', 'Area2'],
                       help='区域选择')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='数据目录')

    parser.add_argument('--baseline_model', type=str, required=True,
                       help='基线LSTM模型路径')
    parser.add_argument('--proposed_model', type=str, required=True,
                       help='提出方法LSTM-Transformer模型路径')

    parser.add_argument('--total_steps', type=int, default=100,
                       help='实验总步数')
    parser.add_argument('--feedback_coefficient', type=float, default=0.1,
                       help='误差反馈系数')

    parser.add_argument('--save_results', action='store_true',
                       help='是否保存结果')

    return parser.parse_args()

def main():

    args = parse_args()

    print("=" * 60)
    print("优化调度实验")
    print("=" * 60)
    print(f"区域: {args.area}")
    print(f"基线模型: {args.baseline_model}")
    print(f"提出模型: {args.proposed_model}")
    print(f"实验步数: {args.total_steps}")
    print(f"反馈系数: {args.feedback_coefficient}")
    print()

    if not os.path.exists(args.baseline_model):
        print(f"错误: 基线模型文件不存在 - {args.baseline_model}")
        return

    if not os.path.exists(args.proposed_model):
        print(f"错误: 提出模型文件不存在 - {args.proposed_model}")
        return

    data_config = {
        'area': args.area,
        'data_dir': args.data_dir,
        'feedback_coefficient': args.feedback_coefficient
    }

    try:

        comparator = ExperimentComparator(data_config)

        print("正在运行基线实验...")
        baseline_results = comparator.run_baseline_experiment(
            model_path=args.baseline_model,
            total_steps=args.total_steps,
            experiment_name="baseline"
        )

        print("\n正在运行提出方法实验...")
        proposed_results = comparator.run_proposed_experiment(
            model_path=args.proposed_model,
            total_steps=args.total_steps,
            feedback_coefficient=args.feedback_coefficient,
            experiment_name="proposed"
        )

        print("\n正在生成比较报告...")
        comparison = comparator.compare_experiments()

        if args.save_results:
            comparator.save_comparison_results()

        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)

        baseline_metrics = baseline_results.get('overall_metrics', {})
        proposed_metrics = proposed_results.get('overall_metrics', {})

        print(f"基线方法 (LSTM + 传统调度):")
        print(f"  平均预测误差: {baseline_metrics.get('avg_prediction_error', 0):.2f} MW")
        print(f"  MAPE: {baseline_metrics.get('mape', 0):.2f}%")
        print(f"  总成本: {baseline_metrics.get('total_cost', 0):.2f} ¥")

        print(f"\n提出方法 (LSTM-Transformer + 协同滚动优化):")
        print(f"  平均预测误差: {proposed_metrics.get('avg_prediction_error', 0):.2f} MW")
        print(f"  MAPE: {proposed_metrics.get('mape', 0):.2f}%")
        print(f"  总成本: {proposed_metrics.get('total_cost', 0):.2f} ¥")

        if baseline_metrics and proposed_metrics:
            error_improvement = ((baseline_metrics.get('avg_prediction_error', 0) - 
                                proposed_metrics.get('avg_prediction_error', 0)) / 
                               baseline_metrics.get('avg_prediction_error', 1)) * 100

            cost_improvement = ((baseline_metrics.get('total_cost', 0) - 
                               proposed_metrics.get('total_cost', 0)) / 
                              baseline_metrics.get('total_cost', 1)) * 100

            print(f"\n改进效果:")
            print(f"  预测误差减少: {error_improvement:.2f}%")
            print(f"  成本降低: {cost_improvement:.2f}%")

        print("\n实验完成!")

    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()