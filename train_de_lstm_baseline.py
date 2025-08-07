import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
from datetime import datetime

from models.de_lstm_baseline import DeLSTMBaseline, SimpleLSTM
from trainers.lstm_trainer import LSTMTrainer
from utils.dataset import create_data_loaders, get_area_data_paths
from utils.metrics import compare_models

def parse_args():

    parser = argparse.ArgumentParser(description='训练降低性能的LSTM Baseline模型')

    parser.add_argument('--area', type=str, default='Area1', choices=['Area1', 'Area2'],
                       help='训练区域')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='数据目录')
    parser.add_argument('--sequence_length', type=int, default=96,
                       help='输入序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--scaler_type', type=str, default='standard', 
                       choices=['standard', 'minmax'],
                       help='标准化类型')

    parser.add_argument('--input_dim', type=int, default=4,
                       help='输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--model_type', type=str, default='de_lstm', 
                       choices=['de_lstm', 'simple_lstm'],
                       help='模型类型')

    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='早停耐心值')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                       help='学习率调度器耐心值')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                       help='学习率缩放因子')

    parser.add_argument('--device', type=str, default='auto',
                       help='设备选择 (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/de_lstm_baseline',
                       help='模型保存目录')
    parser.add_argument('--print_every', type=int, default=5,
                       help='打印频率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    return parser.parse_args()

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    args = parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model_type}_{args.area}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"降低性能的LSTM模型训练 - {args.area}")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"实验名称: {exp_name}")
    print(f"保存目录: {save_dir}")

    try:
        train_path, val_path, test_path = get_area_data_paths(args.area, args.data_dir)
        print(f"训练数据: {train_path}")
        print(f"验证数据: {val_path}")
        print(f"测试数据: {test_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    print("\n正在创建数据加载器...")
    train_loader, val_loader, test_loader, datasets = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        scaler_type=args.scaler_type,
        num_workers=0
    )

    print(f"训练集样本数: {len(datasets['train']):,}")
    print(f"验证集样本数: {len(datasets['val']):,}")
    print(f"测试集样本数: {len(datasets['test']):,}")

    print(f"\n正在创建{args.model_type}模型...")
    if args.model_type == 'de_lstm':
        model = DeLSTMBaseline(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1,
            dropout=args.dropout
        )
    elif args.model_type == 'simple_lstm':
        model = SimpleLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=1,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,
        factor=args.scheduler_factor
    )

    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=save_dir
    )

    model_info = trainer.get_model_summary()
    print(f"\n模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print(f"\n开始训练...")
    training_history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_best=True,
        print_every=args.print_every
    )

    print(f"\n正在评估测试集...")
    test_metrics = trainer.evaluate(trainer.test_loader, "测试集")

    training_history['test_metrics'] = test_metrics

    def convert_numpy_types(obj):

        if hasattr(obj, 'item'):  
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    converted_history = convert_numpy_types(training_history)

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(converted_history, f, indent=2)

    print(f"\n最终评估结果:")
    final_metrics = training_history['test_metrics']
    print(f"测试集指标:")
    for metric, value in final_metrics.items():
        if isinstance(value, (int, float)):
            if metric in ['MAPE', 'nRMSE']:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.6f}")
        else:
            print(f"  {metric}: {value}")

    print(f"\n训练完成! 模型保存在: {save_dir}")
    print(f"最佳模型: final_model.pth")
    print(f"训练历史: training_history.json")
    print(f"配置文件: config.json")

if __name__ == '__main__':
    main()