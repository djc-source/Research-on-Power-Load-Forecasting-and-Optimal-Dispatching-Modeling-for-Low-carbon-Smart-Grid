import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
from datetime import datetime

from models.lstm_transformer import LSTMTransformerModel
from trainers.lstm_transformer_trainer import LSTMTransformerTrainer
from utils.dataset import create_data_loaders, get_area_data_paths
from utils.metrics import compare_models

def parse_args():

    parser = argparse.ArgumentParser(description='训练LSTM-Transformer混合模型')

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

    parser.add_argument('--input_dim', type=int, default=5,
                       help='输入特征维度 (4基础特征 + 1时间特征 = 5)')
    parser.add_argument('--lstm_hidden_dim', type=int, default=128,
                       help='LSTM隐藏层维度')
    parser.add_argument('--lstm_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--transformer_dim', type=int, default=32,
                       help='Transformer维度')
    parser.add_argument('--transformer_layers', type=int, default=4,
                       help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--use_time_features', action='store_true', default=True,
                       help='是否使用时间特征')

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
    parser.add_argument('--save_dir', type=str, default='checkpoints/lstm_transformer',
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
    exp_name = f"lstm_transformer_{args.area}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print(f"LSTM-Transformer 混合模型训练 - {args.area}")
    print("=" * 60)
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
        num_workers=0,
        use_time_features=args.use_time_features
    )

    actual_input_dim = len(datasets['train'].feature_columns)
    print(f"实际输入特征维度: {actual_input_dim}")
    print(f"特征列: {datasets['train'].feature_columns}")

    print(f"训练集样本数: {len(datasets['train']):,}")
    print(f"验证集样本数: {len(datasets['val']):,}")
    print(f"测试集样本数: {len(datasets['test']):,}")

    print(f"\n正在创建LSTM-Transformer模型...")
    model = LSTMTransformerModel(
        input_dim=actual_input_dim,  
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        transformer_dim=args.transformer_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        sequence_length=args.sequence_length,
        output_dim=1,
        dropout=args.dropout
    )

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

    trainer = LSTMTransformerTrainer(
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

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:

        history = {}
        for key, value in training_history.items():
            if isinstance(value, list):

                converted_list = []
                for item in value:
                    if isinstance(item, dict):

                        converted_dict = {}
                        for k, v in item.items():
                            if hasattr(v, 'item'):  
                                converted_dict[k] = float(v.item())
                            elif isinstance(v, (int, float, str, bool)):
                                converted_dict[k] = v
                            else:
                                converted_dict[k] = str(v)
                        converted_list.append(converted_dict)
                    elif hasattr(item, 'item'):  
                        converted_list.append(float(item.item()))
                    elif isinstance(item, (int, float, str, bool)):
                        converted_list.append(item)
                    else:
                        converted_list.append(str(item))
                history[key] = converted_list
            else:
                history[key] = str(value)
        json.dump(history, f, indent=2)

    print(f"\n正在加载最佳模型进行评估...")
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path, load_optimizer=False)

    print(f"\n正在评估模型...")
    test_metrics = trainer.evaluate(trainer.test_loader, "测试")
    val_metrics = trainer.evaluate(trainer.val_loader, "验证")
    train_metrics = trainer.evaluate(trainer.train_loader, "训练")

    def convert_numpy_types(obj):

        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'item'):  
            return float(obj.item())
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    evaluation_results = {
        'train_metrics': convert_numpy_types(train_metrics),
        'val_metrics': convert_numpy_types(val_metrics),
        'test_metrics': convert_numpy_types(test_metrics),
        'model_info': convert_numpy_types(model_info),
        'best_metrics': convert_numpy_types(trainer.metrics_tracker.get_best_metrics())
    }

    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"训练完成 - {exp_name}")
    print(f"=" * 60)
    print(f"最佳验证损失: {trainer.metrics_tracker.best_val_loss:.6f}")
    print(f"最佳验证MAPE: {trainer.metrics_tracker.best_metrics.get('MAPE', 0):.2f}%")
    print(f"测试集MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"测试集RMSE: {test_metrics['RMSE']:.4f}")
    print(f"测试集R²: {test_metrics['R2']:.4f}")
    print(f"\n模型和结果已保存至: {save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()