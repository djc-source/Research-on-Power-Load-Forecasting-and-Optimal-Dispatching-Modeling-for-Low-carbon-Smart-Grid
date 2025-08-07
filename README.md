# 电力负荷预测与优化调度实验

## 环境要求（基于conda虚拟环境）

**基础预测模块**：
```bash
pip install torch pandas numpy scikit-learn openpyxl
```

**调度模块**：
```bash
pip install pyomo numpy pandas matplotlib seaborn
conda install -c conda-forge glpk
```

## 使用方法

### 1. 数据预处理
```bash
python data/data_preprocess.py
```

### 2. 训练模型

**训练DE-LSTM基线模型**：
```bash
python train_de_lstm_baseline.py --model_type de_lstm --area Area1
```

**训练LSTM基线模型**：
```bash
python train_lstm_baseline.py --area Area1
```

**训练LSTM-Transformer模型**：
```bash
python train_lstm_transformer.py --area Area1
```

### 3. 运行调度实验（要基于模型实际保存实际路径）

```bash
python run_dispatching_experiments.py \
    --area Area1 \
    --baseline_model checkpoints/de_lstm_baseline/de_lstm_Area1_xxx/final_model.pth \
    --proposed_model checkpoints/lstm_transformer/lstm_transformer_Area1_xxx/final_model.pth \
    --total_steps 100 \
    --feedback_coefficient 0.1 \
    --save_results




    python run_dispatching_experiments.py  --area Area1   --baseline_model checkpoints/de_lstm_baseline/de_lstm_Area1_20250705_110332/final_model.pth   --proposed_model checkpoints/lstm_transformer/lstm_transformer_Area1_20250705_100953/final_model.pth  --total_steps 100 --feedback_coefficient 0.1  --save_results 
```





