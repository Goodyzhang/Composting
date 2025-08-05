import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
from tqdm import tqdm
import joblib
from itertools import product
import time
import matplotlib.pyplot as plt

# 检查并创建 GHG-Results 目录
if not os.path.exists('GHG-Results'):
    print("GHG-Results 文件夹不存在，即将创建...")
    os.makedirs('GHG-Results', exist_ok=True)

# 2. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
feature_cols = [
    'Loca_00', 'Loca_11', 'Loca_13', 'Loca_14', 'Loca_15', 'Loca_22', 'Loca_23',
    'Loca_32', 'Loca_33', 'Loca_37', 'Loca_41', 'Loca_42', 'Loca_43', 'Loca_44',
    'Loca_45', 'Loca_50', 'Loca_53', 'Loca_61', 'Loca_62', 'Loca_63', 'Loca_64',
    'Loca_81', '1Mat_BR', '1Mat_CM', '1Mat_ChM', '1Mat_DMa', '1Mat_ExT',
    '1Mat_FW', '1Mat_GW', '1Mat_ISS', '1Mat_PM', '1Mat_RS', '1Mat_SM',
    '1Mat_SS', '1Per_0_', '1Per_0_20_', '1Per_20_40_', '1Per_40_60_',
    '1Per_60_80_', '2Mat_AIW', '2Mat_BS', '2Mat_CGS', '2Mat_CM',
    '2Mat_CS', '2Mat_ChM', '2Mat_ExT', '2Mat_FW', '2Mat_GC', '2Mat_GW',
    '2Mat_MS', '2Mat_N_A', '2Mat_PM', '2Mat_RB', '2Mat_RH', '2Mat_RS',
    '2Mat_SD', '2Mat_SM', '2Mat_SS', '2Mat_WA', '2Mat_WB', '2Mat_WS',
    '2Mat_urea_NH2_2CO', '2Per_0_', '2Per_0_20_', '2Per_20_40_',
    '2Per_40_60_', '3Mat_CS', '3Mat_ChM', '3Mat_DG', '3Mat_ExT',
    '3Mat_GC', '3Mat_MS', '3Mat_N_A', '3Mat_PM', '3Mat_SD', '3Mat_SM',
    '3Mat_SW', '3Mat_WS', '3Per_0_', '3Per_0_20_', '3Per_20_40_',
    '3Per_40_60_', '3Per_60_80_', 'AddT_Bio', 'AddT_Com', 'AddT_InoEx',
    'AddT_InoN', 'AddT_InoP', 'AddT_NoAddi', 'AddT_OrgB', 'AddT_OrgCN',
    'AddT_OrgEx', 'AddT_Ref', 'APer_0_', 'APer_5_', 'APer_1_', 'PreT_NoPret',
    'PreT_PhyP', 'CmpT_Na', 'CmpT_Re', 'CmpT_Tr', 'CmpT_Wd', 'CmpV_10_30L',
    'CmpV_200_500L', 'CmpV_30_80L', 'CmpV_3k_6kL', 'CmpV_500_1_5kL',
    'CmpV_80_200L', 'CmpV_10L', 'CmpV_15kL', 'CmpV_No_Vol', 'VenT_Co',
    'VenT_In', 'VenT_None', 'VenT_Tu', 'd_Lmin_Turn_3d_to_7d',
    'd_Lmin_Turn_7d', 'd_Lmin_Turn_8d_to_15d', 'd_Lmin_Turn_15d',
    'd_Lmin_Turn_Below_3d', 'd_Lmin_Ven_0_2_to_0_4', 'd_Lmin_Ven_0_4_to_0_6',
    'd_Lmin_Ven_0_6_to_0_8', 'd_Lmin_Ven_0_8_to_1', 'd_Lmin_Ven_Below_0_2',
    'A_T_', 'S_T_', 'pH', 'NH4_N', 'NO3_N'
]
target_cols = ['CO2', 'CH4', 'N2O', 'NH3']
# 定义标签名称
label_names = ['CO2', 'CH4', 'N2O', 'NH3']
# 1. 加载生成的序列
print("加载生成的序列...")
try:
    X_train = np.load('DATA/LSTM-GHG/sequences/X_train.npy', allow_pickle=False)
    Y_train = np.load('DATA/LSTM-GHG/sequences/Y_train.npy', allow_pickle=False)
    X_test = np.load('DATA/LSTM-GHG/sequences/X_test.npy', allow_pickle=False)
    Y_test = np.load('DATA/LSTM-GHG/sequences/Y_test.npy', allow_pickle=False)
except FileNotFoundError as e:
    print(f"文件未找到: {e.filename}")
    exit(1)

print(f"训练集序列形状: X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"测试集序列形状: X_test: {X_test.shape}, Y_test: {Y_test.shape}")

# 2. 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# 3. 转换为 PyTorch 张量
print("转换为 PyTorch 张量...")
X_train_tensor = torch.tensor(X_train).float()
Y_train_tensor = torch.tensor(Y_train).float()
X_test_tensor = torch.tensor(X_test).float()
Y_test_tensor = torch.tensor(Y_test).float()

# 4. 创建数据集和数据加载器
batch_size = 1  # 初始设置为1，后续网格搜索将覆盖
print("创建数据集和数据加载器...")
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# 注意：在网格搜索中，batch_size 将在循环中调整
# 因此，这里不创建 DataLoader

print(f"训练集样本数量: {len(train_dataset)}")
print(f"测试集样本数量: {len(test_dataset)}")


os.makedirs('GHG-Results/grid_search_results', exist_ok=True)

# 自定义注意力机制层
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attention_weights = torch.nn.functional.softmax(lstm_output, dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output

# 更新后的 AdvancedLSTMModel，加入注意力机制
class AdvancedLSTMWithAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=256, num_layers=3, output_size=1, dropout=0.3, bidirectional=True):
        super(AdvancedLSTMWithAttentionModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 定义双向或单向 LSTM 层
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # LSTM dropout 仅在层数 >1 时应用
            bidirectional=bidirectional
        )

        # 计算全连接层的输入尺寸
        lstm_output_size = hidden_layer_size * 2 if bidirectional else hidden_layer_size

        # 添加注意力层
        self.attention = AttentionLayer()

        # 添加额外的全连接层
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.ln1 = nn.LayerNorm(128)  # 使用 LayerNorm 代替 BatchNorm1d
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_size)
        self.softplus = nn.Softplus()  # 非负激活函数

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_layer_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_layer_size * num_directions)

        # 通过注意力层
        attention_out = self.attention(out)

        # 通过全连接层
        out = self.fc1(attention_out)  # (batch_size, 128)
        out = self.ln1(out)   # 使用 LayerNorm 代替 BatchNorm1d
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)  # (batch_size, output_size)
        out = self.softplus(out)  # 确保输出非负

        return out

# 定义自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, lambda_penalty=10.0):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_penalty = lambda_penalty  # 惩罚系数

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        # 计算负值预测的平均值
        penalty = torch.mean(torch.relu(-outputs))
        total_loss = mse_loss + self.lambda_penalty * penalty
        return total_loss

# 6. 加载每个目标变量的 scaler
print("\n加载各目标变量的 Scalers...")
scalers = {}
for label in label_names:
    scaler_path = os.path.join('DATA/LSTM-GHG/scalers', f'target_scaler_{label}.pkl')
    try:
        scalers[label] = joblib.load(scaler_path)
        print(f"{label} 的 Scaler 已加载。")
    except FileNotFoundError:
        print(f"Scaler 文件未找到: {scaler_path}")
        exit(1)

# 7. 定义标签特定的超参数网格
param_grid_per_label = {
    'CO2': {
        'hidden_layer_size': [45, 45, 45, 45, 45, 45],
        'num_layers': [6, 6, 6, 6, 6, 6],
        'dropout': [0.25, 0.25, 0.25],
        'bidirectional': [True],
        'learning_rate': [0.0018],
        'batch_size': [44, 46, 48]
    },
    'CH4': {
        'hidden_layer_size': [140, 150, 160],
        'num_layers': [2, 3, 4],
        'dropout': [0.15, 0.20, 0.25],
        'bidirectional': [True, False],
        'learning_rate': [0.001],
        'batch_size': [64, 80, 128]
    },
    'N2O': {
        'hidden_layer_size': [150, 200, 250],
        'num_layers': [2, 3],
        'dropout': [0.06, 0.08, 0.1],
        'bidirectional': [True, False],
        'learning_rate': [0.001, 0.0012],
        'batch_size': [60, 85, 90, 100]
    },
    'NH3': {
        'hidden_layer_size': [90, 90, 90, 90, 90],
        'num_layers': [3,3,3,3,3],
        'dropout': [0.25, 0.25],
        'bidirectional': [True],
        'learning_rate': [0.001],
        'batch_size': [75, 75, 75, 75, 75]
    }
}


'''# 定义通用超参数网格
common_param_grid = {
    'bidirectional': [True, False]
}'''


# 定义训练函数
def train_and_evaluate(model, criterion, optimizer, train_loader, epochs=50):
    model.train()
    epoch_losses = []
    for epoch in tqdm(range(epochs), desc="Training", leave=False):
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
    return epoch_losses

def evaluate_model(model, test_loader, scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            targets.append(Y_batch.cpu().numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    try:
        preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"反归一化失败: {e}")
        preds_original = preds
        targets_original = targets
    mse = mean_squared_error(targets_original, preds_original)
    mae = mean_absolute_error(targets_original, preds_original)
    r2 = r2_score(targets_original, preds_original)
    print(f'R2 = {r2:.4f}')
    return mse, mae, r2, preds_original, targets_original

# 定义保存预测值、真实值以及模型文件的函数
def save_predictions_and_model(preds, targets, label, model, model_save_path, pred_real_save_path):
    pred_real_df = pd.DataFrame({
        'Predict': preds,
        'Real': targets
    })
    pred_real_df.to_csv(pred_real_save_path, index=False)
    print(f"预测值与真实值已保存为 {pred_real_save_path}")
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存为 {model_save_path}")

# 检查模型文件是否可读取
def check_model_readability(model_save_path):
    try:
        model = torch.load(model_save_path)
        print(f"模型 {model_save_path} 已成功加载。")
        return True
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False
# 8. 定义初始训练参数
epochs = 50  # 根据观察收敛情况调整为50

# 9. 初始化结果存储
results = []

# ========== 网格搜索 ==========
for label in label_names:
    # 日志文件：GHG-Results/log_{label}.txt
    label_log_path = os.path.join("GHG-Results", f"log_GRU_{label}.txt")
    # 若文件已存在，清空或根据需要处理
    with open(label_log_path, 'w', encoding='utf-8') as f:
        f.write(f"日志开始: 目标 {label}\n")

    print(f"\n=== 网格搜索开始: 预测目标 {label} ===")
    # 写入日志
    with open(label_log_path, 'a', encoding='utf-8') as f:
        f.write(f"网格搜索开始: 目标 {label}\n")

    scaler = scalers[label]

    # 选择对应的目标
    label_index = label_names.index(label)
    Y_train_label = Y_train_tensor[:, label_index].unsqueeze(1)  # (num_samples, 1)
    Y_test_label = Y_test_tensor[:, label_index].unsqueeze(1)    # (num_samples, 1)

    # 创建新的数据集
    train_label_dataset = TensorDataset(X_train_tensor, Y_train_label)
    test_label_dataset = TensorDataset(X_test_tensor, Y_test_label)

    # 获取该标签的特定超参数
    label_specific_grid = param_grid_per_label[label]

    # 4) 生成所有可能的超参数组合
    keys = label_specific_grid.keys()
    values = (label_specific_grid[k] for k in keys)
    hyperparameter_combinations = list(product(*values))
    print(f"总超参数组合数: {len(hyperparameter_combinations)}")

    best_r2 = -float('inf')  # 初始的R2设置为负无穷
    best_params = None
    best_mse = None
    best_mae = None
    best_pred = None
    best_targ = None
    best_model = None

    # 分块输出
    chunk_size = 50
    lines_buffer = []

    for idx, combo in enumerate(hyperparameter_combinations, start=1):
        # combo 就是 (hidden_layer_size, num_layers, dropout, bidirectional, learning_rate, batch_size)
        # 按keys顺序解包
        hidden_layer_size, num_layers, dropout, bidirectional, learning_rate, bs = combo

        msg_combo = (f"\n训练组合 {idx}/{len(hyperparameter_combinations)}: "
                     f"hidden_layer_size={hidden_layer_size}, num_layers={num_layers}, "
                     f"dropout={dropout}, bidirectional={bidirectional}, "
                     f"learning_rate={learning_rate}, batch_size={bs}")
        print(msg_combo)
        lines_buffer.append(msg_combo)

        # 创建 DataLoader
        train_label_loader = DataLoader(train_label_dataset, batch_size=bs, shuffle=True)
        test_label_loader = DataLoader(test_label_dataset, batch_size=bs, shuffle=False)

        # 定义模型
        input_size = X_train.shape[2]  # 假设为130
        model = AdvancedLSTMWithAttentionModel(
            input_size=input_size,
            hidden_layer_size=hidden_layer_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(device)

        # 定义损失函数和优化器
        criterion = CustomLoss(lambda_penalty=10.0)  # 使用自定义损失函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练
        epoch_losses = []
        model.train()
        for epoch in tqdm(range(epochs), desc="Training", leave=False):
            epoch_loss = 0
            for X_batch, Y_batch in train_label_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_label_loader)
            epoch_losses.append(avg_loss)

        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in DataLoader(test_label_dataset, batch_size=64, shuffle=False):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                preds.append(outputs.cpu().numpy())
                targets.append(Y_batch.cpu().numpy())
        preds = np.concatenate(preds, axis=0).flatten()
        targets = np.concatenate(targets, axis=0).flatten()

        # 反归一化
        try:
            preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"反归一化失败: {e}")
            preds_original = preds
            targets_original = targets

        # 获取评估结果
        mse, mae, r2, pred, targ= evaluate_model(model, test_label_loader, scaler)

        # 更新最佳超参数组合
        if r2 > best_r2:
            best_r2 = r2
            best_mse = mse
            best_mae = mae
            best_params = {
                'hidden_layer_size': hidden_layer_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'bidirectional': bidirectional,
                'learning_rate': learning_rate,
                'batch_size': bs
            }

            # 保存最佳的预测结果和真实值
            best_pred = pred
            best_targ = targ
            best_model = model

            # 计算当前最佳预测和真实值的 R2
            calculated_r2 = r2_score(best_targ, best_pred)
            if not np.isclose(calculated_r2, best_r2, atol=0.1):
                print(f"警告: 计算得到的 R2({calculated_r2:.6f}) 与记录的 best_r2({best_r2:.6f}) 不一致！")
                # 可选择抛出异常或记录日志
            else:
                print("检查通过: best_pred 与 best_targ 的 R2 与 best_r2 一致。")

    # 保存最佳评估结果到文件
    with open(label_log_path, 'a') as f:
        f.write(f"最佳超参数组合: {best_params}\n")
        f.write(f"最佳评估结果： R2={best_r2:.4f}, RMSE={np.sqrt(best_mse):.4f}, MAE={best_mae:.4f}\n")

    # 输出最佳评估结果
    print(f"目标 {label} 的最佳超参数组合： R2={best_r2:.4f}, RMSE={np.sqrt(best_mse):.4f}, MAE={best_mae:.4f}")

    # 保存预测结果与真实值为 txt 格式
    pred_real_save_path_txt = os.path.join('GHG-Results', f"oh_GRU_{label}.txt")
    with open(pred_real_save_path_txt, 'w') as f:
        # 写入表头
        f.write("Predict\tReal\n")
        # 遍历预测值和真实值并逐行写入
        for pred, real in zip(best_pred, best_targ):
            f.write(f"{pred:.4f}\t{real:.4f}\n")

    print(f"预测结果与真实值已保存为 {pred_real_save_path_txt}")

    # 保存最佳模型
    if best_model is not None:
        model_save_path = os.path.join('GHG-Results', f"GRU_{label}_BestModel.pkl")
        torch.save(best_model.state_dict(), model_save_path)
        print(f"最佳模型已保存为 {model_save_path}")
