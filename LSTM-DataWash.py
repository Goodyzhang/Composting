import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import numpy as np


# 定义序列生成函数
def create_sequences(df, feature_cols, target_cols, look_back, forecast_horizon=1):
    X, Y = [], []
    grouped = df.groupby(['Arti', 'Grup'])
    for name, group in tqdm(grouped, desc='Generating sequences', leave=False):
        group_sorted = group.sort_values(by='TIME')
        if group_sorted['TIME'].nunique() < look_back + forecast_horizon:
            continue
        for i in range(len(group_sorted) - look_back - forecast_horizon + 1):
            seq_x = group_sorted.iloc[i:i + look_back][feature_cols].values
            seq_y = group_sorted.iloc[i + look_back + forecast_horizon - 1][target_cols].values
            X.append(seq_x)
            Y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# 设置参数
look_back = 12
forecast_horizon = 1
feature_cols = ['Loca_00', 'Loca_11', 'Loca_13', 'Loca_14', 'Loca_15', 'Loca_22', 'Loca_23',
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
                'A_T_', 'S_T_', 'pH', 'NH4_N', 'NO3_N']
target_cols = ['CO2', 'CH4', 'N2O', 'NH3']

# 1. 加载标准化后的训练集和测试集
train_scaled_path = 'DATA/LSTM-GHG/standardized_data/train_scaled.csv'
test_scaled_path = 'DATA/LSTM-GHG/standardized_data/test_scaled.csv'

train_df_scaled = pd.read_csv(train_scaled_path)
test_df_scaled = pd.read_csv(test_scaled_path)

# 确保所有特征列为数值类型
train_df_scaled[feature_cols] = train_df_scaled[feature_cols].astype(np.float32)
test_df_scaled[feature_cols] = test_df_scaled[feature_cols].astype(np.float32)

# 确保所有目标列为数值类型
train_df_scaled[target_cols] = train_df_scaled[target_cols].astype(np.float32)
test_df_scaled[target_cols] = test_df_scaled[target_cols].astype(np.float32)

# 处理缺失值（示例：前向填充）
train_df_scaled.fillna(method='ffill', inplace=True)
test_df_scaled.fillna(method='ffill', inplace=True)

# 2. 对目标变量进行 Min-Max 归一化，确保非负
target_scaler = MinMaxScaler(feature_range=(0, 1))
train_df_scaled[target_cols] = target_scaler.fit_transform(train_df_scaled[target_cols])
test_df_scaled[target_cols] = target_scaler.transform(test_df_scaled[target_cols])

# 保存 scaler 以便后续反归一化
os.makedirs('DATA/LSTM-GHG/scalers', exist_ok=True)
joblib.dump(target_scaler, 'DATA/LSTM-GHG/scalers/target_scaler.pkl')

print(f"训练集标准化后数据形状: {train_df_scaled.shape}")
print(f"测试集标准化后数据形状: {test_df_scaled.shape}")

# 3. 生成训练集序列
print("\n生成训练集序列...")
X_train, Y_train = create_sequences(
    train_df_scaled,
    feature_cols=feature_cols,
    target_cols=target_cols,
    look_back=look_back,
    forecast_horizon=forecast_horizon
)
print(f"训练集序列形状: X_train: {X_train.shape}, Y_train: {Y_train.shape}")

# 4. 生成测试集序列
print("\n生成测试集序列...")
X_test, Y_test = create_sequences(
    test_df_scaled,
    feature_cols=feature_cols,
    target_cols=target_cols,
    look_back=look_back,
    forecast_horizon=forecast_horizon
)
print(f"测试集序列形状: X_test: {X_test.shape}, Y_test: {Y_test.shape}")

# 5. 保存生成的序列（可选）
os.makedirs('DATA/LSTM-GHG/sequences', exist_ok=True)

np.save('DATA/LSTM-GHG/sequences/X_train.npy', X_train)
np.save('DATA/LSTM-GHG/sequences/Y_train.npy', Y_train)
np.save('DATA/LSTM-GHG/sequences/X_test.npy', X_test)
np.save('DATA/LSTM-GHG/sequences/Y_test.npy', Y_test)

print("\n序列生成完成并保存到 'DATA/LSTM-GHG/sequences' 目录。")



# 设置参数
look_back = 12
forecast_horizon = 1
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

# 1. 加载并过滤数据
print("加载并过滤数据...")
df = pd.read_csv('DATA/GHG.csv')
print(f"原始数据形状: {df.shape}")

# 按'Arti'和'Grup'分组，并过滤每组至少有29天数据的实验组
filtered_df = df.groupby(['Arti', 'Grup']).filter(lambda x: len(x) >= 29)
print(f"过滤后数据形状: {filtered_df.shape}")

# 确保时间步从0到28
if filtered_df['TIME'].min() != 0 or filtered_df['TIME'].max() < 28:
    # 假设每个实验组的时间步连续且完整
    filtered_df['TIME'] = filtered_df.groupby(['Arti', 'Grup']).cumcount()

    # 重新过滤确保每组时间步覆盖0到28
    filtered_df = filtered_df.groupby(['Arti', 'Grup']).filter(lambda x: x['TIME'].max() >= 28)
    print(f"重新过滤后数据形状: {filtered_df.shape}")

# 2. 训练集/测试集分割
print("\n进行训练集和测试集的分割...")
# 获取所有满足条件的实验组的唯一标识
experiment_groups = list(filtered_df.groupby(['Arti', 'Grup']).groups.keys())
print(f"满足条件的实验组数量: {len(experiment_groups)}")

# 定义训练集和测试集的比例
train_size = 0.8

# 使用train_test_split按实验组进行分割
train_groups, test_groups = train_test_split(
    experiment_groups,
    train_size=train_size,
    random_state=42
)

print(f"训练集实验组数量: {len(train_groups)}")
print(f"测试集实验组数量: {len(test_groups)}")

# 根据分割的实验组获取对应的数据
train_df = filtered_df.set_index(['Arti', 'Grup']).loc[train_groups].reset_index()
test_df = filtered_df.set_index(['Arti', 'Grup']).loc[test_groups].reset_index()

print(f"训练集数据形状: {train_df.shape}")
print(f"测试集数据形状: {test_df.shape}")

# 3. 确保每个实验组中的行顺序不被打乱
print("\n确保每个实验组中的行顺序不被打乱...")
train_df = train_df.sort_values(by=['Arti', 'Grup', 'TIME']).reset_index(drop=True)
test_df = test_df.sort_values(by=['Arti', 'Grup', 'TIME']).reset_index(drop=True)

# 4. 保存分割后的数据集
print("\n保存分割后的数据集...")
os.makedirs('DATA/LSTM-GHG/split_data', exist_ok=True)
train_df.to_csv('split_data/train.csv', index=False)
test_df.to_csv('split_data/test.csv', index=False)

print("数据分割完成并保存到 'split_data' 目录。")

# 5. 目标变量归一化
print("\n对目标变量进行 Min-Max 归一化，确保非负...")
# 为每个目标变量单独初始化 MinMaxScaler
scalers = {}
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

for target in target_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df_scaled[target] = scaler.fit_transform(train_df[[target]])
    test_df_scaled[target] = scaler.transform(test_df[[target]])
    scalers[target] = scaler
    print(f"{target} 归一化完成。")

# 6. 保存归一化后的数据和 Scalers
print("\n保存归一化后的数据和 Scalers...")
os.makedirs('DATA/LSTM-GHG/scalers', exist_ok=True)
os.makedirs('DATA/LSTM-GHG/standardized_data', exist_ok=True)

train_df_scaled.to_csv('standardized_data/train_scaled.csv', index=False)
test_df_scaled.to_csv('standardized_data/test_scaled.csv', index=False)

# 保存每个目标变量的 scaler
for target, scaler in scalers.items():
    scaler_path = os.path.join('DATA/LSTM-GHG/scalers', f'target_scaler_{target}.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"{target} 的 Scaler 已保存为 {scaler_path}。")

print("归一化后的数据已保存到 'standardized_data' 目录。")
print("所有 Scaler 对象已分别保存。")