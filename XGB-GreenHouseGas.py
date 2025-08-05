import xgboost as xgb
import torch
import sys
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from skopt import BayesSearchCV

# 检查 CUDA 可用性及 GPU 信息
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

def check_xgb_gpu_support():
    try:
        # 尝试创建一个使用 GPU 的 XGBRegressor 模型
        model = XGBRegressor(
            tree_method='gpu_hist',      # 使用 GPU 的直方图算法
            predictor='gpu_predictor',   # 使用 GPU 预测器
            random_state=42,
            n_jobs=-1
        )
        print("XGBoost 已成功配置为使用 GPU 运算。")
    except ValueError as e:
        print("XGBoost 未配置为使用 GPU 运算。")
        print(f"错误信息: {e}")

check_xgb_gpu_support()

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import joblib
import warnings
import time
from tqdm import tqdm
import os

# 忽略警告信息
warnings.filterwarnings("ignore")

# 确保日志和模型目录存在
os.makedirs('GHG-Results', exist_ok=True)
os.makedirs('GHG-Results/Models', exist_ok=True)

# 加载数据并处理缺失值
try:
    df = pd.read_csv("DATA/GHG.csv")
    print("数据已成功加载。")
except FileNotFoundError:
    print("错误：指定的CSV文件未找到。请检查路径是否正确。")
    sys.exit()
except pd.errors.EmptyDataError:
    print("错误：CSV文件为空。请检查数据源。")
    sys.exit()
except Exception as e:
    print(f"加载数据时发生未知错误：{e}")
    sys.exit()

# 输出数据概况
print("\n数据概况：")
df.info()
print("\n数据描述性统计：")
print(df.describe())

# 检查并填充缺失值
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"列 '{col}' 存在缺失值，已使用中位数 {median_value} 填充。")
        else:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col].fillna(mode_value[0], inplace=True)
                print(f"列 '{col}' 存在缺失值，已使用众数 '{mode_value[0]}' 填充。")
            else:
                df[col].fillna('', inplace=True)
                print(f"列 '{col}' 存在缺失值，已使用空字符串填充。")

# 设置 MultiIndex
if 'Arti' not in df.columns or 'Grup' not in df.columns:
    raise ValueError("数据中缺少 'Arti' 或 'Grup' 列，请检查数据。")

df.set_index(['Arti', 'Grup'], inplace=True)
print("\n已将 'Arti' 和 'Grup' 列设置为 MultiIndex。")

# 重置索引，方便后续处理
df.reset_index(inplace=True)

# 生成分组信息
groups = df['Arti'].astype(str) + '_' + df['Grup'].astype(str)
print("\n分组信息已生成。")

# 检查分组样本数量，统计样本数不足的分组
group_counts = groups.value_counts()
insufficient_groups = group_counts[group_counts < 2].index.tolist()
if insufficient_groups:
    print(f"\n以下分组的样本数少于 2，将被删除：{insufficient_groups}")
    # 删除样本数不足的分组
    mask = ~groups.isin(insufficient_groups)
    df = df.loc[mask].copy()
    groups = groups.loc[mask].copy()
    print(f"删除后剩余样本数：{len(df)}")
else:
    print("\n所有分组的样本数均不少于 2，无需删除。")

# **新增：按照TIME列仅保留每组TIME为0～28天的数据**
print("\n开始按照TIME列裁剪数据，保留每组TIME为0～28天的数据。")
# initial_group_counts = df.groupby(['Arti', 'Grup'])['TIME'].count().sum()  # 未使用，已移除
df = df[df['TIME'].between(0, 28)]
print("按照TIME列裁剪完成。")
print(f"裁剪后数据总行数：{len(df)}")

# 输出分组后的 TIME 描述性统计
print("\n分组后的 TIME 描述性统计：")
print(df.groupby(['Arti', 'Grup'])['TIME'].describe())

# 生成分组信息（重新生成，以防裁剪后需要更新）
groups = df['Arti'].astype(str) + '_' + df['Grup'].astype(str)
print("\n分组信息已重新生成。")

# 可选：再次检查分组样本数量，确保裁剪后各组仍有足够样本
group_counts_after_crop = groups.value_counts()
insufficient_groups_after_crop = group_counts_after_crop[group_counts_after_crop < 2].index.tolist()
if insufficient_groups_after_crop:
    print(f"\n裁剪后以下分组的样本数少于 2，将被删除：{insufficient_groups_after_crop}")
    # 删除样本数不足的分组
    mask = ~groups.isin(insufficient_groups_after_crop)
    df = df.loc[mask].copy()
    groups = groups.loc[mask].copy()
    print(f"裁剪并删除后剩余样本数：{len(df)}")
else:
    print("\n裁剪后所有分组的样本数均不少于 2，无需删除。")

# 划分特征和目标变量
all_columns = df.columns.tolist()
exclude_columns = ['Arti', 'Grup'] + ['CO2', 'CH4', 'N2O', 'NH3']
feature_columns = [col for col in all_columns if col not in exclude_columns]
X = df[feature_columns]
Y = df[['CO2', 'CH4', 'N2O', 'NH3']]
print("\n特征和目标变量已成功划分。")
print(f"特征数量：{X.shape[1]}")
if "TIME" in X.columns:
    print("TIME列已经加入")
else:
    print("完了")
print(f"目标变量数量：{Y.shape[1]}")

# 更新分组信息
groups = df['Arti'].astype(str) + '_' + df['Grup'].astype(str)

print("df rows:", len(df))
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("groups length:", len(groups))

# 使用 GroupShuffleSplit 划分训练集和测试集
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, Y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

print(f"\n训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")

# **不对特征进行缩放，保持原始值**
print("\n不对特征进行缩放，保持原始值。")
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# 初始化结果 DataFrame
results_df = pd.DataFrame()
metrics_df = pd.DataFrame(columns=['Target', 'RMSE', 'MAE', 'R2'])

# 定义单独的超参数网格
param_grids = {
    'CO2': {
        'n_estimators': [200, 275, 300, 325, 400],
        'max_depth': [5, 11, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.1, 0.6, 0.8],
        'colsample_bytree': [0.1, 0.2, 0.5],
        'gamma': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    },
    'CH4': {
        'n_estimators': [300, 400, 500],
        'max_depth': [5, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1],
        'subsample': [0.2, 0.6, 0.8],
        'colsample_bytree': [0.2, 0.4, 0.8],
        'gamma': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    },
    'N2O': {
        'n_estimators': [300, 375, 400, 500],
        'max_depth': [3, 5, 9],
        'learning_rate': [0.01, 0.1, 0.12],
        'subsample': [0.3, 0.5, 0.7],
        'colsample_bytree': [0.5, 0.9],
        'gamma': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    },
    'NH3': {
        'n_estimators': [300, 380, 395, 400],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.2, 0.6, 0.8],
        'colsample_bytree': [0.2, 0.6, 0.8],
        'gamma': [1],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }
}

param_space = {
    'CO2': {
        'n_estimators': (300, 600),
        'max_depth': (5, 20),
        'learning_rate': (0.001, 0.1),
        'subsample': (0.4, 0.8),
        'colsample_bytree': (0.4, 0.8),
    },
    'CH4': {
        'n_estimators': (200, 600),
        'max_depth': (5, 20),
        'learning_rate': (0.001, 0.1),
        'subsample': (0.1, 0.8),
        'colsample_bytree': (0.1, 0.8),
    },
    'N2O': {
        'n_estimators': (200, 600),
        'max_depth': (5, 20),
        'learning_rate': (0.001, 0.1),
        'subsample': (0.1, 0.8),
        'colsample_bytree': (0.1, 0.8),
    },
    'NH3': {
        'n_estimators': (200, 600),
        'max_depth': (5, 20),
        'learning_rate': (0.001, 0.1),
        'subsample': (0.1, 0.8),
        'colsample_bytree': (0.1, 0.8),
    }
}
# 设置迭代次数（在网格搜索中通常不需要 n_iter，因此这里将其用作参数网格大小的参考）
n_iterations = 50  # 已移除

# 定义自定义 RMSE 评分函数
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# 定义 SHAP分析
# 2. 修改SHAP分析函数，展示前10个特征并调整画布尺寸：
def shap_analysis(model, X_test_scaled, target):
    explainer = shap.Explainer(model, X_test_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # 显示前10个特征
    shap.summary_plot(shap_values, X_test_scaled, max_display=10, show=False)

    # 调整画布尺寸为4:3，并设置字体大小和间距
    plt.gcf().set_size_inches(12, 9)  # 设置尺寸为4:3的比例
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(f"ohTime_Shap{target}.png")
    plt.clf()  # 清空画布以便下次使用


# 针对每个目标变量训练单目标模型
cv = GroupKFold(n_splits=5)

# 初始化总体进度条
#overall_progress = tqdm(total=len(param_grids), desc="训练目标变量", position=0, leave=True)

for target in param_space: # 记得修改
    print(f"\n正在训练目标变量 '{target}' 的模型...")

    # 获取当前目标变量的训练和测试集
    Y_train_target = Y_train[target]
    Y_test_target = Y_test[target]


    # 定义模型
    xgb_model = XGBRegressor(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        random_state=42,
        n_jobs=-1
    )

    # 获取当前目标的超参数网格
    param_grid = param_grids[target]
    param_space_ = param_space[target]

    # 定义网格搜索
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv= cv,
        n_jobs=-1
    )

    # 初始化一个 tqdm 进度条，总迭代次数为 n_iterations
    pbar = tqdm(total=n_iterations, desc="迭代进度")


    # 定义回调函数，每迭代一次调用一次
    def progress_callback(result):
        pbar.update(1)
        return False  # 返回 False 表示不提前停止优化

    bayes_search = BayesSearchCV(
        estimator= xgb_model,
        search_spaces= param_space_,
        n_iter= n_iterations,
        cv= cv,
        scoring= 'r2',
        n_jobs= -1,
        random_state= 42,
    )

    print("\n开始模型训练和超参数优化...")
    #grid_search.fit(X_train, Y_train_target, groups=groups_train)
    bayes_search.fit(X_train, Y_train_target, groups=groups_train, callback=[progress_callback])
    pbar.close()

    # 使用最佳模型对测试集进行预测
    #Y_pred = grid_search.predict(X_test)
    Y_pred = bayes_search.predict(X_test)

    # 获取评估指标
    r2 = r2_score(Y_test_target, Y_pred)
    rmse_value = mean_squared_error(Y_test_target, Y_pred, squared=False)
    mae_value = mean_absolute_error(Y_test_target, Y_pred)

    # 获取最佳超参数
    #best_params = grid_search.best_params_
    best_params = bayes_search.best_params_

    # 输出结果到文件
    results_filename = "GHG-Results/results_ohTime.txt"
    with open(results_filename, 'a') as f:
        f.write(f"{target}\t{r2:.4f}\t{rmse_value:.4f}\t{mae_value:.4f}\t{best_params}\n")

    # 打印结果
    print(f"评估结果已保存：R2={r2:.4f}, RMSE={rmse_value:.4f}, MAE={mae_value:.4f}")
    print(f'最佳参数：{best_params}')

    # 保存模型
    model_filename = f"GHG-Results/ohTime_{target}.pkl"
    joblib.dump(bayes_search.best_estimator_, model_filename)
    print(f"模型已保存为 {model_filename}")

    # 预测并保存预测结果
    with open(f"GHG-Results/ohTime_{target}.txt", 'w') as f:
        for real, pred in zip(Y_test_target, Y_pred):
            f.write(f"{real},{pred}\n")
    print(f"预测结果已保存为 'GHG-Results/ohTime_{target}.txt'")

    # SHAP分析
    shap_analysis(bayes_search.best_estimator_, X_test, target)
    print(f"SHAP分析图像已保存为 'ohTime_Shap{target}.png'")

# 保存所有评估结果
metrics_df.to_csv("GHG-Results/GridSearch_All_True_VS_Pred.csv", index=False)
print("\n所有目标变量的评估结果已保存。")