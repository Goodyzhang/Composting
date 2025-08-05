import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import matplotlib.gridspec as gridspec
from sklearn.impute import SimpleImputer
import joblib

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
SepW2_Ver8 = pd.read_csv("DATA/GI.csv")
print(SepW2_Ver8.info())
SepW2_Ver8_copy = SepW2_Ver8.copy()

# 设置MultiIndex
SepW2_Ver8_copy.set_index(["Article", "Group"], inplace=True)

unique_level_0_values = SepW2_Ver8_copy.index.get_level_values(0).unique()
num_unique_values = len(unique_level_0_values)
print(f"一级索引（试验批次）的个数为： {num_unique_values} 个")

# 批次太多，这里采用反选(NovW4已经提前删除掉了)
selected_columns = SepW2_Ver8_copy


def load_and_preprocess_data():
    # 导入数据
    data_GI = selected_columns  # **这一行的数据可以替换！！**
    print("Data shape: ", data_GI.shape)
    # print("数据集的前五行：\n", data_GI.head(5))
    print("数据读取完毕！")
    # 检查有无数据缺失
    if data_GI.isnull().sum().sum() > 0:
        print("检测到缺失值.")
        imputer = SimpleImputer(strategy="median")  # 采用属性的中位数替换该属性的缺失值
        imputer.fit(data_GI)  # 将imputer实例适配到训练数据
        print("处理中...")
        data_GI = pd.DataFrame(imputer.transform(data_GI), columns=data_GI.columns, index=data_GI.index)
        print("缺失值已采用中位数填充.")
    print("数据缺失检查完毕！")
    # 选取特征和预测值
    X = data_GI.iloc[1:, :-1]
    Y = data_GI.iloc[1:, -1]
    # 删除包含"Unnamed"的列
    cols_to_drop = [col for col in X.columns if "Unnamed" in col]
    X.drop(columns=cols_to_drop, inplace=True)

    print("特征值包括：\n", X.columns)
    print("预测值包括：\n", Y.name)
    return X, Y


def save_output_to_file(grid_search, r2_train, r2, mae, feature_importances):
    with open("GI-Results/model_results_ET.txt", "w") as f:
        # 1. 保存最优的模型参数
        f.write("Best parameters:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\n")

        # 2. 保存模型在训练和测试集上的评估指标
        f.write("Evaluation Metrics:\n")
        f.write(f"R2 score on train set: {r2_train:.4f}\n")
        f.write(f"R2 score on test set: {r2:.4f}\n")
        f.write(f"MAE on test set: {mae:.4f}\n")
        f.write("\n")


def plot_true_vs_predicted(y_true, y_pred, xlabel="True Values", ylabel="Predicted Values"):
    # 设置seaborn风格
    sns.set_style("whitegrid")

    # 创建图形和子图的布局
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

    ax_main = plt.subplot(gs[1, 0])
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_xhist.grid(False)
    ax_yhist.grid(False)
    ax_xhist.yaxis.set_visible(False)
    ax_yhist.xaxis.set_visible(False)

    # 主散点图
    ax_main.scatter(y_true, y_pred, c=np.abs(y_true - y_pred), cmap="viridis", alpha=0.6, edgecolor=None)
    ax_main.set_xlabel('Real Values')
    ax_main.set_ylabel('Predicted Values')
    limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax_main.plot(limits, limits, '--', color='red', alpha=0.75, zorder=0)

    # 使用seaborn画上方的直方图
    sns.histplot(y_true, ax=ax_xhist, bins=50, color='lightgray', alpha=0.6, kde=True)

    # 用这个来设置直方图的y轴范围
    hist_range = [min(y_pred), max(y_pred)]

    # 使用matplotlib的hist函数绘制横向直方图
    ax_yhist.hist(y_pred, bins=50, color='lightgray', alpha=0.6, orientation="horizontal", range=hist_range)

    # 使用seaborn的kdeplot绘制横向KDE
    sns.kdeplot(y=y_pred, ax=ax_yhist, color='gray', bw_adjust=0.5, clip=hist_range)



    # 隐藏直方图的刻度标签
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)
    for spine in ["top", "right", "left", "bottom"]:
        ax_xhist.spines[spine].set_visible(False)
        ax_yhist.spines[spine].set_visible(False)

    # 计算并在主图上显示指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    n = len(y_true)
    text = f'R2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nN = {n}'
    ax_main.text(0.05, 0.95, text, transform=ax_main.transAxes, fontsize=9, verticalalignment='top')
    plt.savefig("real_vs_predicted_with_histograms_seaborn.png", dpi=1000)
    plt.show()


def train_model(X, Y):
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("测试集占比：", x_test.shape[0] / X.shape[0])
    # 定义随机森林模型
    et = ExtraTreesRegressor(random_state=0)
    print("极端随机树模型定义完毕！")
    # 定义超参数搜索范围
    param_grid = {
        'n_estimators': [395, 400, 405],
        'max_depth': [39, 40, 41],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4]
    }
    print("超参数搜索范围是：", param_grid)
    # 定义交叉验证方法
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    print("交叉验证方法：", kf)
    # 定义网格搜索对象
    grid_search = GridSearchCV(et, param_grid, cv=kf, scoring='r2', n_jobs=-1)  # 这里调用所有CPU进行运算

    # 训练模型
    print("开始训练模型！")
    grid_search.fit(x_train, y_train)

    # 输出最优参数和最优得分
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    # 在训练集上评估模型性能
    rf_train_pred = grid_search.predict(x_train)
    r2_train = r2_score(y_train, rf_train_pred)
    print("R2 score on train set: ", r2_train)

    # 在测试集上评估模型性能
    rf_pred = grid_search.predict(x_test)
    r2 = r2_score(y_test, rf_pred)
    mae = mean_absolute_error(y_test, rf_pred)
    print("R2 score on test set: ", r2)
    print("MAE on test set: ", mae)

    # 提取并按得分排序超参数组合
    params_scores = list(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']))
    sorted_params_scores = sorted(params_scores, key=lambda x: x[1], reverse=True)

    # 输出得分最高的前20个超参数组合
    top_20 = sorted_params_scores[:20]
    for index, (params, score) in enumerate(top_20, start=1):
        print(f"Rank {index}:")
        print(f"Mean Test Score: {score:.4f}")
        print(f"Parameters: {params}")
        print("-" * 50)

    # 画图
    #plt.plot(range(len(grid_search.cv_results_['mean_test_score'])), grid_search.cv_results_['mean_test_score'])
    #plt.xlabel("Parameter combinations")
    #plt.ylabel("R2 score")
    #plt.show()
    # 添加真实值与预测值的图
    #plot_true_vs_predicted(y_test, rf_pred)

    # 3. 保存特征重要性
    feature_importances = grid_search.best_estimator_.feature_importances_
    with open("GI-Results/feature_importances_ET_GI.txt", "w") as f:
        for feature, importance in zip(X.columns, feature_importances):
            f.write(f"{feature}: {importance:.4f}\n")

    # 4. 保存模型
    joblib.dump(grid_search.best_estimator_, 'GI-Results/ET_GI_model.pkl')


def main():
    X, Y = load_and_preprocess_data()
    train_model(X, Y)


if __name__ == "__main__":
    main()