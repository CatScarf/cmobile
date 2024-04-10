# 导入必要的库
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from train.load_data import load_onlinefood
from sklearn.metrics import roc_auc_score, f1_score
import joblib

# 加载鸢尾花数据集（二分类任务）
# data = load_iris()
def train_tpot():
    X, y = load_onlinefood()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化TPOTClassifier
    tpot = TPOTClassifier(
        generations=5,  # 运行优化过程的代数
        population_size=20,  # 每代中的个体数量
        verbosity=2,  # 设置详细程度（0：静默，1：最小，2：高）
        random_state=42,  # 设置随机种子以保证可重复性
        config_dict='TPOT sparse',  # 使用预定义的配置字典
        scoring='accuracy',  # 优化的评估指标
        cv=5,  # 交叉验证折数
        n_jobs=-1  # 使用的CPU核心数（-1：使用所有可用核心）
    )

    # 在训练数据上拟合TPOT
    tpot.fit(X_train, y_train)

    # 使用交叉验证评估最佳流水线的性能
    # cv_scores = cross_val_score(tpot.fitted_pipeline_, X_train, y_train, cv=5)
    # print(f"交叉验证准确率：{cv_scores.mean():.2f} (标准差：{cv_scores.std():.2f})")

    yhat_test = tpot.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, yhat_test)
    f1 = f1_score(y_test>0.5, yhat_test>0.5, average='macro')
    print(f"auc: {auc:.2f}, f1: {f1:.2f}")

    # 保存模型为json
    best_pipeline = tpot.fitted_pipeline_

    # 保存最佳流水线到文件
    joblib.dump(best_pipeline, 'tpot_auc_{:.4f}_f1_{:.4f}.pkl'.format(auc,f1))
