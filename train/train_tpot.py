# 导入必要的库
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# 加载鸢尾花数据集（二分类任务）
data = load_iris()
X, y = data.data, data.target
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
cv_scores = cross_val_score(tpot.fitted_pipeline_, X_train, y_train, cv=5)
print(f"交叉验证准确率：{cv_scores.mean():.2f} (标准差：{cv_scores.std():.2f})")
