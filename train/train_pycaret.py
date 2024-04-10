import numpy as np
import pandas as pd
from pycaret.classification import *
from train.load_data import load_onlinefood
from sklearn.model_selection import train_test_split

X, y = load_onlinefood()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 将 NumPy 数组转换为 Pandas DataFrame
df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(0, X.shape[-1])])
df["target"] = y_train

# 初始化分类设置
clf_setup = setup(data=df, target="target", silent=True)

# 比较不同模型
best_model = compare_models()

# 训练最佳模型
final_model = finalize_model(best_model)

# 评估模型
evaluate_model(final_model)

# 在新数据上进行预测（您可以将此部分替换为您自己的测试数据）
new_data = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(0, X.shape[-1])])
predictions = predict_model(final_model, data=new_data)

# 打印预测结果
print(predictions)
