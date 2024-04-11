import pandas as pd
from pycaret import classification as caret

import util

def train_pycaret():
    # 加载数据
    x_train, x_test, y_train, y_test = util.load_onlinefood_splited()
    train_df = pd.DataFrame(x_train, columns=[f"feature_{i}" for i in range(0, x_train.shape[-1])])
    train_df["target"] = y_train

    # 训练
    caret.setup(data=train_df, target="target")
    best_model = caret.compare_models()
    final_model = caret.finalize_model(best_model)

    # 评估模型
    eva = caret.evaluate_model(final_model)
    print(eva)

    # 预测
    test_df = pd.DataFrame(x_test, columns=[f"feature_{i}" for i in range(0, x_test.shape[-1])])
    pred = caret.predict_model(final_model, data=test_df, raw_score=True)

    # 打印预测结果
    print(pred)

if __name__ == '__main__':
    train_pycaret()