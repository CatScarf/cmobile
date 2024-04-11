import xgboost

import util

def train_xgb():
    # 加载数据
    x_train, x_test, y_train, y_test = util.load_onlinefood_splited()

    # 初始化
    model = xgboost.XGBClassifier(
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='auc',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    )

    # 训练
    model.fit(
        x_train, 
        y_train, 
        eval_set=[(x_test, y_test)], 
        verbose=False
    )

    # 评估
    yhat_test = model.predict_proba(x_test)[:,1]
    result = util.evaluate(y_test, yhat_test)

    # 保存
    x, _ = util.load_onlinefood()
    yhat = model.predict_proba(x)[:,1] > 0.5
    util.save(yhat, result + '.txt')

if __name__ == '__main__':
    train_xgb()