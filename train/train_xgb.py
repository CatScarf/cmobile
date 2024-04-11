<<<<<<< HEAD
=======
"""
pip install xgboost
"""

import xgboost
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
>>>>>>> b736e63d4e9ed8d40b4eba1d92ae06ef97ac197a


import util

def train_xgb():
    import xgboost
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
<<<<<<< HEAD
    result = util.evaluate(y_test, yhat_test)

    # 保存
    x, _ = util.load_onlinefood()
    yhat = model.predict_proba(x)[:,1] > 0.5
    util.save(yhat, result + '.txt')

=======
    auc = roc_auc_score(y_test, yhat_test)
    f1 = f1_score(y_test>0.5, yhat_test>0.5, average='macro')
    print(f"auc: {auc:.2f}, f1: {f1:.2f}")

    # 保存模型为json
    path = f"xgb_auc_{auc:.2f}_f1_{f1:.2f}.json"
    model.save_model(path)
    print(f'model saved to {path}')
    
>>>>>>> b736e63d4e9ed8d40b4eba1d92ae06ef97ac197a
if __name__ == '__main__':
    train_xgb()