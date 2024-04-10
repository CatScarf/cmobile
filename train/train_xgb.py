import xgboost
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from load_data import load_onlinefood

if __name__ == '__main__':

    # 读取数据并划分数据集
    x, y = load_onlinefood()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state = 0)

    # 初始化模型
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
    auc = roc_auc_score(y_test, yhat_test)
    f1 = f1_score(y_test>0.5, yhat_test>0.5, average='macro')
    print(f"auc: {auc:.2f}, f1: {f1:.2f}")

    # 保存模型为json
    path = f"xgb_auc_{auc:.2f}_f1_{f1:.2f}.json"
    model.save_model(path)
    print(f'model saved to {path}')
    