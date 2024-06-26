import xgboost

import util

def xgb_classification():
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
    result = util.evaluate_classification(y_test, yhat_test)

    # 保存
    x, _ = util.load_onlinefood()
    yhat = model.predict_proba(x)[:,1] > 0.5
    util.save(yhat, result + '.txt')

def xgb_regression():
    # Load data
    x_train, x_test, y_train, y_test = util.load_estate_splited()

    # Initialize the model
    model = xgboost.XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',  # Using squared error for regression
        early_stopping_rounds=20,
    )

    # Train the model
    model.fit(
        x_train, 
        y_train, 
        verbose=False,
        eval_set=[(x_test, y_test)]
    )

    # Predict
    yhat_test = model.predict(x_test)
    result = util.evaluate_regression(y_test, yhat_test)  # Assume a utility function for regression evaluation

    # Save
    x, _ = util.load_estate()
    yhat = model.predict(x)
    util.save(yhat, result + '.txt')

    return result

if __name__ == '__main__':
    xgb_classification()
    xgb_regression()