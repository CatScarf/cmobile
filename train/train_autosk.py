from autosklearn import classification as ask

import util

def train_autosk():
    # 加载数据
    x_train, x_test, y_train, y_test = util.load_onlinefood_splited()

    # 初始化
    model = ask.AutoSklearnClassifier()

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
    train_autosk()