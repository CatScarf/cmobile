from tpot import TPOTClassifier

import util

def train_tpot():
    # 加载数据
    x_train, x_test, y_train, y_test = util.load_onlinefood_splited()

    # 初始化
    tpot = TPOTClassifier(
        generations=5,              # 运行优化过程的代数
        population_size=20,         # 每代中的个体数量
        verbosity=1,                # 设置详细程度（0：静默，1：最小，2：高）
        random_state=42,            # 设置随机种子以保证可重复性
        generations=5,              # 运行优化过程的代数
        population_size=20,         # 每代中的个体数量
        verbosity=1,                # 设置详细程度（0：静默，1：最小，2：高）
        random_state=42,            # 设置随机种子以保证可重复性
        config_dict='TPOT sparse',  # 使用预定义的配置字典
        scoring='accuracy',         # 优化的评估指标
        cv=5,                       # 交叉验证折数
        n_jobs=-1                   # 使用的CPU核心数（-1：使用所有可用核心）
        scoring='accuracy',         # 优化的评估指标
        cv=5,                       # 交叉验证折数
        n_jobs=-1                   # 使用的CPU核心数（-1：使用所有可用核心）
    )

    # 训练
    tpot.fit(x_train, y_train)

    # 评估
    yhat_test = tpot.predict_proba(x_test)[:,1]
    result = util.evaluate(y_test, yhat_test)

    # 保存
    x, _ = util.load_onlinefood()
    yhat = tpot.predict_proba(x)[:,1] > 0.5
    util.save(yhat, result + '.txt')


if __name__ == '__main__':
    train_tpot()