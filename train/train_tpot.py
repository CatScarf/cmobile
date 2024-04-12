from tpot import TPOTClassifier

import util

def tpot_classification():
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
        config_dict='TPOT sparse',  # 使用预定义的配置字典
        scoring='accuracy',         # 优化的评估指标
        cv=5,                       # 交叉验证折数
    )

    # 训练
    tpot.fit(x_train, y_train)

    # 评估
    yhat_test = tpot.predict_proba(x_test)[:,1]
    result = util.evaluate_classification(y_test, yhat_test)

    # 保存
    x, _ = util.load_onlinefood()
    yhat = tpot.predict_proba(x)[:,1] > 0.5
    util.save(yhat, result + '.txt')

def tpot_regression():
    # Load data
    x_train, x_test, y_train, y_test = util.load_onlinefood_splited()

    # Initialize
    tpot = TPOTRegressor(
        generations=5,                     # Number of generations to run the optimization process
        population_size=20,                # Number of individuals in each generation
        verbosity=2,                       # Verbosity level (0: silent, 1: minimal, 2: high)
        random_state=42,                   # Set a random seed for reproducibility
        config_dict='TPOT light',          # Use a predefined configuration dictionary
        scoring='neg_mean_squared_error',  # Metric to optimize, using negative MSE to maximize
        cv=5                               # Number of cross-validation folds
    )

    # Train
    tpot.fit(x_train, y_train)

    # Evaluate
    yhat_test = tpot.predict(x_test)
    result = util.evaluate_regression(y_test, yhat_test)  # Assuming evaluate_regression is similar to evaluate_classification

    # Save predictions
    x, _ = util.load_onlinefood()
    yhat = tpot.predict(x)
    util.save(yhat, 'regression_' + result + '.txt')

    # Optional: Export the pipeline
    tpot.export('best_regression_pipeline.py')

    return result



if __name__ == '__main__':
    tpot_classification()