import pandas as pd

def preprocess(path: str, heads: list[str], formats: list[str], isnorm: bool=True):

    # 读取
    df = pd.read_csv(path)[heads]

    # 预处理
    assert len(heads) == len(formats)
    for key, format in zip(heads, formats):
        if format == 'num':
            df[key] = pd.to_numeric(df[key], errors='coerce')
        elif format == 'str':
            df[key] = pd.factorize(df[key])[0]
        elif format == 'pos':
            df[key] = df[key].map({'Positive': 1, 'Negative': 0})
        else:
            raise ValueError(f"key {key} not in {df.columns}")

    # 填充NaN
    df.fillna(0, inplace=True)  # 将0替换为df.mean()可以改为均值填充

    # 获取x, y
    x = df[heads[:-1]].astype(float).to_numpy()
    y = df[heads[-1]].astype(int).to_numpy()

    # 归一化
    if isnorm:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, y

def load_onlinefood():
    return preprocess(
        "data/onlinefoods.csv",
        ['Age','Gender','Marital Status','Occupation','Monthly Income','Educational Qualifications','Family size','latitude','longitude','Pin code','Output', 'Feedback'],
        ['num', 'str', 'str', 'str', 'str', 'str', 'num', 'num', 'num', 'num', 'str', 'pos']
    )

def load_onlinefood_splited():
    from sklearn.model_selection import train_test_split
    x, y = load_onlinefood()
    return train_test_split(x, y, test_size=0.2, random_state=42)

def evaluate(y, y_hat):
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
    auc = roc_auc_score(y, y_hat)
    f1 = f1_score(y, y_hat>0.5, average='macro')
    acc = accuracy_score(y, y_hat>0.5)
    rec = recall_score(y, y_hat>0.5)
    print(f"auc: {auc:.2%}, f1: {f1:.2%}, acc: {acc:.2%}, rec: {rec:.2%}")
    return f"auc_{auc:.2f}_f1_{f1:.2f}"

def save(y, path):
    with open(path, 'w') as f:
        for i in y:
            f.write(f'{i}\n')
    print(f'data saved to {path}')

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_onlinefood_splited()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)