import pandas as pd

def preprocess(path: str, heads: list[str], formats: list[str], isnorm: bool=True):
    """ 数据预处理 """

    # 读取数据并只保留heads里的列
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

if __name__ == '__main__':
    x, y = load_onlinefood()
    print(x, y)