import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats as st


def read_xlsx(path):
    df = pd.read_excel(path)
    return df


def split_dataset(df: DataFrame, values=[1, 0], key='flag') -> tuple:

    """
    Partition the dataset into two symmetric divides.
    :param df:
    :param values:
    :param key:
    :return:
    """
    positives, negatives = (df[df[key] == v] for v in values)
    (p_train, p_test), (n_train, n_test) = map(
        lambda dataset: train_test_split(dataset, test_size=0.5, shuffle=True, random_state=None),
        (positives, negatives),
    )

    return (p_train.append(n_train), p_test.append(n_test))


# def split_train_valid(Xtest, values=[1, 0]) -> tuple:
#
#     """
#     Partition the train set into two symmetric divides  ==>  train set and valid set
#     :param Xtest: origin train set
#     :param values:
#     :return:
#     """
#     positives, negatives = (Xtest[Xtest[-1:] == v] for v in values)
#     (p_train, p_test), (n_train, n_test) = map(
#         lambda dataset: train_test_split(dataset, test_size=0.5, shuffle=True, random_state=None),
#         (positives, negatives),
#     )
#
#     return (p_train.append(n_train), p_test.append(n_test))


def split_train_test_label(x_train, x_test):
    """
    Split the features and labels
    :param x_train:
    :param x_test:
    :return:
    """
    x_train, x_test = x_train.loc[:].values, x_test.loc[:].values
    y_train = x_train[:, [-1]]
    x_train = x_train[:, : -1]
    y_test = x_test[:, [-1]]
    x_test = x_test[:, : -1]

    return x_train, x_test, y_train, y_test


def z_score(X_train, X_test):
    """
    standard the data
    :param X_train:
    :param X_test:
    :return:
    """
    scaler_1 = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_trian_scaled = scaler_1.fit_transform(X_train)
    scaler_2 = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_test_scaled = scaler_2.fit_transform(X_test)
    return X_trian_scaled, X_test_scaled


# def transform(y):
#     """
#     标签转化为1和-1
#     :param y:
#     :return:
#     """
#     y[y != '1'] = '-1'
#     y = y.astype(int)
#     return y

def dump_dataframe(path, df):
    'Save an dataframe in three files: data.csv, attr.txt and name.nfo.'
    print(path)
    data_file = new_path(path, 'csv')
    df.to_csv(data_file, header=False, index=False)
    return data_file


def new_path(path: str, ext='') -> str:
    'Return the original path, but with a different suffix name.'
    path = list(os.path.splitext(path))
    path[1] = ext
    return ''.join(path)


if __name__ == '__main__':

    # 读取数据
    df = read_xlsx('../../datasets/codec/codec.xlsx')

    train, test = split_dataset(df)
    Xtrain, Xtest, ytrain, ytest = split_train_test_label(train, test)
    x_train_scaled, x_test_scaled = z_score(Xtrain, Xtest)

    # 检验高斯分布（正态分布）
    # columns = df.columns.tolist()
    # for col in columns:
    #     _, p = st.shapiro(df[col])
    #     if p > 0.05:
    #         print('True')
    #     else:
    #         print('False')

    for i in range(len(x_train_scaled[0])):
        _, p = st.shapiro(x_train_scaled[:, i])
        if p > 0.05:
            print('True')
        else:
            print('False')

    # print(df.shape)
    # print('初始')
    # print(df.head())
    # print('划分')
    # x_train, y_trian, x_test, y_test = split_dataset(df)


    # 划分数据集
    # x_train, x_test = split_dataset(df)
    # x_train1, x_valid = split_dataset(x_train)
    #x_train, x_test, y_train, y_test = split_trian_test_label(x_train, x_test)
    #x_train.sort_values('ST01', inplace=True)
    #x_test.sort_values('ST01', inplace=True)
    # print(x_train1.shape)
    # print(x_valid.shape)
    # print(x_test.shape)
    # print(x_train1[0: 5])
    # print(x_valid[0: 5])
    # print(x_test[0: 5])

    # x_train, x_test = split_dataset(df)
    # x_train, x_test, y_train, y_test = split_trian_test_label(x_train, x_test)
    # print('第二次')

    #x_train.sort_values('ST01', inplace=True)
    #x_test.sort_values('ST01', inplace=True)
    # print(x_train[0: ])
    # print(x_test[0: ])

    # print('特征')
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_train)
    # print(x_test)
    #
    # print('标签')
    # print(y_train.shape)
    # print(y_test.shape)
    # print(y_train)
    # print(y_test)

    # 标准化
    # x_train_scaled, x_test_scaled = z_score(x_train, x_test)
    # print('标准化')
    # print(x_train_scaled.shape)
    # print(x_test_scaled.shape)
    # print(x_train_scaled)
    # print(x_test_scaled)

    # x_train = np.array([[1., -1., 2.],
    #                     [2., 0., 0.],
    #                     [0., 1., -1.]])
    # x_test = np.array([[1., -1., 2.],
    #                     [2., 0., 0.],
    #                     [0., 1., -1.]])
    # x_train_scaled, x_test_scaled = z_score(x_train, x_test)

    # print('均值', x_train_scaled[:, 0].mean())
    # print('标准差', x_train_scaled[:, 0].var())
    # x_train_scaled, x_test_scaled = np.array(x_train_scaled), np.array(x_test_scaled)
    # print('均值', np.mean(x_train_scaled[:, 0]))
    # print('标准差', np.std(x_train_scaled[:, 0]))

