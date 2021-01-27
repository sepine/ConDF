import os
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
