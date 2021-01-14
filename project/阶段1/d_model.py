import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn.linear_model import *
from collections import *
from dateutil import *
import sklearn.metrics
import sys
from sklearn.tree import DecisionTreeClassifier as DTC


def data_nor(dn_df):
    minmaxscaler = MinMaxScaler()
    nor_data = minmaxscaler.fit_transform(dn_df)
    return pd.DataFrame(nor_data, columns=dn_df.columns)


class Model():
    def __init__(self):
        self.model = DTC(criterion='entropy',max_depth=5)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        self.predict_y = self.model.predict(X)
        return self.predict_y


if __name__ == '__main__':

    # 导入数据 测试集路径和训练集路径
    test_path = sys.argv[1]
    train_path = "./d_train.csv"
    train = pd.read_csv(train_path, encoding='gbk')
    test = pd.read_csv(test_path, encoding='gbk')

    # 综合分析可以认为结果和体检日期、性别无关，故删去
    del(train['体检日期'])
    del(train['性别'])

    del(test['体检日期'])
    del(test['性别'])

    # 用中位数填充
    train = train.drop(['id'], axis=1)
    train.fillna(train.median(axis=0), inplace=True)

    test = test.drop(['id'], axis=1)
    test.fillna(test.median(axis=0), inplace=True)

    # 将年龄进行离散化
    train['年龄'].astype(int)
    train['年龄'].value_counts().sort_index().head().plot.bar()

    test['年龄'].astype(int)
    test['年龄'].value_counts().sort_index().head().plot.bar()

    # 将数据进行规格化
    train = data_nor(train)
    test = data_nor(test)

    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]

    print(train)
    print(np.isnan(train).any())
    print(np.isnan(test).any())

    train_col = list(train.columns.values)
    train_col.remove('血糖')
    X = train[train_col]
    y = train[['血糖']].astype(int)
    X_train = X
    y_train = y

    test_col = list(test.columns.values)
    test_col.remove('血糖')
    X = test[test_col]
    y = test[['血糖']].astype(int)
    X_test = X
    y_test = y

    d_model = Model()
    d_model.train(X_train, y_train)
    predict_y = d_model.predict(X_test)

    # 平均绝对值误差
    mean_absolute_error = sklearn.metrics.mean_absolute_error(y_test, predict_y)
    print("平均绝对值误差", mean_absolute_error)
    # 均方差
    mean_squared_error = sklearn.metrics.mean_squared_error(y_test, predict_y)
    print("均方差", mean_squared_error)
    # 中值绝对误差
    median_absolute_error = sklearn.metrics.median_absolute_error(y_test, predict_y)
    print("中值绝对值误差", median_absolute_error)