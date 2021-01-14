import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import *
from collections import *
from dateutil import *
import sys


def data_nor(dn_df):
    minmaxscaler = MinMaxScaler()
    nor_data = minmaxscaler.fit_transform(dn_df)
    return pd.DataFrame(nor_data, columns=dn_df.columns)


class Model():
    def __init__(self):
        self.model = DTC(criterion='entropy', max_depth=5)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        self.predict_y = self.model.predict(X)
        return self.predict_y

if __name__ == '__main__':

    # 导入数据 测试集路径和训练集路径
    test_path = sys.argv[1]
    train_path = "./f_train.csv"
    train = pd.read_csv(train_path, encoding='gbk')
    test = pd.read_csv(test_path, encoding='gbk')

    train[['id', 'SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
           'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
           'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
           'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
           'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
           '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
           'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
           'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
           'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
           'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
           'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
           'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
           'SNP54', 'SNP55', 'ACEID', 'label']].astype(float)

    test[['id', 'SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
          'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
          'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
          'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
          'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
          '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
          'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
          'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
          'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
          'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
          'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
          'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
          'SNP54', 'SNP55', 'ACEID', 'label']].astype(float)

    # 用中位数填充
    train.fillna(train[['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
                        'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
                        'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
                        'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
                        'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
                        '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
                        'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
                        'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
                        'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
                        'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
                        'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
                        'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
                        'SNP54', 'SNP55', 'ACEID', 'label']].median(axis=0), inplace=True)

    test.fillna(test[['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
                      'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
                      'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
                      'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
                      'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
                      '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
                      'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
                      'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
                      'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
                      'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
                      'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
                      'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
                      'SNP54', 'SNP55', 'ACEID', 'label']].median(axis=0), inplace=True)

    # 将数据进行规格化
    train_dn = data_nor(train[['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
                               'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
                               'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
                               'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
                               'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
                               '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
                               'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
                               'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
                               'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
                               'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
                               'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
                               'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
                               'SNP54', 'SNP55', 'ACEID']])

    test_dn = data_nor(test[['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5',
                             'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10', 'SNP11',
                             'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17',
                             'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23',
                             'RBP4', '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类',
                             '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周', 'VAR00007',
                             'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC',
                             'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'SNP24', 'SNP25', 'SNP26',
                             'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
                             'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史', 'SNP39',
                             'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46',
                             'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53',
                             'SNP54', 'SNP55', 'ACEID']])
    train = pd.concat((train_dn, train['label']), axis=1)
    test = pd.concat((test_dn, test['label']), axis=1)

    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]

    train_col = list(train.columns.values)
    train_col.remove('label')
    X_train = train[train_col]
    y_train = train[['label']].astype(int)

    test_col = list(test.columns.values)
    test_col.remove('label')
    X_test = test[test_col]
    y_test = test['label'].astype(int)

    f_model = Model()
    f_model.train(X_train, y_train)
    
    predict_y = f_model.predict(X_test)
    total_predict = 0
    print(predict_y)

    for i in range(len(predict_y)):
        if i > 0:
            total_predict = total_predict + 1
    correct = 0
    j = 0

    for i in range(len(predict_y)):
        if (y_test[i] == predict_y[i]):
            correct = correct + 1
    total_data = 0
    for i in range(len(y_test)):
        if i > 0:
            total_data = total_data + 1

    P = correct / total_predict
    R = correct / total_data
    F1 = 2*P*R / (P+R)
    print(F1)
