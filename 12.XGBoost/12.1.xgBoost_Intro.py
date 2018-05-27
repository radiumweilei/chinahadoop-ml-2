# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np


# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
# 自定义损失函数，理论根据在回归讲义中
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()  # 一阶导数 p7.23
    h = p * (1.0 - p)  # 二阶导数 p7.47
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


# xgboost需要很少的几棵树，每棵树的层数很浅，就能得到很低的错误率； 将 max_depth 改成3看看
if __name__ == "__main__":
    # 读取数据
    data_train = xgb.DMatrix('12.agaricus_train.txt')
    data_test = xgb.DMatrix('12.agaricus_test.txt')

    # 设置参数
    # max_depth 树的深度
    # eta 讲义公示中的参数v
    # silent 1 生成过程不输出, 0 输出
    # objective  binary:logitraw 二分类
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw'}  # logitraw
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 3  # 树的个数
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0))  # 阈值 设成0.5 配合 logistic试试
    error_rate = float(error) / len(y_hat)
    print('样本总数：\t', len(y_hat))
    print('错误数目：\t%4d' % error)
    print('错误率：\t%.5f%%' % (100 * error_rate))
