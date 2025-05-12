from math import exp

import pandas as pd
import numpy as np
from scipy.special import gamma as gamma_func


def get_fractional_series(x0: np.ndarray, zeta: float) -> np.ndarray:
    """
    计算序列的分数阶
    :param x0: 待计算分数阶的序列
    :param zeta: zeta
    :return: 分数阶序列
    """
    len_x0 = len(x0)

    # 初始化数组
    coefficient = np.zeros((len_x0, len_x0))
    fractional_series = np.zeros(len_x0)

    for k in range(len_x0):
        tmp = 0
        for i in range(k + 1):
            coefficient[k, i] = gamma_func(zeta + k - i) / (gamma_func(k - i + 1) * gamma_func(zeta))
            tmp += coefficient[k, i] * x0[i]
        fractional_series[k] = tmp

    return fractional_series


# 配置超参数

zeta, tau, gamma = 0.68467508,  0.0714406,  -0.34604633#结果可行，备用

# 读取数据
data = pd.read_excel("data/case1_prediction_data_begin2012.xlsx")
data = data.iloc[:, :]  # 从12年开始算
data = data/data.iloc[0, :]
n = data.shape[0]
train_n = n - 3

"""
一阶段：去掉后两位数据，用来最小二乘算参数
"""
# 第一列是年份，故先舍弃
X0 = data.iloc[:train_n, 1:].to_numpy()
# 计算分数阶
X1 = np.array([get_fractional_series(X0[:, 0], zeta),
               get_fractional_series(X0[:, 1], zeta),
               get_fractional_series(X0[:, 2], zeta)
               ]).T

# 构建 Y
y1 = X1[:, 0] ** (1 - gamma)
Y = np.diff(y1)

# 构建 B
Zy = (y1[1:] + y1[:-1]) / 2
M2 = (X1[1:, 1] + X1[:-1, 1]) / 2
M3 = (X1[1:, 2] + X1[:-1, 2]) / 2
TAU = (np.exp(tau) - 1) / tau * np.exp(tau * (np.arange(1, train_n)))
ONE = np.ones(train_n - 1)
B = (1 - gamma) * np.array([-Zy, M2, M3, TAU, ONE]).T

# 最小二乘法
a, b2, b3, h1, h2 = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
print(f"a = {a}, b2 = {b2}, b3 = {b3}, h1 = {h1}, h2 = {h2}")

# 求 mu
mu1 = (1-gamma)/(1+0.5*a*(1-gamma))
mu2 = (1-0.5*a*(1-gamma))/(1+0.5*a*(1-gamma))
mu3 = (h1*(1-gamma)*(exp(tau)-1))/(1+0.5*a*(1-gamma))/tau
mu4 = (h2*(1-gamma))/(1+0.5*a*(1-gamma))
print(f"mu1 = {mu1}, mu2 = {mu2}, mu3 = {mu3}, mu4 = {mu4}")

"""
二阶段：全部数据进行预测
"""
X0 = data.iloc[:, 1:].to_numpy()
X1 = np.array([get_fractional_series(X0[:, 0], zeta),
               get_fractional_series(X0[:, 1], zeta),
               get_fractional_series(X0[:, 2], zeta)
               ]).T

# 求 hat y1(1为上标的那个haty^1)
hat_y1 = [y1[0]]
for m in range(2, n + 1):
    hat_ym = 0
    hat_ym += mu2 ** (m - 1) * y1[0]
    for v in range(1, m):
        hat_ym += mu2 ** (v - 1) * mu1 * (b2 * X1[m - v, 1] + b3 * X1[m - v, 2])
    for w in range(0, m - 1):
        hat_ym += mu2 ** w * (mu4 + mu3 * exp(tau * (m - w - 2)))
    hat_y1.append(hat_ym)
print("hat_y1: \n", hat_y1)

hat_x1 = np.array(hat_y1) ** (1 / (1 - gamma))
print("hat_x1: \n", hat_x1)

hat_x0 = []
for k in range(2, n + 1):
    hat_x0_m = 0
    for j in range(0, k):
        hat_x0_m += (-1) ** j * gamma_func(zeta + 1) / (gamma_func(j + 1) * gamma_func(zeta - j + 1)) * hat_x1[k - j - 1]
    hat_x0.append(hat_x0_m)
print("hat_x0: \n")
for year, i in zip(range(2013, 2027), hat_x0):
    print(year, i)
pd.DataFrame(hat_x0).to_excel('case1-future forecasting.xlsx')