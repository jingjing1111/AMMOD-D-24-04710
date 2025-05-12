from sko.PSO import PSO
from math import exp

import pandas as pd
import numpy as np
from scipy.special import gamma as GAMMA

import warnings
# 忽略警告
warnings.filterwarnings('ignore')

def get_fractional_series(X0, zeta):
    """
    计算序列的分数阶
    :param X0: 待计算分数阶的序列
    :param zeta: zeta
    :return: 分数阶序列
    """
    len_x0 = len(X0)

    # 初始化数组
    xishu = np.zeros((len_x0, len_x0))
    Xr = np.zeros(len_x0)

    for k in range(len_x0):
        tmp = 0
        for i in range(k + 1):
            xishu[k, i] = GAMMA(zeta + k - i) / (GAMMA(k - i + 1) * GAMMA(zeta))
            tmp += xishu[k, i] * X0[i]
        Xr[k] = tmp

    return Xr


def func(XX):
    zeta, tau, gamma = XX

    # 第一列是年份，故先舍弃
    X0 = data.iloc[:train_n, 1:].to_numpy()
    X1 = np.array([get_fractional_series(X0[:, 0], zeta),
                   get_fractional_series(X0[:, 1], zeta),
                   get_fractional_series(X0[:, 2], zeta)]).T

    y1 = X1[:, 0] ** (1 - gamma)

    Zy = (y1[1:] + y1[:-1]) / 2
    M2 = (X1[1:, 1] + X1[:-1, 1]) / 2
    M3 = (X1[1:, 2] + X1[:-1, 2]) / 2

    # 构建 B
    TAU = (np.exp(tau) - 1) / tau * np.exp(tau * (np.arange(1, train_n)))
    ONE = np.ones(train_n - 1)
    B = (1 - gamma) * np.array([-Zy, M2, M3, TAU, ONE]).T

    # 构建 Y
    Y = np.diff(y1)

    # 最小二乘法
    # 最小二乘法
    try:
        a, b2, b3, h1, h2 = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    except:
        return 1

    L2 = 0.001 * (a ** 2 + b2 ** 2 + b3 ** 2)

    # 求 mu
    mu1 = (1 - gamma) / (1 + 0.5 * a * (1 - gamma))
    mu2 = (1 - 0.5 * a * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma))
    mu3 = (h1 * (1 - gamma) * (exp(tau) - 1)) / (1 + 0.5 * a * (1 - gamma)) / tau
    mu4 = (h2 * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma))

    # 这里重新获取下原数据，以实现所有数据的预测
    # 第一列是年份，故先舍弃
    X0 = data.iloc[:, 1:].to_numpy()
    X1 = np.array([get_fractional_series(X0[:, 0], zeta),
                   get_fractional_series(X0[:, 1], zeta),
                   get_fractional_series(X0[:, 2], zeta)]).T

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

    hat_x1 = np.array(hat_y1) ** (1 / (1 - gamma))

    hat_x0 = []
    for k in range(2, n + 1):
        hat_x0_m = 0
        for j in range(0, k):
            hat_x0_m += (-1) ** j * GAMMA(zeta + 1) / (GAMMA(j + 1) * GAMMA(zeta - j + 1)) * hat_x1[k - j - 1]
        hat_x0.append(hat_x0_m)

    original_x1 = data.iloc[1:, 1]
    APE = np.abs(hat_x0 - original_x1) / original_x1
    if APE.isnull().any():
        return 1
    else:
        return np.sqrt(np.mean(APE ** 2))+L2  # 均方根误差和作为损失函数



for i in range(12):
    data = pd.read_excel("case1-使用版2005-2023-2个相关因素_2位小数.xlsx")
    data = data.iloc[i:, :]
    n = data.shape[0]
    train_n = n-2

    # 调用粒子群，指定上述五个变量的取值范围，设置粒子群参数，最终得到最优的 zeta, tau, gamma
    pso = PSO(func=func, n_dim=3, pop=100, max_iter=200, lb=[0, -10, -10], ub=[1, 10, 10])
    pso.record_mode = True
    pso.run()
    print( 'i=', i , 'best_zeta, tau, gamma is ', pso.gbest_x, 'best_MAPE is', pso.gbest_y)
