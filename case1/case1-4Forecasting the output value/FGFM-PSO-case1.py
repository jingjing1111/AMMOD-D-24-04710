from sko.PSO import PSO
from math import exp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def func(params):
    # 得到该次超参数
    zeta, tau, gamma = params
    """
    一阶段：去掉后两位数据，用来算出
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
    try:
        a, b2, b3, h1, h2 = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    except Exception as e:
        return 1

    #L2 = 0.001 * (a ** 2 + b2 ** 2 + b3 ** 2)
    L2=0
    # 求 mu
    mu1 = (1 - gamma) / (1 + 0.5 * a * (1 - gamma))
    mu2 = (1 - 0.5 * a * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma))
    mu3 = (h1 * (1 - gamma) * (exp(tau) - 1)) / (1 + 0.5 * a * (1 - gamma)) / tau
    mu4 = (h2 * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma))

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

    hat_x1 = np.array(hat_y1) ** (1 / (1 - gamma))
    hat_x0 = []
    for k in range(2, n + 1):
        hat_x0_m = 0
        for j in range(0, k):
            hat_x0_m += (-1) ** j * gamma_func(zeta + 1) / (gamma_func(j + 1) * gamma_func(zeta - j + 1)) * hat_x1[
                k - j - 1]
        hat_x0.append(hat_x0_m)

    original_x1 = data.iloc[1:, 1]
    APE = np.abs(hat_x0 - original_x1) / original_x1
    if APE.isnull().any():
        return 1
    else:
        return (np.sqrt(np.mean(APE ** 2)) + L2)  # 均方根误差和作为损失函数#最终7-4用这个


"""
w:  惯性权重
c1: 个体学习因子 控制粒子向自身历史最优位置移动的强度   c1太大，粒子会过于依赖自身经验，导致过早收敛
c2: 社会学习因子 控制粒子向群体最优位置移动的强度。     c2太大，可能会过早陷入局部最优
"""


class DynamicPSO(PSO):
    def __init__(self, func, dim, pop, iter_sum, lb, ub, w_init=0.9, c1_init=2.5, c2_init=0.5):
        super().__init__(func, dim, pop, 1, lb, ub, w=w_init, c1=c1_init, c2=c2_init)
        self.iter_sum = iter_sum
        self.w_init = w_init  # 初始惯性权重
        self.c1_init = c1_init  # 初始个体学习因子
        self.c2_init = c2_init  # 初始社会学习因子

    def run(self):
        for iter_num in range(self.iter_sum):
            # 动态调整参数（示例：线性递减惯性权重，学习因子自适应）
            self.w = self.w_init - 0.1 * (iter_num / self.iter_sum)  # 从0.9递减到0.4
            self.c1 = self.c1_init - 0.1 * (iter_num / self.iter_sum)   # 从2.5递减到0.5
            self.c2 = self.c2_init + 0.1 * (iter_num / self.iter_sum)   # 从0.5递增到2.5
            super().run()  # 调用父类的单次迭代更新
        return self.gbest_x, self.gbest_y


for _ in range(5):
    # 读取数据
    data = pd.read_excel("data/case1_train_data_begin2012.xlsx")
    data = data.iloc[:, :]  # 从第i年开始算
    data = data/data.iloc[0, :]
    n = data.shape[0]
    train_n = n
    #train_n = n - 2
    pso = DynamicPSO(func=func, dim=3, pop=500, iter_sum=3000,
                     lb=[0, -10, -1], ub=[1, 10, 1])
    best_x, best_y = pso.run()
    print(f"i={7} , 最优解:", best_x, "目标值:", best_y)

    plt.plot(pso.gbest_y_hist)
    plt.show()
