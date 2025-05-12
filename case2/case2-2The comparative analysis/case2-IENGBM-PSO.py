from sko.PSO import PSO
import pandas as pd
import numpy as np


def func(XX):
    # 原始数据
    gamma = XX[0]

    X1 = X1_all[:train_n, :]

    # 对第一列一阶累加生成求 1-gamma 次幂
    y1 = X1[:, 0] ** (1 - gamma)

    # 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
    Zy = (y1[1:] + y1[:-1]) / 2
    Z2 = (X1[1:, 1] + X1[:-1, 1]) / 2
    Z3 = (X1[1:, 2] + X1[:-1, 2]) / 2

    # 根据一阶累加生成的第二三列相乘得到序列 pq， 进而得到 Z23
    pq = X1[:, 1] * X1[:, 2]
    Z23 = (pq[1:] + pq[:-1]) / 2

    B = np.array([-Zy, Z2, Z3, Z23]).T
    ones_column = np.ones((B.shape[0], 1))
    B = (1 - gamma) * np.hstack((B, ones_column))

    A = y1[1:] - y1[:-1]
    # 共五个值，分别为a, b2, b3, b23, b1
    try:
        hat_a = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), A)
    except:
        return 1
    a, b2, b3, b23, b1 = hat_a

    # 这里重新获取下原数据，以实现所有数据的预测
    # 一阶累加生成
    X1 = X1_all

    # 对第一列一阶累加生成求 1-gamma 次幂
    y1 = X1[:, 0] ** (1 - gamma)
    # 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
    Zy = (y1[1:] + y1[:-1]) / 2
    Z2 = (X1[1:, 1] + X1[:-1, 1]) / 2
    Z3 = (X1[1:, 2] + X1[:-1, 2]) / 2

    # 根据一阶累加生成的第二三列相乘得到序列 pq， 进而得到 Z23
    pq = X1[:, 1] * X1[:, 2]
    Z23 = (pq[1:] + pq[:-1]) / 2

    hat_y1 = [y1[0]]
    for i in range(N - 1):
        hat_y1.append(
            (1 - gamma) / (1 + 0.5 * a * (1 - gamma)) * (b2 * Z2[i] + b3 * Z3[i]) +
            (1 - 0.5 * a * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma)) * hat_y1[-1] +
            (1 - gamma) * b1 / (1 + 0.5 * a * (1 - gamma)) +
            (1 - gamma) / (1 + 0.5 * a * (1 - gamma)) * (b23 * Z23[i])
        )

    hat_y1 = np.array(hat_y1)
    hat_x1 = np.exp(np.log(hat_y1) / (1 - gamma))
    hat_x0 = np.diff(hat_x1)
    APE = (np.abs(hat_x0 - original_y) / original_y)
    MSE = np.mean(APE ** 2)
    MAPE = np.mean(np.abs(hat_x0 - original_y) / original_y)
    # return MAPE
    return np.sqrt(np.mean(APE ** 2))
    #return MAPE
    # return MAPE + gamma**2
    # return np.sqrt(np.mean(APE**2))
    # return MSE


data = pd.read_excel("京津冀3省2014-2023使用版数据-2位小数.xlsx")
data = data.iloc[:, :]
N = data.shape[0]
train_n = N - 2

# 第一列是年份，故先舍弃
X0 = data.iloc[:, 1:].to_numpy()
# 一阶累加生成
X1_all = np.cumsum(X0, 0)
# 用来计算MAPE的原始真实Y值 （一般来说，PSO寻优函数中尽量不要有切片操作，毕竟寻优函数中的内容会频繁执行）
original_y = data.iloc[1:, 1]
# 调用粒子群，指定上述五个变量的取值范围，设置粒子群参数，最终得到最优的 gamma
pso = PSO(func=func, n_dim=1, pop=200, max_iter=400, lb=[0], ub=[10])
pso.record_mode = True
pso.run()
print('best_gamma is ', pso.gbest_x, 'best_globalX is', pso.gbest_y)
