from sko.PSO import PSO
import pandas as pd
import numpy as np

def func(XX):
    # 原始数据
    gamma = XX[0]
    X0 = data.iloc[:train_n, 1:].to_numpy()  # 第一列是年份，故先舍弃
    X1 = np.cumsum(X0, 0)
    # 对第一列一阶累加生成求 1-gamma 次幂
    y1 = X1[:, 0] ** (1 - gamma)

    # 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
    Zy = (y1[1:] + y1[:-1]) / 2
    Z2 = (X1[1:, 1] + X1[:-1, 1]) / 2
    Z3 = (X1[1:, 2] + X1[:-1, 2]) / 2


    B = np.array([-Zy, Z2, Z3]).T
    ones_column = np.ones((B.shape[0], 1))
    B = (1-gamma)*np.hstack((B, ones_column))

    A = y1[1:] - y1[:-1]
    # 共五个值，分别为a, b2, b3, b23, b1
    try:
        hat_a = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), A)
    except:
        return 1
    a, b2, b3, b1 = hat_a

    # 重新读入完整data
    X0 = data.iloc[:, 1:].to_numpy()  # 第一列是年份，故先舍弃
    X1 = np.cumsum(X0, 0)
    # 对第一列一阶累加生成求 1-gamma 次幂
    y1 = X1[:, 0] ** (1 - gamma)

    # 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
    Zy = (y1[1:] + y1[:-1]) / 2
    Z2 = (X1[1:, 1] + X1[:-1, 1]) / 2
    Z3 = (X1[1:, 2] + X1[:-1, 2]) / 2

    hat_y1 = [y1[0]]
    for i in range(N-1):
        hat_y1.append(
            (1 - gamma) / (1 + 0.5 * a * (1 - gamma)) * (b2 * Z2[i] + b3 * Z3[i]) +
            (1 - 0.5 * a * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma)) * hat_y1[-1] +
            (1 - gamma) * b1 / (1 + 0.5 * a * (1 - gamma))
        )

    hat_y1 = np.array(hat_y1)
    hat_x1 = np.exp(np.log(hat_y1) / (1 - gamma))
    hat_x0 = np.diff(hat_x1)
    original_x1 = data.iloc[1:, 1]

    APE = np.abs(hat_x0 - original_x1) / original_x1
    return np.sqrt(np.mean(APE ** 2))

data = pd.read_excel("京津冀3省2014-2023使用版数据-2位小数.xlsx")
N = data.shape[0]
train_n = N - 2

# 调用粒子群，指定上述五个变量的取值范围，设置粒子群参数，最终得到最优的 gamma
pso = PSO(func=func, n_dim=1, pop=100, max_iter=200, lb=[0], ub=[10])

pso.record_mode = True
pso.run()
print('best_gamma is ', pso.gbest_x, 'best_MAPE is', pso.gbest_y)
