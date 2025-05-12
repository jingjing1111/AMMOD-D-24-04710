import numpy as np
import pandas as pd


gamma =0.15406108


# 设置打印选项，禁用科学计数法
np.set_printoptions(suppress=True)

# 原始数据
data = pd.read_excel("使用版2012-2023-2个相关因素_2位小数.xlsx", sheet_name = 'Sheet1')
N = data.shape[0]


train_n = N-2
X0 = data.iloc[:train_n, 1:].to_numpy()


# 一阶累加生成
X1 = np.cumsum(X0, 0)
print("一阶累加生成：\n", X1)

# 对第一列一阶累加生成求 1-gamma 次幂
y1 = X1[:, 0]**(1-gamma)
print("第一列一阶累加生成求 1-gamma 次幂: \n", y1)
# 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
Zy = (y1[1:]+y1[:-1])/2
Z2 = (X1[1:, 1]+X1[:-1, 1])/2
Z3 = (X1[1:, 2]+X1[:-1, 2])/2


#print("Zy:\n",list(Zy),"\nZ2:\n",list(Z2),"\nZ3:\n",list(Z3),"\nZ23:\n")

B = np.array([-Zy, Z2, Z3]).T
ones_column = np.ones((B.shape[0], 1))
B =  (1-gamma)*np.hstack((B, ones_column))

A = y1[1:]-y1[:-1]
# 共五个值，分别为a, b2, b3, b23, b1
hat_a = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), A)
a, b2, b3, b1 = hat_a

print("共4个值，分别为a, b2, b3,  b1:\n", list(hat_a))


# 这里重新获取下原数据，以实现所有数据的预测
# 第一列是年份，故先舍弃
X0 = data.iloc[:, 1:].to_numpy()

X1 = np.cumsum(X0, 0)
# 对第一列一阶累加生成求 1-gamma 次幂
y1 = X1[:, 0]**(1-gamma)

# 根据 y1 求 Zy，再分别根据一阶累加生成的第二三列生成Z2、Z3
Zy = (y1[1:]+y1[:-1])/2
Z2 = (X1[1:, 1]+X1[:-1, 1])/2
Z3 = (X1[1:, 2]+X1[:-1, 2])/2
hat_y1 = [y1[0]]



for i in range(N-1):
    hat_y1.append(
        (1 - gamma) / (1 + 0.5 * a * (1 - gamma)) * (b2 * Z2[i] + b3 * Z3[i]) +
         (1 - 0.5 * a * (1 - gamma)) / (1 + 0.5 * a * (1 - gamma)) * hat_y1[-1] +
        (1 - gamma) * b1 / (1 + 0.5 * a * (1 - gamma))
    )

hat_y1 = np.array(hat_y1)
print("hat_y1: \n", list(hat_y1))

#hat_x1 = np.exp(np.log(hat_y1)/(1-gamma))
hat_x1 = (hat_y1)**(1/(1-gamma))
print("hat_x1: \n", list(hat_x1))

hat_x0 = np.diff(hat_x1)
print("hat_x0: \n", list(hat_x0))
original_x1 = data.iloc[1:, 1]
APE = np.abs(hat_x0-original_x1)/original_x1
print("APE: \n", list(APE))

MAPE = np.mean(APE)

print("MAPE: ", MAPE)
pd.DataFrame(hat_x0).to_excel('NGBM-hat_x0-case1.xlsx')
pd.DataFrame(APE).to_excel('NGBM-APE-case1.xlsx')
