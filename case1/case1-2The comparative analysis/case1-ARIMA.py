import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA




data = pd.read_excel('使用版2012-2023-2个相关因素_2位小数.xlsx')
N = data.shape[0]

#训练集个数（前train_n个数据被用作训练）
train_n =N-2

train_y = data.iloc[:train_n,1]
train_x = data.iloc[:train_n, [2,3]]
test_y  = data.iloc[train_n:,1]
test_x  = data.iloc[train_n:, [2,3]]
all_x = data.iloc[:, [2,3]]
all_y = data.iloc[:,1]

# 构建模型
model = ARIMA(train_y, order=(1, 2, 0), exog=train_x)

# 训练模型
model_fit = model.fit()

#模拟值
simulation = model_fit.predict(steps=train_n,  exog=train_x)
print(f"模拟 {train_n} 个值为：\n", simulation)
            
# 预测未来 N-train_n 个时间点的值
forecast = model_fit.forecast(steps=N-train_n, exog=test_x)
print(f"未来的 {N-train_n} 个值为：\n", forecast)

#计算APE
APE1 = (np.abs(train_y-simulation)/train_y)
APE2 = (np.abs(test_y-forecast)/test_y)
print("APE为：",APE1, APE2)
# 计算 MAPE
MAPE1 = (np.abs(train_y-simulation)/train_y).mean()
MAPE2 = (np.abs(test_y-forecast)/test_y).mean()
print("MAPE为：",MAPE1,MAPE2)


#forecast = model_fit.forecast(steps=N, exog=data.iloc[:, [2,3]])
#MAPE = (np.abs(all_y-forecast)/all_y).mean()

pd.DataFrame(simulation).to_excel('simulation_ARIMA.xlsx')
pd.DataFrame(forecast).to_excel('forecast_ARIMA.xlsx')
pd.DataFrame(APE1).to_excel('APE_ARIMA1.xlsx')
pd.DataFrame(APE2).to_excel('APE_ARIMA2.xlsx')