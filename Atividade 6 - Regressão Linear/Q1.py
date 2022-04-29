import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import sklearn.metrics

df = pd.read_csv("FuelConsumptionCo2.csv")

'''
LETRA A
'''

for column in df.columns[0:-1]:
    print(f"Showing chart for {column}")
    plt.scatter(df[column], df["CO2EMISSIONS"])
    plt.show()

'''
LETRA B
'''

corr_matrix = df.corr().sort_values(by=['CO2EMISSIONS'], ascending=False)
print(corr_matrix)

'''
Variável mais correlacionada:

FUELCONSUMPTION_CITY
'''

'''
LETRA C
'''

# 80% Pra treino e 20% pra teste
PERCENTAGE_TRAIN_DATA = 0.8

train_x = np.array(df["FUELCONSUMPTION_CITY"][ : int(PERCENTAGE_TRAIN_DATA*len(df["FUELCONSUMPTION_CITY"]))]).reshape(-1, 1)
test_x  = np.array(df["FUELCONSUMPTION_CITY"][int(PERCENTAGE_TRAIN_DATA*len(df["FUELCONSUMPTION_CITY"])) + 1 :]).reshape(-1, 1)

train_y = np.array(df["CO2EMISSIONS"][ : int(PERCENTAGE_TRAIN_DATA*len(df["CO2EMISSIONS"]))]).reshape(-1, 1)
test_y  = np.array(df["CO2EMISSIONS"][int(PERCENTAGE_TRAIN_DATA*len(df["CO2EMISSIONS"])) + 1 :]).reshape(-1, 1)

# Fazendo a predicao utilizando train_x e train_y
reg = LinearRegression().fit(train_x, train_y)

# Passando test_x para o modelo de regressao linear
prediction = reg.predict(test_x)

# Plotting chart
plt.plot(test_y)
plt.plot(prediction)

plt.show()

my_r2_score = sklearn.metrics.r2_score(test_y, prediction)
print(f"R2 score = {my_r2_score}")

# Deu 429, mas pq? Os erros estao puxando muito pra cima?
my_mse_score = sklearn.metrics.mean_squared_error(test_y, prediction)
print(f"MSE score = {my_mse_score}")