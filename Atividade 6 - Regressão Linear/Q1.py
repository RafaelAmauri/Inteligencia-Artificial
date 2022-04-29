import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

# Ordenando de acordo com a variável
df = df.sort_values(by="FUELCONSUMPTION_CITY")

# 95% Pra treino e 5% pra teste
PERCENTAGE_TRAIN_DATA = 0.95

x_train, x_test = train_test_split(df["FUELCONSUMPTION_CITY"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)
y_train, y_test = train_test_split(df["CO2EMISSIONS"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)

x_train = x_train.to_numpy().reshape(-1, 1)
x_test  = x_test.to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test  = y_test.to_numpy().reshape(-1, 1)

# Fazendo a predicao utilizando train_x e train_y
reg = LinearRegression().fit(x_train, y_train)

# Passando test_x para o modelo de regressao linear
prediction = reg.predict(x_test)

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score = {my_mse_score}")

# Montando o grafico
plt.scatter(x_train, y_train, c="orange", label = "Training Data")
plt.scatter(x_test, y_test, c="green", label="Test data")
plt.plot(x_test, prediction, c="red", label="Prediction")
plt.plot(x_train, reg.predict(x_train), c="blue", label="Train")
plt.legend()
plt.show()