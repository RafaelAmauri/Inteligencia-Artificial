import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("FuelConsumptionCo2.csv")

'''
LETRA A
'''

# Para ficar mais fácil de visualizar as relações entre as variáveis
'''
plt.scatter(df['FUELCONSUMPTION_COMB'], df['ENGINESIZE'], color='red')
plt.title('FUELCONSUMPTION_COMB Vs ENGINESIZE', fontsize=14)
plt.xlabel('ENGINESIZE', fontsize=14)
plt.ylabel('FUELCONSUMPTION_COMB', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['FUELCONSUMPTION_COMB'], df['CYLINDERS'], color='red')
plt.title('FUELCONSUMPTION_COMB Vs CYLINDERS', fontsize=14)
plt.xlabel('CYLINDERS', fontsize=14)
plt.ylabel('FUELCONSUMPTION_COMB', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['CYLINDERS'], df['ENGINESIZE'], color='red')
plt.title('CYLINDERS Vs ENGINESIZE', fontsize=14)
plt.xlabel('ENGINESIZE', fontsize=14)
plt.ylabel('CYLINDERS', fontsize=14)
plt.grid(True)
plt.show()
'''

# 95% Pra treino e 5% para teste
PERCENTAGE_TRAIN_DATA = 0.95

x_train, x_test = train_test_split(df[['FUELCONSUMPTION_COMB', 'ENGINESIZE', 'CYLINDERS']], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)
y_train, y_test = train_test_split(df["CO2EMISSIONS"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)

reg = LinearRegression()
reg.fit(x_train, y_train)

prediction = reg.predict(x_test)

print("Métricas para predição com FUELCONSUMPTION_COMB, ENGINESIZE e CYLINDERS:")
my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score = {my_mse_score}")

print("\n")

'''
LETRA B
'''

# Mais mapas para mostrar a relação entre as variáveis. FUELCONSUMPTION_CITY parece ser bem mais 
# correlatado a FUELCONSUMPTION_COMB do que FUELCONSUMPTION_HWY
'''
plt.scatter(df['FUELCONSUMPTION_CITY'], df['FUELCONSUMPTION_COMB'], color='red')
plt.title('FUELCONSUMPTION_CITY Vs FUELCONSUMPTION_COMB', fontsize=14)
plt.xlabel('FUELCONSUMPTION_COMB', fontsize=14)
plt.ylabel('FUELCONSUMPTION_CITY', fontsize=14)
plt.grid(True)
plt.show()


plt.scatter(df['FUELCONSUMPTION_HWY'], df['FUELCONSUMPTION_COMB'], color='red')
plt.title('FUELCONSUMPTION_HWY Vs FUELCONSUMPTION_COMB', fontsize=14)
plt.xlabel('FUELCONSUMPTION_COMB', fontsize=14)
plt.ylabel('FUELCONSUMPTION_HWY', fontsize=14)
plt.grid(True)
plt.show()
'''

# 95% Pra treino e 5% para teste
PERCENTAGE_TRAIN_DATA = 0.95

x_train, x_test = train_test_split(df[['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'ENGINESIZE', 'CYLINDERS']], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)
y_train, y_test = train_test_split(df["CO2EMISSIONS"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)

reg = LinearRegression()
reg.fit(x_train, y_train)

prediction = reg.predict(x_test)

print("Métricas para predição com FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, ENGINESIZE e CYLINDERS:")

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score = {my_mse_score}")

# TERMINAR DE RELACIONAR AS METRICAS E EXPLICAR AS MELHORAS