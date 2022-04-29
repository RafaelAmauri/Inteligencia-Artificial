import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("FuelConsumptionCo2.csv")

# Ordenando de acordo com a variável
df = df.sort_values(by="ENGINESIZE")

# 95% Pra treino e 5% para teste
PERCENTAGE_TRAIN_DATA = 0.95

x_train, x_test = train_test_split(df["ENGINESIZE"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)
y_train, y_test = train_test_split(df["CO2EMISSIONS"], train_size=PERCENTAGE_TRAIN_DATA, shuffle=False)

x_train = x_train.to_numpy().reshape(-1, 1)
x_test  = x_test.to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test  = y_test.to_numpy().reshape(-1, 1)

'''
Teste mock sem utilizar PolynomialFeatures só pra ver o acerto do modelo de regressao linear normal
'''

reg = LinearRegression().fit(x_train, y_train)

prediction = reg.predict(x_test)

print("Métricas da regressao linear normal:")

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score utilizando só o modelo de regressão = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score utilizando só o modelo de regressão = {my_mse_score}\n")

# Plotting chart
plt.scatter(x_train, y_train, c="orange", label = "Training Data")
plt.scatter(x_test, y_test, c="green", label="Test data")
plt.plot(x_test, prediction, c="red", label="Prediction")
plt.plot(x_train, reg.predict(x_train), c="blue", label="Train")
plt.legend()
plt.show()

'''
Primeiro teste com PolynomialFeatures
'''

# Utilizando grau = 2 para o primeiro teste
degree1 = 2

poly = PolynomialFeatures(degree=degree1)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Fazendo a predicao utilizando x_train e y_train
reg = LinearRegression().fit(x_train_poly, y_train)

# Passando x_test para o modelo de regressao linear
prediction = reg.predict(x_test_poly)

print(f"Métricas utilizando degree = {degree1}:")

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score com grau {degree1} = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score com grau {degree1} = {my_mse_score}\n")

# Plotting chart
plt.scatter(x_train, y_train, c="orange", label = "Training Data")
plt.scatter(x_test, y_test, c="green", label="Test data")
plt.plot(x_test, prediction, c="red", label="Prediction")
plt.plot(x_train, reg.predict(x_train_poly), c="blue", label="Train")
plt.legend()
plt.show()

'''
Teste 2
'''

# Utilizando grau = 3 para o proximo teste
degree2 = 3

poly = PolynomialFeatures(degree2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly  = poly.transform(x_test)

# Fazendo a predicao utilizando x_train e y_train
reg = LinearRegression().fit(x_train_poly, y_train)

# Passando x_test para o modelo de regressao linear
prediction = reg.predict(x_test_poly)

print(f"Métricas utilizando degree = {degree2}:")

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score com grau {degree2} = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score com grau {degree2} = {my_mse_score}\n")

# Plotting chart
plt.scatter(x_train, y_train, c="orange", label = "Training Data")
plt.scatter(x_test, y_test, c="green", label="Test data")
plt.plot(x_test, prediction, c="red", label="Prediction")
plt.plot(x_train, reg.predict(x_train_poly), c="blue", label="Train")
plt.legend()
plt.show()

'''
Teste 3
'''

# Utilizando grau = 4 para o proximo teste
degree3 = 4

poly = PolynomialFeatures(degree3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly  = poly.transform(x_test)

# Fazendo a predicao utilizando x_train e y_train
reg = LinearRegression().fit(x_train_poly, y_train)

# Passando x_test para o modelo de regressao linear
prediction = reg.predict(x_test_poly)

print(f"Métricas utilizando degree = {degree3}:")

my_r2_score = sklearn.metrics.r2_score(y_test, prediction)
print(f"R2 score com grau {degree3} = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(y_test, prediction)
print(f"MSE score com grau {degree3} = {my_mse_score}")

# Plotting chart
plt.scatter(x_train, y_train, c="orange", label = "Training Data")
plt.scatter(x_test, y_test, c="green", label="Test data")
plt.plot(x_test, prediction, c="red", label="Prediction")
plt.plot(x_train, reg.predict(x_train_poly), c="blue", label="Train")
plt.legend()
plt.show()

'''
A princípio o modelo parece não ter melhorado muito após o pré-processamento feito com PolynomialFeatures.
O r2_score e o MSE mudaram um pouco, mas ainda estão longe de valores bons.

A minha hipótese é que talvez seja o formato da distribuição dos dados em ENGINESIZE que esteja causando isso, porque os dados estão MUITO mais 
espalhados do que em outras variáveis do dataframe, o que indica que o número de outliers em ENGINESIZE deve ser muito alto. 
Ao meu ver, o resultado ruim não é uma fraqueza do modelo ou do pré-processamento, mas da variável ENGINESIZE que é uma péssima candidata para ser prevista, porque os dados
nela estão muito espalhados para fazer uma previsão precisa utilizando esses modelos :P
'''
