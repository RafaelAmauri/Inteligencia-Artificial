from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics

df = pd.read_csv("FuelConsumptionCo2.csv")

# 80% Pra treino e 20% pra teste de novo
PERCENTAGE_TRAIN_DATA = 0.8

train_x = np.array(df["ENGINESIZE"][ : int(PERCENTAGE_TRAIN_DATA*len(df["ENGINESIZE"]))]).reshape(-1, 1)
test_x  = np.array(df["ENGINESIZE"][int(PERCENTAGE_TRAIN_DATA*len(df["ENGINESIZE"])) + 1 :]).reshape(-1, 1)

train_y = np.array(df["CO2EMISSIONS"][ : int(PERCENTAGE_TRAIN_DATA*len(df["CO2EMISSIONS"]))]).reshape(-1, 1)
test_y  = np.array(df["CO2EMISSIONS"][int(PERCENTAGE_TRAIN_DATA*len(df["CO2EMISSIONS"])) + 1 :]).reshape(-1, 1)

'''
Teste mock sem utilizar PolynomialFeatures só pra ver o acerto do modelo normal
'''

reg = LinearRegression().fit(train_x, train_y)

prediction = reg.predict(test_x)

my_r2_score = sklearn.metrics.r2_score(test_y, prediction)
print(f"R2 score utilizando só o modelo de regressão = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(test_y, prediction)
print(f"MSE score utilizando só o modelo de regressão = {my_mse_score}\n")

'''
Primeiro teste com PolynomialFeatures
'''

# Utilizando grau = 2 para o primeiro teste
degree1 = 2

train_x_poly = PolynomialFeatures(degree1)
train_x_poly = train_x_poly.fit_transform(train_x)


# Fazendo a predicao utilizando train_x e train_y
reg = LinearRegression().fit(train_x_poly, train_y)

test_x_poly = PolynomialFeatures(degree1)
test_x_poly = test_x_poly.fit_transform(test_x)

# Passando test_x para o modelo de regressao linear
prediction = reg.predict(test_x_poly)

my_r2_score = sklearn.metrics.r2_score(test_y, prediction)
print(f"R2 score com grau {degree1} = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(test_y, prediction)
print(f"MSE score com grau {degree1} = {my_mse_score}\n")

'''
Teste 2
'''

# Utilizando grau = 3 para o proximo teste
degree2 = 3

train_x_poly = PolynomialFeatures(degree2)
train_x_poly = train_x_poly.fit_transform(train_x)


# Fazendo a predicao utilizando train_x e train_y
reg = LinearRegression().fit(train_x_poly, train_y)

test_x_poly = PolynomialFeatures(degree2)
test_x_poly = test_x_poly.fit_transform(test_x)

# Passando test_x para o modelo de regressao linear
prediction = reg.predict(test_x_poly)


my_r2_score = sklearn.metrics.r2_score(test_y, prediction)
print(f"R2 score com grau {degree2} = {my_r2_score}")

my_mse_score = sklearn.metrics.mean_squared_error(test_y, prediction)
print(f"MSE score com grau {degree2} = {my_mse_score}")

'''
A princípio, o modelo parece não ter respondido muito bem ao pré-processamento feito com PolynomialFeatures.
O r2_score mudou pouco, assim como o MSE continua bem alto.

Curiosamente, quando fiz os testes utilizando as variáveis FUELCONSUMPTION_CITY e FUELCONSUMPTION_COMB ao invés de ENGINESIZE,
houve uma melhora muito alta! Um último teste também foi feito com a variável FUELCONSUMPTION_HWY, que na teoria tem uma correlação ainda menor que ENGINESIZE
com CO2EMISSIONS, mas os resultados para ela também foram muito melhorados! A minha hipótese é que talvez seja o formato da distribuição dos dados em ENGINESIZE 
que esteja causando isso, porque os dados estão MUITO mais espalhados do que nas outras variáveis que eu testei, o que indica que o número de outliers 
em ENGINESIZE deve ser muito maior. Ao meu ver, não é uma fraqueza do modelo ou do pré-processamento, mas da variável ENGINESIZE que é
uma péssima candidata para uma regressão linear :P

Também testei com a variável CYLINDERS, que tem uma distribuição parecida com a de ENGINESIZE, e a melhora foi bem baixa, igual com ENGINESIZE.
Isso me convenceu ainda mais de que o problema é com a variável escolhida para treinar o modelo, não com o modelo.
'''
