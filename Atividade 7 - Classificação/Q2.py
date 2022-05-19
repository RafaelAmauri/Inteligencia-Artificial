import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelos
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Funções úteis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


heart_disease = pd.read_csv("./heart-disease.csv")

### Q1
# Selecionando as dimensões de idade e batimentos máximos
X = heart_disease.drop("target", axis=1).to_numpy()
y = heart_disease["target"]


## TODO separe em treino e teste

# Porcentagem usada para treino
TRAIN_PERCENTAGE = 75

X = heart_disease.drop("target", axis=1).to_numpy()
y = heart_disease["target"]

x_train, x_test, y_train, y_test  = train_test_split(X, y, train_size=TRAIN_PERCENTAGE, shuffle=True)

scaler = StandardScaler()

scaler.fit(x_train)

## TODO normalize os dados

x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)
### Q1 end

## TODO Selecione as dimensões de idade e batimentos máximos (só descomentar se Xtrain e ytrain já foram definidos)
X2d = x_train_scaled[:, [0, 7]]
y2d = y_train

## TODO Visualize a distribuição (só descomentar)
plt.scatter(X2d[:, 0], X2d[:, 1], c=y2d, cmap='bwr')
print(X2d.shape, y2d.shape)

## TODO Separar os dados (X2d, y2d) em treino e validação (X2d_train, X2d_val, y2d_train, y2d_val)
