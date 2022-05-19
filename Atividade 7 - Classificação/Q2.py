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

## TODO Selecione as dimensões de idade e batimentos máximos (só descomentar se Xtrain e ytrain já foram definidos)
# X2d = Xtrain[:, [0, 7]]
# y2d = ytrain

## TODO Visualize a distribuição (só descomentar)
# plt.scatter(X2d[:, 0], X2d[:, 1], c=y2d, cmap='bwr')
# print(X2d.shape, y2d.shape)

## TODO Separar os dados (X2d, y2d) em treino e validação (X2d_train, X2d_val, y2d_train, y2d_val)
# ...