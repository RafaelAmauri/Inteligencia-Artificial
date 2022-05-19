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


heart_disease = pd.read_csv("heart-disease.csv")
print(heart_disease.head())


# Porcentagem usada para treino
TRAIN_PERCENTAGE = 75

X = heart_disease.drop("target", axis=1).to_numpy()
y = heart_disease["target"]

x_train, x_test, y_train, y_test  = train_test_split(X, y, train_size=TRAIN_PERCENTAGE, shuffle=True)

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)